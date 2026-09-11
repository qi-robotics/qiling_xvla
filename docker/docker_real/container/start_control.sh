#!/usr/bin/env bash
set -Eeo pipefail

set +u
source /opt/ros/humble/setup.bash
source /opt/qiling/control_ws/install/setup.bash
set -u

converter_pid=""
bridge_pid=""

cleanup() {
  trap - TERM INT EXIT
  if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
    kill -TERM "$bridge_pid" 2>/dev/null || true
  fi
  if [[ -n "$converter_pid" ]] && kill -0 "$converter_pid" 2>/dev/null; then
    kill -TERM "$converter_pid" 2>/dev/null || true
  fi
  wait "$bridge_pid" "$converter_pid" 2>/dev/null || true
}
trap cleanup TERM INT EXIT

echo "[control] starting topic_convertor"
ros2 run topic_convertor topic_converter_node --ros-args \
  -p expected_motor_count:=26 \
  -p enable_state_bridge:=true \
  -p enable_command_bridge:=true \
  -p strict_command_size:=true &
converter_pid=$!

sleep 1
if ! kill -0 "$converter_pid" 2>/dev/null; then
  echo "[control] topic_convertor exited during startup" >&2
  exit 1
fi

state_wait="${QILING_STATE_WAIT_SEC:-30}"
echo "[control] waiting up to ${state_wait}s for /human_lower_state"
if ! timeout "$state_wait" ros2 topic echo /human_lower_state --once >/dev/null 2>&1; then
  echo "[control] no /human_lower_state received; verify robot SDK, ROS_DOMAIN_ID and DDS interface" >&2
  exit 2
fi

echo "[control] state received; starting rollout bridge in ${QILING_EXECUTION_MODE:-shadow} mode"
ros2 launch qiling_rollout_ros rollout_host.launch.py \
  config_file:=/opt/qiling/runtime/rollout_host.yaml \
  execution_mode:="${QILING_EXECUTION_MODE:-shadow}" &
bridge_pid=$!

wait -n "$converter_pid" "$bridge_pid"
status=$?
echo "[control] a required process exited with status ${status}; stopping control container" >&2
exit "$status"
