#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker

# Latch the current measured pose before removing publishers. This is not the
# normal task-completion path; use finish_rollout.sh to return to home first.
qiling_compose exec -T control bash -lc '
  set +u
  source /opt/ros/humble/setup.bash
  source /opt/qiling/control_ws/install/setup.bash
  ros2 service call "${QILING_ABORT_SERVICE:-/rollout/abort}" std_srvs/srv/Trigger "{}"
' >/dev/null 2>&1 || true
sleep 1
qiling_compose down --timeout 5
