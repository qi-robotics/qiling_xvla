#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
qiling_compose exec -T control bash -lc '
  set +u
  source /opt/ros/humble/setup.bash
  source /opt/qiling/control_ws/install/setup.bash
  ros2 service call "${QILING_ABORT_SERVICE:-/rollout/abort}" std_srvs/srv/Trigger "{}"
'
echo "Abort requested: the bridge will hold the latest measured arm pose."
