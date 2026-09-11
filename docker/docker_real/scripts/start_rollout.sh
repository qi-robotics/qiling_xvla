#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
qiling_render_config

set -a
source "$QILING_ENV_FILE"
set +a

if [[ "$QILING_EXECUTION_MODE" == "armed" ]]; then
  echo "WARNING: armed mode will command the real robot."
  echo "Stop teleoperation publishers and keep an emergency stop ready."
  read -r -p "Type ARM to continue: " confirmation
  if [[ "$confirmation" != "ARM" ]]; then
    echo "Rollout cancelled."
    exit 1
  fi
fi

cleanup() {
  trap - TERM INT EXIT
  qiling_compose down --timeout 5 >/dev/null 2>&1 || true
}
trap cleanup TERM INT EXIT

echo "[rollout] starting control and worker containers"
echo "[rollout] control order: topic_convertor -> state check -> rollout bridge"
echo "[rollout] worker waits until the bridge opens localhost IPC after homing"
qiling_compose up --abort-on-container-exit --remove-orphans control worker
