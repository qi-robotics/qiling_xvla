#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
qiling_render_config

echo "[train] outputs are stored in $QILING_RELEASE_ROOT/outputs"
qiling_compose run --rm --no-deps trainer \
  /opt/qi-reasoning/bin/python /opt/qiling/app/run_xvla_train.py \
    --config /opt/qiling/runtime/xvla_training.yaml \
    "$@"
