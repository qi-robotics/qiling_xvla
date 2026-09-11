#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
selection="${1:-all}"

download_asset() {
  qiling_run_reasoning_tool \
    /opt/qi-reasoning/bin/python /opt/qiling/release/tools/download_modelscope_asset.py \
    --config /opt/qiling/config/qiling.yaml \
    --asset "$1" \
    --destination-root /opt/qiling/models
}

case "$selection" in
  base)
    download_asset xvla_base
    ;;
  rollout)
    download_asset rollout_checkpoint
    ;;
  all)
    download_asset xvla_base
    download_asset rollout_checkpoint
    ;;
  *)
    echo "Usage: $0 [base|rollout|all]" >&2
    exit 2
    ;;
esac
