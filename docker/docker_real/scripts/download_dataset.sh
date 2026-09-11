#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
qiling_run_reasoning_tool \
  /opt/qi-reasoning/bin/python /opt/qiling/release/tools/download_modelscope_asset.py \
  --config /opt/qiling/config/qiling.yaml \
  --asset dataset \
  --destination-root /opt/qiling/datasets
