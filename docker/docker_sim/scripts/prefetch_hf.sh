#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_docker
prepare_workspace

echo "[hf] cache ${QILING_ROOT}/.cache/huggingface  endpoint=${HF_ENDPOINT}"
run_prefetch_hf "$@"
