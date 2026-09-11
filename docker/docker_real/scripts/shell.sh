#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker
qiling_render_config

case "${1:-}" in
  control)
    qiling_compose run --rm --no-deps --entrypoint bash control
    ;;
  reasoning)
    qiling_compose run --rm --no-deps --entrypoint bash trainer
    ;;
  *)
    echo "Usage: $0 [control|reasoning]" >&2
    exit 2
    ;;
esac
