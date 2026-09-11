#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage: ./docker/docker_sim/scripts/fetch_modelscope.sh [--dry-run] [--with-optimizer]

Download only the XVLA dataset + 200k weights from ModelScope into ~/X-VLA.
Does not download bottleInBowl / smolVLA in the same repo.

  --dry-run           list files and sizes, do not download
  --with-optimizer    also fetch training_state (~3.5 GB; only to resume that job)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to download from ModelScope." >&2
  exit 1
fi

prepare_workspace

echo "[fetch] ModelScope keno123/qi-studio_embodied_edu  (xvla/ only)"
python3 "${SIM_DIR}/tools/fetch_modelscope.py" --dest "${QILING_ROOT}" "$@"
chmod -R a+rwX "${QILING_ROOT}/datasets" "${QILING_ROOT}/outputs" 2>/dev/null || true

if [[ " $* " != *" --dry-run "* ]]; then
  echo
  echo "[fetch] next: ./docker/docker_sim/scripts/rollout.sh --seed 40"
  echo "        (run ./docker/docker_sim/scripts/up.sh first if images are not ready)"
fi
