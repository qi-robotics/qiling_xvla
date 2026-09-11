#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker

profile="${1:-${QI_BUILD_PROFILE:-domestic}}"
case "$profile" in
  domestic|proxy) ;;
  *)
    echo "Usage: $0 [domestic|proxy]" >&2
    exit 2
    ;;
esac

profile_file="$QILING_RELEASE_ROOT/dependencies/build.$profile.env"
[[ -f "$profile_file" ]] || {
  echo "Build profile not found: $profile_file" >&2
  exit 2
}

set -a
source "$profile_file"
set +a

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This release currently supports only x86_64 hosts." >&2
  exit 1
fi

export DOCKER_BUILDKIT=1

pull_args=()
if [[ "${QI_PULL_BASE:-0}" == "1" ]]; then
  pull_args=(--pull)
fi

echo "[build] profile=$profile; building qi-reasoning:local"
qiling_compose build "${pull_args[@]}" control

echo "[build] rendering and validating the single YAML configuration"
qiling_render_config

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build] verifying NVIDIA GPU access inside qi-reasoning"
  qiling_compose run --rm --no-deps trainer \
    /opt/qi-reasoning/bin/python /opt/qiling/app/verify_runtime.py
else
  echo "[build] WARNING: nvidia-smi is unavailable; image build succeeded but GPU runtime was not verified." >&2
fi

echo "[build] complete"
echo "Download models: ./scripts/download_models.sh all"
echo "Start rollout:   ./scripts/start_rollout.sh"
echo "Start training:  ./scripts/train.sh"
