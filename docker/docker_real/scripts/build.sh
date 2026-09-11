#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qiling_require_docker

# Optional delivery test: remove shell proxy variables and reject a Docker
# daemon that still routes pulls through a proxy. The script never changes the
# host's systemd/Docker configuration by itself.
if [[ "${QI_BUILD_WITHOUT_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy all_proxy no_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  if ! docker_info="$(docker info 2>&1)"; then
    echo "[build] Unable to inspect Docker daemon in strict no-proxy mode:" >&2
    echo "$docker_info" >&2
    exit 3
  fi
  if grep -Eq '^[[:space:]]*(HTTP|HTTPS) Proxy:[[:space:]]*[^[:space:]]' <<<"$docker_info"; then
    echo "[build] QI_BUILD_WITHOUT_PROXY=1, but Docker daemon still has a proxy configured." >&2
    echo "[build] Disable the Docker systemd proxy, restart Docker, and retry." >&2
    exit 3
  fi
  echo "[build] strict no-proxy mode verified"
fi

set -a
source "$QILING_RELEASE_ROOT/dependencies/build.env"
set +a

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This release currently supports only x86_64 hosts." >&2
  exit 1
fi

echo "[build] pulling domestic ROS bases and building qi-reasoning:local"
qiling_compose build --pull control

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
