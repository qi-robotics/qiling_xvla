#!/usr/bin/env bash
# Started by Isaac rollout as XVLA_POLICY_PYTHON. Forwards stdio pickle IPC
# into a sibling XVLA container via the host Docker daemon.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: xvla_policy_python.sh <serve_script> [args...]" >&2
  exit 2
fi

SCRIPT="$1"
shift

IMAGE="${QILING_XVLA_IMAGE:-}"
HOST_ROOT="${QILING_ROOT_HOST:-}"

if [[ -z "${IMAGE}" ]]; then
  echo "QILING_XVLA_IMAGE is not set." >&2
  exit 1
fi
if [[ -z "${HOST_ROOT}" ]]; then
  echo "QILING_ROOT_HOST is not set." >&2
  exit 1
fi
if [[ ! -S /var/run/docker.sock ]]; then
  echo "docker.sock is not mounted; cannot start the XVLA policy container." >&2
  exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "XVLA image not found: ${IMAGE}. Run ./docker/docker_sim/up.sh first." >&2
  exit 1
fi

exec docker run --rm -i \
  --gpus all \
  --ipc=host \
  --user 1234:1234 \
  --network none \
  --log-driver=none \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e HF_HOME=/workspace/X-VLA/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/workspace/X-VLA/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/workspace/X-VLA/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "${HOST_ROOT}:/workspace/X-VLA" \
  --entrypoint /opt/venv/bin/python \
  "${IMAGE}" \
  "${SCRIPT}" "$@"
