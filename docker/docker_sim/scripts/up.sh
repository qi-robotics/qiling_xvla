#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_docker
require_nvidia
prepare_workspace
acr_login

echo "[up] workspace ${QILING_ROOT}"
echo "[up] xvla image ${QILING_XVLA_IMAGE}"
echo "[up] docker.sock gid ${DOCKER_GID}"

echo "[up] pulling Isaac Sim 5.1.0 from NVIDIA NGC (no China mirror; license forbids copying it to ACR)"
docker pull nvcr.io/nvidia/isaac-sim:5.1.0

if docker image inspect "${QILING_XVLA_IMAGE}" >/dev/null 2>&1; then
  echo "[up] using local ${QILING_XVLA_IMAGE}"
else
  echo "[up] pulling ${QILING_XVLA_IMAGE}"
  docker pull "${QILING_XVLA_IMAGE}"
fi

echo "[up] building Isaac overlay qiling-isaac:5.1.0 (PyPI ${PIP_INDEX_URL})"
qiling_compose build isaac

echo
echo "[up] ready. Next:"
echo "  skip collect/train:  ./docker/docker_sim/scripts/fetch_modelscope.sh && ./docker/docker_sim/scripts/rollout.sh --seed 40"
echo "  or collect+train:    ./docker/docker_sim/scripts/record.sh --count 3 && ./docker/docker_sim/scripts/train.sh && ./docker/docker_sim/scripts/rollout.sh --seed 40"
