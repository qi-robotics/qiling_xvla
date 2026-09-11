#!/usr/bin/env bash
set -euo pipefail

WORK="${QILING_WORKSPACE:-/workspace/X-VLA}"
OPT="/opt/qiling_xvla"

mkdir -p "${WORK}/datasets" "${WORK}/outputs" "${WORK}/configs" "${WORK}/reports" || {
  echo "Cannot write ${WORK}. On the host run: chmod -R a+rwX \"\$HOME/X-VLA\"" >&2
  exit 1
}

ln -sfn "${WORK}/datasets" "${OPT}/datasets"
ln -sfn "${WORK}/outputs" "${OPT}/outputs"
ln -sfn "${WORK}/reports" "${OPT}/reports"

export XVLA_POLICY_PYTHON="${XVLA_POLICY_PYTHON:-/opt/qiling_xvla/docker/docker_sim/container/xvla_policy_python.sh}"

ISAAC_PYTHON="/isaac-sim/python.sh"
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac python.sh not found at ${ISAAC_PYTHON}" >&2
  exit 1
fi

cd "${OPT}"

if [[ $# -eq 0 ]]; then
  exec bash
fi
if [[ "$1" == "bash" || "$1" == "sh" ]]; then
  exec "$@"
fi

exec "${ISAAC_PYTHON}" "$@"
