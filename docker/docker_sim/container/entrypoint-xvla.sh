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

cd "${OPT}"

if [[ $# -eq 0 ]]; then
  exec bash
fi

exec "$@"
