#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COUNT=3
GUI=0
OVERWRITE=0
RESUME=1

usage() {
  cat <<'EOF'
Usage: ./docker/docker_sim/record.sh [--count N] [--gui] [--overwrite] [--no-resume]

Default is headless. Writes to:
  ~/X-VLA/datasets/raw_slender_pin_v1
  ~/X-VLA/datasets/recorded_slender_pin_v1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count)
      COUNT="${2:?}"
      shift 2
      ;;
    --gui)
      GUI=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_docker
require_nvidia
prepare_workspace

if [[ "${GUI}" -eq 1 ]]; then
  allow_x11
  QILING_ISAAC_TTY=1
  export QILING_ISAAC_TTY
fi

TASK_CONFIG="$(container_config task_slender_pin_insertion_right_arm.yaml)"
RAW="${CONTAINER_WORKSPACE}/datasets/raw_slender_pin_v1"
REC="${CONTAINER_WORKSPACE}/datasets/recorded_slender_pin_v1"
SUMMARY="${CONTAINER_WORKSPACE}/reports/slender_pin_v1_summary.json"

generate_args=(
  scripts/generate_slender_pin_recovery_smoke.py
  --output-root "${RAW}"
  --task-config "${TASK_CONFIG}"
  --count "${COUNT}"
)
if [[ "${OVERWRITE}" -eq 1 ]]; then
  generate_args+=(--overwrite)
fi

if [[ "${OVERWRITE}" -eq 1 || ! -f "${QILING_ROOT}/datasets/raw_slender_pin_v1/manifest.json" ]]; then
  echo "[record] generate count=${COUNT}"
  run_isaac "${generate_args[@]}"
else
  echo "[record] reuse existing ${QILING_ROOT}/datasets/raw_slender_pin_v1 (pass --overwrite to regenerate)"
fi

record_args=(
  scripts/record_slender_pin_recovery_smoke_headless.py
  --raw-root "${RAW}"
  --recorded-root "${REC}"
  --summary-json "${SUMMARY}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  record_args+=(--resume)
fi
if [[ "${GUI}" -eq 1 ]]; then
  record_args+=(--gui)
fi

echo "[record] record gui=${GUI}"
run_isaac "${record_args[@]}"

echo "[record] validate"
run_isaac \
  scripts/validate_slender_pin_recovery_dataset.py \
  --recorded-root "${REC}" \
  --json-out "${CONTAINER_WORKSPACE}/reports/slender_pin_v1_validation.json"

echo "[record] done. recorded data is in ${QILING_ROOT}/datasets/recorded_slender_pin_v1"
