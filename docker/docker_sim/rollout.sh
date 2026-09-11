#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEED=40
HEADLESS=0
CHECKPOINT_REL=""
DATASET_REL=""

usage() {
  cat <<'EOF'
Usage: ./docker/docker_sim/rollout.sh [--seed N] [--headless] [--checkpoint PATH] [--dataset PATH]

Default opens the Isaac GUI and writes a rollout video under ~/X-VLA/outputs/.
PATH is relative to ~/X-VLA or an absolute host path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:?}"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --checkpoint)
      CHECKPOINT_REL="${2:?}"
      shift 2
      ;;
    --dataset)
      DATASET_REL="${2:?}"
      shift 2
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
ensure_bart_tokenizer

if [[ "${HEADLESS}" -eq 0 ]]; then
  allow_x11
  QILING_ISAAC_TTY=1
  export QILING_ISAAC_TTY
fi

if [[ -z "${CHECKPOINT_REL}" ]]; then
  CHECKPOINT_REL="$(find_checkpoint || true)"
fi
if [[ -z "${CHECKPOINT_REL}" ]]; then
  echo "No checkpoint found under ${QILING_ROOT}/outputs." >&2
  echo "Run ./docker/docker_sim/fetch_modelscope.sh, or pass --checkpoint." >&2
  exit 1
fi

if [[ -z "${DATASET_REL}" ]]; then
  DATASET_REL="$(find_lerobot_dataset || true)"
fi
if [[ -z "${DATASET_REL}" ]]; then
  echo "No LeRobot dataset found under ${QILING_ROOT}/datasets." >&2
  echo "Run ./docker/docker_sim/fetch_modelscope.sh (or record + train) first." >&2
  exit 1
fi

CHECKPOINT="$(to_container_path "${CHECKPOINT_REL}")"
DATASET="$(to_container_path "${DATASET_REL}")"

OUT_REL="outputs/slender_pin_xvla_rollout/seed${SEED}"
if [[ -d "${QILING_ROOT}/${OUT_REL}" ]] && [[ -n "$(ls -A "${QILING_ROOT}/${OUT_REL}" 2>/dev/null || true)" ]]; then
  OUT_REL="outputs/slender_pin_xvla_rollout/seed${SEED}_$(date +%Y%m%d_%H%M%S)"
fi
OUT_DIR="${CONTAINER_WORKSPACE}/${OUT_REL}"

TASK_CONFIG="$(container_config task_slender_pin_insertion_right_arm.yaml)"
ROBOT_CONFIG="$(container_config robot_dual_arm.yaml)"
CAMERA_CONFIG="$(container_config camera_bimanual.yaml)"

rollout_args=(
  scripts/run_slender_pin_xvla_rollout.py
  --checkpoint "${CHECKPOINT}"
  --dataset "${DATASET}"
  --task-config "${TASK_CONFIG}"
  --robot-config "${ROBOT_CONFIG}"
  --camera-config "${CAMERA_CONFIG}"
  --policy-device cuda
  --execution-horizon 32
  --seed "${SEED}"
  --record-video
  --out-dir "${OUT_DIR}"
)
if [[ "${HEADLESS}" -eq 1 ]]; then
  rollout_args+=(--headless)
fi

echo "[rollout] checkpoint ${CHECKPOINT_REL}"
echo "[rollout] dataset ${DATASET_REL}"
echo "[rollout] out ${OUT_REL} headless=${HEADLESS}"

run_isaac "${rollout_args[@]}"

echo "[rollout] done. artifacts: ${QILING_ROOT}/${OUT_REL}"
