#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STEPS=200000
BATCH_SIZE=4
SAVE_FREQ=50000
LOG_FREQ=100
NUM_WORKERS=4
OUTPUT_REL="outputs/xvla_slender_pin_full"
PRETRAINED_REL=""
FROM_BASE=0
RESUME=0

RENAME_MAP='{"observation.images.chest":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'

usage() {
  cat <<'EOF'
Usage: ./docker/docker_sim/train.sh [options]

Full XVLA training (default 200000 steps, batch 4). Writes under ~/X-VLA/outputs/.

  --steps N
  --batch-size N
  --save-freq N
  --num-workers N
  --output-dir PATH          relative to ~/X-VLA, or absolute
  --pretrained-path PATH     continue from a local ckpt
  --from-base                start from lerobot/xvla-base (downloads it once)
  --resume                   resume the job in --output-dir
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="${2:?}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?}"
      shift 2
      ;;
    --save-freq)
      SAVE_FREQ="${2:?}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:?}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_REL="${2:?}"
      shift 2
      ;;
    --pretrained-path)
      PRETRAINED_REL="${2:?}"
      shift 2
      ;;
    --from-base)
      FROM_BASE=1
      shift
      ;;
    --resume)
      RESUME=1
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

if [[ "${FROM_BASE}" -eq 1 && -n "${PRETRAINED_REL}" ]]; then
  echo "Use only one of --from-base or --pretrained-path." >&2
  exit 2
fi

require_docker
require_nvidia
prepare_workspace

DATASET_REL="$(ensure_lerobot_dataset)"
DATASET="$(to_container_path "${DATASET_REL}")"

if [[ "${OUTPUT_REL}" == /* ]]; then
  OUTPUT_HOST="${OUTPUT_REL}"
  OUTPUT="$(to_container_path "${OUTPUT_REL}")"
  OUTPUT_REL="${OUTPUT_HOST#"${QILING_ROOT}"/}"
else
  OUTPUT_HOST="${QILING_ROOT}/${OUTPUT_REL}"
  OUTPUT="$(to_container_path "${OUTPUT_REL}")"
fi

if [[ "${RESUME}" -eq 0 && -z "${PRETRAINED_REL}" && -d "${OUTPUT_HOST}" ]]; then
  OUTPUT_REL="${OUTPUT_REL}_$(date +%Y%m%d_%H%M%S)"
  OUTPUT_HOST="${QILING_ROOT}/${OUTPUT_REL}"
  OUTPUT="$(to_container_path "${OUTPUT_REL}")"
  echo "[train] ${OUTPUT_HOST} already exists; writing to ${OUTPUT_REL}"
fi

ensure_bart_tokenizer
if [[ "${FROM_BASE}" -eq 1 ]]; then
  echo "[train] fetching lerobot/xvla-base into the workspace cache"
  run_xvla python docker/prefetch_hf.py --with-base
fi

train_args=(
  lerobot-train
  --policy.dtype=bfloat16
  --policy.device=cuda
  --policy.action_mode=auto
  --rename_map="${RENAME_MAP}"
  --dataset.root="${DATASET}"
  --dataset.repo_id=qiling/slender_pin_xvla_v1
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --num_workers="${NUM_WORKERS}"
  --output_dir="${OUTPUT}"
  --job_name=xvla_slender_pin_full
  --wandb.enable=false
  --policy.push_to_hub=false
)

if [[ "${FROM_BASE}" -eq 1 ]]; then
  train_args+=(--policy.path=lerobot/xvla-base)
else
  train_args+=(--policy.type=xvla)
fi

if [[ -n "${PRETRAINED_REL}" ]]; then
  train_args+=(--policy.pretrained_path="$(to_container_path "${PRETRAINED_REL}")")
fi
if [[ "${RESUME}" -eq 1 ]]; then
  train_args+=(--resume=true)
fi

echo "[train] dataset ${DATASET_REL}"
echo "[train] output ${OUTPUT_REL} steps=${STEPS} batch_size=${BATCH_SIZE} save_freq=${SAVE_FREQ}"

run_xvla "${train_args[@]}"

echo "[train] checkpoints are under ${OUTPUT_HOST}"
