#!/usr/bin/env bash
# Shared paths and docker compose wrapper for docker/docker_sim/*.sh.
# shellcheck disable=SC2034

set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DOCKER_DIR}/../.." && pwd)"

QILING_ROOT="${QILING_ROOT:-${HOME}/X-VLA}"
QILING_XVLA_TAG="${QILING_XVLA_TAG:-lerobot050}"
QILING_ACR_HOST="${QILING_ACR_HOST:-registry.cn-hangzhou.aliyuncs.com}"
QILING_ACR_USER="${QILING_ACR_USER:-694407146@qq.com}"
QILING_ACR_PASSWORD="${QILING_ACR_PASSWORD:-qirobotics@1234}"
QILING_XVLA_LOCAL="${QILING_XVLA_LOCAL:-qiling-xvla:${QILING_XVLA_TAG}}"
QILING_XVLA_IMAGE="${QILING_XVLA_IMAGE:-${QILING_ACR_HOST}/keno/qi-xvla:${QILING_XVLA_TAG}}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export QILING_ROOT QILING_XVLA_TAG QILING_ACR_HOST QILING_ACR_USER QILING_ACR_PASSWORD
export QILING_XVLA_LOCAL QILING_XVLA_IMAGE HF_ENDPOINT PIP_INDEX_URL

if [[ -S /var/run/docker.sock ]]; then
  DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
  export DOCKER_GID
fi

WORKSPACE_DATASETS="${QILING_ROOT}/datasets"
WORKSPACE_OUTPUTS="${QILING_ROOT}/outputs"
WORKSPACE_CONFIGS="${QILING_ROOT}/configs"

CONTAINER_WORKSPACE="/workspace/X-VLA"

DEFAULT_CONFIGS=(
  task_slender_pin_insertion_right_arm.yaml
  robot_dual_arm.yaml
  camera_bimanual.yaml
)

qiling_compose() {
  docker compose \
    --project-directory "${REPO_ROOT}" \
    -f "${DOCKER_DIR}/docker-compose.yml" \
    "$@"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker daemon is not running, or this user cannot talk to it." >&2
    exit 1
  fi
  if [[ ! -S /var/run/docker.sock ]]; then
    echo "/var/run/docker.sock is missing; rollout needs it to start the XVLA container." >&2
    exit 1
  fi
  if [[ -z "${DOCKER_GID:-}" ]]; then
    echo "Could not read the group id of /var/run/docker.sock." >&2
    exit 1
  fi
}

require_nvidia() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Install an NVIDIA driver first." >&2
    exit 1
  fi
}

# Login to the private ACR so docker pull/push work as the current user (not root).
acr_login() {
  echo "[acr] docker login ${QILING_ACR_HOST}  user=${QILING_ACR_USER}"
  printf '%s\n' "${QILING_ACR_PASSWORD}" | docker login \
    --username "${QILING_ACR_USER}" \
    --password-stdin \
    "${QILING_ACR_HOST}"
}

prepare_workspace() {
  mkdir -p \
    "${WORKSPACE_DATASETS}" \
    "${WORKSPACE_OUTPUTS}" \
    "${WORKSPACE_CONFIGS}" \
    "${QILING_ROOT}/reports" \
    "${QILING_ROOT}/.isaac-cache/cache/main/ov" \
    "${QILING_ROOT}/.isaac-cache/cache/main/warp" \
    "${QILING_ROOT}/.isaac-cache/cache/computecache" \
    "${QILING_ROOT}/.isaac-cache/config" \
    "${QILING_ROOT}/.isaac-cache/data/documents" \
    "${QILING_ROOT}/.isaac-cache/data/Kit" \
    "${QILING_ROOT}/.isaac-cache/logs" \
    "${QILING_ROOT}/.isaac-cache/pkg" \
    "${QILING_ROOT}/.cache/huggingface/hub" \
    "${QILING_ROOT}/.cache/huggingface/transformers" \
    "${QILING_ROOT}/.cache/huggingface/datasets"
  touch "${HOME}/.Xauthority" 2>/dev/null || true
  chmod -R a+rwX "${QILING_ROOT}" 2>/dev/null || true

  local name
  for name in "${DEFAULT_CONFIGS[@]}"; do
    if [[ ! -e "${WORKSPACE_CONFIGS}/${name}" && -f "${REPO_ROOT}/configs/${name}" ]]; then
      cp "${REPO_ROOT}/configs/${name}" "${WORKSPACE_CONFIGS}/${name}"
    fi
  done
}

container_config() {
  local name="$1"
  if [[ -f "${WORKSPACE_CONFIGS}/${name}" ]]; then
    echo "${CONTAINER_WORKSPACE}/configs/${name}"
  else
    echo "configs/${name}"
  fi
}

first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "${QILING_ROOT}/${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

find_lerobot_dataset() {
  first_existing \
    datasets/slender_pin_lerobot_v3_xvla_v1 \
    datasets/slender_pin_lerobot_v3_xvla_datasets \
    datasets/slender_pin_lerobot_v3_pi05_v1
}

find_checkpoint() {
  local found latest
  found="$(first_existing \
    outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/last/pretrained_model \
    outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/200000/pretrained_model \
    outputs/xvla_slender_pin_full/checkpoints/last/pretrained_model \
    outputs/xvla_slender_pin_smoke/checkpoints/last/pretrained_model || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi
  latest="$(ls -d "${QILING_ROOT}"/outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/*/pretrained_model 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    echo "${latest#"${QILING_ROOT}"/}"
    return 0
  fi
  latest="$(ls -d "${QILING_ROOT}"/outputs/xvla_slender_pin_smoke_*/checkpoints/last/pretrained_model 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    echo "${latest#"${QILING_ROOT}"/}"
    return 0
  fi
  return 1
}

allow_x11() {
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is empty. GUI needs a local X11 display." >&2
    exit 1
  fi
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || xhost +local: >/dev/null 2>&1 || true
  fi
}

run_isaac() {
  local tty_flag=()
  if [[ "${QILING_ISAAC_TTY:-0}" == "1" ]]; then
    tty_flag=()
  else
    tty_flag=(-T)
  fi
  qiling_compose run --rm --no-deps "${tty_flag[@]}" isaac "$@"
}

run_xvla() {
  qiling_compose run --rm --no-deps -T xvla "$@"
}

ensure_bart_tokenizer() {
  local marker="${QILING_ROOT}/.cache/huggingface/hub/models--facebook--bart-large"
  if [[ -d "${marker}" ]]; then
    return 0
  fi
  echo "[hf] facebook/bart-large is not in ${QILING_ROOT}/.cache/huggingface"
  echo "[hf] downloading facebook/bart-large via ${HF_ENDPOINT}"
  run_xvla python docker/prefetch_hf.py
}

# Prints a dataset path relative to QILING_ROOT. Converts recorded episodes if needed.
ensure_lerobot_dataset() {
  local found recorded
  found="$(find_lerobot_dataset || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  recorded="${QILING_ROOT}/datasets/recorded_slender_pin_v1"
  if [[ ! -d "${recorded}" ]]; then
    echo "No LeRobot dataset and no recorded episodes." >&2
    echo "Run ./docker/docker_sim/record.sh first, or ./docker/docker_sim/fetch_modelscope.sh" >&2
    return 1
  fi
  echo "[train] converting recorded episodes to LeRobot v3" >&2
  run_xvla python scripts/convert_slender_pin_to_lerobot_v3.py \
    --recorded-root "${CONTAINER_WORKSPACE}/datasets/recorded_slender_pin_v1" \
    --output-dir "${CONTAINER_WORKSPACE}/datasets/slender_pin_lerobot_v3_xvla_v1" \
    --repo-id qiling/slender_pin_xvla_v1 \
    --json-out "${CONTAINER_WORKSPACE}/reports/convert_slender_pin_lerobot_v3_xvla_v1.json" \
    --overwrite
  printf '%s\n' "datasets/slender_pin_lerobot_v3_xvla_v1"
}

to_container_path() {
  local value="$1"
  if [[ "${value}" == /workspace/X-VLA/* ]]; then
    echo "${value}"
  elif [[ "${value}" == /* ]]; then
    echo "${CONTAINER_WORKSPACE}/${value#"${QILING_ROOT}"/}"
  else
    echo "${CONTAINER_WORKSPACE}/${value}"
  fi
}
