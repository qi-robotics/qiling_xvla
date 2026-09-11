#!/usr/bin/env bash

set -Eeuo pipefail

QILING_RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QILING_COMPOSE_FILE="$QILING_RELEASE_ROOT/compose.yaml"
QILING_CONFIG_FILE="$QILING_RELEASE_ROOT/config/qiling.yaml"
QILING_RUNTIME_DIR="$QILING_RELEASE_ROOT/runtime"
QILING_ENV_FILE="$QILING_RUNTIME_DIR/release.env"

export QILING_UID="$(id -u)"
export QILING_GID="$(id -g)"

mkdir -p \
  "$QILING_RELEASE_ROOT/models" \
  "$QILING_RELEASE_ROOT/datasets" \
  "$QILING_RELEASE_ROOT/outputs" \
  "$QILING_RUNTIME_DIR"

qiling_compose() {
  local env_args=()
  if [[ -f "$QILING_ENV_FILE" ]]; then
    env_args=(--env-file "$QILING_ENV_FILE")
  fi
  docker compose "${env_args[@]}" -f "$QILING_COMPOSE_FILE" "$@"
}

qiling_require_docker() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is not installed" >&2
    exit 1
  }
  docker compose version >/dev/null 2>&1 || {
    echo "Docker Compose v2 is not available" >&2
    exit 1
  }
}

qiling_render_config() {
  docker run --rm \
    --user "$QILING_UID:$QILING_GID" \
    -v "$QILING_RELEASE_ROOT/config:/opt/qiling/config:ro" \
    -v "$QILING_RUNTIME_DIR:/opt/qiling/runtime" \
    qi-reasoning:local \
    /opt/qi-reasoning/bin/python /opt/qiling/release/tools/render_config.py \
      --config /opt/qiling/config/qiling.yaml \
      --output /opt/qiling/runtime
}

qiling_run_reasoning_tool() {
  local token_args=()
  if [[ -n "${MS_TOKEN:-}" ]]; then
    token_args=(--env MS_TOKEN)
  fi
  docker run --rm \
    --user "$QILING_UID:$QILING_GID" \
    "${token_args[@]}" \
    -v "$QILING_RELEASE_ROOT/config:/opt/qiling/config:ro" \
    -v "$QILING_RELEASE_ROOT/models:/opt/qiling/models" \
    -v "$QILING_RELEASE_ROOT/datasets:/opt/qiling/datasets" \
    -v "$QILING_RUNTIME_DIR:/opt/qiling/runtime" \
    qi-reasoning:local "$@"
}
