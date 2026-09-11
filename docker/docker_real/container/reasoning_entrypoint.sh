#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/opt/qiling/runtime"
mkdir -p "$runtime_root/hf_cache" "$runtime_root/cache" "$runtime_root/home"

if [[ ! -d "$runtime_root/hf_cache/hub/models--facebook--bart-large" ]]; then
  echo "[qi-reasoning] seeding offline facebook/bart-large tokenizer cache"
  cp -a /opt/qiling/hf_seed/. "$runtime_root/hf_cache/"
fi

export HOME="$runtime_root/home"
export HF_HOME="$runtime_root/hf_cache"
export XDG_CACHE_HOME="$runtime_root/cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec "$@"
