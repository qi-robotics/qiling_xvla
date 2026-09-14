#!/usr/bin/env python3
"""Download runtime Hugging Face files into the workspace cache, not the image.

Default is facebook/bart-large tokenizer files only (~2MB). The full BART repo
also has ~6GB of pytorch/flax/tf/safetensors weights that XVLA never loads.
Pass --with-base to also fetch lerobot/xvla-base (optional fine-tune start).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

os.environ.setdefault("HF_HOME", "/workspace/X-VLA/.cache/huggingface")
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", "/workspace/X-VLA/.cache/huggingface/hub"
)
# huggingface_hub 1.x talks to xethub.hf.co unless this is set; that hangs in CN.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
socket.setdefaulttimeout(30)

from huggingface_hub import hf_hub_download, snapshot_download

try:
    from huggingface_hub.errors import EntryNotFoundError
except ImportError:  # huggingface_hub < 0.24
    from huggingface_hub.utils import EntryNotFoundError  # type: ignore

BASE_REPO = "lerobot/xvla-base"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
TOKENIZER_REQUIRED = ("vocab.json", "merges.txt", "tokenizer.json")


def _hub() -> str:
    return os.environ["HUGGINGFACE_HUB_CACHE"]


def bart_tokenizer_cached() -> bool:
    snap = os.path.join(_hub(), "models--facebook--bart-large", "snapshots")
    if not os.path.isdir(snap):
        return False
    for name in os.listdir(snap):
        folder = os.path.join(snap, name)
        if not os.path.isdir(folder):
            continue
        if all(os.path.exists(os.path.join(folder, f)) for f in TOKENIZER_REQUIRED):
            return True
    return False


def prefetch_bart_tokenizer() -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(
        f"[prefetch] facebook/bart-large tokenizer -> {os.environ['HF_HOME']}",
        flush=True,
    )
    if bart_tokenizer_cached():
        print("[prefetch] ok facebook/bart-large (already in cache, skip download)", flush=True)
        return
    print(f"[prefetch] fetching ~2MB via {endpoint} (not the 6GB BART weights)", flush=True)
    for name in TOKENIZER_FILES:
        print(f"[prefetch]   {name}", flush=True)
        try:
            hf_hub_download(repo_id="facebook/bart-large", filename=name)
        except EntryNotFoundError:
            print(f"[prefetch]   skip missing {name}", flush=True)
    if not bart_tokenizer_cached():
        raise SystemExit(
            "tokenizer download finished but vocab.json/merges.txt/tokenizer.json "
            f"are not in {_hub()}/models--facebook--bart-large/snapshots"
        )
    print("[prefetch] ok facebook/bart-large", flush=True)


def prefetch_full(repo: str) -> None:
    print(f"[prefetch] {repo} full snapshot -> {os.environ['HF_HOME']}", flush=True)
    snapshot_download(repo_id=repo)
    print(f"[prefetch] ok {repo}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-base",
        action="store_true",
        help="also download lerobot/xvla-base (large; only if you fine-tune from it)",
    )
    args = parser.parse_args()

    prefetch_bart_tokenizer()
    if args.with_base:
        prefetch_full(BASE_REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
