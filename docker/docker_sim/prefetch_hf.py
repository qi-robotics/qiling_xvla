#!/usr/bin/env python3
"""Download runtime Hugging Face files into the workspace cache, not the image.

Default is facebook/bart-large (tokenizer for train/rollout).
Pass --with-base to also fetch lerobot/xvla-base (optional fine-tune start).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HOME", "/workspace/X-VLA/.cache/huggingface")
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", "/workspace/X-VLA/.cache/huggingface/hub"
)

from huggingface_hub import snapshot_download

DEFAULT_REPOS = ("facebook/bart-large",)
BASE_REPO = "lerobot/xvla-base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-base",
        action="store_true",
        help="also download lerobot/xvla-base (large; only if you fine-tune from it)",
    )
    args = parser.parse_args()

    repos = list(DEFAULT_REPOS)
    if args.with_base:
        repos.append(BASE_REPO)

    for repo in repos:
        print(f"[prefetch] {repo} -> {os.environ['HF_HOME']}", flush=True)
        snapshot_download(repo_id=repo)
        print(f"[prefetch] ok {repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
