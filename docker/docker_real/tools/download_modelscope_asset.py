#!/usr/bin/env python3
"""Download one configured model or dataset subtree from ModelScope."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess

import yaml


def relative_path(value: object, name: str) -> PurePosixPath:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{name} must be a non-empty relative path")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    assets = config.get("assets", {})
    if args.asset not in assets:
        raise RuntimeError(f"unknown asset {args.asset!r}; available: {', '.join(assets)}")
    asset = assets[args.asset] or {}
    repo_id = str(asset.get("repo_id", "")).strip()
    remote = relative_path(asset.get("remote_path"), f"assets.{args.asset}.remote_path")
    local_subdir = relative_path(asset.get("local_subdir"), f"assets.{args.asset}.local_subdir")
    if not repo_id:
        raise RuntimeError(f"assets.{args.asset}.repo_id is empty; configure it before downloading")

    destination = args.destination_root / local_subdir
    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("MS_TOKEN", "").strip()
    if token and not args.dry_run:
        subprocess.run(["modelscope", "login", "--token", token], check=True)

    command = [
        "modelscope",
        "download",
        repo_id,
        "--repo-type",
        str(asset.get("repo_type", "dataset")),
        "--revision",
        str(asset.get("revision", "master")),
        "--include",
        f"{remote}/**",
        "--local-dir",
        str(destination),
    ]
    print(
        f"Downloading {repo_id}:{remote} -> {destination} "
        f"(revision={asset.get('revision', 'master')})",
        flush=True,
    )
    if args.dry_run:
        print("Command: " + " ".join(command), flush=True)
        return
    subprocess.run(command, check=True)

    downloaded_root = destination / remote
    if args.asset in {"xvla_base", "rollout_checkpoint"}:
        for required in ("config.json", "model.safetensors"):
            if not (downloaded_root / required).is_file():
                raise RuntimeError(f"download completed but required file is missing: {downloaded_root / required}")
    print(f"Asset ready: {downloaded_root}", flush=True)


if __name__ == "__main__":
    main()
