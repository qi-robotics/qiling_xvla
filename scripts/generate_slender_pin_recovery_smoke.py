#!/usr/bin/env python3
"""Create a slender-pin normal/recovery recording manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="datasets/raw_slender_pin_v1")
    parser.add_argument("--task-config", default="configs/task_slender_pin_insertion_right_arm.yaml")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 3:
        raise ValueError("--count must be at least 3 to include normal and both recovery types")
    output = resolve(args.output_root)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output root is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    task_cfg = yaml.safe_load(resolve(args.task_config).read_text(encoding="utf-8"))
    instruction = task_cfg["task"]["language_instruction"]
    normal_count = max(1, int(args.count * 0.7))
    recovery_types = ["normal"] * normal_count
    recovery_kinds = ("lateral_offset", "angular_offset")
    for index in range(args.count - normal_count):
        recovery_types.append(recovery_kinds[index % len(recovery_kinds)])
    episodes = []
    for index, recovery_type in enumerate(recovery_types):
        name = f"episode_{index:06d}"
        row = {
            "episode": name,
            "seed": int(args.seed + index),
            "recovery_type": recovery_type,
            "language_instruction": instruction,
            "task_config": args.task_config,
            "status": "PLANNED",
        }
        episode_dir = output / name
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "metadata.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        episodes.append(row)
    manifest = {
        "schema": "qiling_groot_assemble.slender_pin_recovery_manifest.v1",
        "task_config": args.task_config,
        "episode_count": len(episodes),
        "normal_count": recovery_types.count("normal"),
        "recovery_count": len(episodes) - recovery_types.count("normal"),
        "episodes": episodes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "manifest": str(output / "manifest.json"),
        "episodes": len(episodes),
        "normal": recovery_types.count("normal"),
        "recovery": len(episodes) - recovery_types.count("normal"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
