#!/usr/bin/env python3
"""Headlessly record the slender-pin smoke manifest with three synchronized cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="datasets/raw_slender_pin_v1")
    parser.add_argument("--recorded-root", default="datasets/recorded_slender_pin_v1")
    parser.add_argument("--summary-json", default="reports/slender_pin_v1_summary.json")
    parser.add_argument("--episode-timeout-seconds", type=float, default=420.0)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Record only the named episode; repeat this option to select several.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def complete(path: Path) -> bool:
    return (path / "metadata.json").exists() and (path / "episode.npz").exists() and all(
        (path / "videos" / f"{name}.mp4").exists()
        for name in ("chest", "left_wrist", "right_wrist")
    )


def main() -> int:
    args = parse_args()
    raw_root = resolve(args.raw_root)
    recorded_root = resolve(args.recorded_root)
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if args.episode:
        requested_names = set(args.episode)
        episodes = [row for row in episodes if row["episode"] in requested_names]
        found_names = {row["episode"] for row in episodes}
        missing_names = sorted(requested_names - found_names)
        if missing_names:
            raise ValueError(f"Episodes not found in manifest: {missing_names}")
    if recorded_root.exists() and not args.resume:
        raise FileExistsError(f"Recorded root exists: {recorded_root}; use --resume")
    recorded_root.mkdir(parents=True, exist_ok=True)
    results = []
    started_all = time.monotonic()
    for index, episode in enumerate(episodes, start=1):
        name = episode["episode"]
        output = recorded_root / name
        if args.resume and complete(output):
            results.append({"episode": name, "status": "PASS", "resumed": True})
            continue
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        log_path = output / "record.log"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_handle_pin_grasp_gui.py"),
            "--headless",
            "--complete-insertion",
            "--task-config", str(episode.get(
                "task_config", "configs/task_slender_pin_insertion_right_arm.yaml"
            )),
            "--record-out-dir", str(output),
            "--recovery-type", str(episode["recovery_type"]),
            "--episode-seed", str(int(episode["seed"])),
        ]
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.episode_timeout_seconds, check=False,
            )
        metadata_path = output / "metadata.json"
        passed = process.returncode == 0 and complete(output)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            passed = passed and metadata.get("status") == "PASS" and bool(metadata.get("task_success"))
        row = {
            "episode": name,
            "recovery_type": episode["recovery_type"],
            "status": "PASS" if passed else "FAIL",
            "returncode": process.returncode,
            "wall_seconds": time.monotonic() - started,
            "log": str(log_path.relative_to(ROOT)),
        }
        results.append(row)
        print(f"[slender-record] {index}/{len(episodes)} {name} {row['status']}", flush=True)
    successful = sum(row["status"] == "PASS" for row in results)
    summary = {
        "status": "PASS" if successful == len(episodes) else "PARTIAL",
        "requested": len(episodes),
        "successful": successful,
        "failed": len(episodes) - successful,
        "success_rate": successful / float(len(episodes)),
        "raw_root": str(raw_root.relative_to(ROOT)),
        "recorded_root": str(recorded_root.relative_to(ROOT)),
        "camera_storage": "three H.264 MP4 files per episode",
        "record_hz": 20.0,
        "wall_seconds": time.monotonic() - started_all,
        "results": results,
    }
    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if successful == len(episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
