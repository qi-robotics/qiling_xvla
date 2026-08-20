#!/usr/bin/env python3
"""Validate slender-pin recorded episodes and all three camera streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


STATE_DIM = 17
ACTION_DIM = 10
JOINT_DIM = 13


def validate_episode(path: Path, limits: dict) -> dict:
    errors = []
    metadata_path = path / "metadata.json"
    episode_path = path / "episode.npz"
    if not metadata_path.exists():
        return {"status": "FAIL", "episode": path.name, "frame_count": 0,
                "recovery_type": None, "phase_counts": {},
                "errors": ["missing metadata.json"]}
    if not episode_path.exists():
        return {"status": "FAIL", "episode": path.name, "frame_count": 0,
                "recovery_type": None, "phase_counts": {},
                "errors": ["missing episode.npz"]}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = 0
    with np.load(episode_path, allow_pickle=False) as data:
        required = {
            "observation_state", "target_observation_state", "action", "timestamp",
            "phase", "actual_joint_position", "target_joint_position",
        }
        missing = sorted(required - set(data.files))
        if missing:
            errors.append(f"missing arrays: {missing}")
        if "observation_state" not in data.files:
            return {"status": "FAIL", "episode": path.name, "frame_count": 0,
                    "recovery_type": metadata.get("recovery_type"),
                    "phase_counts": metadata.get("phase_counts", {}),
                    "errors": errors}
        frame_count = int(np.asarray(data["observation_state"]).shape[0])
        expected = {
            "observation_state": (frame_count, STATE_DIM),
            "target_observation_state": (frame_count, STATE_DIM),
            "action": (frame_count, ACTION_DIM),
            "timestamp": (frame_count,),
            "phase": (frame_count,),
            "actual_joint_position": (frame_count, JOINT_DIM),
            "target_joint_position": (frame_count, JOINT_DIM),
        }
        for name, shape in expected.items():
            if name not in data.files:
                continue
            if np.asarray(data[name]).shape != shape:
                errors.append(f"{name} shape {np.asarray(data[name]).shape}, expected {shape}")
            if name != "phase" and not np.isfinite(np.asarray(data[name])).all():
                errors.append(f"{name} contains NaN or inf")
        timestamps = np.asarray(data["timestamp"]) if "timestamp" in data.files else np.empty(0)
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 0.05, atol=1e-6):
            errors.append("timestamps are not continuous at 20 Hz")
    for name in ("chest", "left_wrist", "right_wrist"):
        video = path / "videos" / f"{name}.mp4"
        if not video.exists() or video.stat().st_size == 0:
            errors.append(f"missing or empty video: {name}")
            continue
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            errors.append(f"cannot decode video: {name}")
            continue
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        decoded = 0
        sampled_std = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if decoded == 1 or decoded % max(1, frame_count // 5) == 0:
                sampled_std.append(float(frame.std()))
        capture.release()
        if (width, height) != (640, 480):
            errors.append(f"{name} resolution {(width, height)}, expected (640, 480)")
        if fps > 0.0 and abs(fps - 20.0) > 1.0:
            errors.append(f"{name} fps {fps:.2f}, expected about 20")
        if decoded != frame_count:
            errors.append(f"{name} decoded {decoded} frames, expected {frame_count}")
        if not sampled_std or max(sampled_std) < 1.0:
            errors.append(f"{name} appears blank in sampled frames")
    phase_counts = metadata.get("phase_counts", {})
    if metadata.get("recovery_type") != "normal" and not phase_counts.get("recovery_inject"):
        errors.append("recovery episode has no recovery_inject phase")
    if metadata.get("status") != "PASS" or not metadata.get("task_success"):
        errors.append("episode metadata is not PASS/task_success")
    if int(metadata.get("state_dim", 0)) != STATE_DIM:
        errors.append(f"state_dim {metadata.get('state_dim')}, expected {STATE_DIM}")
    if int(metadata.get("action_dim", 0)) != ACTION_DIM:
        errors.append(f"action_dim {metadata.get('action_dim')}, expected {ACTION_DIM}")
    lift = float(metadata.get("physical_grasp_lift_m", 0.0))
    if lift < limits["minimum_lift_m"]:
        errors.append(f"physical grasp lift {lift * 1000.0:.2f} mm is too small")
    depth = float(metadata.get("insertion_depth_m", 0.0))
    if depth < limits["minimum_depth_m"]:
        errors.append(f"insertion depth {depth * 1000.0:.2f} mm is too shallow")
    lateral = float(metadata.get("lateral_error_m", 1.0))
    if lateral > limits["maximum_lateral_m"]:
        errors.append(f"final lateral error {lateral * 1000.0:.2f} mm is too large")
    orientation_deg = math.degrees(float(metadata.get("orientation_error_rad", math.pi)))
    if orientation_deg > limits["maximum_orientation_deg"]:
        errors.append(f"final orientation error {orientation_deg:.2f} deg is too large")
    stable_frames = int(metadata.get("insertion_stable_frames", 0))
    if stable_frames < limits["minimum_stable_frames"]:
        errors.append(f"seated stability hold is only {stable_frames} frames")
    if int(phase_counts.get("seated_stability_hold", 0)) < limits["minimum_stable_frames"]:
        errors.append("recorded seated_stability_hold phase is too short")
    drift = float(metadata.get("seated_depth_drift_m", 1.0))
    if drift > limits["maximum_seated_drift_m"]:
        errors.append(f"seated depth drifted {drift * 1000.0:.2f} mm during the hold")
    return {
        "status": "PASS" if not errors else "FAIL",
        "episode": path.name,
        "frame_count": frame_count,
        "recovery_type": metadata.get("recovery_type"),
        "phase_counts": metadata.get("phase_counts", {}),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-root", default="datasets/recorded_slender_pin_v1")
    parser.add_argument("--json-out", default="reports/slender_pin_v1_validation.json")
    parser.add_argument("--minimum-lift-m", type=float, default=0.015)
    parser.add_argument("--minimum-depth-m", type=float, default=0.040)
    parser.add_argument("--maximum-lateral-m", type=float, default=0.0036)
    parser.add_argument("--maximum-orientation-deg", type=float, default=8.0)
    parser.add_argument("--minimum-stable-frames", type=int, default=20)
    parser.add_argument("--maximum-seated-drift-m", type=float, default=0.0010)
    args = parser.parse_args()
    limits = {
        "minimum_lift_m": args.minimum_lift_m,
        "minimum_depth_m": args.minimum_depth_m,
        "maximum_lateral_m": args.maximum_lateral_m,
        "maximum_orientation_deg": args.maximum_orientation_deg,
        "minimum_stable_frames": args.minimum_stable_frames,
        "maximum_seated_drift_m": args.maximum_seated_drift_m,
    }
    root = Path(args.recorded_root)
    reports = [
        validate_episode(path, limits)
        for path in sorted(root.glob("episode_*"))
        if path.is_dir()
    ]
    result = {
        "status": "PASS" if reports and all(row["status"] == "PASS" for row in reports) else "FAIL",
        "episode_count": len(reports),
        "passed": sum(row["status"] == "PASS" for row in reports),
        "reports": reports,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
