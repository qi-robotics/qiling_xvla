#!/usr/bin/env python3
"""Shared LeRobot v3 conversion for expert EEF recordings (17D state / 10D action)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


ROOT = Path(__file__).resolve().parents[1]
EPISODE_RE = re.compile(r"episode_(\d{6})$")
CAMERAS = ("chest", "left_wrist", "right_wrist")
FPS = 20
IMAGE_SHAPE = (480, 640, 3)

RIGHT_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "RH_thumb_cmc_yaw",
    "RH_thumb_cmc_pitch",
    "RH_index_mcp_pitch",
    "RH_middle_mcp_pitch",
    "RH_ring_mcp_pitch",
    "RH_pinky_mcp_pitch",
]

EEF_STATE_NAMES = [
    "right_eef.position.x",
    "right_eef.position.y",
    "right_eef.position.z",
    "right_eef.rotation6d.0",
    "right_eef.rotation6d.1",
    "right_eef.rotation6d.2",
    "right_eef.rotation6d.3",
    "right_eef.rotation6d.4",
    "right_eef.rotation6d.5",
    *RIGHT_JOINT_NAMES[:7],
    "right_hand_grasp",
]

EEF_ACTION_NAMES = EEF_STATE_NAMES[:9] + ["right_hand_grasp"]


@dataclass(frozen=True)
class ConversionSpec:
    mode: str
    default_output_dir: str
    default_repo_id: str
    default_report: str
    default_recorded_root: str
    robot_type: str
    state_key: str
    action_key: str
    state_width: int
    action_width: int
    state_names: list[str]
    action_names: list[str]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def discover_episode_dirs(root: Path) -> list[Path]:
    rows: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = EPISODE_RE.fullmatch(child.name) if child.is_dir() else None
        if match:
            rows.append((int(match.group(1)), child))
    return [path for _, path in sorted(rows)]


def source_is_valid(episode_dir: Path) -> tuple[bool, str]:
    metadata_path = episode_dir / "metadata.json"
    episode_path = episode_dir / "episode.npz"
    if not metadata_path.is_file() or not episode_path.is_file():
        return False, "missing metadata.json or episode.npz"
    metadata = load_json(metadata_path)
    if metadata.get("status") != "PASS" or not metadata.get("task_success"):
        return False, "recording metadata is not PASS"
    validation_path = episode_dir / "validation_report.json"
    if validation_path.is_file() and load_json(validation_path).get("status") != "PASS":
        return False, "validation report is not PASS"
    for camera in CAMERAS:
        if not (episode_dir / "videos" / f"{camera}.mp4").is_file():
            return False, f"missing {camera} video"
    return True, ""


def dataset_features(spec: ConversionSpec) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (spec.state_width,),
            "names": spec.state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (spec.action_width,),
            "names": spec.action_names,
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        }
    return features


def open_cameras(episode_dir: Path) -> dict[str, cv2.VideoCapture]:
    captures = {
        camera: cv2.VideoCapture(str(episode_dir / "videos" / f"{camera}.mp4"))
        for camera in CAMERAS
    }
    failed = [camera for camera, capture in captures.items() if not capture.isOpened()]
    if failed:
        for capture in captures.values():
            capture.release()
        raise RuntimeError(f"{episode_dir}: cannot open cameras {failed}")
    return captures


def read_rgb_frames(
    captures: dict[str, cv2.VideoCapture], episode_dir: Path, frame_index: int
) -> dict[str, np.ndarray]:
    images = {}
    for camera, capture in captures.items():
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise RuntimeError(
                f"{episode_dir}: {camera} ended before frame {frame_index}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != IMAGE_SHAPE:
            raise ValueError(
                f"{episode_dir}: {camera} frame {frame_index} has shape {rgb.shape}; "
                f"expected {IMAGE_SHAPE}"
            )
        images[f"observation.images.{camera}"] = np.ascontiguousarray(rgb)
    return images


def convert(spec: ConversionSpec, args: argparse.Namespace) -> dict[str, Any]:
    recorded_root = resolve(args.recorded_root)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.json_out)
    if not recorded_root.is_dir():
        raise FileNotFoundError(recorded_root)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)

    all_dirs = discover_episode_dirs(recorded_root)
    accepted: list[Path] = []
    skipped: list[dict[str, str]] = []
    for episode_dir in all_dirs:
        valid, reason = source_is_valid(episode_dir)
        if valid:
            accepted.append(episode_dir)
        else:
            skipped.append({"source_episode": episode_dir.name, "reason": reason})
    if args.max_episodes is not None:
        accepted = accepted[: args.max_episodes]
    if not accepted:
        raise RuntimeError(f"No validated recordings found in {recorded_root}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=FPS,
        robot_type=spec.robot_type,
        features=dataset_features(spec),
        use_videos=True,
        image_writer_threads=0 if args.streaming_encoding else 4,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
        vcodec=args.video_codec,
        video_backend="pyav",
    )

    source_rows = []
    total_frames = 0
    try:
        for output_index, episode_dir in enumerate(accepted):
            metadata = load_json(episode_dir / "metadata.json")
            instruction = str(metadata["language_instruction"])
            with np.load(episode_dir / "episode.npz", allow_pickle=False) as data:
                state = np.asarray(data[spec.state_key], dtype=np.float32)
                action = np.asarray(data[spec.action_key], dtype=np.float32)
                phase = np.asarray(data["phase"]).astype(str)
                expert_mask = np.asarray(data["expert_mask"], dtype=bool)
            if state.shape != (len(state), spec.state_width):
                raise ValueError(f"{episode_dir}: invalid state shape {state.shape}")
            if action.shape != (len(state), spec.action_width):
                raise ValueError(f"{episode_dir}: invalid action shape {action.shape}")
            if len(action) != len(state) or phase.shape != (len(state),):
                raise ValueError(f"{episode_dir}: frame arrays are not aligned")
            if expert_mask.shape != (len(state),) or not bool(np.all(expert_mask)):
                raise ValueError(f"{episode_dir}: contains invalid expert labels")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{episode_dir}: state/action contains NaN or inf")

            captures = open_cameras(episode_dir)
            try:
                for frame_index in range(len(state)):
                    frame = {
                        "observation.state": state[frame_index],
                        "action": action[frame_index],
                        "task": instruction,
                    }
                    frame.update(read_rgb_frames(captures, episode_dir, frame_index))
                    dataset.add_frame(frame)
                for camera, capture in captures.items():
                    extra, _ = capture.read()
                    if extra:
                        raise ValueError(
                            f"{episode_dir}: {camera} contains more than {len(state)} frames"
                        )
            finally:
                for capture in captures.values():
                    capture.release()

            dataset.save_episode(parallel_encoding=not args.streaming_encoding)
            source_rows.append(
                {
                    "episode_index": output_index,
                    "source_episode": episode_dir.name,
                    "source_frame_count": len(state),
                    "phases": list(dict.fromkeys(phase.tolist())),
                    "rj45_plug_world_xyz": metadata.get("rj45_plug_world_xyz"),
                    "insertion_depth_m": metadata.get("insertion_depth_m"),
                    "lateral_error_m": metadata.get("lateral_error_m"),
                }
            )
            total_frames += len(state)
            print(
                f"[rj45-{spec.mode}-v3] {output_index + 1}/{len(accepted)} "
                f"{episode_dir.name} frames={len(state)}",
                flush=True,
            )
    finally:
        dataset.finalize()

    converted_metadata = LeRobotDatasetMetadata(args.repo_id, root=output_dir)
    if converted_metadata.total_episodes != len(accepted):
        raise RuntimeError("LeRobot metadata episode count does not match conversion")
    if converted_metadata.total_frames != total_frames:
        raise RuntimeError("LeRobot metadata frame count does not match conversion")

    manifest = {
        "schema": "qiling_groot_assemble.lerobot_v3_conversion.v1",
        "mode": spec.mode,
        "source": str(recorded_root),
        "repo_id": args.repo_id,
        "state_source": spec.state_key,
        "action_source": spec.action_key,
        "state_dim": spec.state_width,
        "action_dim": spec.action_width,
        "camera_names": list(CAMERAS),
        "fps": FPS,
        "episodes": source_rows,
        "skipped": skipped,
    }
    write_json(output_dir / "conversion_manifest.json", manifest)
    report = {
        "status": "PASS",
        "mode": spec.mode,
        "dataset_dir": str(output_dir),
        "repo_id": args.repo_id,
        "source_directory_count": len(all_dirs),
        "converted_episode_count": len(accepted),
        "skipped_episode_count": len(skipped),
        "total_frames": total_frames,
        "state_dim": spec.state_width,
        "action_dim": spec.action_width,
        "video_count": len(accepted) * len(CAMERAS),
        "codebase_version": str(converted_metadata.info["codebase_version"]),
        "skipped": skipped,
    }
    write_json(report_path, report)
    return report


def run_cli(spec: ConversionSpec) -> int:
    parser = argparse.ArgumentParser(
        description=f"Convert expert recordings to LeRobot v3 for {spec.mode}."
    )
    parser.add_argument(
        "--recorded-root",
        default=spec.default_recorded_root,
    )
    parser.add_argument("--output-dir", default=spec.default_output_dir)
    parser.add_argument("--repo-id", default=spec.default_repo_id)
    parser.add_argument("--json-out", default=spec.default_report)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=90)
    parser.add_argument("--encoder-threads", type=int, default=2)
    args = parser.parse_args()
    report = convert(spec, args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
