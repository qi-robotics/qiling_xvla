#!/usr/bin/env python3
"""Synchronous closed-loop XVLA rollout for slender-pin insertion in Isaac.

Architecture
------------
Isaac Sim owns the scene, cameras, IK, and hand mapping. The trained XVLA
policy runs in an isolated LeRobot subprocess (``XVLA_POLICY_PYTHON``).
Unlike chunk-based pi0.5 rollout, XVLA uses an internal action queue:
per-step calls return one action at a time from a pre-generated 32-step chunk.

Control loop
------------
    for each policy call:
        1. observe (cameras + robot state)
        2. reset XVLA queue           <- call "reset" IPC before new obs
        3. predict_action             <- XVLA select_action: returns 1 action
        4. execute that 1 action via IK
    (re-observe every step, i.e. execution_horizon=1 is the default)

With --execution-horizon N > 1 the same observation is reused: the policy
server keeps its queue alive and pops the next action for steps 2..N without
a new image, then we re-observe at step N+1.  Queue is reset at every new obs.

Inputs to XVLA
--------------
  RGB 480×640: chest, left_wrist, right_wrist  (uint8, renamed internally by preprocessor)
  state 17D: xyz(3) + rot6d(6) + arm_q(7) + grasp(1)
  text instruction from task config
  (no privileged pin/socket pose)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from isaacsim import SimulationApp

from run_rj45_isaac_handoff_scene import (
    build_scene,
    coupled_hand_joint_names,
    coupled_hand_positions,
    load_yaml,
    quaternion_xyzw_to_matrix,
    startup_joint_names,
)
from run_rj45_fixed_socket_scene import (
    apply_left_observation_pose,
    apply_task_hand_presets,
    validate_fixed_upward_socket,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CAMERA_NAMES = ("chest", "left_wrist", "right_wrist")
ACTION_DIM_NAMES = (
    "right_eef.position.x",
    "right_eef.position.y",
    "right_eef.position.z",
    "right_eef.rotation6d.0",
    "right_eef.rotation6d.1",
    "right_eef.rotation6d.2",
    "right_eef.rotation6d.3",
    "right_eef.rotation6d.4",
    "right_eef.rotation6d.5",
    "right_hand_grasp",
)
STATE_DIM_NAMES = ACTION_DIM_NAMES[:9] + (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_grasp",
)
DEFAULT_CHECKPOINT = (
    "outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/last/pretrained_model"
)
DEFAULT_DATASET = "datasets/slender_pin_lerobot_v3_pi05_v1"
HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ROBOT_TYPE = "S4_RIGHT_ARM_O6_FIXED_SOCKET_RJ45_XVLA"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", "--episode-seed", type=int, default=0, dest="seed")
    parser.add_argument(
        "--plug-xyz", "--pin-xyz", nargs=3, type=float, default=None, dest="plug_xyz"
    )
    parser.add_argument("--reference-episode", type=int, default=None)
    parser.add_argument(
        "--reference-recorded-root", default="datasets/recorded_slender_pin_v1"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to pretrained_model. Default is checkpoints/last "
            "(currently 200000). For step 150000 use "
            "outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/150000/pretrained_model"
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-seed", type=int, default=3)
    parser.add_argument(
        "--execution-horizon", type=int, default=1,
        help=(
            "Number of steps to execute before re-observing. "
            "1 = every step re-observes (recommended for XVLA). "
            "N > 1 reuses the same observation for N steps (queue stays open)."
        ),
    )
    parser.add_argument("--max-policy-calls", type=int, default=800)
    parser.add_argument("--max-control-steps", type=int, default=800)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--verbose-actions", action="store_true")
    parser.add_argument(
        "--out-dir", default="outputs/slender_pin_xvla_rollout/rollout_000000"
    )
    parser.add_argument(
        "--task-config", default="configs/task_slender_pin_insertion_right_arm.yaml"
    )
    parser.add_argument("--robot-config", default="configs/robot_dual_arm.yaml")
    parser.add_argument("--camera-config", default="configs/camera_bimanual.yaml")
    parser.add_argument("--stl-scale", type=float, default=1.0)
    parser.add_argument("--max-eef-step-m", type=float, default=0.025)
    parser.add_argument("--max-eef-rotation-step-deg", type=float, default=12.0)
    parser.add_argument("--max-grasp-step", type=float, default=0.06)
    parser.add_argument(
        "--eef-min-xyz-base", nargs=3, type=float, default=[0.35, -0.32, 0.13]
    )
    parser.add_argument(
        "--eef-max-xyz-base", nargs=3, type=float, default=[0.57, -0.02, 0.44]
    )
    parser.add_argument("--ik-max-iters", type=int, default=800)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=0.002)
    parser.add_argument("--ik-rotation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--ik-hard-position-error-m", type=float, default=0.008)
    parser.add_argument("--ik-hard-rotation-error-deg", type=float, default=8.0)
    parser.add_argument("--success-stable-steps", type=int, default=20)
    parser.add_argument("--open-grasp-threshold", type=float, default=0.20)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def resolve_policy_python() -> str:
    """Interpreter that has LeRobot XVLA installed.

    Set ``XVLA_POLICY_PYTHON`` to override. Falls back to ``qiling_xvla``,
    then a local legacy env if present.
    """
    override = os.environ.get("XVLA_POLICY_PYTHON")
    if override:
        return override
    candidates = [
        Path.home() / "miniconda3/envs/qiling_xvla/bin/python",
        Path.home() / "anaconda3/envs/qiling_xvla/bin/python",
        Path.home() / "miniconda3/envs/so101_lerobot/bin/python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "Set XVLA_POLICY_PYTHON to the Python that has LeRobot XVLA installed, "
        "for example: export XVLA_POLICY_PYTHON=$HOME/miniconda3/envs/qiling_xvla/bin/python"
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# IPC helpers (identical protocol to pi0.5 server)
# ---------------------------------------------------------------------------

def recv_exact(stream, size: int, shutdown: dict[str, Any] | None = None) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if shutdown is not None and shutdown.get("requested"):
            raise InterruptedError("rollout interrupted")
        ready, _, _ = select.select([stream], [], [], 0.25)
        if not ready:
            continue
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("XVLA policy server disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(stream, shutdown: dict[str, Any] | None = None) -> Any:
    size = HEADER.unpack(recv_exact(stream, HEADER.size, shutdown))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid XVLA response size: {size}")
    return pickle.loads(recv_exact(stream, size, shutdown))  # noqa: S301


def send_message(stream, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"XVLA request is too large: {len(payload)}")
    stream.write(HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


class XVLAPolicyClient:
    def __init__(
        self,
        process: subprocess.Popen,
        shutdown: dict[str, Any] | None = None,
    ) -> None:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("XVLA policy process does not expose stdio pipes")
        self.process = process
        self.input = process.stdin
        self.output = process.stdout
        self.shutdown = shutdown

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"XVLA policy process exited with code {self.process.returncode}"
            )
        send_message(self.input, request)
        response = recv_message(self.output, self.shutdown)
        if not isinstance(response, dict) or not response.get("ok"):
            raise RuntimeError(
                response.get("error", f"invalid XVLA response: {response!r}")
            )
        return response


def start_policy_server(args: argparse.Namespace) -> subprocess.Popen:
    checkpoint = resolve(args.checkpoint)
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        environment.pop(name, None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    python = resolve_policy_python()
    command = [
        python,
        str(ROOT / "scripts/serve_xvla_policy.py"),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.policy_device,
        "--seed",
        str(args.policy_seed),
    ]
    print(
        f"[Slender-pin XVLA] starting isolated policy server "
        f"python={python} checkpoint={checkpoint} policy_seed={args.policy_seed}",
        flush=True,
    )
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )


def _kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=2.0)
    except Exception:
        pass


def stop_policy_server(
    process: subprocess.Popen | None,
    client: XVLAPolicyClient | None,
    *,
    force: bool = False,
) -> None:
    if process is None:
        return
    if force:
        _kill_process_group(process)
        return
    try:
        if client is not None and process.poll() is None:
            client.call({"type": "shutdown"})
    except Exception:
        pass
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from pi0.5 rollout)
# ---------------------------------------------------------------------------

def sample_pin_xyz(task_cfg: dict, seed: int) -> list[float]:
    scene = task_cfg["scene"]
    sampled = np.asarray(scene["rj45_plug_initial_xyz"], dtype=float)
    randomization = scene.get("rj45_plug_position_randomization_m")
    if randomization is not None:
        rng = np.random.default_rng(int(seed))
        sampled[0] += rng.uniform(*map(float, randomization["x"]))
        sampled[1] += rng.uniform(*map(float, randomization["y"]))
    scene["rj45_plug_initial_xyz"] = sampled.tolist()
    return [float(value) for value in sampled]


def world_pose(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom
    matrix = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    xyz = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        np.asarray([xyz[0], xyz[1], xyz[2]], dtype=float),
        quaternion_xyzw_to_matrix(
            [imaginary[0], imaginary[1], imaginary[2], quaternion.GetReal()]
        ),
    )


def rotation_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=float).reshape(3, 3)[:, :2].T.reshape(-1)


def rot6d_to_rotation(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=float).reshape(2, 3)
    first = values[0]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1.0e-6:
        raise ValueError("Predicted rot6d first axis is degenerate")
    first = first / first_norm
    second = values[1] - float(values[1] @ first) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1.0e-6:
        raise ValueError("Predicted rot6d second axis is degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(cosine))


def rotation_to_rpy_zyx_deg(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-6:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees(np.asarray([roll, pitch, yaw], dtype=float))


def limit_rotation_step(
    current: np.ndarray, target: np.ndarray, max_angle: float
) -> np.ndarray:
    relative = np.asarray(target) @ np.asarray(current).T
    angle = rotation_angle(relative)
    if angle <= max_angle or angle < 1.0e-9:
        return np.asarray(target, dtype=float)
    skew = np.asarray([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    sine = float(np.linalg.norm(skew)) * 0.5
    axis = np.asarray([1.0, 0.0, 0.0]) if sine < 1.0e-8 else skew / (2.0 * sine)
    cross = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    increment = np.eye(3) + math.sin(max_angle) * cross + (1.0 - math.cos(max_angle)) * (
        cross @ cross
    )
    return increment @ np.asarray(current)


def limit_xyz_step(
    current: np.ndarray, target: np.ndarray, max_step: float
) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance > float(max_step) and distance > 1.0e-12:
        return current + delta * (float(max_step) / distance)
    return target


def grasp_scalar(hand_q: np.ndarray, robot_cfg: dict) -> float:
    hand = robot_cfg["right_o6_hand"]
    names = hand["active_driver_joints"]
    opened = np.asarray([hand["open_preset"][name] for name in names], dtype=float)
    closed = np.asarray([hand["pinch_preset"][name] for name in names], dtype=float)
    delta = closed - opened
    return float(np.clip((np.asarray(hand_q) - opened) @ delta / (delta @ delta), 0.0, 1.0))


def hand_target(value: float, robot_cfg: dict) -> np.ndarray:
    hand = robot_cfg["right_o6_hand"]
    names = hand["active_driver_joints"]
    opened = np.asarray([hand["open_preset"][name] for name in names], dtype=float)
    approach = np.asarray([hand["approach_preset"][name] for name in names], dtype=float)
    pinched = np.asarray([hand["pinch_preset"][name] for name in names], dtype=float)
    pinch_delta = pinched - opened
    denominator = float(pinch_delta @ pinch_delta)
    if denominator < 1.0e-12:
        raise ValueError("Right-hand open and pinch presets must differ")
    approach_scalar = float(
        np.clip((approach - opened) @ pinch_delta / denominator, 0.0, 1.0)
    )
    if not 1.0e-6 < approach_scalar < 1.0 - 1.0e-6:
        raise ValueError(f"Invalid projected approach scalar: {approach_scalar}")
    scalar = float(np.clip(value, 0.0, 1.0))
    if scalar <= approach_scalar:
        alpha = scalar / approach_scalar
        return opened + alpha * (approach - opened)
    alpha = (scalar - approach_scalar) / (1.0 - approach_scalar)
    return approach + alpha * (pinched - approach)


def load_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    return json.loads(info_path.read_text(encoding="utf-8"))


def describe_action(action: np.ndarray) -> str:
    values = np.asarray(action, dtype=float).reshape(-1)
    xyz = values[:3]
    grasp = float(values[9]) if values.size >= 10 else float("nan")
    return (
        f"xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
        f"grasp={grasp:.3f} dim={values.size}"
    )


# ---------------------------------------------------------------------------
# Policy IPC call
# ---------------------------------------------------------------------------

def predict_one_action(
    client: XVLAPolicyClient,
    images: dict[str, np.ndarray],
    state: np.ndarray,
    instruction: str,
    reset_queue: bool,
) -> tuple[np.ndarray, float]:
    """Send one observation to XVLA server, get back a single (10,) action."""
    if reset_queue:
        client.call({"type": "reset"})

    request = {
        "type": "predict",
        "state_bytes": np.asarray(state, dtype=np.float32).tobytes(order="C"),
        "image_bytes": {
            name: np.ascontiguousarray(images[name], dtype=np.uint8).tobytes(order="C")
            for name in CAMERA_NAMES
        },
        "instruction": instruction,
        "robot_type": ROBOT_TYPE,
    }
    response = client.call(request)
    shape = tuple(int(v) for v in response["action_shape"])
    action = np.frombuffer(response["action_bytes"], dtype=np.float32).reshape(shape)
    if action.shape[-1] != 10:
        raise RuntimeError(f"Unexpected XVLA action shape: {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError("XVLA returned NaN or inf")
    raw = np.asarray(action[0], dtype=np.float32)  # (10,)
    return raw, float(response["inference_ms"])


# ---------------------------------------------------------------------------
# Video writer (unchanged)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

class _RgbVideoWriter:
    """Write RGB frames to a playable MP4.

    OpenCV `mp4v` only writes the moov index on release(), so Ctrl+C leaves a
    dead file. ffmpeg fragmented MP4 is playable even if the process dies
    mid-run; we remux to a regular MP4 on clean close.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._tmp_path = path.with_suffix(".frag.mp4")
        self._width = int(width)
        self._height = int(height)
        self._proc: subprocess.Popen | None = None
        self._cv_writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self._width}x{self._height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                str(self._tmp_path),
            ]
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(
            str(path), fourcc, fps, (self._width, self._height)
        )

    def write(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"video frame shape {frame.shape} != {(self._height, self._width, 3)}"
            )
        if self._proc is not None:
            if self._proc.stdin is None:
                return
            self._proc.stdin.write(frame.tobytes(order="C"))
            self._proc.stdin.flush()
            return
        import cv2

        self._cv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self, commit: bool = True) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=8.0)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    pass
            self._proc = None
            if commit and self._tmp_path.is_file() and self._tmp_path.stat().st_size > 0:
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is not None:
                    remux = subprocess.run(
                        [
                            ffmpeg,
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(self._tmp_path),
                            "-c",
                            "copy",
                            "-movflags",
                            "+faststart",
                            str(self._path),
                        ],
                        check=False,
                    )
                    if remux.returncode != 0 or not self._path.is_file():
                        shutil.copy2(self._tmp_path, self._path)
                else:
                    shutil.copy2(self._tmp_path, self._path)
            if self._tmp_path.is_file():
                try:
                    self._tmp_path.unlink()
                except Exception:
                    pass
            if not commit and self._path.exists():
                self._path.unlink(missing_ok=True)
            return
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
            if not commit and self._path.exists():
                self._path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.plug_xyz is not None and args.reference_episode is not None:
        raise ValueError("Use only one of --pin-xyz/--plug-xyz and --reference-episode")
    if abs(float(args.control_hz) - 20.0) > 1.0e-6:
        print(
            f"[Slender-pin XVLA] WARNING: control_hz={args.control_hz} but dataset fps=20",
            flush=True,
        )

    dataset_dir = resolve(args.dataset)
    dataset_info = load_dataset_info(dataset_dir)

    out_dir = resolve(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Rollout output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    server_process = None
    policy_client = None
    app = None
    video_writers: dict[str, _RgbVideoWriter] = {}
    report: dict[str, Any] = {"status": "ERROR", "success": False}
    arrays: dict[str, np.ndarray] = {}

    observation_states: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []
    executed_eef: list[np.ndarray] = []
    executed_grasp: list[float] = []
    measured_grasp_list: list[float] = []
    actual_joint_positions: list[np.ndarray] = []
    insertion_history: list[list[float]] = []
    pin_xyz_hist: list[list[float]] = []
    pin_rpy_hist: list[list[float]] = []
    ik_errors: list[list[float]] = []
    policy_latencies: list[float] = []
    action_sources: list[list[int]] = []
    inference_ms_hist: list[float] = []
    shutdown = {"requested": False, "count": 0}
    outputs_saved = False

    def _on_signal(signum, _frame) -> None:
        shutdown["requested"] = True
        shutdown["count"] += 1
        print(
            f"\n[Slender-pin XVLA] signal {signum} "
            f"({shutdown['count']}) — finishing current step and saving. "
            "Press Ctrl+C again to force-kill the policy process.",
            flush=True,
        )
        if shutdown["count"] >= 2 and server_process is not None:
            _kill_process_group(server_process)
        if shutdown["count"] >= 3:
            persist_outputs()
            os._exit(130)

    def _install_signal_handlers() -> None:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

    def persist_outputs() -> None:
        nonlocal arrays, outputs_saved
        if outputs_saved:
            return
        outputs_saved = True
        try:
            if not arrays and observation_states:
                arrays = {
                    "observation_state": np.asarray(observation_states, dtype=np.float32),
                    "predicted_action": np.asarray(predicted_actions, dtype=np.float32),
                    "executed_right_eef": np.asarray(executed_eef, dtype=np.float32),
                    "executed_right_hand_grasp": np.asarray(
                        executed_grasp, dtype=np.float32
                    ).reshape(-1, 1),
                    "measured_grasp": np.asarray(
                        measured_grasp_list, dtype=np.float32
                    ).reshape(-1, 1),
                    "actual_right_joint_position": np.asarray(
                        actual_joint_positions, dtype=np.float32
                    ),
                    "insertion_metrics": np.asarray(insertion_history, dtype=np.float32),
                    "pin_xyz": np.asarray(pin_xyz_hist, dtype=np.float32),
                    "pin_rpy_world_deg": np.asarray(pin_rpy_hist, dtype=np.float32),
                    "ik_error": np.asarray(ik_errors, dtype=np.float32),
                    "policy_latency_ms": np.asarray(policy_latencies, dtype=np.float32),
                    "step_inference_ms": np.asarray(inference_ms_hist, dtype=np.float32),
                    "action_source": np.asarray(action_sources, dtype=np.int32),
                }
            commit_video = bool(arrays) or bool(observation_states) or bool(video_writers)
            for writer in list(video_writers.values()):
                try:
                    writer.close(commit=commit_video)
                except Exception as exc:
                    print(f"[Slender-pin XVLA] video close warning: {exc}", flush=True)
            video_writers.clear()
            chest_video = out_dir / "videos" / "chest.mp4"
            if commit_video and chest_video.is_file():
                shutil.copy2(chest_video, out_dir / "rollout.mp4")
            if arrays:
                np.savez_compressed(out_dir / "rollout.npz", **arrays)
            report.setdefault("executed_control_steps", len(observation_states))
            report.setdefault("policy_calls", len(policy_latencies))
            with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    report,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
                handle.write("\n")
            print(
                f"[Slender-pin XVLA] result={report.get('status')} "
                f"success={report.get('success')} steps={len(observation_states)} "
                f"out={out_dir}",
                flush=True,
            )
        except Exception as exc:
            print(f"[Slender-pin XVLA] persist warning: {exc}", flush=True)
            traceback.print_exc()

    _install_signal_handlers()

    try:
        # Load the XVLA checkpoint BEFORE Isaac allocates GPU resources.
        server_process = start_policy_server(args)
        policy_client = XVLAPolicyClient(server_process, shutdown=shutdown)

        ping = policy_client.call({"type": "ping"})
        schema = policy_client.call({"type": "schema"})
        print(
            f"[Slender-pin XVLA] policy ready checkpoint={ping.get('checkpoint')}",
            flush=True,
        )
        print(
            f"[Slender-pin XVLA] schema: "
            f"input_features={schema.get('input_features')} "
            f"output_features={schema.get('output_features')} "
            f"chunk_size={schema.get('chunk_size')} "
            f"n_action_steps={schema.get('n_action_steps')} "
            f"action_mode={schema.get('action_mode')} "
            f"max_state_dim={schema.get('max_state_dim')} "
            f"normalization={schema.get('normalization_mapping')}",
            flush=True,
        )
        policy_action_shape = schema["output_features"]["action"]["shape"]
        if list(policy_action_shape) != [10]:
            raise RuntimeError(f"XVLA action shape is {policy_action_shape}, expected [10]")

        # ------------------------------------------------------------------
        # Isaac Sim setup
        # ------------------------------------------------------------------
        app = SimulationApp({"headless": bool(args.headless), "width": 1280, "height": 720})
        _install_signal_handlers()
        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.camera import Camera
        from pxr import PhysxSchema, UsdPhysics

        from qiling_xvla.control.dual_arm_pinocchio import ArmPinocchioKinematics

        robot_cfg = load_yaml(ROOT / args.robot_config)
        task_cfg = load_yaml(ROOT / args.task_config)
        camera_cfg = load_yaml(ROOT / args.camera_config)
        validate_fixed_upward_socket(task_cfg)
        startup_scale = float(
            task_cfg.get("simulation", {}).get("startup_motion_duration_scale", 1.0)
        )
        for startup_phase in ("spread", "home"):
            robot_cfg["startup_motion"][startup_phase]["duration_hint"] *= startup_scale
        apply_task_hand_presets(robot_cfg, task_cfg)
        apply_left_observation_pose(robot_cfg, task_cfg)

        reference_episode_name = None
        if args.reference_episode is not None:
            reference_episode_name = f"episode_{int(args.reference_episode):06d}"
            reference_root = resolve(args.reference_recorded_root)
            reference_metadata_path = (
                reference_root / reference_episode_name / "metadata.json"
            )
            if not reference_metadata_path.exists():
                raise FileNotFoundError(reference_metadata_path)
            reference_metadata = json.loads(
                reference_metadata_path.read_text(encoding="utf-8")
            )
            if reference_metadata.get("status") != "PASS":
                raise ValueError(
                    f"Reference episode is not PASS: {reference_episode_name}"
                )
            plug_xyz = [float(v) for v in reference_metadata["rj45_plug_world_xyz"]]
        elif args.plug_xyz is not None:
            plug_xyz = [float(v) for v in args.plug_xyz]
        else:
            plug_xyz = sample_pin_xyz(task_cfg, args.seed)

        task_cfg["scene"]["rj45_plug_initial_xyz"] = plug_xyz
        print(
            f"[Slender-pin XVLA] seed={args.seed} pin_xyz={np.round(plug_xyz, 8).tolist()} "
            f"reference_episode={reference_episode_name} synchronous=True "
            f"execution_horizon={int(args.execution_horizon)} privileged_policy_input=False",
            flush=True,
        )

        camera_paths, _, articulation_root, robot_model_root = build_scene(
            app, robot_cfg, task_cfg, camera_cfg, args.stl_scale,
            startup_realtime=not args.headless,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        _install_signal_handlers()
        articulation = Articulation(articulation_root)
        articulation.initialize()
        stage = omni.usd.get_context().get_stage()

        controlled_names = startup_joint_names(robot_cfg)
        controlled_indices = np.asarray(
            [articulation.get_dof_index(name) for name in controlled_names]
        )
        coupled_names = (
            coupled_hand_joint_names("left") + coupled_hand_joint_names("right")
        )
        coupled_indices = np.asarray(
            [articulation.get_dof_index(name) for name in coupled_names]
        )
        if np.any(controlled_indices < 0) or np.any(coupled_indices < 0):
            raise RuntimeError("Isaac articulation missing controlled or coupled joints")

        right_arm_indices = controlled_indices[13:20]
        right_hand_indices = controlled_indices[20:26]
        current_command = np.asarray(
            articulation.get_joint_positions(joint_indices=controlled_indices), dtype=float
        ).reshape(-1)
        left_target = current_command[:13].copy()

        right_kin = ArmPinocchioKinematics(
            ROOT / robot_cfg["robot"]["preferred_urdf_for_first_pass"],
            [ROOT],
            robot_cfg["right_arm"]["joints"],
            robot_cfg["robot"]["right_eef_link"],
        )
        q_seed = right_kin.set_arm(right_kin.neutral(), current_command[13:20])

        sensors: dict[str, Camera] = {}
        for index, (name, path) in enumerate(zip(CAMERA_NAMES, camera_paths)):
            settings = camera_cfg["cameras"][name]
            sensor = Camera(
                prim_path=path,
                name=f"slender_pin_xvla_{name}_{index}",
                resolution=(int(settings["width"]), int(settings["height"])),
            )
            sensor.initialize()
            sensors[name] = sensor
            if args.record_video:
                video_writers[name] = _RgbVideoWriter(
                    out_dir / "videos" / f"{name}.mp4",
                    int(settings["width"]),
                    int(settings["height"]),
                    float(args.control_hz),
                )
        for _ in range(12):
            app.update()

        last_images: dict[str, np.ndarray] = {}

        def read_images() -> dict[str, np.ndarray]:
            images = {}
            for name, sensor in sensors.items():
                settings = camera_cfg["cameras"][name]
                expected = (int(settings["height"]), int(settings["width"]), 4)
                rgba = np.asarray(sensor.get_rgba())
                for _ in range(6):
                    if rgba.shape == expected and float(rgba[:, :, :3].std()) >= 1.0:
                        break
                    app.update()
                    rgba = np.asarray(sensor.get_rgba())
                if rgba.shape == expected and float(rgba[:, :, :3].std()) >= 1.0:
                    images[name] = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
                elif name in last_images:
                    print(
                        f"[Slender-pin XVLA] camera {name} blank; reusing last good frame",
                        flush=True,
                    )
                    images[name] = last_images[name]
                else:
                    raise RuntimeError(f"Camera {name} is blank: {rgba.shape}")
            last_images.update(images)
            return images

        def actual_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            base_xyz, base_rotation = world_pose(stage, f"{robot_model_root}/base_link")
            palm_xyz, palm_rotation = world_pose(
                stage, f"{robot_model_root}/{robot_cfg['robot']['right_eef_link']}"
            )
            eef_xyz = base_rotation.T @ (palm_xyz - base_xyz)
            eef_rotation = base_rotation.T @ palm_rotation
            arm_q = np.asarray(
                articulation.get_joint_positions(joint_indices=right_arm_indices), dtype=float
            ).reshape(-1)
            hand_q = np.asarray(
                articulation.get_joint_positions(joint_indices=right_hand_indices), dtype=float
            ).reshape(-1)
            grasp = grasp_scalar(hand_q, robot_cfg)
            state = np.concatenate(
                [eef_xyz, rotation_to_rot6d(eef_rotation), arm_q, [grasp]]
            ).astype(np.float32)
            return state, eef_xyz, eef_rotation, arm_q, grasp

        expert = task_cfg["expert_auto_ik"]
        validation = expert["validation"]
        socket_xyz, socket_rotation = world_pose(stage, "/World/RJ45Socket")
        socket_entry = socket_xyz + socket_rotation @ np.asarray(
            expert["socket_entry_local_xyz_m"], dtype=float
        )
        opening_axis = socket_rotation @ np.asarray(
            expert["opening_axis_local_xyz"], dtype=float
        )
        opening_axis /= np.linalg.norm(opening_axis)
        pin_tip_local = np.asarray(expert["plug_tip_local_xyz_m"], dtype=float)
        pin_axis_local = np.asarray(expert["plug_axis_local_xyz"], dtype=float)
        min_depth = float(validation["minimum_inserted_depth_m"])
        max_depth = float(validation["maximum_inserted_depth_m"])
        max_lateral = float(validation["maximum_final_lateral_error_m"])
        max_orientation_deg = float(validation["maximum_final_orientation_error_deg"])

        def insertion_metrics() -> dict[str, float | bool | list[float]]:
            position, rotation = world_pose(stage, "/World/RJ45Plug")
            tip = position + rotation @ pin_tip_local
            delta = socket_entry - tip
            depth = float(delta @ opening_axis)
            lateral_vector = delta - depth * opening_axis
            lateral = float(np.linalg.norm(lateral_vector))
            pin_axis = rotation @ pin_axis_local
            orientation = math.acos(
                np.clip(float(pin_axis @ -opening_axis), -1.0, 1.0)
            )
            valid = bool(
                depth >= min_depth
                and depth <= max_depth
                and lateral <= max_lateral
                and math.degrees(orientation) <= max_orientation_deg
            )
            return {
                "depth_m": depth,
                "lateral_error_m": lateral,
                "orientation_error_rad": orientation,
                "valid": valid,
                "pin_xyz": [float(v) for v in position],
                "pin_rpy_deg": [float(v) for v in rotation_to_rpy_zyx_deg(rotation)],
            }

        instruction = str(task_cfg["task"]["language_instruction"])
        physics_hz = float(task_cfg["simulation"]["physics_hz"])
        dataset_fps = float(dataset_info.get("fps", 20))
        substeps = max(1, int(round(physics_hz / float(args.control_hz))))
        print(
            f"[Slender-pin XVLA] dataset_fps={dataset_fps} "
            f"control_hz={float(args.control_hz)} physics_hz={physics_hz} "
            f"substeps_per_action={substeps} instruction={instruction!r}",
            flush=True,
        )

        plug_body = UsdPhysics.RigidBodyAPI.Get(stage, "/World/RJ45Plug")
        if not plug_body:
            raise RuntimeError("Slender-pin rigid body API is missing")
        plug_body.GetKinematicEnabledAttr().Set(False)
        plug_physx = PhysxSchema.PhysxRigidBodyAPI.Get(stage, "/World/RJ45Plug")
        plug_physx.GetEnableCCDAttr().Set(True)
        initial_plug_world, _ = world_pose(stage, "/World/RJ45Plug")

        current_images = read_images()
        state, current_xyz, current_rotation, current_arm_q, current_grasp = actual_state()
        print(
            f"[Slender-pin XVLA] state shape={state.shape} "
            f"measured_eef={np.round(current_xyz, 5).tolist()} grasp={current_grasp:.3f}",
            flush=True,
        )

        last_commanded_xyz = current_xyz.copy()
        last_commanded_rotation = current_rotation.copy()
        last_commanded_grasp = float(current_grasp)

        insertion_streak = 0
        ever_inserted = False
        ever_lifted = False
        ever_closed = False
        release_damping_applied = False
        finish_reason = "max_control_steps"
        control_step = 0
        policy_calls = 0
        printed_first_action = False

        workspace_min = np.asarray(args.eef_min_xyz_base, dtype=float)
        workspace_max = np.asarray(args.eef_max_xyz_base, dtype=float)

        # ------------------------------------------------------------------
        # Control loop
        #
        # Each outer iteration: observe → (reset + predict 1 action) → execute N steps
        # where N = execution_horizon.  For N > 1, subsequent steps 2..N also call
        # predict (no re-observe) without queue reset so XVLA pops next queue entries.
        # ------------------------------------------------------------------
        _install_signal_handlers()
        while app.is_running() and control_step < int(args.max_control_steps):
            if shutdown["requested"]:
                finish_reason = "interrupted"
                break
            if policy_calls >= int(args.max_policy_calls):
                finish_reason = "max_policy_calls"
                break

            cycle_started = time.perf_counter()
            current_images = read_images()
            state, current_xyz, current_rotation, current_arm_q, current_grasp = actual_state()

            execute_n = int(args.execution_horizon)
            first_in_group = True

            for horizon_index in range(execute_n):
                if shutdown["requested"]:
                    finish_reason = "interrupted"
                    break
                if control_step >= int(args.max_control_steps):
                    break

                # Reset queue only at the start of each new obs group.
                raw_action, inference_ms = predict_one_action(
                    policy_client,
                    current_images,
                    state,
                    instruction,
                    reset_queue=first_in_group,
                )
                if first_in_group:
                    policy_calls += 1
                    policy_latencies.append(inference_ms)
                first_in_group = False

                if not printed_first_action:
                    printed_first_action = True
                    print(
                        f"[Slender-pin XVLA] first action inference={inference_ms:.1f}ms "
                        f"{describe_action(raw_action)}",
                        flush=True,
                    )
                    for idx, name in enumerate(ACTION_DIM_NAMES):
                        print(
                            f"  action[0][{idx:02d}] {name:<28} {float(raw_action[idx]): .6f}",
                            flush=True,
                        )
                    xyz0 = np.asarray(raw_action[:3], dtype=float)
                    if np.any(np.abs(xyz0) > 2.0):
                        print(
                            "[Slender-pin XVLA] WARNING: action xyz looks out of workspace; "
                            "unnormalization may be wrong",
                            flush=True,
                        )

                predicted_xyz = np.asarray(raw_action[:3], dtype=float)
                predicted_rotation = rot6d_to_rotation(raw_action[3:9])
                predicted_grasp = float(np.clip(raw_action[9], 0.0, 1.0))

                clamped_xyz = np.clip(predicted_xyz, workspace_min, workspace_max)
                target_xyz = limit_xyz_step(
                    current_xyz, clamped_xyz, float(args.max_eef_step_m)
                )
                target_rotation = limit_rotation_step(
                    current_rotation,
                    predicted_rotation,
                    math.radians(float(args.max_eef_rotation_step_deg)),
                )
                grasp_delta = float(np.clip(
                    predicted_grasp - last_commanded_grasp,
                    -float(args.max_grasp_step),
                    float(args.max_grasp_step),
                ))
                target_grasp = float(np.clip(last_commanded_grasp + grasp_delta, 0.0, 1.0))

                for threshold_name, threshold in (
                    ("approach", 0.57684276),
                    ("closing", 0.70),
                    ("closed", 0.90),
                ):
                    if last_commanded_grasp < threshold <= target_grasp:
                        ever_closed = ever_closed or threshold_name == "closed"
                        print(
                            f"[Slender-pin XVLA] grasp event={threshold_name} "
                            f"step={control_step} grasp={target_grasp:.3f} "
                            f"xyz={np.round(current_xyz, 5).tolist()}",
                            flush=True,
                        )

                q_seed = right_kin.set_arm(q_seed, current_arm_q)
                solution = right_kin.solve_pose_ik(
                    target_xyz,
                    target_rotation,
                    q_seed=q_seed,
                    max_iters=int(args.ik_max_iters),
                    position_tolerance_m=float(args.ik_position_tolerance_m),
                    rotation_tolerance_rad=math.radians(float(args.ik_rotation_tolerance_deg)),
                    nullspace_config=expert.get("solver", {}).get("nullspace"),
                    q_reference_arm=current_arm_q,
                )
                ik_errors.append([solution.position_error_m, solution.rotation_error_rad])
                if (
                    solution.position_error_m > float(args.ik_hard_position_error_m)
                    or solution.rotation_error_rad
                    > math.radians(float(args.ik_hard_rotation_error_deg))
                ):
                    raise RuntimeError(
                        "Policy target rejected by IK safety: "
                        f"pos={solution.position_error_m * 1000.0:.2f}mm "
                        f"rot={math.degrees(solution.rotation_error_rad):.2f}deg"
                    )
                q_seed = solution.q_full
                target_arm_q = solution.q_arm
                right_hand = hand_target(target_grasp, robot_cfg)
                target = np.concatenate([left_target, target_arm_q, right_hand])
                coupled = np.asarray(
                    coupled_hand_positions("left", left_target[7:13])
                    + coupled_hand_positions("right", right_hand),
                    dtype=float,
                )
                for substep in range(substeps):
                    articulation.set_joint_position_targets(
                        np.asarray([target], dtype=np.float32),
                        joint_indices=controlled_indices,
                    )
                    articulation.set_joint_velocity_targets(
                        np.zeros((1, len(controlled_indices)), dtype=np.float32),
                        joint_indices=controlled_indices,
                    )
                    articulation.set_joint_position_targets(
                        np.asarray([coupled], dtype=np.float32),
                        joint_indices=coupled_indices,
                    )
                    articulation.set_joint_velocity_targets(
                        np.zeros((1, len(coupled_indices)), dtype=np.float32),
                        joint_indices=coupled_indices,
                    )
                    if substep == substeps - 1:
                        app.update()
                    else:
                        SimulationManager.step(render=False)

                last_commanded_xyz = target_xyz.copy()
                last_commanded_rotation = target_rotation.copy()
                if (
                    not release_damping_applied
                    and (ever_lifted or ever_inserted)
                    and last_commanded_grasp > float(args.open_grasp_threshold)
                    and float(target_grasp) <= float(args.open_grasp_threshold)
                ):
                    plug_physx.GetLinearDampingAttr().Set(
                        float(task_cfg["simulation"].get("object_release_linear_damping", 0.05))
                    )
                    plug_physx.GetAngularDampingAttr().Set(
                        float(task_cfg["simulation"].get("object_release_angular_damping", 0.05))
                    )
                    release_damping_applied = True
                    print("[Slender-pin XVLA] dropped pin damping after grasp opened", flush=True)
                last_commanded_grasp = float(target_grasp)

                metrics = insertion_metrics()
                plug_world = np.asarray(metrics["pin_xyz"], dtype=float)
                ever_lifted = ever_lifted or float(
                    plug_world[2] - initial_plug_world[2]
                ) >= float(validation["required_lift_m"])
                insertion_streak = insertion_streak + 1 if metrics["valid"] else 0
                ever_inserted = ever_inserted or bool(metrics["valid"])

                (
                    measured_state,
                    measured_xyz,
                    measured_rotation,
                    measured_arm_q,
                    measured_g,
                ) = actual_state()
                observation_states.append(state.copy())
                predicted_actions.append(raw_action.copy())
                executed_eef.append(
                    np.concatenate([target_xyz, rotation_to_rot6d(target_rotation)])
                )
                executed_grasp.append(target_grasp)
                measured_grasp_list.append(measured_g)
                actual_joint_positions.append(
                    np.asarray(
                        articulation.get_joint_positions(
                            joint_indices=controlled_indices[13:26]
                        ),
                        dtype=float,
                    ).reshape(-1)
                )
                insertion_history.append([
                    metrics["depth_m"],
                    metrics["lateral_error_m"],
                    metrics["orientation_error_rad"],
                    float(metrics["valid"]),
                ])
                pin_xyz_hist.append(metrics["pin_xyz"])
                pin_rpy_hist.append(metrics["pin_rpy_deg"])
                action_sources.append([policy_calls - 1, horizon_index, 1])
                inference_ms_hist.append(inference_ms if horizon_index == 0 else 0.0)

                if args.record_video:
                    frame_images = current_images if horizon_index == 0 else read_images()
                    for name in CAMERA_NAMES:
                        video_writers[name].write(frame_images[name])
                    current_images = frame_images

                if control_step % 20 == 0 or args.verbose_actions:
                    print(
                        f"[Slender-pin XVLA] step={control_step:04d} "
                        f"call={policy_calls - 1:03d}:{horizon_index:02d} "
                        f"xyz={np.round(target_xyz, 4).tolist()} "
                        f"grasp={target_grasp:.3f} measured={measured_g:.3f} "
                        f"pin_lateral={metrics['lateral_error_m'] * 1000.0:.2f}mm "
                        f"pin_orient={math.degrees(metrics['orientation_error_rad']):.2f}deg "
                        f"depth={metrics['depth_m'] * 1000.0:.2f}mm "
                        f"lift={ever_lifted} seated={ever_inserted} "
                        f"infer={inference_ms:.1f}ms",
                        flush=True,
                    )

                current_xyz = measured_xyz
                current_rotation = measured_rotation
                current_arm_q = measured_arm_q
                current_grasp = measured_g
                state = measured_state
                control_step += 1

                if insertion_streak >= int(args.success_stable_steps) and ever_lifted:
                    finish_reason = "seated_stability"
                    break
                if control_step >= int(args.max_control_steps):
                    break

            if finish_reason in {"seated_stability", "interrupted"}:
                break

            if args.realtime:
                delay = 1.0 / float(args.control_hz) - (time.perf_counter() - cycle_started)
                if delay > 0.0:
                    time.sleep(delay)

        final_metrics = insertion_metrics()
        success = bool(ever_lifted and ever_inserted)
        if shutdown["requested"]:
            finish_reason = "interrupted"
        if finish_reason == "interrupted":
            status = "INTERRUPTED"
            success = False
        else:
            status = "PASS" if success else "FAIL"
        report = {
            "status": status,
            "success": success,
            "finish_reason": finish_reason,
            "policy": "LeRobot XVLA",
            "policy_loader": "XVLAPolicy.from_pretrained + make_pre_post_processors",
            "checkpoint": str(resolve(args.checkpoint)),
            "dataset": str(dataset_dir),
            "synchronous_closed_loop": True,
            "asynchronous_policy": False,
            "privileged_policy_input": False,
            "action_space": "absolute_eef_xyz_rot6d_grasp",
            "normalization": schema.get("normalization_mapping"),
            "action_mode": schema.get("action_mode"),
            "max_state_dim": schema.get("max_state_dim"),
            "action_dim_names": list(ACTION_DIM_NAMES),
            "state_dim_names": list(STATE_DIM_NAMES),
            "camera_names": list(CAMERA_NAMES),
            "camera_resolution": [640, 480],
            "dataset_fps": float(dataset_fps),
            "control_hz": float(args.control_hz),
            "physics_hz": float(physics_hz),
            "execution_horizon": int(args.execution_horizon),
            "policy_chunk_size": int(schema.get("chunk_size", 32)),
            "language_instruction": instruction,
            "seed": int(args.seed),
            "reference_expert_episode": reference_episode_name,
            "pin_initial_xyz": [float(v) for v in plug_xyz],
            "executed_control_steps": control_step,
            "policy_calls": policy_calls,
            "ever_lifted": bool(ever_lifted),
            "ever_closed_grasp": bool(ever_closed),
            "ever_inserted": bool(ever_inserted),
            "release_damping_applied": bool(release_damping_applied),
            "final_insertion_metrics": final_metrics,
            "mean_policy_latency_ms": (
                float(np.mean(policy_latencies)) if policy_latencies else None
            ),
            "max_policy_latency_ms": (
                float(np.max(policy_latencies)) if policy_latencies else None
            ),
            "max_ik_position_error_m": (
                float(np.max(np.asarray(ik_errors)[:, 0])) if ik_errors else None
            ),
            "max_ik_rotation_error_rad": (
                float(np.max(np.asarray(ik_errors)[:, 1])) if ik_errors else None
            ),
        }
        arrays = {
            "observation_state": np.asarray(observation_states, dtype=np.float32),
            "predicted_action": np.asarray(predicted_actions, dtype=np.float32),
            "executed_right_eef": np.asarray(executed_eef, dtype=np.float32),
            "executed_right_hand_grasp": np.asarray(
                executed_grasp, dtype=np.float32
            ).reshape(-1, 1),
            "measured_grasp": np.asarray(
                measured_grasp_list, dtype=np.float32
            ).reshape(-1, 1),
            "actual_right_joint_position": np.asarray(
                actual_joint_positions, dtype=np.float32
            ),
            "insertion_metrics": np.asarray(insertion_history, dtype=np.float32),
            "pin_xyz": np.asarray(pin_xyz_hist, dtype=np.float32),
            "pin_rpy_world_deg": np.asarray(pin_rpy_hist, dtype=np.float32),
            "ik_error": np.asarray(ik_errors, dtype=np.float32),
            "policy_latency_ms": np.asarray(policy_latencies, dtype=np.float32),
            "step_inference_ms": np.asarray(inference_ms_hist, dtype=np.float32),
            "action_source": np.asarray(action_sources, dtype=np.int32),
        }
        return 0 if success else 2

    except (KeyboardInterrupt, InterruptedError):
        report = {
            "status": "INTERRUPTED",
            "success": False,
            "finish_reason": "interrupted",
            "executed_control_steps": len(observation_states),
            "policy_calls": len(policy_latencies),
            "checkpoint": str(resolve(args.checkpoint)),
        }
        print("[Slender-pin XVLA] interrupted — saving video/npz/metadata", flush=True)
        return 130

    except BaseException as exc:
        report = {
            "status": "ERROR",
            "success": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "executed_control_steps": len(observation_states),
            "policy_calls": len(policy_latencies),
        }
        print(f"[Slender-pin XVLA] ERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1

    finally:
        persist_outputs()
        stop_policy_server(
            server_process,
            policy_client,
            force=report.get("status") in {"INTERRUPTED", "ERROR"},
        )
        if app is not None:
            try:
                app.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
