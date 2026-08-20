"""Synchronized three-camera recorder for the fixed-socket right-arm task."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


def rotation_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Encode the first two rotation-matrix columns, matching GR00T conventions."""

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    return matrix[:, :2].T.reshape(-1)


class _RgbVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        self.width = int(width)
        self.height = int(height)
        self.frame_count = 0
        self.closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", f"{float(fps):.8g}", "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.partial_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if frame.shape != expected:
            raise ValueError(f"Video frame shape {frame.shape}, expected {expected}")
        if self.closed or self.process.stdin is None:
            raise RuntimeError(f"Video writer is closed: {self.path}")
        self.process.stdin.write(frame.tobytes())
        self.frame_count += 1

    def close(self, commit: bool) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        returncode = self.process.wait()
        if commit and returncode == 0:
            self.partial_path.replace(self.path)
        else:
            self.partial_path.unlink(missing_ok=True)
        if commit and returncode != 0:
            raise RuntimeError(f"ffmpeg failed with return code {returncode}: {self.path}")


class FixedSocketEpisodeRecorder:
    """Record right-arm state/action and synchronized RGB observations at 20 Hz."""

    camera_names = ("chest", "left_wrist", "right_wrist")

    def __init__(
        self,
        *,
        app,
        stage,
        output_dir: Path,
        camera_paths: list[str],
        camera_cfg: dict,
        robot_cfg: dict,
        robot_model_root: str,
        right_kin,
        record_hz: float,
        instruction: str,
    ) -> None:
        from isaacsim.sensors.camera import Camera

        if len(camera_paths) != 3:
            raise ValueError(f"Expected three camera paths, got {camera_paths}")
        self.app = app
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.camera_cfg = camera_cfg
        self.robot_cfg = robot_cfg
        self.robot_model_root = robot_model_root
        self.right_kin = right_kin
        self.record_hz = float(record_hz)
        self.instruction = str(instruction)
        self.sensors = {}
        self.writers = {}
        self.observation_state = []
        self.target_observation_state = []
        self.action = []
        self.actual_joint_position = []
        self.target_joint_position = []
        self.phases = []
        self.closed = False

        for index, (name, path) in enumerate(zip(self.camera_names, camera_paths)):
            settings = camera_cfg["cameras"][name]
            width, height = int(settings["width"]), int(settings["height"])
            sensor = Camera(
                prim_path=path,
                name=f"fixed_socket_record_{name}_{index}",
                resolution=(width, height),
            )
            sensor.initialize()
            self.sensors[name] = sensor
            self.writers[name] = _RgbVideoWriter(
                self.output_dir / "videos" / f"{name}.mp4",
                width,
                height,
                self.record_hz,
            )
        for _ in range(12):
            app.update()
        self._validate_cameras()

    def _world_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        from pxr import Usd, UsdGeom

        matrix = UsdGeom.Xformable(
            self.stage.GetPrimAtPath(prim_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xyz = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        x, y, z, w = (
            float(imaginary[0]), float(imaginary[1]),
            float(imaginary[2]), float(quaternion.GetReal()),
        )
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        return np.asarray(xyz, dtype=float), rotation

    def _validate_cameras(self) -> None:
        for name, sensor in self.sensors.items():
            settings = self.camera_cfg["cameras"][name]
            expected = (int(settings["height"]), int(settings["width"]), 4)
            rgba = np.asarray(sensor.get_rgba())
            if rgba.shape != expected or float(rgba[:, :, :3].std()) < 1.0:
                raise RuntimeError(
                    f"Recorder camera {name} is unavailable or blank: shape={rgba.shape}"
                )
            print(f"Recorder camera ready: {name} shape={rgba.shape}", flush=True)

    def _eef_pose_from_arm(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full = self.right_kin.set_arm(self.right_kin.neutral(), arm_q)
        pose = self.right_kin.frame_pose(q_full)
        return np.asarray(pose.translation), np.asarray(pose.rotation)

    def _actual_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        base_xyz, base_rotation = self._world_pose(f"{self.robot_model_root}/base_link")
        eef_link = self.robot_cfg["robot"]["right_eef_link"]
        eef_xyz, eef_rotation = self._world_pose(f"{self.robot_model_root}/{eef_link}")
        return base_rotation.T @ (eef_xyz - base_xyz), base_rotation.T @ eef_rotation

    def _grasp_scalar(self, hand_q: np.ndarray) -> float:
        hand_cfg = self.robot_cfg["right_o6_hand"]
        names = hand_cfg["active_driver_joints"]
        opened = np.asarray([hand_cfg["open_preset"][name] for name in names])
        closed = np.asarray([hand_cfg["pinch_preset"][name] for name in names])
        delta = closed - opened
        denominator = float(delta @ delta)
        if denominator < 1.0e-12:
            return 0.0
        return float(np.clip((np.asarray(hand_q) - opened) @ delta / denominator, 0.0, 1.0))

    def capture(self, phase: str, actual_full_q: np.ndarray, target_full_q: np.ndarray) -> None:
        actual = np.asarray(actual_full_q, dtype=float).reshape(-1)
        target = np.asarray(target_full_q, dtype=float).reshape(-1)
        actual_arm, actual_hand = actual[13:20], actual[20:26]
        target_arm, target_hand = target[13:20], target[20:26]
        actual_xyz, actual_rotation = self._actual_eef_pose()
        target_xyz, target_rotation = self._eef_pose_from_arm(target_arm)
        actual_grasp = self._grasp_scalar(actual_hand)
        target_grasp = self._grasp_scalar(target_hand)
        self.observation_state.append(
            np.concatenate([actual_xyz, rotation_to_rot6d(actual_rotation), actual_arm, [actual_grasp]])
        )
        self.target_observation_state.append(
            np.concatenate([target_xyz, rotation_to_rot6d(target_rotation), target_arm, [target_grasp]])
        )
        self.action.append(
            np.concatenate([target_xyz, rotation_to_rot6d(target_rotation), [target_grasp]])
        )
        self.actual_joint_position.append(np.concatenate([actual_arm, actual_hand]))
        self.target_joint_position.append(np.concatenate([target_arm, target_hand]))
        self.phases.append(str(phase))
        for name, sensor in self.sensors.items():
            rgba = np.asarray(sensor.get_rgba())
            settings = self.camera_cfg["cameras"][name]
            expected = (int(settings["height"]), int(settings["width"]), 4)
            if rgba.shape != expected or float(rgba[:, :, :3].std()) < 1.0:
                raise RuntimeError(f"Invalid recorded frame from {name}: {rgba.shape}")
            self.writers[name].write(rgba[:, :, :3])

    def finalize(
        self,
        *,
        seed: int,
        plug_xyz: list[float],
        socket_xyz: list[float],
        metrics: dict,
    ) -> None:
        if self.closed:
            return
        frame_count = len(self.phases)
        if frame_count < 2:
            raise RuntimeError("Recorded episode contains fewer than two frames")
        for writer in self.writers.values():
            if writer.frame_count != frame_count:
                raise RuntimeError("Camera frame counts are not synchronized")
            writer.close(commit=True)
        self.closed = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output_dir / "episode.npz",
            observation_state=np.asarray(self.observation_state, dtype=np.float32),
            target_observation_state=np.asarray(self.target_observation_state, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.float32),
            timestamp=np.arange(frame_count, dtype=np.float64) / self.record_hz,
            phase=np.asarray(self.phases),
            actual_joint_position=np.asarray(self.actual_joint_position, dtype=np.float32),
            target_joint_position=np.asarray(self.target_joint_position, dtype=np.float32),
            expert_mask=np.ones(frame_count, dtype=bool),
            language_instruction=np.full(frame_count, self.instruction),
        )
        metadata = {
            "schema": "qiling_groot_assemble.fixed_socket_right_arm.v1",
            "status": "PASS",
            "seed": int(seed),
            "rj45_plug_world_xyz": [float(value) for value in plug_xyz],
            "socket_box_world_xyz": [float(value) for value in socket_xyz],
            "frame_count": frame_count,
            "record_hz": self.record_hz,
            "state_dim": 17,
            "action_dim": 10,
            "joint_position_dim": 13,
            "camera_names": list(self.camera_names),
            "camera_resolution": [640, 480],
            "camera_storage": "video",
            "camera_files": {name: f"videos/{name}.mp4" for name in self.camera_names},
            "camera_frame_counts": {
                name: self.writers[name].frame_count for name in self.camera_names
            },
            "language_instruction": self.instruction,
            "observation_action_alignment": "pre_command_observation_to_expert_target",
            **metrics,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def abort(self) -> None:
        if self.closed:
            return
        for writer in self.writers.values():
            writer.close(commit=False)
        self.closed = True
