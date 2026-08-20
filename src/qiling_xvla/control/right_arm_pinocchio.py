"""Pinocchio FK/IK helpers for the S4 right arm."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IKSolution:
    q_full: np.ndarray
    q_right_arm: np.ndarray
    success: bool
    position_error_m: float
    iterations: int


def import_pinocchio_without_ros_pythonpath():
    """Import conda Pinocchio even if ROS Humble Python paths leaked into sys.path.

    ROS Humble on this workstation provides Python 3.10 Pinocchio bindings. The
    so101_isaac environment uses Python 3.11, so importing the ROS package first
    raises ``ModuleNotFoundError: pinocchio.pinocchio_pywrap_default``.
    """

    removed_paths: list[str] = []
    kept_paths: list[str] = []
    current_py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for path in sys.path:
        path_text = str(path)
        is_ros_python = "/opt/ros/" in path_text
        is_wrong_python = "/python3." in path_text and current_py_tag not in path_text
        if is_ros_python or is_wrong_python:
            removed_paths.append(path_text)
        else:
            kept_paths.append(path)

    if removed_paths:
        sys.path[:] = kept_paths
        sys.modules.pop("pinocchio", None)

    try:
        return importlib.import_module("pinocchio")
    except ModuleNotFoundError as exc:
        hint = (
            "Failed to import Pinocchio. If you are running from a ROS-sourced shell, "
            "try: unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD, then activate so101_isaac again."
        )
        raise ModuleNotFoundError(hint) from exc


class RightArmPinocchioKinematics:
    """Kinematics wrapper around the fixed-lower-body S4 URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: Iterable[str | Path],
        right_arm_joints: list[str],
        eef_frame: str,
    ) -> None:
        pin = import_pinocchio_without_ros_pythonpath()

        self.pin = pin
        self.urdf_path = Path(urdf_path)
        package_dir_list = [str(Path(path)) for path in package_dirs]
        try:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path), package_dir_list)
        except Exception:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.right_arm_joints = right_arm_joints
        self.eef_frame = eef_frame
        self.eef_frame_id = self.model.getFrameId(eef_frame)
        if self.eef_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame not found in model: {eef_frame}")

        self.joint_ids = [self.model.getJointId(name) for name in right_arm_joints]
        missing = [
            name
            for name, joint_id in zip(right_arm_joints, self.joint_ids)
            if joint_id >= len(self.model.joints)
        ]
        if missing:
            raise ValueError(f"Joints not found in model: {missing}")

        self.idx_q = [self.model.joints[joint_id].idx_q for joint_id in self.joint_ids]
        self.idx_v = [self.model.joints[joint_id].idx_v for joint_id in self.joint_ids]

    def neutral(self) -> np.ndarray:
        return self.pin.neutral(self.model)

    def right_arm_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray([q_full[idx] for idx in self.idx_q], dtype=float)

    def set_right_arm(self, q_full: np.ndarray, q_right_arm: np.ndarray) -> np.ndarray:
        q_out = np.array(q_full, dtype=float, copy=True)
        for idx, value in zip(self.idx_q, q_right_arm):
            q_out[idx] = value
        return q_out

    def frame_pose(self, q_full: np.ndarray):
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.eef_frame_id].copy()

    def solve_position_ik(
        self,
        target_xyz: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 200,
        tolerance_m: float = 1e-3,
        damping: float = 1e-4,
        step_scale: float = 0.35,
    ) -> IKSolution:
        pin = self.pin
        q = self.neutral() if q_seed is None else np.array(q_seed, dtype=float, copy=True)
        target = np.asarray(target_xyz, dtype=float).reshape(3)

        success = False
        position_error = float("inf")
        iteration = 0

        for iteration in range(1, max_iters + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pose = self.data.oMf[self.eef_frame_id]
            error = target - pose.translation
            position_error = float(np.linalg.norm(error))
            if position_error <= tolerance_m:
                success = True
                break

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.eef_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, self.idx_v]
            lhs = jacobian @ jacobian.T + damping * np.eye(3)
            delta_right = jacobian.T @ np.linalg.solve(lhs, error)

            velocity = np.zeros(self.model.nv)
            for idx_v, value in zip(self.idx_v, delta_right):
                velocity[idx_v] = value * step_scale

            q = pin.integrate(self.model, q, velocity)
            q = self._clamp_right_arm(q)

        return IKSolution(
            q_full=q,
            q_right_arm=self.right_arm_from_full(q),
            success=success,
            position_error_m=position_error,
            iterations=iteration,
        )

    def _clamp_right_arm(self, q_full: np.ndarray) -> np.ndarray:
        q_out = np.array(q_full, dtype=float, copy=True)
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        for idx_q in self.idx_q:
            if np.isfinite(lower[idx_q]) and np.isfinite(upper[idx_q]):
                q_out[idx_q] = min(upper[idx_q], max(lower[idx_q], q_out[idx_q]))
        return q_out
