"""Pinocchio FK/IK helpers for S4 dual-arm scripted data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qiling_xvla.control.right_arm_pinocchio import import_pinocchio_without_ros_pythonpath


@dataclass(frozen=True)
class ArmIKSolution:
    q_full: np.ndarray
    q_arm: np.ndarray
    success: bool
    position_error_m: float
    iterations: int
    rotation_error_rad: float = 0.0


class ArmPinocchioKinematics:
    """Kinematics wrapper for one named arm in the full S4 URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: Iterable[str | Path],
        arm_joints: list[str],
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
        self.arm_joints = arm_joints
        self.eef_frame = eef_frame
        self.eef_frame_id = self.model.getFrameId(eef_frame)
        if self.eef_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame not found in model: {eef_frame}")

        self.joint_ids = [self.model.getJointId(name) for name in arm_joints]
        missing = [name for name, joint_id in zip(arm_joints, self.joint_ids) if joint_id >= len(self.model.joints)]
        if missing:
            raise ValueError(f"Joints not found in model: {missing}")

        self.idx_q = [self.model.joints[joint_id].idx_q for joint_id in self.joint_ids]
        self.idx_v = [self.model.joints[joint_id].idx_v for joint_id in self.joint_ids]

    def neutral(self) -> np.ndarray:
        return self.pin.neutral(self.model)

    def arm_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray([q_full[idx] for idx in self.idx_q], dtype=float)

    def set_arm(self, q_full: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        q_out = np.array(q_full, dtype=float, copy=True)
        for idx, value in zip(self.idx_q, q_arm):
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
    ) -> ArmIKSolution:
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
            delta_arm = jacobian.T @ np.linalg.solve(lhs, error)

            velocity = np.zeros(self.model.nv)
            for idx_v, value in zip(self.idx_v, delta_arm):
                velocity[idx_v] = value * step_scale

            q = pin.integrate(self.model, q, velocity)
            q = self._clamp_arm(q)

        return ArmIKSolution(
            q_full=q,
            q_arm=self.arm_from_full(q),
            success=success,
            position_error_m=position_error,
            iterations=iteration,
        )

    def solve_pose_ik(
        self,
        target_xyz: np.ndarray,
        target_rotation: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 300,
        position_tolerance_m: float = 1e-3,
        rotation_tolerance_rad: float = 0.08,
        damping: float = 1e-4,
        step_scale: float = 0.30,
        rotation_weight: float = 0.45,
        nullspace_config: dict[str, Any] | None = None,
        q_reference_arm: np.ndarray | None = None,
    ) -> ArmIKSolution:
        """Solve a palm pose with an optional continuous 7-DoF nullspace task."""

        pin = self.pin
        q = self.neutral() if q_seed is None else np.array(q_seed, dtype=float, copy=True)
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
        nullspace = dict(nullspace_config or {})
        nullspace_enabled = bool(nullspace.get("enabled", False))
        reference_arm = (
            self.arm_from_full(q)
            if q_reference_arm is None
            else np.asarray(q_reference_arm, dtype=float).reshape(len(self.idx_q))
        )
        elbow_frame_id = None
        if nullspace_enabled and nullspace.get("elbow_frame"):
            elbow_frame_id = self.model.getFrameId(str(nullspace["elbow_frame"]))
            if elbow_frame_id >= len(self.model.frames):
                raise ValueError(
                    f"Nullspace elbow frame not found: {nullspace['elbow_frame']}"
                )

        success = False
        position_error = float("inf")
        rotation_error = float("inf")
        iteration = 0
        weights = np.diag([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight])

        for iteration in range(1, max_iters + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pose = self.data.oMf[self.eef_frame_id]
            position_delta = target - pose.translation
            rotation_delta = pin.log3(target_rotation @ pose.rotation.T)
            position_error = float(np.linalg.norm(position_delta))
            rotation_error = float(np.linalg.norm(rotation_delta))
            secondary_converged = True
            if nullspace_enabled:
                secondary_converged = self._nullspace_converged(
                    q,
                    elbow_frame_id,
                    nullspace,
                )
            if (
                position_error <= position_tolerance_m
                and rotation_error <= rotation_tolerance_rad
                and secondary_converged
            ):
                success = True
                break

            error = weights @ np.concatenate([position_delta, rotation_delta])
            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.eef_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, self.idx_v]
            weighted_jacobian = weights @ jacobian
            lhs = weighted_jacobian @ weighted_jacobian.T + damping * np.eye(6)
            jacobian_pinv = weighted_jacobian.T @ np.linalg.solve(
                lhs, np.eye(weighted_jacobian.shape[0])
            )
            delta_arm = jacobian_pinv @ error
            if nullspace_enabled:
                nullspace_projector = (
                    np.eye(len(self.idx_v)) - jacobian_pinv @ weighted_jacobian
                )
                secondary = self._nullspace_secondary_velocity(
                    q,
                    reference_arm,
                    elbow_frame_id,
                    nullspace,
                )
                delta_arm = delta_arm + nullspace_projector @ secondary
                max_joint_step = float(nullspace.get("max_joint_step_rad", 0.20))
                delta_norm = float(np.linalg.norm(delta_arm))
                if delta_norm > max_joint_step > 0.0:
                    delta_arm *= max_joint_step / delta_norm

            velocity = np.zeros(self.model.nv)
            for idx_v, value in zip(self.idx_v, delta_arm):
                velocity[idx_v] = value * step_scale
            q = pin.integrate(self.model, q, velocity)
            q = self._clamp_arm(q)

        if not success:
            success = bool(
                position_error <= position_tolerance_m
                and rotation_error <= rotation_tolerance_rad
            )

        return ArmIKSolution(
            q_full=q,
            q_arm=self.arm_from_full(q),
            success=success,
            position_error_m=position_error,
            iterations=iteration,
            rotation_error_rad=rotation_error,
        )

    def _nullspace_secondary_velocity(
        self,
        q_full: np.ndarray,
        reference_arm: np.ndarray,
        elbow_frame_id: int | None,
        config: dict[str, Any],
    ) -> np.ndarray:
        pin = self.pin
        q_arm = self.arm_from_full(q_full)
        secondary = float(config.get("reference_gain", 0.18)) * (
            reference_arm - q_arm
        )

        lower = np.asarray(self.model.lowerPositionLimit[self.idx_q], dtype=float)
        upper = np.asarray(self.model.upperPositionLimit[self.idx_q], dtype=float)
        finite = np.isfinite(lower) & np.isfinite(upper) & ((upper - lower) > 1.0e-6)
        if np.any(finite):
            midpoint = 0.5 * (lower + upper)
            half_width = np.maximum(0.5 * (upper - lower), 1.0e-6)
            normalized = np.zeros_like(q_arm)
            normalized[finite] = (q_arm[finite] - midpoint[finite]) / half_width[finite]
            barrier = np.zeros_like(q_arm)
            denominator = np.maximum(1.0 - np.square(normalized[finite]), 0.05)
            barrier[finite] = -normalized[finite] / denominator
            barrier = np.clip(barrier, -5.0, 5.0)
            secondary += float(config.get("joint_limit_gain", 0.025)) * barrier

        if elbow_frame_id is not None:
            sign = float(config.get("outward_sign", -1.0))
            elbow_pose = self.data.oMf[elbow_frame_id]
            outward = sign * float(elbow_pose.translation[1])
            target = float(config.get("elbow_target_outward_m", outward))
            elbow_error = target - outward
            if bool(config.get("elbow_only_below_target", True)):
                elbow_error = max(0.0, elbow_error)
            elbow_error = float(np.clip(elbow_error, -0.05, 0.12))
            elbow_jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_full,
                elbow_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, self.idx_v]
            outward_gradient = sign * elbow_jacobian[1, :]
            secondary += (
                float(config.get("elbow_gain", 6.0))
                * elbow_error
                * outward_gradient
            )

        max_norm = float(config.get("max_secondary_velocity_norm", 0.18))
        norm = float(np.linalg.norm(secondary))
        if norm > max_norm > 0.0:
            secondary *= max_norm / norm
        return secondary

    def _nullspace_converged(
        self,
        q_full: np.ndarray,
        elbow_frame_id: int | None,
        config: dict[str, Any],
    ) -> bool:
        q_arm = self.arm_from_full(q_full)
        lower = np.asarray(self.model.lowerPositionLimit[self.idx_q], dtype=float)
        upper = np.asarray(self.model.upperPositionLimit[self.idx_q], dtype=float)
        widths = np.maximum(upper - lower, 1.0e-9)
        margin = float(np.min(np.minimum(q_arm - lower, upper - q_arm) / widths))
        if margin < float(config.get("minimum_joint_margin_ratio", 0.0)):
            return False
        if elbow_frame_id is None:
            return True
        sign = float(config.get("outward_sign", -1.0))
        outward = sign * float(self.data.oMf[elbow_frame_id].translation[1])
        target = float(config.get("elbow_target_outward_m", outward))
        tolerance = float(config.get("elbow_target_tolerance_m", 0.015))
        return outward >= target - tolerance

    def _clamp_arm(self, q_full: np.ndarray) -> np.ndarray:
        q_out = np.array(q_full, dtype=float, copy=True)
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        for idx_q in self.idx_q:
            if np.isfinite(lower[idx_q]) and np.isfinite(upper[idx_q]):
                q_out[idx_q] = min(upper[idx_q], max(lower[idx_q], q_out[idx_q]))
        return q_out
