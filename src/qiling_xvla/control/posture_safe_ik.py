"""Posture-constrained IK helpers for bimanual RJ45 assembly."""

from __future__ import annotations

from typing import Any

import numpy as np

from qiling_xvla.control.dual_arm_pinocchio import (
    ArmIKSolution,
    ArmPinocchioKinematics,
)


def arm_posture_report(
    kin: ArmPinocchioKinematics,
    side: str,
    q_arm: np.ndarray,
    posture_cfg: dict[str, Any],
) -> dict[str, float | bool]:
    q_arm = np.asarray(q_arm, dtype=float).reshape(7)
    q_full = kin.set_arm(kin.neutral(), q_arm)
    kin.pin.forwardKinematics(kin.model, kin.data, q_full)
    kin.pin.updateFramePlacements(kin.model, kin.data)
    sign = 1.0 if side == "left" else -1.0
    elbow = np.asarray(
        kin.data.oMf[kin.model.getFrameId(f"{side}_elbow_link")].translation,
        dtype=float,
    )
    wrist = np.asarray(
        kin.data.oMf[kin.model.getFrameId(f"{side}_wrist_roll_link")].translation,
        dtype=float,
    )
    lower = np.asarray(kin.model.lowerPositionLimit[kin.idx_q], dtype=float)
    upper = np.asarray(kin.model.upperPositionLimit[kin.idx_q], dtype=float)
    widths = np.maximum(upper - lower, 1.0e-9)
    margin = float(np.min(np.minimum(q_arm - lower, upper - q_arm) / widths))
    jacobian = kin.pin.computeFrameJacobian(
        kin.model,
        kin.data,
        q_full,
        kin.eef_frame_id,
        kin.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )[:, kin.idx_v]
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    report = {
        "outward_shoulder_roll_rad": float(sign * q_arm[1]),
        "elbow_flexion_rad": float(-q_arm[3]),
        "elbow_outward_y_m": float(sign * elbow[1]),
        "wrist_outward_y_m": float(sign * wrist[1]),
        "minimum_joint_margin_ratio": margin,
        "minimum_jacobian_singular_value": float(singular_values[-1]),
    }
    report["valid"] = bool(
        report["outward_shoulder_roll_rad"]
        >= float(posture_cfg.get("min_outward_shoulder_roll_rad", -0.15))
        and report["elbow_flexion_rad"]
        >= float(posture_cfg.get("min_elbow_flexion_rad", 0.50))
        and report["elbow_flexion_rad"]
        <= float(posture_cfg.get("max_elbow_flexion_rad", 1.84))
        and report["elbow_outward_y_m"]
        >= float(posture_cfg.get("min_elbow_outward_y_m", 0.20))
        and report["wrist_outward_y_m"]
        >= float(posture_cfg.get("min_wrist_outward_y_m", 0.13))
        and margin >= float(posture_cfg.get("min_joint_margin_ratio", 0.02))
        and report["minimum_jacobian_singular_value"]
        >= float(posture_cfg.get("min_jacobian_singular_value", 0.0))
    )
    return report


def solve_posture_safe_pose(
    kin: ArmPinocchioKinematics,
    side: str,
    xyz: np.ndarray,
    rotation: np.ndarray,
    current_q: np.ndarray,
    home_q: np.ndarray,
    solver_cfg: dict[str, Any],
    posture_cfg: dict[str, Any],
) -> tuple[ArmIKSolution, dict[str, float | bool]]:
    sign = 1.0 if side == "left" else -1.0
    assembly_seed = np.asarray(
        [-0.39, 0.16, -0.62, -1.36, -0.31, -0.27, -0.23]
        if side == "left"
        else [-0.35, -0.23, -0.65, -1.35, 0.38, -0.36, -0.25],
        dtype=float,
    )
    nullspace_cfg = dict(solver_cfg.get("nullspace", {}))
    nullspace_enabled = bool(nullspace_cfg.get("enabled", False))
    if nullspace_enabled:
        nullspace_cfg.setdefault("elbow_frame", f"{side}_elbow_link")
        nullspace_cfg.setdefault("outward_sign", sign)
    seeds = [np.asarray(current_q, dtype=float)]
    if not (nullspace_enabled and bool(nullspace_cfg.get("continuous_only", False))):
        seeds.extend([assembly_seed, np.asarray(home_q, dtype=float)])
        for shoulder_roll in (0.16, 0.32, 0.50):
            for shoulder_yaw in (-0.80, -0.25, 0.35):
                for elbow in (-1.00, -1.45):
                    seed = np.asarray(home_q, dtype=float).copy()
                    seed[1] = sign * shoulder_roll
                    seed[2] = shoulder_yaw
                    seed[3] = elbow
                    seeds.append(seed)

    best: ArmIKSolution | None = None
    best_report: dict[str, float | bool] | None = None
    for seed in seeds:
        solution = kin.solve_pose_ik(
            np.asarray(xyz, dtype=float),
            np.asarray(rotation, dtype=float),
            q_seed=kin.set_arm(kin.neutral(), seed),
            max_iters=int(solver_cfg.get("ik_max_iters", 600)),
            position_tolerance_m=float(
                solver_cfg.get("waypoint_position_tolerance_m", 0.006)
            ),
            rotation_tolerance_rad=float(
                solver_cfg.get("waypoint_rotation_tolerance_rad", 0.10)
            ),
            damping=float(solver_cfg.get("ik_damping", 1.0e-4)),
            step_scale=float(solver_cfg.get("ik_step_scale", 0.25)),
            rotation_weight=float(solver_cfg.get("ik_rotation_weight", 0.40)),
            nullspace_config=nullspace_cfg,
            q_reference_arm=np.asarray(current_q, dtype=float),
        )
        report = arm_posture_report(kin, side, solution.q_arm, posture_cfg)
        if best is None or (
            solution.position_error_m,
            solution.rotation_error_rad,
        ) < (best.position_error_m, best.rotation_error_rad):
            best = solution
            best_report = report
        if solution.success and bool(report["valid"]):
            return solution, report

    if best is None or best_report is None:
        raise RuntimeError(f"No {side} IK candidates were evaluated")
    return (
        ArmIKSolution(
            best.q_full,
            best.q_arm,
            False,
            best.position_error_m,
            best.iterations,
            best.rotation_error_rad,
        ),
        best_report,
    )


def torso_safe_joint_path(
    kin: ArmPinocchioKinematics,
    side: str,
    start_q: np.ndarray,
    end_q: np.ndarray,
    posture_cfg: dict[str, Any],
) -> bool:
    sign = 1.0 if side == "left" else -1.0
    count = int(posture_cfg.get("path_samples", 9))
    elbow_limit = float(posture_cfg.get("path_min_elbow_outward_y_m", 0.18))
    wrist_limit = float(posture_cfg.get("path_min_wrist_outward_y_m", 0.11))
    for alpha in np.linspace(0.0, 1.0, count):
        q_arm = np.asarray(start_q) * (1.0 - alpha) + np.asarray(end_q) * alpha
        q_full = kin.set_arm(kin.neutral(), q_arm)
        kin.pin.forwardKinematics(kin.model, kin.data, q_full)
        kin.pin.updateFramePlacements(kin.model, kin.data)
        elbow_y = sign * float(
            kin.data.oMf[kin.model.getFrameId(f"{side}_elbow_link")].translation[1]
        )
        wrist_y = sign * float(
            kin.data.oMf[
                kin.model.getFrameId(f"{side}_wrist_roll_link")
            ].translation[1]
        )
        if elbow_y < elbow_limit or wrist_y < wrist_limit:
            return False
    return True
