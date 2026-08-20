"""Scripted bimanual Auto-IK waypoints for RJ45 cable insertion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from qiling_xvla.control.dual_arm_pinocchio import ArmIKSolution, ArmPinocchioKinematics


ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)


def xyz_rot6d(xyz: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(xyz, dtype=float).reshape(3), ROT6D_IDENTITY])


def rotation_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def arm_home(robot_cfg: dict, side: str) -> np.ndarray:
    arm_cfg = robot_cfg[f"{side}_arm"]
    return np.asarray([arm_cfg.get("home_position", {}).get(name, 0.0) for name in arm_cfg["joints"]], dtype=float)


def solve_with_restarts(
    kin: ArmPinocchioKinematics,
    target_xyz_base: np.ndarray,
    q_seed: np.ndarray,
    home_q: np.ndarray,
    scripted: dict,
    target_rotation: np.ndarray | None = None,
) -> ArmIKSolution:
    seeds: list[np.ndarray] = [
        q_seed,
        kin.set_arm(kin.neutral(), home_q),
        kin.set_arm(kin.neutral(), np.zeros_like(home_q)),
    ]
    for base in (kin.arm_from_full(q_seed), home_q):
        for joint_index in range(len(base)):
            for delta in (-0.35, 0.35):
                candidate = np.array(base, dtype=float, copy=True)
                candidate[joint_index] += delta
                seeds.append(kin.set_arm(kin.neutral(), candidate))

    tolerance_m = float(scripted.get("waypoint_position_tolerance_m", 1.0e-3))
    rotation_tolerance_rad = float(scripted.get("waypoint_rotation_tolerance_rad", 0.15))
    max_iters = int(scripted.get("ik_max_iters", 350))
    damping = float(scripted.get("ik_damping", 1.0e-4))
    step_scale = float(scripted.get("ik_step_scale", 0.35))
    if target_rotation is None:
        solutions = [
            kin.solve_position_ik(
                target_xyz_base,
                q_seed=seed,
                max_iters=max_iters,
                tolerance_m=tolerance_m,
                damping=damping,
                step_scale=step_scale,
            )
            for seed in seeds
        ]
    else:
        solutions = [
            kin.solve_pose_ik(
                target_xyz_base,
                target_rotation,
                q_seed=seed,
                max_iters=max_iters,
                position_tolerance_m=tolerance_m,
                rotation_tolerance_rad=rotation_tolerance_rad,
                damping=damping,
                step_scale=step_scale,
                rotation_weight=float(scripted.get("ik_rotation_weight", 0.45)),
            )
            for seed in seeds
        ]
    successful = [solution for solution in solutions if solution.success]
    reference = kin.arm_from_full(q_seed)
    if successful:
        return min(
            successful,
            key=lambda solution: (
                float(np.linalg.norm(solution.q_arm - reference)),
                float(solution.position_error_m),
                float(solution.rotation_error_rad),
                int(solution.iterations),
            ),
        )
    return min(solutions, key=lambda solution: float(solution.position_error_m))


def _hand_preset(robot_cfg: dict, side: str, preset_name: str) -> np.ndarray:
    hand_cfg = robot_cfg[f"{side}_o6_hand"]
    preset = hand_cfg[preset_name]
    return np.asarray([preset[name] for name in hand_cfg["active_driver_joints"]], dtype=float)


def _supported_grasp_preset(robot_cfg: dict, side: str) -> np.ndarray:
    """Keep the calibrated thumb/index pinch and close the remaining fingers."""
    hand_cfg = robot_cfg[f"{side}_o6_hand"]
    pinch = hand_cfg["pinch_preset"]
    close = hand_cfg["close_preset"]
    values = []
    for joint_name in hand_cfg["active_driver_joints"]:
        # The thumb and index establish the calibrated grasp. Middle, ring, and
        # pinky close only after the object has cleared the table.
        use_close = any(
            f"_{finger}_" in joint_name for finger in ("middle", "ring", "pinky")
        )
        values.append(close[joint_name] if use_close else pinch[joint_name])
    return np.asarray(values, dtype=float)


def make_rj45_pickup_auto_ik_plan(
    robot_cfg: dict,
    task_cfg: dict,
    root: str | Path,
    *,
    include_assembly: bool = False,
    socket_box_world_xyz: list[float] | np.ndarray | None = None,
    rj45_plug_world_xyz: list[float] | np.ndarray | None = None,
) -> list[dict]:
    """Build calibrated pickup waypoints and optionally continue through insertion."""

    root_path = Path(root)
    robot = robot_cfg["robot"]
    scene = task_cfg["scene"]
    pickup = task_cfg["pickup_auto_ik"]
    scripted = task_cfg["scripted_auto_ik"]
    pickup_solver = dict(scripted)
    for key in (
        "waypoint_position_tolerance_m",
        "waypoint_rotation_tolerance_rad",
        "ik_max_iters",
        "ik_damping",
        "ik_step_scale",
        "ik_rotation_weight",
    ):
        if key in pickup:
            pickup_solver[key] = pickup[key]
    base_world_xyz = np.asarray(scene["robot_base_world_xyz"], dtype=float)
    socket_world = np.asarray(
        socket_box_world_xyz
        if socket_box_world_xyz is not None
        else scene["socket_box_initial_xyz"],
        dtype=float,
    )
    plug_world = np.asarray(
        rj45_plug_world_xyz
        if rj45_plug_world_xyz is not None
        else scene["rj45_plug_initial_xyz"],
        dtype=float,
    )
    socket_base = socket_world - base_world_xyz
    plug_base = plug_world - base_world_xyz

    urdf_path = root_path / robot["preferred_urdf_for_first_pass"]
    left_kin = ArmPinocchioKinematics(
        urdf_path, [root_path], robot_cfg["left_arm"]["joints"], robot["left_eef_link"]
    )
    right_kin = ArmPinocchioKinematics(
        urdf_path, [root_path], robot_cfg["right_arm"]["joints"], robot["right_eef_link"]
    )
    left_home = arm_home(robot_cfg, "left")
    right_home = arm_home(robot_cfg, "right")
    left_seed = left_kin.set_arm(left_kin.neutral(), left_home)
    right_seed = right_kin.set_arm(right_kin.neutral(), right_home)
    left_home_target_pose = left_kin.frame_pose(left_seed)
    right_home_target_pose = right_kin.frame_pose(right_seed)
    left_open = _hand_preset(robot_cfg, "left", "open_preset")
    right_open = _hand_preset(robot_cfg, "right", "open_preset")
    left_approach = _hand_preset(robot_cfg, "left", "approach_preset")
    right_approach = _hand_preset(robot_cfg, "right", "approach_preset")
    left_closed = _hand_preset(robot_cfg, "left", "pinch_preset")
    right_closed = _hand_preset(robot_cfg, "right", "pinch_preset")
    left_supported = _supported_grasp_preset(robot_cfg, "left")
    right_supported = _supported_grasp_preset(robot_cfg, "right")
    left_grasp_rotation = np.asarray(pickup["left_grasp_rotation_matrix"], dtype=float).reshape(3, 3)
    left_hover_rotation = np.asarray(
        pickup.get("left_hover_rotation_matrix", pickup["left_grasp_rotation_matrix"]),
        dtype=float,
    ).reshape(3, 3)
    right_grasp_rotation = np.asarray(pickup["right_grasp_rotation_matrix"], dtype=float).reshape(3, 3)
    right_hover_rotation = np.asarray(
        pickup.get("right_hover_rotation_matrix", pickup["right_grasp_rotation_matrix"]),
        dtype=float,
    ).reshape(3, 3)
    table = scene["table"]
    table_top_world_z = float(table["center_xyz"][2]) + float(table["size_xyz"][2]) * 0.5
    fingertip_vertices: dict[tuple[str, str], np.ndarray] = {}
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            mesh_path = (
                root_path
                / "robot_description"
                / "S4"
                / "meshes"
                / "o6"
                / side
                / "meshes"
                / f"{finger}_distal.STL"
            )
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            fingertip_vertices[(side, finger)] = np.asarray(mesh.vertices, dtype=float)

    def fingertip_table_clearance(
        kin: ArmPinocchioKinematics,
        solution: ArmIKSolution,
        side: str,
        hand_q: np.ndarray,
    ) -> float:
        q_full = np.array(solution.q_full, dtype=float, copy=True)
        hand_cfg = robot_cfg[f"{side}_o6_hand"]
        for joint_name, value in zip(hand_cfg["active_driver_joints"], hand_q):
            joint_id = kin.model.getJointId(joint_name)
            q_full[kin.model.joints[joint_id].idx_q] = float(value)
        prefix = "LH" if side == "left" else "RH"
        thumb_multiplier = 2.29 if side == "left" else 1.86
        mimic_values = {
            f"{prefix}_thumb_ip": min(1.08, float(hand_q[1]) * thumb_multiplier),
            **{
                f"{prefix}_{finger}_dip": min(1.43, float(hand_q[index]) * 0.89)
                for index, finger in enumerate(("index", "middle", "ring", "pinky"), start=2)
            },
        }
        for joint_name, value in mimic_values.items():
            joint_id = kin.model.getJointId(joint_name)
            q_full[kin.model.joints[joint_id].idx_q] = value
        kin.pin.forwardKinematics(kin.model, kin.data, q_full)
        kin.pin.updateFramePlacements(kin.model, kin.data)
        world_heights = []
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            placement = kin.data.oMf[kin.model.getFrameId(f"{prefix}_{finger}_distal")]
            vertices_world = (
                np.asarray(placement.rotation, dtype=float) @ fingertip_vertices[(side, finger)].T
            ).T + np.asarray(placement.translation, dtype=float)
            world_heights.append(float(vertices_world[:, 2].min() + base_world_xyz[2]))
        return min(world_heights) - table_top_world_z

    left_grasp_center = socket_base + np.asarray(
        [0.0, 0.0, pickup["left_grasp_center_world_z_offset_m"]], dtype=float
    )
    right_grasp_center = plug_base + np.asarray(
        [0.0, 0.0, pickup["right_grasp_center_world_z_offset_m"]], dtype=float
    )
    left_grasp = left_grasp_center - left_grasp_rotation @ np.asarray(
        pickup["left_grasp_center_from_palm_m"], dtype=float
    )
    right_grasp = right_grasp_center - right_grasp_rotation @ np.asarray(
        pickup["right_grasp_center_from_palm_m"], dtype=float
    )
    left_grasp[0] += float(pickup.get("left_grasp_palm_x_offset_m", 0.0))
    right_grasp[0] += float(pickup.get("right_grasp_palm_x_offset_m", 0.0))
    right_grasp[1] += float(pickup.get("right_grasp_palm_y_offset_m", 0.0))
    pregrasp_height = float(pickup["pregrasp_clearance_m"])
    lift_height = float(pickup["lift_height_m"])
    movement_specs = [
        (
            "pre_grasp",
            left_grasp + [0.0, 0.0, pregrasp_height],
            right_grasp + [0.0, 0.0, pregrasp_height],
            left_open,
            right_open,
            0.0,
            0.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "shape_grippers",
            left_grasp + [0.0, 0.0, pregrasp_height],
            right_grasp + [0.0, 0.0, pregrasp_height],
            left_approach,
            right_approach,
            0.0,
            0.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "descend_to_objects",
            left_grasp,
            right_grasp,
            left_approach,
            right_approach,
            0.0,
            0.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "grasp_objects",
            left_grasp,
            right_grasp,
            left_closed,
            right_closed,
            1.0,
            1.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "grasp_hold",
            left_grasp,
            right_grasp,
            left_closed,
            right_closed,
            1.0,
            1.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "lift_objects",
            left_grasp + [0.0, 0.0, lift_height],
            right_grasp + [0.0, 0.0, lift_height],
            left_supported,
            right_supported,
            1.0,
            1.0,
            left_grasp_rotation,
            right_grasp_rotation,
        ),
        (
            "orient_socket_for_assembly",
            left_grasp + [0.0, 0.0, lift_height],
            right_grasp + [0.0, 0.0, lift_height],
            left_supported,
            right_supported,
            1.0,
            1.0,
            left_hover_rotation,
            right_hover_rotation,
        ),
        (
            "hover_objects",
            np.asarray(pickup["left_hover_palm_xyz_base"], dtype=float),
            np.asarray(pickup["right_hover_palm_xyz_base"], dtype=float),
            left_supported,
            right_supported,
            1.0,
            1.0,
            left_hover_rotation,
            right_hover_rotation,
        ),
    ]

    if include_assembly:
        # Keep the supporting three-finger closure throughout the airborne
        # assembly. The release waypoints below explicitly reopen each hand.
        left_closed = left_supported
        right_closed = right_supported
        assembly = task_cfg["assembly_auto_ik"]
        left_compensated_rotation = np.asarray(
            assembly["left_compensated_rotation_matrix"], dtype=float
        ).reshape(3, 3)
        right_compensated_rotation = np.asarray(
            assembly["right_compensated_rotation_matrix"], dtype=float
        ).reshape(3, 3)
        left_object_from_palm = np.asarray(
            assembly["left_object_from_palm_xyz_m"], dtype=float
        )
        right_object_from_palm = np.asarray(
            assembly["right_object_from_palm_xyz_m"], dtype=float
        )
        socket_target = np.asarray(assembly["socket_target_xyz_base"], dtype=float)
        plug_lateral_alignment = np.asarray(
            assembly.get("plug_lateral_alignment_offset_base_m", [0.0, 0.0, 0.0]),
            dtype=float,
        )

        def palm_for_object(
            object_xyz: np.ndarray,
            palm_rotation: np.ndarray,
            object_from_palm: np.ndarray,
        ) -> np.ndarray:
            return np.asarray(object_xyz, dtype=float) - palm_rotation @ object_from_palm

        left_assembly_palm = palm_for_object(
            socket_target, left_compensated_rotation, left_object_from_palm
        )
        right_object_targets = {
            name: socket_target + np.asarray(assembly[key], dtype=float)
            for name, key in (
                ("assembly_staging", "plug_pre_insert_offset_from_socket_m"),
                ("approach_slot_far", "plug_far_offset_from_socket_m"),
                ("approach_slot_near", "plug_near_offset_from_socket_m"),
                ("seat_at_slot_mouth", "plug_mouth_offset_from_socket_m"),
                ("insert_plug_shallow", "plug_shallow_insert_offset_from_socket_m"),
                ("insert_plug", "plug_final_insert_offset_from_socket_m"),
            )
        }
        right_object_targets["align_plug_tip"] = (
            right_object_targets["approach_slot_far"] + plug_lateral_alignment
        )
        right_object_targets["alignment_correction_1"] = right_object_targets["align_plug_tip"]
        right_object_targets["alignment_correction_2"] = right_object_targets["align_plug_tip"]
        for name in (
            "approach_slot_near",
            "seat_at_slot_mouth",
            "insert_plug_shallow",
            "insert_plug",
        ):
            right_object_targets[name] = right_object_targets[name] + plug_lateral_alignment

        if bool(assembly.get("require_base_y_only_insertion", False)):
            local_head_axis = np.asarray([0.0, 0.0, -1.0], dtype=float)
            socket_object_rotation = left_compensated_rotation @ left_grasp_rotation.T
            plug_object_rotation = right_compensated_rotation @ right_grasp_rotation.T
            socket_axis = socket_object_rotation @ local_head_axis
            plug_axis = plug_object_rotation @ local_head_axis
            if not np.allclose(socket_axis, [0.0, -1.0, 0.0], atol=2.0e-5):
                raise ValueError(f"Socket insertion axis is not base Y-: {socket_axis.tolist()}")
            if not np.allclose(plug_axis, [0.0, 1.0, 0.0], atol=2.0e-5):
                raise ValueError(f"Plug insertion axis is not base Y+: {plug_axis.tolist()}")

            insertion_targets = np.asarray(
                [
                    right_object_targets[name]
                    for name in (
                        "assembly_staging",
                        "approach_slot_far",
                        "align_plug_tip",
                        "alignment_correction_1",
                        "alignment_correction_2",
                        "approach_slot_near",
                        "seat_at_slot_mouth",
                        "insert_plug_shallow",
                        "insert_plug",
                    )
                ],
                dtype=float,
            )
            if not np.allclose(insertion_targets[:, [0, 2]], socket_target[[0, 2]], atol=1.0e-9):
                raise ValueError("Insertion targets change outside base Y")
            socket_entry = socket_target + socket_axis * 0.01758
            plug_tip_at_mouth = right_object_targets["seat_at_slot_mouth"] + plug_axis * 0.019
            if not np.allclose(socket_entry, plug_tip_at_mouth, atol=1.0e-6):
                raise ValueError(
                    "Plug tip and socket entry do not align at the mouth: "
                    f"socket={socket_entry.tolist()}, plug={plug_tip_at_mouth.tolist()}"
                )
            insertion_delta = (
                right_object_targets["insert_plug"]
                - right_object_targets["insert_plug_shallow"]
            )
            if not np.allclose(insertion_delta, [0.0, 0.010, 0.0], atol=1.0e-9):
                raise ValueError(f"Final insertion is not base Y+ 10 mm: {insertion_delta.tolist()}")

        def assembly_spec(name: str, right_object_xyz: np.ndarray) -> tuple:
            return (
                name,
                left_assembly_palm,
                palm_for_object(
                    right_object_xyz,
                    right_compensated_rotation,
                    right_object_from_palm,
                ),
                left_closed,
                right_closed,
                1.0,
                1.0,
                left_compensated_rotation,
                right_compensated_rotation,
            )

        socket_release_target = np.asarray(
            assembly["socket_release_target_xyz_base"], dtype=float
        )
        right_insert_palm = palm_for_object(
            right_object_targets["insert_plug"],
            right_compensated_rotation,
            right_object_from_palm,
        )
        right_release_rotation = np.asarray(
            assembly["right_release_rotation_matrix"], dtype=float
        ).reshape(3, 3)
        right_release_object = (
            right_object_targets["insert_plug"] + socket_release_target - socket_target
        )
        right_release_palm = palm_for_object(
            right_release_object,
            right_release_rotation,
            right_object_from_palm,
        )

        assembly_specs = []
        if not bool(assembly.get("skip_attitude_compensation", False)):
            assembly_specs.append(
                (
                    "compensate_grasp_attitude",
                    np.asarray(pickup["left_hover_palm_xyz_base"], dtype=float),
                    np.asarray(pickup["right_hover_palm_xyz_base"], dtype=float),
                    left_closed,
                    right_closed,
                    1.0,
                    1.0,
                    left_compensated_rotation,
                    right_compensated_rotation,
                )
            )
        assembly_specs.extend(
            [
                assembly_spec("assembly_staging", right_object_targets["assembly_staging"]),
                assembly_spec("assembly_staging_hold", right_object_targets["assembly_staging"]),
                assembly_spec("approach_slot_far", right_object_targets["approach_slot_far"]),
                assembly_spec("align_plug_tip", right_object_targets["align_plug_tip"]),
                assembly_spec(
                    "alignment_correction_1",
                    right_object_targets["alignment_correction_1"],
                ),
                assembly_spec(
                    "alignment_correction_2",
                    right_object_targets["alignment_correction_2"],
                ),
                assembly_spec("aligned_hold", right_object_targets["align_plug_tip"]),
                assembly_spec("approach_slot_near", right_object_targets["approach_slot_near"]),
                assembly_spec("seat_at_slot_mouth", right_object_targets["seat_at_slot_mouth"]),
                assembly_spec("insert_plug_shallow", right_object_targets["insert_plug_shallow"]),
                assembly_spec("insert_plug", right_object_targets["insert_plug"]),
                assembly_spec("insert_hold", right_object_targets["insert_plug"]),
                (
                    "release_left_hand",
                    left_assembly_palm,
                    right_insert_palm,
                    left_open,
                    right_closed,
                    0.0,
                    1.0,
                    left_compensated_rotation,
                    right_compensated_rotation,
                ),
                (
                    "lower_assembly_right",
                    left_assembly_palm,
                    right_release_palm,
                    left_open,
                    right_closed,
                    0.0,
                    1.0,
                    left_compensated_rotation,
                    right_release_rotation,
                ),
                (
                    "release_right_hand",
                    left_assembly_palm,
                    right_release_palm,
                    left_open,
                    right_open,
                    0.0,
                    0.0,
                    left_compensated_rotation,
                    right_release_rotation,
                ),
                (
                    "post_release_hold",
                    left_assembly_palm,
                    right_release_palm,
                    left_open,
                    right_open,
                    0.0,
                    0.0,
                    left_compensated_rotation,
                    right_release_rotation,
                ),
                (
                    "return_home",
                    left_home_target_pose.translation,
                    right_home_target_pose.translation,
                    left_open,
                    right_open,
                    0.0,
                    0.0,
                    left_home_target_pose.rotation,
                    right_home_target_pose.rotation,
                ),
                (
                    "episode_end",
                    left_home_target_pose.translation,
                    right_home_target_pose.translation,
                    left_open,
                    right_open,
                    0.0,
                    0.0,
                    left_home_target_pose.rotation,
                    right_home_target_pose.rotation,
                ),
            ]
        )
        movement_specs.extend(assembly_specs)

    plan: list[dict] = []

    def append_waypoint(
        name: str,
        left_xyz: np.ndarray,
        right_xyz: np.ndarray,
        left_solution: ArmIKSolution,
        right_solution: ArmIKSolution,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
        left_grasp_scalar: float,
        right_grasp_scalar: float,
        left_rotation: np.ndarray,
        right_rotation: np.ndarray,
    ) -> None:
        left_clearance = fingertip_table_clearance(left_kin, left_solution, "left", left_hand_q)
        right_clearance = fingertip_table_clearance(right_kin, right_solution, "right", right_hand_q)
        plan.append(
            {
                "name": name,
                "left_target_xyz_base": np.asarray(left_xyz, dtype=float).tolist(),
                "left_target_xyz_world": (np.asarray(left_xyz) + base_world_xyz).tolist(),
                "left_target_rot6d": rotation_to_rot6d(left_rotation).tolist(),
                "left_arm_q": left_solution.q_arm.tolist(),
                "left_hand_q": np.asarray(left_hand_q, dtype=float).tolist(),
                "left_grasp": float(left_grasp_scalar),
                "left_ik_success": bool(left_solution.success),
                "left_ik_position_error_m": float(left_solution.position_error_m),
                "left_ik_rotation_error_rad": float(left_solution.rotation_error_rad),
                "left_ik_iterations": int(left_solution.iterations),
                "left_min_fingertip_table_clearance_m": float(left_clearance),
                "right_target_xyz_base": np.asarray(right_xyz, dtype=float).tolist(),
                "right_target_xyz_world": (np.asarray(right_xyz) + base_world_xyz).tolist(),
                "right_target_rot6d": rotation_to_rot6d(right_rotation).tolist(),
                "right_arm_q": right_solution.q_arm.tolist(),
                "right_hand_q": np.asarray(right_hand_q, dtype=float).tolist(),
                "right_grasp": float(right_grasp_scalar),
                "right_ik_success": bool(right_solution.success),
                "right_ik_position_error_m": float(right_solution.position_error_m),
                "right_ik_rotation_error_rad": float(right_solution.rotation_error_rad),
                "right_ik_iterations": int(right_solution.iterations),
                "right_min_fingertip_table_clearance_m": float(right_clearance),
                "socket_box_world_xyz": socket_world.tolist(),
                "rj45_plug_world_xyz": plug_world.tolist(),
                "slot_target_world_xyz": socket_world.tolist(),
            }
        )

    left_home_pose = left_kin.frame_pose(left_seed)
    right_home_pose = right_kin.frame_pose(right_seed)
    left_home_solution = ArmIKSolution(left_seed, left_home, True, 0.0, 0, 0.0)
    right_home_solution = ArmIKSolution(right_seed, right_home, True, 0.0, 0, 0.0)
    append_waypoint(
        "home",
        left_home_pose.translation,
        right_home_pose.translation,
        left_home_solution,
        right_home_solution,
        left_open,
        right_open,
        0.0,
        0.0,
        left_home_pose.rotation,
        right_home_pose.rotation,
    )
    append_waypoint(
        "home_hold",
        left_home_pose.translation,
        right_home_pose.translation,
        left_home_solution,
        right_home_solution,
        left_open,
        right_open,
        0.0,
        0.0,
        left_home_pose.rotation,
        right_home_pose.rotation,
    )

    previous_left_xyz: np.ndarray | None = None
    previous_right_xyz: np.ndarray | None = None
    previous_left_solution: ArmIKSolution | None = None
    previous_right_solution: ArmIKSolution | None = None
    previous_left_rotation: np.ndarray | None = None
    previous_right_rotation: np.ndarray | None = None
    for (
        name,
        left_xyz,
        right_xyz,
        left_hand_q,
        right_hand_q,
        left_scalar,
        right_scalar,
        left_rotation,
        right_rotation,
    ) in movement_specs:
        left_xyz = np.asarray(left_xyz, dtype=float)
        right_xyz = np.asarray(right_xyz, dtype=float)
        left_rotation = np.asarray(left_rotation, dtype=float)
        right_rotation = np.asarray(right_rotation, dtype=float)
        force_home = name in ("return_home", "episode_end")
        if force_home:
            left_solution = left_home_solution
        elif (
            previous_left_xyz is not None
            and previous_left_rotation is not None
            and np.allclose(left_xyz, previous_left_xyz, atol=1e-9)
            and np.allclose(left_rotation, previous_left_rotation, atol=1e-9)
        ):
            left_solution = previous_left_solution
        else:
            left_solution = solve_with_restarts(
                left_kin, left_xyz, left_seed, left_home, pickup_solver, left_rotation
            )
        if force_home:
            right_solution = right_home_solution
        elif (
            previous_right_xyz is not None
            and previous_right_rotation is not None
            and np.allclose(right_xyz, previous_right_xyz, atol=1e-9)
            and np.allclose(right_rotation, previous_right_rotation, atol=1e-9)
        ):
            right_solution = previous_right_solution
        else:
            right_solution = solve_with_restarts(
                right_kin, right_xyz, right_seed, right_home, pickup_solver, right_rotation
            )
        if left_solution is None or right_solution is None:
            raise RuntimeError("Missing cached IK solution")
        left_seed = left_solution.q_full
        right_seed = right_solution.q_full
        previous_left_xyz = left_xyz
        previous_right_xyz = right_xyz
        previous_left_rotation = left_rotation
        previous_right_rotation = right_rotation
        previous_left_solution = left_solution
        previous_right_solution = right_solution
        append_waypoint(
            name,
            left_xyz,
            right_xyz,
            left_solution,
            right_solution,
            left_hand_q,
            right_hand_q,
            left_scalar,
            right_scalar,
            left_rotation,
            right_rotation,
        )
    minimum_clearance = float(pickup["min_fingertip_table_clearance_m"])
    unsafe = [
        waypoint
        for waypoint in plan
        if min(
            waypoint["left_min_fingertip_table_clearance_m"],
            waypoint["right_min_fingertip_table_clearance_m"],
        )
        < minimum_clearance
    ]
    if unsafe:
        details = [
            (
                waypoint["name"],
                round(waypoint["left_min_fingertip_table_clearance_m"], 4),
                round(waypoint["right_min_fingertip_table_clearance_m"], 4),
            )
            for waypoint in unsafe
        ]
        raise ValueError(
            f"Pickup plan violates {minimum_clearance:.3f} m fingertip/table clearance: {details}"
        )
    return plan


def make_rj45_insertion_auto_ik_plan(
    robot_cfg: dict,
    task_cfg: dict,
    root: str | Path,
    *,
    socket_box_world_xyz: list[float] | np.ndarray | None = None,
    rj45_plug_world_xyz: list[float] | np.ndarray | None = None,
) -> list[dict]:
    """Build synchronized left/right arm waypoints for RJ45 insertion.

    This is a kinematic scaffold for raw episode generation. Object grasp and
    insertion contact must still be validated in Isaac with real collision assets.
    """

    root_path = Path(root)
    robot = robot_cfg["robot"]
    scene = task_cfg["scene"]
    frames = task_cfg["frames"]
    scripted = task_cfg["scripted_auto_ik"]

    urdf_path = root_path / robot["preferred_urdf_for_first_pass"]
    left_kin = ArmPinocchioKinematics(
        urdf_path=urdf_path,
        package_dirs=[root_path],
        arm_joints=robot_cfg["left_arm"]["joints"],
        eef_frame=robot["left_eef_link"],
    )
    right_kin = ArmPinocchioKinematics(
        urdf_path=urdf_path,
        package_dirs=[root_path],
        arm_joints=robot_cfg["right_arm"]["joints"],
        eef_frame=robot["right_eef_link"],
    )

    base_world_xyz = np.asarray(scene["robot_base_world_xyz"], dtype=float)
    socket_world = np.asarray(
        socket_box_world_xyz if socket_box_world_xyz is not None else scene["socket_box_initial_xyz"],
        dtype=float,
    )
    plug_world = np.asarray(
        rj45_plug_world_xyz if rj45_plug_world_xyz is not None else scene["rj45_plug_initial_xyz"],
        dtype=float,
    )
    socket_base = socket_world - base_world_xyz
    plug_base = plug_world - base_world_xyz

    socket_grasp = socket_base + np.asarray(frames["socket_grasp_offset_m"], dtype=float)
    socket_hold = socket_base + np.asarray(frames["socket_hold_offset_m"], dtype=float)
    slot = socket_base + np.asarray(frames["socket_slot_offset_m"], dtype=float)
    plug_grasp = plug_base + np.asarray(frames["plug_grasp_offset_m"], dtype=float)
    plug_pre_insert = slot + np.asarray(frames["plug_pre_insert_offset_from_slot_m"], dtype=float)
    plug_insert = slot + np.asarray(frames["plug_insert_offset_from_slot_m"], dtype=float)
    plug_retreat = slot + np.asarray(frames["plug_retreat_offset_from_slot_m"], dtype=float)

    pre_grasp_height = float(scripted["pre_grasp_height_m"])
    lift_height = float(scripted["lift_height_m"])
    left_closed = float(scripted["left_grasp_closed_scalar"])
    right_closed = float(scripted["right_grasp_closed_scalar"])
    close_hold_segments = max(0, int(scripted.get("close_hold_segments", 0)))
    align_hold_segments = max(0, int(scripted.get("align_hold_segments", 0)))
    insert_hold_segments = max(0, int(scripted.get("insert_hold_segments", 0)))

    waypoint_specs = [
        ("home", socket_grasp + [0.0, 0.0, pre_grasp_height], 0.0, plug_grasp + [0.0, 0.0, pre_grasp_height], 0.0),
        ("pre_grasp", socket_grasp + [0.0, 0.0, pre_grasp_height], 0.0, plug_grasp + [0.0, 0.0, pre_grasp_height], 0.0),
        ("descend_to_objects", socket_grasp, 0.0, plug_grasp, 0.0),
        ("grasp_objects", socket_grasp, left_closed, plug_grasp, right_closed),
        *[
            (f"grasp_hold_{index + 1}", socket_grasp, left_closed, plug_grasp, right_closed)
            for index in range(close_hold_segments)
        ],
        ("lift_objects", socket_hold + [0.0, 0.0, lift_height], left_closed, plug_grasp + [0.0, 0.0, lift_height], right_closed),
        ("align_plug_to_slot", socket_hold, left_closed, plug_pre_insert, right_closed),
        *[
            (f"align_hold_{index + 1}", socket_hold, left_closed, plug_pre_insert, right_closed)
            for index in range(align_hold_segments)
        ],
        ("insert_plug", socket_hold, left_closed, plug_insert, right_closed),
        *[
            (f"insert_hold_{index + 1}", socket_hold, left_closed, plug_insert, right_closed)
            for index in range(insert_hold_segments)
        ],
        ("release_right_hand", socket_hold, left_closed, plug_insert, 0.0),
        ("retreat_right_hand", socket_hold, left_closed, plug_retreat, 0.0),
    ]

    left_home = arm_home(robot_cfg, "left")
    right_home = arm_home(robot_cfg, "right")
    left_seed = left_kin.set_arm(left_kin.neutral(), left_home)
    right_seed = right_kin.set_arm(right_kin.neutral(), right_home)

    plan: list[dict] = []
    for name, left_xyz_base, left_grasp, right_xyz_base, right_grasp in waypoint_specs:
        left_xyz = np.asarray(left_xyz_base, dtype=float)
        right_xyz = np.asarray(right_xyz_base, dtype=float)
        left_solution = solve_with_restarts(left_kin, left_xyz, left_seed, left_home, scripted)
        right_solution = solve_with_restarts(right_kin, right_xyz, right_seed, right_home, scripted)
        left_seed = left_solution.q_full
        right_seed = right_solution.q_full
        plan.append(
            {
                "name": name,
                "left_target_xyz_base": left_xyz.tolist(),
                "left_target_xyz_world": (left_xyz + base_world_xyz).tolist(),
                "left_arm_q": left_solution.q_arm.tolist(),
                "left_grasp": float(left_grasp),
                "left_ik_success": bool(left_solution.success),
                "left_ik_position_error_m": float(left_solution.position_error_m),
                "left_ik_iterations": int(left_solution.iterations),
                "right_target_xyz_base": right_xyz.tolist(),
                "right_target_xyz_world": (right_xyz + base_world_xyz).tolist(),
                "right_arm_q": right_solution.q_arm.tolist(),
                "right_grasp": float(right_grasp),
                "right_ik_success": bool(right_solution.success),
                "right_ik_position_error_m": float(right_solution.position_error_m),
                "right_ik_iterations": int(right_solution.iterations),
                "socket_box_world_xyz": socket_world.tolist(),
                "rj45_plug_world_xyz": plug_world.tolist(),
                "slot_target_world_xyz": (slot + base_world_xyz).tolist(),
            }
        )

    return plan
