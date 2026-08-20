#!/usr/bin/env python3
"""Quick physical two-finger grasp/lift verifier for the slender pin."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from isaacsim import SimulationApp

from run_rj45_isaac_handoff_scene import (
    build_scene,
    coupled_hand_joint_names,
    coupled_hand_positions,
    euler_xyz_deg_to_matrix,
    load_yaml,
    quaternion_xyzw_to_matrix,
    startup_joint_names,
)
from run_rj45_fixed_socket_scene import (
    apply_left_observation_pose,
    apply_task_hand_presets,
    smootherstep,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--seconds-after", type=float, default=0.0)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument(
        "--complete-insertion",
        action="store_true",
        help="Continue past the validated grasp/lift into the full assembly.",
    )
    parser.add_argument(
        "--record-out-dir",
        default=None,
        help="Write a three-camera expert episode to this directory.",
    )
    parser.add_argument(
        "--recovery-type",
        default="normal",
        choices=("normal", "lateral_offset", "angular_offset"),
        help="Perturbation injected at the socket mouth and then corrected.",
    )
    parser.add_argument(
        "--camera-preview-dir",
        default=None,
        help="Save one PNG per camera at each assembly checkpoint for review.",
    )
    parser.add_argument(
        "--task-config",
        default="configs/task_slender_pin_insertion_right_arm.yaml",
    )
    parser.add_argument("--robot-config", default="configs/robot_dual_arm.yaml")
    parser.add_argument("--camera-config", default="configs/camera_bimanual.yaml")
    return parser.parse_args()


def world_pose(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath(prim_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        np.asarray(translation, dtype=float),
        quaternion_xyzw_to_matrix(
            [imaginary[0], imaginary[1], imaginary[2], quaternion.GetReal()]
        ),
    )


def rotation_error_rad(actual: np.ndarray, reference: np.ndarray) -> float:
    cosine = float(
        np.clip((np.trace(reference.T @ actual) - 1.0) * 0.5, -1.0, 1.0)
    )
    return float(math.acos(cosine))


def rotation_vector_deg(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    delta = reference.T @ actual
    angle = rotation_error_rad(actual, reference)
    if angle < 1.0e-8:
        return np.zeros(3, dtype=float)
    axis = np.asarray(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(angle))
    return axis * math.degrees(angle)


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    skew = np.asarray(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3)
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )


def perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, unit))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=float)
    perpendicular = reference - float(np.dot(reference, unit)) * unit
    return perpendicular / np.linalg.norm(perpendicular)


def hand_preset(task_cfg: dict, name: str) -> np.ndarray:
    result = np.asarray(
        task_cfg["expert_auto_ik"]["right_hand_presets"][name], dtype=float
    )
    if result.shape != (6,):
        raise ValueError(f"right_hand_presets.{name} must contain six values")
    return result


def main() -> int:
    args = parse_args()
    if args.replay_speed <= 0.0:
        raise ValueError("--replay-speed must be positive")
    if args.record_out_dir and not args.complete_insertion:
        raise ValueError("--record-out-dir requires --complete-insertion")
    if args.recovery_type != "normal" and not args.complete_insertion:
        raise ValueError("--recovery-type requires --complete-insertion")
    sys.argv[:] = [sys.argv[0]] + (["--headless"] if args.headless else [])
    app = SimulationApp(
        {"headless": bool(args.headless), "width": 1280, "height": 720}
    )
    try:
        import carb
        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.api.sensors import RigidContactView
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.carb import set_carb_setting
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        from qiling_xvla.control.dual_arm_pinocchio import ArmPinocchioKinematics
        from qiling_xvla.control.posture_safe_ik import (
            solve_posture_safe_pose,
            torso_safe_joint_path,
        )

        robot_cfg = load_yaml(ROOT / args.robot_config)
        task_cfg = load_yaml(ROOT / args.task_config)
        camera_cfg = load_yaml(ROOT / args.camera_config)
        scene_cfg = task_cfg["scene"]
        randomization = scene_cfg.get("rj45_plug_position_randomization_m")
        if randomization is not None:
            sampled_xyz = np.asarray(
                scene_cfg["rj45_plug_initial_xyz"], dtype=float
            )
            rng = np.random.default_rng(args.episode_seed)
            sampled_xyz[0] += rng.uniform(*map(float, randomization["x"]))
            sampled_xyz[1] += rng.uniform(*map(float, randomization["y"]))
            scene_cfg["rj45_plug_initial_xyz"] = sampled_xyz.tolist()
            print(
                "Slender-pin grasp sample: "
                f"seed={args.episode_seed} xyz={np.round(sampled_xyz, 5).tolist()} "
                "orientation=fixed",
                flush=True,
            )

        startup_scale = float(
            task_cfg.get("simulation", {}).get(
                "startup_motion_duration_scale", 1.0
            )
        )
        for startup_phase in ("spread", "home"):
            robot_cfg["startup_motion"][startup_phase][
                "duration_hint"
            ] *= startup_scale
        apply_task_hand_presets(robot_cfg, task_cfg)
        apply_left_observation_pose(robot_cfg, task_cfg)
        camera_paths, _, articulation_root, robot_model_root = build_scene(
            app,
            robot_cfg,
            task_cfg,
            camera_cfg,
            1.0,
            startup_realtime=not args.headless,
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        articulation = Articulation(articulation_root)
        articulation.initialize()
        plug_rigid = RigidPrim(
            "/World/RJ45Plug",
            name="slender_pin_mass_probe",
            reset_xform_properties=False,
        )
        plug_rigid.initialize()
        print(
            "Slender-pin PhysX mass kg="
            f"{np.asarray(plug_rigid.get_masses(), dtype=float).reshape(-1).tolist()}",
            flush=True,
        )
        stage = omni.usd.get_context().get_stage()
        contact_filter_labels = (
            "thumb_base",
            "thumb_metacarpal",
            "thumb_distal",
            "index_proximal",
            "index_distal",
            "table",
        )
        contact_view = RigidContactView(
            prim_paths_expr="/World/RJ45Plug",
            filter_paths_expr=[[
                f"{robot_model_root}/RH_thumb_metacarpals_base2",
                f"{robot_model_root}/RH_thumb_metacarpals",
                f"{robot_model_root}/RH_thumb_distal",
                f"{robot_model_root}/RH_index_proximal",
                f"{robot_model_root}/RH_index_distal",
                "/World/Table",
            ]],
            name="slender_pin_contact_view",
            max_contact_count=64,
        )
        contact_view.initialize()

        controlled_names = startup_joint_names(robot_cfg)
        controlled_indices = np.asarray(
            [articulation.get_dof_index(name) for name in controlled_names]
        )
        coupled_names = coupled_hand_joint_names("left") + coupled_hand_joint_names(
            "right"
        )
        coupled_indices = np.asarray(
            [articulation.get_dof_index(name) for name in coupled_names]
        )
        if np.any(controlled_indices < 0) or np.any(coupled_indices < 0):
            raise RuntimeError("The Isaac articulation is missing controlled O6 joints")

        expert = task_cfg["expert_auto_ik"]
        phase_frames = expert["phase_frames"]
        base_world = np.asarray(scene_cfg["robot_base_world_xyz"], dtype=float)
        right_kin = ArmPinocchioKinematics(
            ROOT / robot_cfg["robot"]["preferred_urdf_for_first_pass"],
            [ROOT],
            robot_cfg["right_arm"]["joints"],
            robot_cfg["robot"]["right_eef_link"],
        )
        right_home = np.asarray(
            [
                robot_cfg["right_arm"]["home_position"][name]
                for name in robot_cfg["right_arm"]["joints"]
            ],
            dtype=float,
        )
        left_hold = np.asarray(
            task_cfg["left_observation"]["joint_positions"], dtype=float
        )
        left_open = np.zeros(6, dtype=float)
        right_open = np.zeros(6, dtype=float)
        right_approach = hand_preset(task_cfg, "approach")
        right_seat = hand_preset(task_cfg, "seat")
        right_preload = hand_preset(task_cfg, "preload")
        right_grasp = hand_preset(task_cfg, "grasp")
        right_release_break = hand_preset(task_cfg, "release_break")
        for name, values in (
            ("approach", right_approach),
            ("seat", right_seat),
            ("preload", right_preload),
            ("grasp", right_grasp),
        ):
            if not np.allclose(values[3:], 0.0):
                raise ValueError(
                    f"Two-finger verifier requires middle/ring/pinky open in {name}"
                )

        grasp_rotation = np.asarray(
            expert["right_grasp_rotation_matrix"], dtype=float
        )
        grasp_offset = np.asarray(
            expert["right_grasp_center_from_palm_m"], dtype=float
        )
        pin_initial_xyz, pin_initial_rotation = world_pose(stage, "/World/RJ45Plug")
        pin_base = pin_initial_xyz - base_world
        grasp_center = pin_base + np.asarray(
            [0.0, 0.0, float(expert["right_grasp_center_world_z_offset_m"])]
        )
        nominal_grasp_palm = grasp_center - grasp_rotation @ grasp_offset
        grasp_palm = nominal_grasp_palm.copy()
        grasp_palm[0] += float(expert.get("right_grasp_palm_x_offset_m", 0.0))
        grasp_palm[1] += float(expert.get("right_grasp_palm_y_offset_m", 0.0))
        approach_direction = np.asarray(
            expert["grasp_approach_direction_base"], dtype=float
        )
        approach_direction /= np.linalg.norm(approach_direction)
        pregrasp_palm = grasp_palm + approach_direction * float(
            expert["pregrasp_clearance_m"]
        )
        contact_palm = grasp_palm + approach_direction * float(
            expert["dynamic_contact_clearance_m"]
        )
        breakaway_palm = grasp_palm + np.asarray([0.0, 0.0, 0.004])
        lift_height = float(expert["lift_height_m"])
        lift_mid_palm = grasp_palm + np.asarray(
            [0.0, 0.0, min(0.020, lift_height)]
        )
        lift_palm = grasp_palm + np.asarray(
            [0.0, 0.0, lift_height]
        )

        current_q = right_home.copy()

        def solve(
            name: str,
            target_xyz: np.ndarray,
            target_rotation: np.ndarray | None = None,
        ) -> np.ndarray:
            nonlocal current_q
            if target_rotation is None:
                target_rotation = grasp_rotation
            solution, report = solve_posture_safe_pose(
                right_kin,
                "right",
                target_xyz,
                target_rotation,
                current_q,
                right_home,
                expert["solver"],
                expert["posture"],
            )
            path_safe = torso_safe_joint_path(
                right_kin,
                "right",
                current_q,
                solution.q_arm,
                expert["posture"],
            )
            print(
                f"Two-finger IK {name}: success={solution.success} "
                f"path_safe={path_safe} target={np.round(target_xyz, 4).tolist()} "
                f"position={solution.position_error_m * 1000.0:.2f}mm "
                f"rotation={math.degrees(solution.rotation_error_rad):.2f}deg "
                f"posture_valid={bool(report['valid'])} "
                f"elbow_out={report['elbow_outward_y_m']:.3f} "
                f"joint_margin={report['minimum_joint_margin_ratio']:.3f}",
                flush=True,
            )
            if not solution.success or not path_safe:
                raise RuntimeError(f"Unsafe or unreachable two-finger target: {name}")
            current_q = solution.q_arm.copy()
            return current_q.copy()

        pregrasp_q = solve("pregrasp", pregrasp_palm)
        contact_q = solve("contact", contact_palm)
        grasp_q = solve("grasp", grasp_palm)
        breakaway_q = solve("lift_breakaway", breakaway_palm)
        lift_mid_q = solve("lift_mid", lift_mid_palm)
        lift_q = solve("lift", lift_palm)

        physics_hz = float(task_cfg["simulation"]["physics_hz"])
        control_hz = float(task_cfg["simulation"]["record_hz"])
        substeps = max(1, int(round(physics_hz / control_hz)))
        render_settings = carb.settings.get_settings()
        render_counter = 0
        render_interval = max(1, int(round(physics_hz / 60.0)))

        def render_without_physics() -> None:
            # SimulationManager owns the deterministic PhysX clock below.
            # Disable Kit's implicit simulation while refreshing the viewport.
            set_carb_setting(
                render_settings, "/app/player/playSimulations", False
            )
            try:
                app.update()
            finally:
                set_carb_setting(
                    render_settings, "/app/player/playSimulations", True
                )

        def step_physics() -> None:
            # Keep GUI and headless replay on the same fixed PhysX clock. Using
            # app.update() here made closure depend on viewport render timing.
            nonlocal render_counter
            SimulationManager.step(render=False)
            if not args.headless:
                render_counter += 1
                if render_counter >= render_interval:
                    render_without_physics()
                    render_counter = 0
                time.sleep(1.0 / (physics_hz * args.replay_speed))

        def capture_frame(
            phase: str, frame_target: np.ndarray, step_index: int
        ) -> None:
            """Record one 20 Hz sample at the start of each control frame.

            The observation is read before the command that is about to run, so
            behavior cloning sees the same ordering the policy will see.
            """

            if recorder is None or step_index % substeps != 0:
                return
            render_without_physics()
            actual = np.asarray(
                articulation.get_joint_positions(joint_indices=controlled_indices),
                dtype=float,
            ).reshape(-1)
            recorder.capture(phase, actual, frame_target)

        current_command = np.asarray(
            articulation.get_joint_positions(joint_indices=controlled_indices),
            dtype=float,
        ).reshape(-1)
        current_command[:7] = left_hold
        current_command[7:13] = left_open
        current_coupled = np.asarray(
            coupled_hand_positions("left", left_open)
            + coupled_hand_positions("right", current_command[20:26]),
            dtype=float,
        )

        def measured_contact_forces() -> np.ndarray:
            forces = contact_view.get_contact_force_matrix(dt=1.0 / physics_hz)
            if forces is None:
                return np.zeros((len(contact_filter_labels), 3), dtype=float)
            return np.asarray(forces, dtype=float).reshape(-1, 3)

        def finger_contact_force_norms(forces: np.ndarray) -> np.ndarray:
            thumb_force = np.sum(forces[:3], axis=0)
            index_force = np.sum(forces[3:5], axis=0)
            return np.asarray(
                [np.linalg.norm(thumb_force), np.linalg.norm(index_force)],
                dtype=float,
            )

        def print_contact_details(label: str) -> None:
            normal_forces, points, normals, distances, starts, counts = (
                contact_view.get_contact_force_data(dt=1.0 / physics_hz)
            )
            normal_forces = np.asarray(normal_forces, dtype=float).reshape(-1)
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            normals = np.asarray(normals, dtype=float).reshape(-1, 3)
            distances = np.asarray(distances, dtype=float).reshape(-1)
            starts = np.asarray(starts, dtype=int).reshape(-1)
            counts = np.asarray(counts, dtype=int).reshape(-1)
            details = {}
            for filter_index, filter_label in enumerate(contact_filter_labels):
                start = int(starts[filter_index])
                count = int(counts[filter_index])
                if count <= 0:
                    continue
                details[filter_label] = [
                    {
                        "force_n": round(float(normal_forces[item]), 3),
                        "point": np.round(points[item], 5).tolist(),
                        "normal": np.round(normals[item], 3).tolist(),
                        "separation_mm": round(float(distances[item]) * 1000.0, 3),
                    }
                    for item in range(start, start + count)
                ]
            print(f"Two-finger contact details {label}: {details}", flush=True)

        def print_collision_bounds() -> None:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            print("Two-finger runtime collision bounds:", flush=True)
            for link_name in (
                "RH_thumb_metacarpals_base2",
                "RH_thumb_metacarpals",
                "RH_thumb_distal",
                "RH_index_proximal",
                "RH_index_distal",
            ):
                link_path = f"{robot_model_root}/{link_name}"
                link_prim = stage.GetPrimAtPath(link_path)
                link_xyz, _ = world_pose(stage, link_path)
                colliders = []
                for descendant in Usd.PrimRange(link_prim):
                    descendant_path = str(descendant.GetPath())
                    if (
                        not descendant.HasAPI(UsdPhysics.CollisionAPI)
                        and "/collisions" not in descendant_path.lower()
                    ):
                        continue
                    world_range = bbox_cache.ComputeWorldBound(
                        descendant
                    ).ComputeAlignedRange()
                    lower = np.asarray(world_range.GetMin(), dtype=float)
                    upper = np.asarray(world_range.GetMax(), dtype=float)
                    colliders.append(
                        {
                            "path": descendant_path,
                            "physics_material": [
                                str(target)
                                for relationship in descendant.GetRelationships()
                                if relationship.GetName() == "material:binding:physics"
                                for target in relationship.GetTargets()
                            ],
                            "min": np.round(lower, 5).tolist(),
                            "max": np.round(upper, 5).tolist(),
                        }
                    )
                link_materials = [
                    str(target)
                    for relationship in link_prim.GetRelationships()
                    if relationship.GetName() == "material:binding:physics"
                    for target in relationship.GetTargets()
                ]
                print(
                    f"  {link_name}: origin={np.round(link_xyz, 5).tolist()} "
                    f"physics_material={link_materials} "
                    f"colliders={colliders}",
                    flush=True,
                )

        # The approach and closure sequence is tuned around the tracking lag of
        # a loaded position drive, so the hand command is only held across phase
        # boundaries once the grip is locked onto the pin.
        grip_locked = False

        def command(
            name: str,
            q_arm: np.ndarray,
            q_hand: np.ndarray,
            right_coupled_override: np.ndarray | None = None,
            frames: int | None = None,
        ) -> None:
            nonlocal current_command, current_coupled
            count = int(phase_frames[name] if frames is None else frames)
            target = np.concatenate([left_hold, left_open, q_arm, q_hand])
            right_coupled_target = (
                np.asarray(right_coupled_override, dtype=float)
                if right_coupled_override is not None
                else np.asarray(coupled_hand_positions("right", q_hand), dtype=float)
            )
            target_coupled = np.concatenate(
                [
                    np.asarray(coupled_hand_positions("left", left_open), dtype=float),
                    right_coupled_target,
                ]
            )
            start = current_command.copy()
            coupled_start = current_coupled.copy()
            print(f"Two-finger replay phase: {name}", flush=True)
            total_substeps = max(1, count * substeps)
            for step_index in range(total_substeps):
                alpha = smootherstep((step_index + 1) / float(total_substeps))
                frame_target = start + alpha * (target - start)
                frame_coupled = coupled_start + alpha * (
                    target_coupled - coupled_start
                )
                capture_frame(name, frame_target, step_index)
                articulation.set_joint_position_targets(
                    np.asarray([frame_target], dtype=np.float32),
                    joint_indices=controlled_indices,
                )
                articulation.set_joint_velocity_targets(
                    np.zeros((1, len(controlled_indices)), dtype=np.float32),
                    joint_indices=controlled_indices,
                )
                articulation.set_joint_position_targets(
                    np.asarray([frame_coupled], dtype=np.float32),
                    joint_indices=coupled_indices,
                )
                articulation.set_joint_velocity_targets(
                    np.zeros((1, len(coupled_indices)), dtype=np.float32),
                    joint_indices=coupled_indices,
                )
                step_physics()
            measured = np.asarray(
                articulation.get_joint_positions(joint_indices=controlled_indices),
                dtype=float,
            ).reshape(-1)
            current_command = measured.copy()
            current_command[:7] = left_hold
            current_command[7:13] = left_open
            current_coupled = np.asarray(
                articulation.get_joint_positions(joint_indices=coupled_indices),
                dtype=float,
            ).reshape(-1)
            if grip_locked:
                # Continue the next phase from the commanded posture, not the
                # measured one. A loaded position drive sits behind its target,
                # so re-seeding from the measurement drops the holding torque:
                # the grip releases at every phase boundary, and the arm command
                # jumps back to the sagged pose before advancing again.
                current_command[13:20] = target[13:20]
                current_command[20:26] = target[20:26]
                current_coupled[5:] = target_coupled[5:]
            pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
            palm_xyz, palm_rotation = world_pose(
                stage, f"{robot_model_root}/RH_palm_center"
            )
            palm_rotation_error_deg = math.degrees(
                rotation_error_rad(palm_rotation, grasp_rotation)
            )
            pin_rotation_delta_deg = math.degrees(
                rotation_error_rad(pin_rotation, pin_initial_rotation)
            )
            pin_rotation_vector_deg = rotation_vector_deg(
                pin_rotation, pin_initial_rotation
            )
            actual_right_hand = measured[20:26]
            contact_forces = measured_contact_forces()
            print(
                f"Two-finger phase end {name}: pin_xyz="
                f"{np.round(pin_xyz, 5).tolist()} displacement_mm="
                f"{np.round((pin_xyz - pin_initial_xyz) * 1000.0, 2).tolist()} "
                f"pin_rotation_deg={pin_rotation_delta_deg:.2f} "
                f"pin_rotation_vector_deg="
                f"{np.round(pin_rotation_vector_deg, 2).tolist()} "
                f"palm_xyz={np.round(palm_xyz, 5).tolist()} "
                f"palm_rotation_error_deg={palm_rotation_error_deg:.2f} "
                f"contact_forces_N={dict(zip(contact_filter_labels, np.round(contact_forces, 3).tolist()))} "
                f"actual_index={actual_right_hand[2]:.4f}",
                flush=True,
            )

        def pin_from_palm() -> tuple[np.ndarray, np.ndarray]:
            palm_xyz, palm_rotation = world_pose(
                stage, f"{robot_model_root}/RH_palm_center"
            )
            pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
            return (
                palm_rotation.T @ (pin_xyz - palm_xyz),
                palm_rotation.T @ pin_rotation,
            )

        recorder = None
        if args.record_out_dir:
            from qiling_xvla.data.slender_pin_episode_recorder import (
                SlenderPinEpisodeRecorder,
            )

            record_dir = Path(args.record_out_dir)
            recorder = SlenderPinEpisodeRecorder(
                app=app,
                stage=stage,
                output_dir=record_dir if record_dir.is_absolute() else ROOT / record_dir,
                camera_paths=camera_paths,
                camera_cfg=camera_cfg,
                robot_cfg=robot_cfg,
                robot_model_root=robot_model_root,
                right_kin=right_kin,
                record_hz=control_hz,
                instruction=task_cfg["task"]["language_instruction"],
                seed=args.episode_seed,
                recovery_type=args.recovery_type,
            )

        preview_sensors: dict = {}
        preview_dir = None
        if args.camera_preview_dir:
            import cv2
            from isaacsim.sensors.camera import Camera

            preview_dir = Path(args.camera_preview_dir)
            if not preview_dir.is_absolute():
                preview_dir = ROOT / preview_dir
            preview_dir.mkdir(parents=True, exist_ok=True)
            if recorder is not None:
                preview_sensors = dict(recorder.sensors)
            else:
                for index, (name, path) in enumerate(
                    zip(("chest", "left_wrist", "right_wrist"), camera_paths)
                ):
                    settings = camera_cfg["cameras"][name]
                    sensor = Camera(
                        prim_path=path,
                        name=f"slender_pin_preview_{name}_{index}",
                        resolution=(int(settings["width"]), int(settings["height"])),
                    )
                    sensor.initialize()
                    preview_sensors[name] = sensor
                for _ in range(12):
                    app.update()

        def save_camera_preview(label: str) -> None:
            if not preview_sensors:
                return
            render_without_physics()
            for name, sensor in preview_sensors.items():
                rgba = np.asarray(sensor.get_rgba())
                if rgba.ndim != 3 or rgba.shape[2] < 3:
                    print(f"Camera preview unavailable: {label} {name}", flush=True)
                    continue
                cv2.imwrite(
                    str(preview_dir / f"{label}_{name}.png"),
                    np.asarray(rgba[:, :, 2::-1], dtype=np.uint8),
                )
            print(f"Saved camera preview: {label}", flush=True)

        command("home_hold", right_home, right_open)
        command("pre_grasp", pregrasp_q, right_open)
        command("shape_hand", pregrasp_q, right_approach)
        command("descend_to_handle", contact_q, right_approach)
        command("contact_handle", grasp_q, right_approach)

        body = UsdPhysics.RigidBodyAPI.Get(stage, "/World/RJ45Plug")
        if not body:
            raise RuntimeError("Slender pin is missing PhysicsRigidBodyAPI")
        body.GetKinematicEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Get(
            stage, "/World/RJ45Plug"
        ).GetEnableCCDAttr().Set(True)
        print("Slender pin switched to dynamic mode before index closure", flush=True)
        command("settle_contact", grasp_q, right_approach)
        command("close_preload", grasp_q, right_seat)
        command("close_grasp", grasp_q, right_preload)

        print("Two-finger replay phase: close_until_bilateral_contact", flush=True)
        contact_cfg = expert.get("two_finger_contact", {})
        minimum_contact_force = float(contact_cfg.get("minimum_force_n", 0.15))
        required_bilateral_steps = int(
            contact_cfg.get("stable_physics_steps", 5)
        )
        maximum_closure_steps = int(
            contact_cfg.get("maximum_closure_physics_steps", 150)
        )
        bilateral_steps = 0
        closure_steps = 0
        target = np.concatenate([left_hold, left_open, grasp_q, right_grasp])
        target_coupled = np.asarray(
            coupled_hand_positions("left", left_open)
            + coupled_hand_positions("right", right_grasp),
            dtype=float,
        )
        for closure_steps in range(1, maximum_closure_steps + 1):
            capture_frame("close_until_contact", target, closure_steps - 1)
            articulation.set_joint_position_targets(
                np.asarray([target], dtype=np.float32),
                joint_indices=controlled_indices,
            )
            articulation.set_joint_velocity_targets(
                np.zeros((1, len(controlled_indices)), dtype=np.float32),
                joint_indices=controlled_indices,
            )
            articulation.set_joint_position_targets(
                np.asarray([target_coupled], dtype=np.float32),
                joint_indices=coupled_indices,
            )
            articulation.set_joint_velocity_targets(
                np.zeros((1, len(coupled_indices)), dtype=np.float32),
                joint_indices=coupled_indices,
            )
            step_physics()
            forces = measured_contact_forces()
            finger_force_norms = finger_contact_force_norms(forces)
            if np.all(finger_force_norms >= minimum_contact_force):
                bilateral_steps += 1
                if bilateral_steps >= required_bilateral_steps:
                    break
            else:
                bilateral_steps = 0
        measured = np.asarray(
            articulation.get_joint_positions(joint_indices=controlled_indices),
            dtype=float,
        ).reshape(-1)
        current_command = measured.copy()
        current_command[:7] = left_hold
        current_command[7:13] = left_open
        current_coupled = np.asarray(
            articulation.get_joint_positions(joint_indices=coupled_indices),
            dtype=float,
        ).reshape(-1)
        forces = measured_contact_forces()
        finger_force_norms = finger_contact_force_norms(forces)
        pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
        print(
            "Two-finger contact closure result: "
            f"steps={closure_steps} stable={bilateral_steps}/"
            f"{required_bilateral_steps} threshold={minimum_contact_force:.3f}N "
            f"forces_N={dict(zip(contact_filter_labels, np.round(forces, 3).tolist()))} "
            f"finger_force_norms_N={np.round(finger_force_norms, 3).tolist()} "
            f"actual_hand={np.round(measured[20:26], 4).tolist()} "
            f"pin_displacement_mm="
            f"{np.round((pin_xyz - pin_initial_xyz) * 1000.0, 2).tolist()} "
            f"pin_rotation_deg="
            f"{math.degrees(rotation_error_rad(pin_rotation, pin_initial_rotation)):.2f}",
            flush=True,
        )
        if bilateral_steps < required_bilateral_steps:
            print_collision_bounds()
            raise RuntimeError("PhysX did not establish bilateral finger contact")
        print_collision_bounds()
        print_contact_details("closed")
        locked_hand = measured[20:26].copy()
        thumb_preload = float(contact_cfg.get("thumb_lock_preload_rad", 0.008))
        index_preload = float(contact_cfg.get("index_lock_preload_rad", 0.025))
        locked_hand[0] = min(float(right_grasp[0]), locked_hand[0] + thumb_preload)
        locked_hand[1] = min(float(right_grasp[1]), locked_hand[1] + thumb_preload)
        locked_hand[2] = min(float(right_grasp[2]), locked_hand[2] + index_preload)
        locked_hand[3:] = 0.0
        locked_right_coupled = current_coupled[5:].copy()
        coupled_preload = np.asarray(
            contact_cfg.get("coupled_lock_preload_rad", [0.008, 0.020]),
            dtype=float,
        )
        grasp_coupled = np.asarray(
            coupled_hand_positions("right", right_grasp), dtype=float
        )
        locked_right_coupled[0] = min(
            float(grasp_coupled[0]), locked_right_coupled[0] + float(coupled_preload[0])
        )
        locked_right_coupled[1] = min(
            float(grasp_coupled[1]), locked_right_coupled[1] + float(coupled_preload[1])
        )
        locked_right_coupled[2:] = 0.0
        # From here on the commanded posture is carried across phase boundaries.
        # grasp_hold still ramps into the lock from the measured angles; jumping
        # straight to it drives a step into a stiff position drive and hammers
        # the pin.
        grip_locked = True
        print(
            "Two-finger contact lock: active="
            f"{np.round(locked_hand, 4).tolist()} coupled="
            f"{np.round(locked_right_coupled, 4).tolist()}",
            flush=True,
        )

        force_low = float(contact_cfg.get("force_control_low_n", 0.055))
        force_high = float(contact_cfg.get("force_control_high_n", 0.30))
        thumb_step = float(
            contact_cfg.get("force_control_thumb_step_rad", 0.00035)
        )
        index_step = float(
            contact_cfg.get("force_control_index_step_rad", 0.00150)
        )
        maximum_hand = np.asarray(
            [
                contact_cfg.get("force_control_max_thumb_yaw_rad", 0.80),
                contact_cfg.get("force_control_max_thumb_pitch_rad", 0.34),
                contact_cfg.get("force_control_max_index_rad", 0.82),
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        # Measured contact force saturates far below what a deeper closure would
        # imply, so the absolute joint caps sit beyond the point where the jaws
        # eject this light pin. Bound the travel against the closure that first
        # held it instead.
        extra_closure = float(
            contact_cfg.get("force_control_max_extra_closure_rad", 0.070)
        )
        maximum_hand[:3] = np.minimum(maximum_hand[:3], locked_hand[:3] + extra_closure)
        maximum_coupled = locked_right_coupled[:2] + extra_closure * np.asarray(
            [1.86, 0.89], dtype=float
        )
        persistence_steps = int(
            contact_cfg.get("force_control_persistence_steps", 8)
        )
        print(
            "Two-finger closure bounds: "
            f"hand_max={np.round(maximum_hand[:3], 4).tolist()} "
            f"coupled_max={np.round(maximum_coupled, 4).tolist()} "
            f"persistence={persistence_steps}",
            flush=True,
        )

        def force_controlled_command(
            name: str, q_arm: np.ndarray, frames: int | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
            """Move the arm while maintaining bilateral PhysX contact force."""

            nonlocal current_command, current_coupled, locked_hand
            nonlocal locked_right_coupled
            count = int(phase_frames[name] if frames is None else frames)
            start = current_command.copy()
            coupled_start = current_coupled.copy()
            arm_target = np.concatenate(
                [left_hold, left_open, q_arm, locked_hand]
            )
            total_substeps = max(1, count * substeps)
            minimum_seen = np.full(2, np.inf, dtype=float)
            maximum_seen = np.zeros(2, dtype=float)
            final_forces = np.zeros(2, dtype=float)
            low_streak = np.zeros(2, dtype=int)
            print(f"Two-finger force-control phase: {name}", flush=True)
            for step_index in range(total_substeps):
                alpha = smootherstep((step_index + 1) / float(total_substeps))
                frame_target = start + alpha * (arm_target - start)
                # Track the grip command directly. Ramping it in over the phase
                # both releases the pin at the phase boundary and attenuates the
                # increments the force controller makes below.
                frame_target[20:26] = locked_hand
                left_coupled = np.asarray(
                    coupled_hand_positions("left", left_open), dtype=float
                )
                frame_coupled = coupled_start.copy()
                frame_coupled[:5] = coupled_start[:5] + alpha * (
                    left_coupled - coupled_start[:5]
                )
                frame_coupled[5:] = locked_right_coupled
                capture_frame(name, frame_target, step_index)
                articulation.set_joint_position_targets(
                    np.asarray([frame_target], dtype=np.float32),
                    joint_indices=controlled_indices,
                )
                articulation.set_joint_velocity_targets(
                    np.zeros((1, len(controlled_indices)), dtype=np.float32),
                    joint_indices=controlled_indices,
                )
                articulation.set_joint_position_targets(
                    np.asarray([frame_coupled], dtype=np.float32),
                    joint_indices=coupled_indices,
                )
                articulation.set_joint_velocity_targets(
                    np.zeros((1, len(coupled_indices)), dtype=np.float32),
                    joint_indices=coupled_indices,
                )
                step_physics()

                final_forces = finger_contact_force_norms(
                    measured_contact_forces()
                )
                minimum_seen = np.minimum(minimum_seen, final_forces)
                maximum_seen = np.maximum(maximum_seen, final_forces)

                # Increase only the jaw that has been unloaded for several
                # consecutive steps, and never while either jaw is already
                # highly loaded. Reacting to a single zero reading turned this
                # into a closing ramp that ejected the pin.
                low_streak = np.where(final_forces < force_low, low_streak + 1, 0)
                if np.any(final_forces > force_high):
                    continue
                if low_streak[0] >= persistence_steps:
                    low_streak[0] = 0
                    locked_hand[0] = min(
                        maximum_hand[0], locked_hand[0] + thumb_step
                    )
                    locked_hand[1] = min(
                        maximum_hand[1], locked_hand[1] + thumb_step
                    )
                    locked_right_coupled[0] = min(
                        maximum_coupled[0],
                        locked_right_coupled[0] + thumb_step * 1.86,
                    )
                if low_streak[1] >= persistence_steps:
                    low_streak[1] = 0
                    locked_hand[2] = min(
                        maximum_hand[2], locked_hand[2] + index_step
                    )
                    locked_right_coupled[1] = min(
                        maximum_coupled[1],
                        locked_right_coupled[1] + index_step * 0.89,
                    )

            measured = np.asarray(
                articulation.get_joint_positions(joint_indices=controlled_indices),
                dtype=float,
            ).reshape(-1)
            current_command = measured.copy()
            current_command[:7] = left_hold
            current_command[7:13] = left_open
            current_command[13:20] = arm_target[13:20]
            current_command[20:26] = locked_hand
            current_coupled = np.asarray(
                articulation.get_joint_positions(joint_indices=coupled_indices),
                dtype=float,
            ).reshape(-1)
            current_coupled[5:] = locked_right_coupled
            pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
            print(
                f"Two-finger force-control end {name}: pin_displacement_mm="
                f"{np.round((pin_xyz - pin_initial_xyz) * 1000.0, 2).tolist()} "
                f"pin_rotation_deg="
                f"{math.degrees(rotation_error_rad(pin_rotation, pin_initial_rotation)):.2f} "
                f"force_min_N={np.round(minimum_seen, 3).tolist()} "
                f"force_max_N={np.round(maximum_seen, 3).tolist()} "
                f"force_final_N={np.round(final_forces, 3).tolist()} "
                f"hand_target={np.round(locked_hand, 4).tolist()}",
                flush=True,
            )
            return final_forces, maximum_seen

        command(
            "grasp_hold",
            grasp_q,
            locked_hand,
            locked_right_coupled,
        )
        hold_forces = finger_contact_force_norms(measured_contact_forces())
        minimum_hold_force = float(
            contact_cfg.get("minimum_hold_force_n", minimum_contact_force)
        )
        if np.any(hold_forces < minimum_hold_force):
            raise RuntimeError(
                "Two-finger preload did not remain bilateral: "
                f"forces={np.round(hold_forces, 3).tolist()}"
            )

        pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
        object_size = np.asarray(expert["object_size_m"], dtype=float)
        object_center = pin_xyz + pin_rotation @ np.asarray(
            [0.0, 0.0, object_size[2] * 0.5]
        )
        half = object_size * 0.5
        contacts: dict[str, dict[str, object]] = {}
        for finger in ("thumb", "index"):
            mesh_path = ROOT / (
                f"robot_description/S4/meshes/o6/right/meshes/{finger}_distal.STL"
            )
            mesh = __import__("trimesh").load_mesh(mesh_path)
            vertices = np.asarray(mesh.vertices, dtype=float)
            normals = np.asarray(mesh.vertex_normals, dtype=float)
            tip_mask = vertices[:, 2] >= float(vertices[:, 2].max()) - 0.012
            local = vertices[tip_mask]
            local_normals = normals[tip_mask]
            matrix = UsdGeom.Xformable(
                stage.GetPrimAtPath(f"{robot_model_root}/RH_{finger}_distal")
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world = np.asarray(
                [matrix.Transform(Gf.Vec3d(*point)) for point in local], dtype=float
            )
            block_local = (pin_rotation.T @ (world - object_center).T).T
            outside = np.maximum(np.abs(block_local) - half, 0.0)
            distances = np.linalg.norm(outside, axis=1)
            closest = int(np.argmin(distances))
            link_rotation = np.asarray(matrix.ExtractRotationMatrix(), dtype=float)
            normal_world = link_rotation @ local_normals[closest]
            contacts[finger] = {
                "distance_m": float(distances[closest]),
                "point_local_m": block_local[closest],
                "normal_local": pin_rotation.T @ normal_world,
            }
        thumb_point = np.asarray(contacts["thumb"]["point_local_m"])
        index_point = np.asarray(contacts["index"]["point_local_m"])
        actual_hand = np.asarray(
            articulation.get_joint_positions(joint_indices=controlled_indices),
            dtype=float,
        ).reshape(-1)[20:26]
        contact_tolerance = 0.0015
        opposing = bool(
            thumb_point[0] <= -half[0] + 0.004
            and index_point[0] >= half[0] - 0.004
        )
        bilateral_contact = bool(
            opposing
            and float(contacts["thumb"]["distance_m"]) <= contact_tolerance
            and float(contacts["index"]["distance_m"]) <= contact_tolerance
        )
        closure_ready = bool(
            opposing
            and float(contacts["thumb"]["distance_m"]) <= 0.003
            and float(contacts["index"]["distance_m"]) <= contact_tolerance
        )
        print(
            "Two-finger contact geometry: "
            f"thumb={float(contacts['thumb']['distance_m']) * 1000.0:.2f}mm@"
            f"{np.round(thumb_point * 1000.0, 1).tolist()} "
            f"index={float(contacts['index']['distance_m']) * 1000.0:.2f}mm@"
            f"{np.round(index_point * 1000.0, 1).tolist()} "
            f"normals={dict((name, np.round(np.asarray(data['normal_local']), 3).tolist()) for name, data in contacts.items())} "
            f"actual_hand={np.round(actual_hand, 4).tolist()} "
            f"opposing={opposing} bilateral={bilateral_contact} "
            f"closure_ready={closure_ready}",
            flush=True,
        )
        if not closure_ready:
            raise RuntimeError("Thumb/index closure is not positioned across the pin")

        force_controlled_command("lift_breakaway", breakaway_q)
        print_contact_details("after_breakaway")
        # The first 4 mm is also the loaded self-centering phase: the index can
        # slide the pin laterally into the fixed thumb before vertical support
        # is established. Judge pickup only after the intermediate lift waypoint.
        force_controlled_command("lift_mid", lift_mid_q)
        lift_mid_xyz, _ = world_pose(stage, "/World/RJ45Plug")
        lift_mid_height = float(lift_mid_xyz[2] - pin_initial_xyz[2])
        if lift_mid_height < 0.010:
            raise RuntimeError(
                "Two-finger pickup failed at lift_mid: "
                f"lift={lift_mid_height * 1000.0:.2f}mm"
            )
        force_controlled_command("lift_handle", lift_q)
        hold_translation_before, hold_rotation_before = pin_from_palm()
        force_controlled_command("lift_hold", lift_q)
        final_xyz, _ = world_pose(stage, "/World/RJ45Plug")
        hold_translation_after, hold_rotation_after = pin_from_palm()

        lift = float(final_xyz[2] - pin_initial_xyz[2])
        translation_slip = float(
            np.linalg.norm(hold_translation_after - hold_translation_before)
        )
        rotation_slip = rotation_error_rad(
            hold_rotation_after, hold_rotation_before
        )
        lift_valid = lift >= float(expert["validation"]["required_lift_m"])
        slip_valid = translation_slip <= float(
            expert["validation"]["maximum_hold_slip_m"]
        ) and math.degrees(rotation_slip) <= float(
            expert["validation"].get("maximum_hold_rotation_error_deg", 5.0)
        )
        print(
            "Two-finger grasp result: "
            f"lift={lift * 1000.0:.2f}mm "
            f"hold_translation_slip={translation_slip * 1000.0:.2f}mm "
            f"hold_rotation_slip={math.degrees(rotation_slip):.2f}deg "
            f"lift_valid={lift_valid} slip_valid={slip_valid}",
            flush=True,
        )
        if not lift_valid or not slip_valid:
            raise RuntimeError("Two-finger physical grasp/lift validation failed")
        print("Slender-pin two-finger grasp/lift verifier PASSED", flush=True)
        save_camera_preview("after_lift")

        if args.complete_insertion:
            # Let the closure settle before transport and confirm the pin is
            # still in the jaws. The peak force is the meaningful signal here;
            # single-step readings drop to zero even under load.
            _, preload_peak = force_controlled_command("grip_preload_hold", lift_q)
            preload_pin_xyz, _ = world_pose(stage, "/World/RJ45Plug")
            preload_lift = float(preload_pin_xyz[2] - pin_initial_xyz[2])
            if np.any(preload_peak < minimum_hold_force) or preload_lift < 0.010:
                raise RuntimeError(
                    "Two-finger grip was not holding the pin before transport: "
                    f"peak_forces={np.round(preload_peak, 3).tolist()}N "
                    f"lift={preload_lift * 1000.0:.2f}mm"
                )
            print(
                "Two-finger transport preload ready: "
                f"peak_forces={np.round(preload_peak, 3).tolist()}N "
                f"lift={preload_lift * 1000.0:.2f}mm",
                flush=True,
            )

            validation = expert["validation"]
            socket_world, socket_rotation_world = world_pose(stage, "/World/RJ45Socket")
            socket_entry = socket_world + socket_rotation_world @ np.asarray(
                expert["socket_entry_local_xyz_m"], dtype=float
            )
            opening_axis = socket_rotation_world @ np.asarray(
                expert["opening_axis_local_xyz"], dtype=float
            )
            opening_axis /= np.linalg.norm(opening_axis)
            pin_tip_local = np.asarray(expert["plug_tip_local_xyz_m"], dtype=float)
            pin_axis_local = np.asarray(expert["plug_axis_local_xyz"], dtype=float)
            # The pin is picked up upright and the socket opens along +Z, so the
            # assembly attitude equals the spawn attitude. No wrist flip is
            # needed; only the small tilt acquired while grasping is corrected.
            desired_pin_rotation = euler_xyz_deg_to_matrix(
                expert["plug_assembly_rotate_xyz_deg"]
            )
            assembly_bias = np.zeros(3, dtype=float)

            def insertion_metrics() -> tuple[float, float, float, np.ndarray]:
                pin_xyz, pin_rotation = world_pose(stage, "/World/RJ45Plug")
                tip = pin_xyz + pin_rotation @ pin_tip_local
                delta = socket_entry - tip
                depth = float(np.dot(delta, opening_axis))
                lateral_vector = delta - depth * opening_axis
                pin_axis = pin_rotation @ pin_axis_local
                orientation = math.acos(
                    float(np.clip(np.dot(pin_axis, -opening_axis), -1.0, 1.0))
                )
                return (
                    depth,
                    float(np.linalg.norm(lateral_vector)),
                    orientation,
                    lateral_vector,
                )

            def corridor_center(height_m: float) -> np.ndarray:
                return socket_entry + opening_axis * float(height_m) + assembly_bias

            transport_reference_offset, _ = pin_from_palm()
            maximum_transport_slip = float(
                expert["validation"].get("maximum_transport_slip_m", 0.008)
            )

            def verify_grip(phase: str) -> None:
                """Fail on the phase that drops the pin, not three phases later.

                Once the pin is on the table, the measured pin-from-palm offset
                is meaningless and every later waypoint solves for a palm pose
                that drifts further from the socket. Instantaneous contact force
                reads zero too often to be used as the signal, so track where
                the pin sits inside the jaws instead.
                """

                offset, _ = pin_from_palm()
                slip = float(np.linalg.norm(offset - transport_reference_offset))
                pin_xyz, _ = world_pose(stage, "/World/RJ45Plug")
                held_height = float(pin_xyz[2] - pin_initial_xyz[2])
                if slip > maximum_transport_slip or held_height < 0.010:
                    forces = finger_contact_force_norms(measured_contact_forces())
                    raise RuntimeError(
                        f"Slender pin lost during {phase}: "
                        f"slip_in_jaws={slip * 1000.0:.2f}mm "
                        f"height={held_height * 1000.0:.2f}mm "
                        f"forces={np.round(forces, 3).tolist()}N"
                    )

            def move_pin_to(
                phase: str,
                center_world: np.ndarray,
                *,
                pin_rotation_target: np.ndarray | None = None,
                frames: int | None = None,
                check_grip: bool = True,
                grasp_transform: tuple[np.ndarray, np.ndarray] | None = None,
            ) -> np.ndarray:
                # Re-measure the physical grasp every waypoint. Replaying a
                # stale palm offset cannot track slip or post-contact shifts.
                # Callers that drive the pin against a constraint pass a frozen
                # transform instead, because there the measured pose reflects
                # contact deflection and solving against it fights the socket.
                if grasp_transform is None:
                    pin_offset, pin_rotation_in_palm = pin_from_palm()
                else:
                    pin_offset, pin_rotation_in_palm = grasp_transform
                target_rotation = (
                    desired_pin_rotation
                    if pin_rotation_target is None
                    else np.asarray(pin_rotation_target, dtype=float)
                )
                palm_rotation = target_rotation @ pin_rotation_in_palm.T
                palm_xyz = (
                    np.asarray(center_world, dtype=float)
                    - palm_rotation @ pin_offset
                    - base_world
                )
                q_arm = solve(phase, palm_xyz, palm_rotation)
                force_controlled_command(phase, q_arm, frames=frames)
                if check_grip:
                    verify_grip(phase)
                return q_arm

            def correct_alignment(
                phase: str,
                height_m: float,
                *,
                passes: int,
                gain: float,
                lateral_tolerance: float,
                axial_tolerance: float,
                orientation_tolerance_deg: float,
                max_lateral_step: float,
                max_axial_step: float,
                max_total_lateral: float,
                max_total_axial: float,
                settle_frames: int,
                minimum_passes: int = 0,
                grasp_transform: tuple[np.ndarray, np.ndarray] | None = None,
            ) -> bool:
                nonlocal assembly_bias
                total_lateral = 0.0
                total_axial = 0.0
                for index in range(1, int(passes) + 1):
                    depth, lateral, orientation, lateral_vector = insertion_metrics()
                    axial_error = depth + float(height_m)
                    converged = bool(
                        lateral <= lateral_tolerance
                        and abs(axial_error) <= axial_tolerance
                        and math.degrees(orientation) <= orientation_tolerance_deg
                    )
                    print(
                        f"Slender-pin {phase} {index}: "
                        f"lateral={lateral * 1000.0:.2f}mm "
                        f"axial={axial_error * 1000.0:.2f}mm "
                        f"orientation={math.degrees(orientation):.2f}deg "
                        f"converged={converged}",
                        flush=True,
                    )
                    if converged and index > minimum_passes:
                        return True
                    lateral_step = lateral_vector * float(gain)
                    step_norm = float(np.linalg.norm(lateral_step))
                    if step_norm > max_lateral_step:
                        lateral_step = lateral_step * (max_lateral_step / step_norm)
                        step_norm = max_lateral_step
                    if total_lateral + step_norm > max_total_lateral:
                        print(
                            f"Slender-pin {phase} lateral budget exhausted",
                            flush=True,
                        )
                        lateral_step = np.zeros(3, dtype=float)
                        step_norm = 0.0
                    axial_step = float(
                        np.clip(
                            axial_error * float(gain), -max_axial_step, max_axial_step
                        )
                    )
                    if total_axial + abs(axial_step) > max_total_axial:
                        print(
                            f"Slender-pin {phase} axial budget exhausted", flush=True
                        )
                        axial_step = 0.0
                    if step_norm <= 0.0 and axial_step == 0.0:
                        break
                    total_lateral += step_norm
                    total_axial += abs(axial_step)
                    assembly_bias = (
                        assembly_bias + lateral_step + opening_axis * axial_step
                    )
                    q_hold = move_pin_to(
                        phase,
                        corridor_center(height_m),
                        grasp_transform=grasp_transform,
                    )
                    # The next pass re-solves the palm pose from the measured pin
                    # orientation, so measuring mid-swing feeds the swing back
                    # into the arm. Hold still until the pin stops moving.
                    if int(settle_frames) > 0:
                        force_controlled_command(
                            phase, q_hold, frames=int(settle_frames)
                        )
                depth, lateral, orientation, _ = insertion_metrics()
                axial_error = depth + float(height_m)
                converged = bool(
                    lateral <= lateral_tolerance
                    and abs(axial_error) <= axial_tolerance
                    and math.degrees(orientation) <= orientation_tolerance_deg
                )
                print(
                    f"Slender-pin {phase} final: "
                    f"lateral={lateral * 1000.0:.2f}mm "
                    f"axial={axial_error * 1000.0:.2f}mm "
                    f"orientation={math.degrees(orientation):.2f}deg "
                    f"converged={converged}",
                    flush=True,
                )
                return converged

            transport_height = float(expert["transport_clearance_above_socket_m"])
            preinsert_height = float(expert["preinsert_clearance_m"])
            mouth_height = float(expert["mouth_alignment_clearance_m"])
            print(
                "Slender-pin assembly geometry: "
                f"socket_entry={np.round(socket_entry, 5).tolist()} "
                f"opening_axis={np.round(opening_axis, 4).tolist()} "
                f"transport={transport_height * 1000.0:.1f}mm "
                f"preinsert={preinsert_height * 1000.0:.1f}mm "
                f"mouth={mouth_height * 1000.0:.1f}mm",
                flush=True,
            )

            held_pin_xyz, _ = world_pose(stage, "/World/RJ45Plug")
            move_pin_to(
                "transport_high",
                np.asarray(
                    [
                        float(held_pin_xyz[0]),
                        float(held_pin_xyz[1]),
                        float(socket_entry[2] + transport_height),
                    ]
                ),
            )
            move_pin_to("transport_mid", corridor_center(transport_height))
            above_socket_q = move_pin_to(
                "above_socket", corridor_center(transport_height)
            )
            save_camera_preview("above_socket")

            move_pin_to("preinsert", corridor_center(preinsert_height))
            if not correct_alignment(
                "alignment_correction",
                preinsert_height,
                passes=int(expert["alignment_correction_passes"]),
                gain=float(validation["alignment_correction_gain"]),
                lateral_tolerance=float(
                    validation["maximum_preinsert_lateral_error_m"]
                ),
                axial_tolerance=float(validation["maximum_preinsert_axial_error_m"]),
                orientation_tolerance_deg=float(
                    validation["maximum_preinsert_orientation_error_deg"]
                ),
                max_lateral_step=float(
                    validation["maximum_preinsert_lateral_step_m"]
                ),
                max_axial_step=float(validation["maximum_alignment_axial_step_m"]),
                max_total_lateral=float(
                    validation["maximum_preinsert_total_lateral_correction_m"]
                ),
                max_total_axial=float(
                    validation["maximum_preinsert_total_axial_correction_m"]
                ),
                settle_frames=int(expert["alignment_settle_frames"]),
                grasp_transform=pin_from_palm(),
            ):
                raise RuntimeError("Slender-pin preinsert corridor alignment failed")

            move_pin_to("seat_at_socket_mouth", corridor_center(mouth_height))
            save_camera_preview("socket_mouth")

            # A disturbance along one hard-coded direction is not a disturbance:
            # the grasp geometry leaves a systematic mouth misalignment, and a
            # fixed offset either cancels it or reinforces it every single time.
            # It also teaches one correction direction. Give each episode its own
            # direction in the plane normal to the opening axis, drawn from the
            # episode seed so the dataset stays reproducible.
            recovery_rng = np.random.default_rng([int(args.episode_seed), 0x5EC0])
            plane_u = perpendicular_axis(opening_axis)
            plane_v = np.cross(opening_axis, plane_u)
            plane_v = plane_v / np.linalg.norm(plane_v)
            recovery_angle = float(recovery_rng.uniform(0.0, 2.0 * math.pi))
            recovery_direction = (
                math.cos(recovery_angle) * plane_u
                + math.sin(recovery_angle) * plane_v
            )

            minimum_mouth_passes = 0
            if args.recovery_type == "lateral_offset":
                magnitude = float(expert["lateral_recovery_offset_m"])
                injected = magnitude * recovery_direction
                assembly_bias = assembly_bias + injected
                print(
                    "Slender-pin recovery injection lateral_mm="
                    f"{np.round(injected * 1000.0, 3).tolist()} "
                    f"magnitude_mm={magnitude * 1000.0:.2f} "
                    f"direction_deg={math.degrees(recovery_angle):.1f}",
                    flush=True,
                )
                move_pin_to("recovery_inject", corridor_center(mouth_height))
                minimum_mouth_passes = 2
            elif args.recovery_type == "angular_offset":
                injected_deg = float(expert["angular_recovery_offset_deg"])
                perturbed_rotation = (
                    axis_angle_matrix(
                        recovery_direction, math.radians(injected_deg)
                    )
                    @ desired_pin_rotation
                )
                print(
                    f"Slender-pin recovery injection angular_deg={injected_deg:.3f} "
                    f"axis_deg={math.degrees(recovery_angle):.1f}",
                    flush=True,
                )
                move_pin_to(
                    "recovery_inject",
                    corridor_center(mouth_height),
                    pin_rotation_target=perturbed_rotation,
                )
                minimum_mouth_passes = 2

            mouth_converged = correct_alignment(
                "mouth_alignment_correction",
                mouth_height,
                passes=int(expert["mouth_alignment_correction_passes"]),
                gain=float(validation["mouth_alignment_correction_gain"]),
                lateral_tolerance=float(validation["maximum_mouth_lateral_error_m"]),
                axial_tolerance=float(validation["maximum_mouth_axial_error_m"]),
                orientation_tolerance_deg=float(
                    validation["maximum_mouth_orientation_error_deg"]
                ),
                max_lateral_step=float(validation["maximum_mouth_lateral_step_m"]),
                max_axial_step=float(validation["maximum_mouth_axial_step_m"]),
                max_total_lateral=float(
                    validation["maximum_mouth_total_lateral_correction_m"]
                ),
                max_total_axial=float(
                    validation["maximum_mouth_total_axial_correction_m"]
                ),
                settle_frames=int(expert["alignment_settle_frames"]),
                minimum_passes=minimum_mouth_passes,
                grasp_transform=pin_from_palm(),
            )
            if not mouth_converged:
                depth, lateral, orientation, _ = insertion_metrics()
                mouth_converged = bool(
                    lateral <= float(validation["coarse_mouth_lateral_error_m"])
                    and abs(depth + mouth_height)
                    <= float(validation["coarse_mouth_axial_error_m"])
                    and math.degrees(orientation)
                    <= float(validation["coarse_mouth_orientation_error_deg"])
                )
                print(
                    f"Slender-pin coarse mouth acceptance={mouth_converged}", flush=True
                )
            if not mouth_converged:
                raise RuntimeError("Slender-pin socket-mouth alignment failed")

            # Axial bias belongs to the alignment stage only; the insertion
            # depth below is commanded directly along the opening axis.
            lateral_bias = (
                assembly_bias
                - float(np.dot(assembly_bias, opening_axis)) * opening_axis
            )
            insertion_step = float(expert["insertion_step_m"])
            insertion_step_frames = int(expert["insertion_step_frames"])
            insertion_depth = float(expert["insertion_depth_m"])
            transient_lateral = float(
                validation["maximum_insertion_transient_lateral_error_m"]
            )
            transient_orientation = float(
                validation["maximum_insertion_transient_orientation_error_deg"]
            )
            insertion_steps = int(
                math.ceil((mouth_height + insertion_depth) / insertion_step)
            )
            inserted_q = above_socket_q
            depth = lateral = orientation = 0.0
            for step_index in range(1, insertion_steps + 1):
                commanded_depth = min(
                    insertion_depth, -mouth_height + step_index * insertion_step
                )
                phase = (
                    "descend_centered_to_mouth"
                    if commanded_depth < 0.0
                    else "insert_pin"
                )
                inserted_q = move_pin_to(
                    phase,
                    socket_entry - opening_axis * commanded_depth + lateral_bias,
                    frames=insertion_step_frames,
                    # The socket walls progressively carry the pin, so finger
                    # force is no longer the right integrity signal here.
                    check_grip=False,
                )
                depth, lateral, orientation, _ = insertion_metrics()
                if step_index % 10 == 0 or depth >= insertion_depth:
                    print(
                        f"Slender-pin insertion feedback {step_index:03d}: "
                        f"commanded={commanded_depth * 1000.0:.1f}mm "
                        f"depth={depth * 1000.0:.2f}mm "
                        f"lateral={lateral * 1000.0:.2f}mm "
                        f"orientation={math.degrees(orientation):.2f}deg",
                        flush=True,
                    )
                if (
                    lateral > transient_lateral
                    or math.degrees(orientation) > transient_orientation
                ):
                    raise RuntimeError(
                        "Slender-pin insertion left the guided corridor: "
                        f"lateral={lateral * 1000.0:.2f}mm "
                        f"orientation={math.degrees(orientation):.2f}deg"
                    )
                if depth >= insertion_depth:
                    break

            force_controlled_command("insert_hold", inserted_q)
            depth, lateral, orientation, _ = insertion_metrics()
            print(
                "Slender-pin insertion before release: "
                f"depth={depth * 1000.0:.2f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )
            save_camera_preview("inserted")

            # Sweeping both jaws open in one motion drags the pin with them:
            # fingertip friction is deliberately enormous and the thumb drive
            # carries 160 N, which is more than enough to push a 2 g pin through
            # a static socket wall. Retract only the moving jaw far enough to
            # unload the pin, let gravity seat it, and open fully afterwards.
            command("release_break", inserted_q, right_release_break)
            depth, lateral, orientation, _ = insertion_metrics()
            print(
                "Slender-pin after jaw unload: "
                f"depth={depth * 1000.0:.2f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )
            pin_physx = PhysxSchema.PhysxRigidBodyAPI.Get(stage, "/World/RJ45Plug")
            pin_physx.GetLinearDampingAttr().Set(
                float(
                    task_cfg["simulation"].get("object_release_linear_damping", 0.05)
                )
            )
            pin_physx.GetAngularDampingAttr().Set(
                float(
                    task_cfg["simulation"].get("object_release_angular_damping", 0.05)
                )
            )
            thumb_xyz, _ = world_pose(
                stage, f"{robot_model_root}/RH_thumb_distal"
            )
            pin_xyz, _ = world_pose(stage, "/World/RJ45Plug")
            peel = thumb_xyz - pin_xyz
            peel = peel - opening_axis * float(np.dot(peel, opening_axis))
            peel_norm = float(np.linalg.norm(peel))
            palm_xyz, palm_rotation = world_pose(
                stage, f"{robot_model_root}/RH_palm_center"
            )
            clear_q = inserted_q
            if peel_norm > 1e-6:
                peel = peel / peel_norm
                clear_offset = float(expert.get("release_clear_offset_m", 0.018))
                clear_q = solve(
                    "release_clear",
                    palm_xyz + peel * clear_offset - base_world,
                    palm_rotation,
                )
                command("release_clear", clear_q, right_release_break)
            depth, lateral, orientation, _ = insertion_metrics()
            print(
                "Slender-pin after thumb peel: "
                f"depth={depth * 1000.0:.2f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )
            command("release_pin", clear_q, right_open)
            # Get the fingers out of the channel mouth before the pin falls, so
            # residual pad contact cannot wedge it. The drop is then observed
            # from the already-cleared above-socket pose.
            command("retreat_from_socket", above_socket_q, right_open)
            # A tilted pin slides down the channel far slower than it would fall
            # freely, so the settle is split into chunks that log the descent.
            # Sampling it as one block cannot tell a jam from an unfinished fall.
            settle_total = int(phase_frames["release_settle"])
            settle_chunks = max(1, int(expert["release_settle_chunks"]))
            chunk_frames = max(1, settle_total // settle_chunks)
            for chunk in range(1, settle_chunks + 1):
                command("release_settle", above_socket_q, right_open, frames=chunk_frames)
                depth, lateral, orientation, _ = insertion_metrics()
                print(
                    f"Slender-pin release settle {chunk}: "
                    f"depth={depth * 1000.0:.2f}mm "
                    f"lateral={lateral * 1000.0:.2f}mm "
                    f"orientation={math.degrees(orientation):.2f}deg",
                    flush=True,
                )
            depth, lateral, orientation, _ = insertion_metrics()
            print(
                "Slender-pin after release drop: "
                f"depth={depth * 1000.0:.2f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )

            minimum_depth = float(validation["minimum_inserted_depth_m"])
            maximum_depth = float(validation["maximum_inserted_depth_m"])
            maximum_final_lateral = float(validation["maximum_final_lateral_error_m"])
            maximum_final_orientation = float(
                validation["maximum_final_orientation_error_deg"]
            )
            stable_frames = int(validation["insertion_success_stable_frames"])
            stable_depths = []
            for _ in range(stable_frames):
                command(
                    "seated_stability_hold", above_socket_q, right_open, frames=1
                )
                depth, lateral, orientation, _ = insertion_metrics()
                stable_depths.append(depth)
                if (
                    depth < minimum_depth
                    or depth > maximum_depth
                    or lateral > maximum_final_lateral
                    or math.degrees(orientation) > maximum_final_orientation
                ):
                    raise RuntimeError(
                        "Slender-pin seated pose failed the stability hold: "
                        f"depth={depth * 1000.0:.2f}mm "
                        f"lateral={lateral * 1000.0:.2f}mm "
                        f"orientation={math.degrees(orientation):.2f}deg"
                    )
            seated_drift = float(max(stable_depths) - min(stable_depths))
            print(
                "Slender-pin seated stability hold: "
                f"frames={stable_frames} "
                f"depth={depth * 1000.0:.2f}mm "
                f"drift={seated_drift * 1000.0:.3f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )
            save_camera_preview("released")

            command("return_home", right_home, right_open)
            depth, lateral, orientation, _ = insertion_metrics()
            print(
                "Slender-pin insertion result after retreat: "
                f"depth={depth * 1000.0:.2f}mm "
                f"lateral={lateral * 1000.0:.2f}mm "
                f"orientation={math.degrees(orientation):.2f}deg",
                flush=True,
            )
            if (
                depth < minimum_depth
                or depth > maximum_depth
                or lateral > maximum_final_lateral
                or math.degrees(orientation) > maximum_final_orientation
            ):
                raise RuntimeError("Slender-pin assembly was disturbed by the retreat")

            if recorder is not None:
                recorder.finalize(
                    plug_xyz=pin_initial_xyz.tolist(),
                    socket_xyz=socket_world.tolist(),
                    metrics={
                        "physical_grasp_lift_m": lift,
                        "hold_slip_m": translation_slip,
                        "hold_rotation_slip_rad": rotation_slip,
                        "insertion_depth_m": depth,
                        "lateral_error_m": lateral,
                        "orientation_error_rad": orientation,
                        "seated_depth_drift_m": seated_drift,
                        "insertion_stable_frames": stable_frames,
                    },
                )
            print(
                f"Slender-pin expert episode PASS seed={args.episode_seed} "
                f"recovery={args.recovery_type}",
                flush=True,
            )
        else:
            # Keep the final force-controlled closure active while the GUI
            # remains open; commanding the measured angles here would release
            # the preload that is holding the pin.
            current_command[20:26] = locked_hand
            current_coupled[5:] = locked_right_coupled

        if args.headless:
            return 0
        started = time.monotonic()
        while app.is_running():
            articulation.set_joint_position_targets(
                np.asarray([current_command], dtype=np.float32),
                joint_indices=controlled_indices,
            )
            articulation.set_joint_position_targets(
                np.asarray([current_coupled], dtype=np.float32),
                joint_indices=coupled_indices,
            )
            SimulationManager.step(render=False)
            render_without_physics()
            if (
                args.seconds_after > 0.0
                and time.monotonic() - started >= args.seconds_after
            ):
                break
            time.sleep(1.0 / 60.0)
        return 0
    except Exception as exc:
        # A failed episode must never leave a committed video or metadata file
        # behind; partial expert data silently poisons the training set.
        if "recorder" in locals() and recorder is not None:
            recorder.abort()
        print(
            f"Slender-pin two-finger verifier failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
