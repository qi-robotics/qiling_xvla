#!/usr/bin/env python3
"""Preview the fixed upward-facing socket and right-arm RJ45 task scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
from isaacsim import SimulationApp

from run_rj45_isaac_handoff_scene import (
    build_scene,
    euler_xyz_deg_to_matrix,
    load_yaml,
    smoke_test_camera_frames,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-randomize-plug", action="store_true")
    parser.add_argument("--camera-smoke-test", action="store_true")
    parser.add_argument(
        "--task-config",
        default="configs/task_rj45_fixed_socket_right_arm.yaml",
    )
    parser.add_argument("--robot-config", default="configs/robot_dual_arm.yaml")
    parser.add_argument("--camera-config", default="configs/camera_bimanual.yaml")
    parser.add_argument("--stl-scale", type=float, default=0.001)
    return parser.parse_args()


def apply_left_observation_pose(robot_cfg: dict, task_cfg: dict) -> list[float]:
    values = [float(value) for value in task_cfg["left_observation"]["joint_positions"]]
    joint_names = robot_cfg["left_arm"]["joints"]
    if len(values) != len(joint_names):
        raise ValueError(
            f"left_observation.joint_positions has {len(values)} values; "
            f"expected {len(joint_names)}"
        )
    robot_cfg["startup_motion"]["home"]["joint_positions"][:7] = values
    for name, value in zip(joint_names, values):
        robot_cfg["left_arm"]["home_position"][name] = value
        robot_cfg["initial_joint_positions"][name] = value
    return values


def apply_task_hand_presets(robot_cfg: dict, task_cfg: dict) -> None:
    """Let a task override the O6 gripper states used by control and recording."""

    configured = task_cfg.get("expert_auto_ik", {}).get("right_hand_presets", {})
    drive_override = task_cfg.get("expert_auto_ik", {}).get("right_hand_drive", {})
    if drive_override:
        drive_cfg = robot_cfg["isaac_drive"]
        drive_cfg["right_hand_stiffness"] = float(
            drive_override.get("stiffness", drive_cfg["right_hand_stiffness"])
        )
        drive_cfg["right_hand_damping"] = float(
            drive_override.get("damping", drive_cfg["right_hand_damping"])
        )
        drive_cfg["right_hand_max_force"] = float(
            drive_override.get("max_force", drive_cfg.get("right_hand_max_force", 1000.0))
        )
        joint_overrides = drive_override.get("joint_overrides")
        if joint_overrides:
            drive_cfg["right_hand_joint_overrides"] = joint_overrides
    hand = robot_cfg["right_o6_hand"]
    names = hand["active_driver_joints"]
    for task_name, robot_name in (
        ("approach", "approach_preset"),
        ("grasp", "pinch_preset"),
    ):
        values = configured.get(task_name)
        if values is None:
            continue
        if len(values) != len(names):
            raise ValueError(f"right_hand_presets.{task_name} must contain {len(names)} values")
        hand[robot_name] = {name: float(value) for name, value in zip(names, values)}


def smootherstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, float(alpha)))
    return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)


def randomize_plug_position(
    task_cfg: dict,
    seed: int,
    position_bounds_world_m: dict | None = None,
) -> list[float]:
    scene = task_cfg["scene"]
    nominal = [float(value) for value in scene["rj45_plug_initial_xyz"]]
    if position_bounds_world_m is None:
        offsets = scene["rj45_plug_position_randomization_m"]
        bounds = {
            "x": [nominal[0] + float(value) for value in offsets["x"]],
            "y": [nominal[1] + float(value) for value in offsets["y"]],
        }
    else:
        bounds = position_bounds_world_m
    rng = random.Random(seed)
    sampling = scene.get("sampling", {})
    if sampling.get("method") == "stratified_jittered_grid_with_boundary_boost":
        x_low, x_high = (float(value) for value in bounds["x"])
        y_low, y_high = (float(value) for value in bounds["y"])
        x_center, y_center = (x_low + x_high) * 0.5, (y_low + y_high) * 0.5
        x_limit, y_limit = (x_high - x_low) * 0.5, (y_high - y_low) * 0.5
        slot = int(seed) % 10
        if slot == 0:
            dx = rng.choice((-1.0, 1.0)) * rng.uniform(0.78 * x_limit, x_limit)
            dy = rng.choice((-1.0, 1.0)) * rng.uniform(0.78 * y_limit, y_limit)
        elif slot in (1, 2):
            if rng.random() < 0.5:
                dx = rng.choice((-1.0, 1.0)) * rng.uniform(0.78 * x_limit, x_limit)
                dy = rng.uniform(-0.78 * y_limit, 0.78 * y_limit)
            else:
                dx = rng.uniform(-0.78 * x_limit, 0.78 * x_limit)
                dy = rng.choice((-1.0, 1.0)) * rng.uniform(0.78 * y_limit, y_limit)
        else:
            dx = rng.uniform(-0.78 * x_limit, 0.78 * x_limit)
            dy = rng.uniform(-0.78 * y_limit, 0.78 * y_limit)
        sampled = [x_center + dx, y_center + dy, nominal[2]]
    else:
        sampled = [
            rng.uniform(*[float(value) for value in bounds["x"]]),
            rng.uniform(*[float(value) for value in bounds["y"]]),
            nominal[2],
        ]
    scene["rj45_plug_initial_xyz"] = sampled
    return sampled


def validate_fixed_upward_socket(task_cfg: dict) -> None:
    import numpy as np

    socket = task_cfg["assets"]["socket_box"]
    if not bool(socket.get("fixed", False)):
        raise ValueError("The fixed-socket scene requires assets.socket_box.fixed=true")
    rotation = euler_xyz_deg_to_matrix(
        task_cfg["scene"]["socket_box_initial_rotate_xyz_deg"]
    )
    expert = task_cfg["expert_auto_ik"]
    opening_world = rotation @ np.asarray(
        expert.get("opening_axis_local_xyz", [0.0, 0.0, -1.0]), dtype=float
    )
    plug_rotation = euler_xyz_deg_to_matrix(
        task_cfg["expert_auto_ik"]["plug_assembly_rotate_xyz_deg"]
    )
    plug_axis = plug_rotation @ np.asarray(
        expert.get("plug_axis_local_xyz", [0.0, 0.0, -1.0]), dtype=float
    )
    opposing = float(np.dot(opening_world, -plug_axis))
    opening_lateral = float(np.linalg.norm(opening_world[:2]))
    plug_lateral = float(np.linalg.norm(plug_axis[:2]))
    if (
        opening_lateral > 1.0e-6
        or plug_lateral > 1.0e-6
        or float(opening_world[2]) < 0.999999
        or float(plug_axis[2]) > -0.999999
        or opposing < 0.999999
    ):
        raise ValueError(
            "The fixed socket opening must point exactly upward and oppose the "
            f"held plug axis; opening={opening_world.tolist()} "
            f"plug_axis={plug_axis.tolist()} dot={opposing:.6f}"
        )


def main() -> int:
    args = parse_args()
    app = SimulationApp(
        {"headless": bool(args.headless), "width": 1280, "height": 720}
    )
    try:
        robot_cfg = load_yaml(ROOT / args.robot_config)
        task_cfg = load_yaml(ROOT / args.task_config)
        camera_cfg = load_yaml(ROOT / args.camera_config)
        validate_fixed_upward_socket(task_cfg)
        left_pose = apply_left_observation_pose(robot_cfg, task_cfg)
        plug_xyz = [float(value) for value in task_cfg["scene"]["rj45_plug_initial_xyz"]]
        if not args.no_randomize_plug:
            plug_xyz = randomize_plug_position(task_cfg, args.seed)

        print(f"Fixed socket base XY={task_cfg['scene']['socket_box_initial_xyz'][:2]}", flush=True)
        print(
            f"Plug sample seed={args.seed} xyz={plug_xyz}; orientation is fixed",
            flush=True,
        )
        print(f"Frozen left observation joints={left_pose}", flush=True)

        camera_paths, object_points, _, _ = build_scene(
            app,
            robot_cfg,
            task_cfg,
            camera_cfg,
            args.stl_scale,
            startup_realtime=not args.headless,
        )
        corridor_height = float(
            task_cfg["left_observation"].get("insertion_corridor_height_m", 0.10)
        )
        socket_point = list(object_points[0])
        socket_rotation = euler_xyz_deg_to_matrix(
            task_cfg["scene"]["socket_box_initial_rotate_xyz_deg"]
        )
        opening_axis = socket_rotation @ np.asarray([0.0, 0.0, -1.0])
        inspection_points = [
            socket_point,
            list(object_points[1]),
            (
                np.asarray(socket_point) + opening_axis * corridor_height
            ).tolist(),
        ]
        if args.camera_smoke_test:
            smoke_test_camera_frames(
                app,
                camera_paths,
                inspection_points,
                camera_cfg,
                required_object_indices={
                    "chest": [0, 1],
                    "left_wrist": [0, 2],
                    "right_wrist": [1],
                },
            )

        if args.headless:
            return 0
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        print("Scene ready. Use the GUI camera menu to inspect chest/left/right wrist views.", flush=True)
        started = time.monotonic()
        while app.is_running():
            app.update()
            if args.seconds > 0.0 and time.monotonic() - started >= args.seconds:
                break
            time.sleep(1.0 / 60.0)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
