#!/usr/bin/env python3
"""Load the S4 bimanual RJ45 insertion handoff scene in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class RgbVideoWriter:
    """Stream RGB frames to an H.264 MP4 without retaining them in memory."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        self.width = int(width)
        self.height = int(height)
        self.frame_count = 0
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{float(fps):.8g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.partial_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, rgb) -> None:
        import numpy as np

        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Video frame has shape {frame.shape}, expected "
                f"{(self.height, self.width, 3)}"
            )
        if self._closed or self._process.stdin is None:
            raise RuntimeError(f"Video writer is closed: {self.path}")
        self._process.stdin.write(frame.tobytes())
        self.frame_count += 1

    def close(self, commit: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        returncode = self._process.wait()
        if commit and returncode == 0:
            self.partial_path.replace(self.path)
            return
        self.partial_path.unlink(missing_ok=True)
        if commit:
            raise RuntimeError(
                f"ffmpeg failed with return code {returncode}: {self.path}"
            )

    def __del__(self) -> None:
        try:
            self.close(commit=False)
        except Exception:
            pass


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_stl_mesh(stl_path: Path, scale: float, center_all_axes: bool = True):
    import numpy as np
    import trimesh

    mesh = trimesh.load_mesh(stl_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float) * float(scale)
    faces = np.asarray(mesh.faces, dtype=int)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"No triangles found in {stl_path}")
    if center_all_axes:
        vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    return vertices, faces


def define_mesh(stage, prim_path: str, stl_path: Path, color: tuple[float, float, float], scale: float) -> None:
    from pxr import Gf, UsdGeom

    vertices, faces = load_stl_mesh(stl_path, scale=scale)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def make_dynamic_mesh_body(
    stage,
    root_path: str,
    mesh_path: str,
    mass_kg: float,
    start_kinematic: bool = False,
) -> None:
    import numpy as np
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body.CreateKinematicEnabledAttr(bool(start_kinematic))
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(float(mass_kg))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_body.CreateEnableCCDAttr(not bool(start_kinematic))
    physx_body.CreateSolverPositionIterationCountAttr(16)
    physx_body.CreateSolverVelocityIterationCountAttr(4)
    physx_body.CreateMaxDepenetrationVelocityAttr(1.0)
    points = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=float)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    collision = UsdGeom.Cube.Define(stage, f"{root_path}/collision_proxy")
    collision.CreateSizeAttr(1.0)
    collision.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.25, 0.20)])
    set_xform(
        collision.GetPrim(),
        translate=(bounds_min + bounds_max) * 0.5,
        scale=bounds_max - bounds_min,
    )
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()


def bind_material_to_colliders(stage, root_prim, material_path) -> int:
    from omni.physx.scripts import physicsUtils
    from pxr import Usd, UsdPhysics

    count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
            count += 1
    return count


def bind_material_to_geometries(stage, root_prim, material_path) -> int:
    """Bind directly to imported geometry prims.

    Isaac's URDF importer does not consistently expose CollisionAPI through
    HasAPI() on mesh descendants.  A direct strong physics binding keeps the
    fingertip contact material independent of that importer detail.
    """

    from omni.physx.scripts import physicsUtils
    from pxr import Usd, UsdGeom

    count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Gprim):
            physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
            relationship = prim.GetRelationship("material:binding:physics")
            if relationship.IsValid():
                relationship.SetMetadata("bindMaterialAs", "strongerThanDescendants")
            count += 1
    return count


def bind_material_strongly(stage, prim, material_path) -> None:
    from omni.physx.scripts import physicsUtils

    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    relationship = prim.GetRelationship("material:binding:physics")
    if relationship.IsValid():
        relationship.SetMetadata("bindMaterialAs", "strongerThanDescendants")


def bind_loaded_finger_colliders(stage, robot_model_root: str, material_path) -> int:
    """Bind after URDF composition so imported collision meshes are present."""

    from omni.physx.scripts import physicsUtils
    from pxr import Usd, UsdPhysics

    count = 0
    for side in ("LH", "RH"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            link = stage.GetPrimAtPath(f"{robot_model_root}/{side}_{finger}_distal")
            if not link.IsValid():
                continue
            for prim in Usd.PrimRange(link):
                path = str(prim.GetPath()).lower()
                is_collision_prim = prim.HasAPI(UsdPhysics.CollisionAPI) or "/collisions" in path
                if not is_collision_prim:
                    continue
                physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
                relationship = prim.GetRelationship("material:binding:physics")
                if relationship.IsValid():
                    relationship.SetMetadata("bindMaterialAs", "strongerThanDescendants")
                count += 1
    return count


def configure_torsional_friction(root_prim, radius: float, minimum_radius: float) -> int:
    """Apply soft-contact torsional friction to existing collision shapes."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    count = 0
    for prim in Usd.PrimRange(root_prim):
        path = str(prim.GetPath()).lower()
        if not prim.HasAPI(UsdPhysics.CollisionAPI) and "/collisions" not in path:
            continue
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateTorsionalPatchRadiusAttr(float(radius))
        collision.CreateMinTorsionalPatchRadiusAttr(float(minimum_radius))
        count += 1
    return count


def make_functional_socket_body(
    stage,
    root_path: str,
    socket_extents,
    plug_extents,
    mass_kg: float | None,
    clearance_m: float = 0.0015,
    channel_depth_m: float = 0.022,
    opening_axis_sign: float = -1.0,
) -> None:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    outer_x = float(socket_extents[0]) * 0.5
    outer_y = float(socket_extents[1]) * 0.5
    front_z = math.copysign(float(socket_extents[2]) * 0.5, opening_axis_sign)
    aperture_x = float(plug_extents[0]) * 0.5 + clearance_m
    aperture_y = float(plug_extents[1]) * 0.5 + clearance_m
    if aperture_x >= outer_x or aperture_y >= outer_y:
        raise ValueError("Functional socket aperture does not fit inside socket outer bounds")

    channel_center_z = front_z - math.copysign(channel_depth_m * 0.5, opening_axis_sign)
    boxes = {
        "left_wall": (-(outer_x + aperture_x) * 0.5, 0.0, channel_center_z,
                      outer_x - aperture_x, outer_y * 2.0, channel_depth_m),
        "right_wall": ((outer_x + aperture_x) * 0.5, 0.0, channel_center_z,
                       outer_x - aperture_x, outer_y * 2.0, channel_depth_m),
        "lower_wall": (0.0, -(outer_y + aperture_y) * 0.5, channel_center_z,
                       aperture_x * 2.0, outer_y - aperture_y, channel_depth_m),
        "upper_wall": (0.0, (outer_y + aperture_y) * 0.5, channel_center_z,
                       aperture_x * 2.0, outer_y - aperture_y, channel_depth_m),
    }
    channel_inner_z = front_z - math.copysign(channel_depth_m, opening_axis_sign)
    back_outer_z = -front_z
    back_depth = abs(back_outer_z - channel_inner_z)
    if back_depth > 1.0e-6:
        boxes["back_wall"] = (
            0.0,
            0.0,
            0.5 * (channel_inner_z + back_outer_z),
            outer_x * 2.0,
            outer_y * 2.0,
            back_depth,
        )
    for name, (x, y, z, size_x, size_y, size_z) in boxes.items():
        path = f"{root_path}/functional_collision/{name}"
        add_box(stage, path, (x, y, z), (size_x, size_y, size_z), (0.15, 0.55, 0.95))
        prim = stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).MakeInvisible()

    root_prim = stage.GetPrimAtPath(root_path)
    if mass_kg is not None:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(float(mass_kg))
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_body.CreateEnableCCDAttr(True)
        physx_body.CreateSolverPositionIterationCountAttr(16)
        physx_body.CreateSolverVelocityIterationCountAttr(4)
        physx_body.CreateMaxDepenetrationVelocityAttr(1.0)


def set_xform(prim, translate=None, rotate_xyz=None, scale=None) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_xyz is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def add_box(stage, prim_path: str, center, size, color) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_xform(cube.GetPrim(), translate=center, scale=size)


def add_camera(stage, prim_path: str, translate, rotate_xyz, fov_mm: float = 20.0) -> None:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.CreateFocalLengthAttr(fov_mm)
    camera.CreateHorizontalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
    set_xform(camera.GetPrim(), translate=translate, rotate_xyz=rotate_xyz)


def normalize_quaternion_xyzw(quaternion):
    import numpy as np

    array = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize zero quaternion: {quaternion}")
    return array / norm


def wxyz_to_xyzw(quaternion):
    w, x, y, z = quaternion
    return (x, y, z, w)


def quaternion_xyzw_to_matrix(rotation_xyzw):
    import numpy as np

    x, y, z, w = normalize_quaternion_xyzw(rotation_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_xyz_deg_to_matrix(rotation_xyz_deg):
    import numpy as np

    rx, ry, rz = np.radians(np.asarray(rotation_xyz_deg, dtype=float))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotate_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotate_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotate_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotate_z @ rotate_y @ rotate_x


def mesh_resting_center_z(stl_path: Path, scale: float, rotation_matrix, support_z: float) -> float:
    import numpy as np

    vertices, _ = load_stl_mesh(stl_path, scale=scale)
    rotated_vertices = vertices @ np.asarray(rotation_matrix, dtype=float).T
    return float(support_z - rotated_vertices[:, 2].min())


def ros_camera_to_usd_xyzw(rotation_xyzw):
    """Convert ROS optical axes (+Z forward, +Y down) to USD camera axes (-Z forward, +Y up)."""
    x, y, z, w = normalize_quaternion_xyzw(rotation_xyzw)
    # q_usd = q_ros * q_x_180, where q_x_180 = [1, 0, 0, 0] in xyzw order.
    return normalize_quaternion_xyzw((w, z, -y, -x))


def set_pose_xform(prim, translate, rotation_xyzw) -> None:
    from pxr import Gf, UsdGeom

    x, y, z, w = normalize_quaternion_xyzw(rotation_xyzw)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def add_line_segments(stage, prim_path: str, points, indices, color, width: float = 0.003) -> None:
    from pxr import Gf, UsdGeom

    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve_points = []
    for start, end in indices:
        curve_points.append(Gf.Vec3f(*points[start]))
        curve_points.append(Gf.Vec3f(*points[end]))
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([2] * len(indices))
    curves.CreatePointsAttr(curve_points)
    curves.CreateWidthsAttr([width] * len(curve_points))
    curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def add_pose_camera_rig(
    stage,
    camera_path: str,
    translation_m,
    rotation_xyzw,
    color: tuple[float, float, float],
    fovy_deg: float,
    convention: str = "ros",
    visualization_depth_m: float = 0.70,
    visualize_frustum: bool = True,
) -> str:
    from pxr import Gf, UsdGeom

    if convention == "ros":
        usd_rotation_xyzw = ros_camera_to_usd_xyzw(rotation_xyzw)
    elif convention in ("usd", "opengl"):
        usd_rotation_xyzw = normalize_quaternion_xyzw(rotation_xyzw)
    else:
        raise ValueError(f"Unsupported camera convention: {convention}")

    camera = UsdGeom.Camera.Define(stage, camera_path)
    vertical_aperture_mm = 18.0
    focal_length_mm = vertical_aperture_mm / (2.0 * math.tan(math.radians(fovy_deg) * 0.5))
    camera.CreateFocalLengthAttr(focal_length_mm)
    camera.CreateHorizontalApertureAttr(24.0)
    camera.CreateVerticalApertureAttr(vertical_aperture_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
    set_pose_xform(camera.GetPrim(), translation_m, usd_rotation_xyzw)

    # The visible housing is a sibling of the Camera prim. Keeping all housing
    # geometry behind the optical center prevents the camera from filming itself.
    visual_path = f"{camera_path}_visual"
    visual_root = UsdGeom.Xform.Define(stage, visual_path)
    set_pose_xform(visual_root.GetPrim(), translation_m, usd_rotation_xyzw)

    body = UsdGeom.Cube.Define(stage, f"{visual_path}/body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_xform(body.GetPrim(), translate=(0.0, 0.0, 0.014), scale=(0.026, 0.018, 0.014))

    lens = UsdGeom.Cube.Define(stage, f"{visual_path}/lens")
    lens.CreateSizeAttr(1.0)
    lens.CreateDisplayColorAttr([Gf.Vec3f(0.02, 0.02, 0.02)])
    set_xform(lens.GetPrim(), translate=(0.0, 0.0, 0.004), scale=(0.014, 0.010, 0.004))

    if visualize_frustum:
        depth = float(visualization_depth_m)
        half_h = depth * math.tan(math.radians(fovy_deg) * 0.5)
        half_w = half_h * (24.0 / 18.0)
        frustum_points = [
            (0.0, 0.0, 0.0),
            (-half_w, -half_h, -depth),
            (half_w, -half_h, -depth),
            (half_w, half_h, -depth),
            (-half_w, half_h, -depth),
        ]
        add_line_segments(
            stage,
            f"{visual_path}/frustum",
            frustum_points,
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)],
            color,
            width=0.002,
        )
        add_line_segments(
            stage,
            f"{visual_path}/optical_axis",
            [(0.0, 0.0, 0.0), (0.0, 0.0, -depth)],
            [(0, 1)],
            (1.0, 0.1, 0.1),
            width=0.004,
        )
    return camera_path


def import_robot(robot_cfg: dict):
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0
    # PhysX mimic constraints are path-dependent for the O6 high-ratio finger
    # couplings. Import them as regular DOFs and command the URDF ratios below.
    import_config.parse_mimic = False
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(ROOT / robot_cfg["robot"]["preferred_urdf_for_first_pass"]),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError("URDFParseAndImportFile failed")
    return str(prim_path)


def robot_model_root_from_articulation(articulation_root_path: str) -> str:
    path = articulation_root_path.rstrip("/")
    if path.endswith("/root_joint"):
        return path[: -len("/root_joint")]
    return "/".join(path.split("/")[:-1]) or path


def startup_joint_names(robot_cfg: dict) -> list[str]:
    return (
        robot_cfg["left_arm"]["joints"]
        + robot_cfg["left_o6_hand"]["active_driver_joints"]
        + robot_cfg["right_arm"]["joints"]
        + robot_cfg["right_o6_hand"]["active_driver_joints"]
    )


def coupled_hand_joint_names(side: str) -> list[str]:
    prefix = "LH" if side == "left" else "RH"
    return [
        f"{prefix}_thumb_ip",
        f"{prefix}_index_dip",
        f"{prefix}_middle_dip",
        f"{prefix}_ring_dip",
        f"{prefix}_pinky_dip",
    ]


def coupled_hand_positions(side: str, active_positions) -> list[float]:
    values = [float(value) for value in active_positions]
    thumb_multiplier = 2.29 if side == "left" else 1.86
    return [values[1] * thumb_multiplier, *(value * 0.89 for value in values[2:6])]


def configure_robot_drives(stage, robot_model_root: str, robot_cfg: dict) -> None:
    """Apply damped position drives to the controlled arm and hand joints."""

    from pxr import UsdPhysics

    drive_cfg = robot_cfg["isaac_drive"]
    groups = (
        (
            robot_cfg["left_arm"]["joints"] + robot_cfg["right_arm"]["joints"],
            float(drive_cfg["arm_stiffness"]),
            float(drive_cfg["arm_damping"]),
            drive_cfg.get("arm_max_force"),
            "arm",
        ),
        (
            robot_cfg["left_o6_hand"]["active_driver_joints"]
            + coupled_hand_joint_names("left"),
            float(drive_cfg["hand_stiffness"]),
            float(drive_cfg["hand_damping"]),
            drive_cfg.get("hand_max_force"),
            "left hand",
        ),
        (
            robot_cfg["right_o6_hand"]["active_driver_joints"]
            + coupled_hand_joint_names("right"),
            float(drive_cfg.get("right_hand_stiffness", drive_cfg["hand_stiffness"])),
            float(drive_cfg.get("right_hand_damping", drive_cfg["hand_damping"])),
            drive_cfg.get("right_hand_max_force"),
            "right hand",
        ),
    )
    for joint_names, stiffness, damping, max_force, label in groups:
        for joint_name in joint_names:
            prim = stage.GetPrimAtPath(f"{robot_model_root}/joints/{joint_name}")
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not prim.IsValid() or not drive:
                raise RuntimeError(f"Missing angular drive for {joint_name}")
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping)
            if max_force is not None:
                drive.GetMaxForceAttr().Set(float(max_force))
        print(
            f"Configured {label} drives: stiffness={stiffness:.1f}, "
            f"damping={damping:.1f}, max_force={max_force}",
            flush=True,
        )

    for joint_name, override in drive_cfg.get(
        "right_hand_joint_overrides", {}
    ).items():
        prim = stage.GetPrimAtPath(f"{robot_model_root}/joints/{joint_name}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not prim.IsValid() or not drive:
            raise RuntimeError(f"Missing angular drive override target {joint_name}")
        stiffness = float(override["stiffness"])
        damping = float(override["damping"])
        max_force = override.get("max_force")
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        if max_force is not None:
            drive.GetMaxForceAttr().Set(float(max_force))
        print(
            f"Configured {joint_name} drive override: stiffness={stiffness:.1f}, "
            f"damping={damping:.1f}, max_force={max_force}",
            flush=True,
        )

    print(
        "Configured O6 coupled joints as explicit critically damped position drives",
        flush=True,
    )


def run_startup_motion(
    app, articulation_root_path: str, robot_cfg: dict, realtime: bool = True
) -> None:
    import numpy as np
    import omni.timeline
    from isaacsim.core.prims import Articulation

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    art = Articulation(articulation_root_path)
    art.initialize()

    joint_names = startup_joint_names(robot_cfg)
    indices = [art.get_dof_index(name) for name in joint_names]
    joint_indices = np.asarray(indices)
    current = np.asarray(art.get_joint_positions(joint_indices=joint_indices), dtype=float).reshape(-1)
    motion_cfg = robot_cfg["startup_motion"]
    control_hz = float(motion_cfg.get("control_hz", 60.0))

    for phase_name in ("spread", "home"):
        phase = motion_cfg[phase_name]
        target = np.asarray(phase["joint_positions"], dtype=float)
        if target.shape != current.shape:
            raise ValueError(
                f"startup_motion.{phase_name} has {target.size} values; expected {current.size} for {joint_names}"
            )
        steps = max(1, int(round(float(phase["duration_hint"]) * control_hz)))
        start = current.copy()
        for step in range(1, steps + 1):
            linear_alpha = step / steps
            alpha = linear_alpha * linear_alpha * (3.0 - 2.0 * linear_alpha)
            positions = start + alpha * (target - start)
            art.set_joint_positions(np.asarray([positions], dtype=np.float32), joint_indices=joint_indices)
            app.update()
            if realtime:
                time.sleep(1.0 / control_hz)
        current = target

    art.set_joint_position_targets(
        np.asarray([current], dtype=np.float32), joint_indices=joint_indices
    )
    timeline.pause()


def build_scene(
    app,
    robot_cfg: dict,
    task_cfg: dict,
    camera_cfg: dict,
    scale: float,
    startup_realtime: bool = True,
):
    import numpy as np
    from omni.physx.scripts import physicsUtils, utils as physx_utils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr(float(task_cfg["simulation"]["physics_hz"]))
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableEnhancedDeterminismAttr(True)

    UsdGeom.Scope.Define(stage, "/World/PhysicsMaterials")
    table_material_path = Sdf.Path("/World/PhysicsMaterials/Table")
    finger_material_path = Sdf.Path("/World/PhysicsMaterials/FingerGrasp")
    object_material_path = Sdf.Path("/World/PhysicsMaterials/Object")
    socket_material_path = Sdf.Path("/World/PhysicsMaterials/Socket")

    table = task_cfg["scene"]["table"]
    physx_utils.addRigidBodyMaterial(
        stage,
        table_material_path,
        staticFriction=float(table["static_friction"]),
        dynamicFriction=float(table["dynamic_friction"]),
        restitution=float(table["restitution"]),
    )
    physx_utils.addRigidBodyMaterial(
        stage,
        finger_material_path,
        staticFriction=float(
            task_cfg["simulation"].get("fingertip_static_friction", 20.0)
        ),
        dynamicFriction=float(
            task_cfg["simulation"].get("fingertip_dynamic_friction", 15.0)
        ),
        restitution=0.0,
    )
    finger_material_api = PhysxSchema.PhysxMaterialAPI(
        stage.GetPrimAtPath(finger_material_path)
    )
    finger_material_api.CreateFrictionCombineModeAttr("max")
    physx_utils.addRigidBodyMaterial(
        stage,
        object_material_path,
        staticFriction=float(task_cfg["simulation"].get("object_static_friction", 1.0)),
        dynamicFriction=float(task_cfg["simulation"].get("object_dynamic_friction", 0.8)),
        restitution=float(task_cfg["simulation"].get("object_restitution", 0.0)),
    )
    object_material_api = PhysxSchema.PhysxMaterialAPI(
        stage.GetPrimAtPath(object_material_path)
    )
    object_material_api.CreateFrictionCombineModeAttr(
        str(task_cfg["simulation"].get("object_friction_combine_mode", "average"))
    )
    # The insert needs a grabbable surface against the fingers and a slippery
    # one against the socket channel, so the channel gets its own material. Which
    # coefficient a pair ends up with is decided by the combine mode with the
    # higher PhysX priority (max > multiply > min > average), so this material is
    # only reachable when the object material does not force max.
    physx_utils.addRigidBodyMaterial(
        stage,
        socket_material_path,
        staticFriction=float(
            task_cfg["simulation"].get("socket_static_friction", 0.05)
        ),
        dynamicFriction=float(
            task_cfg["simulation"].get("socket_dynamic_friction", 0.04)
        ),
        restitution=float(task_cfg["simulation"].get("object_restitution", 0.0)),
    )
    PhysxSchema.PhysxMaterialAPI(
        stage.GetPrimAtPath(socket_material_path)
    ).CreateFrictionCombineModeAttr("average")
    table_color = tuple(float(value) for value in table.get("rgba", [0.50, 0.34, 0.22])[:3])
    add_box(stage, "/World/Table", table["center_xyz"], table["size_xyz"], table_color)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Table"))
    physicsUtils.add_physics_material_to_prim(
        stage, stage.GetPrimAtPath("/World/Table"), table_material_path
    )
    articulation_root = import_robot(robot_cfg)
    robot_model_root = robot_model_root_from_articulation(articulation_root)
    robot_root = stage.GetPrimAtPath(robot_model_root)
    # Let the asynchronous URDF reference composition publish its collision
    # descendants before PhysX creates shapes. Material bindings authored after
    # the first simulation run do not reliably refresh already-created shapes.
    for _ in range(4):
        app.update()
    configure_robot_drives(stage, robot_model_root, robot_cfg)
    set_xform(robot_root, translate=task_cfg["scene"]["robot_base_world_xyz"])
    bind_material_strongly(stage, robot_root, finger_material_path)
    pad_radius = float(task_cfg["simulation"].get("right_fingertip_pad_radius_m", 0.0))
    if pad_radius > 0.0:
        pad_shape = str(
            task_cfg["simulation"].get("right_fingertip_pad_shape", "sphere")
        ).lower()
        pad_cube_size = tuple(
            float(value)
            for value in task_cfg["simulation"].get(
                "right_fingertip_pad_cube_size_m", [0.010, 0.014, 0.014]
            )
        )
        configured_pad_radii = task_cfg["simulation"].get(
            "right_fingertip_pad_radii_m", {}
        )
        enabled_pad_links = set(
            task_cfg["simulation"].get(
                "right_fingertip_pad_links",
                ["RH_thumb_distal", "RH_index_distal", "RH_middle_distal"],
            )
        )
        pad_centers = {
            "RH_thumb_distal": (-0.0020, -0.0017, 0.0458),
            "RH_index_distal": (0.0173, 0.0017, 0.0308),
            "RH_middle_distal": (0.0173, 0.0017, 0.0308),
        }
        for link_name, center in pad_centers.items():
            if link_name not in enabled_pad_links:
                continue
            link_pad_radius = float(configured_pad_radii.get(link_name, pad_radius))
            link_prim = stage.GetPrimAtPath(f"{robot_model_root}/{link_name}")
            # Use one well-defined convex contact at each fingertip. The
            # imported concave STL remains visible but no longer competes with
            # the pad in PhysX contact generation.
            for descendant in Usd.PrimRange(link_prim):
                if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(False)
            pad_path = f"{robot_model_root}/{link_name}/grasp_pad"
            if pad_shape == "cube":
                pad = UsdGeom.Cube.Define(stage, pad_path)
                pad.CreateSizeAttr(1.0)
                set_xform(pad.GetPrim(), translate=center, scale=pad_cube_size)
            else:
                pad = UsdGeom.Sphere.Define(stage, pad_path)
                pad.CreateRadiusAttr(link_pad_radius)
                set_xform(pad.GetPrim(), translate=center)
            pad.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
            pad_physx = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
            # Keep speculative contact narrow. A 1 mm offset on each side
            # stopped the fingers before the pads touched the 30 mm block.
            pad_physx.CreateContactOffsetAttr(0.0001)
            pad_physx.CreateRestOffsetAttr(-0.0003)
            pad_physx.CreateTorsionalPatchRadiusAttr(0.004)
            pad_physx.CreateMinTorsionalPatchRadiusAttr(0.002)
            bind_material_strongly(stage, pad.GetPrim(), finger_material_path)
        print(
            f"Configured physical fingertip pads {sorted(enabled_pad_links)}: "
            f"shape={pad_shape} "
            "radii_mm="
            f"{dict((name, float(configured_pad_radii.get(name, pad_radius)) * 1000.0) for name in sorted(enabled_pad_links))}",
            flush=True,
        )
    finger_link_count = 0
    finger_geometry_count = 0
    for link_name in (
        "LH_thumb_distal",
        "LH_index_distal",
        "LH_middle_distal",
        "RH_thumb_distal",
        "RH_index_distal",
        "RH_middle_distal",
        "RH_ring_distal",
    ):
        link_prim = stage.GetPrimAtPath(f"{robot_model_root}/{link_name}")
        if link_prim.IsValid():
            bind_material_strongly(stage, link_prim, finger_material_path)
            finger_geometry_count += bind_material_to_geometries(
                stage, link_prim, finger_material_path
            )
            finger_link_count += 1
    robot_collider_count = bind_material_to_colliders(
        stage, robot_root, finger_material_path
    )
    prephysics_finger_colliders = bind_loaded_finger_colliders(
        stage, robot_model_root, finger_material_path
    )
    torsional_radius = float(
        task_cfg["simulation"].get("contact_torsional_patch_radius_m", 0.0)
    )
    min_torsional_radius = float(
        task_cfg["simulation"].get("contact_min_torsional_patch_radius_m", 0.0)
    )
    if torsional_radius > 0.0:
        for link_name in ("RH_thumb_distal", "RH_index_distal"):
            configure_torsional_friction(
                stage.GetPrimAtPath(f"{robot_model_root}/{link_name}"),
                torsional_radius,
                min_torsional_radius,
            )
    print(
        f"Pre-physics O6 finger collision materials: colliders={prephysics_finger_colliders}",
        flush=True,
    )
    if prephysics_finger_colliders == 0:
        raise RuntimeError("URDF finger collision meshes were not composed before physics startup")

    table_top_z = float(table["center_xyz"][2]) + float(table["size_xyz"][2]) * 0.5
    socket_asset = task_cfg["assets"]["socket_box"]
    plug_asset = task_cfg["assets"]["rj45_plug"]
    socket_path = ROOT / socket_asset["visual_mesh"]
    plug_path = ROOT / plug_asset["visual_mesh"]
    socket_rotation = wxyz_to_xyzw(task_cfg["scene"]["socket_box_initial_quat_wxyz"])
    plug_rotation = wxyz_to_xyzw(task_cfg["scene"]["rj45_plug_initial_quat_wxyz"])
    socket_rotate_xyz = task_cfg["scene"].get("socket_box_initial_rotate_xyz_deg")
    plug_rotate_xyz = task_cfg["scene"].get("rj45_plug_initial_rotate_xyz_deg")
    socket_rotation_matrix = (
        euler_xyz_deg_to_matrix(socket_rotate_xyz)
        if socket_rotate_xyz is not None
        else quaternion_xyzw_to_matrix(socket_rotation)
    )
    plug_rotation_matrix = (
        euler_xyz_deg_to_matrix(plug_rotate_xyz)
        if plug_rotate_xyz is not None
        else quaternion_xyzw_to_matrix(plug_rotation)
    )
    socket_config_xyz = task_cfg["scene"]["socket_box_initial_xyz"]
    plug_config_xyz = task_cfg["scene"]["rj45_plug_initial_xyz"]
    surface_clearance = float(task_cfg["scene"].get("object_surface_clearance_m", 0.002))
    support_z = table_top_z + surface_clearance
    use_usd_assets = (
        str(socket_asset.get("format", "stl")).lower() == "usd_reference"
        and str(plug_asset.get("format", "stl")).lower() == "usd_reference"
    )
    if use_usd_assets:
        socket_xyz = (
            float(socket_config_xyz[0]),
            float(socket_config_xyz[1]),
            support_z,
        )
        plug_xyz = (
            float(plug_config_xyz[0]),
            float(plug_config_xyz[1]),
            support_z,
        )
    else:
        socket_xyz = (
            float(socket_config_xyz[0]),
            float(socket_config_xyz[1]),
            mesh_resting_center_z(socket_path, scale, socket_rotation_matrix, support_z),
        )
        plug_xyz = (
            float(plug_config_xyz[0]),
            float(plug_config_xyz[1]),
            mesh_resting_center_z(plug_path, scale, plug_rotation_matrix, support_z),
        )

    socket_root = UsdGeom.Xform.Define(stage, "/World/RJ45Socket")
    if socket_rotate_xyz is not None:
        set_xform(socket_root.GetPrim(), translate=socket_xyz, rotate_xyz=socket_rotate_xyz)
    else:
        set_pose_xform(socket_root.GetPrim(), socket_xyz, socket_rotation)
    socket_fixed = bool(socket_asset.get("fixed", False))
    if use_usd_assets:
        socket_root.GetPrim().GetReferences().AddReference(str(socket_path.resolve()))
    else:
        define_mesh(
            stage,
            "/World/RJ45Socket/visual",
            socket_path,
            (0.05, 0.07, 0.11),
            scale,
        )
        socket_vertices, _ = load_stl_mesh(socket_path, scale=scale)
        plug_vertices, _ = load_stl_mesh(plug_path, scale=scale)
        make_functional_socket_body(
            stage,
            "/World/RJ45Socket",
            socket_vertices.max(axis=0) - socket_vertices.min(axis=0),
            plug_vertices.max(axis=0) - plug_vertices.min(axis=0),
            None if socket_fixed else float(socket_asset["mass_kg"]),
        )

    plug_root = UsdGeom.Xform.Define(stage, "/World/RJ45Plug")
    if plug_rotate_xyz is not None:
        set_xform(plug_root.GetPrim(), translate=plug_xyz, rotate_xyz=plug_rotate_xyz)
    else:
        set_pose_xform(plug_root.GetPrim(), plug_xyz, plug_rotation)
    if use_usd_assets:
        plug_root.GetPrim().GetReferences().AddReference(str(plug_path.resolve()))
        plug_body = UsdPhysics.RigidBodyAPI.Get(stage, "/World/RJ45Plug")
        if not plug_body:
            raise RuntimeError("Referenced insert is missing PhysicsRigidBodyAPI")
        plug_body.GetKinematicEnabledAttr().Set(
            bool(plug_asset.get("start_kinematic", False))
        )
        # A reference authored onto an existing Xform does not reliably carry
        # the referenced root MassAPI into the PhysX actor. Author the task
        # mass explicitly on the composed actor so PhysX does not fall back to
        # density-derived mass (about 34 g for the slender-pin box).
        UsdPhysics.MassAPI.Apply(plug_root.GetPrim()).CreateMassAttr(
            float(plug_asset["mass_kg"])
        )
        plug_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(plug_root.GetPrim())
        plug_physx.CreateEnableCCDAttr(
            not bool(plug_asset.get("start_kinematic", False))
        )
        plug_physx.CreateSolverPositionIterationCountAttr(32)
        plug_physx.CreateSolverVelocityIterationCountAttr(16)
        plug_physx.CreateMaxDepenetrationVelocityAttr(
            float(task_cfg["simulation"].get("object_max_depenetration_velocity_m", 0.5))
        )
        plug_physx.CreateLinearDampingAttr(
            float(task_cfg["simulation"].get("object_linear_damping", 0.0))
        )
        plug_physx.CreateAngularDampingAttr(
            float(task_cfg["simulation"].get("object_angular_damping", 0.0))
        )
    else:
        define_mesh(
            stage,
            "/World/RJ45Plug/visual",
            plug_path,
            (0.02, 0.02, 0.02),
            scale,
        )
        make_dynamic_mesh_body(
            stage,
            "/World/RJ45Plug",
            "/World/RJ45Plug/visual",
            plug_asset["mass_kg"],
            start_kinematic=bool(plug_asset.get("start_kinematic", False)),
        )
    bind_material_strongly(stage, socket_root.GetPrim(), socket_material_path)
    bind_material_strongly(stage, plug_root.GetPrim(), object_material_path)
    socket_collider_count = bind_material_to_colliders(
        stage, socket_root.GetPrim(), socket_material_path
    )
    plug_collider_count = bind_material_to_colliders(
        stage, plug_root.GetPrim(), object_material_path
    )
    if torsional_radius > 0.0:
        configure_torsional_friction(
            plug_root.GetPrim(), torsional_radius, min_torsional_radius
        )
    print(
        "Bound grasp material to collision shapes: "
        f"robot={robot_collider_count}, finger_links={finger_link_count}, "
        f"finger_geometries={finger_geometry_count}, "
        f"socket={socket_collider_count}, "
        f"plug={plug_collider_count}",
        flush=True,
    )

    print(
        f"Task assets placed {surface_clearance * 1000.0:.1f} mm above table z={table_top_z:.5f}: "
        f"socket={socket_xyz}, plug={plug_xyz}",
        flush=True,
    )
    print(
        f"RJ45 socket physics mode: {'static' if socket_fixed else 'dynamic'}",
        flush=True,
    )

    # Create the legs after the robot and task rigid bodies so their PhysX actor
    # registration order remains identical to the validated tabletop-only scene.
    leg_size = np.asarray(table.get("leg_size_xyz", [0.06, 0.06, 1.0]), dtype=float)
    leg_inset = np.asarray(table.get("leg_inset_xy_m", [0.05, 0.05]), dtype=float)
    table_center = np.asarray(table["center_xyz"], dtype=float)
    table_size = np.asarray(table["size_xyz"], dtype=float)
    leg_offsets = 0.5 * table_size[:2] - leg_inset - 0.5 * leg_size[:2]
    leg_center_z = table_center[2] - 0.5 * table_size[2] - 0.5 * leg_size[2]
    for index, (x_sign, y_sign) in enumerate(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)), start=1
    ):
        leg_path = f"/World/TableLeg{index}"
        leg_center = (
            table_center[0] + x_sign * leg_offsets[0],
            table_center[1] + y_sign * leg_offsets[1],
            leg_center_z,
        )
        add_box(stage, leg_path, leg_center, leg_size, table_color)
        leg_prim = stage.GetPrimAtPath(leg_path)
        UsdPhysics.CollisionAPI.Apply(leg_prim)
        physicsUtils.add_physics_material_to_prim(stage, leg_prim, table_material_path)

    cameras = camera_cfg["cameras"]
    chest = cameras["chest"]
    left_wrist = cameras["left_wrist"]
    right_wrist = cameras["right_wrist"]
    camera_paths = [
        add_pose_camera_rig(
            stage,
            f"{robot_model_root}/base_link/chest_camera",
            translation_m=chest["translation_m"],
            rotation_xyzw=chest["rotation_xyzw"],
            convention=chest["convention"],
            fovy_deg=chest["fovy_deg"],
            visualization_depth_m=chest.get("visualization_depth_m", 1.0),
            visualize_frustum=chest.get("visualize_frustum", True),
            color=(0.95, 0.55, 0.10),
        ),
        add_pose_camera_rig(
            stage,
            f"{robot_model_root}/LH_hand_base_link/left_wrist_camera",
            translation_m=left_wrist["translation_m"],
            rotation_xyzw=left_wrist["rotation_xyzw"],
            convention=left_wrist["convention"],
            fovy_deg=left_wrist["fovy_deg"],
            visualization_depth_m=left_wrist.get("visualization_depth_m", 0.70),
            visualize_frustum=left_wrist.get("visualize_frustum", True),
            color=(0.1, 0.35, 0.95),
        ),
        add_pose_camera_rig(
            stage,
            f"{robot_model_root}/RH_hand_base_link/right_wrist_camera",
            translation_m=right_wrist["translation_m"],
            rotation_xyzw=right_wrist["rotation_xyzw"],
            convention=right_wrist["convention"],
            fovy_deg=right_wrist["fovy_deg"],
            visualization_depth_m=right_wrist.get("visualization_depth_m", 0.70),
            visualize_frustum=right_wrist.get("visualize_frustum", True),
            color=(0.1, 0.75, 0.55),
        ),
    ]
    add_camera(stage, "/World/OverviewCamera", (0.72, -0.62, 1.55), (62.0, 0.0, 42.0), 24.0)

    print("Isaac sensor camera prims:", flush=True)
    for camera_path in camera_paths:
        print(f"  {camera_path}", flush=True)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(700.0)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(1600.0)
    set_xform(key.GetPrim(), rotate_xyz=(-45.0, 0.0, 35.0))

    try:
        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera("/World/OverviewCamera")
    except Exception:
        pass

    run_startup_motion(
        app, articulation_root, robot_cfg, realtime=startup_realtime
    )
    late_finger_colliders = bind_loaded_finger_colliders(
        stage, robot_model_root, finger_material_path
    )
    print(
        f"Late-bound O6 finger collision materials: colliders={late_finger_colliders}",
        flush=True,
    )
    if late_finger_colliders == 0:
        raise RuntimeError("URDF finger collision meshes were not available for material binding")
    object_points = [socket_xyz, plug_xyz]
    return camera_paths, object_points, articulation_root, robot_model_root


def calibrate_o6_pinches(
    app,
    articulation_root_path: str,
    robot_model_root: str,
    robot_cfg: dict,
    task_cfg: dict | None = None,
) -> None:
    """Search pinch presets against the meshes actually imported into Isaac."""

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial import cKDTree

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    articulation = Articulation(articulation_root_path)
    articulation.initialize()
    stage = omni.usd.get_context().get_stage()

    def imported_points(prefix: str, finger: str) -> np.ndarray:
        side = "left" if prefix == "LH" else "right"
        mesh_path = ROOT / f"robot_description/S4/meshes/o6/{side}/meshes/{finger}_distal.STL"
        points = np.asarray(__import__("trimesh").load_mesh(mesh_path).vertices, dtype=float)
        return points[points[:, 2] >= float(points[:, 2].max()) - 0.012]

    def world_points(link_path: str, local_points: np.ndarray) -> np.ndarray:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(link_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        return np.asarray(
            [matrix.Transform(Gf.Vec3d(*point)) for point in local_points], dtype=float
        )

    # The pinch is treated as a two-state gripper: the thumb is the fixed jaw and
    # only the index closes.  Targets are the measured object widths minus 0.5 mm
    # of preload (socket 29.2 mm, RJ45 plug 11.34 mm).
    default_targets = {
        "left": 0.02870,
        "right": 0.01084,
    }
    if task_cfg is not None:
        configured_right_gap = task_cfg.get("expert_auto_ik", {}).get(
            "right_pinch_target_gap_m"
        )
        if configured_right_gap is not None:
            default_targets["right"] = float(configured_right_gap)
    for side, prefix, target_gap_m in (
        ("left", "LH", 0.02870),
        ("right", "RH", default_targets["right"]),
    ):
        hand_cfg = robot_cfg[f"{side}_o6_hand"]
        hand_indices = np.asarray(
            [articulation.get_dof_index(name) for name in hand_cfg["active_driver_joints"]]
        )
        coupled_indices = np.asarray(
            [articulation.get_dof_index(name) for name in coupled_hand_joint_names(side)]
        )
        thumb_local = imported_points(prefix, "thumb")
        index_local = imported_points(prefix, "index")
        best: dict[str, object] = {}
        fixed_preset = hand_cfg["pinch_preset"]
        fixed_yaw = float(fixed_preset[f"{prefix}_thumb_cmc_yaw"])
        fixed_pitch = float(fixed_preset[f"{prefix}_thumb_cmc_pitch"])
        thumb_pad = stage.GetPrimAtPath(
            f"{robot_model_root}/{prefix}_thumb_distal/grasp_pad"
        )
        index_pad = stage.GetPrimAtPath(
            f"{robot_model_root}/{prefix}_index_distal/grasp_pad"
        )
        use_pads = thumb_pad.IsValid() and index_pad.IsValid()
        pad_radius = (
            float(UsdGeom.Sphere(thumb_pad).GetRadiusAttr().Get()) if use_pads else 0.0
        )

        def evaluate(index_pitch: float, keep_points: bool = False) -> tuple[float, float]:
            command = np.asarray(
                [[fixed_yaw, fixed_pitch, float(index_pitch), 0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
            zeros = np.zeros_like(command)
            coupled = np.asarray(
                [coupled_hand_positions(side, command.reshape(-1))], dtype=np.float32
            )
            coupled_zeros = np.zeros_like(coupled)
            articulation.set_joint_position_targets(command, joint_indices=hand_indices)
            articulation.set_joint_velocity_targets(zeros, joint_indices=hand_indices)
            articulation.set_joint_position_targets(coupled, joint_indices=coupled_indices)
            articulation.set_joint_velocity_targets(coupled_zeros, joint_indices=coupled_indices)
            articulation.set_joint_positions(command, joint_indices=hand_indices)
            articulation.set_joint_velocities(zeros, joint_indices=hand_indices)
            articulation.set_joint_positions(coupled, joint_indices=coupled_indices)
            articulation.set_joint_velocities(coupled_zeros, joint_indices=coupled_indices)
            # Fabric exposes the imported visual transform one update after the
            # articulation state changes; the second update makes the geometry
            # and the commanded joint value refer to the same candidate.
            for _ in range(10):
                app.update()
            if use_pads:
                thumb_matrix = UsdGeom.Xformable(thumb_pad).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                index_matrix = UsdGeom.Xformable(index_pad).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                thumb_center = np.asarray(
                    thumb_matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=float
                )
                index_center = np.asarray(
                    index_matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=float
                )
                direction = index_center - thumb_center
                center_gap = float(np.linalg.norm(direction))
                direction /= max(center_gap, 1.0e-9)
                thumb_world = np.asarray([thumb_center + direction * pad_radius])
                index_world = np.asarray([index_center - direction * pad_radius])
                thumb_index = 0
                index_index = 0
                gap = center_gap - 2.0 * pad_radius
            else:
                thumb_world = world_points(
                    f"{robot_model_root}/{prefix}_thumb_distal", thumb_local
                )
                index_world = world_points(
                    f"{robot_model_root}/{prefix}_index_distal", index_local
                )
                distances, indices = cKDTree(index_world).query(thumb_world)
                thumb_index = int(np.argmin(distances))
                index_index = int(indices[thumb_index])
                gap = float(distances[thumb_index])
            if keep_points:
                best.update(
                    gap_m=gap,
                    thumb_point=thumb_world[thumb_index],
                    index_point=index_world[index_index],
                    actual_q=np.asarray(
                        articulation.get_joint_positions(joint_indices=hand_indices), dtype=float
                    ).reshape(-1),
                )
            return (gap - target_gap_m) ** 2, gap

        # The fingertip gap is non-monotonic after the fingers cross. Only the
        # first closing branch represents a usable opposed pinch.
        # Both the authored distal meshes and optional pads cross near 0.9 rad.
        # Searching past that point can select the second, reopening branch and
        # report a numerically correct gap with physically crossed fingers.
        branch_max = 0.88
        coarse = np.linspace(0.0, branch_max, 15)
        coarse_results = [(float(q), *evaluate(float(q))) for q in coarse]
        print(
            f"Isaac pinch scan {side}: "
            + ", ".join(f"{q:.3f}->{gap * 1000.0:.1f}mm" for q, _, gap in coarse_results),
            flush=True,
        )
        coarse_best = min(coarse_results, key=lambda item: item[1])
        coarse_step = float(coarse[1] - coarse[0])
        refine_low = max(0.0, coarse_best[0] - coarse_step)
        refine_high = min(branch_max, coarse_best[0] + coarse_step)
        fine = np.linspace(refine_low, refine_high, 9)
        fine_results = [(float(q), *evaluate(float(q))) for q in fine]
        best_q, objective, _ = min(fine_results, key=lambda item: item[1])
        _, measured_gap = evaluate(best_q, keep_points=True)
        midpoint_world = (
            np.asarray(best["thumb_point"]) + np.asarray(best["index_point"])
        ) * 0.5
        palm_world = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"{robot_model_root}/{prefix}_palm_center")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        thumb_palm = palm_world.GetInverse().Transform(
            Gf.Vec3d(*np.asarray(best["thumb_point"], dtype=float))
        )
        index_palm = palm_world.GetInverse().Transform(
            Gf.Vec3d(*np.asarray(best["index_point"], dtype=float))
        )
        midpoint_palm = palm_world.GetInverse().Transform(Gf.Vec3d(*midpoint_world))
        print(
            f"Isaac pinch calibration {side}: "
            f"geometry={'pads' if use_pads else 'meshes'}, "
            f"q={[round(fixed_yaw, 6), round(fixed_pitch, 6), round(best_q, 6)]}, "
            f"target_gap={target_gap_m * 1000.0:.3f} mm, "
            f"surface_gap={measured_gap * 1000.0:.3f} mm, "
            f"actual_q={[round(float(value), 6) for value in best['actual_q'][:3]]}, "
            f"center_from_palm={[round(float(value), 6) for value in midpoint_palm]}, "
            f"thumb_from_palm={[round(float(value), 6) for value in thumb_palm]}, "
            f"index_from_palm={[round(float(value), 6) for value in index_palm]}, "
            f"objective={objective:.8f}",
            flush=True,
        )
    timeline.pause()


def replay_pickup_episode(
    app,
    episode_path: Path,
    articulation_root_path: str,
    robot_model_root: str,
    robot_cfg: dict,
    task_cfg: dict,
    record_hz: float,
    replay_speed: float,
    realtime: bool,
    max_replay_frames: int | None = None,
    grasp_capture_path: Path | None = None,
    final_capture_path: Path | None = None,
    camera_paths: list[str] | None = None,
    camera_cfg: dict | None = None,
    record_out_dir: Path | None = None,
    record_format: str = "npz",
) -> list[list[float]]:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    from qiling_xvla.control.dual_arm_pinocchio import ArmPinocchioKinematics
    from qiling_xvla.control.posture_safe_ik import (
        solve_posture_safe_pose,
        torso_safe_joint_path,
    )
    from qiling_xvla.data.rj45_insertion_plan import (
        arm_home,
        rotation_to_rot6d,
        solve_with_restarts,
    )

    with np.load(episode_path) as episode:
        required = {
            "phase",
            "debug_left_arm_q",
            "debug_right_arm_q",
            "debug_left_hand_q",
            "debug_right_hand_q",
        }
        missing = sorted(required - set(episode.files))
        if missing:
            raise ValueError(f"Pickup episode is missing replay arrays: {missing}")
        phases = np.asarray(episode["phase"]).astype(str)
        left_arm = np.asarray(episode["debug_left_arm_q"], dtype=float)
        right_arm = np.asarray(episode["debug_right_arm_q"], dtype=float)
        left_hand = np.asarray(episode["debug_left_hand_q"], dtype=float)
        right_hand = np.asarray(episode["debug_right_hand_q"], dtype=float)
        source_observation_state = (
            np.asarray(episode["observation_state"], dtype=np.float32)
            if "observation_state" in episode.files
            else None
        )
        source_action = (
            np.asarray(episode["action"], dtype=np.float32)
            if "action" in episode.files
            else None
        )
        source_expert_mask = (
            np.asarray(episode["expert_mask"], dtype=bool)
            if "expert_mask" in episode.files
            else np.ones(len(phases), dtype=bool)
        )
        source_language_instruction = (
            np.asarray(episode["language_instruction"]).astype(str)
            if "language_instruction" in episode.files
            else None
        )

    nominal_left_arm = np.array(left_arm, copy=True)
    nominal_right_arm = np.array(right_arm, copy=True)

    if max_replay_frames is not None and max_replay_frames > 0:
        limit = min(int(max_replay_frames), len(phases))
        phases = phases[:limit]
        left_arm = left_arm[:limit]
        right_arm = right_arm[:limit]
        left_hand = left_hand[:limit]
        right_hand = right_hand[:limit]
        if source_observation_state is not None:
            source_observation_state = source_observation_state[:limit]
        if source_action is not None:
            source_action = source_action[:limit]
        source_expert_mask = source_expert_mask[:limit]
        if source_language_instruction is not None:
            source_language_instruction = source_language_instruction[:limit]

    frame_count = len(phases)
    if not all(len(array) == frame_count for array in (left_arm, right_arm, left_hand, right_hand)):
        raise ValueError("Pickup replay arrays have different frame counts")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    articulation = Articulation(articulation_root_path)
    articulation.initialize()
    joint_names = startup_joint_names(robot_cfg)
    joint_indices = np.asarray([articulation.get_dof_index(name) for name in joint_names])
    if np.any(joint_indices < 0):
        missing_joints = [name for name, index in zip(joint_names, joint_indices) if index < 0]
        raise ValueError(f"Replay joints not found in Isaac articulation: {missing_joints}")
    coupled_names = coupled_hand_joint_names("left") + coupled_hand_joint_names("right")
    coupled_indices = np.asarray([articulation.get_dof_index(name) for name in coupled_names])
    if np.any(coupled_indices < 0):
        missing_joints = [name for name, index in zip(coupled_names, coupled_indices) if index < 0]
        raise ValueError(f"Coupled O6 joints not found in Isaac articulation: {missing_joints}")

    stage = omni.usd.get_context().get_stage()
    grasp_checked = False
    grasp_success = False

    def world_position(prim_path: str) -> np.ndarray:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        value = matrix.ExtractTranslation()
        return np.asarray([value[0], value[1], value[2]], dtype=float)

    def world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        rotation = quaternion_xyzw_to_matrix(
            [imaginary[0], imaginary[1], imaginary[2], quaternion.GetReal()]
        )
        return (
            np.asarray([translation[0], translation[1], translation[2]], dtype=float),
            rotation,
        )

    def world_direction(prim_path: str, local_direction) -> np.ndarray:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        value = matrix.TransformDir(Gf.Vec3d(*local_direction))
        direction = np.asarray([value[0], value[1], value[2]], dtype=float)
        return direction / np.linalg.norm(direction)

    if record_format not in ("npz", "video"):
        raise ValueError(f"Unsupported recording format: {record_format}")

    camera_sensors: dict[str, object] = {}
    video_writers: dict[str, RgbVideoWriter] = {}
    camera_frame_counts = {"chest": 0, "left_wrist": 0, "right_wrist": 0}
    recorded_camera_frames: dict[str, list[np.ndarray]] = {
        "chest": [],
        "left_wrist": [],
        "right_wrist": [],
    }
    recorded_observation_state: list[np.ndarray] = []
    recorded_target_observation_state: list[np.ndarray] = []
    recorded_action: list[np.ndarray] = []
    recorded_actual_joint_position: list[np.ndarray] = []
    recorded_target_joint_position: list[np.ndarray] = []

    if record_out_dir is not None:
        if camera_paths is None or camera_cfg is None:
            raise ValueError("Three-camera recording requires camera_paths and camera_cfg")
        from isaacsim.sensors.camera import Camera

        camera_names = ("chest", "left_wrist", "right_wrist")
        if len(camera_paths) != len(camera_names):
            raise ValueError(f"Expected three camera paths, got {camera_paths}")
        for index, (name, camera_path) in enumerate(zip(camera_names, camera_paths)):
            settings = camera_cfg["cameras"][name]
            sensor = Camera(
                prim_path=camera_path,
                name=f"rj45_record_{name}_{index}",
                resolution=(int(settings["width"]), int(settings["height"])),
            )
            sensor.initialize()
            camera_sensors[name] = sensor
            if record_format == "video":
                video_writers[name] = RgbVideoWriter(
                    record_out_dir / "videos" / f"{name}.mp4",
                    int(settings["width"]),
                    int(settings["height"]),
                    float(record_hz),
                )
        for _ in range(12):
            app.update()

        home_object_points = np.asarray(
            [world_position("/World/RJ45Socket"), world_position("/World/RJ45Plug")],
            dtype=float,
        )
        for name, camera_path in zip(camera_names, camera_paths):
            settings = camera_cfg["cameras"][name]
            width = int(settings["width"])
            height = int(settings["height"])
            sensor = camera_sensors[name]
            rgba = np.asarray(sensor.get_rgba())
            if rgba.shape != (height, width, 4):
                raise RuntimeError(
                    f"Camera {name} returned {rgba.shape}, expected {(height, width, 4)}"
                )
            rgb = rgba[:, :, :3]
            if float(rgb.std()) < 1.0:
                raise RuntimeError(f"Camera {name} returned a blank home frame")
            pixels = np.asarray(sensor.get_image_coords_from_world_points(home_object_points))
            inside = (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
            if name in ("left_wrist", "right_wrist") and not bool(np.all(inside)):
                raise RuntimeError(
                    f"Home {name} camera does not see both objects: "
                    f"pixels={np.round(pixels, 1).tolist()} inside={inside.tolist()}"
                )
            print(
                f"Recorder camera ready: {name} shape={rgba.shape} "
                f"object_pixels={np.round(pixels, 1).tolist()} inside={inside.tolist()}",
                flush=True,
            )
        # Keep render products enabled so RTX temporal denoising and antialiasing
        # history survive between recorded frames. The headless replay loop still
        # performs the five intermediate PhysX substeps without rendering.

    def preinsert_alignment_metrics() -> dict[str, float | np.ndarray | bool]:
        alignment_cfg = task_cfg.get("assembly_auto_ik", {}).get(
            "preinsert_alignment", {}
        )
        socket_position, socket_rotation = world_pose("/World/RJ45Socket")
        plug_position, plug_rotation = world_pose("/World/RJ45Plug")
        socket_axis = socket_rotation @ np.asarray([0.0, 0.0, -1.0], dtype=float)
        plug_axis = plug_rotation @ np.asarray([0.0, 0.0, -1.0], dtype=float)
        socket_entry = socket_position + socket_axis * 0.01758
        plug_tip = plug_position + plug_axis * 0.019
        socket_axis_error = math.acos(
            min(1.0, max(-1.0, float(np.dot(socket_axis, [0.0, -1.0, 0.0]))))
        )
        plug_axis_error = math.acos(
            min(1.0, max(-1.0, float(np.dot(plug_axis, [0.0, 1.0, 0.0]))))
        )
        opposing_error = math.acos(
            min(1.0, max(-1.0, float(np.dot(socket_axis, -plug_axis))))
        )
        feature_delta = plug_tip - socket_entry
        lateral_error = math.hypot(float(feature_delta[0]), float(feature_delta[2]))
        axial_separation = float(feature_delta[1])
        max_lateral = float(alignment_cfg.get("max_lateral_error_m", 0.006))
        max_axis_error = float(alignment_cfg.get("max_axis_error_rad", 0.12))
        valid = (
            lateral_error <= max_lateral
            and socket_axis_error <= max_axis_error
            and plug_axis_error <= max_axis_error
            and opposing_error <= max_axis_error
            and axial_separation < 0.0
        )
        return {
            "lateral_error": lateral_error,
            "socket_axis_error": socket_axis_error,
            "plug_axis_error": plug_axis_error,
            "opposing_error": opposing_error,
            "axial_separation": axial_separation,
            "socket_entry": socket_entry,
            "plug_tip": plug_tip,
            "feature_delta": feature_delta,
            "valid": valid,
        }

    def validate_preinsert_alignment(label: str, *, raise_on_failure: bool) -> bool:
        metrics = preinsert_alignment_metrics()
        print(
            f"Auto-IK {label} alignment check: "
            f"tip/entry lateral={float(metrics['lateral_error']) * 1000.0:.2f} mm, "
            f"socket_axis={math.degrees(float(metrics['socket_axis_error'])):.2f} deg, "
            f"plug_axis={math.degrees(float(metrics['plug_axis_error'])):.2f} deg, "
            f"opposing={math.degrees(float(metrics['opposing_error'])):.2f} deg, "
            f"tip/entry axial={float(metrics['axial_separation']) * 1000.0:.2f} mm, "
            f"valid={bool(metrics['valid'])}",
            flush=True,
        )
        if raise_on_failure and not bool(metrics["valid"]):
            raise RuntimeError("RJ45 plug and socket are not aligned; insertion is blocked")
        return bool(metrics["valid"])

    assembly_cfg = task_cfg.get("assembly_auto_ik", {})
    adaptive_cfg = assembly_cfg.get("adaptive_alignment", {})
    posture_cfg = assembly_cfg.get("posture_safety", {})
    adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
    adaptive_left_kin = None
    adaptive_right_kin = None
    adaptive_left_home = None
    adaptive_right_home = None
    adaptive_solver_cfg = None

    # Recording always needs FK to encode target EEF poses in the 34-D state,
    # even when runtime adaptive trajectory rewriting is disabled.
    if adaptive_enabled or record_out_dir is not None:
        robot = robot_cfg["robot"]
        urdf_path = ROOT / robot["preferred_urdf_for_first_pass"]
        adaptive_left_kin = ArmPinocchioKinematics(
            urdf_path,
            [ROOT],
            robot_cfg["left_arm"]["joints"],
            robot["left_eef_link"],
        )
        adaptive_right_kin = ArmPinocchioKinematics(
            urdf_path,
            [ROOT],
            robot_cfg["right_arm"]["joints"],
            robot["right_eef_link"],
        )
        adaptive_left_home = arm_home(robot_cfg, "left")
        adaptive_right_home = arm_home(robot_cfg, "right")

    if adaptive_enabled:
        adaptive_solver_cfg = dict(task_cfg["scripted_auto_ik"])
        adaptive_solver_cfg.update(
            {
                "waypoint_position_tolerance_m": float(
                    adaptive_cfg.get("ik_position_tolerance_m", 0.002)
                ),
                "waypoint_rotation_tolerance_rad": float(
                    adaptive_cfg.get("ik_rotation_tolerance_rad", 0.04)
                ),
                "ik_max_iters": int(adaptive_cfg.get("ik_max_iters", 500)),
                "ik_step_scale": float(adaptive_cfg.get("ik_step_scale", 0.25)),
            }
        )

    base_world_xyz = np.asarray(task_cfg["scene"]["robot_base_world_xyz"], dtype=float)
    base_world_rotation = quaternion_xyzw_to_matrix(
        wxyz_to_xyzw(task_cfg["scene"]["robot_base_world_quat_wxyz"])
    )
    pickup_cfg = task_cfg["pickup_auto_ik"]
    left_grasp_rotation = np.asarray(
        pickup_cfg["left_grasp_rotation_matrix"], dtype=float
    ).reshape(3, 3)
    right_grasp_rotation = np.asarray(
        pickup_cfg["right_grasp_rotation_matrix"], dtype=float
    ).reshape(3, 3)
    socket_goal_rotation_base = np.asarray(
        assembly_cfg["left_compensated_rotation_matrix"], dtype=float
    ).reshape(3, 3) @ left_grasp_rotation.T
    plug_goal_rotation_base = np.asarray(
        assembly_cfg["right_compensated_rotation_matrix"], dtype=float
    ).reshape(3, 3) @ right_grasp_rotation.T
    nominal_socket_goal_position_base = np.asarray(
        assembly_cfg["socket_target_xyz_base"], dtype=float
    )
    socket_goal_position_base = np.array(nominal_socket_goal_position_base, copy=True)
    nominal_socket_release_position_base = np.asarray(
        assembly_cfg["socket_release_target_xyz_base"], dtype=float
    )
    socket_release_position_base = np.array(
        nominal_socket_release_position_base, copy=True
    )
    plug_alignment_bias_base = np.zeros(3, dtype=float)

    right_object_offset_keys = {
        "assembly_staging": "plug_pre_insert_offset_from_socket_m",
        "assembly_staging_hold": "plug_pre_insert_offset_from_socket_m",
        "approach_slot_far": "plug_far_offset_from_socket_m",
        "align_plug_tip": "plug_far_offset_from_socket_m",
        "alignment_correction_1": "plug_far_offset_from_socket_m",
        "alignment_correction_2": "plug_far_offset_from_socket_m",
        "aligned_hold": "plug_far_offset_from_socket_m",
        "approach_slot_near": "plug_near_offset_from_socket_m",
        "seat_at_slot_mouth": "plug_mouth_offset_from_socket_m",
        "insert_plug_shallow": "plug_shallow_insert_offset_from_socket_m",
        "insert_plug": "plug_final_insert_offset_from_socket_m",
        "insert_hold": "plug_final_insert_offset_from_socket_m",
    }
    object_constrained_left_phases = set(right_object_offset_keys)

    def measured_object_from_palm(
        object_path: str, palm_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        object_position, object_rotation = world_pose(object_path)
        palm_position, palm_rotation = world_pose(palm_path)
        return (
            palm_rotation.T @ (object_position - palm_position),
            palm_rotation.T @ object_rotation,
        )

    def measure_held_transforms(label: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        held = {
            "left": measured_object_from_palm(
                "/World/RJ45Socket", f"{robot_model_root}/LH_palm_center"
            ),
            "right": measured_object_from_palm(
                "/World/RJ45Plug", f"{robot_model_root}/RH_palm_center"
            ),
        }
        for side in ("left", "right"):
            translation, rotation = held[side]
            print(
                f"Adaptive SE(3) calibration {label} {side}: "
                f"object_from_palm_xyz={np.round(translation, 6).tolist()}, "
                f"object_from_palm_rotation={np.round(rotation, 5).tolist()}",
                flush=True,
            )
        return held

    def palm_pose_for_object_goal(
        object_position_base: np.ndarray,
        object_rotation_base: np.ndarray,
        object_from_palm: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        object_position_world = base_world_xyz + base_world_rotation @ object_position_base
        object_rotation_world = base_world_rotation @ object_rotation_base
        relative_translation, relative_rotation = object_from_palm
        palm_rotation_world = object_rotation_world @ relative_rotation.T
        palm_position_world = object_position_world - palm_rotation_world @ relative_translation
        return (
            base_world_rotation.T @ (palm_position_world - base_world_xyz),
            base_world_rotation.T @ palm_rotation_world,
        )

    def adaptive_phase_targets(
        phase: str,
        held: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
        if phase in right_object_offset_keys:
            right_object_position = socket_goal_position_base + np.asarray(
                assembly_cfg[right_object_offset_keys[phase]], dtype=float
            ) + plug_alignment_bias_base
            right_target = palm_pose_for_object_goal(
                right_object_position, plug_goal_rotation_base, held["right"]
            )
            if phase in object_constrained_left_phases:
                left_target = palm_pose_for_object_goal(
                    socket_goal_position_base,
                    socket_goal_rotation_base,
                    held["left"],
                )
            else:
                left_target = (
                    np.asarray(assembly_cfg["left_post_release_hover_xyz_base"], dtype=float),
                    np.asarray(assembly_cfg["left_release_rotation_matrix"], dtype=float),
                )
            return left_target, right_target

        return None

    def solve_adaptive_solution(
        side: str,
        target: tuple[np.ndarray, np.ndarray],
        seed_arm: np.ndarray,
    ):
        kin = adaptive_left_kin if side == "left" else adaptive_right_kin
        home = adaptive_left_home if side == "left" else adaptive_right_home
        if kin is None or home is None or adaptive_solver_cfg is None:
            raise RuntimeError("Adaptive IK was not initialized")
        seed = kin.set_arm(kin.neutral(), seed_arm)
        if bool(posture_cfg.get("enabled", False)):
            solution, report = solve_posture_safe_pose(
                kin,
                side,
                np.asarray(target[0], dtype=float),
                np.asarray(target[1], dtype=float),
                np.asarray(seed_arm, dtype=float),
                np.asarray(home, dtype=float),
                adaptive_solver_cfg,
                posture_cfg,
            )
            if solution.success:
                print(
                    f"Adaptive posture {side}: "
                    f"elbow_y={float(report['elbow_outward_y_m']) * 1000.0:.1f} mm, "
                    f"wrist_y={float(report['wrist_outward_y_m']) * 1000.0:.1f} mm",
                    flush=True,
                )
            return solution
        return solve_with_restarts(
            kin,
            np.asarray(target[0], dtype=float),
            seed,
            home,
            adaptive_solver_cfg,
            np.asarray(target[1], dtype=float),
        )

    def solve_adaptive_arm(
        side: str,
        target: tuple[np.ndarray, np.ndarray],
        seed_arm: np.ndarray,
    ) -> np.ndarray:
        solution = solve_adaptive_solution(side, target, seed_arm)
        print(
            f"Adaptive IK {side}: success={solution.success}, "
            f"position_error={solution.position_error_m * 1000.0:.2f} mm, "
            f"rotation_error={math.degrees(solution.rotation_error_rad):.2f} deg",
            flush=True,
        )
        if not solution.success:
            raise RuntimeError(f"Adaptive {side} arm IK failed before insertion")
        return np.asarray(solution.q_arm, dtype=float)

    def solve_adaptive_or_preserve(
        side: str,
        target: tuple[np.ndarray, np.ndarray],
        seed_arm: np.ndarray,
        fallback_arm: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        try:
            return solve_adaptive_arm(side, target, seed_arm)
        except RuntimeError as exc:
            # A recovery correction must not stop the episode because one arm's
            # measured disturbed pose is outside the strict posture-safe IK set.
            # Keep that arm on its verified offline waypoint while the other arm
            # can still receive the measured alignment correction.
            print(
                f"Adaptive IK fallback {side} phase={phase}: {exc}; "
                "preserving offline target",
                flush=True,
            )
            return np.asarray(fallback_arm, dtype=float)

    def select_shared_assembly_goal(
        held: dict[str, tuple[np.ndarray, np.ndarray]],
        left_seed: np.ndarray,
        right_seed: np.ndarray,
    ) -> None:
        search_cfg = adaptive_cfg.get("shared_workspace_search", {})
        x_offsets = search_cfg.get("x_offsets_m", [0.0])
        y_offsets = search_cfg.get("y_offsets_m", [0.0])
        z_offsets = search_cfg.get("z_offsets_m", [0.0])
        right_keys = (
            "plug_pre_insert_offset_from_socket_m",
            "plug_far_offset_from_socket_m",
            "plug_near_offset_from_socket_m",
            "plug_mouth_offset_from_socket_m",
            "plug_final_insert_offset_from_socket_m",
        )
        best_failure = None
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                for z_offset in z_offsets:
                    offset = np.asarray([x_offset, y_offset, z_offset], dtype=float)
                    candidate = nominal_socket_goal_position_base + offset
                    left_target = palm_pose_for_object_goal(
                        candidate, socket_goal_rotation_base, held["left"]
                    )
                    left_solution = solve_adaptive_solution("left", left_target, left_seed)
                    right_solutions = []
                    candidate_right_seed = np.asarray(right_seed, dtype=float)
                    for key in right_keys:
                        right_target = palm_pose_for_object_goal(
                            candidate + np.asarray(assembly_cfg[key], dtype=float),
                            plug_goal_rotation_base,
                            held["right"],
                        )
                        solution = solve_adaptive_solution(
                            "right", right_target, candidate_right_seed
                        )
                        right_solutions.append(solution)
                        candidate_right_seed = np.asarray(solution.q_arm, dtype=float)
                    solutions = [left_solution, *right_solutions]
                    max_position_error = max(
                        float(solution.position_error_m) for solution in solutions
                    )
                    max_rotation_error = max(
                        float(solution.rotation_error_rad) for solution in solutions
                    )
                    all_success = all(solution.success for solution in solutions)
                    print(
                        "Adaptive shared-workspace candidate: "
                        f"offset_mm={np.round(offset * 1000.0, 1).tolist()}, "
                        f"max_position_error={max_position_error * 1000.0:.2f} mm, "
                        f"max_rotation_error={math.degrees(max_rotation_error):.2f} deg, "
                        f"valid={all_success}",
                        flush=True,
                    )
                    failure_score = max_position_error + 0.05 * max_rotation_error
                    if best_failure is None or failure_score < best_failure[0]:
                        best_failure = (failure_score, offset, max_position_error, max_rotation_error)
                    if all_success:
                        socket_goal_position_base[:] = candidate
                        print(
                            "Adaptive shared assembly goal selected: "
                            f"socket_xyz_base={np.round(candidate, 6).tolist()}, "
                            f"offset_mm={np.round(offset * 1000.0, 1).tolist()}",
                            flush=True,
                        )
                        select_release_goal(held, right_solutions[-1].q_arm)
                        return
        if best_failure is None:
            raise RuntimeError("Adaptive shared-workspace search has no candidates")
        _, offset, position_error, rotation_error = best_failure
        raise RuntimeError(
            "No shared bimanual assembly goal passed IK; best "
            f"offset_mm={np.round(offset * 1000.0, 1).tolist()}, "
            f"position_error={position_error * 1000.0:.2f} mm, "
            f"rotation_error={math.degrees(rotation_error):.2f} deg"
        )

    def select_release_goal(
        held: dict[str, tuple[np.ndarray, np.ndarray]],
        right_seed: np.ndarray,
    ) -> None:
        search_cfg = adaptive_cfg.get("release_workspace_search", {})
        final_offset = np.asarray(
            assembly_cfg["plug_final_insert_offset_from_socket_m"], dtype=float
        )
        best_failure = None
        for x_offset in search_cfg.get("x_offsets_m", [0.0]):
            for y_offset in search_cfg.get("y_offsets_m", [0.0]):
                for z_offset in search_cfg.get("z_offsets_m", [0.0]):
                    offset = np.asarray([x_offset, y_offset, z_offset], dtype=float)
                    candidate_socket = nominal_socket_release_position_base + offset
                    target = palm_pose_for_object_goal(
                        candidate_socket + final_offset,
                        plug_goal_rotation_base,
                        held["right"],
                    )
                    solution = solve_adaptive_solution("right", target, right_seed)
                    print(
                        "Adaptive release-workspace candidate: "
                        f"offset_mm={np.round(offset * 1000.0, 1).tolist()}, "
                        f"position_error={solution.position_error_m * 1000.0:.2f} mm, "
                        f"rotation_error={math.degrees(solution.rotation_error_rad):.2f} deg, "
                        f"valid={solution.success}",
                        flush=True,
                    )
                    score = float(solution.position_error_m) + 0.05 * float(
                        solution.rotation_error_rad
                    )
                    if best_failure is None or score < best_failure[0]:
                        best_failure = (score, offset, solution)
                    if solution.success:
                        socket_release_position_base[:] = candidate_socket
                        print(
                            "Adaptive release goal selected: "
                            f"socket_xyz_base={np.round(candidate_socket, 6).tolist()}, "
                            f"offset_mm={np.round(offset * 1000.0, 1).tolist()}",
                            flush=True,
                        )
                        return
        if best_failure is None:
            raise RuntimeError("Adaptive release-workspace search has no candidates")
        _, offset, solution = best_failure
        raise RuntimeError(
            "No release goal passed right-arm IK; best "
            f"offset_mm={np.round(offset * 1000.0, 1).tolist()}, "
            f"position_error={solution.position_error_m * 1000.0:.2f} mm, "
            f"rotation_error={math.degrees(solution.rotation_error_rad):.2f} deg"
        )

    def apply_lateral_residual_compensation(label: str) -> None:
        metrics = preinsert_alignment_metrics()
        feature_delta = np.asarray(metrics["feature_delta"], dtype=float)
        gain = float(adaptive_cfg.get("residual_compensation_gain", 1.0))
        plug_alignment_bias_base[0] -= gain * feature_delta[0]
        plug_alignment_bias_base[2] -= gain * feature_delta[2]
        maximum = float(adaptive_cfg.get("max_residual_compensation_m", 0.008))
        lateral_norm = float(np.linalg.norm(plug_alignment_bias_base[[0, 2]]))
        if lateral_norm > maximum:
            plug_alignment_bias_base[[0, 2]] *= maximum / lateral_norm
        print(
            f"Adaptive lateral residual compensation {label}: "
            f"measured_xz_mm={np.round(feature_delta[[0, 2]] * 1000.0, 3).tolist()}, "
            f"accumulated_bias_xz_mm="
            f"{np.round(plug_alignment_bias_base[[0, 2]] * 1000.0, 3).tolist()}",
            flush=True,
        )

    def rewrite_adaptive_trajectory(first_phase: str, label: str) -> None:
        if not adaptive_enabled:
            return
        candidates = np.flatnonzero(phases == first_phase)
        if candidates.size == 0:
            raise RuntimeError(f"Adaptive phase is missing from episode: {first_phase}")
        start_index = int(candidates[0])
        held = measure_held_transforms(label)
        actual = np.asarray(
            articulation.get_joint_positions(joint_indices=joint_indices), dtype=float
        ).reshape(-1)
        previous_left = np.asarray(actual[:7], dtype=float)
        previous_right = np.asarray(actual[13:20], dtype=float)
        if label == "hover_objects":
            select_shared_assembly_goal(held, previous_left, previous_right)
        cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        cursor = start_index
        rewritten_phases = []
        while cursor < frame_count:
            phase = str(phases[cursor])
            end = cursor + 1
            while end < frame_count and phases[end] == phase:
                end += 1
            targets = adaptive_phase_targets(phase, held)
            if targets is not None:
                key = (
                    tuple(np.round(targets[0][0], 8)),
                    tuple(np.round(targets[0][1].reshape(-1), 8)),
                    tuple(np.round(targets[1][0], 8)),
                    tuple(np.round(targets[1][1].reshape(-1), 8)),
                )
                if key in cache:
                    target_left, target_right = cache[key]
                else:
                    target_left = solve_adaptive_or_preserve(
                        "left",
                        targets[0],
                        previous_left,
                        nominal_left_arm[end - 1],
                        phase,
                    )
                    target_right = solve_adaptive_or_preserve(
                        "right",
                        targets[1],
                        previous_right,
                        nominal_right_arm[end - 1],
                        phase,
                    )
                    cache[key] = (target_left, target_right)
            elif phase == "lower_assembly_right":
                release_plug_position = (
                    socket_release_position_base
                    + np.asarray(
                        assembly_cfg["plug_final_insert_offset_from_socket_m"],
                        dtype=float,
                    )
                    + plug_alignment_bias_base
                )
                release_right_target = palm_pose_for_object_goal(
                    release_plug_position,
                    plug_goal_rotation_base,
                    held["right"],
                )
                # Keep the left arm at its final adaptive insertion target while
                # the right arm alone carries the latched assembly to the table.
                target_left = np.asarray(previous_left, dtype=float)
                target_right = solve_adaptive_or_preserve(
                    "right",
                    release_right_target,
                    previous_right,
                    nominal_right_arm[end - 1],
                    phase,
                )
            elif phase in (
                "release_left_hand",
                "release_right_hand",
                "post_release_hold",
            ):
                # The adaptive assembly solve may differ from the offline left-arm
                # and right-arm solutions even at the same EEF pose.  Holding both
                # final targets here prevents a discontinuity while either hand
                # opens or the placed assembly settles.
                target_left = np.asarray(previous_left, dtype=float)
                target_right = np.asarray(previous_right, dtype=float)
            elif phase in ("return_home", "episode_end"):
                target_left = np.asarray(nominal_left_arm[end - 1], dtype=float)
                target_right = np.asarray(nominal_right_arm[end - 1], dtype=float)
            else:
                # Curriculum disturbance and recovery phases are intentionally
                # absent from adaptive_phase_targets.  Preserve their offline
                # perturbed endpoint, but interpolate to it from the previous
                # adaptive target so the runtime rewrite cannot jump at a phase
                # boundary.  The next adaptive phase then starts from this
                # actual endpoint rather than from a stale pre-disturbance seed.
                target_left = np.asarray(nominal_left_arm[end - 1], dtype=float)
                target_right = np.asarray(nominal_right_arm[end - 1], dtype=float)

            count = end - cursor
            if bool(posture_cfg.get("enabled", False)):
                if not torso_safe_joint_path(
                    adaptive_left_kin,
                    "left",
                    previous_left,
                    target_left,
                    posture_cfg,
                ):
                    raise RuntimeError(
                        f"Adaptive left path enters torso corridor at phase {phase}"
                    )
                if not torso_safe_joint_path(
                    adaptive_right_kin,
                    "right",
                    previous_right,
                    target_right,
                    posture_cfg,
                ):
                    raise RuntimeError(
                        f"Adaptive right path enters torso corridor at phase {phase}"
                    )
            for offset, frame in enumerate(range(cursor, end)):
                linear = offset / float(max(1, count - 1))
                alpha = linear**3 * (linear * (linear * 6.0 - 15.0) + 10.0)
                left_arm[frame] = previous_left * (1.0 - alpha) + target_left * alpha
                right_arm[frame] = previous_right * (1.0 - alpha) + target_right * alpha
            previous_left = np.asarray(target_left, dtype=float)
            previous_right = np.asarray(target_right, dtype=float)
            rewritten_phases.append(phase)
            cursor = end
        print(
            f"Adaptive trajectory rewrite {label}: start={first_phase}, "
            f"phases={rewritten_phases}",
            flush=True,
        )

    latch_engaged = False
    latch_mode = None
    latched_plug_from_socket = None
    latched_socket_from_plug = None

    socket_prim = stage.GetPrimAtPath("/World/RJ45Socket")
    plug_prim = stage.GetPrimAtPath("/World/RJ45Plug")
    socket_translate_attr = socket_prim.GetAttribute("xformOp:translate")
    socket_orient_attr = socket_prim.GetAttribute("xformOp:orient")
    plug_translate_attr = plug_prim.GetAttribute("xformOp:translate")
    plug_orient_attr = plug_prim.GetAttribute("xformOp:orient")
    if not all(
        (socket_translate_attr, socket_orient_attr, plug_translate_attr, plug_orient_attr)
    ):
        raise RuntimeError("RJ45 assets are missing translate/orient xform attributes")

    def set_world_pose(translate_attr, orient_attr, matrix) -> None:
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        translate_attr.Set(Gf.Vec3d(*[float(value) for value in translation]))
        orient_attr.Set(
            Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(*[float(value) for value in imaginary]),
            )
        )

    def set_kinematic(prim, enabled: bool) -> None:
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        kinematic_attr = rigid_body.GetKinematicEnabledAttr()
        if not kinematic_attr:
            kinematic_attr = rigid_body.CreateKinematicEnabledAttr()
        kinematic_attr.Set(enabled)

    def set_collision_tree_enabled(root_prim, enabled: bool) -> None:
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision = UsdPhysics.CollisionAPI(prim)
            collision_attr = collision.GetCollisionEnabledAttr()
            if not collision_attr:
                collision_attr = collision.CreateCollisionEnabledAttr()
            collision_attr.Set(enabled)

    def update_latched_pair_pose() -> None:
        if not latch_engaged:
            return
        if latch_mode == "socket_follows_plug" and latched_socket_from_plug is not None:
            plug_matrix = UsdGeom.Xformable(plug_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            set_world_pose(
                socket_translate_attr,
                socket_orient_attr,
                latched_socket_from_plug * plug_matrix,
            )
        elif latch_mode == "plug_follows_socket" and latched_plug_from_socket is not None:
            socket_matrix = UsdGeom.Xformable(socket_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            set_world_pose(
                plug_translate_attr,
                plug_orient_attr,
                latched_plug_from_socket * socket_matrix,
            )

    def add_latched_assembly_support_collisions() -> None:
        socket_support_path = "/World/RJ45Socket/latched_socket_support"
        if not stage.GetPrimAtPath(socket_support_path).IsValid():
            socket_points = np.asarray(
                UsdGeom.Mesh(
                    stage.GetPrimAtPath("/World/RJ45Socket/visual")
                ).GetPointsAttr().Get(),
                dtype=float,
            )
            socket_bounds_min = socket_points.min(axis=0)
            socket_bounds_max = socket_points.max(axis=0)
            add_box(
                stage,
                socket_support_path,
                (socket_bounds_min + socket_bounds_max) * 0.5,
                socket_bounds_max - socket_bounds_min + 0.001,
                (0.05, 0.07, 0.11),
            )
            socket_support_prim = stage.GetPrimAtPath(socket_support_path)
            UsdPhysics.CollisionAPI.Apply(socket_support_prim)
            UsdGeom.Imageable(socket_support_prim).MakeInvisible()

        support_path = "/World/RJ45Socket/latched_plug_support"
        if stage.GetPrimAtPath(support_path).IsValid():
            return
        relative_translation = latched_plug_from_socket.ExtractTranslation()
        relative_rotation = latched_plug_from_socket.ExtractRotationQuat()
        relative_imaginary = relative_rotation.GetImaginary()
        support = UsdGeom.Xform.Define(stage, support_path)
        set_pose_xform(
            support.GetPrim(),
            relative_translation,
            (
                float(relative_imaginary[0]),
                float(relative_imaginary[1]),
                float(relative_imaginary[2]),
                float(relative_rotation.GetReal()),
            ),
        )
        points = np.asarray(
            UsdGeom.Mesh(stage.GetPrimAtPath("/World/RJ45Plug/visual")).GetPointsAttr().Get(),
            dtype=float,
        )
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        proxy_path = f"{support_path}/collision_proxy"
        add_box(
            stage,
            proxy_path,
            (bounds_min + bounds_max) * 0.5,
            bounds_max - bounds_min + 0.001,
            (0.02, 0.02, 0.02),
        )
        proxy_prim = stage.GetPrimAtPath(proxy_path)
        UsdPhysics.CollisionAPI.Apply(proxy_prim)
        UsdGeom.Imageable(proxy_prim).MakeInvisible()

    def transfer_latch_to_socket_support() -> None:
        nonlocal latch_mode
        if latch_mode != "socket_follows_plug":
            return
        update_latched_pair_pose()
        set_collision_tree_enabled(socket_prim, True)
        add_latched_assembly_support_collisions()
        set_kinematic(socket_prim, False)
        PhysxSchema.PhysxRigidBodyAPI(plug_prim).GetEnableCCDAttr().Set(False)
        set_kinematic(plug_prim, True)
        set_collision_tree_enabled(plug_prim, False)
        latch_mode = "plug_follows_socket"
        update_latched_pair_pose()
        print(
            "RJ45 latch support transferred to the dynamic socket for table placement",
            flush=True,
        )

    def engage_post_insert_latch(carrier: str) -> bool:
        nonlocal latch_engaged, latch_mode, latched_plug_from_socket, latched_socket_from_plug
        latch_cfg = task_cfg.get("assembly_auto_ik", {}).get("post_insert_latch", {})
        latch_enabled = bool(latch_cfg.get("enabled", False))
        socket_matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath("/World/RJ45Socket")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        plug_matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath("/World/RJ45Plug")
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        plug_from_socket = plug_matrix * socket_matrix.GetInverse()
        relative_translation = plug_from_socket.ExtractTranslation()
        relative_rotation = plug_from_socket.ExtractRotationQuat()
        relative_imaginary = relative_rotation.GetImaginary()
        insertion_depth = 0.03658 + float(relative_translation[2])
        lateral_error = math.hypot(
            float(relative_translation[0]), float(relative_translation[1])
        )
        orientation_error = 2.0 * math.acos(
            min(1.0, max(0.0, abs(float(relative_imaginary[0]))))
        )
        print(
            "Auto-IK insertion before release: "
            f"depth={insertion_depth * 1000.0:.1f} mm, "
            f"lateral={lateral_error * 1000.0:.1f} mm, "
            f"orientation={math.degrees(orientation_error):.1f} deg",
            flush=True,
        )
        if not latch_enabled:
            return False
        valid = (
            insertion_depth >= float(latch_cfg.get("min_insert_depth_m", 0.008))
            and lateral_error <= float(latch_cfg.get("max_lateral_error_m", 0.008))
            and orientation_error <= float(latch_cfg.get("max_orientation_error_rad", 0.20))
        )
        if not valid:
            print(
                "RJ45 latch not engaged: "
                f"depth={insertion_depth * 1000.0:.1f} mm, "
                f"lateral={lateral_error * 1000.0:.1f} mm, "
                f"orientation={math.degrees(orientation_error):.1f} deg",
                flush=True,
            )
            return False

        latched_plug_from_socket = Gf.Matrix4d(plug_from_socket)
        latched_socket_from_plug = Gf.Matrix4d(plug_from_socket.GetInverse())
        if carrier == "plug":
            PhysxSchema.PhysxRigidBodyAPI(socket_prim).GetEnableCCDAttr().Set(False)
            set_kinematic(socket_prim, True)
            set_collision_tree_enabled(socket_prim, False)
            latch_mode = "socket_follows_plug"
        else:
            PhysxSchema.PhysxRigidBodyAPI(plug_prim).GetEnableCCDAttr().Set(False)
            set_kinematic(plug_prim, True)
            set_collision_tree_enabled(plug_prim, False)
            add_latched_assembly_support_collisions()
            latch_mode = "plug_follows_socket"
        latch_engaged = True
        update_latched_pair_pose()
        print(
            "RJ45 internal adhesive latch engaged after insertion: "
            f"depth={insertion_depth * 1000.0:.1f} mm, "
            f"lateral={lateral_error * 1000.0:.1f} mm, "
            f"orientation={math.degrees(orientation_error):.1f} deg, carrier={carrier}",
            flush=True,
        )
        return True

    def validate_grasp(frame_index: int) -> bool:
        pickup_cfg = task_cfg["pickup_auto_ik"]
        palm_limit = float(pickup_cfg["grasp_check_max_palm_distance_m"])
        finger_limit = float(pickup_cfg["grasp_check_max_finger_distance_m"])
        socket_position = world_position("/World/RJ45Socket")
        plug_position = world_position("/World/RJ45Plug")
        left_palm_distance = float(
            np.linalg.norm(world_position(f"{robot_model_root}/LH_palm_center") - socket_position)
        )
        right_palm_distance = float(
            np.linalg.norm(world_position(f"{robot_model_root}/RH_palm_center") - plug_position)
        )
        left_finger_distances = [
            float(np.linalg.norm(world_position(f"{robot_model_root}/LH_{finger}_distal") - socket_position))
            for finger in ("thumb", "index")
        ]
        right_finger_distances = [
            float(np.linalg.norm(world_position(f"{robot_model_root}/RH_{finger}_distal") - plug_position))
            for finger in ("thumb", "index")
        ]

        left_cfg = robot_cfg["left_o6_hand"]
        right_cfg = robot_cfg["right_o6_hand"]
        left_expected = np.asarray(
            [left_cfg["pinch_preset"][name] for name in left_cfg["active_driver_joints"]], dtype=float
        )
        right_expected = np.asarray(
            [right_cfg["pinch_preset"][name] for name in right_cfg["active_driver_joints"]], dtype=float
        )
        left_closed = bool(np.max(np.abs(left_hand[frame_index] - left_expected)) <= 0.05)
        right_pinched = bool(np.max(np.abs(right_hand[frame_index] - right_expected)) <= 0.05)
        left_valid = (
            left_closed
            and left_palm_distance <= palm_limit
            and all(distance <= finger_limit for distance in left_finger_distances)
        )
        right_valid = (
            right_pinched
            and right_palm_distance <= palm_limit
            and all(distance <= finger_limit for distance in right_finger_distances)
        )
        print(
            "Auto-IK grasp check: "
            f"left palm={left_palm_distance * 1000.0:.1f} mm "
            f"thumb/index={[round(value * 1000.0, 1) for value in left_finger_distances]} valid={left_valid}; "
            f"right palm={right_palm_distance * 1000.0:.1f} mm "
            f"thumb/index={[round(value * 1000.0, 1) for value in right_finger_distances]} valid={right_valid}",
            flush=True,
        )
        return bool(left_valid and right_valid)

    def transform_points(prim_path: str, points: np.ndarray) -> np.ndarray:
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        return np.asarray(
            [matrix.Transform(Gf.Vec3d(*point)) for point in np.asarray(points, dtype=float)],
            dtype=float,
        )

    def visual_mesh_points_world(prim_path: str) -> np.ndarray:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
        return transform_points(prim_path, np.asarray(mesh.GetPointsAttr().Get(), dtype=float))

    def closest_surface_pair(first: np.ndarray, second: np.ndarray):
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(second).query(first)
        first_index = int(np.argmin(distances))
        second_index = int(indices[first_index])
        return float(distances[first_index]), first[first_index], second[second_index]

    def report_grasp_surfaces() -> None:
        for side, prefix, object_visual in (
            ("left", "LH", "/World/RJ45Socket/visual"),
            ("right", "RH", "/World/RJ45Plug/visual"),
        ):
            object_points = visual_mesh_points_world(object_visual)
            finger_points = {}
            for finger in ("thumb", "index"):
                mesh_path = (
                    ROOT
                    / "robot_description"
                    / "S4"
                    / "meshes"
                    / "o6"
                    / side
                    / "meshes"
                    / f"{finger}_distal.STL"
                )
                local_points = np.asarray(__import__("trimesh").load_mesh(mesh_path).vertices)
                local_points = local_points[
                    local_points[:, 2] >= float(local_points[:, 2].max()) - 0.012
                ]
                finger_points[finger] = transform_points(
                    f"{robot_model_root}/{prefix}_{finger}_distal", local_points
                )

            thumb_gap, thumb_point, thumb_object_point = closest_surface_pair(
                finger_points["thumb"], object_points
            )
            index_gap, index_point, index_object_point = closest_surface_pair(
                finger_points["index"], object_points
            )
            pinch_gap, pinch_thumb_point, pinch_index_point = closest_surface_pair(
                finger_points["thumb"], finger_points["index"]
            )
            pinch_midpoint = (pinch_thumb_point + pinch_index_point) * 0.5
            print(
                f"{side} grasp surfaces: thumb/object={thumb_gap * 1000.0:.2f} mm "
                f"at {np.round(thumb_point, 4).tolist()} -> {np.round(thumb_object_point, 4).tolist()}, "
                f"index/object={index_gap * 1000.0:.2f} mm "
                f"at {np.round(index_point, 4).tolist()} -> {np.round(index_object_point, 4).tolist()}, "
                f"thumb/index={pinch_gap * 1000.0:.2f} mm "
                f"midpoint={np.round(pinch_midpoint, 4).tolist()}",
                flush=True,
            )

        actual = np.asarray(
            articulation.get_joint_positions(joint_indices=joint_indices), dtype=float
        ).reshape(-1)
        desired = np.concatenate(
            [left_arm[frame_index], left_hand[frame_index], right_arm[frame_index], right_hand[frame_index]]
        )
        print(
            "grasp joint tracking: "
            f"left arm max_error={np.max(np.abs(desired[:7] - actual[:7])):.4f}, "
            f"left hand target={np.round(desired[7:13], 4).tolist()} "
            f"actual={np.round(actual[7:13], 4).tolist()}; "
            f"right arm max_error={np.max(np.abs(desired[13:20] - actual[13:20])):.4f}, "
            f"right hand target={np.round(desired[20:26], 4).tolist()} "
            f"actual={np.round(actual[20:26], 4).tolist()}",
            flush=True,
        )
        coupled_actual = np.asarray(
            articulation.get_joint_positions(joint_indices=coupled_indices), dtype=float
        ).reshape(-1)
        for side, offset in (("left", 0), ("right", 5)):
            coupled_degrees = np.degrees(coupled_actual[offset : offset + 2])
            print(
                f"{side} coupled joint positions deg thumb_ip/index_dip="
                f"{[round(float(value), 3) for value in coupled_degrees]}",
                flush=True,
            )

        if grasp_capture_path is not None:
            from isaacsim.sensors.camera import Camera
            from PIL import Image

            overview = Camera(
                prim_path="/World/OverviewCamera",
                name="rj45_grasp_overview_capture",
                resolution=(1280, 720),
            )
            overview.initialize()
            for _ in range(12):
                app.update()
            rgba = np.asarray(overview.get_rgba())
            grasp_capture_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(grasp_capture_path)
            print(f"Saved grasp overview: {grasp_capture_path}", flush=True)

    initial_object_positions = {
        "socket": world_position("/World/RJ45Socket"),
        "plug": world_position("/World/RJ45Plug"),
    }

    def grasp_scalar(side: str, hand_positions: np.ndarray) -> float:
        hand_cfg = robot_cfg[f"{side}_o6_hand"]
        names = hand_cfg["active_driver_joints"]
        opened = np.asarray([hand_cfg["open_preset"][name] for name in names], dtype=float)
        closed = np.asarray([hand_cfg["pinch_preset"][name] for name in names], dtype=float)
        delta = closed - opened
        denominator = float(np.dot(delta, delta))
        if denominator < 1.0e-12:
            return 0.0
        value = float(np.dot(np.asarray(hand_positions, dtype=float) - opened, delta) / denominator)
        return float(np.clip(value, 0.0, 1.0))

    def target_eef_pose(kin, arm_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full = kin.set_arm(kin.neutral(), np.asarray(arm_positions, dtype=float))
        pose = kin.frame_pose(q_full)
        return np.asarray(pose.translation, dtype=float), np.asarray(pose.rotation, dtype=float)

    def actual_eef_pose(palm_path: str) -> tuple[np.ndarray, np.ndarray]:
        base_position, base_rotation = world_pose(f"{robot_model_root}/base_link")
        palm_position, palm_rotation = world_pose(palm_path)
        return (
            base_rotation.T @ (palm_position - base_position),
            base_rotation.T @ palm_rotation,
        )

    def record_frame(
        actual_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> None:
        if record_out_dir is None:
            return
        if adaptive_left_kin is None or adaptive_right_kin is None:
            raise RuntimeError("Recording requires initialized left and right kinematics")

        left_actual_xyz, left_actual_rotation = actual_eef_pose(
            f"{robot_model_root}/LH_palm_center"
        )
        right_actual_xyz, right_actual_rotation = actual_eef_pose(
            f"{robot_model_root}/RH_palm_center"
        )
        left_target_xyz, left_target_rotation = target_eef_pose(
            adaptive_left_kin, target_positions[:7]
        )
        right_target_xyz, right_target_rotation = target_eef_pose(
            adaptive_right_kin, target_positions[13:20]
        )
        left_actual_grasp = grasp_scalar("left", actual_positions[7:13])
        right_actual_grasp = grasp_scalar("right", actual_positions[20:26])
        left_target_grasp = grasp_scalar("left", target_positions[7:13])
        right_target_grasp = grasp_scalar("right", target_positions[20:26])

        actual_state = np.concatenate(
            [
                left_actual_xyz,
                rotation_to_rot6d(left_actual_rotation),
                actual_positions[:7],
                [left_actual_grasp],
                right_actual_xyz,
                rotation_to_rot6d(right_actual_rotation),
                actual_positions[13:20],
                [right_actual_grasp],
            ]
        ).astype(np.float32)
        target_state = np.concatenate(
            [
                left_target_xyz,
                rotation_to_rot6d(left_target_rotation),
                target_positions[:7],
                [left_target_grasp],
                right_target_xyz,
                rotation_to_rot6d(right_target_rotation),
                target_positions[13:20],
                [right_target_grasp],
            ]
        ).astype(np.float32)
        action = np.concatenate(
            [
                left_target_xyz,
                rotation_to_rot6d(left_target_rotation),
                [left_target_grasp],
                right_target_xyz,
                rotation_to_rot6d(right_target_rotation),
                [right_target_grasp],
            ]
        ).astype(np.float32)
        recorded_observation_state.append(actual_state)
        recorded_target_observation_state.append(target_state)
        recorded_action.append(action)
        recorded_actual_joint_position.append(np.asarray(actual_positions, dtype=np.float32))
        recorded_target_joint_position.append(np.asarray(target_positions, dtype=np.float32))

        for name, sensor in camera_sensors.items():
            settings = camera_cfg["cameras"][name]
            expected_shape = (int(settings["height"]), int(settings["width"]), 4)
            rgba = np.asarray(sensor.get_rgba())
            if rgba.shape != expected_shape:
                raise RuntimeError(
                    f"Camera {name} frame has shape {rgba.shape}, expected {expected_shape}"
                )
            rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
            if float(rgb.std()) < 1.0:
                raise RuntimeError(f"Camera {name} produced a blank frame")
            if record_format == "video":
                video_writers[name].write(rgb)
            else:
                recorded_camera_frames[name].append(rgb.copy())
            camera_frame_counts[name] += 1

    previous_phase = None
    physics_hz = float(task_cfg["simulation"]["physics_hz"])
    physics_steps_per_frame = max(1, int(round(physics_hz / float(record_hz))))
    physics_step_period = 1.0 / max(1e-6, physics_hz * float(replay_speed))
    max_arm_tracking_error = 0.0
    max_hand_tracking_error = 0.0
    max_object_heights = {
        "socket": float(initial_object_positions["socket"][2]),
        "plug": float(initial_object_positions["plug"][2]),
    }
    max_object_height_phases = {"socket": "initial", "plug": "initial"}
    arm_tracking_indices = np.asarray(list(range(0, 7)) + list(range(13, 20)), dtype=int)
    hand_tracking_indices = np.asarray(list(range(7, 13)) + list(range(20, 26)), dtype=int)
    previous_command = np.asarray(
        articulation.get_joint_positions(joint_indices=joint_indices), dtype=float
    ).reshape(-1)
    previous_coupled = np.concatenate(
        [
            coupled_hand_positions("left", previous_command[7:13]),
            coupled_hand_positions("right", previous_command[20:26]),
        ]
    )
    for frame_index, phase in enumerate(phases):
        if phase == "release_left_hand" and previous_phase != phase and not latch_engaged:
            engage_post_insert_latch(carrier="plug")
        if phase == "release_right_hand" and previous_phase != phase:
            if not latch_engaged:
                engage_post_insert_latch(carrier="socket")
            else:
                transfer_latch_to_socket_support()
        positions = np.concatenate([left_arm[frame_index], left_hand[frame_index], right_arm[frame_index], right_hand[frame_index]])
        velocity_target = (positions - previous_command) * float(record_hz)
        coupled_positions = np.asarray(
            coupled_hand_positions("left", positions[7:13])
            + coupled_hand_positions("right", positions[20:26]),
            dtype=float,
        )
        coupled_velocity_target = (coupled_positions - previous_coupled) * float(record_hz)
        for physics_step in range(1, physics_steps_per_frame + 1):
            alpha = physics_step / float(physics_steps_per_frame)
            substep_target = previous_command + alpha * (positions - previous_command)
            coupled_substep_target = previous_coupled + alpha * (
                coupled_positions - previous_coupled
            )
            articulation.set_joint_position_targets(
                np.asarray([substep_target], dtype=np.float32), joint_indices=joint_indices
            )
            articulation.set_joint_velocity_targets(
                np.asarray([velocity_target], dtype=np.float32), joint_indices=joint_indices
            )
            articulation.set_joint_position_targets(
                np.asarray([coupled_substep_target], dtype=np.float32),
                joint_indices=coupled_indices,
            )
            articulation.set_joint_velocity_targets(
                np.asarray([coupled_velocity_target], dtype=np.float32),
                joint_indices=coupled_indices,
            )
            if latch_engaged:
                update_latched_pair_pose()
            capture_step = bool(
                record_out_dir is not None
                and physics_step == physics_steps_per_frame
            )
            if realtime or capture_step:
                app.update()
            else:
                # Headless collection needs 120 Hz contacts and drives, but
                # rendering the five intermediate substeps is unnecessary.
                SimulationManager.step(render=False)
            if latch_engaged:
                update_latched_pair_pose()
            if realtime:
                time.sleep(physics_step_period)
            actual_positions = np.asarray(
                articulation.get_joint_positions(joint_indices=joint_indices), dtype=float
            ).reshape(-1)
            tracking_error = np.abs(actual_positions - substep_target)
            max_arm_tracking_error = max(
                max_arm_tracking_error, float(tracking_error[arm_tracking_indices].max())
            )
            max_hand_tracking_error = max(
                max_hand_tracking_error, float(tracking_error[hand_tracking_indices].max())
            )
            for object_name, object_path in (
                ("socket", "/World/RJ45Socket"),
                ("plug", "/World/RJ45Plug"),
            ):
                height = float(world_position(object_path)[2])
                if height > max_object_heights[object_name]:
                    max_object_heights[object_name] = height
                    max_object_height_phases[object_name] = str(phase)
        previous_command = positions
        previous_coupled = coupled_positions

        if phase != previous_phase:
            print(f"Auto-IK replay phase: {phase}", flush=True)
            previous_phase = phase
        grasp_hold_complete = phase == "grasp_hold" and (
            frame_index + 1 == frame_count or phases[frame_index + 1] != "grasp_hold"
        )
        if grasp_hold_complete and not grasp_checked:
            grasp_checked = True
            grasp_success = validate_grasp(frame_index)
            report_grasp_surfaces()
            if not grasp_success:
                raise RuntimeError("Physical grasp validation failed")
        hover_complete = phase == "hover_objects" and (
            frame_index + 1 == frame_count or phases[frame_index + 1] != "hover_objects"
        )
        if hover_complete and adaptive_cfg.get("rewrite_from_hover", True):
            rewrite_adaptive_trajectory("assembly_staging", "hover_objects")
        align_plug_tip_complete = phase == "align_plug_tip" and (
            frame_index + 1 == frame_count or phases[frame_index + 1] != "align_plug_tip"
        )
        if align_plug_tip_complete and adaptive_enabled:
            validate_preinsert_alignment("before correction 1", raise_on_failure=False)
            apply_lateral_residual_compensation("correction_pass_1")
            rewrite_adaptive_trajectory("alignment_correction_1", "correction_pass_1")
        correction_1_complete = phase == "alignment_correction_1" and (
            frame_index + 1 == frame_count
            or phases[frame_index + 1] != "alignment_correction_1"
        )
        if correction_1_complete and adaptive_enabled:
            correction_1_valid = validate_preinsert_alignment(
                "after correction 1", raise_on_failure=False
            )
            if correction_1_valid:
                print(
                    "Adaptive correction pass 2 skipped: correction pass 1 already passed",
                    flush=True,
                )
            else:
                apply_lateral_residual_compensation("correction_pass_2")
                rewrite_adaptive_trajectory(
                    "alignment_correction_2", "correction_pass_2"
                )
        correction_2_complete = phase == "alignment_correction_2" and (
            frame_index + 1 == frame_count
            or phases[frame_index + 1] != "alignment_correction_2"
        )
        if correction_2_complete and adaptive_enabled:
            validate_preinsert_alignment("after correction 2", raise_on_failure=False)
        aligned_hold_complete = phase == "aligned_hold" and (
            frame_index + 1 == frame_count or phases[frame_index + 1] != "aligned_hold"
        )
        if aligned_hold_complete:
            validate_preinsert_alignment("pre-insert", raise_on_failure=True)
        record_frame(actual_positions, positions)
    timeline.pause()
    print(
        f"Auto-IK replay complete: {frame_count} frames, "
        f"{physics_steps_per_frame} physics steps/frame, "
        f"max arm tracking error={max_arm_tracking_error:.4f} rad, "
        f"max hand tracking error={max_hand_tracking_error:.4f} rad from {episode_path}",
        flush=True,
    )
    final_object_points: list[list[float]] = []
    final_object_matrices = []
    for object_path, palm_path in (
        ("/World/RJ45Socket", f"{robot_model_root}/LH_palm_center"),
        ("/World/RJ45Plug", f"{robot_model_root}/RH_palm_center"),
    ):
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(object_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        translation = matrix.ExtractTranslation()
        final_object_matrices.append(matrix)
        final_object_points.append([float(translation[0]), float(translation[1]), float(translation[2])])
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        palm_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(palm_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        object_from_palm = matrix * palm_matrix.GetInverse()
        relative_translation = object_from_palm.ExtractTranslation()
        relative_rotation = object_from_palm.ExtractRotationQuat()
        relative_imaginary = relative_rotation.GetImaginary()
        print(
            f"Auto-IK hover pose {object_path}: "
            f"world_xyz={[round(float(value), 6) for value in translation]}, "
            f"world_quat_wxyz={[round(float(rotation.GetReal()), 7), *[round(float(value), 7) for value in imaginary]]}, "
            f"object_from_palm_xyz={[round(float(value), 6) for value in relative_translation]}, "
            f"object_from_palm_quat_wxyz={[round(float(relative_rotation.GetReal()), 7), *[round(float(value), 7) for value in relative_imaginary]]}",
            flush=True,
        )
    print(f"Auto-IK final object centers: {final_object_points}", flush=True)
    table_cfg = task_cfg["scene"]["table"]
    table_top_z = float(table_cfg["center_xyz"][2]) + 0.5 * float(
        table_cfg["size_xyz"][2]
    )
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    final_visual_min_z = {}
    for object_name, visual_path in (
        ("socket", "/World/RJ45Socket/visual"),
        ("plug", "/World/RJ45Plug/visual"),
    ):
        world_range = bbox_cache.ComputeWorldBound(
            stage.GetPrimAtPath(visual_path)
        ).ComputeAlignedRange()
        final_visual_min_z[object_name] = float(world_range.GetMin()[2])
    minimum_visual_z = min(final_visual_min_z.values())
    print(
        "Final table clearance: "
        f"socket={(final_visual_min_z['socket'] - table_top_z) * 1000.0:.2f} mm, "
        f"plug={(final_visual_min_z['plug'] - table_top_z) * 1000.0:.2f} mm",
        flush=True,
    )
    if minimum_visual_z < table_top_z - 0.001:
        raise RuntimeError(
            "Placed RJ45 assembly penetrates the table by "
            f"{(table_top_z - minimum_visual_z) * 1000.0:.2f} mm"
        )
    plug_from_socket = final_object_matrices[1] * final_object_matrices[0].GetInverse()
    relative_translation = plug_from_socket.ExtractTranslation()
    relative_rotation = plug_from_socket.ExtractRotationQuat()
    relative_imaginary = relative_rotation.GetImaginary()
    channel_entry_separation_m = 0.03658
    insertion_depth = channel_entry_separation_m + float(relative_translation[2])
    lateral_error = math.hypot(float(relative_translation[0]), float(relative_translation[1]))
    relative_x180_error = 2.0 * math.acos(
        min(1.0, max(0.0, abs(float(relative_imaginary[0]))))
    )
    result_label = (
        "Auto-IK final post-release relative pose"
        if "release_left_hand" in phases
        else "Auto-IK insertion result"
    )
    print(
        f"{result_label}: "
        f"plug_from_socket_xyz={[round(float(value), 6) for value in relative_translation]}, "
        f"quat_wxyz={[round(float(relative_rotation.GetReal()), 7), *[round(float(value), 7) for value in relative_imaginary]]}, "
        f"depth={insertion_depth * 1000.0:.1f} mm, "
        f"lateral_error={lateral_error * 1000.0:.1f} mm, "
        f"orientation_error={math.degrees(relative_x180_error):.1f} deg",
        flush=True,
    )
    print(
        "Auto-IK peak object lifts: "
        f"left={(max_object_heights['socket'] - initial_object_positions['socket'][2]) * 1000.0:.1f} mm "
        f"during {max_object_height_phases['socket']}, "
        f"right={(max_object_heights['plug'] - initial_object_positions['plug'][2]) * 1000.0:.1f} mm "
        f"during {max_object_height_phases['plug']}",
        flush=True,
    )
    pickup_cfg = task_cfg["pickup_auto_ik"]
    required_lift = float(pickup_cfg.get("required_physx_lift_m", 0.04))
    max_held_distance = float(pickup_cfg.get("max_held_object_palm_distance_m", 0.16))
    final_socket = np.asarray(final_object_points[0], dtype=float)
    final_plug = np.asarray(final_object_points[1], dtype=float)
    socket_lift = float(final_socket[2] - initial_object_positions["socket"][2])
    plug_lift = float(final_plug[2] - initial_object_positions["plug"][2])
    socket_hand_distance = float(
        np.linalg.norm(world_position(f"{robot_model_root}/LH_palm_center") - final_socket)
    )
    plug_hand_distance = float(
        np.linalg.norm(world_position(f"{robot_model_root}/RH_palm_center") - final_plug)
    )
    left_success = socket_lift >= required_lift and socket_hand_distance <= max_held_distance
    right_success = plug_lift >= required_lift and plug_hand_distance <= max_held_distance
    if "release_left_hand" in phases:
        release_mode = (
            "the verified internal socket latch remained engaged"
            if latch_engaged
            else "no post-insertion latch was engaged"
        )
        print(
            "PhysX release result: "
            f"socket final dz={socket_lift * 1000.0:.1f} mm, "
            f"plug final dz={plug_lift * 1000.0:.1f} mm; "
            f"both hands released and {release_mode}.",
            flush=True,
        )
    else:
        print(
            "PhysX-only pickup result: "
            f"left lift={socket_lift * 1000.0:.1f} mm held={left_success}, "
            f"right lift={plug_lift * 1000.0:.1f} mm held={right_success}; "
            "no attachment or kinematic object following was used.",
            flush=True,
        )
    if record_out_dir is not None:
        success_cfg = task_cfg["task"]["success"]
        latch_cfg = task_cfg.get("assembly_auto_ik", {}).get("post_insert_latch", {})
        final_arm_error = float(
            np.max(
                np.abs(
                    np.asarray(recorded_actual_joint_position[-1])[
                        np.asarray(list(range(0, 7)) + list(range(13, 20)), dtype=int)
                    ]
                    - np.asarray(recorded_target_joint_position[-1])[
                        np.asarray(list(range(0, 7)) + list(range(13, 20)), dtype=int)
                    ]
                )
            )
        )
        task_success = bool(
            grasp_success
            and latch_engaged
            and insertion_depth >= float(success_cfg["min_insert_depth_m"])
            and lateral_error <= float(success_cfg["max_lateral_error_m"])
            and relative_x180_error <= float(latch_cfg.get("max_orientation_error_rad", 0.20))
            and minimum_visual_z >= table_top_z - 0.001
        )
        if not task_success:
            raise RuntimeError(
                "Recorded episode failed final task validation: "
                f"grasp={grasp_success}, latch={latch_engaged}, "
                f"depth={insertion_depth:.6f}, lateral={lateral_error:.6f}, "
                f"orientation={relative_x180_error:.6f}"
            )

        frame_total = len(recorded_observation_state)
        if frame_total != frame_count:
            raise RuntimeError(f"Recorded {frame_total} frames for a {frame_count}-frame replay")
        for name, count in camera_frame_counts.items():
            if count != frame_total:
                raise RuntimeError(
                    f"Camera {name} recorded {count} frames, expected {frame_total}"
                )

        record_out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "observation_state": np.asarray(recorded_observation_state, dtype=np.float32),
            "target_observation_state": np.asarray(
                recorded_target_observation_state, dtype=np.float32
            ),
            "action": np.asarray(recorded_action, dtype=np.float32),
            "timestamp": np.arange(frame_total, dtype=np.float64) / float(record_hz),
            "phase": np.asarray(phases),
            "actual_joint_position": np.asarray(
                recorded_actual_joint_position, dtype=np.float32
            ),
            "target_joint_position": np.asarray(
                recorded_target_joint_position, dtype=np.float32
            ),
        }
        if record_format == "npz":
            arrays.update(
                {
                    "observation_images_chest": np.stack(
                        recorded_camera_frames["chest"], axis=0
                    ),
                    "observation_images_left_wrist": np.stack(
                        recorded_camera_frames["left_wrist"], axis=0
                    ),
                    "observation_images_right_wrist": np.stack(
                        recorded_camera_frames["right_wrist"], axis=0
                    ),
                }
            )
        if source_observation_state is not None:
            arrays["source_observation_state"] = source_observation_state
        if source_action is not None:
            arrays["source_action"] = source_action
        arrays["expert_mask"] = source_expert_mask
        if source_language_instruction is not None:
            arrays["language_instruction"] = source_language_instruction
        if record_format == "video":
            for writer in video_writers.values():
                writer.close(commit=True)
        np.savez_compressed(record_out_dir / "episode.npz", **arrays)

        source_metadata = {}
        source_metadata_path = episode_path.parent / "metadata.json"
        if source_metadata_path.exists():
            with source_metadata_path.open("r", encoding="utf-8") as handle:
                source_metadata = json.load(handle)
        metadata = {
            "schema": (
                "qiling_groot_assemble.isaac_recorded_rj45.v2"
                if record_format == "video"
                else "qiling_groot_assemble.isaac_recorded_rj45.v1"
            ),
            "status": "PASS",
            "source_episode": str(episode_path),
            "frame_count": frame_total,
            "physics_hz": physics_hz,
            "record_hz": float(record_hz),
            "physics_steps_per_frame": physics_steps_per_frame,
            "state_dim": 34,
            "action_dim": 20,
            "camera_names": ["chest", "left_wrist", "right_wrist"],
            "camera_resolution": [640, 480],
            "camera_storage": record_format,
            "camera_files": (
                {name: f"videos/{name}.mp4" for name in camera_frame_counts}
                if record_format == "video"
                else {}
            ),
            "camera_codec": "h264" if record_format == "video" else None,
            "camera_frame_counts": camera_frame_counts,
            "grasp_success": grasp_success,
            "latch_engaged": latch_engaged,
            "insertion_depth_m": insertion_depth,
            "lateral_error_m": lateral_error,
            "orientation_error_rad": relative_x180_error,
            "final_table_clearance_m": {
                name: value - table_top_z for name, value in final_visual_min_z.items()
            },
            "max_arm_tracking_error_rad": max_arm_tracking_error,
            "max_hand_tracking_error_rad": max_hand_tracking_error,
            "final_arm_tracking_error_rad": final_arm_error,
            "source_metadata": source_metadata,
        }
        with (record_out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            f"Recorded synchronized RJ45 episode: {record_out_dir} "
            f"frames={frame_total}, cameras=3, status=PASS",
            flush=True,
        )
    if final_capture_path is not None:
        from isaacsim.sensors.camera import Camera
        from PIL import Image

        overview = Camera(
            prim_path="/World/OverviewCamera",
            name="rj45_final_overview_capture",
            resolution=(1280, 720),
        )
        overview.initialize()
        for _ in range(12):
            app.update()
        rgba = np.asarray(overview.get_rgba())
        final_capture_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(final_capture_path)
        print(f"Saved final overview: {final_capture_path}", flush=True)
    return final_object_points


def smoke_test_camera_frames(
    app,
    camera_paths: list[str],
    object_points,
    camera_cfg: dict,
    wrist_require_both: bool = True,
    required_object_indices: dict[str, list[int]] | None = None,
) -> None:
    import numpy as np
    import omni.timeline
    from isaacsim.sensors.camera import Camera

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    camera_names = ("chest", "left_wrist", "right_wrist")
    camera_settings = [camera_cfg["cameras"][name] for name in camera_names]
    if len(camera_paths) != len(camera_settings):
        raise RuntimeError("Camera path and configuration counts do not match")

    sensors = []
    for index, (camera_path, settings) in enumerate(zip(camera_paths, camera_settings)):
        width = int(settings["width"])
        height = int(settings["height"])
        sensor = Camera(
            prim_path=camera_path,
            name=f"rj45_camera_smoke_{index}",
            resolution=(width, height),
        )
        sensor.initialize()
        sensors.append((sensor, width, height))

    for _ in range(12):
        app.update()

    for name, camera_path, (sensor, width, height) in zip(
        camera_names, camera_paths, sensors
    ):
        rgba = np.asarray(sensor.get_rgba())
        if rgba.shape != (height, width, 4):
            raise RuntimeError(f"Camera {camera_path} returned invalid frame shape {rgba.shape}")
        rgb = rgba[:, :, :3].astype(np.float32)
        print(
            f"Camera frame OK: {camera_path} "
            f"shape={rgba.shape} rgb_min={rgb.min():.0f} rgb_max={rgb.max():.0f} rgb_std={rgb.std():.2f}",
            flush=True,
        )
        if float(rgb.max()) <= float(rgb.min()):
            raise RuntimeError(f"Camera {camera_path} returned a constant/blank RGB frame")
        pixels = np.asarray(sensor.get_image_coords_from_world_points(np.asarray(object_points, dtype=float)))
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        print(f"  task asset pixels={np.round(pixels, 1).tolist()} inside={inside.tolist()}", flush=True)
        if required_object_indices is not None:
            required = required_object_indices.get(name, [])
            if required and not bool(np.all(inside[np.asarray(required, dtype=int)])):
                raise RuntimeError(
                    f"Camera {name} does not contain required task points {required}"
                )
            continue
        if "left_wrist_camera" in camera_path:
            valid = bool(np.all(inside)) if wrist_require_both else bool(inside[0])
            if not valid:
                raise RuntimeError(f"Left wrist camera {camera_path} does not contain its required task assets")
        if "right_wrist_camera" in camera_path:
            valid = bool(np.all(inside)) if wrist_require_both else bool(inside[1])
            if not valid:
                raise RuntimeError(f"Right wrist camera {camera_path} does not contain its required task assets")
    timeline.pause()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0, help="Close automatically after N seconds; 0 means wait.")
    parser.add_argument("--robot-config", default="configs/robot_dual_arm.yaml")
    parser.add_argument("--task-config", default="configs/task_rj45_insertion.yaml")
    parser.add_argument("--camera-config", default="configs/camera_bimanual.yaml")
    parser.add_argument("--stl-scale", type=float, default=0.001)
    parser.add_argument("--camera-smoke-test", action="store_true", help="Render and validate all sensor cameras.")
    parser.add_argument("--replay-pickup-episode", default=None, help="Replay a generated pickup episode after home.")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--max-replay-frames", type=int, default=None, help="Stop replay after this many frames.")
    parser.add_argument("--grasp-capture", default=None, help="Save an overview RGB image at the end of grasp_hold.")
    parser.add_argument("--final-capture", default=None, help="Save an overview RGB image after replay completes.")
    parser.add_argument(
        "--record-out-dir",
        default=None,
        help="Record actual state/action and synchronized chest/left/right RGB into this episode directory.",
    )
    parser.add_argument(
        "--record-format",
        choices=("npz", "video"),
        default="npz",
        help="Store RGB inside episode.npz or stream each camera to H.264 MP4.",
    )
    parser.add_argument("--export-stage", default=None, help="Export the constructed USD stage before replay.")
    parser.add_argument("--calibrate-pinches", action="store_true", help="Calibrate both O6 pinch presets in Isaac.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = SimulationApp({"headless": bool(args.headless), "width": 1280, "height": 720})
    try:
        robot_cfg = load_yaml(ROOT / args.robot_config)
        task_cfg = load_yaml(ROOT / args.task_config)
        camera_cfg = load_yaml(ROOT / args.camera_config)
        if args.replay_pickup_episode:
            replay_path = ROOT / args.replay_pickup_episode
            source_metadata_path = replay_path.parent / "metadata.json"
            if source_metadata_path.exists():
                with source_metadata_path.open("r", encoding="utf-8") as handle:
                    source_metadata = json.load(handle)
                socket_xyz = source_metadata.get("socket_box_world_xyz")
                plug_xyz = source_metadata.get("rj45_plug_world_xyz")
                if socket_xyz is not None and plug_xyz is not None:
                    task_cfg["scene"]["socket_box_initial_xyz"] = socket_xyz
                    task_cfg["scene"]["rj45_plug_initial_xyz"] = plug_xyz
                    print(
                        "Applied episode object randomization to Isaac scene: "
                        f"socket={socket_xyz}, plug={plug_xyz}",
                        flush=True,
                    )
        try:
            camera_paths, object_points, articulation_root, robot_model_root = build_scene(
                app,
                robot_cfg,
                task_cfg,
                camera_cfg,
                args.stl_scale,
                startup_realtime=not args.headless,
            )
        except Exception as exc:
            print(f"Scene construction failed: {type(exc).__name__}: {exc}", flush=True)
            raise
        if args.export_stage:
            import omni.usd

            export_path = ROOT / args.export_stage
            export_path.parent.mkdir(parents=True, exist_ok=True)
            omni.usd.get_context().get_stage().Export(str(export_path))
            print(f"Exported stage: {export_path}", flush=True)
        if args.calibrate_pinches:
            calibrate_o6_pinches(
                app, articulation_root, robot_model_root, robot_cfg, task_cfg
            )
            return 0
        if args.replay_pickup_episode:
            try:
                object_points = replay_pickup_episode(
                    app,
                    ROOT / args.replay_pickup_episode,
                    articulation_root,
                    robot_model_root,
                    robot_cfg,
                    task_cfg,
                    float(task_cfg["simulation"]["record_hz"]),
                    args.replay_speed,
                    realtime=not args.headless,
                    max_replay_frames=args.max_replay_frames,
                    grasp_capture_path=ROOT / args.grasp_capture if args.grasp_capture else None,
                    final_capture_path=ROOT / args.final_capture if args.final_capture else None,
                    camera_paths=camera_paths,
                    camera_cfg=camera_cfg,
                    record_out_dir=ROOT / args.record_out_dir if args.record_out_dir else None,
                    record_format=args.record_format,
                )
            except Exception as exc:
                print(f"Episode replay failed: {type(exc).__name__}: {exc}", flush=True)
                raise
        if args.camera_smoke_test:
            smoke_test_camera_frames(
                app,
                camera_paths,
                object_points,
                camera_cfg,
                wrist_require_both=not bool(args.replay_pickup_episode),
            )

        if args.headless and (
            args.replay_pickup_episode
            or args.camera_smoke_test
            or args.export_stage
            or args.record_out_dir
        ):
            return 0

        start = time.monotonic()
        while app.is_running():
            app.update()
            if args.seconds > 0.0 and time.monotonic() - start >= args.seconds:
                break
            time.sleep(1.0 / 60.0)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
