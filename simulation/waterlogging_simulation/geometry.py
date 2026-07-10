"""Deterministic road geometry and Ground Truth calculations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def ground_elevation(x: np.ndarray, y: np.ndarray, sensors: dict[str, Any]) -> np.ndarray:
    road = sensors["road"]
    basin = sensors["basin"]
    base = float(road["longitudinal_slope"]) * x + float(road["cross_slope"]) * y
    rx = (x - float(basin["center_x_m"])) / float(basin["half_length_m"])
    ry = (y - float(basin["center_y_m"])) / float(basin["half_width_m"])
    radius2 = rx * rx + ry * ry
    inside = radius2 < 1.0
    depression = np.zeros_like(base, dtype=np.float64)
    exponent = float(basin["shape_exponent"])
    depression[inside] = float(basin["depth_m"]) * np.power(1.0 - radius2[inside], exponent)
    return (base - depression).astype(np.float32)


def dem_grid(sensors: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    road = sensors["road"]
    resolution = float(road["dem_resolution_m"])
    length = float(road["length_m"])
    width = float(road["width_m"])
    nx = int(round(length / resolution))
    ny = int(round(width / resolution))
    xs = -length / 2.0 + resolution * (np.arange(nx, dtype=np.float64) + 0.5)
    ys = -width / 2.0 + resolution * (np.arange(ny, dtype=np.float64) + 0.5)
    xx, yy = np.meshgrid(xs, ys)
    return xx.astype(np.float32), yy.astype(np.float32), ground_elevation(xx, yy, sensors)


def water_ground_truth(
    ground_dem: np.ndarray,
    scenario: dict[str, Any],
    sensors: dict[str, Any],
) -> tuple[float | None, np.ndarray, np.ndarray, float, float]:
    depth_cm = int(scenario["water_depth_cm"])
    if depth_cm == 0:
        mask = np.zeros_like(ground_dem, dtype=bool)
        depth = np.zeros_like(ground_dem, dtype=np.float32)
        return None, mask, depth, 0.0, 0.0

    water_level_m = float(np.min(ground_dem)) + depth_cm / 100.0
    depth = np.maximum(water_level_m - ground_dem.astype(np.float64), 0.0).astype(np.float32)
    mask = depth > 0.0
    resolution = float(sensors["road"]["dem_resolution_m"])
    cell_area_m2 = resolution * resolution
    area_m2 = float(np.count_nonzero(mask) * cell_area_m2)
    volume_m3 = float(np.sum(depth.astype(np.float64)) * cell_area_m2)
    return water_level_m, mask, depth, area_m2, volume_m3


def camera_intrinsics(sensors: dict[str, Any]) -> dict[str, Any]:
    camera = sensors["camera"]
    width = int(camera["width_px"])
    height = int(camera["height_px"])
    hfov_rad = math.radians(float(camera["horizontal_fov_deg"]))
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    return {
        "width_px": width,
        "height_px": height,
        "horizontal_fov_deg": float(camera["horizontal_fov_deg"]),
        "fx": fx,
        "fy": fx,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def camera_world_pose(sensors: dict[str, Any]) -> dict[str, float]:
    rig = sensors["sensor_rig"]["pose_map"]
    camera = sensors["camera"]["pose_on_rig"]
    return {
        "x_m": float(rig["x_m"]) + float(camera["x_m"]),
        "y_m": float(rig["y_m"]) + float(camera["y_m"]),
        "z_m": float(rig["z_m"]) + float(camera["z_m"]),
        "roll_deg": float(camera["roll_deg"]),
        "pitch_down_deg": float(camera["pitch_down_deg"]),
        "yaw_deg": float(camera["yaw_deg"]),
    }


def _project_points(points_xyz: np.ndarray, sensors: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = camera_intrinsics(sensors)
    pose = camera_world_pose(sensors)
    yaw = math.radians(pose["yaw_deg"])
    pitch = math.radians(pose["pitch_down_deg"])
    forward = np.asarray(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    down = np.cross(forward, right)
    origin = np.asarray([pose["x_m"], pose["y_m"], pose["z_m"]], dtype=np.float64)
    delta = points_xyz.astype(np.float64) - origin
    z_cam = delta @ forward
    x_cam = delta @ right
    y_cam = delta @ down
    valid = z_cam > float(sensors["camera"]["near_clip_m"])
    uv = np.zeros((points_xyz.shape[0], 2), dtype=np.float64)
    safe_z = np.where(valid, z_cam, 1.0)
    uv[:, 0] = intrinsics["fx"] * x_cam / safe_z + intrinsics["cx"]
    uv[:, 1] = intrinsics["fy"] * y_cam / safe_z + intrinsics["cy"]
    return uv, valid


def camera_water_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    dem_mask: np.ndarray,
    water_level_m: float | None,
    sensors: dict[str, Any],
) -> np.ndarray:
    intrinsics = camera_intrinsics(sensors)
    width = int(intrinsics["width_px"])
    height = int(intrinsics["height_px"])
    image = Image.new("L", (width, height), 0)
    if water_level_m is None or not np.any(dem_mask):
        return np.asarray(image, dtype=np.uint8)

    resolution = float(sensors["road"]["dem_resolution_m"])
    draw = ImageDraw.Draw(image)
    rows, cols = np.where(dem_mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        x = float(xx[row, col])
        y = float(yy[row, col])
        half = resolution / 2.0
        corners = np.asarray(
            [
                [x - half, y - half, water_level_m],
                [x + half, y - half, water_level_m],
                [x + half, y + half, water_level_m],
                [x - half, y + half, water_level_m],
            ],
            dtype=np.float64,
        )
        uv, valid = _project_points(corners, sensors)
        if not np.all(valid):
            continue
        polygon = [(int(round(u)), int(round(v))) for u, v in uv]
        draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8)


def write_road_obj(path: str | Path, sensors: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    road = sensors["road"]
    resolution = float(road["mesh_resolution_m"])
    length = float(road["length_m"])
    width = float(road["width_m"])
    nx = int(round(length / resolution)) + 1
    ny = int(round(width / resolution)) + 1
    xs = np.linspace(-length / 2.0, length / 2.0, nx)
    ys = np.linspace(-width / 2.0, width / 2.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = ground_elevation(xx, yy, sensors)
    dz_dy, dz_dx = np.gradient(zz.astype(np.float64), resolution, resolution)
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(zz, dtype=np.float64)), axis=-1)
    normal_lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-12)

    lines = [
        "# Deterministic road basin mesh generated from sensors.yaml",
        "mtllib road_basin.mtl",
        "o road_basin",
        "usemtl road_asphalt",
        "s 1",
    ]
    for x, y, z in zip(xx.ravel(), yy.ravel(), zz.ravel()):
        lines.append(f"v {float(x):.6f} {float(y):.6f} {float(z):.6f}")
    for nx_value, ny_value, nz_value in normals.reshape(-1, 3):
        lines.append(
            f"vn {float(nx_value):.8f} {float(ny_value):.8f} {float(nz_value):.8f}"
        )
    for row in range(ny - 1):
        for col in range(nx - 1):
            a = row * nx + col + 1
            b = a + 1
            c = a + nx
            d = c + 1
            lines.append(f"f {a}//{a} {b}//{b} {d}//{d}")
            lines.append(f"f {a}//{a} {d}//{d} {c}//{c}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    material_path = output.with_name("road_basin.mtl")
    material_path.write_text(
        "\n".join(
            [
                "newmtl road_asphalt",
                "Ka 0.18 0.20 0.22",
                "Kd 0.18 0.20 0.22",
                "Ks 0.15 0.15 0.15",
                "Ns 16.0",
                "d 1.0",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    validate_road_obj(output)
    return output


def validate_road_obj(path: str | Path) -> dict[str, int]:
    """Reject empty, out-of-range, degenerate, or inconsistently-normaled OBJ meshes."""
    obj_path = Path(path).expanduser().resolve()
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    faces: list[tuple[list[int], list[int]]] = []
    for line_number, raw_line in enumerate(obj_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("v "):
            values = line.split()
            if len(values) != 4:
                raise ValueError(f"Invalid vertex at line {line_number}: {raw_line}")
            vertices.append(np.asarray([float(value) for value in values[1:]], dtype=np.float64))
        elif line.startswith("vn "):
            values = line.split()
            if len(values) != 4:
                raise ValueError(f"Invalid normal at line {line_number}: {raw_line}")
            normal = np.asarray([float(value) for value in values[1:]], dtype=np.float64)
            if not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12:
                raise ValueError(f"Invalid zero/non-finite normal at line {line_number}")
            normals.append(normal)
        elif line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) != 3:
                raise ValueError(f"Face must be a triangle at line {line_number}: {raw_line}")
            vertex_indices: list[int] = []
            normal_indices: list[int] = []
            for token in tokens:
                parts = token.split("/")
                if len(parts) != 3 or not parts[0] or not parts[2]:
                    raise ValueError(f"Face must reference vertex and normal at line {line_number}")
                vertex_indices.append(int(parts[0]))
                normal_indices.append(int(parts[2]))
            faces.append((vertex_indices, normal_indices))

    if not vertices:
        raise ValueError("OBJ has no vertices")
    if not faces:
        raise ValueError("OBJ has no triangle faces")
    if len(normals) != len(vertices):
        raise ValueError(
            f"OBJ normal count {len(normals)} does not match vertex count {len(vertices)}"
        )

    vertex_count = len(vertices)
    normal_count = len(normals)
    for face_number, (vertex_indices, normal_indices) in enumerate(faces, 1):
        if any(index < 1 or index > vertex_count for index in vertex_indices):
            raise ValueError(f"OBJ face {face_number} has an out-of-range vertex index")
        if any(index < 1 or index > normal_count for index in normal_indices):
            raise ValueError(f"OBJ face {face_number} has an out-of-range normal index")
        triangle = [vertices[index - 1] for index in vertex_indices]
        twice_area = np.linalg.norm(
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        )
        if not np.isfinite(twice_area) or twice_area <= 1e-12:
            raise ValueError(f"OBJ face {face_number} is degenerate")

    return {
        "vertex_count": vertex_count,
        "normal_count": normal_count,
        "face_count": len(faces),
    }
