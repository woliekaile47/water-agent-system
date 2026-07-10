#!/usr/bin/env python3
"""Project a camera-space binary water mask onto a dry ground DEM.

Prediction inputs are deliberately allow-listed here.  This module never
opens a water-case manifest, DEM Ground Truth mask, depth Ground Truth, or
Ground Truth metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PREDICTION_SOURCE = "camera_mask_ground_dem_boundary_inversion"
ALGORITHM_VERSION = "phase2a_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_prediction_inputs(
    project_root: str | Path,
    case_id: str,
    projection_config: dict[str, Any],
) -> dict[str, Any]:
    """Load only allow-listed prediction inputs.

    Paths are constructed from configured case ids instead of being obtained
    from a water-case manifest, because that manifest also contains answers.
    """
    root = Path(project_root).expanduser().resolve()
    dry_case_id = str(projection_config["dry_case_id"])
    dry_dem_path = root / "data" / "simulation" / dry_case_id / "ground_truth" / "ground_dem_gt.npy"
    camera_mask_path = root / "data" / "simulation" / case_id / "ground_truth" / "camera_water_mask_gt.png"
    sensors_path = root / str(projection_config.get("sensors_config", "simulation/config/sensors.yaml"))
    for path in (dry_dem_path, camera_mask_path, sensors_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction input: {path}")

    ground_dem = np.load(dry_dem_path).astype(np.float32)
    camera_mask = np.asarray(Image.open(camera_mask_path).convert("L"), dtype=np.uint8)
    sensors_document = _load_yaml(sensors_path)
    sensors = sensors_document
    if "road" not in sensors or "camera" not in sensors:
        raise ValueError(f"Invalid sensors configuration: {sensors_path}")

    return {
        "ground_dem": ground_dem,
        "camera_mask": camera_mask,
        "sensors": sensors,
        "paths": {
            "dry_ground_dem": str(dry_dem_path),
            "camera_water_mask": str(camera_mask_path),
            "sensors_config": str(sensors_path),
        },
        "prediction_inputs": [
            "dry_ground_dem",
            "camera_water_mask_gt_as_phase2a_input",
            "camera_intrinsics_from_sensors_config",
            "T_camera_optical_map_from_sensors_config",
            "dem_grid_geometry_from_sensors_config",
        ],
    }


def dem_cell_centres(shape: tuple[int, int], sensors: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    road = sensors["road"]
    resolution = float(road["dem_resolution_m"])
    length = float(road["length_m"])
    width = float(road["width_m"])
    expected = (int(round(width / resolution)), int(round(length / resolution)))
    if tuple(shape) != expected:
        raise ValueError(f"DEM shape {shape} does not match configured grid {expected}")
    xs = -length / 2.0 + resolution * (np.arange(shape[1], dtype=np.float64) + 0.5)
    ys = -width / 2.0 + resolution * (np.arange(shape[0], dtype=np.float64) + 0.5)
    return np.meshgrid(xs, ys)


def camera_model(sensors: dict[str, Any]) -> dict[str, Any]:
    camera = sensors["camera"]
    rig = sensors["sensor_rig"]["pose_map"]
    pose = camera["pose_on_rig"]
    width = int(camera["width_px"])
    height = int(camera["height_px"])
    fx = width / (2.0 * math.tan(math.radians(float(camera["horizontal_fov_deg"])) / 2.0))
    world_pose = {
        "x_m": float(rig["x_m"]) + float(pose["x_m"]),
        "y_m": float(rig["y_m"]) + float(pose["y_m"]),
        "z_m": float(rig["z_m"]) + float(pose["z_m"]),
        "roll_deg": float(pose.get("roll_deg", 0.0)),
        "pitch_down_deg": float(pose["pitch_down_deg"]),
        "yaw_deg": float(pose["yaw_deg"]),
    }
    if abs(world_pose["roll_deg"]) > 1e-9:
        raise ValueError("Phase 2A camera model currently requires roll_deg=0")
    return {
        "frame_id": sensors["coordinate_frames"]["camera_optical"],
        "map_frame_id": sensors["coordinate_frames"]["map"],
        "width_px": width,
        "height_px": height,
        "fx": fx,
        "fy": fx,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "near_m": float(camera["near_clip_m"]),
        "far_m": float(camera["far_clip_m"]),
        "world_pose": world_pose,
        "optical_axes": {"x": "right", "y": "down", "z": "forward"},
    }


def map_points_to_camera_optical(points_map: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Apply T_camera_optical_map using the Phase 1A optical-axis convention."""
    pose = model["world_pose"]
    yaw = math.radians(float(pose["yaw_deg"]))
    pitch = math.radians(float(pose["pitch_down_deg"]))
    forward = np.asarray(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    down = np.cross(forward, right)
    origin = np.asarray([pose["x_m"], pose["y_m"], pose["z_m"]], dtype=np.float64)
    delta = np.asarray(points_map, dtype=np.float64) - origin
    return np.column_stack((delta @ right, delta @ down, delta @ forward))


def project_camera_mask_to_dem(
    ground_dem: np.ndarray,
    camera_mask: np.ndarray,
    sensors: dict[str, Any],
    mask_threshold: int = 127,
) -> tuple[np.ndarray, dict[str, Any]]:
    if ground_dem.ndim != 2 or camera_mask.ndim != 2:
        raise ValueError("ground_dem and camera_mask must both be 2D")
    model = camera_model(sensors)
    if camera_mask.shape != (model["height_px"], model["width_px"]):
        raise ValueError(
            f"Camera mask shape {camera_mask.shape} does not match CameraInfo "
            f"{(model['height_px'], model['width_px'])}"
        )
    xx, yy = dem_cell_centres(tuple(ground_dem.shape), sensors)
    finite = np.isfinite(ground_dem)
    points = np.column_stack((xx.ravel(), yy.ravel(), ground_dem.astype(np.float64).ravel()))
    optical = map_points_to_camera_optical(points, model)
    x_cam, y_cam, z_cam = optical.T
    finite_flat = finite.ravel()
    behind_or_near = finite_flat & (z_cam <= model["near_m"])
    beyond_far = finite_flat & (z_cam > model["far_m"])
    range_valid = finite_flat & ~behind_or_near & ~beyond_far

    u = np.full(z_cam.shape, np.nan, dtype=np.float64)
    v = np.full(z_cam.shape, np.nan, dtype=np.float64)
    u[range_valid] = model["fx"] * x_cam[range_valid] / z_cam[range_valid] + model["cx"]
    v[range_valid] = model["fy"] * y_cam[range_valid] / z_cam[range_valid] + model["cy"]
    inside_image = range_valid & (u >= 0.0) & (u < model["width_px"]) & (v >= 0.0) & (v < model["height_px"])
    rows = np.rint(v[inside_image]).astype(np.int64)
    cols = np.rint(u[inside_image]).astype(np.int64)
    rows = np.clip(rows, 0, model["height_px"] - 1)
    cols = np.clip(cols, 0, model["width_px"] - 1)

    predicted_flat = np.zeros(ground_dem.size, dtype=bool)
    projected_indices = np.flatnonzero(inside_image)
    predicted_flat[projected_indices] = camera_mask[rows, cols] > int(mask_threshold)
    predicted = predicted_flat.reshape(ground_dem.shape)
    valid_dem_count = int(np.count_nonzero(finite_flat))
    projected_count = int(projected_indices.size)
    water_pixels = camera_mask > int(mask_threshold)
    sampled_unique = np.unique(rows.astype(np.int64) * model["width_px"] + cols.astype(np.int64))
    water_sampled = int(np.count_nonzero(camera_mask[rows, cols] > int(mask_threshold))) if rows.size else 0
    diagnostics = {
        "data_role": "prediction_diagnostics",
        "source": PREDICTION_SOURCE,
        "algorithm_version": ALGORITHM_VERSION,
        "camera_frame": model["frame_id"],
        "map_frame": model["map_frame_id"],
        "transform": "T_camera_optical_map",
        "optical_axes": model["optical_axes"],
        "dem_cell_count": int(ground_dem.size),
        "valid_dem_cell_count": valid_dem_count,
        "projected_dem_cell_count": projected_count,
        "projection_coverage": float(projected_count / max(1, valid_dem_count)),
        "predicted_water_cell_count_before_cleanup": int(np.count_nonzero(predicted)),
        "camera_water_pixel_count": int(np.count_nonzero(water_pixels)),
        "sampled_camera_pixel_count_unique": int(sampled_unique.size),
        "sampled_water_dem_cell_count": water_sampled,
        "invalid_reasons": {
            "nonfinite_dem": int(np.count_nonzero(~finite_flat)),
            "behind_or_near_camera": int(np.count_nonzero(behind_or_near)),
            "beyond_far_clip": int(np.count_nonzero(beyond_far)),
            "outside_image": int(np.count_nonzero(range_valid & ~inside_image)),
        },
        "camera_model": model,
    }
    return predicted, diagnostics
