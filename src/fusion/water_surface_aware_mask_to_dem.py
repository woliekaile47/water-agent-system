#!/usr/bin/env python3
"""Water-surface-aware Camera mask geometry for Phase 2B.

This prediction module only consumes the allow-listed Phase 2A inputs. It
does not load simulation water levels, DEM Ground Truth masks, or depth maps.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.fusion.project_camera_mask_to_dem import (
    camera_model,
    dem_cell_centres,
    map_points_to_camera_optical,
)
from src.hydrology.estimate_water_level_from_boundary import connected_components
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


PREDICTION_SOURCE = "water_surface_aware_camera_mask_dem_inversion"
ALGORITHM_VERSION = "phase2b_v1"


def camera_ray_map(u: float, v: float, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return Camera origin and a unit pixel ray in map coordinates."""
    x_optical = (float(u) - float(model["cx"])) / float(model["fx"])
    y_optical = (float(v) - float(model["cy"])) / float(model["fy"])
    pose = model["world_pose"]
    yaw = math.radians(float(pose["yaw_deg"]))
    pitch = math.radians(float(pose["pitch_down_deg"]))
    forward = np.asarray(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    down = np.cross(forward, right)
    direction = right * x_optical + down * y_optical + forward
    direction /= np.linalg.norm(direction)
    origin = np.asarray([pose["x_m"], pose["y_m"], pose["z_m"]], dtype=np.float64)
    return origin, direction


def bilinear_dem_height(
    x_m: float,
    y_m: float,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
) -> float | None:
    dem = np.asarray(ground_dem, dtype=np.float64)
    road = sensors["road"]
    resolution = float(road["dem_resolution_m"])
    x0 = -float(road["length_m"]) / 2.0 + resolution / 2.0
    y0 = -float(road["width_m"]) / 2.0 + resolution / 2.0
    col = (float(x_m) - x0) / resolution
    row = (float(y_m) - y0) / resolution
    if col < 0.0 or row < 0.0 or col > dem.shape[1] - 1 or row > dem.shape[0] - 1:
        return None
    col0 = int(np.floor(col))
    row0 = int(np.floor(row))
    col1 = min(col0 + 1, dem.shape[1] - 1)
    row1 = min(row0 + 1, dem.shape[0] - 1)
    values = np.asarray([dem[row0, col0], dem[row0, col1], dem[row1, col0], dem[row1, col1]])
    if not np.isfinite(values).all():
        return None
    tx = col - col0
    ty = row - row0
    top = values[0] * (1.0 - tx) + values[1] * tx
    bottom = values[2] * (1.0 - tx) + values[3] * tx
    return float(top * (1.0 - ty) + bottom * ty)


def intersect_ray_with_dem(
    origin: np.ndarray,
    direction: np.ndarray,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, float] | None, str]:
    """March through the DEM, then bisect the first above-to-below crossing."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(direction).all():
        return None, "invalid_ray"
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None, "invalid_ray"
    direction = direction / norm
    step = float(config["ray_step_m"])
    near = float(config.get("ray_min_m", 0.1))
    far = float(config["ray_max_m"])
    previous: tuple[float, float] | None = None
    saw_dem = False
    for t in np.arange(near, far + step * 0.5, step, dtype=np.float64):
        point = origin + direction * t
        height = bilinear_dem_height(point[0], point[1], ground_dem, sensors)
        if height is None:
            previous = None
            continue
        saw_dem = True
        residual = float(point[2] - height)
        if previous is None:
            if residual <= 0.0:
                return None, "entered_dem_below_surface"
            previous = (float(t), residual)
            continue
        if previous[1] > 0.0 and residual <= 0.0:
            low, high = previous[0], float(t)
            for _ in range(int(config.get("bisection_iterations", 18))):
                middle = (low + high) / 2.0
                middle_point = origin + direction * middle
                middle_height = bilinear_dem_height(middle_point[0], middle_point[1], ground_dem, sensors)
                if middle_height is None:
                    return None, "bisection_left_dem_bounds"
                middle_residual = float(middle_point[2] - middle_height)
                if middle_residual > 0.0:
                    low = middle
                else:
                    high = middle
            distance = (low + high) / 2.0
            intersection = origin + direction * distance
            dem_height = bilinear_dem_height(intersection[0], intersection[1], ground_dem, sensors)
            if dem_height is None:
                return None, "intersection_outside_dem"
            return {
                "x_m": float(intersection[0]),
                "y_m": float(intersection[1]),
                "z_ray_m": float(intersection[2]),
                "dem_height_m": float(dem_height),
                "ray_distance_m": float(distance),
                "residual_m": float(intersection[2] - dem_height),
            }, "success"
        previous = (float(t), residual)
    return None, "no_surface_crossing" if saw_dem else "ray_missed_dem_bounds"


def intersect_camera_shoreline(
    camera_mask: np.ndarray,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = np.asarray(camera_mask) > int(config.get("mask_threshold", 127))
    model = camera_model(sensors)
    components = connected_components(mask, int(config.get("connectivity", 8)))
    minimum_pixels = int(config.get("min_image_component_pixels", 10))
    stride = max(1, int(config.get("shoreline_sample_stride_px", 2)))
    records: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    sampled_count = 0
    boundary_count = 0
    retained_components = 0
    for component_index, component in enumerate(components):
        if int(np.count_nonzero(component)) < minimum_pixels:
            continue
        retained_components += 1
        # The geometric binary-mask shoreline lies between an inside pixel and
        # an adjacent outside pixel. Casting from the inside pixel centre
        # biases intersections toward lower ground, especially on shallow
        # slopes. Use deterministic half-pixel interface points instead.
        interface_points: set[tuple[float, float]] = set()
        height, width = component.shape
        for row, col in zip(*np.where(component)):
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row, neighbor_col = int(row + drow), int(col + dcol)
                outside = not (0 <= neighbor_row < height and 0 <= neighbor_col < width)
                if outside or not component[neighbor_row, neighbor_col]:
                    interface_points.add((float(row) + 0.5 * drow, float(col) + 0.5 * dcol))
        pixels = sorted(interface_points)
        boundary_count += len(pixels)
        for row, col in pixels[::stride]:
            sampled_count += 1
            origin, direction = camera_ray_map(float(col), float(row), model)
            hit, reason = intersect_ray_with_dem(origin, direction, ground_dem, sensors, config)
            if hit is None:
                failure_reasons[reason] += 1
                continue
            records.append({
                "component_index": int(component_index),
                "pixel_u": float(col),
                "pixel_v": float(row),
                "ray_direction_map": direction.tolist(),
                **hit,
            })
    edge_band = max(1, int(config.get("image_edge_band_px", 2)))
    image_boundary = extract_boundary_mask(mask)
    rows, cols = np.where(image_boundary)
    touches_edge = (
        (rows < edge_band)
        | (rows >= mask.shape[0] - edge_band)
        | (cols < edge_band)
        | (cols >= mask.shape[1] - edge_band)
    )
    diagnostics = {
        "image_component_count": len(components),
        "retained_image_component_count": retained_components,
        "image_shoreline_pixel_count": boundary_count,
        "sampled_shoreline_ray_count": sampled_count,
        "successful_intersection_count": len(records),
        "shoreline_intersection_success_rate": float(len(records) / max(1, sampled_count)),
        "intersection_failure_reasons": dict(sorted(failure_reasons.items())),
        "camera_mask_edge_touch_ratio": float(np.count_nonzero(touches_edge) / max(1, rows.size)),
        "camera_frame": model["frame_id"],
        "map_frame": model["map_frame_id"],
        "transform": "T_map_camera_optical",
    }
    return records, diagnostics


def reconstruct_connected_lowland(
    ground_dem: np.ndarray,
    estimated_water_level_m: float,
    seed_mask: np.ndarray,
    config: dict[str, Any],
    observed_camera_mask: np.ndarray | None = None,
    sensors: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    dem = np.asarray(ground_dem, dtype=np.float64)
    seeds = np.asarray(seed_mask, dtype=bool)
    if dem.shape != seeds.shape:
        raise ValueError("ground_dem and seed_mask shapes differ")
    candidate = np.isfinite(dem) & (dem < float(estimated_water_level_m) - float(config.get("lowland_margin_m", 0.0)))
    valid_seeds = seeds & candidate
    components = connected_components(candidate, int(config.get("connectivity", 8)))
    minimum_seed_cells = int(config.get("min_seed_cells_per_basin", 1))
    selected: list[np.ndarray] = []
    overlaps: list[int] = []
    camera_support: list[dict[str, Any]] = []
    ambiguous_indices: list[int] = []
    unobserved_indices: list[int] = []
    observed = None if observed_camera_mask is None else np.asarray(observed_camera_mask) > 127
    minimum_camera_precision = float(config.get("min_candidate_camera_precision", 0.90))
    minimum_camera_pixels = int(config.get("min_candidate_camera_overlap_pixels", 10))
    ambiguous_precision = float(config.get("ambiguous_candidate_camera_precision", 0.20))
    for component_index, component in enumerate(components):
        overlap = int(np.count_nonzero(component & valid_seeds))
        overlaps.append(overlap)
        support = {
            "component_index": component_index,
            "cell_count": int(np.count_nonzero(component)),
            "seed_overlap_cells": overlap,
            "camera_overlap_pixels": 0,
            "camera_projection_precision": 0.0,
            "camera_supported": False,
        }
        if observed is not None and sensors is not None:
            projected, _ = reproject_water_surface(component, estimated_water_level_m, sensors)
            projected_binary = projected > 127
            projected_count = int(np.count_nonzero(projected_binary))
            camera_overlap = int(np.count_nonzero(projected_binary & observed))
            precision = float(camera_overlap / max(1, projected_count))
            support.update({
                "camera_overlap_pixels": camera_overlap,
                "camera_projected_pixels": projected_count,
                "camera_projection_precision": precision,
                "camera_supported": bool(camera_overlap >= minimum_camera_pixels and precision >= minimum_camera_precision),
            })
        camera_support.append(support)
        if overlap >= minimum_seed_cells or support["camera_supported"]:
            selected.append(component)
        elif support["camera_overlap_pixels"] >= minimum_camera_pixels and support["camera_projection_precision"] >= ambiguous_precision:
            ambiguous_indices.append(component_index)
        elif observed is not None and sensors is not None and support.get("camera_projected_pixels", 0) == 0:
            # A below-level basin with neither a seed nor any Camera projection
            # cannot be classified as wet or dry from the allowed inputs.
            ambiguous_indices.append(component_index)
            unobserved_indices.append(component_index)
    predicted = np.logical_or.reduce(selected) if selected else np.zeros(dem.shape, dtype=bool)
    selected_count = len(selected)
    return predicted, {
        "candidate_basin_count": len(components),
        "selected_basin_count": selected_count,
        "candidate_seed_overlap_cells": overlaps,
        "seed_cell_count": int(np.count_nonzero(seeds)),
        "valid_seed_cell_count": int(np.count_nonzero(valid_seeds)),
        "seed_valid": bool(np.any(valid_seeds) and selected_count > 0),
        "ambiguous_candidate_basins": bool(ambiguous_indices),
        "ambiguous_candidate_indices": ambiguous_indices,
        "ambiguous_candidate_basin_count": len(ambiguous_indices),
        "unobserved_candidate_basin_count": len(unobserved_indices),
        "unobserved_candidate_indices": unobserved_indices,
        "camera_observable_candidate_basin_count": int(
            sum(int(item.get("camera_projected_pixels", 0)) > 0 for item in camera_support)
        ),
        "candidate_camera_support": camera_support,
        "candidate_lowland_cell_count": int(np.count_nonzero(candidate)),
        "predicted_water_cell_count": int(np.count_nonzero(predicted)),
        "barrier_rule": "flood_fill_only_within_ground_dem_below_estimated_water_level",
    }


def reproject_water_surface(
    predicted_dem_mask: np.ndarray,
    estimated_water_level_m: float,
    sensors: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(predicted_dem_mask, dtype=bool)
    model = camera_model(sensors)
    xx, yy = dem_cell_centres(mask.shape, sensors)
    resolution = float(sensors["road"]["dem_resolution_m"])
    image = Image.new("L", (model["width_px"], model["height_px"]), 0)
    draw = ImageDraw.Draw(image)
    valid_cells = 0
    for row, col in zip(*np.where(mask)):
        half = resolution / 2.0
        corners = np.asarray([
            [xx[row, col] - half, yy[row, col] - half, estimated_water_level_m],
            [xx[row, col] + half, yy[row, col] - half, estimated_water_level_m],
            [xx[row, col] + half, yy[row, col] + half, estimated_water_level_m],
            [xx[row, col] - half, yy[row, col] + half, estimated_water_level_m],
        ])
        optical = map_points_to_camera_optical(corners, model)
        z = optical[:, 2]
        valid = (z > model["near_m"]) & (z <= model["far_m"])
        if not np.all(valid):
            continue
        u = model["fx"] * optical[:, 0] / z + model["cx"]
        v = model["fy"] * optical[:, 1] / z + model["cy"]
        if np.all((u < 0) | (u >= model["width_px"]) | (v < 0) | (v >= model["height_px"])):
            continue
        draw.polygon([(int(round(x)), int(round(y))) for x, y in zip(u, v)], fill=255)
        valid_cells += 1
    array = np.asarray(image, dtype=np.uint8)
    return array, {
        "predicted_water_cell_count": int(np.count_nonzero(mask)),
        "reprojectable_water_cell_count": valid_cells,
        "water_surface_projection_coverage": float(valid_cells / max(1, np.count_nonzero(mask))),
    }


def _nearest_boundary_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    source_points = np.column_stack(np.where(source)).astype(np.float64)
    target_points = np.column_stack(np.where(target)).astype(np.float64)
    if source_points.size == 0 or target_points.size == 0:
        return np.asarray([], dtype=np.float64)
    distances: list[np.ndarray] = []
    for start in range(0, source_points.shape[0], chunk_size):
        chunk = source_points[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def camera_reprojection_consistency(
    observed_camera_mask: np.ndarray,
    reprojected_camera_mask: np.ndarray,
    projection_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    observed = np.asarray(observed_camera_mask) > 127
    predicted = np.asarray(reprojected_camera_mask) > 127
    intersection = int(np.count_nonzero(observed & predicted))
    union = int(np.count_nonzero(observed | predicted))
    observed_boundary = extract_boundary_mask(observed)
    predicted_boundary = extract_boundary_mask(predicted)
    forward = _nearest_boundary_distances(observed_boundary, predicted_boundary)
    reverse = _nearest_boundary_distances(predicted_boundary, observed_boundary)
    symmetric = np.concatenate((forward, reverse)) if forward.size and reverse.size else np.asarray([])
    return {
        "camera_reprojection_iou": float(intersection / max(1, union)),
        "camera_reprojection_precision": float(intersection / max(1, np.count_nonzero(predicted))),
        "camera_reprojection_recall": float(intersection / max(1, np.count_nonzero(observed))),
        "boundary_reprojection_mean_px": float(np.mean(symmetric)) if symmetric.size else None,
        "boundary_reprojection_p95_px": float(np.percentile(symmetric, 95)) if symmetric.size else None,
        "observed_camera_water_pixel_count": int(np.count_nonzero(observed)),
        "reprojected_camera_water_pixel_count": int(np.count_nonzero(predicted)),
        **projection_diagnostics,
    }
