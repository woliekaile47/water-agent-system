#!/usr/bin/env python3
"""Adapt a manually prompted SAM 2 shoreline to Phase 2B geometry.

Only prediction-side inputs are consumed: the manually prompted candidate,
the saved dry Ground DEM, camera geometry, and the source RGB frame hash.
Water-state Ground Truth is deliberately outside this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion.project_camera_mask_to_dem import camera_model
from src.fusion.water_surface_aware_mask_to_dem import (
    _nearest_boundary_distances,
    camera_ray_map,
    camera_reprojection_consistency,
    intersect_ray_with_dem,
    reconstruct_connected_lowland,
    reproject_water_surface,
)
from src.hydrology.estimate_water_level_from_boundary import connected_components, robust_filter
from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


SEMANTIC_LABEL = "manual_prompt_visible_water_candidate"
RESULT_SCOPE = "single_frame_camera_visible_region"
RESULT_NOTE = (
    "This result is derived from a manually prompted SAM 2 visible-water candidate "
    "and dry Ground DEM. It is an offline geometry diagnostic, not an autonomous or "
    "authoritative water measurement."
)


def prediction_semantics() -> dict[str, Any]:
    return {
        "semantic_label": SEMANTIC_LABEL,
        "authoritative": False,
        "ground_truth_used": False,
        "result_scope": RESULT_SCOPE,
        "area_volume_semantics": "camera_visible_candidate_estimate",
        "eligible_for_formal_s5_s8": False,
        "eligible_for_downstream": False,
        "result_note": RESULT_NOTE,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path, root_key: str | None = None) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if root_key is None:
        return document
    if root_key not in document:
        raise ValueError(f"Missing {root_key} in {path}")
    return document[root_key]


def _validate_xy(name: str, points: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2] in [x, y] order")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not (
        np.all(values[:, 0] >= 0)
        and np.all(values[:, 0] < width)
        and np.all(values[:, 1] >= 0)
        and np.all(values[:, 1] < height)
    ):
        raise ValueError(f"{name} contains coordinates outside the Camera image")
    return values.astype(np.float64, copy=True)


def load_prediction_inputs(
    input_dir: str | Path,
    source_image: str | Path,
    ground_dem_path: str | Path,
    sensors_path: str | Path,
) -> dict[str, Any]:
    """Load the strict prediction allow-list and validate provenance.

    source_trace.json is required and hashed for custody only. It is not parsed,
    because it intentionally contains later-evaluation metadata that prediction
    must not consume.
    """
    directory = Path(input_dir)
    paths = {
        "mask": directory / "selected_component_mask.npy",
        "full_shoreline": directory / "prompted_outer_shoreline_full_xy.npy",
        "sampled_shoreline": directory / "prompted_outer_shoreline_sampled_128_xy.npy",
        "shoreline_metadata": directory / "prompted_outer_shoreline.json",
        "manual_prompt": directory / "water_test_manual_prompt.json",
        "source_trace_custody_only": directory / "source_trace.json",
        "source_image": Path(source_image),
        "dry_ground_dem": Path(ground_dem_path),
        "sensors": Path(sensors_path),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction input files: {missing}")

    sensors = load_yaml(paths["sensors"])
    model = camera_model(sensors)
    mask = np.load(paths["mask"], allow_pickle=False)
    if mask.shape != (model["height_px"], model["width_px"]):
        raise ValueError(f"SAM 2 mask shape {mask.shape} does not match Camera {(model['height_px'], model['width_px'])}")
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("SAM 2 selected component mask is empty")
    full = _validate_xy(
        "full shoreline",
        np.load(paths["full_shoreline"], allow_pickle=False),
        model["width_px"],
        model["height_px"],
    )
    sampled = _validate_xy(
        "sampled shoreline",
        np.load(paths["sampled_shoreline"], allow_pickle=False),
        model["width_px"],
        model["height_px"],
    )
    if sampled.shape[0] != 128:
        raise ValueError(f"Expected 128 sampled shoreline points, got {sampled.shape[0]}")

    shoreline_metadata = json.loads(paths["shoreline_metadata"].read_text(encoding="utf-8"))
    manual_prompt = json.loads(paths["manual_prompt"].read_text(encoding="utf-8"))
    if shoreline_metadata.get("coordinate_convention", {}).get("order") != "[x, y]":
        raise ValueError("Shoreline coordinate convention is not [x, y]")
    source_hash = sha256_file(paths["source_image"])
    expected_hash = str(shoreline_metadata.get("sha256", {}).get("input_image", ""))
    if source_hash != expected_hash:
        raise ValueError("Source image SHA-256 does not match shoreline provenance")
    prompt_hash = sha256_file(paths["manual_prompt"])
    expected_prompt_hash = str(shoreline_metadata.get("sha256", {}).get("manual_prompt_json", ""))
    if prompt_hash != expected_prompt_hash:
        raise ValueError("Manual prompt SHA-256 does not match shoreline provenance")

    ground_dem = np.load(paths["dry_ground_dem"], allow_pickle=False).astype(np.float32)
    expected_dem_shape = (
        int(round(float(sensors["road"]["width_m"]) / float(sensors["road"]["dem_resolution_m"]))),
        int(round(float(sensors["road"]["length_m"]) / float(sensors["road"]["dem_resolution_m"]))),
    )
    if ground_dem.shape != expected_dem_shape or not np.isfinite(ground_dem).all():
        raise ValueError("Dry Ground DEM shape or finite-value validation failed")

    positive_points = _validate_xy(
        "manual positive points",
        np.asarray(manual_prompt.get("positive_points", []), dtype=np.float64),
        model["width_px"],
        model["height_px"],
    )
    validation = {
        **prediction_semantics(),
        "input_files": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "source_image_sha256_strict_match": True,
        "source_trace_consumed_by_prediction": False,
        "source_trace_role": "custody_copy_only_not_parsed",
        "image_shape_hw": [model["height_px"], model["width_px"]],
        "selected_component_mask_shape": list(mask.shape),
        "selected_component_mask_pixel_count": int(np.count_nonzero(mask)),
        "full_shoreline_point_count": int(full.shape[0]),
        "sampled_shoreline_point_count": int(sampled.shape[0]),
        "coordinate_order": "[x, y]",
        "all_coordinates_in_image": True,
        "manual_positive_point_count": int(positive_points.shape[0]),
        "dry_ground_dem_shape": list(ground_dem.shape),
        "dry_ground_dem_all_finite": True,
        "camera_model": model,
        "validation_status": "pass",
    }
    return {
        "mask": mask,
        "full_shoreline_xy": full,
        "sampled_shoreline_xy": sampled,
        "positive_points_xy": positive_points,
        "manual_prompt": manual_prompt,
        "shoreline_metadata": shoreline_metadata,
        "ground_dem": ground_dem,
        "sensors": sensors,
        "camera_model": model,
        "validation": validation,
    }


def optical_ray_direction(u: float, v: float, model: dict[str, Any]) -> np.ndarray:
    direction = np.asarray(
        [(float(u) - model["cx"]) / model["fx"], (float(v) - model["cy"]) / model["fy"], 1.0],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    return direction


def intersect_pixel_rays(
    points_xy: np.ndarray,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
    config: dict[str, Any],
    sample_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = camera_model(sensors)
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for index, (u, v) in enumerate(np.asarray(points_xy, dtype=np.float64)):
        optical = optical_ray_direction(float(u), float(v), model)
        origin, direction = camera_ray_map(float(u), float(v), model)
        if optical[2] <= 0.0 or not np.isfinite(direction).all():
            hit, reason = None, "invalid_forward_ray"
        else:
            hit, reason = intersect_ray_with_dem(origin, direction, ground_dem, sensors, config)
        success = hit is not None
        if not success:
            failures[reason] += 1
        record: dict[str, Any] = {
            "sample_id": f"{sample_prefix}_{index:03d}",
            "sample_index": int(index),
            "component_index": 0,
            "pixel_u": float(u),
            "pixel_v": float(v),
            "ray_direction_optical_xyz": optical.tolist(),
            "ray_direction_map_xyz": direction.tolist(),
            "ray_origin_map_xyz": origin.tolist(),
            "hit_status": "success" if success else "failed",
            "failure_reason": None if success else reason,
            "intersection_map_xyz": None,
            "dem_z_m": None,
            "intersection_residual_m": None,
            "iteration_count": int(config.get("bisection_iterations", 18)) if success else 0,
            "iteration_count_semantics": "bisection_iterations_after_first_sign_change",
            "distance_from_camera_m": None,
        }
        if hit is not None:
            record.update({
                "intersection_map_xyz": [hit["x_m"], hit["y_m"], hit["z_ray_m"]],
                "dem_z_m": hit["dem_height_m"],
                "intersection_residual_m": hit["residual_m"],
                "distance_from_camera_m": hit["ray_distance_m"],
            })
        records.append(record)
    successes = sum(record["hit_status"] == "success" for record in records)
    diagnostics = {
        "total_ray_count": len(records),
        "successful_intersection_count": int(successes),
        "intersection_success_ratio": float(successes / max(1, len(records))),
        "intersection_failure_reasons": dict(sorted(failures.items())),
    }
    return records, diagnostics


def height_statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: None for key in (
            "shoreline_height_min_m", "p10_m", "p25_m", "median_m", "p75_m",
            "p90_m", "shoreline_height_max_m", "mean_m", "std_m", "MAD_m", "IQR_m"
        )} | {"count": 0}
    p10, p25, median, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {
        "count": int(array.size),
        "shoreline_height_min_m": float(np.min(array)),
        "p10_m": float(p10),
        "p25_m": float(p25),
        "median_m": float(median),
        "p75_m": float(p75),
        "p90_m": float(p90),
        "shoreline_height_max_m": float(np.max(array)),
        "mean_m": float(np.mean(array)),
        "std_m": float(np.std(array)),
        "MAD_m": float(np.median(np.abs(array - median))),
        "IQR_m": float(p75 - p25),
    }


def estimate_level_from_ray_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    successful = [record for record in records if record["hit_status"] == "success"]
    phase2b_records = [
        {"component_index": 0, "dem_height_m": float(record["dem_z_m"])} for record in successful
    ]
    raw = np.asarray([record["dem_z_m"] for record in successful], dtype=np.float64)
    filtered, filter_diagnostics = robust_filter(raw, config)
    level, phase2b = estimate_water_level_from_shoreline(phase2b_records, config)
    diagnostics = {
        **prediction_semantics(),
        "total_shoreline_ray_count": len(records),
        "successful_intersection_count": len(successful),
        "intersection_success_ratio": float(len(successful) / max(1, len(records))),
        "water_level_estimation_method": phase2b["method"],
        "estimated_water_level_m": float(level),
        "raw_height_statistics": height_statistics(raw),
        "filtered_height_statistics": height_statistics(filtered),
        "raw_height_samples_m": raw.tolist(),
        "filtered_height_samples_m": filtered.tolist(),
        "inlier_count": int(filtered.size),
        "outlier_count": int(raw.size - filtered.size),
        "outlier_filter": filter_diagnostics,
        "converged": bool(phase2b["water_level_converged"]),
        "confidence_diagnostic": {
            "basis": "prediction_side_shoreline_sample_quality_only",
            "intersection_success_ratio": float(len(successful) / max(1, len(records))),
            "filtered_sample_count": int(filtered.size),
            "MAD_m": height_statistics(filtered)["MAD_m"],
            "IQR_m": height_statistics(filtered)["IQR_m"],
        },
        **phase2b,
    }
    return float(level), diagnostics


def deterministic_seed_pixels(
    mask: np.ndarray,
    positive_points_xy: np.ndarray,
    maximum_mask_seed_count: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    binary = np.asarray(mask, dtype=bool)
    candidates: list[tuple[float, float, str]] = []
    for u, v in np.asarray(positive_points_xy, dtype=np.float64):
        row = int(np.clip(round(float(v)), 0, binary.shape[0] - 1))
        col = int(np.clip(round(float(u)), 0, binary.shape[1] - 1))
        if binary[row, col]:
            candidates.append((float(u), float(v), "manual_positive_point"))
    rows, cols = np.where(binary)
    count = min(maximum_mask_seed_count, rows.size)
    if count:
        indices = np.linspace(0, rows.size - 1, count, dtype=np.int64)
        for index in indices.tolist():
            candidates.append((float(cols[index]), float(rows[index]), "deterministic_mask_interior"))
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    records: list[dict[str, Any]] = []
    for u, v, source in candidates:
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
        records.append({"pixel_u": u, "pixel_v": v, "seed_source": source})
    return np.asarray(unique, dtype=np.float64).reshape((-1, 2)), records


def ray_hits_to_dem_seed_mask(
    records: list[dict[str, Any]],
    dem_shape: tuple[int, int],
    sensors: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    resolution = float(sensors["road"]["dem_resolution_m"])
    x0 = -float(sensors["road"]["length_m"]) / 2.0 + resolution / 2.0
    y0 = -float(sensors["road"]["width_m"]) / 2.0 + resolution / 2.0
    seed_mask = np.zeros(dem_shape, dtype=bool)
    mapped: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["dem_row"] = None
        item["dem_col"] = None
        item["dem_cell_valid"] = False
        if record["hit_status"] == "success":
            x_m, y_m, _ = record["intersection_map_xyz"]
            col = int(round((float(x_m) - x0) / resolution))
            row = int(round((float(y_m) - y0) / resolution))
            if 0 <= row < dem_shape[0] and 0 <= col < dem_shape[1]:
                seed_mask[row, col] = True
                item.update({"dem_row": row, "dem_col": col, "dem_cell_valid": True})
        mapped.append(item)
    return seed_mask, mapped


def reconstruct_seed_connected_lowland(
    ground_dem: np.ndarray,
    estimated_water_level_m: float,
    seed_mask: np.ndarray,
    config: dict[str, Any],
    observed_camera_mask: np.ndarray,
    sensors: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse Phase 2B reconstruction while enforcing seed-connected output."""
    predicted, diagnostics = reconstruct_connected_lowland(
        ground_dem,
        estimated_water_level_m,
        seed_mask,
        config,
        observed_camera_mask=None,
        sensors=None,
    )
    candidate = np.isfinite(ground_dem) & (
        ground_dem < float(estimated_water_level_m) - float(config.get("lowland_margin_m", 0.0))
    )
    valid_seeds = seed_mask & candidate
    components = connected_components(candidate, int(config.get("connectivity", 8)))
    support: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    unselected_indices: list[int] = []
    unobservable_indices: list[int] = []
    for index, component in enumerate(components):
        seed_cells = int(np.count_nonzero(component & valid_seeds))
        projected, projection = reproject_water_surface(component, estimated_water_level_m, sensors)
        projected_binary = projected > 127
        projected_count = int(np.count_nonzero(projected_binary))
        overlap = int(np.count_nonzero(projected_binary & np.asarray(observed_camera_mask, dtype=bool)))
        if seed_cells:
            selected_indices.append(index)
        else:
            unselected_indices.append(index)
            if projected_count == 0:
                unobservable_indices.append(index)
        support.append({
            "component_index": index,
            "cell_count": int(np.count_nonzero(component)),
            "seed_overlap_cells": seed_cells,
            "camera_projected_pixels": projected_count,
            "camera_overlap_pixels": overlap,
            "camera_projection_precision": float(overlap / max(1, projected_count)),
            "projection_coverage": projection["water_surface_projection_coverage"],
            "selected_by_seed": bool(seed_cells),
        })
    diagnostics.update({
        "candidate_basin_count": len(components),
        "selected_basin_count": len(selected_indices),
        "selected_basin_indices": selected_indices,
        "seed_basin_indices": selected_indices,
        "ambiguous_candidate_basins": bool(unselected_indices),
        "ambiguous_candidate_indices": unselected_indices,
        "ambiguous_candidate_basin_count": len(unselected_indices),
        "unobserved_candidate_basin_count": len(unobservable_indices),
        "unobserved_candidate_indices": unobservable_indices,
        "camera_observable_candidate_basin_count": sum(item["camera_projected_pixels"] > 0 for item in support),
        "candidate_camera_support": support,
        "seed_valid": bool(np.any(valid_seeds) and len(selected_indices) > 0),
        "selection_rule": "seed_connected_components_only_no_camera_unsupported_basin_fill",
        "result_scope": "camera_visible_manual_prompt_candidate",
    })
    return predicted, diagnostics


def outer_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return water pixels adjacent to exterior background, excluding holes."""
    water = np.asarray(mask, dtype=bool)
    background = ~water
    exterior = np.zeros_like(background)
    queue: deque[tuple[int, int]] = deque()
    height, width = water.shape
    for row in (0, height - 1):
        for col in range(width):
            if background[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                queue.append((row, col))
    for col in (0, width - 1):
        for row in range(height):
            if background[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + drow, col + dcol
            if 0 <= rr < height and 0 <= cc < width and background[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                queue.append((rr, cc))
    boundary = np.zeros_like(water)
    padded = np.pad(exterior, 1, mode="constant", constant_values=False)
    for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        boundary |= water & padded[1 + drow : 1 + drow + height, 1 + dcol : 1 + dcol + width]
    return boundary


def _symmetric_boundary_percentiles(first: np.ndarray, second: np.ndarray) -> tuple[float | None, float | None]:
    forward = _nearest_boundary_distances(first, second)
    reverse = _nearest_boundary_distances(second, first)
    if not forward.size or not reverse.size:
        return None, None
    distances = np.concatenate((forward, reverse))
    return float(np.percentile(distances, 50)), float(np.percentile(distances, 95))


def extended_reprojection_consistency(
    observed_camera_mask: np.ndarray,
    reprojected_camera_mask: np.ndarray,
    projection_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    observed_binary = np.asarray(observed_camera_mask, dtype=bool)
    observed_u8 = np.where(observed_binary, 255, 0).astype(np.uint8)
    base = camera_reprojection_consistency(
        observed_u8,
        reprojected_camera_mask,
        projection_diagnostics,
    )
    observed = observed_binary
    predicted = np.asarray(reprojected_camera_mask) > 127
    p50, p95 = _symmetric_boundary_percentiles(extract_boundary_mask(observed), extract_boundary_mask(predicted))
    outer_p50, outer_p95 = _symmetric_boundary_percentiles(outer_boundary_mask(observed), outer_boundary_mask(predicted))
    border = np.zeros_like(predicted)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    base.update({
        "boundary_reprojection_p50_px": p50,
        "boundary_reprojection_p95_px": p95,
        "outer_boundary_reprojection_p50_px": outer_p50,
        "outer_boundary_reprojection_p95_px": outer_p95,
        "valid_projection_ratio": float(projection_diagnostics.get("water_surface_projection_coverage", 0.0)),
        "camera_coverage_ratio": float(base["camera_reprojection_recall"]),
        "camera_coverage_ratio_semantics": "fraction_of_manual_prompt_candidate_covered_by_reprojection",
        "predicted_mask_touches_image_border": bool(np.any(predicted & border)),
    })
    return base
