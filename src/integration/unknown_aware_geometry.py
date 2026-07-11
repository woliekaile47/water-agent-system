#!/usr/bin/env python3
"""Unknown-aware geometry adapters around the frozen Phase 2B algorithms."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.fusion.project_camera_mask_to_dem import camera_model
from src.fusion.water_surface_aware_mask_to_dem import (
    _nearest_boundary_distances,
    camera_ray_map,
    intersect_ray_with_dem,
    reproject_water_surface,
)
from src.hydrology.estimate_water_level_from_boundary import connected_components


NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
UNKNOWN_SEMANTICS = "no_temporal_evidence_not_confirmed_dry"


def _validate_camera_masks(water: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    water_mask = np.asarray(water, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    if water_mask.ndim != 2 or water_mask.shape != unknown_mask.shape:
        raise ValueError("camera water and unknown masks must be matching 2D arrays")
    if np.any(water_mask & unknown_mask):
        raise ValueError("camera water and unknown masks must not overlap")
    return water_mask, unknown_mask


def build_trusted_shoreline(
    camera_water_mask: np.ndarray, camera_unknown_mask: np.ndarray, connectivity: int = 8,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Keep only water-to-known-nonwater four-neighbor interfaces inside the image."""
    water, unknown = _validate_camera_masks(camera_water_mask, camera_unknown_mask)
    known_nonwater = ~water & ~unknown
    trusted = np.zeros_like(water)
    interface_points: list[dict[str, Any]] = []
    candidate_count = 0
    unknown_adjacent = 0
    image_edge = 0
    height, width = water.shape
    for row, col in zip(*np.where(water)):
        for drow, dcol in NEIGHBORS_4:
            neighbor_row, neighbor_col = int(row + drow), int(col + dcol)
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                candidate_count += 1
                image_edge += 1
                continue
            if water[neighbor_row, neighbor_col]:
                continue
            candidate_count += 1
            if unknown[neighbor_row, neighbor_col]:
                unknown_adjacent += 1
                continue
            if not known_nonwater[neighbor_row, neighbor_col]:
                continue
            trusted[row, col] = True
            interface_points.append({
                "water_row": int(row), "water_col": int(col),
                "pixel_v": float(row) + 0.5 * drow,
                "pixel_u": float(col) + 0.5 * dcol,
            })
    components = connected_components(water, connectivity)
    diagnostics = {
        "water_component_count": len(components),
        "retained_water_component_count": len(components),
        "candidate_interface_count": candidate_count,
        "trusted_interface_count": len(interface_points),
        "unknown_adjacent_interface_count": unknown_adjacent,
        "image_edge_interface_count": image_edge,
        "trusted_shoreline_fraction": float(len(interface_points) / max(1, candidate_count)),
        "unknown_region_semantics": UNKNOWN_SEMANTICS,
        "ground_truth_used": False,
    }
    return trusted, interface_points, diagnostics


def intersect_trusted_camera_shoreline(
    camera_water_mask: np.ndarray, camera_unknown_mask: np.ndarray, ground_dem: np.ndarray,
    sensors: dict[str, Any], mapping_config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    water, unknown = _validate_camera_masks(camera_water_mask, camera_unknown_mask)
    trusted, points, diagnostics = build_trusted_shoreline(
        water, unknown, int(mapping_config.get("connectivity", 8)),
    )
    components = connected_components(water, int(mapping_config.get("connectivity", 8)))
    minimum_pixels = int(mapping_config.get("min_image_component_pixels", 10))
    component_by_pixel: dict[tuple[int, int], int] = {}
    retained: set[int] = set()
    for index, component in enumerate(components):
        if int(np.count_nonzero(component)) < minimum_pixels:
            continue
        retained.add(index)
        for row, col in zip(*np.where(component)):
            component_by_pixel[(int(row), int(col))] = index
    filtered_points = [point for point in points if (point["water_row"], point["water_col"]) in component_by_pixel]
    stride = max(1, int(mapping_config.get("shoreline_sample_stride_px", 2)))
    model = camera_model(sensors)
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for point in filtered_points[::stride]:
        origin, direction = camera_ray_map(point["pixel_u"], point["pixel_v"], model)
        hit, reason = intersect_ray_with_dem(origin, direction, ground_dem, sensors, mapping_config)
        if hit is None:
            failures[reason] += 1
            continue
        records.append({
            "component_index": component_by_pixel[(point["water_row"], point["water_col"])],
            "pixel_u": point["pixel_u"], "pixel_v": point["pixel_v"],
            "ray_direction_map": direction.tolist(), **hit,
        })
    edge_band = max(1, int(mapping_config.get("image_edge_band_px", 2)))
    water_rows, water_cols = np.where(water)
    edge_water = (
        (water_rows < edge_band) | (water_rows >= water.shape[0] - edge_band)
        | (water_cols < edge_band) | (water_cols >= water.shape[1] - edge_band)
    )
    sampled = len(filtered_points[::stride])
    diagnostics.update({
        "retained_water_component_count": len(retained),
        "sampled_shoreline_ray_count": sampled,
        "successful_intersection_count": len(records),
        "shoreline_intersection_success_rate": float(len(records) / max(1, sampled)),
        "intersection_failure_reasons": dict(sorted(failures.items())),
        "camera_mask_edge_touch_ratio": float(np.count_nonzero(edge_water) / max(1, water_rows.size)),
        "camera_frame": model["frame_id"], "map_frame": model["map_frame_id"],
        "transform": "T_map_camera_optical",
    })
    return trusted, records, diagnostics


def reconstruct_connected_lowland_unknown_aware(
    ground_dem: np.ndarray, estimated_water_level_m: float, seed_mask: np.ndarray,
    reconstruction_config: dict[str, Any], observed_camera_water_mask: np.ndarray,
    observed_camera_unknown_mask: np.ndarray, sensors: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    dem = np.asarray(ground_dem, dtype=np.float64)
    seeds = np.asarray(seed_mask, dtype=bool)
    water, unknown = _validate_camera_masks(observed_camera_water_mask, observed_camera_unknown_mask)
    if dem.ndim != 2 or dem.shape != seeds.shape:
        raise ValueError("ground_dem and seed_mask must be matching 2D arrays")
    if not np.isfinite(estimated_water_level_m):
        raise ValueError("estimated_water_level_m must be finite")
    candidate = np.isfinite(dem) & (
        dem < float(estimated_water_level_m) - float(reconstruction_config.get("lowland_margin_m", 0.0))
    )
    valid_seeds = seeds & candidate
    components = connected_components(candidate, int(reconstruction_config.get("connectivity", 8)))
    minimum_seed = int(reconstruction_config.get("min_seed_cells_per_basin", 1))
    minimum_precision = float(reconstruction_config.get("min_candidate_camera_precision", .9))
    minimum_pixels = int(reconstruction_config.get("min_candidate_camera_overlap_pixels", 10))
    ambiguous_precision = float(reconstruction_config.get("ambiguous_candidate_camera_precision", .2))
    selected: list[np.ndarray] = []
    ambiguous_indices: list[int] = []
    unobserved_indices: list[int] = []
    unknown_only_indices: list[int] = []
    support_rows = []
    overlaps = []
    known_nonwater = ~water & ~unknown
    for index, component in enumerate(components):
        seed_overlap = int(np.count_nonzero(component & valid_seeds))
        overlaps.append(seed_overlap)
        projected, _ = reproject_water_surface(component, estimated_water_level_m, sensors)
        projected_mask = projected > 127
        projected_pixels = int(np.count_nonzero(projected_mask))
        known_projected = int(np.count_nonzero(projected_mask & ~unknown))
        water_overlap = int(np.count_nonzero(projected_mask & water))
        known_nonwater_overlap = int(np.count_nonzero(projected_mask & known_nonwater))
        unknown_overlap = int(np.count_nonzero(projected_mask & unknown))
        known_precision = float(water_overlap / known_projected) if known_projected else None
        camera_supported = bool(
            known_projected > 0 and water_overlap >= minimum_pixels
            and known_precision is not None and known_precision >= minimum_precision
        )
        support = {
            "component_index": index, "cell_count": int(np.count_nonzero(component)),
            "seed_overlap_cells": seed_overlap,
            "projected_pixels": projected_pixels, "camera_projected_pixels": projected_pixels,
            "known_projected_pixels": known_projected,
            "water_overlap_pixels": water_overlap, "camera_overlap_pixels": water_overlap,
            "known_nonwater_overlap_pixels": known_nonwater_overlap,
            "unknown_overlap_pixels": unknown_overlap,
            "known_camera_precision": known_precision,
            "camera_projection_precision": known_precision if known_precision is not None else 0.0,
            "camera_supported": camera_supported,
            "observation_status": "known_observable",
        }
        if seed_overlap >= minimum_seed or camera_supported:
            selected.append(component)
            support["selected"] = True
        elif projected_pixels == 0:
            support.update({"selected": False, "observation_status": "unobserved"})
            ambiguous_indices.append(index)
            unobserved_indices.append(index)
        elif known_projected == 0:
            support.update({"selected": False, "observation_status": "unknown_only"})
            ambiguous_indices.append(index)
            unknown_only_indices.append(index)
        elif water_overlap >= minimum_pixels and known_precision is not None and known_precision >= ambiguous_precision:
            support.update({"selected": False, "observation_status": "ambiguous_camera_support"})
            ambiguous_indices.append(index)
        else:
            support["selected"] = False
        support_rows.append(support)
    predicted = np.logical_or.reduce(selected) if selected else np.zeros_like(candidate)
    return predicted, {
        "candidate_basin_count": len(components), "selected_basin_count": len(selected),
        "candidate_seed_overlap_cells": overlaps,
        "seed_cell_count": int(np.count_nonzero(seeds)),
        "valid_seed_cell_count": int(np.count_nonzero(valid_seeds)),
        "seed_valid": bool(np.any(valid_seeds) and selected),
        "ambiguous_candidate_basins": bool(ambiguous_indices),
        "ambiguous_candidate_indices": ambiguous_indices,
        "ambiguous_candidate_basin_count": len(ambiguous_indices),
        "unobserved_candidate_basin_count": len(unobserved_indices),
        "unobserved_candidate_indices": unobserved_indices,
        "unknown_only_candidate_basin_count": len(unknown_only_indices),
        "unknown_only_candidate_indices": unknown_only_indices,
        "camera_observable_candidate_basin_count": sum(row["projected_pixels"] > 0 for row in support_rows),
        "camera_known_observable_candidate_basin_count": sum(row["known_projected_pixels"] > 0 for row in support_rows),
        "candidate_camera_support": support_rows,
        "candidate_lowland_cell_count": int(np.count_nonzero(candidate)),
        "predicted_water_cell_count": int(np.count_nonzero(predicted)),
        "barrier_rule": "flood_fill_only_within_ground_dem_below_level_and_camera_supported_basins",
        "unknown_region_semantics": UNKNOWN_SEMANTICS,
        "ground_truth_used": False,
    }


def camera_reprojection_consistency_unknown_aware(
    observed_camera_water_mask: np.ndarray, observed_camera_unknown_mask: np.ndarray,
    reprojected_camera_mask: np.ndarray, projection_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    observed, unknown = _validate_camera_masks(observed_camera_water_mask, observed_camera_unknown_mask)
    predicted = np.asarray(reprojected_camera_mask) > 127
    if predicted.shape != observed.shape:
        raise ValueError("reprojected mask shape differs from observed Camera mask")
    known = ~unknown
    predicted_known = predicted & known
    intersection = int(np.count_nonzero(observed & predicted_known))
    union = int(np.count_nonzero((observed | predicted_known) & known))
    observed_boundary, _, _ = build_trusted_shoreline(observed, unknown)
    predicted_boundary, _, _ = build_trusted_shoreline(predicted_known, unknown)
    forward = _nearest_boundary_distances(observed_boundary, predicted_boundary)
    reverse = _nearest_boundary_distances(predicted_boundary, observed_boundary)
    symmetric = np.concatenate((forward, reverse)) if forward.size and reverse.size else np.asarray([])
    predicted_count = int(np.count_nonzero(predicted_known))
    observed_count = int(np.count_nonzero(observed))
    iou = float(intersection / max(1, union))
    return {
        "camera_reprojection_iou": iou,
        "camera_reprojection_precision": float(intersection / max(1, predicted_count)),
        "camera_reprojection_recall": float(intersection / max(1, observed_count)),
        "boundary_reprojection_mean_px": float(np.mean(symmetric)) if symmetric.size else None,
        "boundary_reprojection_p95_px": float(np.percentile(symmetric, 95)) if symmetric.size else None,
        "water_surface_projection_coverage": float(projection_diagnostics.get("water_surface_projection_coverage", 0.0)),
        "observed_camera_water_pixel_count": observed_count,
        "reprojected_camera_water_pixel_count": predicted_count,
        "known_camera_pixel_count": int(np.count_nonzero(known)),
        "unknown_camera_pixel_count": int(np.count_nonzero(unknown)),
        "known_region_reprojection_iou": iou,
        "trusted_observed_boundary_pixel_count": int(np.count_nonzero(observed_boundary)),
        "trusted_reprojected_boundary_pixel_count": int(np.count_nonzero(predicted_boundary)),
        "metric_domain": "camera_known_region_excluding_unknown",
        "unknown_region_semantics": UNKNOWN_SEMANTICS,
        "ground_truth_used": False,
        **projection_diagnostics,
    }
