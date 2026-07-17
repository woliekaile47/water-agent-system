#!/usr/bin/env python3
"""Deterministic, GT-free conversion of temporal evidence into SAM 2 prompts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_records(
    water_mask: np.ndarray,
    probability: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        water_mask.astype(np.uint8), connectivity=connectivity
    )
    records: list[dict[str, Any]] = []
    for label in range(1, count):
        component = labels == label
        records.append({
            "label": int(label),
            "area_pixels": int(stats[label, cv2.CC_STAT_AREA]),
            "probability_mass": float(np.sum(probability[component], dtype=np.float64)),
            "bbox_xywh": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ],
        })
    records.sort(key=lambda item: (-item["probability_mass"], -item["area_pixels"], item["label"]))
    return labels, records


def _select_positive_points(
    component: np.ndarray,
    probability: np.ndarray,
    target: int,
    min_boundary_distance: float,
    min_spacing: float,
) -> tuple[list[list[int]], list[float]]:
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(component & (distance >= min_boundary_distance))
    if ys.size == 0:
        return [], []
    candidates = np.column_stack((xs, ys)).astype(np.int32)
    selected: list[np.ndarray] = []
    selected_boundary_distances: list[float] = []
    available = np.ones(candidates.shape[0], dtype=bool)
    while len(selected) < target and np.any(available):
        indices = np.flatnonzero(available)
        points = candidates[indices]
        point_distances = distance[points[:, 1], points[:, 0]]
        point_probabilities = probability[points[:, 1], points[:, 0]]
        if selected:
            prior = np.asarray(selected, dtype=np.float64)
            spacing = np.min(
                np.linalg.norm(points[:, None, :].astype(np.float64) - prior[None, :, :], axis=2),
                axis=1,
            )
            valid_spacing = spacing >= min_spacing
            if not np.any(valid_spacing):
                break
            indices = indices[valid_spacing]
            points = candidates[indices]
            point_distances = distance[points[:, 1], points[:, 0]]
            point_probabilities = probability[points[:, 1], points[:, 0]]
            spacing = spacing[valid_spacing]
        else:
            spacing = np.zeros(points.shape[0], dtype=np.float64)
        order = np.lexsort((points[:, 0], points[:, 1], -point_probabilities, -point_distances, -spacing))
        chosen_index = int(indices[int(order[0])])
        chosen = candidates[chosen_index]
        selected.append(chosen.copy())
        selected_boundary_distances.append(float(distance[chosen[1], chosen[0]]))
        available[chosen_index] = False
    return [[int(point[0]), int(point[1])] for point in selected], selected_boundary_distances


def _valid_negative_pixel(
    x: int,
    y: int,
    water: np.ndarray,
    unknown: np.ndarray,
) -> bool:
    height, width = water.shape
    return 0 <= x < width and 0 <= y < height and not water[y, x] and not unknown[y, x]


def _sector(x: int, y: int, center_x: float, center_y: float) -> int:
    angle = (np.arctan2(y - center_y, x - center_x) + 2.0 * np.pi) % (2.0 * np.pi)
    return int(np.floor(angle / (2.0 * np.pi / 8.0))) % 8


def _select_negative_points(
    water: np.ndarray,
    unknown: np.ndarray,
    component: np.ndarray,
    probability: np.ndarray,
    classifications: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[list[int]], list[str], list[int]]:
    ys, xs = np.where(component)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    target = int(config["target_negative_points"])
    selected: list[list[int]] = []
    sources: list[str] = []
    occupied_sectors: set[int] = set()

    background = ~component
    count, background_labels = cv2.connectedComponents(
        background.astype(np.uint8), connectivity=8
    )
    border_labels = np.unique(np.concatenate((
        background_labels[0, :], background_labels[-1, :],
        background_labels[:, 0], background_labels[:, -1],
    )))
    border_labels = border_labels[(border_labels > 0) & (border_labels < count)]
    external_background = np.isin(background_labels, border_labels)
    outside_distance = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 5)
    minimum_negative_distance = float(config["negative_ring_inner_distance_px"])
    maximum_negative_distance = float(config["negative_ring_outer_distance_px"])

    dry_candidates: list[tuple[float, int, int, int]] = []
    for item in classifications:
        if item.get("classification") != "dry_splash":
            continue
        confidence = float(item.get("confidence", 0.0))
        if confidence < float(config["dry_track_min_confidence"]):
            continue
        center = item.get("center_mean")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        x, y = int(round(float(center[0]))), int(round(float(center[1])))
        if (
            _valid_negative_pixel(x, y, water, unknown)
            and external_background[y, x]
            and minimum_negative_distance <= float(outside_distance[y, x]) <= maximum_negative_distance
        ):
            dry_candidates.append((-confidence, -int(item.get("duration_frames", 0)), y, x))
    deferred_dry: list[tuple[int, int]] = []
    for _, _, y, x in sorted(dry_candidates):
        point = [x, y]
        if point in selected:
            continue
        sector = _sector(x, y, center_x, center_y)
        if sector in occupied_sectors:
            deferred_dry.append((y, x))
            continue
        selected.append(point)
        sources.append("dry_splash_track")
        occupied_sectors.add(sector)
        if len(selected) >= target:
            return selected, sources, sorted(occupied_sectors)

    known_nonwater = ~water & ~unknown
    ring = (
        known_nonwater
        & external_background
        & (outside_distance >= minimum_negative_distance)
        & (outside_distance <= maximum_negative_distance)
    )
    ring_y, ring_x = np.where(ring)
    if ring_y.size:
        middle = 0.5 * (
            minimum_negative_distance + maximum_negative_distance
        )
        ring_records = []
        for x, y in zip(ring_x.tolist(), ring_y.tolist()):
            sector = _sector(x, y, center_x, center_y)
            ring_records.append((
                sector,
                abs(float(outside_distance[y, x]) - middle),
                float(probability[y, x]),
                y,
                x,
            ))
        for sector in range(8):
            if len(selected) >= target:
                break
            if sector in occupied_sectors:
                continue
            candidates = [item for item in ring_records if item[0] == sector]
            if not candidates:
                continue
            _, _, _, y, x = min(candidates)
            point = [x, y]
            if point not in selected:
                selected.append(point)
                sources.append("known_nonwater_ring")
                occupied_sectors.add(sector)
        if len(selected) < target:
            for y, x in deferred_dry:
                point = [x, y]
                if point in selected:
                    continue
                selected.append(point)
                sources.append("dry_splash_track")
                occupied_sectors.add(_sector(x, y, center_x, center_y))
                if len(selected) >= target:
                    break
        if len(selected) < target:
            for _, _, _, y, x in sorted(ring_records):
                point = [x, y]
                if point in selected:
                    continue
                selected.append(point)
                sources.append("known_nonwater_ring")
                occupied_sectors.add(_sector(x, y, center_x, center_y))
                if len(selected) >= target:
                    break
    return selected, sources, sorted(occupied_sectors)


def generate_temporal_sam2_prompt(
    probability: np.ndarray,
    water_mask: np.ndarray,
    unknown_mask: np.ndarray,
    classifications: list[dict[str, Any]],
    temporal_quality_gate: dict[str, Any],
    config: dict[str, Any],
    *,
    image_path: str,
    image_sha256: str,
    frame_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a non-authoritative SAM 2 prompt from prediction artifacts only."""
    probability = np.asarray(probability, dtype=np.float32)
    water = np.asarray(water_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if probability.ndim != 2 or water.shape != probability.shape or unknown.shape != probability.shape:
        raise ValueError("probability, water_mask and unknown_mask must be same-shape 2-D arrays")
    if int(config.get("connectivity", 8)) not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = probability.shape
    hard_reasons: list[str] = []
    diagnostic_reasons: list[str] = []
    if not np.isfinite(probability).all():
        hard_reasons.append("nonfinite_probability")
    if np.any(water & unknown):
        hard_reasons.append("water_unknown_overlap")
    water = water & ~unknown
    if temporal_quality_gate.get("status") == "reject":
        hard_reasons.append("temporal_quality_gate_reject")
    elif temporal_quality_gate.get("status") != "pass":
        diagnostic_reasons.append("temporal_quality_gate_not_pass")
    if not np.any(water):
        hard_reasons.append("predicted_water_mask_empty")

    labels, components = _component_records(water, probability, int(config.get("connectivity", 8)))
    selected_component = np.zeros_like(water)
    selected_record: dict[str, Any] | None = components[0] if components else None
    ambiguous_count = 0
    if selected_record is not None:
        selected_component = labels == int(selected_record["label"])
        if selected_record["area_pixels"] < int(config["min_component_area_px"]):
            hard_reasons.append("primary_component_too_small")
        threshold = selected_record["probability_mass"] * float(
            config["ambiguous_component_probability_mass_ratio"]
        )
        ambiguous_count = sum(item["probability_mass"] >= threshold for item in components[1:])
        if ambiguous_count > int(config["max_ambiguous_component_count"]):
            hard_reasons.append("ambiguous_temporal_water_components")

    box: list[int] | None = None
    box_area_fraction = 0.0
    box_border_touch_ratio = 0.0
    positive_points: list[list[int]] = []
    positive_boundary_distances: list[float] = []
    negative_points: list[list[int]] = []
    negative_sources: list[str] = []
    negative_sectors: list[int] = []
    if selected_record is not None:
        x, y, component_width, component_height = selected_record["bbox_xywh"]
        margin = int(config["box_margin_px"])
        box = [
            max(0, x - margin),
            max(0, y - margin),
            min(width - 1, x + component_width - 1 + margin),
            min(height - 1, y + component_height - 1 + margin),
        ]
        box_area_fraction = float(
            ((box[2] - box[0] + 1) * (box[3] - box[1] + 1)) / max(1, width * height)
        )
        box_border_touch_ratio = float(sum((
            box[0] == 0, box[1] == 0, box[2] == width - 1, box[3] == height - 1,
        )) / 4.0)
        if box_area_fraction > float(config["max_box_area_fraction"]):
            hard_reasons.append("box_area_abnormally_large")
        if box_border_touch_ratio > float(config["max_box_border_touch_ratio"]):
            hard_reasons.append("box_excessively_touches_image_border")
        elif box_border_touch_ratio > 0:
            diagnostic_reasons.append("box_touches_image_border")
        positive_points, positive_boundary_distances = _select_positive_points(
            selected_component,
            probability,
            int(config["target_positive_points"]),
            float(config["min_positive_boundary_distance_px"]),
            float(config["min_positive_spacing_px"]),
        )
        negative_points, negative_sources, negative_sectors = _select_negative_points(
            water, unknown, selected_component, probability, classifications, config
        )
    if len(positive_points) < int(config["min_positive_points"]):
        hard_reasons.append("insufficient_safe_positive_points")
    if len(negative_points) < int(config["min_negative_points"]):
        hard_reasons.append("insufficient_trusted_negative_points")
    if len(negative_sectors) < int(config["min_negative_direction_sectors"]):
        hard_reasons.append("insufficient_negative_direction_coverage")
    if any(unknown[y, x] or not selected_component[y, x] for x, y in positive_points):
        hard_reasons.append("invalid_positive_point")
    if any(unknown[y, x] or water[y, x] for x, y in negative_points):
        hard_reasons.append("invalid_negative_point")
    if box is None:
        hard_reasons.append("invalid_box")
    elif any(not (box[0] <= x <= box[2] and box[1] <= y <= box[3]) for x, y in positive_points):
        hard_reasons.append("box_does_not_contain_positive_points")

    hard_reasons = list(dict.fromkeys(hard_reasons))
    diagnostic_reasons = list(dict.fromkeys(diagnostic_reasons))
    status = "reject" if hard_reasons else "diagnostic_only" if diagnostic_reasons else "pass"
    all_reasons = hard_reasons + diagnostic_reasons
    prompt = {
        "schema_version": config["schema_version"],
        "algorithm_version": config["algorithm_version"],
        "image_path": str(image_path),
        "image_sha256": str(image_sha256),
        "image_width": int(width),
        "image_height": int(height),
        "frame_index": int(frame_index),
        "prompt_source": "temporal_water_evidence_v1",
        "semantic_label": "unknown_candidate",
        "authoritative": False,
        "box_xyxy": box,
        "positive_points_xy": positive_points,
        "negative_points_xy": negative_points,
        "prompt_quality_status": status,
        "prompt_quality_reasons": all_reasons,
        "ground_truth_used": False,
        "eligible_for_downstream": False,
    }
    diagnostics = {
        "data_role": "prediction_diagnostic",
        "component_count": len(components),
        "components": components,
        "selected_component_label": selected_record["label"] if selected_record else None,
        "selected_component_area_pixels": selected_record["area_pixels"] if selected_record else 0,
        "selected_component_probability_mass": selected_record["probability_mass"] if selected_record else 0.0,
        "ambiguous_component_count": int(ambiguous_count),
        "box_area_fraction": box_area_fraction,
        "box_border_touch_ratio": box_border_touch_ratio,
        "positive_point_count": len(positive_points),
        "positive_boundary_distances_px": positive_boundary_distances,
        "negative_point_count": len(negative_points),
        "negative_point_sources": negative_sources,
        "negative_direction_sector_count": len(negative_sectors),
        "negative_direction_sectors": negative_sectors,
        "unknown_fraction": float(np.mean(unknown)),
        "temporal_quality_gate_status": temporal_quality_gate.get("status", "unavailable"),
        "status": status,
        "hard_reasons": hard_reasons,
        "diagnostic_reasons": diagnostic_reasons,
        "geometry_diagnostic_readiness": "ready" if status == "pass" else status,
        "ground_truth_used": False,
        "eligible_for_downstream": False,
    }
    return prompt, diagnostics
