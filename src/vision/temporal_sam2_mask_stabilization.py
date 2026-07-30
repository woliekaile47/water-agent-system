"""Deterministic short-window stabilization for frozen SAM 2 video masks.

The module is deliberately independent from SAM 2 inference and geometry.  It
only combines already-frozen binary masks, never mutates them, and fails closed
when the configured positive-point evidence is lost.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import cv2
import numpy as np


def validate_stabilization_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the fixed, prediction-side stabilization config."""
    window = config.get("window", {})
    positive_support = config.get("positive_support", {})
    size = int(window.get("size", 0))
    mode = str(window.get("mode", ""))
    support_fraction = float(window.get("support_fraction", math.nan))
    edge_policy = str(window.get("edge_policy", ""))
    minimum_count = int(positive_support.get("minimum_count", 0))
    failure_policy = str(positive_support.get("failure_policy", ""))
    if mode != "centered":
        raise ValueError("mask stabilization window mode must be centered")
    if size < 1 or size % 2 != 1:
        raise ValueError("mask stabilization window size must be a positive odd integer")
    if not 0.0 < support_fraction <= 1.0:
        raise ValueError("mask stabilization support_fraction must be in (0, 1]")
    if edge_policy != "available_frames":
        raise ValueError("mask stabilization edge_policy must be available_frames")
    if positive_support.get("required") is not True:
        raise ValueError("mask stabilization positive-point support must be required")
    if minimum_count < 1:
        raise ValueError("mask stabilization minimum positive support must be at least one")
    if failure_policy != "empty_mask_reject":
        raise ValueError("mask stabilization failure_policy must be empty_mask_reject")
    return {
        "window_size": size,
        "window_radius": size // 2,
        "support_fraction": support_fraction,
        "minimum_positive_count": minimum_count,
    }


def _normalize_masks(masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not masks:
        raise ValueError("mask sequence is empty")
    normalized = [np.asarray(mask, dtype=bool).copy() for mask in masks]
    shape = normalized[0].shape
    if len(shape) != 2:
        raise ValueError("SAM2 masks must be two-dimensional")
    if any(mask.shape != shape for mask in normalized):
        raise ValueError("all SAM2 masks must have the same shape")
    return normalized


def _positive_pixels(
    positive_points_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    points = np.asarray(positive_points_xy, dtype=np.float64)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("positive points must be finite [x,y] coordinates")
    height, width = image_shape
    pixels: list[tuple[int, int]] = []
    for x, y in points:
        column = int(round(float(x)))
        row = int(round(float(y)))
        if 0 <= row < height and 0 <= column < width:
            pixels.append((row, column))
    return pixels


def binary_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return deterministic binary IoU, defining two empty masks as identical."""
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask IoU inputs must have the same shape")
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def mask_content_sha256(mask: np.ndarray) -> str:
    """Hash binary mask shape and pixels independently from any file container."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask content hash requires a two-dimensional mask")
    digest = hashlib.sha256()
    digest.update(b"binary-mask-content-v1\0")
    digest.update(np.asarray(source.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(source, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _select_positive_supported_component(
    candidate: np.ndarray,
    positive_pixels: list[tuple[int, int]],
) -> dict[str, Any]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    component_count = int(labels_count - 1)
    if component_count == 0:
        return {
            "mask": np.zeros_like(candidate, dtype=bool),
            "component_count": 0,
            "selected_label": None,
            "selected_positive_count": 0,
            "selected_area_pixels": 0,
            "excluded_candidate_pixels": 0,
        }
    ranked: list[tuple[int, int, int]] = []
    for label in range(1, labels_count):
        positive_count = sum(
            int(labels[row, column] == label) for row, column in positive_pixels
        )
        area = int(stats[label, cv2.CC_STAT_AREA])
        ranked.append((positive_count, area, -label))
    positive_count, area, negative_label = max(ranked)
    selected_label = -negative_label
    selected = labels == selected_label
    return {
        "mask": selected,
        "component_count": component_count,
        "selected_label": int(selected_label),
        "selected_positive_count": int(positive_count),
        "selected_area_pixels": int(area),
        "excluded_candidate_pixels": int(np.count_nonzero(candidate & ~selected)),
    }


def stabilize_frozen_mask_sequence(
    frozen_masks: Sequence[np.ndarray],
    frozen_positive_points_xy: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply fixed centered-window support and return one result per input mask.

    A non-empty window whose consensus loses all required positive evidence is
    replaced by an empty mask and marked ineligible for geometry.  A genuinely
    empty input window remains empty without inventing water.
    """
    settings = validate_stabilization_config(config)
    source_masks = _normalize_masks(frozen_masks)
    positive_pixels = _positive_pixels(frozen_positive_points_xy, source_masks[0].shape)
    results: list[dict[str, Any]] = []
    for index, raw_mask in enumerate(source_masks):
        start = max(0, index - settings["window_radius"])
        stop = min(len(source_masks), index + settings["window_radius"] + 1)
        window = source_masks[start:stop]
        window_count = len(window)
        required_count = int(math.ceil(settings["support_fraction"] * window_count - 1e-12))
        if not np.any(raw_mask):
            stabilized = np.zeros_like(raw_mask, dtype=bool)
            results.append({
                "mask": stabilized,
                "status": "reject_empty_raw_mask",
                "eligible_for_geometry": False,
                "fail_closed": True,
                "failure_reason": "mask_stabilization_empty_raw_mask",
                "window_start_offset": start,
                "window_stop_offset_exclusive": stop,
                "window_frame_count": window_count,
                "required_support_count": required_count,
                "supported_positive_count": 0,
                "candidate_component_count": 0,
                "selected_component_area_pixels": 0,
                "excluded_candidate_pixels": 0,
                "raw_area_pixels": 0,
                "stabilized_area_pixels": 0,
                "raw_stabilized_iou": 1.0,
                "stabilized_mask_content_sha256": mask_content_sha256(stabilized),
            })
            continue

        support_count = np.sum(np.stack(window, axis=0), axis=0, dtype=np.uint16)
        candidate = support_count >= required_count
        component = _select_positive_supported_component(candidate, positive_pixels)
        selected = component["mask"]
        supported_positive_count = component["selected_positive_count"]
        fail_closed = (
            not np.any(selected)
            or supported_positive_count < settings["minimum_positive_count"]
        )
        if fail_closed:
            stabilized = np.zeros_like(selected, dtype=bool)
            status = "reject_positive_support_lost"
            failure_reason = "mask_stabilization_positive_support_lost"
        else:
            stabilized = selected.copy()
            status = "stabilized"
            failure_reason = None
        results.append({
            "mask": stabilized,
            "status": status,
            "eligible_for_geometry": not fail_closed,
            "fail_closed": fail_closed,
            "failure_reason": failure_reason,
            "window_start_offset": start,
            "window_stop_offset_exclusive": stop,
            "window_frame_count": window_count,
            "required_support_count": required_count,
            "supported_positive_count": int(supported_positive_count),
            "candidate_component_count": component["component_count"],
            "selected_component_area_pixels": component["selected_area_pixels"],
            "excluded_candidate_pixels": component["excluded_candidate_pixels"],
            "raw_area_pixels": int(np.count_nonzero(raw_mask)),
            "stabilized_area_pixels": int(np.count_nonzero(stabilized)),
            "raw_stabilized_iou": binary_mask_iou(raw_mask, stabilized),
            "stabilized_mask_content_sha256": mask_content_sha256(stabilized),
        })
    return results
