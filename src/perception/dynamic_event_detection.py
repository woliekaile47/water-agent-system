#!/usr/bin/env python3
"""Local dynamic-event candidate extraction from temporal residuals."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _component_shape(mask: np.ndarray, residual: np.ndarray, signed: np.ndarray) -> dict[str, float]:
    rows, cols = np.where(mask)
    area = int(rows.size)
    center_row, center_col = float(np.mean(rows)), float(np.mean(cols))
    if area >= 3:
        covariance = np.cov(np.column_stack((rows, cols)), rowvar=False)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-9))
        eccentricity = float(np.sqrt(1.0 - eigenvalues[0] / eigenvalues[-1]))
    else:
        eccentricity = 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, 1e-9))
    distances = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    split = float(np.median(distances)) if distances.size else 0.0
    energy = residual[mask].astype(np.float64)
    outer = energy[distances >= split]
    inner = energy[distances < split]
    ringness = float(np.mean(outer) / max(float(np.mean(inner)) if inner.size else 0.0, 1e-6))
    signed_values = signed[mask]
    polarity_mean = float(np.mean(signed_values))
    polarity = "positive" if polarity_mean > 0.5 else "negative" if polarity_mean < -0.5 else "mixed"
    return {
        "center_u": center_col,
        "center_v": center_row,
        "area": area,
        "mean_residual": float(np.mean(energy)),
        "max_residual": float(np.max(energy)),
        "polarity_mean": polarity_mean,
        "polarity": polarity,
        "eccentricity": eccentricity,
        "compactness": compactness,
        "ringness": ringness,
        "equivalent_radius": float(np.sqrt(area / np.pi)),
    }


def detect_dynamic_event_candidates(
    absolute_residual: np.ndarray,
    signed_residual: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    absolute = np.asarray(absolute_residual, dtype=np.float32)
    signed = np.asarray(signed_residual, dtype=np.float32)
    if absolute.shape != signed.shape or absolute.ndim != 3:
        raise ValueError("absolute and signed residuals must be matching TxHxW arrays")
    candidates_by_frame: list[list[dict[str, Any]]] = []
    candidate_pixels = 0
    thresholds = []
    for frame_index in range(absolute.shape[0]):
        residual = absolute[frame_index]
        threshold = max(
            float(config["residual_threshold"]),
            float(np.percentile(residual, float(config["residual_percentile"]))),
        )
        thresholds.append(threshold)
        binary = (residual >= threshold).astype(np.uint8)
        close_kernel = int(config.get("morphology_close_kernel", 3))
        if close_kernel > 1:
            kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        iterations = int(config.get("morphology_dilate_iterations", 0))
        if iterations > 0:
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=iterations)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        frame_candidates = []
        for label in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[label]]
            if area < int(config["min_area_px"]) or area > int(config["max_area_px"]):
                continue
            aspect = max(width / max(1, height), height / max(1, width))
            if aspect > float(config["max_aspect_ratio"]):
                continue
            component = labels[y : y + height, x : x + width] == label
            local_residual = residual[y : y + height, x : x + width]
            local_signed = signed[frame_index, y : y + height, x : x + width]
            features = _component_shape(component, local_residual, local_signed)
            features["center_u"] += x
            features["center_v"] += y
            features.update({
                "candidate_id": f"f{frame_index:06d}_c{len(frame_candidates):04d}",
                "frame_index": frame_index,
                "bbox": [x, y, x + width - 1, y + height - 1],
                "aspect_ratio": float(aspect),
            })
            frame_candidates.append(features)
            candidate_pixels += area
        candidates_by_frame.append(frame_candidates)
    diagnostics = {
        "candidate_count": int(sum(len(items) for items in candidates_by_frame)),
        "candidate_count_by_frame": [len(items) for items in candidates_by_frame],
        "candidate_pixel_fraction": float(candidate_pixels / max(1, np.prod(absolute.shape))),
        "threshold_median": float(np.median(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "ground_truth_used": False,
    }
    return candidates_by_frame, diagnostics
