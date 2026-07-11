#!/usr/bin/env python3
"""Conservative sparse water evidence accumulation from classified tracks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def build_temporal_water_evidence(
    classifications: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    height, width = image_shape
    evidence = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    water_tracks = []
    dry_tracks = []
    for item in classifications:
        if item["classification"] == "water_ripple":
            water_tracks.append(item)
            center_u, center_v = item["center_mean"]
            estimated_radius = np.sqrt(max(float(item["maximum_area"]), 1.0) / np.pi)
            sigma = max(float(config["minimum_kernel_sigma_px"]), min(float(config["maximum_propagation_radius_px"]) / 2.5, estimated_radius * 1.3))
            support = min(float(config["maximum_propagation_radius_px"]), sigma * 2.5)
            x0, x1 = max(0, int(center_u - support)), min(width, int(center_u + support) + 1)
            y0, y1 = max(0, int(center_v - support)), min(height, int(center_v + support) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            kernel = np.exp(-0.5 * (((xx - center_u) ** 2 + (yy - center_v) ** 2) / (sigma * sigma))).astype(np.float32)
            weight = float(item["confidence"]) * (0.5 + 0.5 * min(1.0, item["duration_frames"] / 15.0))
            evidence[y0:y1, x0:x1] += kernel * weight
            count[y0:y1, x0:x1] += (kernel >= 0.10).astype(np.float32)
        elif item["classification"] == "dry_splash":
            dry_tracks.append(item)
    probability = 1.0 - np.exp(-evidence / max(float(config["probability_scale"]), 1e-6))
    unknown = count < float(config["unknown_evidence_threshold"])
    water = (probability >= float(config["water_probability_threshold"])) & ~unknown
    kernel_size = int(config.get("morphology_close_kernel", 0))
    if kernel_size > 1 and np.any(water):
        water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
    unknown &= ~water
    diagnostics = {
        "water_track_count": len(water_tracks),
        "dry_track_count": len(dry_tracks),
        "uncertain_track_count": len(classifications) - len(water_tracks) - len(dry_tracks),
        "evidence_coverage_fraction": float(np.mean(count > 0)),
        "unknown_fraction": float(np.mean(unknown)),
        "predicted_water_fraction": float(np.mean(water)),
        "maximum_probability": float(np.max(probability)) if probability.size else 0.0,
        "evidence_concentration": float(np.max(evidence) / max(float(np.sum(evidence)), 1e-6)),
        "unknown_region_semantics": "no_temporal_evidence_not_confirmed_dry",
    }
    return {
        "predicted_water_probability": probability.astype(np.float32),
        "predicted_water_mask": water,
        "predicted_unknown_mask": unknown,
        "evidence_count_map": count.astype(np.float32),
    }, diagnostics
