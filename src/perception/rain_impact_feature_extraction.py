#!/usr/bin/env python3
"""Explainable temporal and spatial features for candidate tracks."""

from __future__ import annotations

import numpy as np


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(values.size, dtype=np.float64), values, 1)[0])


def _smooth(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.ones(3) / 3.0, mode="valid")


def extract_track_features(track: dict, context: dict | None = None) -> dict:
    areas = np.asarray(track["area_sequence"], dtype=np.float64)
    radii = np.asarray(track["equivalent_radius_sequence"], dtype=np.float64)
    intensity = np.asarray(track["intensity_sequence"], dtype=np.float64)
    peak = int(np.argmax(intensity)) if intensity.size else 0
    after_peak = intensity[peak:]
    decay_rate = max(0.0, -_slope(after_peak) / max(float(np.max(intensity)) if intensity.size else 1.0, 1e-6))
    radius_differences = np.diff(radii)
    monotonicity = float(np.mean(radius_differences >= -0.05)) if radius_differences.size else 0.0
    post_persistence = float((intensity.size - peak - 1) / max(1, intensity.size - 1))
    polarities = track["polarity_sequence"]
    polarity_changes = sum(left != right for left, right in zip(polarities[:-1], polarities[1:]))
    observations = track["observations"]
    ringness = float(np.mean([item["ringness"] for item in observations]))
    compactness = float(np.mean([item["compactness"] for item in observations]))
    temporal_energy = float(np.sum(intensity * areas))
    before_energy = float(np.sum(intensity[: peak + 1]))
    after_energy = float(np.sum(intensity[peak + 1 :]))
    asymmetry = float((after_energy - before_energy) / max(after_energy + before_energy, 1e-6))
    early_count = max(2, min(5, radii.size))
    radius_steps = np.diff(radii)
    positive_steps = radius_steps[radius_steps > 0]
    expansion_consistency = 0.0
    if positive_steps.size:
        expansion_consistency = float(1.0 / (1.0 + np.std(positive_steps) / max(np.mean(positive_steps), 1e-6)))
    ring_values = np.asarray([item["ringness"] for item in observations], dtype=np.float64)
    exposure_correlation = 0.0
    if context and context.get("exposure_offsets") is not None and intensity.size >= 2:
        frame_indices = np.asarray([item["frame_index"] for item in observations], dtype=np.int64)
        exposure = np.asarray(context["exposure_offsets"], dtype=np.float64)[frame_indices]
        if np.std(exposure) > 1e-9 and np.std(intensity) > 1e-9:
            exposure_correlation = float(np.corrcoef(exposure, intensity)[0, 1])
    return {
        "track_id": track["track_id"],
        "duration_frames": int(track["duration_frames"]),
        "valid_observation_count": int(track["valid_observation_count"]),
        "peak_frame_offset": peak,
        "maximum_area": int(np.max(areas)) if areas.size else 0,
        "area_growth_slope": _slope(areas),
        "radius_growth_slope": _slope(radii),
        "expansion_monotonicity": monotonicity,
        "decay_rate": decay_rate,
        "post_peak_persistence": post_persistence,
        "center_drift": float(track["center_drift"]),
        "spatial_compactness": compactness,
        "radial_energy_profile": ringness,
        "ringness": ringness,
        "polarity_change_count": int(polarity_changes),
        "temporal_energy": temporal_energy,
        "temporal_asymmetry": asymmetry,
        "early_radius_growth_slope": _slope(radii[:early_count]),
        "post_peak_area_integral": float(np.sum(areas[peak:])),
        "ring_energy_duration_fraction": float(np.mean(ring_values > 1.05)) if ring_values.size else 0.0,
        "radius_expansion_consistency": expansion_consistency,
        "polarity_switch_rate": float(polarity_changes / max(1, len(polarities) - 1)),
        "smoothed_area_growth_slope": _slope(_smooth(areas)),
        "smoothed_radius_growth_slope": _slope(_smooth(radii)),
        "observation_fill_ratio": float(track["valid_observation_count"] / max(1, track["duration_frames"])),
        "local_dynamic_density": float((context or {}).get("local_dynamic_density", 0.0)),
        "neighbor_track_count": int((context or {}).get("neighbor_track_count", 0)),
        "exposure_residual_correlation": exposure_correlation,
        "center_mean": track["center_mean"],
        "bbox_union": track["bbox_union"],
        "start_frame": track["start_frame"],
        "end_frame": track["end_frame"],
    }


def extract_all_track_features(
    tracks: list[dict], exposure_offsets: np.ndarray | None = None,
) -> list[dict]:
    results = []
    centers = np.asarray([track["center_mean"] for track in tracks], dtype=np.float64)
    starts = np.asarray([track["start_frame"] for track in tracks], dtype=np.int64)
    ends = np.asarray([track["end_frame"] for track in tracks], dtype=np.int64)
    for index, track in enumerate(tracks):
        distances = np.linalg.norm(centers - centers[index], axis=1)
        nearby = (distances <= 35.0) & (np.arange(len(tracks)) != index)
        temporal_overlap = np.maximum(
            0, np.minimum(ends[index], ends) - np.maximum(starts[index], starts) + 1,
        )
        neighbors = int(np.count_nonzero(nearby))
        overlapping_neighbors = int(np.count_nonzero(nearby & (temporal_overlap > 0)))
        context = {
            "neighbor_track_count": neighbors,
            "local_dynamic_density": overlapping_neighbors / max(1.0, track["duration_frames"]),
            "exposure_offsets": exposure_offsets,
        }
        results.append(extract_track_features(track, context))
    return results
