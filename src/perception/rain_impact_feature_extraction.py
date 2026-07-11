#!/usr/bin/env python3
"""Explainable temporal and spatial features for candidate tracks."""

from __future__ import annotations

import numpy as np


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(values.size, dtype=np.float64), values, 1)[0])


def extract_track_features(track: dict) -> dict:
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
        "center_mean": track["center_mean"],
        "bbox_union": track["bbox_union"],
        "start_frame": track["start_frame"],
        "end_frame": track["end_frame"],
    }


def extract_all_track_features(tracks: list[dict]) -> list[dict]:
    return [extract_track_features(track) for track in tracks]
