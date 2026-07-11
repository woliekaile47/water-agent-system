#!/usr/bin/env python3
"""Sequence-level dataset assembly for the Phase 2C-2B event classifier."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.perception.rain_impact_feature_extraction import extract_all_track_features
from src.perception.temporal_event_label_matching import load_training_event_labels, match_tracks_to_events
from src.perception.temporal_frame_preprocessing import load_detector_frames, preprocess_temporal_frames


FEATURE_NAMES = [
    "duration_frames", "peak_frame_offset", "maximum_area", "area_growth_slope",
    "radius_growth_slope", "expansion_monotonicity", "decay_rate",
    "post_peak_persistence", "center_drift", "spatial_compactness",
    "radial_energy_profile", "ringness", "polarity_change_count", "temporal_energy",
    "temporal_asymmetry", "early_radius_growth_slope", "post_peak_area_integral",
    "ring_energy_duration_fraction", "radius_expansion_consistency", "polarity_switch_rate",
    "smoothed_area_growth_slope", "smoothed_radius_growth_slope", "observation_fill_ratio",
    "local_dynamic_density", "neighbor_track_count", "exposure_residual_correlation",
]


def sequence_identity(sequence_dir: str | Path) -> dict[str, Any]:
    path = Path(sequence_dir).expanduser().resolve()
    return {
        "path": str(path), "case_id": path.parents[1].name, "rain_level": path.parent.name,
        "seed": int(path.name.removeprefix("seed_")),
    }


def discover_sequence_splits(data_root: str | Path, split_config: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    seed_sets = {
        "train": set(map(int, split_config["train_seeds"])),
        "validation": set(map(int, split_config["validation_seeds"])),
        "test": set(map(int, split_config["test_seeds"])),
    }
    overlap = (seed_sets["train"] & seed_sets["validation"]) | (seed_sets["train"] & seed_sets["test"]) | (seed_sets["validation"] & seed_sets["test"])
    if overlap:
        raise ValueError(f"seed overlap across sequence splits: {sorted(overlap)}")
    splits: dict[str, list[dict]] = {name: [] for name in seed_sets}
    for frames_dir in sorted(root.glob("*/*/seed_*/frames")):
        identity = sequence_identity(frames_dir.parent)
        for name, seeds in seed_sets.items():
            if identity["seed"] in seeds:
                splits[name].append(identity)
                break
    manifest = {
        "split_unit": "whole_sequence_and_seed",
        "train_sequences": splits["train"],
        "validation_sequences": splits["validation"],
        "test_sequences": splits["test"],
        "seed_overlap_check": {"overlap": sorted(overlap), "passed": not overlap},
        "case_distribution": {name: dict(Counter(item["case_id"] for item in values)) for name, values in splits.items()},
        "rain_distribution": {name: dict(Counter(item["rain_level"] for item in values)) for name, values in splits.items()},
    }
    return splits, manifest


def load_cached_tracks(sequence_dir: str | Path, data_root: str | Path, baseline_output_root: str | Path) -> list[dict]:
    sequence = Path(sequence_dir).expanduser().resolve()
    relative = sequence.relative_to(Path(data_root).expanduser().resolve())
    path = Path(baseline_output_root).expanduser().resolve() / relative / "event_tracks.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)["tracks"]


def build_labeled_sequence_samples(
    sequence: dict[str, Any], data_root: str | Path, baseline_output_root: str | Path,
    detector_config: dict[str, Any], matching_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence_dir = Path(sequence["path"])
    try:
        tracks = load_cached_tracks(sequence_dir, data_root, baseline_output_root)
    except FileNotFoundError:
        from src.perception.temporal_water_pipeline import run_temporal_prediction
        tracks = run_temporal_prediction(str(sequence_dir / "frames"), detector_config, "full")["tracks"]
    frames, _ = load_detector_frames(sequence_dir / "frames")
    preprocessed, _ = preprocess_temporal_frames(frames, detector_config["preprocessing"])
    features = extract_all_track_features(tracks, preprocessed["exposure_offsets"])
    events = load_training_event_labels(sequence_dir)
    labels, diagnostics = match_tracks_to_events(tracks, events, matching_config)
    labels_by_track = {item["track_id"]: item for item in labels}
    samples = []
    for feature in features:
        label = labels_by_track[feature["track_id"]]
        vector = np.asarray([feature.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
        nonfinite = ~np.isfinite(vector)
        vector[nonfinite] = 0.0
        samples.append({
            "sequence": sequence, "track_id": feature["track_id"], "features": feature,
            "feature_vector": vector, "nonfinite_feature_count": int(np.count_nonzero(nonfinite)),
            **label,
        })
    return samples, diagnostics
