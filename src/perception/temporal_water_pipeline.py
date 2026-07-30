#!/usr/bin/env python3
"""GT-free temporal water prediction pipeline accepting only frames/."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.perception.dynamic_event_detection import detect_dynamic_event_candidates
from src.perception.dynamic_event_tracking import track_dynamic_events
from src.perception.rain_impact_classifier import classify_tracks
from src.perception.rain_impact_feature_extraction import extract_all_track_features
from src.perception.temporal_frame_preprocessing import load_detector_frames, preprocess_temporal_frames
from src.perception.temporal_water_evidence import build_temporal_water_evidence


TEMPORAL_SUPPORT_WINDOW_COUNT = 3


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def build_temporal_support_fraction(
    classifications: list[dict[str, Any]],
    image_shape: tuple[int, int],
    evidence_config: dict[str, Any],
    frame_count: int,
    window_count: int = TEMPORAL_SUPPORT_WINDOW_COUNT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure spatial evidence recurrence across fixed, GT-free time windows."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if window_count < 1:
        raise ValueError("window_count must be positive")
    edges = np.linspace(0, frame_count, window_count + 1, dtype=np.int64)
    window_classifications: list[list[dict[str, Any]]] = [
        [] for _ in range(window_count)
    ]
    for item in classifications:
        start = int(item.get("start_frame", 0))
        end = int(item.get("end_frame", start))
        center = int(np.clip((start + end) // 2, 0, frame_count - 1))
        window_index = int(np.searchsorted(edges[1:], center, side="right"))
        window_index = min(window_index, window_count - 1)
        window_classifications[window_index].append(item)

    support_count = np.zeros(image_shape, dtype=np.float32)
    water_track_counts: list[int] = []
    for items in window_classifications:
        window_evidence, window_diagnostics = build_temporal_water_evidence(
            items,
            image_shape,
            evidence_config,
        )
        support_count += (
            np.asarray(window_evidence["evidence_count_map"], dtype=np.float32) > 0
        )
        water_track_counts.append(int(window_diagnostics["water_track_count"]))

    support_fraction = support_count / float(window_count)
    windows = [
        {
            "window_index": index,
            "start_frame_inclusive": int(edges[index]),
            "end_frame_exclusive": int(edges[index + 1]),
            "classification_count": len(window_classifications[index]),
            "water_track_count": water_track_counts[index],
        }
        for index in range(window_count)
    ]
    return support_fraction.astype(np.float32), {
        "window_count": int(window_count),
        "windows": windows,
        "maximum_support_fraction": float(np.max(support_fraction))
        if support_fraction.size
        else 0.0,
        "recurrent_support_fraction": float(
            np.mean(support_fraction >= (2.0 / window_count))
        ),
        "ground_truth_used": False,
    }


def run_temporal_prediction(
    frames_dir: str,
    detector_config: dict[str, Any],
    mode: str = "full",
) -> dict[str, Any]:
    """Run prediction without a GT or metadata argument by construction."""
    frames, loader = load_detector_frames(frames_dir)
    preprocessed, preprocessing = preprocess_temporal_frames(
        frames,
        detector_config["preprocessing"],
        mode=mode,
        shuffle_seed=int(detector_config["ablation"]["shuffle_seed"]),
    )
    candidates, candidate_diagnostics = detect_dynamic_event_candidates(
        preprocessed["absolute_residual"],
        preprocessed["signed_residual"],
        detector_config["candidates"],
    )
    tracks = track_dynamic_events(candidates, detector_config["tracking"])
    features = extract_all_track_features(tracks, preprocessed.get("exposure_offsets"))
    classifications = classify_tracks(features, detector_config["classification"])
    evidence, evidence_diagnostics = build_temporal_water_evidence(
        classifications,
        (loader["height"], loader["width"]),
        detector_config["evidence"],
    )
    temporal_support_fraction, temporal_support_diagnostics = (
        build_temporal_support_fraction(
            classifications,
            (loader["height"], loader["width"]),
            detector_config["evidence"],
            int(loader["frame_count"]),
        )
    )
    midpoint = max(1, loader["frame_count"] // 2)
    early_classifications = [item for item in classifications if item["start_frame"] < midpoint]
    early_evidence, _ = build_temporal_water_evidence(
        early_classifications,
        (loader["height"], loader["width"]),
        detector_config["evidence"],
    )
    stability = _mask_iou(early_evidence["predicted_water_mask"], evidence["predicted_water_mask"])
    score_margins = [abs(item["water_ripple_score"] - item["dry_splash_score"]) for item in classifications]
    feature_separation = float(np.mean(score_margins)) if score_margins else 0.0
    return {
        "loader": loader,
        "preprocessed": preprocessed,
        "preprocessing_diagnostics": preprocessing,
        "candidates_by_frame": candidates,
        "candidate_diagnostics": candidate_diagnostics,
        "tracks": tracks,
        "features": features,
        "classifications": classifications,
        "evidence": evidence,
        "evidence_diagnostics": evidence_diagnostics,
        "temporal_support_fraction": temporal_support_fraction,
        "temporal_support_diagnostics": temporal_support_diagnostics,
        "water_mask_time_stability": stability,
        "feature_score_separation": feature_separation,
        "mode": mode,
        "ground_truth_used": False,
    }


def run_temporal_model_prediction(
    frames_dir: str, model_dir: str, detector_config: dict[str, Any],
    training_config: dict[str, Any], thresholds: dict[str, float], mode: str = "full",
) -> dict[str, Any]:
    """Run rule/model/hybrid inference from RGB frames and frozen model files only."""
    from src.perception.temporal_event_classifier_inference import (
        build_hybrid_classifications, infer_track_classifications,
    )
    from src.perception.temporal_event_classifier_model import load_model
    from src.perception.temporal_water_evidence import (
        build_model_water_evidence, build_rule_preserving_hybrid_evidence,
    )

    baseline = run_temporal_prediction(frames_dir, detector_config, mode)
    model = load_model(model_dir)
    model_classifications = infer_track_classifications(
        baseline["features"], model, thresholds, detector_config["classification"],
    )
    hybrid_classifications = build_hybrid_classifications(
        model_classifications, thresholds, float(training_config["evidence"]["hybrid_model_weight"]),
    )
    shape = (baseline["loader"]["height"], baseline["loader"]["width"])
    model_evidence, model_diagnostics = build_model_water_evidence(
        model_classifications, shape, training_config["evidence"], "model_water_probability",
    )
    hybrid_model_evidence, hybrid_model_diagnostics = build_model_water_evidence(
        hybrid_classifications, shape, training_config["evidence"], "hybrid_water_probability",
    )
    hybrid_evidence, hybrid_diagnostics = build_rule_preserving_hybrid_evidence(
        baseline["evidence"], hybrid_model_evidence, hybrid_model_diagnostics,
    )
    midpoint = max(1, baseline["loader"]["frame_count"] // 2)
    early = [item for item in model_classifications if item["start_frame"] < midpoint]
    early_evidence, _ = build_model_water_evidence(
        early, shape, training_config["evidence"], "model_water_probability",
    )
    return {
        "baseline": baseline, "model_classifications": model_classifications,
        "hybrid_classifications": hybrid_classifications,
        "model_evidence": model_evidence, "model_evidence_diagnostics": model_diagnostics,
        "hybrid_evidence": hybrid_evidence, "hybrid_evidence_diagnostics": hybrid_diagnostics,
        "model_window_mask_iou": _mask_iou(
            early_evidence["predicted_water_mask"], model_evidence["predicted_water_mask"],
        ),
        "ground_truth_used": False,
    }
