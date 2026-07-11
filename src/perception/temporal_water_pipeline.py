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


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


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
    features = extract_all_track_features(tracks)
    classifications = classify_tracks(features, detector_config["classification"])
    evidence, evidence_diagnostics = build_temporal_water_evidence(
        classifications,
        (loader["height"], loader["width"]),
        detector_config["evidence"],
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
        "water_mask_time_stability": stability,
        "feature_score_separation": feature_separation,
        "mode": mode,
        "ground_truth_used": False,
    }
