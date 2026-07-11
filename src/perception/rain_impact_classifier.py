#!/usr/bin/env python3
"""Unified rule-based dry-splash / water-ripple classifier."""

from __future__ import annotations

from typing import Any


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def classify_track(features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    duration = float(features["duration_frames"])
    observations = float(features["valid_observation_count"])
    duration_score = _clip((duration - 3.0) / 12.0)
    observation_score = _clip((observations - 2.0) / 8.0)
    growth_score = _clip(features["radius_growth_slope"] / 0.65)
    persistence_score = _clip(features["post_peak_persistence"])
    monotonic_score = _clip(features["expansion_monotonicity"])
    ring_score = _clip((features["ringness"] - 0.9) / 1.5)
    stability_score = 1.0 - _clip(features["center_drift"] / 12.0)
    water_score = (
        0.23 * duration_score + 0.12 * observation_score + 0.20 * growth_score
        + 0.16 * persistence_score + 0.12 * monotonic_score + 0.10 * ring_score
        + 0.07 * stability_score
    )
    short_score = 1.0 - _clip((duration - 2.0) / 6.0)
    low_growth = 1.0 - _clip(max(features["radius_growth_slope"], 0.0) / 0.45)
    quick_decay = _clip(features["decay_rate"] / 0.15)
    low_persistence = 1.0 - persistence_score
    irregularity = 1.0 - _clip(features["spatial_compactness"])
    dry_score = 0.32 * short_score + 0.25 * low_growth + 0.18 * quick_decay + 0.15 * low_persistence + 0.10 * irregularity
    margin = abs(water_score - dry_score)
    if features["valid_observation_count"] < int(config["minimum_track_observations"]):
        classification = "uncertain"
        reasons = ["too_few_track_observations"]
    elif water_score >= float(config["water_score_threshold"]) and water_score > dry_score and margin >= float(config["minimum_score_margin"]):
        classification = "water_ripple"
        reasons = ["long_persistent_track", "positive_radius_growth", "stable_or_ring_like_structure"]
    elif dry_score >= float(config["dry_score_threshold"]) and dry_score > water_score and margin >= float(config["minimum_score_margin"]):
        classification = "dry_splash"
        reasons = ["short_or_fast_decaying_track", "limited_persistent_expansion"]
    else:
        classification = "uncertain"
        reasons = ["water_dry_score_margin_too_small"]
    confidence = _clip(max(water_score, dry_score) * (0.6 + 0.4 * margin))
    return {
        **features,
        "dry_splash_score": float(dry_score),
        "water_ripple_score": float(water_score),
        "classification": classification,
        "confidence": float(confidence),
        "classification_reasons": reasons,
        "ground_truth_used": False,
    }


def classify_tracks(features: list[dict], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [classify_track(item, config) for item in features]
