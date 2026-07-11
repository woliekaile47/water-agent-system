#!/usr/bin/env python3
"""GT-free inference for the trained temporal event classifier."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.perception.rain_impact_classifier import classify_track
from src.perception.temporal_event_classifier_model import predict_water_probability


def infer_track_classifications(
    feature_records: list[dict[str, Any]], model: dict[str, Any], thresholds: dict[str, float],
    rule_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify RGB-derived track features; deliberately has no GT/metadata argument."""
    names = model["feature_names"]
    matrix = np.asarray([[record.get(name, 0.0) for name in names] for record in feature_records], dtype=np.float64)
    if not feature_records:
        return []
    probabilities, normalized = predict_water_probability(matrix, model)
    results = []
    for index, (record, water_probability) in enumerate(zip(feature_records, probabilities)):
        baseline = classify_track(record, rule_config)
        if water_probability >= float(thresholds["high_threshold"]):
            classification = "water_ripple"
            reason = "model_probability_above_validation_high_threshold"
        elif water_probability <= float(thresholds["low_threshold"]):
            classification = "dry_splash"
            reason = "model_probability_below_validation_low_threshold"
        else:
            classification = "uncertain"
            reason = "model_probability_inside_validation_uncertain_interval"
        confidence = float(max(water_probability, 1.0 - water_probability))
        results.append({
            **record,
            "baseline_rule_classification": baseline["classification"],
            "baseline_water_score": baseline["water_ripple_score"],
            "model_dry_probability": float(1.0 - water_probability),
            "model_water_probability": float(water_probability),
            "classification": classification,
            "confidence": confidence,
            "classification_reasons": [reason],
            "model_version": model["model_version"],
            "feature_schema_version": model["feature_schema_version"],
            "maximum_absolute_train_z": float(np.max(np.abs(normalized[index]))),
            "ground_truth_used": False,
        })
    return results


def build_hybrid_classifications(
    model_classifications: list[dict[str, Any]], thresholds: dict[str, float], model_weight: float,
) -> list[dict[str, Any]]:
    results = []
    for item in model_classifications:
        baseline_probability = float(np.clip(item["baseline_water_score"], 0.0, 1.0))
        probability = model_weight * item["model_water_probability"] + (1.0 - model_weight) * baseline_probability
        if probability >= thresholds["high_threshold"]:
            classification = "water_ripple"
        elif probability <= thresholds["low_threshold"]:
            classification = "dry_splash"
        else:
            classification = "uncertain"
        results.append({
            **item, "classification": classification,
            "confidence": float(max(probability, 1.0 - probability)),
            "hybrid_water_probability": float(probability),
            "classification_reasons": ["validation_frozen_rule_model_probability_blend"],
        })
    return results
