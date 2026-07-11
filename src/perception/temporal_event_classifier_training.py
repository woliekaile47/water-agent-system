#!/usr/bin/env python3
"""Deterministic sampling, training and validation-only threshold selection."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.perception.temporal_event_classifier_model import fit_logistic_regression, predict_water_probability


def select_training_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    by_sequence: dict[str, list[dict]] = {}
    for sample in samples:
        if sample["label"] == "uncertain":
            continue
        by_sequence.setdefault(sample["sequence"]["path"], []).append(sample)
    selected = []
    per_sequence = {}
    for path, values in sorted(by_sequence.items()):
        water = [item for item in values if item["label"] == "water_ripple"]
        negative = sorted(
            [item for item in values if item["label"] in ("dry_splash", "background_noise")],
            key=lambda item: (item["label"] != "dry_splash", item["track_id"]),
        )
        target = min(
            int(config["maximum_negative_per_sequence"]),
            max(50, int(np.ceil(len(water) * float(config["negative_to_positive_ratio"])))),
        )
        chosen = water + negative[:target]
        selected.extend(chosen)
        per_sequence[path] = {
            "water": len(water), "negative_available": len(negative), "negative_selected": min(target, len(negative)),
        }
    distribution = Counter(item["label"] for item in selected)
    return selected, {"class_distribution": dict(distribution), "per_sequence": per_sequence}


def make_training_arrays(
    samples: list[dict[str, Any]], model_config: dict[str, Any], use_class_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([item["feature_vector"] for item in samples])
    y = np.asarray([item["label"] == "water_ripple" for item in samples], dtype=np.float64)
    sequence_counts = Counter(item["sequence"]["path"] for item in samples)
    rain_counts = Counter(item["sequence"]["rain_level"] for item in samples)
    sample_weight = np.asarray([
        1.0 / sequence_counts[item["sequence"]["path"]] / rain_counts[item["sequence"]["rain_level"]]
        for item in samples
    ], dtype=np.float64)
    sample_weight /= max(np.mean(sample_weight), 1e-12)
    if use_class_weights:
        sample_weight *= np.where(
            y > 0.5, float(model_config["class_weight_water"]), float(model_config["class_weight_dry"]),
        )
    return x, y, sample_weight


def train_classifier(
    samples: list[dict[str, Any]], model_config: dict[str, Any], use_class_weights: bool = True,
) -> dict[str, Any]:
    x, y, sample_weight = make_training_arrays(samples, model_config, use_class_weights)
    return fit_logistic_regression(x, y, sample_weight, model_config)


def classification_metrics(samples: list[dict[str, Any]], probabilities: np.ndarray, low: float, high: float) -> dict[str, Any]:
    truth_water = np.asarray([item["label"] == "water_ripple" for item in samples], dtype=bool)
    truth_dry = np.asarray([item["label"] in ("dry_splash", "background_noise") for item in samples], dtype=bool)
    predicted_water = probabilities >= high
    predicted_dry = probabilities <= low
    uncertain = ~(predicted_water | predicted_dry)

    def class_result(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        tp = int(np.count_nonzero(truth & prediction))
        fp = int(np.count_nonzero(~truth & prediction))
        fn = int(np.count_nonzero(truth & ~prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    water = class_result(truth_water, predicted_water)
    dry = class_result(truth_dry, predicted_dry)
    noise = np.asarray([item["label"] == "background_noise" for item in samples], dtype=bool)
    brier = float(np.mean((probabilities - truth_water.astype(np.float64)) ** 2)) if samples else 0.0
    return {
        "water_ripple": water, "dry_or_noise": dry, "macro_f1": (water["f1"] + dry["f1"]) / 2.0,
        "uncertain_rate": float(np.mean(uncertain)) if samples else 0.0,
        "background_noise_false_water": int(np.count_nonzero(noise & predicted_water)),
        "brier_score": brier,
        "confusion_matrix": {
            "water": {"water": int(np.count_nonzero(truth_water & predicted_water)), "dry": int(np.count_nonzero(truth_water & predicted_dry)), "uncertain": int(np.count_nonzero(truth_water & uncertain))},
            "dry_noise": {"water": int(np.count_nonzero(truth_dry & predicted_water)), "dry": int(np.count_nonzero(truth_dry & predicted_dry)), "uncertain": int(np.count_nonzero(truth_dry & uncertain))},
        },
    }


def select_validation_thresholds(
    samples: list[dict[str, Any]], model: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    matrix = np.stack([item["feature_vector"] for item in samples])
    probability, _ = predict_water_probability(matrix, model)
    candidates = []
    for low in map(float, config["low_candidates"]):
        for high in map(float, config["high_candidates"]):
            if low >= high:
                continue
            metrics = classification_metrics(samples, probability, low, high)
            objective = (
                metrics["macro_f1"]
                - float(config["uncertain_penalty"]) * metrics["uncertain_rate"]
                - float(config["false_water_noise_penalty"])
                * metrics["background_noise_false_water"] / max(1, len(samples))
            )
            candidates.append({"low_threshold": low, "high_threshold": high, "objective": objective, "metrics": metrics})
    best = max(candidates, key=lambda item: (item["objective"], item["high_threshold"], -item["low_threshold"]))
    return {"low_threshold": best["low_threshold"], "high_threshold": best["high_threshold"]}, {
        "selection_split": "validation_only", "selected": best, "candidate_count": len(candidates),
        "probability_recall_curve": [
            {
                "threshold": threshold,
                "precision": classification_metrics(samples, probability, threshold, threshold)["water_ripple"]["precision"],
                "recall": classification_metrics(samples, probability, threshold, threshold)["water_ripple"]["recall"],
            }
            for threshold in np.linspace(0.05, 0.95, 19)
        ],
    }
