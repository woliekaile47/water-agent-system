#!/usr/bin/env python3
"""Train the deterministic Phase 2C-2B track-feature classifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.rain_impact_classifier import classify_track
from src.perception.temporal_event_classifier_model import (
    fit_logistic_regression, predict_water_probability, save_model,
)
from src.perception.temporal_event_classifier_training import (
    classification_metrics, make_training_arrays, select_training_samples,
    select_validation_thresholds, train_classifier,
)
from src.perception.temporal_event_dataset import (
    FEATURE_NAMES, build_labeled_sequence_samples, discover_sequence_splits,
)


def load_config(path: Path, key: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)[key]


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def assemble_split_samples(root: Path, splits: dict, detector: dict, training: dict) -> tuple[dict, dict]:
    samples_by_split, matching = {}, {}
    for split_name, sequences in splits.items():
        samples_by_split[split_name] = []
        matching[split_name] = {}
        for sequence in sequences:
            samples, diagnostics = build_labeled_sequence_samples(
                sequence, root / "data/simulation_dynamic", root / "outputs/temporal_water_detection",
                detector, training["matching"],
            )
            samples_by_split[split_name].extend(samples)
            matching[split_name][sequence["path"]] = diagnostics
    return samples_by_split, matching


def _probabilities(samples: list[dict], model: dict) -> np.ndarray:
    return predict_water_probability(np.stack([item["feature_vector"] for item in samples]), model)[0]


def _group_metrics(samples: list[dict], probabilities: np.ndarray, thresholds: dict) -> dict[str, Any]:
    result = {}
    for field in ("rain_level", "case_id", "seed"):
        values = sorted({item["sequence"][field] for item in samples}, key=str)
        result[field] = {}
        for value in values:
            indices = [index for index, item in enumerate(samples) if item["sequence"][field] == value]
            subset = [samples[index] for index in indices]
            result[field][str(value)] = classification_metrics(
                subset, probabilities[indices], thresholds["low_threshold"], thresholds["high_threshold"],
            )
    return result


def _calibration(samples: list[dict], probabilities: np.ndarray) -> list[dict[str, Any]]:
    truth = np.asarray([item["label"] == "water_ripple" for item in samples], dtype=np.float64)
    bins = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        bins.append({
            "lower": float(lower), "upper": float(upper), "count": int(np.count_nonzero(selected)),
            "mean_probability": float(np.mean(probabilities[selected])) if np.any(selected) else None,
            "observed_water_fraction": float(np.mean(truth[selected])) if np.any(selected) else None,
        })
    return bins


def _with_vectors(samples: list[dict], indices: list[int]) -> list[dict]:
    copied = []
    for sample in samples:
        item = dict(sample)
        item["feature_vector"] = sample["feature_vector"][indices]
        copied.append(item)
    return copied


def run_ablations(train_samples: list[dict], validation_samples: list[dict], config: dict) -> dict:
    groups = {
        "all_features": [], "without_ringness": ["ringness", "radial_energy_profile", "ring_energy_duration_fraction"],
        "without_duration": ["duration_frames", "peak_frame_offset", "observation_fill_ratio"],
        "without_expansion": ["area_growth_slope", "radius_growth_slope", "early_radius_growth_slope", "radius_expansion_consistency", "smoothed_area_growth_slope", "smoothed_radius_growth_slope"],
        "without_post_peak_persistence": ["post_peak_persistence", "post_peak_area_integral"],
    }
    outputs = {}
    for name, removed in groups.items():
        indices = [index for index, feature in enumerate(FEATURE_NAMES) if feature not in removed]
        train_subset = _with_vectors(train_samples, indices)
        validation_subset = _with_vectors(validation_samples, indices)
        model = train_classifier(train_subset, config["model"], use_class_weights=True)
        thresholds, selection = select_validation_thresholds(validation_subset, model, config["threshold_selection"])
        metrics = classification_metrics(validation_subset, _probabilities(validation_subset, model), thresholds["low_threshold"], thresholds["high_threshold"])
        outputs[name] = {"removed_features": removed, "thresholds": thresholds, "validation_objective": selection["selected"]["objective"], "validation_metrics": metrics}
    x, y, sample_weight = make_training_arrays(train_samples, config["model"], use_class_weights=False)
    no_weight_model = fit_logistic_regression(x, y, sample_weight, config["model"])
    thresholds, selection = select_validation_thresholds(validation_samples, no_weight_model, config["threshold_selection"])
    outputs["without_class_weights"] = {
        "removed_features": [], "thresholds": thresholds,
        "validation_objective": selection["selected"]["objective"],
        "validation_metrics": classification_metrics(validation_samples, _probabilities(validation_samples, no_weight_model), thresholds["low_threshold"], thresholds["high_threshold"]),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    detector = load_config(root / "configs/temporal_water_mask_detector.yaml", "temporal_water_mask_detector")
    training = load_config(root / "configs/temporal_event_classifier_training.yaml", "temporal_event_classifier_training")
    splits, split_manifest = discover_sequence_splits(root / "data/simulation_dynamic", training["splits"])
    if any(not splits[name] for name in ("train", "validation", "test")):
        raise RuntimeError("train, validation and test sequence splits must all be non-empty")
    fit_splits = {name: splits[name] for name in ("train", "validation")}
    samples_by_split, matching = assemble_split_samples(root, fit_splits, detector, training)
    train_selected, sampling = select_training_samples(samples_by_split["train"], training["sampling"])
    validation = [item for item in samples_by_split["validation"] if item["label"] != "uncertain"]
    model = train_classifier(train_selected, training["model"], use_class_weights=True)
    model.update({
        "feature_names": FEATURE_NAMES, "model_version": training["model_version"],
        "feature_schema_version": training["feature_schema_version"],
    })
    thresholds, threshold_selection = select_validation_thresholds(validation, model, training["threshold_selection"])
    output = root / "outputs/temporal_event_classifier/model"
    save_model(output, model, model)
    validation_probability = _probabilities(validation, model)
    validation_metrics = classification_metrics(validation, validation_probability, thresholds["low_threshold"], thresholds["high_threshold"])
    ablations = run_ablations(train_selected, validation, training)
    write_json(output / "split_manifest.json", split_manifest)
    write_json(output / "training_manifest.json", {
        "model_version": training["model_version"], "feature_schema_version": training["feature_schema_version"],
        "training_sample_count": len(train_selected), "class_distribution": sampling["class_distribution"],
        "sequence_balancing": "inverse sequence and rain-level frequency",
        "negative_sampling": training["sampling"], "class_weights": {
            "dry": training["model"]["class_weight_dry"], "water": training["model"]["class_weight_water"],
        },
        "ground_truth_usage": "event annotations only for offline track labeling",
        "test_split_used_for_threshold_selection": False,
        "training_loss_trace": model["training_loss_trace"],
    })
    write_json(output / "matching_diagnostics.json", matching)
    write_json(output / "threshold_selection.json", thresholds | threshold_selection)
    write_json(output / "validation_metrics.json", validation_metrics)
    write_json(output / "ablation_metrics.json", ablations)
    print(f"[phase2c2b] train={len(train_selected)} validation={len(validation)} test=not_loaded thresholds={thresholds}")
    print(f"[phase2c2b] validation water_f1={validation_metrics['water_ripple']['f1']:.4f}")


if __name__ == "__main__":
    main()
