#!/usr/bin/env python3
"""Evaluate frozen rule/model/hybrid temporal classifiers on held-out sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_temporal_water_mask import (
    evaluate_labeled_track_classifications, evaluate_water_mask, load_temporal_evaluation_ground_truth,
)
from src.perception.temporal_event_dataset import discover_sequence_splits
from src.perception.temporal_event_label_matching import match_tracks_to_events
from src.perception.temporal_water_pipeline import run_temporal_model_prediction
from src.perception.temporal_water_quality_gate import evaluate_model_classifier_quality_gate


def load_yaml(path: Path, key: str) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[key]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def save_comparison(path: Path, rule: np.ndarray, model: np.ndarray, hybrid: np.ndarray, truth: np.ndarray) -> None:
    panels = []
    for mask, color in ((truth, 2), (rule, 0), (model, 1), (hybrid, 1)):
        panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
        panel[..., color] = np.where(mask, 255, 25)
        panels.append(panel)
    Image.fromarray(np.concatenate(panels, axis=1), mode="RGB").save(path)


def classifications_for_rule(prediction: dict) -> list[dict]:
    return prediction["baseline"]["classifications"]


def probability_diagnostics(classifications: list[dict], track_labels: list[dict]) -> dict[str, Any]:
    labels = {item["track_id"]: item["label"] for item in track_labels}
    usable = [item for item in classifications if labels.get(item["track_id"]) != "uncertain"]
    probability = np.asarray([item["model_water_probability"] for item in usable], dtype=np.float64)
    truth = np.asarray([labels.get(item["track_id"]) == "water_ripple" for item in usable], dtype=bool)
    curve = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predicted = probability >= threshold
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        curve.append({
            "threshold": float(threshold),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })
    calibration = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + .1
        selected = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        calibration.append({
            "lower": float(lower), "upper": float(upper), "count": int(np.count_nonzero(selected)),
            "mean_probability": float(np.mean(probability[selected])) if np.any(selected) else None,
            "observed_water_fraction": float(np.mean(truth[selected])) if np.any(selected) else None,
        })
    return {"water_event_pr_curve": curve, "probability_calibration": calibration}


def run_sequence(sequence: dict, root: Path, detector: dict, training: dict, thresholds: dict) -> dict:
    sequence_dir = Path(sequence["path"])
    relative = sequence_dir.relative_to(root / "data/simulation_dynamic")
    output = root / "outputs/temporal_water_detection_model" / relative
    output.mkdir(parents=True, exist_ok=True)
    modes = {}
    for mode in ("full", "single_frame", "shuffled", "color_normalized"):
        modes[mode] = run_temporal_model_prediction(
            str(sequence_dir / "frames"), str(root / "outputs/temporal_event_classifier/model"),
            detector, training, thresholds, mode,
        )
    prediction = modes["full"]
    gate = evaluate_model_classifier_quality_gate(prediction, training["quality_gate"])
    write_json(output / "model_quality_gate.json", gate)
    write_json(output / "model_event_classifications.json", {
        "ground_truth_used": False, "classifications": prediction["model_classifications"],
    })
    write_json(output / "hybrid_event_classifications.json", {
        "ground_truth_used": False, "classifications": prediction["hybrid_classifications"],
    })
    rule_evidence = prediction["baseline"]["evidence"]
    save_mask(output / "baseline_rule_water_mask.png", rule_evidence["predicted_water_mask"])
    save_mask(output / "model_water_mask.png", prediction["model_evidence"]["predicted_water_mask"])
    save_mask(output / "hybrid_water_mask.png", prediction["hybrid_evidence"]["predicted_water_mask"])
    np.save(output / "model_water_probability.npy", prediction["model_evidence"]["predicted_water_probability"])
    write_json(output / "prediction_manifest.json", {
        "detector_input": str(sequence_dir / "frames"),
        "inference_inputs": ["frames/*.png", "detector_config", "frozen_classifier_parameters"],
        "ground_truth_or_metadata_read_during_prediction": False,
        "quality_gate_completed_before_evaluation": True,
        "eligible_for_downstream": False,
    })

    # Independent evaluation starts only after prediction artifacts and gate exist.
    ground_truth = load_temporal_evaluation_ground_truth(sequence_dir)
    track_labels, matching = match_tracks_to_events(
        prediction["baseline"]["tracks"], ground_truth["events"], training["matching"],
    )
    mode_metrics = {}
    for name, value in modes.items():
        mode_metrics[name] = evaluate_water_mask(
            value["model_evidence"]["predicted_water_mask"],
            value["model_evidence"]["predicted_unknown_mask"], ground_truth["water_mask"],
        )
    mask_metrics = {
        "rule_baseline": evaluate_water_mask(rule_evidence["predicted_water_mask"], rule_evidence["predicted_unknown_mask"], ground_truth["water_mask"]),
        "learned_model": mode_metrics["full"],
        "hybrid": evaluate_water_mask(prediction["hybrid_evidence"]["predicted_water_mask"], prediction["hybrid_evidence"]["predicted_unknown_mask"], ground_truth["water_mask"]),
    }
    event_metrics = {
        "rule_baseline": evaluate_labeled_track_classifications(classifications_for_rule(prediction), track_labels),
        "learned_model": evaluate_labeled_track_classifications(prediction["model_classifications"], track_labels),
        "hybrid": evaluate_labeled_track_classifications(prediction["hybrid_classifications"], track_labels),
    }
    evaluation = {
        "data_role": "evaluation", "ground_truth_used_for_evaluation_only": True,
        "mask_metrics": mask_metrics, "event_metrics": event_metrics,
        "temporal_ablations": mode_metrics, "matching_diagnostics": matching,
        "model_probability_diagnostics": probability_diagnostics(prediction["model_classifications"], track_labels),
        "quality_gate": gate, "eligible_for_downstream": False,
    }
    write_json(output / "evaluation_metrics.json", evaluation)
    save_comparison(
        output / "rule_model_hybrid_comparison.png", rule_evidence["predicted_water_mask"],
        prediction["model_evidence"]["predicted_water_mask"], prediction["hybrid_evidence"]["predicted_water_mask"],
        ground_truth["water_mask"],
    )
    return {"sequence": sequence, **evaluation}


def aggregate_results(results: list[dict]) -> dict[str, Any]:
    modes = ("rule_baseline", "learned_model", "hybrid")
    aggregate: dict[str, Any] = {"sequence_count": len(results), "mask_metrics": {}, "event_metrics": {}}
    for mode in modes:
        mask_rows = [result["mask_metrics"][mode] for result in results]
        water_rows = [row for result, row in zip(results, mask_rows) if "dry_baseline" not in result["sequence"]["case_id"]]
        aggregate["mask_metrics"][mode] = {
            "mean_water_sequence_iou": float(np.mean([row["whole_image_iou"] for row in water_rows])),
            "mean_water_sequence_precision": float(np.mean([row["pixel_precision"] for row in water_rows])),
            "mean_water_sequence_recall": float(np.mean([row["pixel_recall"] for row in water_rows])),
            "mean_water_sequence_f1": float(np.mean([row["pixel_f1"] for row in water_rows])),
            "mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in water_rows])),
            "mean_unknown_fraction": float(np.mean([row["unknown_fraction"] for row in mask_rows])),
            "dry_false_positive_pixels": int(sum(
                row["predicted_water_area_pixels"] for result, row in zip(results, mask_rows)
                if "dry_baseline" in result["sequence"]["case_id"]
            )),
        }
        per_sequence_events = [result["event_metrics"][mode] for result in results]
        classes = {}
        for class_name in ("dry_splash", "water_ripple"):
            tp = sum(item["per_class"][class_name]["tp"] for item in per_sequence_events)
            fp = sum(item["per_class"][class_name]["fp"] for item in per_sequence_events)
            fn = sum(item["per_class"][class_name]["fn"] for item in per_sequence_events)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            classes[class_name] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        aggregate["event_metrics"][mode] = {
            "per_class": classes,
            "macro_f1": float(np.mean([classes[name]["f1"] for name in classes])),
            "background_noise_false_water": int(sum(item["background_noise_false_water"] for item in per_sequence_events)),
            "mean_uncertain_rate": float(np.mean([item["uncertain_rate"] for item in per_sequence_events])),
        }
    aggregate["quality_gate_distribution"] = {
        status: sum(result["quality_gate"]["status"] == status for result in results)
        for status in ("pass", "partial", "reject")
    }
    aggregate["by_rain_level"] = {}
    aggregate["by_case"] = {}
    for field, key in (("rain_level", "by_rain_level"), ("case_id", "by_case")):
        for value in sorted({result["sequence"][field] for result in results}):
            subset = [result for result in results if result["sequence"][field] == value]
            aggregate[key][value] = {
                mode: float(np.mean([item["mask_metrics"][mode]["whole_image_iou"] for item in subset]))
                for mode in modes
            }
    aggregate["temporal_ablations"] = {
        mode: float(np.mean([result["temporal_ablations"][mode]["whole_image_iou"] for result in results]))
        for mode in ("full", "single_frame", "shuffled", "color_normalized")
    }
    aggregate["probability_diagnostics_by_sequence"] = {
        f"{item['sequence']['case_id']}/{item['sequence']['rain_level']}/seed_{item['sequence']['seed']}": item["model_probability_diagnostics"]
        for item in results
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--all-test", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    detector = load_yaml(root / "configs/temporal_water_mask_detector.yaml", "temporal_water_mask_detector")
    training = load_yaml(root / "configs/temporal_event_classifier_training.yaml", "temporal_event_classifier_training")
    _, split_manifest = discover_sequence_splits(root / "data/simulation_dynamic", training["splits"])
    thresholds_document = load_json(root / "outputs/temporal_event_classifier/model/threshold_selection.json")
    thresholds = {key: thresholds_document[key] for key in ("low_threshold", "high_threshold")}
    results = []
    for sequence in split_manifest["test_sequences"]:
        result = run_sequence(sequence, root, detector, training, thresholds)
        results.append(result)
        print(
            f"[phase2c2b] {sequence['case_id']}/{sequence['rain_level']}/seed_{sequence['seed']} "
            f"rule_iou={result['mask_metrics']['rule_baseline']['whole_image_iou']:.4f} "
            f"model_iou={result['mask_metrics']['learned_model']['whole_image_iou']:.4f} "
            f"hybrid_iou={result['mask_metrics']['hybrid']['whole_image_iou']:.4f} "
            f"gate={result['quality_gate']['status']}"
        )
    write_json(root / "outputs/temporal_water_detection_model/test_dataset_summary.json", {"results": results})
    write_json(root / "outputs/temporal_event_classifier/model/test_metrics.json", aggregate_results(results))


if __name__ == "__main__":
    main()
