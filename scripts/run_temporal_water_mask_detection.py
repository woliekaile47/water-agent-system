#!/usr/bin/env python3
"""Run the explainable Phase 2C-2A temporal RGB water-mask baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_temporal_water_mask import (
    evaluate_events,
    evaluate_water_mask,
    load_temporal_evaluation_ground_truth,
)
from src.perception.temporal_water_pipeline import run_temporal_prediction
from src.perception.temporal_water_quality_gate import evaluate_temporal_quality_gate


SOURCE = "explainable_temporal_rain_impact_baseline"


def load_yaml(path: Path, key: str) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return document[key]


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def json_safe_candidates(candidates_by_frame: list[list[dict]]) -> dict[str, Any]:
    return {
        "ground_truth_used": False,
        "frames": [
            {"frame_index": index, "candidates": candidates}
            for index, candidates in enumerate(candidates_by_frame)
        ],
    }


def _save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _save_temporal_preview(path: Path, residual: np.ndarray) -> None:
    value = np.asarray(residual, dtype=np.float32)
    scale = float(np.percentile(value, 99.5)) if value.size else 1.0
    image = np.clip(value / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _save_comparison(path: Path, prediction: np.ndarray, unknown: np.ndarray, truth: np.ndarray) -> None:
    height, width = truth.shape
    gt_panel = np.zeros((height, width, 3), dtype=np.uint8)
    gt_panel[..., 2] = np.where(truth, 255, 30)
    pred_panel = np.zeros_like(gt_panel)
    pred_panel[..., 1] = np.where(prediction, 255, 30)
    pred_panel[..., 0] = np.where(unknown, 120, 0)
    overlay = np.zeros_like(gt_panel)
    overlay[..., 1] = np.where(prediction & truth, 255, 0)
    overlay[..., 0] = np.where(prediction & ~truth, 255, 0)
    overlay[..., 2] = np.where(~prediction & truth, 255, 0)
    separator = np.full((height, 3, 3), 100, dtype=np.uint8)
    Image.fromarray(np.concatenate((gt_panel, separator, pred_panel, separator, overlay), axis=1), mode="RGB").save(path)


def _order_sensitivity(full: dict, shuffled: dict) -> float:
    left = full["evidence"]["predicted_water_probability"]
    right = shuffled["evidence"]["predicted_water_probability"]
    difference = float(np.mean(np.abs(left - right)))
    reference = max(float(np.mean(left) + np.mean(right)), 0.01)
    return float(np.clip(difference / reference, 0.0, 1.0))


def run_sequence(
    sequence_dir: Path,
    output_dir: Path,
    detector_config: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    frames_dir = sequence_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    full = run_temporal_prediction(str(frames_dir), detector_config, "full")
    single = run_temporal_prediction(str(frames_dir), detector_config, "single_frame")
    shuffled = run_temporal_prediction(str(frames_dir), detector_config, "shuffled")
    color_normalized = run_temporal_prediction(str(frames_dir), detector_config, "color_normalized")
    order_sensitivity = _order_sensitivity(full, shuffled)

    np.save(output_dir / "predicted_water_probability.npy", full["evidence"]["predicted_water_probability"])
    np.save(output_dir / "evidence_count_map.npy", full["evidence"]["evidence_count_map"])
    _save_binary(output_dir / "predicted_water_mask.png", full["evidence"]["predicted_water_mask"])
    _save_binary(output_dir / "predicted_unknown_mask.png", full["evidence"]["predicted_unknown_mask"])
    _save_temporal_preview(output_dir / "temporal_activity_preview.png", full["preprocessed"]["temporal_residual_preview"])
    write_json(output_dir / "event_candidates.json", json_safe_candidates(full["candidates_by_frame"]))
    write_json(output_dir / "event_tracks.json", {"ground_truth_used": False, "tracks": full["tracks"]})
    write_json(output_dir / "event_classifications.json", {"ground_truth_used": False, "classifications": full["classifications"]})
    temporal_diagnostics = {
        "loader": full["loader"],
        "preprocessing": full["preprocessing_diagnostics"],
        "candidates": full["candidate_diagnostics"],
        "evidence": full["evidence_diagnostics"],
        "water_mask_time_stability": full["water_mask_time_stability"],
        "feature_score_separation": full["feature_score_separation"],
        "order_sensitivity": order_sensitivity,
        "ground_truth_used": False,
    }
    write_json(output_dir / "temporal_diagnostics.json", temporal_diagnostics)
    gate = evaluate_temporal_quality_gate(
        full["loader"],
        full["preprocessing_diagnostics"],
        full["candidate_diagnostics"],
        full["tracks"],
        full["classifications"],
        full["evidence"],
        full["evidence_diagnostics"],
        order_sensitivity,
        full["water_mask_time_stability"],
        full["feature_score_separation"],
        float(detector_config["fps"]),
        gate_config,
    )
    write_json(output_dir / "quality_gate.json", gate)
    prediction_manifest = {
        "data_role": "prediction",
        "source": SOURCE,
        "algorithm_version": detector_config["algorithm_version"],
        "detector_input": str(frames_dir.resolve()),
        "detector_input_role": "continuous_rgb_frames_only",
        "ground_truth_or_metadata_read_during_prediction": False,
        "observation_source": "temporal_rain_impact_visual_evidence",
        "result_semantics": "camera_water_mask_from_sparse_temporal_evidence",
        "coverage_status": gate["coverage_status"],
        "unknown_region_semantics": "no_temporal_evidence_not_confirmed_dry",
        "synthetic_domain": True,
        "real_world_validated": False,
        "quality_gate_status": gate["status"],
        "eligible_for_downstream": False,
    }
    write_json(output_dir / "prediction_manifest.json", prediction_manifest)

    # Independent evaluation begins only after prediction and GT-free gate.
    ground_truth = load_temporal_evaluation_ground_truth(sequence_dir)
    ablation_predictions = {
        "full_temporal": full,
        "single_frame": single,
        "shuffled_frames": shuffled,
        "color_mean_normalized": color_normalized,
    }
    ablation_metrics = {}
    for name, prediction in ablation_predictions.items():
        ablation_metrics[name] = evaluate_water_mask(
            prediction["evidence"]["predicted_water_mask"],
            prediction["evidence"]["predicted_unknown_mask"],
            ground_truth["water_mask"],
        )
        ablation_metrics[name]["water_track_count"] = prediction["evidence_diagnostics"]["water_track_count"]
    mask_metrics = ablation_metrics["full_temporal"]
    event_metrics = evaluate_events(full["classifications"], ground_truth["events"])
    evaluation = {
        "data_role": "evaluation",
        "ground_truth_used_for_evaluation_only": True,
        "water_mask_metrics": mask_metrics,
        "ablation_metrics": ablation_metrics,
        "quality_gate_status": gate["status"],
        "eligible_for_downstream": False,
        "ground_truth_paths": ground_truth["paths"],
    }
    write_json(output_dir / "evaluation_metrics.json", evaluation)
    write_json(output_dir / "event_evaluation.json", {"data_role": "evaluation", **event_metrics})
    write_json(output_dir / "ablation_metrics.json", {"data_role": "evaluation", "metrics": ablation_metrics})
    _save_comparison(
        output_dir / "water_mask_comparison.png",
        full["evidence"]["predicted_water_mask"],
        full["evidence"]["predicted_unknown_mask"],
        ground_truth["water_mask"],
    )
    return {
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "gate": gate,
        "evaluation": evaluation,
        "event_evaluation": event_metrics,
        "prediction": full,
    }


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "sequence", "gate", "gate_reasons", "water_tracks", "unknown_fraction",
        "whole_image_iou", "known_region_iou", "precision", "recall", "event_water_f1",
        "predicted_water_area_pixels", "false_water_events_on_dry_sequence",
        "single_frame_iou", "shuffled_iou", "color_normalized_iou", "eligible_for_downstream",
    ]
    rows = []
    for result in results:
        evaluation = result["evaluation"]
        mask = evaluation["water_mask_metrics"]
        ablations = evaluation["ablation_metrics"]
        water_events = result["event_evaluation"]["per_class"]["water_ripple"]
        rows.append({
            "sequence": result["sequence_dir"],
            "gate": result["gate"]["status"],
            "gate_reasons": ";".join(result["gate"]["reasons"]),
            "water_tracks": result["prediction"]["evidence_diagnostics"]["water_track_count"],
            "unknown_fraction": mask["unknown_fraction"],
            "whole_image_iou": mask["whole_image_iou"],
            "known_region_iou": mask["evaluated_known_region_iou"],
            "precision": mask["pixel_precision"],
            "recall": mask["pixel_recall"],
            "event_water_f1": water_events["f1"],
            "predicted_water_area_pixels": mask["predicted_water_area_pixels"],
            "false_water_events_on_dry_sequence": result["event_evaluation"]["false_water_events_on_dry_sequence"],
            "single_frame_iou": ablations["single_frame"]["whole_image_iou"],
            "shuffled_iou": ablations["shuffled_frames"]["whole_image_iou"],
            "color_normalized_iou": ablations["color_mean_normalized"]["whole_image_iou"],
            "eligible_for_downstream": False,
        })
    with (output_root / "dataset_evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 2C-2A temporal water baseline summary", "",
        "| sequence | gate | IoU | precision | recall | unknown | water event F1 | single IoU | shuffled IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {Path(row['sequence']).as_posix()} | {row['gate']} | {row['whole_image_iou']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['unknown_fraction']:.4f} | "
            f"{row['event_water_f1']:.4f} | {row['single_frame_iou']:.4f} | {row['shuffled_iou']:.4f} |"
        )
    (output_root / "dataset_evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_results(data_root: Path, output_root: Path) -> list[dict[str, Any]]:
    """Load completed outputs for summary generation without rerunning prediction."""
    results = []
    for evaluation_path in sorted(output_root.glob("*/*/seed_*/evaluation_metrics.json")):
        output_dir = evaluation_path.parent
        relative = output_dir.relative_to(output_root)
        with evaluation_path.open("r", encoding="utf-8") as stream:
            evaluation = json.load(stream)
        with (output_dir / "quality_gate.json").open("r", encoding="utf-8") as stream:
            gate = json.load(stream)
        with (output_dir / "event_evaluation.json").open("r", encoding="utf-8") as stream:
            event_evaluation = json.load(stream)
        results.append({
            "sequence_dir": str(data_root / relative),
            "output_dir": str(output_dir),
            "gate": gate,
            "evaluation": evaluation,
            "event_evaluation": event_evaluation,
            "prediction": {"evidence_diagnostics": {
                "water_track_count": gate["metrics"]["water_track_count"],
            }},
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sequence-dir")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    detector = load_yaml(root / "configs/temporal_water_mask_detector.yaml", "temporal_water_mask_detector")
    gate = load_yaml(root / "configs/temporal_water_quality_gate.yaml", "temporal_water_quality_gate")
    data_root = root / "data/simulation_dynamic"
    output_root = root / "outputs/temporal_water_detection"
    if args.summarize_existing:
        results = load_existing_results(data_root, output_root)
        if not results:
            raise RuntimeError(f"No completed temporal evaluations under {output_root}")
        write_summary(output_root, results)
        print(f"[phase2c2a] summarized {len(results)} completed sequences")
        return
    if args.all:
        sequences = sorted(path.parent for path in data_root.glob("*/*/seed_*/frames"))
    else:
        sequences = [Path(args.sequence_dir).expanduser().resolve()]
    results = []
    for sequence in sequences:
        relative = sequence.relative_to(data_root)
        output = output_root / relative
        result = run_sequence(sequence, output, detector, gate)
        results.append(result)
        metric = result["evaluation"]["water_mask_metrics"]
        print(
            f"[phase2c2a] {relative}: gate={result['gate']['status']} "
            f"iou={metric['whole_image_iou']:.4f} precision={metric['pixel_precision']:.4f} "
            f"recall={metric['pixel_recall']:.4f} unknown={metric['unknown_fraction']:.4f}"
        )
    if args.all:
        write_summary(output_root, results)


if __name__ == "__main__":
    main()
