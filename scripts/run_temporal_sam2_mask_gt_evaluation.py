#!/usr/bin/env python3
"""Evaluate 12 frozen automatic-prompt SAM 2 masks against Camera-only GT."""

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

from src.evaluation.evaluate_temporal_sam2_mask_gt import (  # noqa: E402
    evaluate_camera_mask,
    load_camera_mask_ground_truth,
    verify_frozen_sample_inputs,
)
from src.fusion.sam2_shoreline_geometry_adapter import outer_boundary_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prompt-matrix-summary", type=Path, required=True)
    parser.add_argument("--sam2-matrix-summary", type=Path, required=True)
    parser.add_argument("--sam2-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        key: {
            "sample_count": len(items),
            "iou": metric_summary([float(item["iou"]) for item in items]),
            "precision": metric_summary([float(item["precision"]) for item in items]),
            "recall": metric_summary([float(item["recall"]) for item in items]),
            "f1": metric_summary([float(item["f1"]) for item in items]),
            "outer_boundary_p95_px": metric_summary(
                [float(item["outer_boundary_p95_px"]) for item in items if item["outer_boundary_p95_px"] is not None]
            ),
            "offline_research_criteria_met_count": sum(bool(item["offline_research_criteria_met"]) for item in items),
        }
        for key, items in sorted(groups.items())
    }


def save_comparison(predicted: np.ndarray, truth: np.ndarray, path: Path) -> None:
    image = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    image[predicted & truth] = (40, 190, 80)
    image[predicted & ~truth] = (225, 70, 55)
    image[~predicted & truth] = (50, 120, 235)
    Image.fromarray(image, mode="RGB").save(path)


def save_outer_boundaries(predicted: np.ndarray, truth: np.ndarray, path: Path) -> None:
    image = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    pred_outer = outer_boundary_mask(predicted)
    truth_outer = outer_boundary_mask(truth)
    image[pred_outer] = (255, 210, 20)
    image[truth_outer] = (20, 210, 255)
    image[pred_outer & truth_outer] = (255, 255, 255)
    Image.fromarray(image, mode="RGB").save(path)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    prompt_summary = read_json(args.prompt_matrix_summary.expanduser().resolve())
    sam2_summary = read_json(args.sam2_matrix_summary.expanduser().resolve())
    sam2_root = args.sam2_output_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_root}")
    prompt_by_id = {item["sample_id"]: item for item in prompt_summary["samples"]}
    sam2_by_id = {item["sample_id"]: item for item in sam2_summary["samples"]}
    if set(prompt_by_id) != set(sam2_by_id) or len(sam2_by_id) != 12:
        raise ValueError("Expected the same 12 frozen samples in prompt and SAM 2 summaries")

    # Atomic protocol boundary: every frozen hash is checked before the first GT read.
    frozen_checks = {
        sample_id: verify_frozen_sample_inputs(sam2_by_id[sample_id], prompt_by_id[sample_id], sam2_root)
        for sample_id in sorted(sam2_by_id)
    }

    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(sam2_by_id):
        sample = sam2_by_id[sample_id]
        prompt = prompt_by_id[sample_id]
        gt = load_camera_mask_ground_truth(
            root, sample["case_id"], sample["rain_level"], int(sample["seed"]), int(sample["frame_index"])
        )
        predicted = np.load(frozen_checks[sample_id]["mask_path"], allow_pickle=False).astype(bool)
        truth = np.asarray(gt["camera_mask"], dtype=bool)
        if predicted.shape != truth.shape:
            raise ValueError(f"Mask shape mismatch for {sample_id}")
        evaluation = evaluate_camera_mask(predicted, truth)
        sample_dir = output_root / sample_id
        sample_dir.mkdir()
        write_json(sample_dir / "camera_mask_evaluation.json", {
            "sample": sample,
            "prompt_status": prompt["prompt_status"],
            "prompt_reasons": prompt["prompt_reasons"],
            "frozen_input_verification": frozen_checks[sample_id],
            "ground_truth_validation": gt["validation"],
            "evaluation": evaluation,
            "prediction_recomputed": False,
            "sam2_rerun": False,
            "ground_truth_used_for_prediction": False,
            "eligible_for_downstream": False,
        })
        save_comparison(predicted, truth, sample_dir / "camera_mask_vs_gt.png")
        save_outer_boundaries(predicted, truth, sample_dir / "camera_outer_boundary_vs_gt.png")
        camera = evaluation["camera_mask_metrics"]
        boundary = evaluation["camera_boundary_metrics"]
        row = {
            "sample_id": sample_id,
            "case_id": sample["case_id"],
            "nominal_depth_cm_for_evaluation_grouping": int(sample["case_id"].split("_")[2].removesuffix("cm")),
            "rain_level": sample["rain_level"],
            "seed": int(sample["seed"]),
            "frame_index": int(sample["frame_index"]),
            "prompt_status": prompt["prompt_status"],
            "predicted_pixels": camera["predicted_pixels"],
            "gt_pixels": camera["gt_pixels"],
            "iou": camera["iou"],
            "precision": camera["precision"],
            "recall": camera["recall"],
            "f1": camera["f1"],
            "false_positive_pixels": camera["false_positive_pixels"],
            "false_negative_pixels": camera["false_negative_pixels"],
            "outer_boundary_p50_px": boundary["symmetric_outer_boundary"]["p50_px"],
            "outer_boundary_p95_px": boundary["symmetric_outer_boundary"]["p95_px"],
            "predicted_to_gt_outer_p95_px": boundary["predicted_shoreline_to_gt_boundary"]["p95_px"],
            "gt_to_predicted_outer_p95_px": boundary["gt_boundary_to_predicted_shoreline"]["p95_px"],
            "connected_component_count": camera["connected_component_count"],
            "enclosed_hole_count": evaluation["prediction_topology"]["enclosed_hole_count"],
            "offline_research_criteria_met": evaluation["offline_research_criteria"]["all_met"],
            "frozen_hashes_verified": True,
            "ground_truth_used_for_prediction": False,
            "eligible_for_downstream": False,
        }
        rows.append(row)

    fieldnames = list(rows[0])
    with (output_root / "evaluation_matrix_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_fields = ("iou", "precision", "recall", "f1", "outer_boundary_p95_px")
    summary = {
        "protocol_version": "phase2d_c6b3_camera_mask_gt_evaluation_v1",
        "sample_count": len(rows),
        "frozen_inputs_verified_before_ground_truth_read": True,
        "sam2_rerun_count": 0,
        "prediction_recomputed_count": 0,
        "ground_truth_scope": "camera_water_mask_only",
        "prediction_side_gate_modified": False,
        "eligible_for_downstream": False,
        "overall": {field: metric_summary([float(row[field]) for row in rows]) for field in metric_fields},
        "offline_research_criteria_met_count": sum(bool(row["offline_research_criteria_met"]) for row in rows),
        "by_depth_cm": group_summary(rows, "nominal_depth_cm_for_evaluation_grouping"),
        "by_rain_level": group_summary(rows, "rain_level"),
        "samples": rows,
    }
    write_json(output_root / "evaluation_matrix_summary.json", summary)
    print(json.dumps({
        "sample_count": len(rows),
        "offline_research_criteria_met_count": summary["offline_research_criteria_met_count"],
        "iou": summary["overall"]["iou"],
        "output_root": str(output_root),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
