#!/usr/bin/env python3
"""Audit frozen automatic prompts and SAM 2 masks using existing GT evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.analyze_temporal_sam2_prompt_failures import (  # noqa: E402
    classify_failure_mode,
    prompt_ground_truth_support,
)
from src.evaluation.evaluate_temporal_sam2_mask_gt import (  # noqa: E402
    load_camera_mask_ground_truth,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prompt-output-root", type=Path, required=True)
    parser.add_argument("--sam2-output-root", type=Path, required=True)
    parser.add_argument("--gt-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Draw a dependency-light diagnostic chart with Pillow."""
    width, height = 1500, 820
    left, right = 90, 40
    top_a, bottom_a = 80, 365
    top_b, bottom_b = 450, 735
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), "Frozen Automatic Prompt Support vs SAM 2 Camera-mask Quality", fill="black")
    plot_width = width - left - right
    step = plot_width / max(1, len(rows))

    def y_position(value: float, top: int, bottom: int) -> float:
        return bottom - max(0.0, min(1.0, float(value))) * (bottom - top)

    for top, bottom in ((top_a, bottom_a), (top_b, bottom_b)):
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, width - right, bottom), fill="black", width=2)
        for value in (0.0, 0.5, 0.9, 1.0):
            y = y_position(value, top, bottom)
            draw.line((left, y, width - right, y), fill=(210, 210, 210), width=1)
            draw.text((35, y - 7), f"{value:.1f}", fill="black")

    series = (
        ("precision", (210, 55, 55)),
        ("recall", (45, 90, 210)),
        ("box_gt_water_coverage", (45, 150, 70)),
    )
    for key, color in series:
        points = []
        for index, row in enumerate(rows):
            x = left + (index + 0.5) * step
            y = y_position(float(row[key]), top_a, bottom_a)
            points.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    draw.text((left, top_a - 25), "precision (red) / recall (blue) / box GT coverage (green)", fill="black")

    for index, row in enumerate(rows):
        center = left + (index + 0.5) * step
        half = max(5.0, step * 0.16)
        pos_y = y_position(float(row["positive_point_gt_support_rate"]), top_b, bottom_b)
        neg_y = y_position(float(row["negative_point_gt_correct_rate"]), top_b, bottom_b)
        draw.rectangle((center - 2 * half, pos_y, center - 2, bottom_b), fill=(230, 150, 35))
        draw.rectangle((center + 2, neg_y, center + 2 * half, bottom_b), fill=(120, 75, 185))
        draw.text((center - 30, bottom_b + 10), row["sample_id"], fill="black")
    draw.text((left, top_b - 25), "positive point GT support (orange) / negative point correctness (purple)", fill="black")
    image.save(path)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    prompt_root = args.prompt_output_root.expanduser().resolve()
    sam2_root = args.sam2_output_root.expanduser().resolve()
    evaluation_root = args.gt_evaluation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite failure audit output: {output_root}")
    matrix = _read_json(evaluation_root / "evaluation_matrix_summary.json")
    if matrix.get("sample_count") != 12 or matrix.get("ground_truth_scope") != "camera_water_mask_only":
        raise ValueError("Expected the frozen 12-sample Camera-only GT evaluation")
    rows: list[dict[str, Any]] = []
    for evaluated in matrix["samples"]:
        sample_id = evaluated["sample_id"]
        prompt_path = prompt_root / sample_id / "automatic_prompt.json"
        prompt_diagnostics_path = prompt_root / sample_id / "automatic_prompt_diagnostics.json"
        mask_path = sam2_root / sample_id / "prompted_mask_raw.npy"
        evaluation_path = evaluation_root / sample_id / "camera_mask_evaluation.json"
        for path in (prompt_path, prompt_diagnostics_path, mask_path, evaluation_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        prompt = _read_json(prompt_path)
        prompt_diagnostics = _read_json(prompt_diagnostics_path)
        evaluation = _read_json(evaluation_path)
        frozen = evaluation["frozen_input_verification"]
        if sha256_file(mask_path) != frozen["mask_sha256"]:
            raise ValueError(f"Frozen mask hash changed for {sample_id}")
        gt = load_camera_mask_ground_truth(
            root,
            evaluated["case_id"],
            evaluated["rain_level"],
            int(evaluated["seed"]),
            int(evaluated["frame_index"]),
        )["camera_mask"]
        predicted = np.load(mask_path, allow_pickle=False).astype(bool)
        support = prompt_ground_truth_support(prompt, predicted, gt)
        classification = classify_failure_mode(
            evaluated,
            evaluated["outer_boundary_p95_px"],
            support,
        )
        row = {
            "sample_id": sample_id,
            "case_id": evaluated["case_id"],
            "nominal_depth_cm_for_evaluation_grouping": evaluated["nominal_depth_cm_for_evaluation_grouping"],
            "rain_level": evaluated["rain_level"],
            "prompt_status": evaluated["prompt_status"],
            "temporal_quality_gate_status": prompt_diagnostics["temporal_quality_gate_status"],
            "iou": evaluated["iou"],
            "precision": evaluated["precision"],
            "recall": evaluated["recall"],
            "outer_boundary_p95_px": evaluated["outer_boundary_p95_px"],
            "selected_temporal_component_area_pixels": prompt_diagnostics["selected_component_area_pixels"],
            "sam2_mask_area_pixels": evaluated["predicted_pixels"],
            **support,
            **classification,
            "prediction_recomputed": False,
            "sam2_rerun": False,
            "gate_modified": False,
            "eligible_for_downstream": False,
        }
        rows.append(row)

    output_root.mkdir(parents=True)
    fieldnames = list(rows[0])
    with (output_root / "failure_audit_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if not row["offline_research_criteria_met"]]
    summary = {
        "protocol_version": "phase2d_c6b4_failure_audit_v1",
        "sample_count": len(rows),
        "failure_sample_count": len(failures),
        "failure_mode_counts": dict(Counter(row["evaluation_failure_mode"] for row in failures)),
        "diagnostic_attribution_counts": dict(Counter(row["diagnostic_attribution"] for row in failures)),
        "key_findings": {
            "shallow_heavy_positive_prompt_contamination": all(
                next(row for row in rows if row["sample_id"] == sid)["positive_point_gt_support_rate"] <= 0.60
                for sid in ("c6b2_003", "c6b2_006")
            ),
            "forty_cm_light_box_scope_truncation": (
                next(row for row in rows if row["sample_id"] == "c6b2_010")["diagnostic_attribution"]
                == "prompt_box_scope_truncation"
            ),
            "prompt_gate_is_not_sam2_accuracy_gate": any(
                row["prompt_status"] == "pass" and not row["offline_research_criteria_met"] for row in rows
            ) and any(
                row["prompt_status"] != "pass" and row["offline_research_criteria_met"] for row in rows
            ),
        },
        "failure_samples": failures,
        "all_samples": rows,
        "ground_truth_used_for_prediction": False,
        "prediction_or_prompt_modified": False,
        "gate_modified": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    _write_json(output_root / "failure_audit_summary.json", summary)
    _save_plot(rows, output_root / "prompt_support_vs_mask_quality.png")
    print(json.dumps({
        "sample_count": len(rows),
        "failure_sample_count": len(failures),
        "failure_mode_counts": summary["failure_mode_counts"],
        "diagnostic_attribution_counts": summary["diagnostic_attribution_counts"],
        "output_root": str(output_root),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
