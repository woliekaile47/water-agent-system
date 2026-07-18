#!/usr/bin/env python3
"""Confirm the frozen C8 candidate gate using independent seed-303 GT evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-gate-root", type=Path, required=True)
    parser.add_argument("--camera-evaluation-root", type=Path, required=True)
    parser.add_argument("--geometry-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row["evaluation_available"]]
    passes = [row for row in evaluated if row["camera_visible_status"] == "pass"]
    counterexamples = [row for row in passes if not row["water_level_within_3cm"]]
    return {
        "frame_count": len(rows),
        "evaluated_frame_count": len(evaluated),
        "camera_visible_status_counts": dict(sorted(Counter(row["camera_visible_status"] for row in rows).items())),
        "global_scene_status_counts": dict(sorted(Counter(row["global_scene_status"] for row in rows).items())),
        "visible_pass_count": len(passes),
        "visible_pass_within_3cm_count": sum(row["water_level_within_3cm"] for row in passes),
        "visible_pass_outside_3cm_count": len(counterexamples),
        "visible_pass_outside_3cm_cases": [
            {"sample_id": row["sample_id"], "frame_index": row["frame_index"], "absolute_error_cm": row["water_level_absolute_error_cm"]}
            for row in counterexamples
        ],
        "water_level_absolute_error_cm_all": metric([row["water_level_absolute_error_cm"] for row in evaluated]),
        "water_level_absolute_error_cm_visible_pass": metric([row["water_level_absolute_error_cm"] for row in passes]),
        "camera_mask_iou_all": metric([row["camera_mask_iou"] for row in evaluated]),
        "camera_mask_iou_visible_pass": metric([row["camera_mask_iou"] for row in passes]),
        "visible_area_relative_error_visible_pass": metric([row["visible_area_relative_error"] for row in passes]),
        "visible_volume_relative_error_visible_pass": metric([row["visible_volume_relative_error"] for row in passes]),
    }


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_gate_root.expanduser().resolve()
    camera_root = args.camera_evaluation_root.expanduser().resolve()
    geometry_root = args.geometry_evaluation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite seed-303 confirmation: {output_root}")

    candidate_summary = read_json(candidate_root / "candidate_gate_summary.json")
    candidates = read_json(candidate_root / "candidate_gate_per_frame.json")
    camera_summary = read_json(camera_root / "evaluation_summary.json")
    cameras = read_json(camera_root / "per_frame_evaluation.json")
    geometry_summary = read_json(geometry_root / "evaluation_summary.json")
    geometries = read_json(geometry_root / "per_frame_scalar_evaluation.json")
    expected = int(candidate_summary["frame_count"])
    if len(candidates) != expected or len(cameras) != expected or len(geometries) != expected:
        raise ValueError("candidate/camera/geometry frame counts do not match")
    if candidate_summary.get("ground_truth_used") is not False:
        raise ValueError("candidate gate freeze contains invalid GT provenance")
    if camera_summary.get("prediction_recomputed_count") != 0 or geometry_summary.get("geometry_prediction_recomputed_count") != 0:
        raise ValueError("independent evaluation unexpectedly recomputed prediction")

    camera_by_key = {(row["sample_id"], int(row["frame_index"])): row for row in cameras}
    geometry_by_key = {(row["sample_id"], int(row["frame_index"])): row for row in geometries}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate["sample_id"], int(candidate["frame_index"]))
        camera = camera_by_key[key]
        geometry = geometry_by_key[key]
        row = {
            "sample_id": key[0],
            "case_id": candidate["case_id"],
            "rain_level": candidate["rain_level"],
            "seed": int(candidate["seed"]),
            "frame_index": key[1],
            "camera_visible_status": candidate["camera_visible_status"],
            "global_scene_status": candidate["global_scene_status"],
            "candidate_warnings": candidate["warnings"],
            "candidate_visible_reject_reasons": candidate["visible_reject_reasons"],
            "evaluation_available": bool(geometry["evaluation_available"]),
            "water_level_absolute_error_cm": geometry.get("water_level_absolute_error_cm"),
            "water_level_within_3cm": bool(geometry.get("water_level_within_3cm", False)),
            "visible_area_relative_error": geometry.get("visible_area_relative_error"),
            "visible_volume_relative_error": geometry.get("visible_volume_relative_error"),
            "camera_mask_iou": float(camera["iou"]),
            "camera_outer_boundary_p95_px": float(camera["outer_boundary_p95_px"]),
            "ground_truth_used_for_candidate_gate": False,
            "eligible_for_downstream": False,
        }
        rows.append(row)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        depth = row["case_id"].split("_")[2].replace("cm", "cm")
        groups[f"depth:{depth}"].append(row)
        groups[f"rain:{row['rain_level']}"] .append(row)
        groups[f"sequence:{row['sample_id']}"] .append(row)
    overall = summarize(rows)
    summary = {
        "protocol_version": "phase2d_c8_seed303_candidate_gate_independent_confirmation_v1",
        "confirmation_role": "final_confirmation_not_threshold_selection",
        "sample_count": int(candidate_summary["sample_count"]),
        "frame_count": len(rows),
        "candidate_gate_frozen_before_seed303_prediction": bool(candidate_summary["candidate_gate_frozen_before_confirmation"]),
        "candidate_gate_config_sha256": candidate_summary["candidate_gate_config_sha256"],
        "ground_truth_used_for_candidate_gate": False,
        "thresholds_modified_after_confirmation": False,
        "sam2_rerun_count": 0,
        "geometry_prediction_recomputed_count": 0,
        "authoritative": False,
        "eligible_for_downstream": False,
        "overall": overall,
        "groups": {name: summarize(items) for name, items in sorted(groups.items())},
        "confirmation_status": "pass" if overall["visible_pass_count"] > 0 and overall["visible_pass_outside_3cm_count"] == 0 else "fail",
        "interpretation": "A pass confirms only the offline candidate gate against the project 3 cm water-level target; it does not authorize S5-S8 integration.",
    }
    output_root.mkdir(parents=True)
    write_json(output_root / "gate_confirmation_summary.json", summary)
    write_json(output_root / "gate_confirmation_per_frame.json", rows)
    report = [
        "# Phase 2D-C-8-3C seed 303 独立门控确认",
        "",
        "候选门控在 seed 303 预测和 GT 读取前已经冻结。GT 仅由独立评价读取；没有据此修改提示、prediction 或阈值。",
        "",
        f"- Camera-visible pass: {overall['visible_pass_count']}",
        f"- Pass 且水位误差不超过 3 cm: {overall['visible_pass_within_3cm_count']}",
        f"- Pass 但水位误差超过 3 cm: {overall['visible_pass_outside_3cm_count']}",
        f"- 独立确认结论: {summary['confirmation_status']}",
        "- 边界 P95 继续作为诊断项，不单独触发候选门控拒绝。",
        "- 本结果仍非 authoritative，不能进入正式 S5-S8。",
        "",
    ]
    (output_root / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
