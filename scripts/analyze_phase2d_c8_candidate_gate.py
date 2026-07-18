#!/usr/bin/env python3
"""Compare the offline C8 candidate gate with the frozen legacy gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase2d_c8_candidate_quality_gate import evaluate_candidate_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/sam2_video_geometry_stability_seed302_frame79_119"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/sam2_video_geometry_gt_evaluation_seed302_frame79_119"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2d_c8_candidate_quality_gate.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2d_c8_candidate_gate_seed302_video"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def compare_candidate_with_evaluation(
    prediction_root: Path,
    evaluation_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Finish all candidate decisions before opening independent GT evaluation files."""
    candidate_rows: list[dict[str, Any]] = []
    sequences = list(config["evidence_sequences"])
    for sample_id in sequences:
        frame_rows = read_json(prediction_root / sample_id / "per_frame_geometry_summary.json")
        sequence = read_json(prediction_root / sample_id / "sequence_geometry_stability.json")
        for frame in frame_rows:
            candidate = evaluate_candidate_gate(frame, sequence, config)
            candidate_rows.append({
                "sample_id": sample_id,
                "legacy_quality_status": frame["quality_status"],
                "legacy_observable_region_result_valid": bool(
                    frame["observable_region_result_valid"]
                ),
                "legacy_global_estimate_status": frame["global_estimate_status"],
                "legacy_gate_reasons": list(frame.get("gate_reasons", [])),
                "camera_reprojection_iou": frame["camera_reprojection_iou"],
                "outer_boundary_reprojection_p95_px": frame[
                    "outer_boundary_reprojection_p95_px"
                ],
                **candidate,
            })

    # GT is opened only after every prediction-side candidate decision is complete.
    evaluation_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for sample_id in sequences:
        for evaluation in read_json(
            evaluation_root / sample_id / "per_frame_scalar_evaluation.json"
        ):
            evaluation_by_key[(sample_id, int(evaluation["frame_index"]))] = evaluation

    joined: list[dict[str, Any]] = []
    for row in candidate_rows:
        evaluation = evaluation_by_key[(row["sample_id"], int(row["frame_index"]))]
        within_3cm = bool(evaluation["water_level"]["within_project_3cm_target"])
        joined.append({
            **row,
            "evaluation_water_level_absolute_error_cm": evaluation["water_level"][
                "absolute_error_cm"
            ],
            "evaluation_water_level_within_3cm": within_3cm,
            "evaluation_visible_area_relative_error": evaluation[
                "area_camera_visible_main_basin_m2"
            ]["relative_error"],
            "evaluation_visible_volume_relative_error": evaluation[
                "volume_camera_visible_main_basin_m3"
            ]["relative_error"],
            "ground_truth_used_for_candidate_decision": False,
        })

    summary = {
        "protocol_version": config["protocol_version"],
        "frame_count": len(joined),
        "all_candidate_decisions_completed_before_ground_truth_read": True,
        "prediction_recomputed": False,
        "sam2_rerun_count": 0,
        "legacy_gate_modified": False,
        "candidate_replaces_runtime_gate": False,
        "candidate_camera_visible_status_counts": _count(joined, "camera_visible_status"),
        "candidate_global_scene_status_counts": _count(joined, "global_scene_status"),
        "legacy_quality_status_counts": _count(joined, "legacy_quality_status"),
        "legacy_observable_region_valid_count": sum(
            row["legacy_observable_region_result_valid"] for row in joined
        ),
        "candidate_camera_visible_pass_count": sum(
            row["camera_visible_status"] == "pass" for row in joined
        ),
        "legacy_visible_reject_to_candidate_pass_count": sum(
            not row["legacy_observable_region_result_valid"]
            and row["camera_visible_status"] == "pass"
            for row in joined
        ),
        "legacy_visible_pass_to_candidate_reject_count": sum(
            row["legacy_observable_region_result_valid"]
            and row["camera_visible_status"] == "reject"
            for row in joined
        ),
        "candidate_visible_pass_outside_3cm_count": sum(
            row["camera_visible_status"] == "pass"
            and not row["evaluation_water_level_within_3cm"]
            for row in joined
        ),
        "candidate_visible_reject_within_3cm_count": sum(
            row["camera_visible_status"] == "reject"
            and row["evaluation_water_level_within_3cm"]
            for row in joined
        ),
        "warning_counts": dict(sorted(Counter(
            warning for row in joined for warning in row["warnings"]
        ).items())),
        "visible_reject_reason_counts": dict(sorted(Counter(
            reason for row in joined for reason in row["visible_reject_reasons"]
        ).items())),
        "global_scope_reason_counts": dict(sorted(Counter(
            reason for row in joined for reason in row["global_scope_reasons"]
        ).items())),
        "by_sequence": {},
        "ground_truth_used_for_candidate_decision": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    for sample_id in sequences:
        selected = [row for row in joined if row["sample_id"] == sample_id]
        summary["by_sequence"][sample_id] = {
            "frame_count": len(selected),
            "legacy_quality_status_counts": _count(selected, "legacy_quality_status"),
            "legacy_observable_region_valid_count": sum(
                row["legacy_observable_region_result_valid"] for row in selected
            ),
            "candidate_camera_visible_status_counts": _count(selected, "camera_visible_status"),
            "candidate_global_scene_status_counts": _count(selected, "global_scene_status"),
            "water_level_within_3cm_count": sum(
                row["evaluation_water_level_within_3cm"] for row in selected
            ),
            "candidate_visible_pass_outside_3cm_count": sum(
                row["camera_visible_status"] == "pass"
                and not row["evaluation_water_level_within_3cm"]
                for row in selected
            ),
        }
    return joined, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "frame_index", "legacy_quality_status",
        "legacy_observable_region_result_valid", "camera_visible_status",
        "global_scene_status", "result_semantics", "camera_reprojection_iou",
        "outer_boundary_reprojection_p95_px", "evaluation_water_level_absolute_error_cm",
        "evaluation_water_level_within_3cm", "evaluation_visible_area_relative_error",
        "evaluation_visible_volume_relative_error", "visible_reject_reasons",
        "global_scope_reasons", "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "|".join(row[field]) if isinstance(row.get(field), list) else row.get(field)
                for field in fields
            })


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = raw["phase2d_c8_candidate_quality_gate"]
    rows, summary = compare_candidate_with_evaluation(
        args.prediction_root, args.evaluation_root, config
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "candidate_gate_per_frame.json", rows)
    write_csv(args.output_dir / "candidate_gate_per_frame.csv", rows)
    write_json(args.output_dir / "candidate_gate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
