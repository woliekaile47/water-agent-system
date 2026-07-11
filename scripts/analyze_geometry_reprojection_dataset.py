#!/usr/bin/env python3
"""Analyze existing Phase 2D-A reprojection artifacts without rerunning prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.analyze_geometry_reprojection_error import (
    analyze_sequence_artifacts,
    write_geometry_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--prediction-root", default="outputs/synthetic_visual_to_depth_integration")
    parser.add_argument("--evaluation-json", default="outputs/synthetic_visual_to_depth_integration/dataset_evaluation/dataset_evaluations.json")
    parser.add_argument("--output-root", default="outputs/synthetic_visual_to_depth_integration/geometry_diagnostics")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    prediction_root = resolve(args.prediction_root)
    evaluation_path = resolve(args.evaluation_json)
    output_root = resolve(args.output_root)
    evaluations = json.loads(evaluation_path.read_text(encoding="utf-8"))
    water = sorted(
        (item for item in evaluations if not item.get("is_dry")),
        key=lambda item: (item["case_id"], item["rain_level"], item["seed"]),
    )
    if not water:
        raise RuntimeError(f"No water evaluations found in {evaluation_path}")
    records = []
    for index, evaluation in enumerate(water, start=1):
        relative = Path(evaluation["case_id"]) / evaluation["rain_level"] / f"seed_{evaluation['seed']}"
        print(f"[phase2d-b1] {index}/{len(water)} {relative}", flush=True)
        records.append(analyze_sequence_artifacts(prediction_root / relative, evaluation))
    summary = write_geometry_diagnostics(output_root, records)
    print(
        f"[phase2d-b1] complete sequences={len(records)} "
        f"p95_median={summary['p95_distribution']['median_px']:.6g}px output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
