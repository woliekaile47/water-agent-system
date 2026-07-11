#!/usr/bin/env python3
"""Sequential prediction-then-evaluation runner for the Phase 2D-A dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_synthetic_visual_to_depth import evaluate_synthetic_visual_to_depth_case
from src.evaluation.synthetic_visual_to_depth_summary import write_summary_outputs
from src.integration.synthetic_visual_to_depth import run_synthetic_visual_to_depth_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--data-root", default="data/simulation_dynamic")
    parser.add_argument("--output-root", default="outputs/synthetic_visual_to_depth_integration/dataset_evaluation")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--rain-level")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = root / data_root
    data_root = data_root.resolve()
    summary_root = Path(args.output_root).expanduser()
    if not summary_root.is_absolute():
        summary_root = root / summary_root
    config_path = root / "configs/synthetic_visual_to_depth_integration.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        integration_config = yaml.safe_load(stream)["synthetic_visual_to_depth_integration"]
    prediction_root = root / integration_config["output_root"]
    sequences = []
    for frames in sorted(data_root.glob("*/*/seed_*/frames")):
        sequence = frames.parent
        case_id, rain_level = sequence.parents[1].name, sequence.parent.name
        seed = int(sequence.name.removeprefix("seed_"))
        if args.case_id and case_id != args.case_id:
            continue
        if args.rain_level and rain_level != args.rain_level:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        sequences.append(sequence)
    if not sequences:
        raise RuntimeError(f"No valid dynamic sequences selected under {data_root}")
    evaluations = []
    failures = []
    for index, sequence in enumerate(sequences, start=1):
        relative = sequence.relative_to(data_root)
        prediction_output = prediction_root / relative
        print(f"[phase2d-a-dataset] {index}/{len(sequences)} {relative}", flush=True)
        try:
            if not args.skip_prediction:
                run_synthetic_visual_to_depth_prediction(
                    root, sequence / "frames", prediction_output, integration_config,
                )
            evaluation = evaluate_synthetic_visual_to_depth_case(sequence, prediction_output, root)
            evaluation_path = prediction_output / "evaluation_metrics.json"
            evaluation_path.write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            evaluations.append(evaluation)
            print(
                f"[phase2d-a-dataset] evaluated {relative}: "
                f"visual={evaluation['gate']['visual_gate_status']} "
                f"geometry={evaluation['gate']['geometry_gate_status']} "
                f"integration={evaluation['gate']['integration_gate_status']} "
                f"measurement={evaluation['gate']['measurement_status']}",
                flush=True,
            )
        except Exception as error:  # Batch audit records the real sequence failure and continues.
            failures.append({"sequence": str(relative), "error_type": type(error).__name__, "message": str(error)})
            print(f"[phase2d-a-dataset] FAILED {relative}: {type(error).__name__}: {error}", flush=True)
    summary = write_summary_outputs(summary_root, evaluations)
    (summary_root / "batch_failures.json").write_text(
        json.dumps({"failure_count": len(failures), "failures": failures}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary["selected_sequence_count"] = len(sequences)
    summary["evaluation_success_count"] = len(evaluations)
    summary["batch_failure_count"] = len(failures)
    (summary_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(
        f"[phase2d-a-dataset] complete selected={len(sequences)} evaluated={len(evaluations)} "
        f"failed={len(failures)} output={summary_root}", flush=True,
    )


if __name__ == "__main__":
    main()
