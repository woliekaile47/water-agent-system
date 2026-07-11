#!/usr/bin/env python3
"""Select and visualize existing shoreline artifacts without rerunning prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.audit_shoreline_cases import (
    audit_case,
    case_key,
    select_audit_cases,
    write_audit_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--prediction-root", default="outputs/synthetic_visual_to_depth_integration")
    parser.add_argument("--geometry-diagnostics", default="outputs/synthetic_visual_to_depth_integration/geometry_diagnostics/geometry_diagnostics.json")
    parser.add_argument("--output-root", default="outputs/synthetic_visual_to_depth_integration/shoreline_case_audit")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    prediction_root = resolve(args.prediction_root)
    diagnostics = json.loads(resolve(args.geometry_diagnostics).read_text(encoding="utf-8"))
    output_root = resolve(args.output_root)
    selection = select_audit_cases(diagnostics)
    selected_lookup = {item["case_key"]: item for item in selection["selected_cases"]}
    selected_records = [record for record in diagnostics if case_key(record) in selected_lookup]
    selected_records.sort(key=lambda row: (row["case_id"], row["rain_level"], row["seed"]))
    cases = []
    for index, record in enumerate(selected_records, start=1):
        key = case_key(record)
        relative = Path(record["case_id"]) / record["rain_level"] / f"seed_{record['seed']}"
        print(f"[phase2d-b2a] {index}/{len(selected_records)} {relative}", flush=True)
        cases.append(audit_case(prediction_root / relative, record, selected_lookup[key]["selection_groups"]))
    summary = write_audit_outputs(output_root, selection, cases)
    print(
        f"[phase2d-b2a] complete selected_unique={summary['audited_unique_case_count']} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
