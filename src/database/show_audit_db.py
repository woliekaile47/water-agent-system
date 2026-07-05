#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print recent SQLite audit database records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.audit_db import init_db, load_artifacts, load_recent_runs, load_stage_runs


def show_audit_db(db_path: str | Path, limit: int = 10) -> None:
    db = Path(db_path).expanduser()
    init_db(db)
    recent_runs = load_recent_runs(db, limit=limit)
    print(f"[audit_db] database: {db}")
    print(f"[audit_db] recent pipeline_runs (limit={limit}):")
    if not recent_runs:
        print("  - no pipeline_runs records")
        return
    for run in recent_runs:
        print(
            "  - "
            f"run_id={run['run_id']}, status={run['status']}, "
            f"overall={run['overall_warning_level']}, start={run['start_time']}, "
            f"end={run['end_time']}, forecast_source={run.get('forecast_source')}"
        )
        if run.get("final_forecast_5min_cm") is not None:
            print(
                "    "
                "final forecast 5/15/30/60 min cm: "
                f"{run.get('final_forecast_5min_cm')} / "
                f"{run.get('final_forecast_15min_cm')} / "
                f"{run.get('final_forecast_30min_cm')} / "
                f"{run.get('final_forecast_60min_cm')}"
            )

    latest_run_id = recent_runs[0]["run_id"]
    print(f"[audit_db] stage_runs for latest run: {latest_run_id}")
    for stage in load_stage_runs(db, latest_run_id):
        print(
            "  - "
            f"{stage['stage_name']}: status={stage['status']}, "
            f"return_code={stage['return_code']}, config={stage['config_path']}"
        )

    print(f"[audit_db] artifacts for latest run: {latest_run_id}")
    for artifact in load_artifacts(db, latest_run_id):
        exists_text = "exists" if artifact["exists_flag"] else "missing"
        print(f"  - {artifact['artifact_type']}: {exists_text}, {artifact['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show water_agent_system SQLite audit records.")
    parser.add_argument("--db", required=True, help="Path to water_agent_audit.db")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent pipeline runs to print")
    args = parser.parse_args()
    show_audit_db(args.db, args.limit)


if __name__ == "__main__":
    main()
