#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Append warning audit log as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.warning.generate_warning_decision import load_warning_config, resolve_project_path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _forecast_depth_by_horizon(forecast_results: list[dict[str, Any]], horizon_min: int) -> float | None:
    for item in forecast_results:
        if int(item["horizon_min"]) == int(horizon_min):
            return float(item["forecast_depth_cm"])
    return None


def write_audit_log(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    output_config = config["output"]
    json_dir = resolve_project_path(root, output_config["json_dir"])
    audit_dir = resolve_project_path(root, output_config["audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    decision_path = json_dir / "warning_decision_result.json"
    forecast_path = resolve_project_path(root, config["input"]["deterministic_forecast_result_json"])
    decision = load_json(decision_path)
    forecast = load_json(forecast_path)
    forecast_results = forecast.get("forecast_results", [])
    audit_path = audit_dir / "warning_audit_log.jsonl"

    event = {
        "audit_event": "S8_warning_generated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "warning_decision": str(decision_path),
            "deterministic_forecast": str(forecast_path),
            "area_volume": decision["source_area_volume_json"],
            "weather_correction": decision["source_weather_correction_json"],
        },
        "output_files": decision.get("output_files", {}),
        "current_mean_depth_cm": decision["current_mean_depth_cm"],
        "forecast_5min_depth_cm": _forecast_depth_by_horizon(forecast_results, 5),
        "forecast_15min_depth_cm": _forecast_depth_by_horizon(forecast_results, 15),
        "forecast_30min_depth_cm": _forecast_depth_by_horizon(forecast_results, 30),
        "forecast_60min_depth_cm": _forecast_depth_by_horizon(forecast_results, 60),
        "overall_warning_level": decision["overall_warning_level"],
        "action_suggestion": decision["action_suggestion"],
        "mvp_note": decision["mvp_note"],
    }

    with audit_path.open("a", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False)
        f.write("\n")

    print(f"[S8][audit] audit log path: {audit_path}")
    return {"warning_audit_log": str(audit_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="S8 warning audit log writer.")
    parser.add_argument("--config", required=True, help="Path to configs/warning_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    write_audit_log(args.config, args.project_root)


if __name__ == "__main__":
    main()
