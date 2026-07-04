#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Generate Markdown warning report."""

from __future__ import annotations

import argparse
import json
import sys
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


def _fmt_time(value: Any) -> str:
    if value is None:
        return "not reached in current rising model"
    return f"{float(value):.2f} min"


def generate_warning_report(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    output_config = config["output"]
    json_dir = resolve_project_path(root, output_config["json_dir"])
    report_dir = resolve_project_path(root, output_config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    decision_path = json_dir / "warning_decision_result.json"
    forecast_path = resolve_project_path(root, config["input"]["deterministic_forecast_result_json"])
    decision = load_json(decision_path)
    forecast = load_json(forecast_path)
    report_path = report_dir / "warning_report.md"
    times = decision.get("time_to_thresholds_min", {})

    forecast_lines = [
        f"- {item['horizon_min']} min: {item['forecast_depth_cm']:.2f} cm, warning {item['warning_level']}"
        for item in decision["forecast_warning_results"]
    ]
    input_paths = [
        decision["source_deterministic_forecast_json"],
        decision["source_area_volume_json"],
        decision["source_weather_correction_json"],
    ]
    output_paths = [
        decision["output_files"]["data_warning_decision"],
        decision["output_files"]["output_warning_decision"],
        str(report_path),
        decision["output_files"]["audit_log"],
        decision["output_files"]["warning_summary"],
    ]

    markdown = f"""# S8 Waterlogging Warning Report - MVP

This report is generated from MVP simulation data and is not final real emergency dispatch advice.

## Current Status

- Overall warning level: **{decision['overall_warning_level']}**
- Current mean depth: {decision['current_mean_depth_cm']:.2f} cm
- Current max depth: {decision['current_max_depth_cm']:.2f} cm
- Confidence: {decision['confidence_score']:.2f} ({decision['confidence_mode']})

## S5 Area And Volume

- Water area: {decision['water_area_m2']:.4f} m2
- Water volume: {decision['water_volume_m3']:.4f} m3

## S6 Weather Correction

- Rainfall intensity: {decision['rainfall_intensity_mm_h']:.2f} mm/h
- Rainfall level: {decision['rainfall_level_label']}
- Weather correction factor: {decision['weather_correction_factor']:.2f}

## S7-A Forecast

- k_forecast: {forecast.get('k_forecast_cm_per_min', 0.0):.4f} cm/min
- Time to blue threshold: {_fmt_time(times.get('blue'))}
- Time to yellow threshold: {_fmt_time(times.get('yellow'))}
- Time to orange threshold: {_fmt_time(times.get('orange'))}

{chr(10).join(forecast_lines)}

## S8 Warning Decision

- Overall warning level: **{decision['overall_warning_level']}**
- Action suggestion: {decision['action_suggestion']}

## MVP Note

{decision['mvp_note']}

{decision.get('upstream_mvp_note', '')}

## Input Files

{chr(10).join(f'- `{path}`' for path in input_paths)}

## Output Files

{chr(10).join(f'- `{path}`' for path in output_paths)}
"""
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[S8][report] report path: {report_path}")
    return {"warning_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="S8 Markdown warning report generation.")
    parser.add_argument("--config", required=True, help="Path to configs/warning_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    generate_warning_report(args.config, args.project_root)


if __name__ == "__main__":
    main()
