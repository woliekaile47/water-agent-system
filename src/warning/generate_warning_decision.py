#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Generate warning decision from deterministic forecast output."""

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


WARNING_RANK = {"none": 0, "blue": 1, "yellow": 2, "orange": 3}


def load_warning_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read warning config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Warning config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "warning" not in data:
        raise ValueError("Warning config must contain a top-level 'warning' field.")
    return data["warning"]


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_warning_level(depth_cm: float, thresholds: dict[str, Any]) -> str:
    blue = float(thresholds["blue"])
    yellow = float(thresholds["yellow"])
    orange = float(thresholds["orange"])
    if depth_cm >= orange:
        return "orange"
    if depth_cm >= yellow:
        return "yellow"
    if depth_cm >= blue:
        return "blue"
    return "none"


def get_overall_warning_level(levels: list[str]) -> str:
    if not levels:
        return "none"
    return max(levels, key=lambda level: WARNING_RANK.get(level, 0))


def generate_warning_decision(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    input_config = config["input"]
    output_config = config["output"]
    thresholds = config["warning_thresholds_cm"]

    forecast_path = resolve_project_path(root, input_config["deterministic_forecast_result_json"])
    area_volume_path = resolve_project_path(root, input_config["area_volume_result_json"])
    weather_path = resolve_project_path(root, input_config["weather_correction_result_json"])
    forecast = load_json(forecast_path)
    area_volume = load_json(area_volume_path)
    weather = load_json(weather_path)

    forecast_warning_results: list[dict[str, Any]] = []
    for item in forecast.get("forecast_results", []):
        depth_cm = float(item["forecast_depth_cm"])
        level = classify_warning_level(depth_cm, thresholds)
        forecast_warning_results.append(
            {
                "horizon_min": int(item["horizon_min"]),
                "forecast_depth_cm": depth_cm,
                "warning_level": level,
                "source_warning_level": item.get("warning_level"),
            }
        )

    overall_warning_level = get_overall_warning_level(
        [item["warning_level"] for item in forecast_warning_results]
    )
    action_policy = config.get("action_policy", {})
    action_suggestion = str(action_policy.get(overall_warning_level, action_policy.get("none", "")))
    time_to_thresholds = forecast.get("time_to_thresholds_min", {})
    orange_time = time_to_thresholds.get("orange")
    if orange_time is not None and float(orange_time) <= 5.0:
        action_suggestion = (
            action_suggestion
            + " Orange threshold may be reached very soon. Immediate action is recommended."
        )

    warning_dir = resolve_project_path(root, output_config["warning_dir"])
    json_dir = resolve_project_path(root, output_config["json_dir"])
    figure_dir = resolve_project_path(root, output_config["figure_dir"])
    report_dir = resolve_project_path(root, output_config["report_dir"])
    audit_dir = resolve_project_path(root, output_config["audit_dir"])
    for directory in (warning_dir, json_dir, figure_dir, report_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_output_json = warning_dir / "warning_decision_result.json"
    output_json = json_dir / "warning_decision_result.json"
    confidence = config.get("confidence", {})

    result = {
        "stage": "S8_warning_decision",
        "warning_mode": config.get("mode", "offline_mvp_warning"),
        "source_deterministic_forecast_json": str(forecast_path),
        "source_area_volume_json": str(area_volume_path),
        "source_weather_correction_json": str(weather_path),
        "current_mean_depth_cm": float(forecast["current_mean_depth_cm"]),
        "current_max_depth_cm": float(area_volume["max_depth_cm"]),
        "water_area_m2": float(area_volume["water_area_m2"]),
        "water_volume_m3": float(area_volume["water_volume_m3"]),
        "rainfall_intensity_mm_h": float(weather["current_rainfall_intensity_mm_h"]),
        "rainfall_level_label": str(weather["rainfall_level_label"]),
        "weather_correction_factor": float(weather["weather_correction_factor"]),
        "forecast_warning_results": forecast_warning_results,
        "overall_warning_level": overall_warning_level,
        "action_suggestion": action_suggestion,
        "time_to_thresholds_min": time_to_thresholds,
        "confidence_mode": confidence.get("mode", "mvp_rule_based"),
        "confidence_score": float(confidence.get("default_confidence_score", 0.75)),
        "mvp_note": config.get("mvp_note"),
        "upstream_mvp_note": forecast.get("mvp_note"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "data_warning_decision": str(data_output_json),
            "output_warning_decision": str(output_json),
            "warning_report": str(report_dir / "warning_report.md"),
            "warning_summary": str(figure_dir / "warning_summary.png"),
            "audit_log": str(audit_dir / "warning_audit_log.jsonl"),
        },
    }

    for path in (data_output_json, output_json):
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"[S8][decision] overall warning level: {overall_warning_level}")
    print(f"[S8][decision] current mean depth cm: {result['current_mean_depth_cm']:.2f}")
    for item in forecast_warning_results:
        print(
            "[S8][decision] "
            f"{item['horizon_min']} min warning level: {item['warning_level']} "
            f"({item['forecast_depth_cm']:.2f} cm)"
        )
    print(f"[S8][decision] time to orange threshold min: {orange_time}")
    print(f"[S8][decision] action suggestion: {action_suggestion}")
    print("[S8][decision] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S8 warning decision generation.")
    parser.add_argument("--config", required=True, help="Path to configs/warning_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    generate_warning_decision(args.config, args.project_root)


if __name__ == "__main__":
    main()
