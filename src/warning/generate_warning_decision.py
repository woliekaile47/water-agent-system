#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Generate warning decision from the latest available S7 forecast."""

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


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
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


def compute_time_to_thresholds(
    forecast_results: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, float | None]:
    times: dict[str, float | None] = {"blue": None, "yellow": None, "orange": None}
    sorted_results = sorted(forecast_results, key=lambda item: int(item["horizon_min"]))
    for name in ("blue", "yellow", "orange"):
        threshold_cm = float(thresholds[name])
        for item in sorted_results:
            if float(item["forecast_depth_cm"]) >= threshold_cm:
                times[name] = float(item["horizon_min"])
                break
    return times


def _source_path(root: Path, input_config: dict[str, Any], key: str) -> Path | None:
    value = input_config.get(key)
    if not value:
        return None
    return resolve_project_path(root, value)


def _build_final_forecast_results(
    final_forecast: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in final_forecast.get("final_forecast_results", []):
        depth_cm = float(item["final_forecast_depth_cm"])
        level = classify_warning_level(depth_cm, thresholds)
        results.append(
            {
                "horizon_min": int(item["horizon_min"]),
                "forecast_depth_cm": depth_cm,
                "warning_level": level,
                "source_warning_level": item.get("warning_level"),
                "physical_confidence": item.get("physical_confidence"),
                "physical_check": item.get("physical_check"),
                "source_corrected_depth_cm": item.get("source_corrected_depth_cm"),
            }
        )
    return results


def _build_deterministic_forecast_results(
    deterministic_forecast: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in deterministic_forecast.get("forecast_results", []):
        depth_cm = float(item["forecast_depth_cm"])
        level = classify_warning_level(depth_cm, thresholds)
        results.append(
            {
                "horizon_min": int(item["horizon_min"]),
                "forecast_depth_cm": depth_cm,
                "warning_level": level,
                "source_warning_level": item.get("warning_level"),
                "physical_confidence": None,
            }
        )
    return results


def _first_or_none(values: list[Any]) -> Any:
    return values[0] if values else None


def _make_action_suggestion(
    overall_warning_level: str,
    action_policy: dict[str, Any],
    forecast_warning_results: list[dict[str, Any]],
    time_to_thresholds: dict[str, Any],
) -> str:
    suggestion = str(action_policy.get(overall_warning_level, action_policy.get("none", "")))
    orange_horizons = [
        int(item["horizon_min"])
        for item in forecast_warning_results
        if item.get("warning_level") == "orange"
    ]
    first_orange = _first_or_none(sorted(orange_horizons))
    orange_time = time_to_thresholds.get("orange")
    if first_orange is not None:
        suggestion += f" First orange forecast horizon: {first_orange} min."
    if orange_time is not None and float(orange_time) <= 5.0:
        suggestion += " Orange threshold may be reached very soon. Immediate action is recommended."
    return suggestion


def generate_warning_decision(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    input_config = config["input"]
    output_config = config["output"]
    thresholds = config["warning_thresholds_cm"]

    final_path = _source_path(root, input_config, "final_forecast_result_json")
    physical_path = _source_path(root, input_config, "physical_constraint_result_json")
    case_path = _source_path(root, input_config, "case_retrieval_result_json")
    corrected_path = _source_path(root, input_config, "corrected_forecast_result_json")
    deterministic_path = resolve_project_path(root, input_config["deterministic_forecast_result_json"])
    area_volume_path = resolve_project_path(root, input_config["area_volume_result_json"])
    weather_path = resolve_project_path(root, input_config["weather_correction_result_json"])

    area_volume = load_json(area_volume_path)
    weather = load_json(weather_path)
    deterministic_forecast = load_optional_json(deterministic_path)
    final_forecast = load_optional_json(final_path) if final_path is not None else None
    physical_constraint = load_optional_json(physical_path) if physical_path is not None else None
    case_retrieval = load_optional_json(case_path) if case_path is not None else None
    corrected_forecast = load_optional_json(corrected_path) if corrected_path is not None else None

    warnings: list[str] = []
    if final_forecast is not None:
        forecast_source = "S7C_final_forecast"
        forecast_warning_results = _build_final_forecast_results(final_forecast, thresholds)
        physical_confidence_summary = final_forecast.get("physical_confidence_summary")
        s7_pipeline_used = ["S7A", "S7B", "S7C"]
        source_final_forecast_json: str | None = str(final_path)
        upstream_mvp_note = final_forecast.get("mvp_note")
    else:
        if deterministic_forecast is None:
            deterministic_forecast = load_json(deterministic_path)
        forecast_source = "S7A_deterministic_fallback"
        forecast_warning_results = _build_deterministic_forecast_results(
            deterministic_forecast,
            thresholds,
        )
        physical_confidence_summary = None
        s7_pipeline_used = ["S7A"]
        source_final_forecast_json = None
        upstream_mvp_note = deterministic_forecast.get("mvp_note")
        warnings.append(
            "final_forecast_result_json was not found; S8 fell back to S7-A deterministic forecast."
        )

    if deterministic_forecast is None:
        deterministic_forecast = {}

    overall_warning_level = get_overall_warning_level(
        [item["warning_level"] for item in forecast_warning_results]
    )
    time_to_thresholds = compute_time_to_thresholds(forecast_warning_results, thresholds)
    action_suggestion = _make_action_suggestion(
        overall_warning_level=overall_warning_level,
        action_policy=config.get("action_policy", {}),
        forecast_warning_results=forecast_warning_results,
        time_to_thresholds=time_to_thresholds,
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
    current_mean_depth_cm = float(
        deterministic_forecast.get(
            "current_mean_depth_cm",
            physical_constraint.get("current_mean_depth_cm") if physical_constraint else area_volume["mean_depth_cm"],
        )
    )

    result = {
        "stage": "S8_warning_decision",
        "warning_mode": config.get("mode", "offline_mvp_warning"),
        "forecast_source": forecast_source,
        "source_final_forecast_json": source_final_forecast_json,
        "source_physical_constraint_json": str(physical_path) if physical_path is not None else None,
        "source_case_retrieval_json": str(case_path) if case_path is not None else None,
        "source_corrected_forecast_json": str(corrected_path) if corrected_path is not None else None,
        "source_deterministic_forecast_json": str(deterministic_path),
        "source_area_volume_json": str(area_volume_path),
        "source_weather_correction_json": str(weather_path),
        "current_mean_depth_cm": current_mean_depth_cm,
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
        "physical_confidence_summary": physical_confidence_summary,
        "s7_pipeline_used": s7_pipeline_used,
        "s7_summary": {
            "deterministic_forecast_available": bool(deterministic_forecast),
            "case_retrieval_available": case_retrieval is not None,
            "corrected_forecast_available": corrected_forecast is not None,
            "physical_constraint_available": physical_constraint is not None,
            "final_forecast_available": final_forecast is not None,
        },
        "confidence_mode": confidence.get("mode", "mvp_rule_based"),
        "confidence_score": float(confidence.get("default_confidence_score", 0.75)),
        "mvp_note": config.get("mvp_note"),
        "upstream_mvp_note": upstream_mvp_note,
        "warnings": warnings,
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

    orange_time = time_to_thresholds.get("orange")
    print(f"[S8][decision] forecast source: {forecast_source}")
    print(f"[S8][decision] S7 pipeline used: {', '.join(s7_pipeline_used)}")
    print(f"[S8][decision] overall warning level: {overall_warning_level}")
    print(f"[S8][decision] current mean depth cm: {result['current_mean_depth_cm']:.2f}")
    for item in forecast_warning_results:
        confidence_text = item.get("physical_confidence")
        confidence_suffix = f", physical confidence: {confidence_text}" if confidence_text else ""
        print(
            "[S8][decision] "
            f"{item['horizon_min']} min warning level: {item['warning_level']} "
            f"({item['forecast_depth_cm']:.2f} cm{confidence_suffix})"
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
