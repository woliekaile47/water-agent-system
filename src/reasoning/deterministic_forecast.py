#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-A: Deterministic rule-engine forecast MVP."""

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


MVP_NOTE = (
    "S7-A uses configured_mvp_simulation depth, offline_mock_weather, and "
    "offline_mock_depth_history to validate deterministic forecasting pipeline. "
    "It is not final real short-term waterlogging forecast."
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read prediction config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Prediction config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "prediction" not in data:
        raise ValueError("Prediction config must contain a top-level 'prediction' field.")
    return data["prediction"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def get_history_depth(records: list[dict[str, Any]], minutes_before_now: int) -> float:
    for record in records:
        if int(record["minutes_before_now"]) == int(minutes_before_now):
            return float(record["mean_depth_cm"])
    raise ValueError(f"depth_history.records missing {minutes_before_now} min history point")


def warning_level(depth_cm: float, thresholds: dict[str, float]) -> str:
    blue = float(thresholds["blue"])
    yellow = float(thresholds["yellow"])
    orange = float(thresholds["orange"])
    if depth_cm > orange:
        return "orange"
    if depth_cm >= yellow:
        return "yellow"
    if depth_cm >= blue:
        return "blue"
    return "none"


def time_to_threshold(current_depth_cm: float, k_forecast: float, threshold_cm: float) -> float | None:
    if current_depth_cm >= threshold_cm:
        return 0.0
    if k_forecast <= 0.0:
        return None
    return float((threshold_cm - current_depth_cm) / k_forecast)


def deterministic_forecast(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    input_config = config["input"]
    output_config = config["output"]
    history_config = config["depth_history"]
    slope_windows = config["slope_windows"]
    thresholds = config["warning_thresholds_cm"]
    constraints = config.get("constraints", {})
    warnings: list[str] = []

    area_volume_path = resolve_project_path(root, input_config["area_volume_result_json"])
    weather_path = resolve_project_path(root, input_config["weather_correction_result_json"])
    area_volume = load_json(area_volume_path)
    weather = load_json(weather_path)

    current_mean_depth_cm = float(area_volume["mean_depth_cm"])
    current_max_depth_cm = float(area_volume["max_depth_cm"])
    current_water_area_m2 = float(area_volume["water_area_m2"])
    current_water_volume_m3 = float(area_volume["water_volume_m3"])
    rainfall_intensity_mm_h = float(weather["current_rainfall_intensity_mm_h"])
    rainfall_level_label = str(weather["rainfall_level_label"])
    weather_correction_factor = float(weather["weather_correction_factor"])

    history_records = list(history_config.get("records", []))
    depth_1min = get_history_depth(history_records, 1)
    depth_5min = get_history_depth(history_records, 5)
    depth_10min = get_history_depth(history_records, 10)

    window_1min = slope_windows["window_1min"]
    window_5min = slope_windows["window_5min"]
    window_10min = slope_windows["window_10min"]
    k_1min = (current_mean_depth_cm - depth_1min) / float(window_1min["minutes"])
    k_5min = (current_mean_depth_cm - depth_5min) / float(window_5min["minutes"])
    k_10min = (current_mean_depth_cm - depth_10min) / float(window_10min["minutes"])
    slope_weights = {
        "window_1min": float(window_1min["weight"]),
        "window_5min": float(window_5min["weight"]),
        "window_10min": float(window_10min["weight"]),
    }
    k_base = (
        slope_weights["window_1min"] * k_1min
        + slope_weights["window_5min"] * k_5min
        + slope_weights["window_10min"] * k_10min
    )
    k_forecast_raw = k_base * weather_correction_factor
    min_slope = constraints.get("min_slope_cm_per_min")
    max_slope = constraints.get("max_slope_cm_per_min")
    k_forecast = k_forecast_raw
    if min_slope is not None and k_forecast < float(min_slope):
        warnings.append(
            f"k_forecast clipped from {k_forecast:.4f} to min_slope_cm_per_min={float(min_slope):.4f}"
        )
        k_forecast = float(min_slope)
    if max_slope is not None and k_forecast > float(max_slope):
        warnings.append(
            f"k_forecast clipped from {k_forecast:.4f} to max_slope_cm_per_min={float(max_slope):.4f}"
        )
        k_forecast = float(max_slope)

    max_reasonable_depth = float(constraints.get("max_reasonable_depth_cm", 100.0))
    forecast_horizons = [int(value) for value in config["forecast_horizons_min"]]
    forecast_results: list[dict[str, Any]] = []
    for horizon_min in forecast_horizons:
        raw_depth = current_mean_depth_cm + k_forecast * horizon_min
        clipped = raw_depth > max_reasonable_depth
        forecast_depth = min(raw_depth, max_reasonable_depth)
        if clipped:
            warnings.append(
                f"forecast depth at {horizon_min} min clipped from {raw_depth:.2f} "
                f"to max_reasonable_depth_cm={max_reasonable_depth:.2f}"
            )
        forecast_results.append(
            {
                "horizon_min": horizon_min,
                "forecast_depth_cm": float(forecast_depth),
                "raw_forecast_depth_cm": float(raw_depth),
                "warning_level": warning_level(float(forecast_depth), thresholds),
                "clipped": bool(clipped),
            }
        )

    time_to_thresholds = {
        "blue": time_to_threshold(current_mean_depth_cm, k_forecast, float(thresholds["blue"])),
        "yellow": time_to_threshold(current_mean_depth_cm, k_forecast, float(thresholds["yellow"])),
        "orange": time_to_threshold(current_mean_depth_cm, k_forecast, float(thresholds["orange"])),
    }

    reasoning_dir = resolve_project_path(root, output_config["reasoning_dir"])
    json_dir = resolve_project_path(root, output_config["json_dir"])
    figure_dir = resolve_project_path(root, output_config["figure_dir"])
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_output_json = reasoning_dir / "deterministic_forecast_result.json"
    output_json = json_dir / "deterministic_forecast_result.json"

    result = {
        "stage": "S7A_deterministic_rule_engine",
        "prediction_mode": config.get("mode", "offline_mock_depth_history"),
        "source_area_volume_result_json": str(area_volume_path),
        "source_weather_correction_result_json": str(weather_path),
        "current_mean_depth_cm": current_mean_depth_cm,
        "current_max_depth_cm": current_max_depth_cm,
        "current_water_area_m2": current_water_area_m2,
        "current_water_volume_m3": current_water_volume_m3,
        "rainfall_intensity_mm_h": rainfall_intensity_mm_h,
        "rainfall_level_label": rainfall_level_label,
        "weather_correction_factor": weather_correction_factor,
        "depth_history_source": "offline_mock_depth_history",
        "depth_history_records": history_records,
        "k_1min_cm_per_min": float(k_1min),
        "k_5min_cm_per_min": float(k_5min),
        "k_10min_cm_per_min": float(k_10min),
        "k_base_cm_per_min": float(k_base),
        "k_forecast_cm_per_min": float(k_forecast),
        "k_forecast_raw_cm_per_min": float(k_forecast_raw),
        "slope_weights": slope_weights,
        "forecast_horizons_min": forecast_horizons,
        "forecast_results": forecast_results,
        "time_to_thresholds_min": time_to_thresholds,
        "warning_thresholds_cm": thresholds,
        "mvp_note": MVP_NOTE,
        "config_note": config.get("note"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": warnings,
        "output_files": {
            "data_result_json": str(data_output_json),
            "output_result_json": str(output_json),
        },
    }

    for path in (data_output_json, output_json):
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"[S7-A][forecast] current mean depth cm: {current_mean_depth_cm:.2f}")
    print(f"[S7-A][forecast] k_1min cm/min: {k_1min:.4f}")
    print(f"[S7-A][forecast] k_5min cm/min: {k_5min:.4f}")
    print(f"[S7-A][forecast] k_10min cm/min: {k_10min:.4f}")
    print(f"[S7-A][forecast] k_base cm/min: {k_base:.4f}")
    print(f"[S7-A][forecast] weather correction factor: {weather_correction_factor:.2f}")
    print(f"[S7-A][forecast] k_forecast cm/min: {k_forecast:.4f}")
    for item in forecast_results:
        print(
            "[S7-A][forecast] "
            f"{item['horizon_min']} min depth cm: {item['forecast_depth_cm']:.2f}, "
            f"warning level: {item['warning_level']}"
        )
    print(
        "[S7-A][forecast] time to blue/yellow/orange threshold min: "
        f"{time_to_thresholds['blue']} / {time_to_thresholds['yellow']} / {time_to_thresholds['orange']}"
    )
    print("[S7-A][forecast] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S7-A deterministic rule-engine forecast MVP.")
    parser.add_argument("--config", required=True, help="Path to configs/prediction_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    deterministic_forecast(args.config, args.project_root)


if __name__ == "__main__":
    main()
