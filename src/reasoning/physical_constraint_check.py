#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-C: Physical constraint check using simplified water balance."""

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
    "S7-C uses simplified MVP water balance to validate physical constraint checking. "
    "It is not a full hydrodynamic model or final real forecast."
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read physical constraint config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Physical constraint config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "physical_constraint" not in data:
        raise ValueError("Config must contain a top-level 'physical_constraint' field.")
    return data["physical_constraint"]


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


def warning_level(depth_cm: float, thresholds: dict[str, Any]) -> str:
    if depth_cm >= float(thresholds["orange"]):
        return "orange"
    if depth_cm >= float(thresholds["yellow"]):
        return "yellow"
    if depth_cm >= float(thresholds["blue"]):
        return "blue"
    return "none"


def physical_confidence(error_ratio: float, confidence_config: dict[str, Any]) -> str:
    high_limit = float(confidence_config["high_physical_confidence_max_error_ratio"])
    medium_limit = float(confidence_config["medium_physical_confidence_max_error_ratio"])
    if error_ratio <= high_limit:
        return "high"
    if error_ratio <= medium_limit:
        return "medium"
    return "low"


def rainfall_for_horizon(weather: dict[str, Any], horizon_min: int) -> float:
    if horizon_min == 15:
        return float(weather["forecast_rainfall_15min_mm"])
    if horizon_min == 30:
        return float(weather["forecast_rainfall_30min_mm"])
    if horizon_min == 60:
        return float(weather["forecast_rainfall_60min_mm"])
    if horizon_min == 5:
        return float(weather["current_rainfall_intensity_mm_h"]) * 5.0 / 60.0
    return float(weather["current_rainfall_intensity_mm_h"]) * float(horizon_min) / 60.0


def get_corrected_depth_by_horizon(corrected_forecast: dict[str, Any], horizon_min: int) -> float:
    for item in corrected_forecast["corrected_forecast_results"]:
        if int(item["horizon_min"]) == int(horizon_min):
            return float(item["corrected_forecast_depth_cm"])
    raise ValueError(f"corrected_forecast_results missing horizon {horizon_min} min")


def overall_warning_level(results: list[dict[str, Any]]) -> str:
    rank = {"none": 0, "blue": 1, "yellow": 2, "orange": 3}
    levels = [str(item["warning_level_after_physical_check"]) for item in results]
    if not levels:
        return "none"
    return max(levels, key=lambda level: rank.get(level, 0))


def confidence_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in results:
        counts[str(item["physical_confidence"])] += 1
    if counts["low"] > 0:
        overall = "low"
    elif counts["medium"] > 0:
        overall = "medium"
    else:
        overall = "high"
    return {"overall_physical_confidence": overall, "counts": counts}


def physical_constraint_check(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    input_config = config["input"]
    output_config = config["output"]
    balance = config["water_balance"]
    thresholds = config["warning_thresholds_cm"]
    confidence_config = config["confidence"]

    corrected_path = resolve_project_path(root, input_config["corrected_forecast_result_json"])
    area_volume_path = resolve_project_path(root, input_config["area_volume_result_json"])
    weather_path = resolve_project_path(root, input_config["weather_correction_result_json"])
    corrected_forecast = load_json(corrected_path)
    area_volume = load_json(area_volume_path)
    weather = load_json(weather_path)

    current_mean_depth_cm = float(area_volume["mean_depth_cm"])
    current_max_depth_cm = float(area_volume["max_depth_cm"])
    water_area_m2 = float(area_volume["water_area_m2"])
    current_water_volume_m3 = float(area_volume["water_volume_m3"])
    if current_mean_depth_cm <= 0:
        raise ValueError("current_mean_depth_cm must be positive for linear_volume_depth_proxy")
    volume_per_cm = current_water_volume_m3 / current_mean_depth_cm
    if volume_per_cm <= 0:
        raise ValueError("volume_per_cm must be positive for physical constraint check")

    catchment_area_m2 = float(balance["catchment_area_m2"])
    drainage_rate = float(balance["drainage_rate_m3_per_min"])
    infiltration_rate = float(balance["infiltration_loss_rate_m3_per_min"])
    tolerance_ratio = float(balance["tolerance_ratio"])
    max_callback_ratio = float(balance["max_callback_ratio"])
    callback_ratio = min(max_callback_ratio, 0.5)
    horizons = [int(value) for value in config["horizons_min"]]

    physical_results: list[dict[str, Any]] = []
    for horizon_min in horizons:
        corrected_depth_cm = get_corrected_depth_by_horizon(corrected_forecast, horizon_min)
        rainfall_mm = rainfall_for_horizon(weather, horizon_min)
        rain_input_m3 = rainfall_mm / 1000.0 * catchment_area_m2
        drainage_output_m3 = drainage_rate * horizon_min
        infiltration_loss_m3 = infiltration_rate * horizon_min
        expected_volume_m3 = max(
            current_water_volume_m3 + rain_input_m3 - drainage_output_m3 - infiltration_loss_m3,
            0.0,
        )
        forecast_volume_m3 = corrected_depth_cm * volume_per_cm
        error_m3 = forecast_volume_m3 - expected_volume_m3
        error_ratio = abs(error_m3) / max(expected_volume_m3, 1e-6)
        check = "pass" if error_ratio <= tolerance_ratio else "adjusted"
        if check == "pass":
            adjusted_depth_cm = corrected_depth_cm
        else:
            expected_depth_cm = expected_volume_m3 / volume_per_cm
            adjusted_depth_cm = corrected_depth_cm + callback_ratio * (expected_depth_cm - corrected_depth_cm)
            adjusted_depth_cm = min(max(adjusted_depth_cm, 0.0), 100.0)
        confidence = physical_confidence(error_ratio, confidence_config)
        physical_results.append(
            {
                "horizon_min": horizon_min,
                "corrected_depth_cm": float(corrected_depth_cm),
                "rainfall_mm": float(rainfall_mm),
                "rain_input_m3": float(rain_input_m3),
                "drainage_output_m3": float(drainage_output_m3),
                "infiltration_loss_m3": float(infiltration_loss_m3),
                "expected_volume_m3": float(expected_volume_m3),
                "forecast_volume_m3": float(forecast_volume_m3),
                "error_m3": float(error_m3),
                "error_ratio": float(error_ratio),
                "physical_check": check,
                "physical_confidence": confidence,
                "adjusted_depth_cm": float(adjusted_depth_cm),
                "warning_level_after_physical_check": warning_level(float(adjusted_depth_cm), thresholds),
            }
        )

    reasoning_dir = resolve_project_path(root, output_config["reasoning_dir"])
    json_dir = resolve_project_path(root, output_config["json_dir"])
    figure_dir = resolve_project_path(root, output_config["figure_dir"])
    for directory in (reasoning_dir, json_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_physical_json = reasoning_dir / "physical_constraint_result.json"
    output_physical_json = json_dir / "physical_constraint_result.json"
    data_final_json = reasoning_dir / "final_forecast_result.json"
    output_final_json = json_dir / "final_forecast_result.json"
    timestamp = datetime.now(timezone.utc).isoformat()

    physical_result = {
        "stage": "S7C_physical_constraint_check",
        "physical_constraint_mode": config.get("mode", "simplified_water_balance_mvp"),
        "source_corrected_forecast_json": str(corrected_path),
        "source_area_volume_json": str(area_volume_path),
        "source_weather_correction_json": str(weather_path),
        "current_mean_depth_cm": current_mean_depth_cm,
        "current_max_depth_cm": current_max_depth_cm,
        "current_water_volume_m3": current_water_volume_m3,
        "water_area_m2": water_area_m2,
        "catchment_area_m2": catchment_area_m2,
        "drainage_rate_m3_per_min": drainage_rate,
        "infiltration_loss_rate_m3_per_min": infiltration_rate,
        "tolerance_ratio": tolerance_ratio,
        "volume_per_cm": float(volume_per_cm),
        "physical_check_results": physical_results,
        "rainfall_level_label": weather.get("rainfall_level_label"),
        "weather_correction_factor": weather.get("weather_correction_factor"),
        "mvp_note": MVP_NOTE,
        "config_note": config.get("mvp_note"),
        "timestamp": timestamp,
        "output_files": {
            "data_physical_constraint_result": str(data_physical_json),
            "output_physical_constraint_result": str(output_physical_json),
            "data_final_forecast_result": str(data_final_json),
            "output_final_forecast_result": str(output_final_json),
        },
    }

    final_forecast_results = [
        {
            "horizon_min": item["horizon_min"],
            "final_forecast_depth_cm": item["adjusted_depth_cm"],
            "source_corrected_depth_cm": item["corrected_depth_cm"],
            "physical_check": item["physical_check"],
            "physical_confidence": item["physical_confidence"],
            "warning_level": item["warning_level_after_physical_check"],
        }
        for item in physical_results
    ]
    final_result = {
        "stage": "S7_final_forecast_after_physical_constraint",
        "source_corrected_forecast_json": str(corrected_path),
        "source_physical_constraint_json": str(output_physical_json),
        "final_forecast_results": final_forecast_results,
        "overall_warning_level": overall_warning_level(physical_results),
        "physical_confidence_summary": confidence_summary(physical_results),
        "mvp_note": MVP_NOTE,
        "timestamp": timestamp,
        "output_files": {
            "data_final_forecast_result": str(data_final_json),
            "output_final_forecast_result": str(output_final_json),
            "physical_constraint_figure": str(figure_dir / "physical_constraint_summary.png"),
        },
    }

    for path, payload in (
        (data_physical_json, physical_result),
        (output_physical_json, physical_result),
        (data_final_json, final_result),
        (output_final_json, final_result),
    ):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print("[S7-C][physical] physical check results:")
    for item in physical_results:
        print(
            "  - "
            f"{item['horizon_min']} min: corrected={item['corrected_depth_cm']:.2f} cm, "
            f"adjusted={item['adjusted_depth_cm']:.2f} cm, "
            f"expected_volume={item['expected_volume_m3']:.4f} m3, "
            f"forecast_volume={item['forecast_volume_m3']:.4f} m3, "
            f"error_ratio={item['error_ratio']:.4f}, "
            f"check={item['physical_check']}, "
            f"confidence={item['physical_confidence']}, "
            f"warning={item['warning_level_after_physical_check']}"
        )
    print(f"[S7-C][physical] overall warning level: {final_result['overall_warning_level']}")
    print("[S7-C][physical] output files:")
    for path in physical_result["output_files"].values():
        print(f"  - {path}")
    return final_result


def main() -> None:
    parser = argparse.ArgumentParser(description="S7-C simplified physical constraint check.")
    parser.add_argument("--config", required=True, help="Path to configs/physical_constraint_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    physical_constraint_check(args.config, args.project_root)


if __name__ == "__main__":
    main()
