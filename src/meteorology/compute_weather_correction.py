#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6: Compute a weather correction factor from configured or live data."""

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

from src.meteorology.open_meteo_provider import (  # noqa: E402
    WeatherProviderError,
    fetch_open_meteo_weather,
)


def load_weather_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read weather config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Weather config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "weather" not in data:
        raise ValueError("Weather config must contain a top-level 'weather' field.")
    return data["weather"]


def classify_rainfall(rainfall_intensity_mm_h: float) -> tuple[str, float]:
    if rainfall_intensity_mm_h <= 0.0:
        return "no_rain", 0.7
    if rainfall_intensity_mm_h < 15.0:
        return "light_rain", 1.0
    if rainfall_intensity_mm_h < 30.0:
        return "moderate_rain", 1.3
    return "heavy_rain", 1.8


def _fallback_weather_values(weather: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = weather.get("fallback_current", weather.get("current", {}))
    forecast = weather.get("fallback_forecast", weather.get("forecast", {}))
    return current, forecast


def compute_weather_correction(
    config_path: str | Path,
    project_root: str | Path,
    weather_fetcher=fetch_open_meteo_weather,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    weather = load_weather_config(config_path)
    provider = weather.get("provider", "offline_mock")
    api_status = "not_requested"
    api_error = None
    provider_metadata: dict[str, Any] = {}
    if provider == "open_meteo":
        try:
            live = weather_fetcher(weather)
            current = live["current"]
            forecast = live["forecast"]
            provider_metadata = live["provider_metadata"]
            api_status = "success"
        except WeatherProviderError as exc:
            if weather.get("api", {}).get("fallback_on_error") is not True:
                raise
            current, forecast = _fallback_weather_values(weather)
            api_status = "fallback"
            api_error = str(exc)
    else:
        current, forecast = _fallback_weather_values(weather)

    rainfall_intensity = float(current.get("rainfall_intensity_mm_h", 0.0))
    rainfall_level_label, correction_factor = classify_rainfall(rainfall_intensity)
    forecast_15min = float(forecast.get("rainfall_15min_mm", 0.0))
    forecast_30min = float(forecast.get("rainfall_30min_mm", 0.0))
    forecast_60min = float(forecast.get("rainfall_60min_mm", 0.0))

    meteorology_dir = root / "data" / "meteorology"
    json_dir = root / "outputs" / "json"
    figure_dir = root / "outputs" / "figures"
    meteorology_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data_output_json = meteorology_dir / "weather_correction_result.json"
    output_json = json_dir / "weather_correction_result.json"
    if api_status == "success":
        data_note = (
            "live_weather_api_sandbox: read-only Open-Meteo data; no notification "
            "or real warning action is permitted."
        )
    elif api_status == "fallback":
        data_note = (
            "weather_api_fallback: the live request failed and explicitly configured "
            "fallback values were used."
        )
    else:
        data_note = (
            "offline_configured_weather: S6 uses configured rainfall data and does "
            "not claim a live meteorological observation."
        )

    result = {
        "stage": "S6_weather_correction",
        "provider": provider,
        "data_mode": weather.get("data_mode", "manual_config"),
        "location_name": weather.get("location_name", "unknown"),
        "location": weather.get("location"),
        "observation_time": current.get("observation_time"),
        "current_rainfall_intensity_mm_h": rainfall_intensity,
        "rainfall_level_label": rainfall_level_label,
        "weather_correction_factor": correction_factor,
        "forecast_rainfall_15min_mm": forecast_15min,
        "forecast_rainfall_30min_mm": forecast_30min,
        "forecast_rainfall_60min_mm": forecast_60min,
        "rule_source": "patent_specification_s6",
        "mock_data_note": data_note,
        "weather_data_note": data_note,
        "api_request_attempted": provider == "open_meteo",
        "api_status": api_status,
        "api_error": api_error,
        "provider_metadata": provider_metadata,
        "external_notification_allowed": False,
        "real_warning_allowed": False,
        "config_note": weather.get("note"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "data_result_json": str(data_output_json),
            "output_result_json": str(output_json),
        },
    }

    for path in (data_output_json, output_json):
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"[S6][weather] current rainfall intensity mm/h: {rainfall_intensity:.2f}")
    print(f"[S6][weather] rainfall level label: {rainfall_level_label}")
    print(f"[S6][weather] weather correction factor: {correction_factor:.2f}")
    print(
        "[S6][weather] forecast rainfall 15/30/60 min: "
        f"{forecast_15min:.2f} / {forecast_30min:.2f} / {forecast_60min:.2f} mm"
    )
    print("[S6][weather] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S6 configured/live weather correction.")
    parser.add_argument("--config", required=True, help="Path to configs/weather_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    compute_weather_correction(args.config, args.project_root)


if __name__ == "__main__":
    main()
