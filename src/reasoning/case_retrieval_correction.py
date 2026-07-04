#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-B: Case retrieval correction MVP using offline mock cases."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MVP_NOTE = (
    "S7-B uses offline_mock_case_library to validate case retrieval correction. "
    "It is not final real historical-case correction."
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read case retrieval config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Case retrieval config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "case_retrieval" not in data:
        raise ValueError("Config must contain a top-level 'case_retrieval' field.")
    return data["case_retrieval"]


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


def build_current_feature_vector(
    area_volume: dict[str, Any],
    weather: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, float]:
    return {
        "rainfall_intensity_mm_h": float(weather["current_rainfall_intensity_mm_h"]),
        "forecast_rainfall_15min_mm": float(weather["forecast_rainfall_15min_mm"]),
        "forecast_rainfall_30min_mm": float(weather["forecast_rainfall_30min_mm"]),
        "forecast_rainfall_60min_mm": float(weather["forecast_rainfall_60min_mm"]),
        "current_mean_depth_cm": float(area_volume["mean_depth_cm"]),
        "current_max_depth_cm": float(area_volume["max_depth_cm"]),
        "water_area_m2": float(area_volume["water_area_m2"]),
        "water_volume_m3": float(area_volume["water_volume_m3"]),
        "k_forecast_cm_per_min": float(deterministic["k_forecast_cm_per_min"]),
    }


def weighted_euclidean(
    current_features: dict[str, float],
    case_features: dict[str, Any],
    feature_weights: dict[str, Any],
) -> float:
    distance_sq = 0.0
    for name, current_value in current_features.items():
        case_value = float(case_features[name])
        weight = float(feature_weights.get(name, 1.0))
        distance_sq += weight * (current_value - case_value) ** 2
    return math.sqrt(distance_sq)


def retrieve_cases(
    current_features: dict[str, float],
    cases: list[dict[str, Any]],
    feature_weights: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        distance = weighted_euclidean(current_features, case["features"], feature_weights)
        similarity = 1.0 / (1.0 + distance)
        scored.append(
            {
                "case_id": case["case_id"],
                "distance": float(distance),
                "similarity_score": float(similarity),
                "forecast_bias_cm": case["forecast_bias_cm"],
                "features": case["features"],
                "note": case.get("note", ""),
            }
        )
    scored.sort(key=lambda item: item["distance"])
    return scored[: int(top_k)]


def median_bias_by_horizon(top_cases: list[dict[str, Any]], horizons: list[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in horizons:
        key = str(horizon)
        values = [float(case["forecast_bias_cm"][key]) for case in top_cases]
        result[key] = float(statistics.median(values))
    return result


def correct_forecasts(
    deterministic: dict[str, Any],
    median_biases: dict[str, float],
    thresholds: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    min_depth = float(constraints.get("min_depth_cm", 0.0))
    max_depth = float(constraints.get("max_reasonable_depth_cm", 100.0))
    corrected: list[dict[str, Any]] = []
    for item in deterministic["forecast_results"]:
        horizon = int(item["horizon_min"])
        deterministic_depth = float(item["forecast_depth_cm"])
        bias = float(median_biases[str(horizon)])
        raw_corrected = deterministic_depth + bias
        corrected_depth = min(max(raw_corrected, min_depth), max_depth)
        corrected.append(
            {
                "horizon_min": horizon,
                "deterministic_forecast_depth_cm": deterministic_depth,
                "median_case_bias_cm": bias,
                "raw_corrected_forecast_depth_cm": float(raw_corrected),
                "corrected_forecast_depth_cm": float(corrected_depth),
                "warning_level": warning_level(float(corrected_depth), thresholds),
                "clipped": bool(corrected_depth != raw_corrected),
            }
        )
    return corrected


def case_retrieval_correction(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    input_config = config["input"]
    output_config = config["output"]
    retrieval_config = config["retrieval"]
    thresholds = config["warning_thresholds_cm"]
    constraints = config.get("constraints", {})

    area_volume_path = resolve_project_path(root, input_config["area_volume_result_json"])
    weather_path = resolve_project_path(root, input_config["weather_correction_result_json"])
    deterministic_path = resolve_project_path(root, input_config["deterministic_forecast_result_json"])
    case_library_path = resolve_project_path(root, input_config["case_library_json"])
    area_volume = load_json(area_volume_path)
    weather = load_json(weather_path)
    deterministic = load_json(deterministic_path)
    case_library = load_json(case_library_path)

    current_features = build_current_feature_vector(area_volume, weather, deterministic)
    top_cases = retrieve_cases(
        current_features,
        list(case_library["cases"]),
        retrieval_config["feature_weights"],
        int(retrieval_config["top_k"]),
    )
    horizons = [int(value) for value in config["forecast_horizons_min"]]
    median_biases = median_bias_by_horizon(top_cases, horizons)
    corrected_forecast_results = correct_forecasts(deterministic, median_biases, thresholds, constraints)

    reasoning_dir = resolve_project_path(root, output_config["reasoning_dir"])
    json_dir = resolve_project_path(root, output_config["json_dir"])
    figure_dir = resolve_project_path(root, output_config["figure_dir"])
    for directory in (reasoning_dir, json_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_case_json = reasoning_dir / "case_retrieval_result.json"
    output_case_json = json_dir / "case_retrieval_result.json"
    data_corrected_json = reasoning_dir / "corrected_forecast_result.json"
    output_corrected_json = json_dir / "corrected_forecast_result.json"
    timestamp = datetime.now(timezone.utc).isoformat()

    case_result = {
        "stage": "S7B_case_retrieval_correction",
        "mode": config.get("mode", "offline_mock_case_library"),
        "distance_method": retrieval_config.get("distance_method", "weighted_euclidean"),
        "similarity_method": retrieval_config.get("similarity_method", "1_over_1_plus_distance"),
        "top_k": int(retrieval_config["top_k"]),
        "source_area_volume_result_json": str(area_volume_path),
        "source_weather_correction_result_json": str(weather_path),
        "source_deterministic_forecast_result_json": str(deterministic_path),
        "source_case_library_json": str(case_library_path),
        "current_event_features": current_features,
        "feature_weights": retrieval_config["feature_weights"],
        "retrieved_cases": top_cases,
        "median_bias_cm_by_horizon": median_biases,
        "mvp_note": MVP_NOTE,
        "config_note": config.get("note"),
        "timestamp": timestamp,
        "output_files": {
            "data_case_retrieval_result": str(data_case_json),
            "output_case_retrieval_result": str(output_case_json),
            "data_corrected_forecast_result": str(data_corrected_json),
            "output_corrected_forecast_result": str(output_corrected_json),
        },
    }

    corrected_result = {
        "stage": "S7B_corrected_forecast",
        "prediction_mode": "offline_mock_case_retrieval_correction",
        "source_deterministic_forecast_result_json": str(deterministic_path),
        "source_case_retrieval_result_json": str(output_case_json),
        "current_mean_depth_cm": float(deterministic["current_mean_depth_cm"]),
        "current_max_depth_cm": float(deterministic["current_max_depth_cm"]),
        "k_forecast_cm_per_min": float(deterministic["k_forecast_cm_per_min"]),
        "forecast_horizons_min": horizons,
        "deterministic_forecast_results": deterministic["forecast_results"],
        "corrected_forecast_results": corrected_forecast_results,
        "median_bias_cm_by_horizon": median_biases,
        "top_case_ids": [case["case_id"] for case in top_cases],
        "top_case_similarity_scores": [case["similarity_score"] for case in top_cases],
        "warning_thresholds_cm": thresholds,
        "mvp_note": MVP_NOTE,
        "timestamp": timestamp,
        "output_files": {
            "data_case_retrieval_result": str(data_case_json),
            "output_case_retrieval_result": str(output_case_json),
            "data_corrected_forecast_result": str(data_corrected_json),
            "output_corrected_forecast_result": str(output_corrected_json),
            "case_retrieval_figure": str(figure_dir / "case_retrieval_correction.png"),
        },
    }

    for path, payload in (
        (data_case_json, case_result),
        (output_case_json, case_result),
        (data_corrected_json, corrected_result),
        (output_corrected_json, corrected_result),
    ):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print("[S7-B][case] top_k retrieved case ids:")
    for case in top_cases:
        print(f"  - {case['case_id']} similarity={case['similarity_score']:.4f} distance={case['distance']:.4f}")
    print("[S7-B][case] median bias by horizon cm:")
    for horizon in horizons:
        print(f"  - {horizon} min: {median_biases[str(horizon)]:.2f}")
    print("[S7-B][case] deterministic vs corrected forecast:")
    for item in corrected_forecast_results:
        print(
            "  - "
            f"{item['horizon_min']} min: deterministic={item['deterministic_forecast_depth_cm']:.2f} cm, "
            f"corrected={item['corrected_forecast_depth_cm']:.2f} cm, "
            f"warning={item['warning_level']}"
        )
    print("[S7-B][case] output files:")
    for path in case_result["output_files"].values():
        print(f"  - {path}")
    return corrected_result


def main() -> None:
    parser = argparse.ArgumentParser(description="S7-B offline mock case retrieval correction.")
    parser.add_argument("--config", required=True, help="Path to configs/case_retrieval_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    case_retrieval_correction(args.config, args.project_root)


if __name__ == "__main__":
    main()
