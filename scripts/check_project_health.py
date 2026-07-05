#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check reproducibility inputs and outputs for the offline MVP demo."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DIRECTORIES = [
    "configs",
    "data/dem",
    "data/masks",
    "data/fusion",
    "data/hydrology",
    "data/meteorology",
    "data/reasoning",
    "data/warnings",
    "data/audit_logs",
    "data/db",
    "outputs/json",
    "outputs/figures",
    "outputs/reports",
    "src/agent",
    "src/database",
    "scripts",
]


CONFIG_FILES = [
    "configs/roi_mapping.yaml",
    "configs/weather_config.yaml",
    "configs/prediction_config.yaml",
    "configs/case_retrieval_config.yaml",
    "configs/physical_constraint_config.yaml",
    "configs/warning_config.yaml",
    "configs/agent_config.yaml",
]


PYTHON_SCRIPTS = [
    "run_offline_pipeline.py",
    "src/agent/pipeline_agent.py",
    "src/database/show_audit_db.py",
    "src/hydrology/calculate_area_volume.py",
    "src/meteorology/compute_weather_correction.py",
    "src/reasoning/deterministic_forecast.py",
    "src/reasoning/case_retrieval_correction.py",
    "src/reasoning/physical_constraint_check.py",
    "src/warning/generate_warning_decision.py",
    "src/warning/generate_warning_report.py",
    "scripts/check_project_health.py",
]


TRACKED_INPUTS = [
    "requirements.txt",
    "DATA_MANIFEST.md",
    "PROJECT_STATUS.md",
    "data/dem/ground_dem_metadata.json",
    "data/masks/manual_water_mask_metadata.json",
    "data/cases/mock_historical_cases.json",
]


IGNORED_LOCAL_INPUTS = [
    "data/dem/ground_dem.npy",
    "data/dem/ground_dem_interpolated.npy",
    "data/dem/ground_dem_valid_mask.npy",
    "data/fusion/water_region_mask.npy",
    "data/masks/manual_water_mask.npy",
    "data/hydrology/water_depth_map.npy",
    "data/hydrology/water_depth_valid_mask.npy",
]


EXPECTED_OUTPUTS = [
    "outputs/json/agent_run_summary.json",
    "outputs/json/final_forecast_result.json",
    "outputs/json/warning_decision_result.json",
    "outputs/reports/warning_report.md",
    "outputs/figures/warning_summary.png",
]


def check_path(
    relative_path: str,
    category: str,
    required: bool,
    expected_type: str = "file",
    ignored_large_file: bool = False,
) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if expected_type == "directory":
        exists = path.is_dir()
    else:
        exists = path.is_file()

    if exists and ignored_large_file:
        status = "ignored_large_file"
    elif exists:
        status = "ok"
    elif required:
        status = "missing_required"
    else:
        status = "missing_optional"

    return {
        "category": category,
        "path": relative_path,
        "expected_type": expected_type,
        "required": required,
        "exists": exists,
        "status": status,
        "note": (
            "Ignored by .gitignore but required locally for this offline demo."
            if ignored_large_file
            else ""
        ),
    }


def run_health_check() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for path in DIRECTORIES:
        checks.append(check_path(path, "directory", required=True, expected_type="directory"))
    for path in CONFIG_FILES:
        checks.append(check_path(path, "config", required=True))
    for path in PYTHON_SCRIPTS:
        checks.append(check_path(path, "python_script", required=True))
    for path in TRACKED_INPUTS:
        checks.append(check_path(path, "tracked_input", required=True))
    for path in IGNORED_LOCAL_INPUTS:
        checks.append(
            check_path(
                path,
                "ignored_local_input",
                required=True,
                ignored_large_file=True,
            )
        )
    for path in EXPECTED_OUTPUTS:
        checks.append(check_path(path, "expected_output", required=False))

    counts = {
        "total_checks": len(checks),
        "ok": sum(1 for item in checks if item["status"] == "ok"),
        "missing_required": sum(1 for item in checks if item["status"] == "missing_required"),
        "missing_optional": sum(1 for item in checks if item["status"] == "missing_optional"),
        "ignored_large_file": sum(1 for item in checks if item["status"] == "ignored_large_file"),
    }

    if counts["missing_required"] > 0:
        health_status = "failed"
    elif counts["missing_optional"] > 0:
        health_status = "warning"
    else:
        health_status = "pass"

    result = {
        "stage": "project_reproducibility_health_check",
        "project_root": str(PROJECT_ROOT),
        "health_status": health_status,
        "counts": counts,
        "checks": checks,
        "mvp_note": (
            "Current reproducibility check is for the offline MVP simulation. "
            "It does not start real-time devices, ROS nodes, or rosbag replay."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_path = PROJECT_ROOT / "outputs/json/health_check_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return result


def main() -> None:
    result = run_health_check()
    counts = result["counts"]
    print("[health] project root:", result["project_root"])
    print("[health] total checks:", counts["total_checks"])
    print("[health] ok count:", counts["ok"])
    print("[health] ignored large/local-required count:", counts["ignored_large_file"])
    print("[health] missing required count:", counts["missing_required"])
    print("[health] missing optional count:", counts["missing_optional"])
    print("[health] health status:", result["health_status"])
    print("[health] output: outputs/json/health_check_result.json")
    if counts["missing_required"]:
        print("[health] missing required paths:")
        for item in result["checks"]:
            if item["status"] == "missing_required":
                print("  -", item["path"])


if __name__ == "__main__":
    main()
