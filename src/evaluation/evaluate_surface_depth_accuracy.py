#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate S4-real surface-difference depth against known test depths."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dem.build_surface_dem_from_rosbag import get_case_config, load_yaml, resolve_project_path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _sub(value: float | None, known: float | None) -> float | None:
    if value is None or known is None:
        return None
    return float(value - known)


def _abs(value: float | None) -> float | None:
    return None if value is None else abs(float(value))


def _relative(error: float | None, known: float | None) -> float | None:
    if error is None or known is None or float(known) == 0.0:
        return None
    return float(error / known * 100.0)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _has_warning(warnings: list[Any], prefix: str) -> bool:
    return any(str(item).startswith(prefix) for item in warnings)


def write_report(report_path: Path, json_dir: Path) -> None:
    rows = []
    for path in sorted(json_dir.glob("surface_depth_accuracy_*.json")):
        try:
            rows.append(load_json(path))
        except Exception:
            continue

    lines = [
        "# Surface DEM Depth Accuracy Report",
        "",
        "This report evaluates offline LiDAR surface DEM difference depth for the dormitory simulation scenes.",
        "",
        "The current evaluation is based on dormitory simulated 13cm / 39cm water-depth scenes. "
        "It does not represent final engineering accuracy. It is mainly used to validate the upgrade "
        "from configured_depth to offline LiDAR surface difference.",
        "",
        "| Case | Known depth cm | Mean cm | Median cm | Max cm | Mean error cm | Median error cm | Valid ratio | Coverage | Accuracy | Data quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in rows:
        warnings = item.get("warning") or []
        coverage_warning = _has_warning(warnings, "coverage_warning")
        accuracy_warning = _has_warning(warnings, "accuracy_warning")
        quality = "low_data_quality" if any("low_data_quality" in str(w) for w in warnings) else "ok"
        lines.append(
            "| "
            f"{item.get('case_name')} | "
            f"{_fmt(item.get('known_depth_cm'))} | "
            f"{_fmt(item.get('mean_depth_cm'))} | "
            f"{_fmt(item.get('median_depth_cm'))} | "
            f"{_fmt(item.get('max_depth_cm'))} | "
            f"{_fmt(item.get('mean_error_cm'))} | "
            f"{_fmt(item.get('median_error_cm'))} | "
            f"{float(item.get('valid_depth_ratio_in_water_region', 0.0)):.4f} | "
            f"{'coverage_warning' if coverage_warning else 'ok'} | "
            f"{'accuracy_warning' if accuracy_warning else 'ok'} | "
            f"{quality} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Low valid depth ratio indicates low data quality and should be interpreted cautiously.",
            "- The current method does not use `configured_depth`.",
            "- Accuracy depends on calibration, ROI mapping, point density, and LiDAR returns from the water/surface target.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_surface_depth_accuracy(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", selected_case))
    known_depth_cm = _safe_float(case_config.get("known_depth_cm"))

    json_dir = resolve_project_path(root, config["output"]["json_dir"])
    hydrology_dir = resolve_project_path(root, config["output"]["hydrology_dir"]) / output_name
    report_dir = resolve_project_path(root, config["output"]["report_dir"])
    result_json_path = json_dir / f"surface_water_depth_result_{output_name}.json"
    depth_map_path = hydrology_dir / "surface_water_depth_map.npy"
    valid_mask_path = hydrology_dir / "surface_water_depth_valid_mask.npy"

    depth_result = load_json(result_json_path)
    depth_map = np.load(depth_map_path)
    valid_mask = np.load(valid_mask_path).astype(bool)
    values_cm = depth_map[valid_mask].astype(np.float64) * 100.0

    warnings = list(depth_result.get("warning") or [])
    if values_cm.size == 0:
        mean_depth_cm = median_depth_cm = max_depth_cm = min_depth_cm = None
        warnings.append("low_data_quality: no valid depth cells for accuracy evaluation")
    else:
        mean_depth_cm = float(np.mean(values_cm))
        median_depth_cm = float(np.median(values_cm))
        max_depth_cm = float(np.max(values_cm))
        min_depth_cm = float(np.min(values_cm))

    mean_error_cm = _sub(mean_depth_cm, known_depth_cm)
    median_error_cm = _sub(median_depth_cm, known_depth_cm)
    valid_depth_ratio = float(depth_result.get("valid_depth_ratio_in_water_region", 0.0))
    if valid_depth_ratio < 0.30 and not _has_warning(warnings, "coverage_warning"):
        warnings.append("coverage_warning: valid_depth_ratio_in_water_region is below 0.30")
    if mean_error_cm is not None and abs(mean_error_cm) > 10.0 and not _has_warning(warnings, "accuracy_warning"):
        warnings.append("accuracy_warning: abs(mean_error_cm) is above 10cm")
    output_json = json_dir / f"surface_depth_accuracy_{output_name}.json"
    report_path = report_dir / "surface_depth_accuracy_report.md"

    result = {
        "stage": "S4_real_surface_depth_accuracy_evaluation",
        "case_name": output_name,
        "source_depth_result_json": str(result_json_path),
        "source_depth_map": str(depth_map_path),
        "source_depth_valid_mask": str(valid_mask_path),
        "mean_depth_cm": mean_depth_cm,
        "median_depth_cm": median_depth_cm,
        "max_depth_cm": max_depth_cm,
        "min_depth_cm": min_depth_cm,
        "known_depth_cm": known_depth_cm,
        "mean_error_cm": mean_error_cm,
        "median_error_cm": median_error_cm,
        "abs_mean_error_cm": _abs(mean_error_cm),
        "abs_median_error_cm": _abs(median_error_cm),
        "relative_mean_error_percent": _relative(mean_error_cm, known_depth_cm),
        "valid_depth_cell_count": int(depth_result.get("valid_depth_cell_count", int(np.count_nonzero(valid_mask)))),
        "valid_depth_ratio_in_water_region": valid_depth_ratio,
        "warning": warnings,
        "note": (
            "Evaluation is based on dormitory simulated known-depth scenes. "
            "It does not represent final engineering accuracy."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "surface_depth_accuracy_json": str(output_json),
            "surface_depth_accuracy_report": str(report_path),
        },
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_report(report_path, json_dir)

    print(f"[surface_eval] case_name: {output_name}")
    print(f"[surface_eval] known_depth_cm: {_fmt(known_depth_cm)}")
    print(f"[surface_eval] mean_depth_cm: {_fmt(mean_depth_cm)}")
    print(f"[surface_eval] median_depth_cm: {_fmt(median_depth_cm)}")
    print(f"[surface_eval] max_depth_cm: {_fmt(max_depth_cm)}")
    print(f"[surface_eval] mean_error_cm: {_fmt(mean_error_cm)}")
    print(f"[surface_eval] median_error_cm: {_fmt(median_error_cm)}")
    print(f"[surface_eval] valid_depth_cell_count: {result['valid_depth_cell_count']}")
    print(f"[surface_eval] valid_depth_ratio_in_water_region: {result['valid_depth_ratio_in_water_region']:.4f}")
    if warnings:
        print(f"[surface_eval][WARN] {'; '.join(warnings)}")
    print("[surface_eval] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S4-real surface DEM depth accuracy.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--case", help="Case name, e.g. water_sim_13cm_001")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    evaluate_surface_depth_accuracy(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
