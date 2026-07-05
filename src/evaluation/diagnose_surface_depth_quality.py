#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose S4-real surface depth quality without tuning the algorithm."""

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


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _case_names(config: dict[str, Any], requested_cases: list[str] | None) -> list[str]:
    if requested_cases:
        return requested_cases
    return [
        name
        for name, case_config in config.get("cases", {}).items()
        if case_config.get("data_type") != "dry_baseline" and case_config.get("known_depth_cm") is not None
    ]


def diagnose_case(config: dict[str, Any], project_root: Path, case_name: str) -> dict[str, Any]:
    _, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", case_name))
    known_depth_cm = float(case_config["known_depth_cm"]) if case_config.get("known_depth_cm") is not None else None

    surface_dir = resolve_project_path(project_root, config["output"]["surface_dem_dir"]) / output_name
    hydrology_dir = resolve_project_path(project_root, config["output"]["hydrology_dir"]) / output_name
    json_dir = resolve_project_path(project_root, config["output"]["json_dir"])

    surface_dem_path = surface_dir / "surface_dem.npy"
    surface_valid_path = surface_dir / "surface_dem_valid_mask.npy"
    point_count_path = surface_dir / "surface_dem_point_count.npy"
    depth_map_path = hydrology_dir / "surface_water_depth_map.npy"
    depth_valid_path = hydrology_dir / "surface_water_depth_valid_mask.npy"
    depth_result_path = json_dir / f"surface_water_depth_result_{output_name}.json"
    accuracy_path = json_dir / f"surface_depth_accuracy_{output_name}.json"

    surface_dem = np.load(surface_dem_path)
    surface_valid_mask = np.load(surface_valid_path).astype(bool)
    point_count = np.load(point_count_path)
    depth_map = np.load(depth_map_path)
    depth_valid_mask = np.load(depth_valid_path).astype(bool)
    depth_result = load_json(depth_result_path)
    accuracy = load_json(accuracy_path)

    point_values = point_count[surface_valid_mask].astype(np.float64)
    point_stats = _safe_stats(point_values)
    depth_values_cm = depth_map[depth_valid_mask].astype(np.float64) * 100.0

    water_region_cell_count = int(depth_result.get("water_region_cell_count", 0))
    surface_valid_cell_count = int(np.count_nonzero(surface_valid_mask))
    depth_valid_cell_count = int(np.count_nonzero(depth_valid_mask))
    valid_depth_ratio = float(depth_result.get("valid_depth_ratio_in_water_region", 0.0))

    if known_depth_cm is None or depth_values_cm.size == 0:
        outlier_high_count = 0
        outlier_low_count = 0
    else:
        outlier_high_count = int(np.count_nonzero(depth_values_cm > known_depth_cm + 15.0))
        outlier_low_count = int(np.count_nonzero(depth_values_cm < known_depth_cm - 10.0))

    mean_error_cm = accuracy.get("mean_error_cm")
    high_error_warning = bool(mean_error_cm is not None and abs(float(mean_error_cm)) > 10.0)
    low_coverage_warning = bool(valid_depth_ratio < 0.30)
    sparse_point_warning = bool(point_stats["median"] is not None and point_stats["median"] <= 1.0)

    warnings = list(accuracy.get("warning") or [])
    if high_error_warning and not any(str(w).startswith("accuracy_warning") for w in warnings):
        warnings.append("accuracy_warning: abs(mean_error_cm) is above 10cm")
    if low_coverage_warning and not any(str(w).startswith("coverage_warning") for w in warnings):
        warnings.append("coverage_warning: valid_depth_ratio_in_water_region is below 0.30")
    if sparse_point_warning:
        warnings.append("sparse_point_warning: median surface point count per valid cell is <= 1")

    return {
        "case_name": output_name,
        "known_depth_cm": known_depth_cm,
        "source_files": {
            "surface_dem": str(surface_dem_path),
            "surface_dem_valid_mask": str(surface_valid_path),
            "surface_dem_point_count": str(point_count_path),
            "surface_water_depth_map": str(depth_map_path),
            "surface_water_depth_valid_mask": str(depth_valid_path),
            "surface_water_depth_result": str(depth_result_path),
            "surface_depth_accuracy": str(accuracy_path),
        },
        "water_region_cell_count": water_region_cell_count,
        "surface_valid_cell_count": surface_valid_cell_count,
        "depth_valid_cell_count": depth_valid_cell_count,
        "valid_depth_ratio_in_water_region": valid_depth_ratio,
        "point_count_min": point_stats["min"],
        "point_count_mean": point_stats["mean"],
        "point_count_median": point_stats["median"],
        "point_count_max": point_stats["max"],
        "depth_p10": _percentile(depth_values_cm, 10),
        "depth_p25": _percentile(depth_values_cm, 25),
        "depth_p50": _percentile(depth_values_cm, 50),
        "depth_p75": _percentile(depth_values_cm, 75),
        "depth_p90": _percentile(depth_values_cm, 90),
        "mean_depth_cm": accuracy.get("mean_depth_cm"),
        "median_depth_cm": accuracy.get("median_depth_cm"),
        "max_depth_cm": accuracy.get("max_depth_cm"),
        "mean_error_cm": mean_error_cm,
        "median_error_cm": accuracy.get("median_error_cm"),
        "outlier_high_depth_cell_count": outlier_high_count,
        "outlier_low_depth_cell_count": outlier_low_count,
        "high_error_warning": high_error_warning,
        "low_coverage_warning": low_coverage_warning,
        "sparse_point_warning": sparse_point_warning,
        "warning": warnings,
        "_arrays": {
            "depth_values_cm": depth_values_cm,
            "point_values": point_values,
            "depth_map": depth_map,
            "depth_valid_mask": depth_valid_mask,
        },
    }


def cross_case_compare(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(case_results) < 2:
        return {"warning": "cross-case comparison requires at least two cases"}

    first = case_results[0]
    second = case_results[1]
    mask_a = first["_arrays"]["depth_valid_mask"]
    mask_b = second["_arrays"]["depth_valid_mask"]
    depth_a = first["_arrays"]["depth_map"].astype(np.float64) * 100.0
    depth_b = second["_arrays"]["depth_map"].astype(np.float64) * 100.0

    overlap = mask_a & mask_b
    union = mask_a | mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    differences = depth_b[overlap] - depth_a[overlap]
    return {
        "case_a": first["case_name"],
        "case_b": second["case_name"],
        "overlapping_valid_cell_count": int(np.count_nonzero(overlap)),
        "overlap_ratio": float(np.count_nonzero(overlap) / max(1, np.count_nonzero(union))),
        "overlap_ratio_of_water_region": float(
            np.count_nonzero(overlap) / max(1, int(first.get("water_region_cell_count", 0)))
        ),
        "same_cell_depth_difference_mean_cm": None if differences.size == 0 else float(np.mean(differences)),
        "same_cell_depth_difference_median_cm": None if differences.size == 0 else float(np.median(differences)),
        "cells_valid_only_in_13cm": int(np.count_nonzero(only_a)),
        "cells_valid_only_in_39cm": int(np.count_nonzero(only_b)),
    }


def write_diagnosis_report(report_path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Surface Depth Quality Diagnosis",
        "",
        "This diagnosis is for S4-real offline LiDAR surface DEM depth inversion.",
        "",
        "Important conclusions:",
        "",
        "- Results are diagnostic and should not be treated as final engineering accuracy.",
        "- The 13cm dormitory scene is clearly overestimated when included in this report.",
        "- Playground pit data is used as an additional controlled S4-real validation scene.",
        "- Low valid-depth coverage indicates that results should be interpreted cautiously.",
        "",
        "## Per-case Diagnosis",
        "",
        "| Case | Known cm | Mean cm | Median cm | Max cm | Mean error cm | Coverage | Depth cells | Surface cells | High outliers | Low outliers | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["cases"]:
        warnings = []
        if item["high_error_warning"]:
            warnings.append("high_error")
        if item["low_coverage_warning"]:
            warnings.append("low_coverage")
        if item["sparse_point_warning"]:
            warnings.append("sparse_points")
        lines.append(
            "| "
            f"{item['case_name']} | "
            f"{_fmt(item.get('known_depth_cm'))} | "
            f"{_fmt(item.get('mean_depth_cm'))} | "
            f"{_fmt(item.get('median_depth_cm'))} | "
            f"{_fmt(item.get('max_depth_cm'))} | "
            f"{_fmt(item.get('mean_error_cm'))} | "
            f"{float(item.get('valid_depth_ratio_in_water_region', 0.0)):.4f} | "
            f"{item.get('depth_valid_cell_count')} | "
            f"{item.get('surface_valid_cell_count')} | "
            f"{item.get('outlier_high_depth_cell_count')} | "
            f"{item.get('outlier_low_depth_cell_count')} | "
            f"{', '.join(warnings) if warnings else 'none'} |"
        )

    cross = result.get("cross_case_comparison", {})
    lines.extend(["", "## Cross-case Comparison", ""])
    if cross.get("warning"):
        lines.append(f"- {cross.get('warning')}")
    else:
        lines.extend(
            [
                f"- case_a: {cross.get('case_a')}",
                f"- case_b: {cross.get('case_b')}",
                f"- overlapping_valid_cell_count: {cross.get('overlapping_valid_cell_count')}",
                f"- overlap_ratio: {_fmt(cross.get('overlap_ratio'), 4)}",
                f"- same_cell_depth_difference_mean_cm: {_fmt(cross.get('same_cell_depth_difference_mean_cm'))}",
                f"- same_cell_depth_difference_median_cm: {_fmt(cross.get('same_cell_depth_difference_median_cm'))}",
                f"- cells_valid_only_in_case_a: {cross.get('cells_valid_only_in_13cm')}",
                f"- cells_valid_only_in_case_b: {cross.get('cells_valid_only_in_39cm')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The diagnosis intentionally does not tune ROI, thresholds, or DEM settings to fit known depths. "
            "High error or low coverage should be treated as a real quality signal for the current data and calibration setup.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_comparison_figure(case_results: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate diagnosis figure: {exc}") from exc

    labels = [item["case_name"].replace("water_sim_", "").replace("_001", "") for item in case_results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle("S4-real Surface Depth Quality Diagnosis", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for item, label in zip(case_results, labels):
        values = item["_arrays"]["depth_values_cm"]
        if values.size:
            ax.hist(values, bins=14, alpha=0.55, label=label)
    ax.set_title("Depth distribution")
    ax.set_xlabel("depth (cm)")
    ax.set_ylabel("cell count")
    ax.legend()

    ax = axes[0, 1]
    for item, label in zip(case_results, labels):
        values = item["_arrays"]["point_values"]
        if values.size:
            ax.hist(values, bins=14, alpha=0.55, label=label)
    ax.set_title("Point count distribution")
    ax.set_xlabel("points / valid surface cell")
    ax.set_ylabel("cell count")
    ax.legend()

    ax = axes[1, 0]
    coverage = [item["valid_depth_ratio_in_water_region"] for item in case_results]
    ax.bar(labels, coverage, color=["#2563eb", "#0f766e"][: len(labels)])
    ax.axhline(0.30, color="#dc2626", linestyle="--", linewidth=1.0, label="warning threshold 0.30")
    ax.set_title("Valid depth coverage")
    ax.set_ylabel("valid depth ratio")
    ax.set_ylim(0, max(0.35, max(coverage + [0.0]) * 1.25))
    ax.legend()

    ax = axes[1, 1]
    known = [item["known_depth_cm"] for item in case_results]
    mean = [item["mean_depth_cm"] for item in case_results]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, known, width, label="known depth", color="#94a3b8")
    ax.bar(x + width / 2, mean, width, label="measured mean", color="#ea580c")
    ax.set_title("Error vs known depth")
    ax.set_ylabel("depth (cm)")
    ax.set_xticks(x, labels)
    ax.legend()

    fig.text(
        0.5,
        0.01,
        "S4-real diagnosis: no tuning is applied; high error or low coverage should be interpreted cautiously.",
        ha="center",
        fontsize=9,
        color="#92400e",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def public_case_result(case_result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case_result.items() if key != "_arrays"}


def diagnose_surface_depth_quality(
    config_path: str | Path,
    project_root: str | Path,
    cases: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_cases = _case_names(config, cases)
    case_results = [diagnose_case(config, root, case_name) for case_name in selected_cases]
    cross_case = cross_case_compare(case_results)

    output_config = config["output"]
    json_dir = resolve_project_path(root, output_config["json_dir"])
    report_dir = resolve_project_path(root, output_config["report_dir"])
    figure_dir = resolve_project_path(root, output_config["figure_dir"])
    json_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_json = json_dir / "surface_depth_quality_diagnosis.json"
    output_report = report_dir / "surface_depth_quality_diagnosis.md"
    output_figure = figure_dir / "surface_depth_quality_comparison.png"

    result = {
        "stage": "S4_real_surface_depth_quality_diagnosis",
        "cases": [public_case_result(item) for item in case_results],
        "cross_case_comparison": cross_case,
        "diagnosis_note": (
            "S4-real diagnosis reads existing surface-depth outputs only. It does not tune ROI, "
            "thresholds, or depth values to match known depths."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "surface_depth_quality_diagnosis_json": str(output_json),
            "surface_depth_quality_diagnosis_report": str(output_report),
            "surface_depth_quality_comparison_figure": str(output_figure),
        },
    }

    save_comparison_figure(case_results, output_figure)
    write_diagnosis_report(output_report, result)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[surface_quality] cases:", ", ".join(selected_cases))
    for item in result["cases"]:
        print(
            "[surface_quality] "
            f"{item['case_name']}: mean={_fmt(item['mean_depth_cm'])} cm, "
            f"known={_fmt(item['known_depth_cm'])} cm, "
            f"mean_error={_fmt(item['mean_error_cm'])} cm, "
            f"coverage={item['valid_depth_ratio_in_water_region']:.4f}, "
            f"high_error={item['high_error_warning']}, "
            f"low_coverage={item['low_coverage_warning']}"
        )
    print(
        "[surface_quality] overlap cells:",
        result["cross_case_comparison"].get("overlapping_valid_cell_count"),
    )
    print("[surface_quality] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose S4-real surface depth quality.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--cases", nargs="*", help="Case names to diagnose")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    diagnose_surface_depth_quality(args.config, args.project_root, args.cases)


if __name__ == "__main__":
    main()
