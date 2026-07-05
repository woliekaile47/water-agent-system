#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose DEM-space water masks for manual refinement."""

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

from src.masks.create_dem_space_water_mask import get_grid_resolution, load_json, load_yaml, resolve_project_path


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = values.astype(np.float64)
    if values.size == 0:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "std": None,
        }
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "std": float(np.std(values)),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    all_neighbors_in_region = np.ones(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            all_neighbors_in_region &= neighbor
    return mask & ~all_neighbors_in_region


def outer_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    any_neighbor_in_region = np.zeros(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            any_neighbor_in_region |= neighbor
    return (~mask) & any_neighbor_in_region


def save_mask_diagnosis_on_dem(
    ground_dem: np.ndarray,
    ground_valid_mask: np.ndarray,
    mask: np.ndarray,
    boundary_mask: np.ndarray,
    outlier_10_mask: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate DEM diagnosis figure: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked_dem = np.ma.masked_where(~ground_valid_mask, ground_dem)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    image = ax.imshow(masked_dem, origin="lower", cmap="viridis")
    mask_rows, mask_cols = np.where(mask)
    boundary_rows, boundary_cols = np.where(boundary_mask)
    outlier_rows, outlier_cols = np.where(outlier_10_mask)
    ax.scatter(mask_cols, mask_rows, s=1, c="#38bdf8", alpha=0.15, linewidths=0, label="mask cells")
    ax.scatter(boundary_cols, boundary_rows, s=5, c="#facc15", linewidths=0, label="inner boundary")
    ax.scatter(outlier_cols, outlier_rows, s=12, c="#dc2626", marker="x", linewidths=0.7, label="boundary >10cm from median")
    ax.set_title("Playground Pit DEM-space Mask Diagnosis on Ground DEM")
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("ground z (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_boundary_outlier_figure(
    ground_dem: np.ndarray,
    ground_valid_mask: np.ndarray,
    boundary_mask: np.ndarray,
    median_height_m: float | None,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate outlier figure: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_boundary = boundary_mask & ground_valid_mask & np.isfinite(ground_dem)
    rows, cols = np.where(valid_boundary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle("Playground Pit Mask Boundary Height Outliers", fontweight="bold")
    if rows.size and median_height_m is not None:
        heights_cm = ground_dem[valid_boundary].astype(np.float64) * 100.0
        deviation_cm = np.abs(heights_cm - median_height_m * 100.0)
        scatter = axes[0].scatter(cols, rows, c=deviation_cm, s=15, cmap="inferno")
        axes[0].set_title("Boundary deviation from median")
        axes[0].set_xlabel("grid col")
        axes[0].set_ylabel("grid row")
        cbar = fig.colorbar(scatter, ax=axes[0])
        cbar.set_label("abs deviation (cm)")
        axes[1].hist(deviation_cm, bins=20, color="#0f766e", alpha=0.85)
        axes[1].axvline(10, color="#f59e0b", linestyle="--", label="10 cm")
        axes[1].axvline(20, color="#ef4444", linestyle="--", label="20 cm")
        axes[1].axvline(50, color="#7f1d1d", linestyle="--", label="50 cm")
        axes[1].set_title("Boundary height deviation histogram")
        axes[1].set_xlabel("abs deviation (cm)")
        axes[1].set_ylabel("cell count")
        axes[1].legend()
    else:
        axes[0].text(0.5, 0.5, "No valid boundary cells", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No valid boundary cells", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_mask_on_point_count(
    point_count: np.ndarray,
    mask: np.ndarray,
    boundary_mask: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate point-count diagnosis figure: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked_count = np.ma.masked_where(point_count <= 0, point_count)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    image = ax.imshow(masked_count, origin="lower", cmap="magma")
    mask_rows, mask_cols = np.where(mask)
    boundary_rows, boundary_cols = np.where(boundary_mask)
    ax.scatter(mask_cols, mask_rows, s=1, c="#38bdf8", alpha=0.15, linewidths=0, label="mask cells")
    ax.scatter(boundary_cols, boundary_rows, s=5, c="#22c55e", linewidths=0, label="inner boundary")
    ax.set_title("Playground Pit DEM-space Mask on Point Count")
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("points / cell")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Playground Pit DEM-space Mask Diagnosis",
        "",
        "This report diagnoses the DEM-space water-region mask used by S4-real-B boundary waterline inversion.",
        "",
        "The script does not tune the polygon to match known depth. It only reports mask geometry, boundary height stability, and point-density diagnostics for manual refinement.",
        "",
        "## Summary",
        "",
        f"- mask_shape: {result['mask_shape']}",
        f"- mask_cell_count: {result['mask_cell_count']}",
        f"- mask_area_m2: {_fmt(result['mask_area_m2'], 4)}",
        f"- row range: {result['row_min']} - {result['row_max']}",
        f"- col range: {result['col_min']} - {result['col_max']}",
        f"- boundary_cell_count: {result['boundary_cell_count']}",
        f"- boundary_valid_cell_count: {result['boundary_valid_cell_count']}",
        f"- boundary_valid_ratio: {_fmt(result['boundary_valid_ratio'], 4)}",
        f"- boundary height median/std m: {_fmt(result['boundary_height_stats']['median'], 4)} / {_fmt(result['boundary_height_stats']['std'], 4)}",
        "",
        "## Boundary Outliers",
        "",
        f"- deviation > 10 cm: {result['boundary_outlier_counts']['gt_10cm']}",
        f"- deviation > 20 cm: {result['boundary_outlier_counts']['gt_20cm']}",
        f"- deviation > 50 cm: {result['boundary_outlier_counts']['gt_50cm']}",
        "",
        "## Point Count",
        "",
        f"- mask point_count min/mean/median/max: {_fmt(result['point_count_on_mask_stats']['min'])} / {_fmt(result['point_count_on_mask_stats']['mean'])} / {_fmt(result['point_count_on_mask_stats']['median'])} / {_fmt(result['point_count_on_mask_stats']['max'])}",
        f"- boundary point_count min/mean/median/max: {_fmt(result['point_count_on_boundary_stats']['min'])} / {_fmt(result['point_count_on_boundary_stats']['mean'])} / {_fmt(result['point_count_on_boundary_stats']['median'])} / {_fmt(result['point_count_on_boundary_stats']['max'])}",
        "",
        "## Manual Refinement Guidance",
        "",
        "- If boundary height std is high, inspect the DEM and point-count overlays and shrink or reshape `refined_polygon_points_rc` around the true pit water region.",
        "- If many boundary points are outliers, the polygon likely crosses slope, wall, sparse LiDAR returns, or non-water terrain.",
        "- Recreate the mask with `--use-refined` only after manual review; do not tune the polygon solely to match `known_depth_cm`.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_dem_space_mask(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)

    ground_dem_path = resolve_project_path(root, config["ground_dem_interpolated_path"])
    ground_valid_path = resolve_project_path(root, config["ground_dem_valid_mask_path"])
    point_count_path = resolve_project_path(root, config["ground_dem_point_count_path"])
    mask_path = resolve_project_path(root, config["output_mask_npy"])
    mask_metadata_path = resolve_project_path(root, config["output_metadata"])
    ground_metadata_path = resolve_project_path(root, config["ground_dem_metadata_path"])

    ground_dem = np.load(ground_dem_path).astype(np.float32)
    ground_valid_mask = np.load(ground_valid_path).astype(bool)
    point_count = np.load(point_count_path)
    water_mask = np.load(mask_path).astype(bool)
    mask_metadata = load_json(mask_metadata_path)
    ground_metadata = load_json(ground_metadata_path)
    if not (ground_dem.shape == ground_valid_mask.shape == point_count.shape == water_mask.shape):
        raise ValueError(
            "ground_dem, ground_valid_mask, point_count, and mask shapes must match: "
            f"{ground_dem.shape}, {ground_valid_mask.shape}, {point_count.shape}, {water_mask.shape}"
        )

    mask_rows, mask_cols = np.where(water_mask)
    if mask_rows.size == 0:
        raise RuntimeError("Water mask has zero cells; cannot diagnose.")
    boundary = inner_boundary(water_mask)
    outer = outer_boundary(water_mask)
    finite_ground = np.isfinite(ground_dem)
    valid_boundary = boundary & ground_valid_mask & finite_ground
    valid_outer = outer & ground_valid_mask & finite_ground

    boundary_values_m = ground_dem[valid_boundary].astype(np.float64)
    outer_values_m = ground_dem[valid_outer].astype(np.float64)
    boundary_stats = _stats(boundary_values_m)
    outer_stats = _stats(outer_values_m)

    median_m = boundary_stats["median"]
    if median_m is None:
        deviation_cm = np.array([], dtype=np.float64)
        outlier_10 = np.zeros(water_mask.shape, dtype=bool)
        outlier_20 = np.zeros(water_mask.shape, dtype=bool)
        outlier_50 = np.zeros(water_mask.shape, dtype=bool)
    else:
        deviation_map_cm = np.zeros(water_mask.shape, dtype=np.float32)
        deviation_map_cm[valid_boundary] = np.abs(ground_dem[valid_boundary].astype(np.float64) - median_m) * 100.0
        deviation_cm = deviation_map_cm[valid_boundary].astype(np.float64)
        outlier_10 = valid_boundary & (deviation_map_cm > 10.0)
        outlier_20 = valid_boundary & (deviation_map_cm > 20.0)
        outlier_50 = valid_boundary & (deviation_map_cm > 50.0)

    grid_resolution = get_grid_resolution(ground_metadata)
    mask_cell_count = int(np.count_nonzero(water_mask))
    mask_area_m2 = float(mask_cell_count * grid_resolution * grid_resolution)

    json_path = resolve_project_path(root, "outputs/json/playground_pit_mask_diagnosis.json")
    report_path = resolve_project_path(root, "outputs/reports/playground_pit_mask_diagnosis.md")
    figure_dem_path = resolve_project_path(root, "outputs/figures/playground_pit_mask_diagnosis_on_dem.png")
    figure_outlier_path = resolve_project_path(root, "outputs/figures/playground_pit_mask_boundary_height_outliers.png")
    figure_count_path = resolve_project_path(root, "outputs/figures/playground_pit_mask_on_point_count.png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dem_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "stage": "S3_playground_pit_dem_space_mask_diagnosis",
        "scene_type": config.get("scene_type"),
        "case_name": config.get("case_name"),
        "mask_shape": list(water_mask.shape),
        "mask_cell_count": mask_cell_count,
        "mask_area_m2": mask_area_m2,
        "grid_resolution": grid_resolution,
        "row_min": int(mask_rows.min()),
        "row_max": int(mask_rows.max()),
        "col_min": int(mask_cols.min()),
        "col_max": int(mask_cols.max()),
        "boundary_cell_count": int(np.count_nonzero(boundary)),
        "outer_boundary_cell_count": int(np.count_nonzero(outer)),
        "boundary_valid_cell_count": int(np.count_nonzero(valid_boundary)),
        "outer_boundary_valid_cell_count": int(np.count_nonzero(valid_outer)),
        "boundary_valid_ratio": float(np.count_nonzero(valid_boundary) / max(1, np.count_nonzero(boundary))),
        "outer_boundary_valid_ratio": float(np.count_nonzero(valid_outer) / max(1, np.count_nonzero(outer))),
        "boundary_height_stats": boundary_stats,
        "outer_boundary_height_stats": outer_stats,
        "boundary_height_std_cm": None if boundary_stats["std"] is None else float(boundary_stats["std"] * 100.0),
        "outer_boundary_height_std_cm": None if outer_stats["std"] is None else float(outer_stats["std"] * 100.0),
        "boundary_outlier_counts": {
            "gt_10cm": int(np.count_nonzero(outlier_10)),
            "gt_20cm": int(np.count_nonzero(outlier_20)),
            "gt_50cm": int(np.count_nonzero(outlier_50)),
        },
        "boundary_outlier_ratio": {
            "gt_10cm": float(np.count_nonzero(outlier_10) / max(1, np.count_nonzero(valid_boundary))),
            "gt_20cm": float(np.count_nonzero(outlier_20) / max(1, np.count_nonzero(valid_boundary))),
            "gt_50cm": float(np.count_nonzero(outlier_50) / max(1, np.count_nonzero(valid_boundary))),
        },
        "point_count_on_mask_stats": _stats(point_count[water_mask]),
        "point_count_on_boundary_stats": _stats(point_count[boundary]),
        "source_files": {
            "ground_dem": str(ground_dem_path),
            "ground_valid_mask": str(ground_valid_path),
            "ground_point_count": str(point_count_path),
            "ground_metadata": str(ground_metadata_path),
            "water_mask": str(mask_path),
            "water_mask_metadata": str(mask_metadata_path),
        },
        "mask_metadata_polygon_points_rc": mask_metadata.get("polygon_points_rc"),
        "diagnosis_note": (
            "This diagnosis only evaluates mask geometry, boundary height stability, and point density. "
            "It does not tune the mask to match known water depth."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "mask_diagnosis_json": str(json_path),
            "mask_diagnosis_report": str(report_path),
            "mask_diagnosis_on_dem": str(figure_dem_path),
            "mask_boundary_height_outliers": str(figure_outlier_path),
            "mask_on_point_count": str(figure_count_path),
        },
    }

    save_mask_diagnosis_on_dem(ground_dem, ground_valid_mask, water_mask, boundary, outlier_10, figure_dem_path)
    save_boundary_outlier_figure(ground_dem, ground_valid_mask, boundary, median_m, figure_outlier_path)
    save_mask_on_point_count(point_count, water_mask, boundary, figure_count_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_report(report_path, result)

    print(f"[mask_diagnosis] mask_shape: {result['mask_shape']}")
    print(f"[mask_diagnosis] mask_cell_count: {mask_cell_count}")
    print(f"[mask_diagnosis] mask_area_m2: {mask_area_m2:.4f}")
    print(f"[mask_diagnosis] row range: {result['row_min']} - {result['row_max']}")
    print(f"[mask_diagnosis] col range: {result['col_min']} - {result['col_max']}")
    print(f"[mask_diagnosis] boundary_cell_count: {result['boundary_cell_count']}")
    print(f"[mask_diagnosis] boundary_valid_cell_count: {result['boundary_valid_cell_count']}")
    print(f"[mask_diagnosis] boundary_valid_ratio: {result['boundary_valid_ratio']:.4f}")
    print(f"[mask_diagnosis] boundary_height_median_m: {_fmt(boundary_stats['median'], 4)}")
    print(f"[mask_diagnosis] boundary_height_std_cm: {_fmt(result['boundary_height_std_cm'], 4)}")
    print(
        "[mask_diagnosis] boundary outliers >10/>20/>50 cm: "
        f"{result['boundary_outlier_counts']['gt_10cm']} / "
        f"{result['boundary_outlier_counts']['gt_20cm']} / "
        f"{result['boundary_outlier_counts']['gt_50cm']}"
    )
    point_mask = result["point_count_on_mask_stats"]
    point_boundary = result["point_count_on_boundary_stats"]
    print(
        "[mask_diagnosis] point_count_on_mask min/mean/median/max: "
        f"{_fmt(point_mask['min'])} / {_fmt(point_mask['mean'])} / "
        f"{_fmt(point_mask['median'])} / {_fmt(point_mask['max'])}"
    )
    print(
        "[mask_diagnosis] point_count_on_boundary min/mean/median/max: "
        f"{_fmt(point_boundary['min'])} / {_fmt(point_boundary['mean'])} / "
        f"{_fmt(point_boundary['median'])} / {_fmt(point_boundary['max'])}"
    )
    print("[mask_diagnosis] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose playground_pit DEM-space water mask.")
    parser.add_argument("--config", required=True, help="Path to configs/playground_pit_dem_mask_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    diagnose_dem_space_mask(args.config, args.project_root)


if __name__ == "__main__":
    main()
