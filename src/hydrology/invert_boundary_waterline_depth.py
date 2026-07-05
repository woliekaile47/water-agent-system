#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4-real-B: boundary-based waterline depth inversion.

This method estimates a water level from the ground DEM elevations along
the water-region boundary, then computes depth inside the water region.
It is intended as an offline alternative when direct LiDAR water-surface
returns are unstable.
"""

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

from src.dem.build_surface_dem_from_rosbag import (
    get_case_config,
    get_ground_dem_input_paths,
    load_json,
    load_yaml,
    parse_dem_grid,
    resolve_project_path,
)


NOTE = (
    "S4-real-B estimates water level from the water-region boundary on the ground DEM. "
    "It is an offline MVP method for cases where direct LiDAR water-surface returns are unstable. "
    "The result depends strongly on water mask boundary quality and ground DEM quality."
)

PLAYGROUND_PIT_MASK_PATH = "data/masks/playground_pit_water_region_mask.npy"


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "std": None,
        }
    return {
        "min": float(np.min(values)),
        "p10": _safe_percentile(values, 10),
        "p25": _safe_percentile(values, 25),
        "median": float(np.median(values)),
        "p75": _safe_percentile(values, 75),
        "p90": _safe_percentile(values, 90),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _trimmed_median(values: np.ndarray) -> float:
    if values.size == 0:
        raise RuntimeError("Cannot estimate water level from zero boundary cells.")
    if values.size < 10:
        return float(np.median(values))
    sorted_values = np.sort(values.astype(np.float64))
    trim = int(np.floor(sorted_values.size * 0.10))
    trimmed = sorted_values[trim : sorted_values.size - trim]
    if trimmed.size == 0:
        trimmed = sorted_values
    return float(np.median(trimmed))


def extract_boundary_mask(water_region_mask: np.ndarray) -> np.ndarray:
    mask = water_region_mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"water_region_mask must be 2D, got shape={mask.shape}")
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    all_neighbors_in_region = np.ones(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            all_neighbors_in_region &= neighbor
    return mask & ~all_neighbors_in_region


def load_water_region_mask_for_boundary(
    config: dict[str, Any],
    project_root: Path,
    ground_inputs: dict[str, Any],
    ground_valid_mask: np.ndarray,
) -> tuple[np.ndarray, str, str | None, list[str]]:
    warnings: list[str] = []
    if str(ground_inputs.get("scene_type")) == "playground_pit":
        playground_mask_path = resolve_project_path(project_root, PLAYGROUND_PIT_MASK_PATH)
        if not playground_mask_path.exists():
            raise FileNotFoundError(
                "playground_pit requires a DEM-space water region mask. "
                f"Missing: {playground_mask_path}. Run create_dem_water_mask first."
            )
        mask = np.load(playground_mask_path).astype(bool)
        if mask.shape != ground_valid_mask.shape:
            raise ValueError(
                "playground_pit DEM-space mask shape does not match ground DEM. "
                f"mask_shape={mask.shape}, ground_shape={ground_valid_mask.shape}. "
                "Refusing to use old or incompatible fallback masks."
            )
        return mask, "playground_pit_dem_space_mask", str(playground_mask_path), warnings

    scene_mask_path = ground_inputs.get("water_region_mask")
    if scene_mask_path is not None and Path(scene_mask_path).exists():
        mask = np.load(scene_mask_path).astype(bool)
        if mask.shape != ground_valid_mask.shape:
            raise ValueError(
                "Scene-specific water region mask shape does not match ground DEM: "
                f"{mask.shape} vs {ground_valid_mask.shape}"
            )
        return mask, "scene_specific_water_region_mask", str(scene_mask_path), warnings

    global_mask_path = resolve_project_path(project_root, config["fusion"]["water_region_mask_path"])
    if global_mask_path.exists():
        mask = np.load(global_mask_path).astype(bool)
        if mask.shape == ground_valid_mask.shape:
            warnings.append(
                "No playground_pit-specific water mask is configured; using existing water_region_mask MVP."
            )
            return mask, "existing_water_region_mask_mvp", str(global_mask_path), warnings
        warnings.append(
            "Existing water_region_mask exists but shape is incompatible with the scene ground DEM; "
            f"mask_shape={mask.shape}, ground_shape={ground_valid_mask.shape}."
        )

    if str(ground_inputs.get("water_region_mode")) == "ground_valid_mask":
        warnings.append(
            "No playground_pit-specific water mask is configured. Using the existing S4-real MVP "
            "ground_valid_mask fallback as the water region. Configure a playground_pit-specific mask "
            "before treating this as a formal result."
        )
        return ground_valid_mask.astype(bool), "existing_water_region_mask_mvp", "ground_valid_mask_fallback", warnings

    raise FileNotFoundError(
        "No compatible water region mask is available. Configure a playground_pit-specific mask or "
        "provide a matching existing water_region_mask."
    )


def save_depth_heatmap(
    depth_map_m: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    case_name: str,
    quality_status: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate boundary waterline heatmap: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_cm = depth_map_m.astype(np.float64) * 100.0
    masked = np.ma.masked_where(~valid_mask, depth_cm)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    image = ax.imshow(masked, origin="lower", cmap="Blues")
    ax.set_title(f"S4-real-B Boundary Waterline Depth - {case_name}\nquality={quality_status}")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("depth (cm)")
    fig.text(
        0.5,
        0.01,
        "Boundary-based waterline MVP; depends on mask boundary and ground DEM quality.",
        ha="center",
        fontsize=8,
        color="#92400e",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_boundary_quality(
    boundary_valid_cell_count: int,
    boundary_height_std_cm: float | None,
    valid_depth_ratio: float,
    mean_error_cm: float | None,
) -> tuple[str, list[str], list[str]]:
    reject_reasons: list[str] = []
    warning_reasons: list[str] = []

    if boundary_valid_cell_count < 10:
        reject_reasons.append(f"boundary_valid_cell_count={boundary_valid_cell_count} < 10")

    if boundary_height_std_cm is None:
        reject_reasons.append("boundary_height_std_cm is unavailable")
    elif boundary_height_std_cm > 10.0:
        reject_reasons.append(f"boundary_height_std_cm={boundary_height_std_cm:.2f} > 10 cm")
    elif boundary_height_std_cm > 5.0:
        warning_reasons.append(f"boundary_height_std_cm={boundary_height_std_cm:.2f} > 5 cm")

    if valid_depth_ratio < 0.15:
        reject_reasons.append(f"valid_depth_ratio_in_water_region={valid_depth_ratio:.4f} < 0.15")
    elif valid_depth_ratio < 0.30:
        warning_reasons.append(f"valid_depth_ratio_in_water_region={valid_depth_ratio:.4f} < 0.30")

    if mean_error_cm is not None:
        if abs(mean_error_cm) > 20.0:
            reject_reasons.append(f"abs(mean_error_cm)={abs(mean_error_cm):.2f} > 20 cm")
        elif abs(mean_error_cm) > 10.0:
            warning_reasons.append(f"abs(mean_error_cm)={abs(mean_error_cm):.2f} > 10 cm")

    if reject_reasons:
        return "reject", reject_reasons, warning_reasons
    if warning_reasons:
        return "warning", reject_reasons, warning_reasons
    return "pass", reject_reasons, warning_reasons


def write_boundary_report(report_path: Path, json_dir: Path) -> None:
    rows = []
    for path in sorted(json_dir.glob("boundary_waterline_depth_result_*.json")):
        try:
            rows.append(load_json(path))
        except Exception:
            continue

    lines = [
        "# S4-real-B Boundary Waterline Depth Report",
        "",
        "S4-real-B estimates water level from the water-region boundary on the ground DEM, "
        "then computes depth inside the water region.",
        "",
        "This method is useful when direct LiDAR water-surface returns are unstable, but it depends on "
        "the correctness of the water mask boundary and ground DEM.",
        "",
        "| Case | Quality | Boundary cells | Water level m | Boundary median m | Boundary std cm | Mean cm | Median cm | Max cm | Known cm | Mean error cm | Area m2 | Volume m3 | Mask source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        lines.append(
            "| "
            f"{item.get('case_name')} | "
            f"{item.get('boundary_quality_status')} | "
            f"{item.get('boundary_valid_cell_count')} | "
            f"{_fmt(item.get('estimated_water_level_m'))} | "
            f"{_fmt(item.get('boundary_height_median_m'))} | "
            f"{_fmt(item.get('boundary_height_std_cm'))} | "
            f"{_fmt(item.get('mean_depth_cm'))} | "
            f"{_fmt(item.get('median_depth_cm'))} | "
            f"{_fmt(item.get('max_depth_cm'))} | "
            f"{_fmt(item.get('known_depth_cm'))} | "
            f"{_fmt(item.get('mean_error_cm'))} | "
            f"{_fmt(item.get('area_m2'))} | "
            f"{_fmt(item.get('volume_m3'))} | "
            f"{item.get('mask_source')} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If `mask_source` is `existing_water_region_mask_mvp`, the result is an MVP fallback. "
            "A playground_pit-specific water mask should be configured before formal validation.",
            "- S4-real-B does not tune parameters to known depth and does not modify rosbag data.",
            "- Quality status `reject` means the result should be kept for diagnosis only.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def invert_boundary_waterline_depth(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", selected_case))
    known_depth_cm = None if case_config.get("known_depth_cm") is None else float(case_config["known_depth_cm"])

    ground_inputs = get_ground_dem_input_paths(config, root, case_config)
    ground_dem_path = Path(ground_inputs["ground_dem"])
    ground_valid_mask_path = Path(ground_inputs["ground_valid_mask"])
    ground_metadata_path = Path(ground_inputs["ground_metadata"])

    ground_dem = np.load(ground_dem_path).astype(np.float32)
    ground_valid_mask = np.load(ground_valid_mask_path).astype(bool)
    ground_metadata = load_json(ground_metadata_path)
    grid = parse_dem_grid(ground_metadata, tuple(ground_dem.shape))
    water_region_mask, mask_source, mask_path_or_mode, mask_warnings = load_water_region_mask_for_boundary(
        config,
        root,
        ground_inputs,
        ground_valid_mask,
    )
    if water_region_mask.shape != ground_dem.shape:
        raise ValueError(
            "water_region_mask shape does not match ground DEM: "
            f"{water_region_mask.shape} vs {ground_dem.shape}"
        )

    finite_ground = np.isfinite(ground_dem)
    boundary_mask = extract_boundary_mask(water_region_mask)
    boundary_valid_mask = boundary_mask & ground_valid_mask & finite_ground
    boundary_values_m = ground_dem[boundary_valid_mask].astype(np.float64)
    boundary_stats = _safe_stats(boundary_values_m)
    estimated_water_level_m = _trimmed_median(boundary_values_m)

    depth_valid_mask = water_region_mask & ground_valid_mask & finite_ground
    raw_depth_m = estimated_water_level_m - ground_dem.astype(np.float64)
    depth_map_m = np.zeros(ground_dem.shape, dtype=np.float32)
    depth_map_m[depth_valid_mask] = np.maximum(raw_depth_m[depth_valid_mask], 0.0).astype(np.float32)

    values_m = depth_map_m[depth_valid_mask].astype(np.float64)
    water_region_cell_count = int(np.count_nonzero(water_region_mask))
    valid_depth_cell_count = int(np.count_nonzero(depth_valid_mask))
    valid_depth_ratio = float(valid_depth_cell_count / max(1, water_region_cell_count))
    grid_size_m = float(grid["grid_size_m"])
    cell_area_m2 = grid_size_m * grid_size_m
    area_m2 = float(valid_depth_cell_count * cell_area_m2)
    volume_m3 = float(np.sum(values_m) * cell_area_m2)

    if values_m.size:
        values_cm = values_m * 100.0
        mean_depth_cm = float(np.mean(values_cm))
        median_depth_cm = float(np.median(values_cm))
        max_depth_cm = float(np.max(values_cm))
        min_depth_cm = float(np.min(values_cm))
    else:
        mean_depth_cm = median_depth_cm = max_depth_cm = min_depth_cm = None

    mean_error_cm = None if known_depth_cm is None or mean_depth_cm is None else float(mean_depth_cm - known_depth_cm)
    median_error_cm = (
        None if known_depth_cm is None or median_depth_cm is None else float(median_depth_cm - known_depth_cm)
    )
    boundary_height_std_cm = None if boundary_stats["std"] is None else float(boundary_stats["std"] * 100.0)
    quality_status, reject_reasons, warning_reasons = evaluate_boundary_quality(
        boundary_valid_cell_count=int(np.count_nonzero(boundary_valid_mask)),
        boundary_height_std_cm=boundary_height_std_cm,
        valid_depth_ratio=valid_depth_ratio,
        mean_error_cm=mean_error_cm,
    )

    hydrology_dir = resolve_project_path(root, config["output"]["hydrology_dir"]) / output_name
    json_dir = resolve_project_path(root, config["output"]["json_dir"])
    figure_dir = resolve_project_path(root, config["output"]["figure_dir"])
    report_dir = resolve_project_path(root, config["output"]["report_dir"])
    hydrology_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_depth_map = hydrology_dir / "boundary_waterline_depth_map.npy"
    output_valid_mask = hydrology_dir / "boundary_waterline_depth_valid_mask.npy"
    output_json = json_dir / f"boundary_waterline_depth_result_{output_name}.json"
    output_figure = figure_dir / f"boundary_waterline_depth_heatmap_{output_name}.png"
    output_report = report_dir / "boundary_waterline_depth_report.md"

    np.save(output_depth_map, depth_map_m)
    np.save(output_valid_mask, depth_valid_mask)
    save_depth_heatmap(depth_map_m, depth_valid_mask, output_figure, output_name, quality_status)

    result = {
        "stage": "S4_real_B_boundary_waterline_depth_inversion",
        "case_name": output_name,
        "depth_source": "boundary_based_waterline_inversion",
        "source_ground_dem": str(ground_dem_path),
        "source_ground_valid_mask": str(ground_valid_mask_path),
        "source_ground_metadata": str(ground_metadata_path),
        "scene_type": case_config.get("scene_type", ""),
        "matched_dry_baseline": case_config.get("matched_dry_baseline"),
        "mask_source": mask_source,
        "mask_path_or_mode": mask_path_or_mode,
        "mask_note": (
            "No playground_pit-specific mask is currently configured when mask_source is "
            "existing_water_region_mask_mvp. A dedicated playground_pit water mask is required "
            "for formal validation."
        ),
        "grid_size_m": grid_size_m,
        "cell_area_m2": cell_area_m2,
        "water_region_cell_count": water_region_cell_count,
        "boundary_cell_count": int(np.count_nonzero(boundary_mask)),
        "boundary_valid_cell_count": int(np.count_nonzero(boundary_valid_mask)),
        "boundary_height_min_m": boundary_stats["min"],
        "boundary_height_p10_m": boundary_stats["p10"],
        "boundary_height_p25_m": boundary_stats["p25"],
        "boundary_height_median_m": boundary_stats["median"],
        "boundary_height_p75_m": boundary_stats["p75"],
        "boundary_height_p90_m": boundary_stats["p90"],
        "boundary_height_max_m": boundary_stats["max"],
        "boundary_height_std_m": boundary_stats["std"],
        "boundary_height_std_cm": boundary_height_std_cm,
        "estimated_water_level_m": estimated_water_level_m,
        "water_level_estimation_method": "trimmed_median_drop_low_high_10_percent",
        "valid_depth_cell_count": valid_depth_cell_count,
        "valid_depth_ratio_in_water_region": valid_depth_ratio,
        "area_m2": area_m2,
        "volume_m3": volume_m3,
        "mean_depth_cm": mean_depth_cm,
        "median_depth_cm": median_depth_cm,
        "max_depth_cm": max_depth_cm,
        "min_depth_cm": min_depth_cm,
        "known_depth_cm": known_depth_cm,
        "mean_error_cm": mean_error_cm,
        "median_error_cm": median_error_cm,
        "boundary_quality_status": quality_status,
        "reject_reasons": reject_reasons,
        "warning_reasons": warning_reasons,
        "warning": mask_warnings + reject_reasons + warning_reasons,
        "note": NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "boundary_waterline_depth_map": str(output_depth_map),
            "boundary_waterline_depth_valid_mask": str(output_valid_mask),
            "boundary_waterline_depth_result_json": str(output_json),
            "boundary_waterline_depth_heatmap": str(output_figure),
            "boundary_waterline_depth_report": str(output_report),
        },
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_boundary_report(output_report, json_dir)

    print(f"[boundary_waterline] case_name: {output_name}")
    print(f"[boundary_waterline] mask_source: {mask_source}")
    print(f"[boundary_waterline] boundary_valid_cell_count: {result['boundary_valid_cell_count']}")
    print(f"[boundary_waterline] estimated_water_level_m: {estimated_water_level_m:.4f}")
    print(f"[boundary_waterline] boundary_height_median_m: {_fmt(boundary_stats['median'])}")
    print(f"[boundary_waterline] boundary_height_std_cm: {_fmt(boundary_height_std_cm)}")
    print(f"[boundary_waterline] mean_depth_cm: {_fmt(mean_depth_cm)}")
    print(f"[boundary_waterline] median_depth_cm: {_fmt(median_depth_cm)}")
    print(f"[boundary_waterline] max_depth_cm: {_fmt(max_depth_cm)}")
    print(f"[boundary_waterline] known_depth_cm: {_fmt(known_depth_cm)}")
    print(f"[boundary_waterline] mean_error_cm: {_fmt(mean_error_cm)}")
    print(f"[boundary_waterline] area_m2: {area_m2:.4f}")
    print(f"[boundary_waterline] volume_m3: {volume_m3:.4f}")
    print(f"[boundary_waterline] boundary_quality_status: {quality_status}")
    if result["warning"]:
        print(f"[boundary_waterline][WARN] {'; '.join(result['warning'])}")
    print("[boundary_waterline] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S4-real-B boundary-based waterline inversion.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--case", required=True, help="Case name, e.g. playground_pit_water_sim_6cm_001")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    invert_boundary_waterline_depth(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
