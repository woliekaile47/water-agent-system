#!/usr/bin/env python3
"""Run the Phase 2D-C-3C prediction-side SAM 2 geometry diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

NEAREST_RESAMPLE = getattr(Image, "Resampling", Image).NEAREST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate
from src.fusion.sam2_shoreline_geometry_adapter import (
    extended_reprojection_consistency,
    intersect_pixel_rays,
    deterministic_seed_pixels,
    estimate_level_from_ray_records,
    load_prediction_inputs,
    load_yaml,
    prediction_semantics,
    ray_hits_to_dem_seed_mask,
    reconstruct_seed_connected_lowland,
)
from src.fusion.water_surface_aware_mask_to_dem import reproject_water_surface
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=Path("data/sam2_geometry_inputs/frame_000059"))
    parser.add_argument(
        "--source-image",
        type=Path,
        default=Path("data/simulation_dynamic/sim_water_5cm_001/heavy/seed_43/frames/frame_000059.png"),
    )
    parser.add_argument(
        "--ground-dem",
        type=Path,
        default=Path("data/simulation/sim_dry_baseline_001/ground_truth/ground_dem_gt.npy"),
        help="Saved dry baseline terrain DEM; water-state answer data is forbidden.",
    )
    parser.add_argument("--sensors-config", type=Path, default=Path("simulation/config/sensors.yaml"))
    parser.add_argument("--mapping-config", type=Path, default=Path("configs/water_surface_aware_mapping.yaml"))
    parser.add_argument("--gate-config", type=Path, default=Path("configs/water_surface_aware_quality_gate.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam2_shoreline_geometry/frame_000059"))
    return parser


def _absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _colorize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    selected = finite if mask is None else finite & np.asarray(mask, dtype=bool)
    result = np.full((*array.shape, 3), 245, dtype=np.uint8)
    if not np.any(selected):
        return result
    low, high = float(np.min(array[selected])), float(np.max(array[selected]))
    normalized = np.clip((array - low) / max(high - low, 1e-12), 0.0, 1.0)
    result[..., 0] = (35 + 210 * normalized).astype(np.uint8)
    result[..., 1] = (70 + 155 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
    result[..., 2] = (235 - 185 * normalized).astype(np.uint8)
    result[~finite] = [80, 80, 80]
    return result


def save_shoreline_on_dem(
    path: Path,
    ground_dem: np.ndarray,
    records: list[dict[str, Any]],
    sensors: dict[str, Any],
) -> None:
    scale = 5
    image = Image.fromarray(_colorize(ground_dem)[::-1], mode="RGB").resize(
        (ground_dem.shape[1] * scale, ground_dem.shape[0] * scale), NEAREST_RESAMPLE
    )
    draw = ImageDraw.Draw(image)
    resolution = float(sensors["road"]["dem_resolution_m"])
    x0 = -float(sensors["road"]["length_m"]) / 2.0 + resolution / 2.0
    y0 = -float(sensors["road"]["width_m"]) / 2.0 + resolution / 2.0
    for record in records:
        if record["hit_status"] != "success":
            continue
        x_m, y_m, _ = record["intersection_map_xyz"]
        col = (x_m - x0) / resolution
        row = (y_m - y0) / resolution
        px = int(round((col + 0.5) * scale))
        py = int(round((ground_dem.shape[0] - row - 0.5) * scale))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(0, 255, 255), outline=(0, 0, 0))
    image.save(path)


def save_height_histogram(path: Path, raw: list[float], filtered: list[float], level: float) -> None:
    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = np.asarray(raw, dtype=np.float64)
    if values.size:
        low, high = float(np.min(values)), float(np.max(values))
        padding = max((high - low) * 0.05, 1e-4)
        bins = np.linspace(low - padding, high + padding, 31)
        raw_hist, _ = np.histogram(values, bins=bins)
        filtered_hist, _ = np.histogram(np.asarray(filtered, dtype=np.float64), bins=bins)
        maximum = max(int(np.max(raw_hist)), 1)
        for index, count in enumerate(raw_hist):
            x1 = 60 + index * 26
            x2 = x1 + 22
            y1 = 440 - int(350 * count / maximum)
            draw.rectangle((x1, y1, x2, 440), fill=(180, 200, 230))
            filtered_count = int(filtered_hist[index])
            fy1 = 440 - int(350 * filtered_count / maximum)
            draw.rectangle((x1 + 5, fy1, x2 - 5, 440), fill=(20, 120, 220))
        lx = 60 + int((level - bins[0]) / max(bins[-1] - bins[0], 1e-12) * (30 * 26))
        draw.line((lx, 70, lx, 440), fill=(220, 20, 20), width=3)
        draw.text((60, 25), f"Shoreline Ground DEM heights; estimated level={level:.6f} m", fill="black")
        draw.text((60, 460), f"range [{bins[0]:.6f}, {bins[-1]:.6f}] m; pale=raw, blue=filtered", fill="black")
    image.save(path)


def save_depth_map(path: Path, depth: np.ndarray, mask: np.ndarray) -> None:
    rgb = _colorize(depth, mask)
    rgb[~np.asarray(mask, dtype=bool)] = [242, 242, 242]
    Image.fromarray(rgb[::-1], mode="RGB").resize(
        (depth.shape[1] * 5, depth.shape[0] * 5), NEAREST_RESAMPLE
    ).save(path)


def save_dem_region(path: Path, ground_dem: np.ndarray, mask: np.ndarray) -> None:
    rgb = _colorize(ground_dem)
    selected = np.asarray(mask, dtype=bool)
    rgb[selected] = (0.35 * rgb[selected] + 0.65 * np.asarray([0, 180, 255])).astype(np.uint8)
    Image.fromarray(rgb[::-1], mode="RGB").resize(
        (ground_dem.shape[1] * 5, ground_dem.shape[0] * 5), NEAREST_RESAMPLE
    ).save(path)


def save_camera_comparison(path: Path, observed: np.ndarray, reprojected: np.ndarray) -> None:
    observed_bool = np.asarray(observed, dtype=bool)
    predicted = np.asarray(reprojected) > 127
    left = np.repeat(np.where(observed_bool, 255, 0).astype(np.uint8)[..., None], 3, axis=2)
    middle = np.repeat(np.where(predicted, 255, 0).astype(np.uint8)[..., None], 3, axis=2)
    overlay = np.zeros((*observed_bool.shape, 3), dtype=np.uint8)
    overlay[observed_bool & predicted] = [0, 210, 0]
    overlay[observed_bool & ~predicted] = [0, 0, 255]
    overlay[~observed_bool & predicted] = [255, 0, 0]
    separator = np.full((observed_bool.shape[0], 4, 3), 110, dtype=np.uint8)
    Image.fromarray(np.concatenate((left, separator, middle, separator, overlay), axis=1), mode="RGB").save(path)


def save_intersection_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "sample_index", "pixel_u", "pixel_v", "ray_direction_optical_xyz",
        "ray_direction_map_xyz", "ray_origin_map_xyz", "hit_status", "failure_reason",
        "intersection_map_xyz", "dem_z_m", "intersection_residual_m", "iteration_count",
        "iteration_count_semantics", "distance_from_camera_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for key in ("ray_direction_optical_xyz", "ray_direction_map_xyz", "ray_origin_map_xyz", "intersection_map_xyz"):
                row[key] = json.dumps(row[key], separators=(",", ":")) if row[key] is not None else ""
            writer.writerow(row)


def _camera_edge_touch_ratio(mask: np.ndarray, edge_band: int) -> float:
    boundary = extract_boundary_mask(np.asarray(mask, dtype=bool))
    rows, cols = np.where(boundary)
    if not rows.size:
        return 0.0
    touches = (
        (rows < edge_band)
        | (rows >= mask.shape[0] - edge_band)
        | (cols < edge_band)
        | (cols >= mask.shape[1] - edge_band)
    )
    return float(np.count_nonzero(touches) / rows.size)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.project_root.expanduser().resolve()
    output_dir = _absolute(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semantics = prediction_semantics()
    mapping = load_yaml(_absolute(root, args.mapping_config), "water_surface_aware_mapping")
    gate_config = load_yaml(_absolute(root, args.gate_config), "water_surface_aware_quality_gate")
    inputs = load_prediction_inputs(
        _absolute(root, args.input_dir),
        _absolute(root, args.source_image),
        _absolute(root, args.ground_dem),
        _absolute(root, args.sensors_config),
    )
    write_json(output_dir / "input_validation.json", inputs["validation"])

    ray_records, ray_summary = intersect_pixel_rays(
        inputs["sampled_shoreline_xy"], inputs["ground_dem"], inputs["sensors"], mapping, "shoreline"
    )
    ray_diagnostics = {
        **semantics,
        **ray_summary,
        "sampled_shoreline_ray_count": ray_summary["total_ray_count"],
        "shoreline_intersection_success_rate": ray_summary["intersection_success_ratio"],
        "camera_mask_edge_touch_ratio": _camera_edge_touch_ratio(
            inputs["mask"], max(1, int(mapping.get("image_edge_band_px", 2)))
        ),
    }
    save_intersection_csv(output_dir / "shoreline_ray_intersections.csv", ray_records)
    write_json(
        output_dir / "shoreline_ray_intersections.json",
        {**semantics, "diagnostics": ray_diagnostics, "records": ray_records},
    )
    save_shoreline_on_dem(
        output_dir / "shoreline_intersections_on_dem.png", inputs["ground_dem"], ray_records, inputs["sensors"]
    )

    level, water_level = estimate_level_from_ray_records(ray_records, mapping["shoreline_water_level"])
    write_json(output_dir / "water_level_estimation.json", water_level)
    save_height_histogram(
        output_dir / "shoreline_height_distribution.png",
        water_level["raw_height_samples_m"],
        water_level["filtered_height_samples_m"],
        level,
    )

    seed_pixels, seed_sources = deterministic_seed_pixels(inputs["mask"], inputs["positive_points_xy"])
    seed_rays, seed_ray_summary = intersect_pixel_rays(
        seed_pixels, inputs["ground_dem"], inputs["sensors"], mapping, "seed"
    )
    for record, source in zip(seed_rays, seed_sources):
        record.update(source)
    seed_mask, mapped_seed_records = ray_hits_to_dem_seed_mask(
        seed_rays, inputs["ground_dem"].shape, inputs["sensors"]
    )
    predicted_mask, reconstruction = reconstruct_seed_connected_lowland(
        inputs["ground_dem"], level, seed_mask, mapping["reconstruction"], inputs["mask"], inputs["sensors"]
    )
    reconstruction.update({
        "camera_seed_pixel_count": int(seed_pixels.shape[0]),
        "successful_seed_ray_count": seed_ray_summary["successful_intersection_count"],
        "seed_ray_success_ratio": seed_ray_summary["intersection_success_ratio"],
        "seed_ray_failure_reasons": seed_ray_summary["intersection_failure_reasons"],
        "seed_records": mapped_seed_records,
    })

    cell_size = float(inputs["sensors"]["road"]["dem_resolution_m"])
    depth, valid_water, water_result = invert_depth_from_ground_dem(
        inputs["ground_dem"], predicted_mask, level, cell_size
    )
    np.save(output_dir / "predicted_dem_water_mask.npy", predicted_mask)
    np.save(output_dir / "predicted_depth_map.npy", depth)
    save_depth_map(output_dir / "predicted_depth_map.png", depth, valid_water)
    save_dem_region(output_dir / "predicted_dem_water_region.png", inputs["ground_dem"], predicted_mask)

    reprojected, projection = reproject_water_surface(predicted_mask, level, inputs["sensors"])
    Image.fromarray(reprojected, mode="L").save(output_dir / "camera_reprojected_mask.png")
    consistency = extended_reprojection_consistency(inputs["mask"], reprojected, projection)
    consistency.update({
        "candidate_basin_count": reconstruction["candidate_basin_count"],
        "seed_validity": reconstruction["seed_valid"],
    })
    save_camera_comparison(
        output_dir / "camera_reprojection_comparison.png", inputs["mask"], reprojected
    )

    required = [
        output_dir / "input_validation.json",
        output_dir / "shoreline_ray_intersections.csv",
        output_dir / "shoreline_ray_intersections.json",
        output_dir / "shoreline_intersections_on_dem.png",
        output_dir / "shoreline_height_distribution.png",
        output_dir / "water_level_estimation.json",
        output_dir / "predicted_dem_water_mask.npy",
        output_dir / "predicted_depth_map.npy",
        output_dir / "predicted_depth_map.png",
        output_dir / "predicted_dem_water_region.png",
        output_dir / "camera_reprojected_mask.png",
        output_dir / "camera_reprojection_comparison.png",
    ]
    gate = evaluate_water_surface_aware_quality_gate(
        ray_diagnostics,
        water_level,
        reconstruction,
        consistency,
        water_result,
        depth,
        gate_config,
        required,
    )
    readiness = (
        "ready"
        if gate["status"] == "pass"
        else "reject"
        if ray_summary["successful_intersection_count"] == 0 or not reconstruction["seed_valid"]
        else "diagnostic_only"
    )
    gate_document = {
        **gate,
        "phase2b_gate_area_volume_semantics": gate.get("area_volume_semantics"),
        **semantics,
        "quality_status": gate["status"],
        "geometry_diagnostic_readiness": readiness,
        "eligible_for_formal_s5_s8": False,
    }
    write_json(output_dir / "prediction_side_quality_gate.json", gate_document)

    result = {
        **semantics,
        "algorithm_version": "phase2d_c3c_v1",
        "prediction_inputs": [
            "manual_prompted_sam2_selected_component_mask",
            "manual_prompted_outer_shoreline_128_xy",
            "manual_positive_points",
            "dry_baseline_ground_dem",
            "camera_intrinsics_and_extrinsics_from_phase2b_sensors_config",
        ],
        "ray_diagnostics": ray_diagnostics,
        "water_level_estimation": water_level,
        "seed_diagnostics": reconstruction,
        "water_result": water_result,
        "self_consistency": consistency,
        "prediction_side_quality_gate": gate_document,
        "quality_status": gate["status"],
        "geometry_diagnostic_readiness": readiness,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "sam2_shoreline_geometry_result.json", result)
    report = f"""# Phase 2D-C-3C SAM 2 岸线几何诊断

本结果来自人工提示 SAM 2 可见水域候选和 dry Ground DEM，仅为离线预测侧几何诊断；不是自动或权威积水测量。

- 岸线射线：{ray_summary['successful_intersection_count']} / {ray_summary['total_ray_count']} 成功
- 估计水位：{level:.9f} m
- 过滤样本：{water_level['inlier_count']} inliers，{water_level['outlier_count']} outliers
- Camera seed：{reconstruction['successful_seed_ray_count']} / {reconstruction['camera_seed_pixel_count']} 成功
- candidate basins：{reconstruction['candidate_basin_count']}，selected：{reconstruction['selected_basin_count']}
- 诊断面积：{water_result['water_area_m2']:.6f} m²
- 诊断体积：{water_result['water_volume_m3']:.9f} m³
- mean/median/max depth：{water_result['mean_depth_cm']:.4f} / {water_result['median_depth_cm']:.4f} / {water_result['max_depth_cm']:.4f} cm
- Camera reprojection IoU：{consistency['camera_reprojection_iou']:.6f}
- boundary P50/P95：{consistency['boundary_reprojection_p50_px']} / {consistency['boundary_reprojection_p95_px']} px
- quality status：{gate['status']}
- geometry diagnostic readiness：{readiness}
- authoritative：false
- ground_truth_used：false
- eligible_for_formal_s5_s8：false
"""
    (output_dir / "sam2_shoreline_geometry_report.md").write_text(report, encoding="utf-8")
    log = {
        "status": "completed",
        "elapsed_seconds": result["elapsed_seconds"],
        "output_dir": str(output_dir),
        "ground_truth_used": False,
        "quality_status": gate["status"],
    }
    (output_dir / "run_log.txt").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except Exception as error:
        root = args.project_root.expanduser().resolve()
        output_dir = _absolute(root, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_log.txt").write_text(
            json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps({
        "quality_status": result["quality_status"],
        "geometry_diagnostic_readiness": result["geometry_diagnostic_readiness"],
        "estimated_water_level_m": result["water_level_estimation"]["estimated_water_level_m"],
        "camera_reprojection_iou": result["self_consistency"]["camera_reprojection_iou"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
