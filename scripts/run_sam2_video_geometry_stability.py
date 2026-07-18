#!/usr/bin/env python3
"""Run prediction-side C3C geometry across frozen SAM 2 video masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.sam2_shoreline_geometry_adapter import load_yaml  # noqa: E402
from src.fusion.sam2_video_geometry_pipeline import (  # noqa: E402
    run_video_frame_geometry,
    summarize_video_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--config-key", default="phase2d_c7_video_pilot")
    parser.add_argument("--propagation-root", type=Path, required=True)
    parser.add_argument(
        "--ground-dem",
        type=Path,
        default=Path("data/simulation/sim_dry_baseline_001/ground_truth/ground_dem_gt.npy"),
    )
    parser.add_argument("--sensors-config", type=Path, default=Path("simulation/config/sensors.yaml"))
    parser.add_argument("--mapping-config", type=Path, default=Path("configs/water_surface_aware_mapping.yaml"))
    parser.add_argument("--gate-config", type=Path, default=Path("configs/water_surface_aware_quality_gate.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_frozen_prediction_inputs(
    root: Path,
    propagation_root: Path,
    samples: list[dict[str, Any]],
    window_start: int,
    window_end: int,
    anchor: int,
) -> dict[str, Any]:
    """Verify all frozen inputs before creating any geometry output."""
    expected = list(range(window_start, window_end + 1))
    verified: dict[str, Any] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        directory = propagation_root / sample_id
        summary_path = directory / "video_propagation_summary.json"
        metrics_path = directory / "frame_metrics.json"
        prompt_path = root / sample["prompt_path"]
        summary = read_json(summary_path)
        metrics = read_json(metrics_path)
        prompt = read_json(prompt_path)
        if summary.get("ground_truth_used") is not False or prompt.get("ground_truth_used") is not False:
            raise ValueError(f"Non-GT-free provenance for {sample_id}")
        if summary.get("sam2_video_propagation_completed") is not True:
            raise ValueError(f"Frozen SAM2 propagation incomplete for {sample_id}")
        if [summary["window_start"], summary["window_end"], summary["anchor_frame_index"]] != [
            window_start, window_end, anchor
        ]:
            raise ValueError(f"Frozen frame window mismatch for {sample_id}")
        if sha256_file(prompt_path) != summary["prompt_sha256"]:
            raise ValueError(f"Frozen prompt SHA-256 mismatch for {sample_id}")
        if prompt.get("prompt_quality_status") == "reject":
            raise ValueError(f"Rejected prompt cannot enter geometry for {sample_id}")
        source_records = summary["source_frame_records"]
        if len(metrics) != len(expected) or len(source_records) != len(expected):
            raise ValueError(f"Frozen frame count mismatch for {sample_id}")
        frames = []
        frames_dir = root / sample["frames_dir"]
        for frame_index, row, source in zip(expected, metrics, source_records):
            if int(row["original_frame_index"]) != frame_index or int(source["original_frame_index"]) != frame_index:
                raise ValueError(f"Frozen frame ordering mismatch for {sample_id}")
            mask_path = directory / "masks_npy" / f"frame_{frame_index:06d}.npy"
            image_path = frames_dir / f"frame_{frame_index:06d}.png"
            if sha256_file(mask_path) != row["mask_sha256"]:
                raise ValueError(f"Frozen mask SHA-256 mismatch for {sample_id}/{frame_index}")
            if sha256_file(image_path) != source["source_sha256"]:
                raise ValueError(f"Frozen RGB SHA-256 mismatch for {sample_id}/{frame_index}")
            frames.append({
                "frame_index": frame_index,
                "mask_path": str(mask_path),
                "mask_sha256": row["mask_sha256"],
                "image_path": str(image_path),
                "image_sha256": source["source_sha256"],
            })
        verified[sample_id] = {
            "verified": True,
            "verified_before_geometry": True,
            "summary_sha256": sha256_file(summary_path),
            "frame_metrics_sha256": sha256_file(metrics_path),
            "prompt_path": str(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
            "positive_points_xy": prompt["positive_points_xy"],
            "frame_count": len(frames),
            "frames": frames,
        }
    return verified


def compact_frame_result(frame_index: int, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not result["available"]:
        row = {
            "frame_index": frame_index,
            "geometry_available": False,
            "quality_status": "reject",
            "gate_reasons": result["prediction_side_quality_gate"]["reasons"],
            "failure_reason": result["failure_reason"],
        }
        return row, {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    topology = result["topology"]
    rays = result["ray_diagnostics"]
    level = result["water_level_estimation"]
    seeds = result["seed_diagnostics"]
    water = result["water_result"]
    consistency = result["self_consistency"]
    gate = result["prediction_side_quality_gate"]
    row = {
        "frame_index": frame_index,
        "geometry_available": True,
        "component_count": topology["component_count"],
        "selected_component_pixels": topology["selected_component_pixels"],
        "excluded_fragment_pixels": topology["excluded_fragment_pixels"],
        "full_shoreline_point_count": topology["full_shoreline_point_count"],
        "shoreline_intersection_success_rate": rays["shoreline_intersection_success_rate"],
        "estimated_water_level_m": level["estimated_water_level_m"],
        "raw_shoreline_mad_m": level["raw_height_statistics"]["MAD_m"],
        "raw_shoreline_iqr_m": level["raw_height_statistics"]["IQR_m"],
        "filtered_shoreline_sample_count": level["inlier_count"],
        "filtered_shoreline_mad_m": level["filtered_height_statistics"]["MAD_m"],
        "filtered_shoreline_iqr_m": level["filtered_height_statistics"]["IQR_m"],
        "seed_ray_success_ratio": seeds["seed_ray_success_ratio"],
        "candidate_basin_count": seeds["candidate_basin_count"],
        "selected_basin_count": seeds["selected_basin_count"],
        "unobserved_candidate_basin_count": seeds["unobserved_candidate_basin_count"],
        "ambiguous_candidate_basin_count": seeds["ambiguous_candidate_basin_count"],
        "water_area_m2": water["water_area_m2"],
        "water_volume_m3": water["water_volume_m3"],
        "mean_depth_cm": water["mean_depth_cm"],
        "median_depth_cm": water["median_depth_cm"],
        "max_depth_cm": water["max_depth_cm"],
        "camera_reprojection_iou": consistency["camera_reprojection_iou"],
        "boundary_reprojection_p50_px": consistency["boundary_reprojection_p50_px"],
        "boundary_reprojection_p95_px": consistency["boundary_reprojection_p95_px"],
        "outer_boundary_reprojection_p95_px": consistency["outer_boundary_reprojection_p95_px"],
        "quality_status": gate["status"],
        "gate_reasons": gate["reasons"],
        "global_estimate_status": gate["global_estimate_status"],
        "observable_region_result_valid": gate["observable_region_result_valid"],
        "ground_truth_used": False,
        "eligible_for_downstream": False,
    }
    detail = {
        key: value
        for key, value in result.items()
        if key not in {"predicted_dem_mask", "predicted_depth_m", "reprojected_camera_mask"}
    }
    return row, detail


def _save_chart(rows: list[dict[str, Any]], field: str, path: Path, title: str) -> None:
    valid = [row for row in rows if row["geometry_available"] and row.get(field) is not None]
    width, height = 1000, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 90, 35, 55, 70
    if not valid:
        draw.text((30, 30), f"{title}: unavailable", fill="black")
        image.save(path)
        return
    xs = np.asarray([row["frame_index"] for row in valid], dtype=np.float64)
    ys = np.asarray([row[field] for row in valid], dtype=np.float64)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    padding = max((y_max - y_min) * 0.08, 1e-9)
    y_min, y_max = y_min - padding, y_max + padding

    def point(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_min) / max(x_max - x_min, 1.0) * (width - left - right))
        py = top + int((y_max - y) / max(y_max - y_min, 1e-12) * (height - top - bottom))
        return px, py

    draw.rectangle((left, top, width - right, height - bottom), outline=(50, 50, 50), width=2)
    for fraction in np.linspace(0.0, 1.0, 6):
        value = y_min + fraction * (y_max - y_min)
        _, py = point(x_min, value)
        draw.line((left, py, width - right, py), fill=(225, 225, 225))
        draw.text((8, py - 7), f"{value:.5f}", fill=(30, 30, 30))
    points = [point(float(x), float(y)) for x, y in zip(xs, ys)]
    draw.line(points, fill=(35, 115, 220), width=3)
    for px, py in points:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(35, 115, 220))
    draw.text((left, 12), title, fill=(20, 20, 20))
    draw.text((width // 2 - 40, height - 28), "Frame index", fill=(20, 20, 20))
    image.save(path)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    root = args.project_root.expanduser().resolve()
    propagation_root = args.propagation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite geometry stability output: {output_root}")
    config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))[args.config_key]
    samples = config["samples"]
    window_start = int(config["window_start"])
    window_end = int(config["window_end"])
    anchor = int(config["anchor_frame_index"])
    verified = verify_frozen_prediction_inputs(
        root, propagation_root, samples, window_start, window_end, anchor
    )
    ground_dem_path = root / args.ground_dem
    sensors_path = root / args.sensors_config
    ground_dem = np.load(ground_dem_path, allow_pickle=False).astype(np.float32)
    sensors = load_yaml(sensors_path)
    mapping = load_yaml(root / args.mapping_config, "water_surface_aware_mapping")
    gate_config = load_yaml(root / args.gate_config, "water_surface_aware_quality_gate")
    if not np.isfinite(ground_dem).all():
        raise ValueError("dry Ground DEM contains NaN or Inf")

    output_root.mkdir(parents=True)
    dataset_summary: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        sample_dir = output_root / sample_id
        sample_dir.mkdir()
        positives = np.asarray(verified[sample_id]["positive_points_xy"], dtype=np.float64)
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for frame in verified[sample_id]["frames"]:
            frame_index = int(frame["frame_index"])
            mask = np.load(frame["mask_path"], allow_pickle=False).astype(bool)
            result = run_video_frame_geometry(mask, positives, ground_dem, sensors, mapping, gate_config)
            row, detail = compact_frame_result(frame_index, result)
            row.update({
                "sample_id": sample_id,
                "is_anchor_frame": frame_index == anchor,
                "mask_sha256": frame["mask_sha256"],
            })
            detail.update({"frame_index": frame_index, "sample_id": sample_id})
            rows.append(row)
            details.append(detail)
            if frame_index == anchor and result["available"]:
                np.save(sample_dir / "anchor_predicted_dem_mask.npy", result["predicted_dem_mask"])
                np.save(sample_dir / "anchor_predicted_depth_m.npy", result["predicted_depth_m"])
                Image.fromarray(
                    np.where(result["reprojected_camera_mask"], 255, 0).astype(np.uint8), mode="L"
                ).save(sample_dir / "anchor_reprojected_camera_mask.png")
        summary = summarize_video_geometry(rows, anchor)
        summary.update({
            "sample_id": sample_id,
            "role": sample["role"],
            "frozen_input_verification": verified[sample_id],
        })
        dataset_summary[sample_id] = summary
        all_rows.extend(rows)
        write_json(sample_dir / "per_frame_geometry.json", details)
        write_json(sample_dir / "per_frame_geometry_summary.json", rows)
        write_json(sample_dir / "sequence_geometry_stability.json", summary)
        with (sample_dir / "per_frame_geometry.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _save_chart(rows, "estimated_water_level_m", sample_dir / "water_level_over_time.png", "Estimated water level")
        _save_chart(rows, "water_area_m2", sample_dir / "area_over_time.png", "Camera-visible candidate area")
        _save_chart(rows, "water_volume_m3", sample_dir / "volume_over_time.png", "Camera-visible candidate volume")
        _save_chart(rows, "max_depth_cm", sample_dir / "max_depth_over_time.png", "Maximum depth")

    final = {
        "protocol_version": "phase2d_c7_video_geometry_stability_v1",
        "input_config_key": args.config_key,
        "sample_count": len(samples),
        "frame_count": len(all_rows),
        "window_start": window_start,
        "window_end": window_end,
        "anchor_frame_index": anchor,
        "algorithm": "existing_phase2d_c3c_geometry_with_external_main_shoreline_per_frame",
        "ground_dem_path": str(ground_dem_path),
        "ground_dem_sha256": sha256_file(ground_dem_path),
        "sensors_config_path": str(sensors_path),
        "sensors_config_sha256": sha256_file(sensors_path),
        "mapping_config_sha256": sha256_file(root / args.mapping_config),
        "gate_config_sha256": sha256_file(root / args.gate_config),
        "gate_thresholds_modified": False,
        "ground_truth_used": False,
        "sam2_rerun_count": 0,
        "prompt_modified": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "sequences": dataset_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_root / "geometry_stability_summary.json", final)
    with (output_root / "per_frame_geometry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output_root / "per_frame_geometry_summary.json", all_rows)
    (output_root / "run_log.txt").write_text(
        json.dumps({
            "status": "completed",
            "frame_count": len(all_rows),
            "elapsed_seconds": final["elapsed_seconds"],
            "ground_truth_used": False,
            "sam2_rerun_count": 0,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "frame_count": len(all_rows),
        "elapsed_seconds": final["elapsed_seconds"],
        "quality_status_counts": {
            sample_id: summary["quality_status_counts"] for sample_id, summary in dataset_summary.items()
        },
        "output_root": str(output_root),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
