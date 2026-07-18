#!/usr/bin/env python3
"""Independently evaluate frozen C7 video geometry scalar outputs against GT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_sam2_shoreline_geometry_gt import (  # noqa: E402
    load_ground_truth_evaluation_inputs,
)
from src.evaluation.evaluate_sam2_video_geometry_gt import (  # noqa: E402
    derive_camera_visible_basin_ground_truth,
    evaluate_frozen_scalar_frame,
    summarize_scalar_evaluation,
)
from src.fusion.sam2_shoreline_geometry_adapter import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--sensors-config", type=Path, default=Path("simulation/config/sensors.yaml"))
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


def verify_frozen_geometry_outputs(
    geometry_root: Path,
    samples: list[dict[str, Any]],
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    """Freeze and verify every scalar prediction artifact before any GT read."""
    dataset_path = geometry_root / "geometry_stability_summary.json"
    dataset = read_json(dataset_path)
    if dataset.get("ground_truth_used") is not False or dataset.get("sam2_rerun_count") != 0:
        raise ValueError("C7-3 dataset provenance is not prediction-only")
    if dataset.get("gate_thresholds_modified") is not False or dataset.get("frame_count") != 123:
        raise ValueError("C7-3 dataset protocol differs from the frozen pilot")
    expected = list(range(window_start, window_end + 1))
    verified: dict[str, Any] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        directory = geometry_root / sample_id
        paths = {
            "rows": directory / "per_frame_geometry_summary.json",
            "details": directory / "per_frame_geometry.json",
            "sequence_summary": directory / "sequence_geometry_stability.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing frozen C7-3 outputs: {missing}")
        rows = read_json(paths["rows"])
        details = read_json(paths["details"])
        sequence_summary = read_json(paths["sequence_summary"])
        if len(rows) != len(expected) or len(details) != len(expected):
            raise ValueError(f"Frozen C7-3 frame count mismatch for {sample_id}")
        if [int(row["frame_index"]) for row in rows] != expected:
            raise ValueError(f"Frozen C7-3 frame order mismatch for {sample_id}")
        if any(row.get("ground_truth_used") is not False for row in rows if row["geometry_available"]):
            raise ValueError(f"Frozen C7-3 row has invalid GT provenance for {sample_id}")
        if sequence_summary.get("ground_truth_used") is not False:
            raise ValueError(f"Frozen C7-3 sequence summary has invalid GT provenance for {sample_id}")
        verified[sample_id] = {
            "verified_before_ground_truth_read": True,
            "frame_count": len(rows),
            "rows": rows,
            "hashes": {name: sha256_file(path) for name, path in paths.items()},
            "paths": {name: str(path) for name, path in paths.items()},
        }
    return {
        "dataset_summary_path": str(dataset_path),
        "dataset_summary_sha256": sha256_file(dataset_path),
        "all_samples_verified_before_ground_truth_read": True,
        "samples": verified,
    }


def flatten_row(sample_id: str, row: dict[str, Any]) -> dict[str, Any]:
    if not row["evaluation_available"]:
        return {
            "sample_id": sample_id,
            "frame_index": row["frame_index"],
            "evaluation_available": False,
            "failure_reason": row["failure_reason"],
        }
    return {
        "sample_id": sample_id,
        "frame_index": row["frame_index"],
        "evaluation_available": True,
        "prediction_side_quality_status": row["prediction_side_quality_status"],
        "prediction_global_estimate_status": row["prediction_global_estimate_status"],
        "prediction_observable_region_result_valid": row["prediction_observable_region_result_valid"],
        "water_level_predicted_m": row["water_level"]["predicted"],
        "water_level_gt_m": row["water_level"]["ground_truth"],
        "water_level_signed_error_cm": row["water_level"]["signed_error_cm"],
        "water_level_absolute_error_cm": row["water_level"]["absolute_error_cm"],
        "water_level_within_3cm": row["water_level"]["within_project_3cm_target"],
        "visible_area_predicted_m2": row["area_camera_visible_main_basin_m2"]["predicted"],
        "visible_area_gt_m2": row["area_camera_visible_main_basin_m2"]["ground_truth"],
        "visible_area_relative_error": row["area_camera_visible_main_basin_m2"]["relative_error"],
        "global_area_gt_m2": row["area_global_scene_m2"]["ground_truth"],
        "global_area_relative_error": row["area_global_scene_m2"]["relative_error"],
        "visible_volume_predicted_m3": row["volume_camera_visible_main_basin_m3"]["predicted"],
        "visible_volume_gt_m3": row["volume_camera_visible_main_basin_m3"]["ground_truth"],
        "visible_volume_relative_error": row["volume_camera_visible_main_basin_m3"]["relative_error"],
        "global_volume_gt_m3": row["volume_global_scene_m3"]["ground_truth"],
        "global_volume_relative_error": row["volume_global_scene_m3"]["relative_error"],
        "visible_mean_depth_absolute_error_cm": row["mean_depth_camera_visible_main_basin_cm"]["absolute_error"],
        "global_max_depth_absolute_error_cm": row["max_depth_global_scene_cm"]["absolute_error"],
        "per_cell_metrics_available": False,
        "ground_truth_used_for_prediction": False,
        "eligible_for_downstream": False,
    }


def save_error_chart(rows: list[dict[str, Any]], field: str, path: Path, title: str) -> None:
    valid = [row for row in rows if row["evaluation_available"]]
    width, height = 1000, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 90, 35, 55, 70
    xs = np.asarray([row["frame_index"] for row in valid], dtype=np.float64)
    ys = np.asarray([row[field] for row in valid], dtype=np.float64)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = 0.0, float(max(np.max(ys) * 1.08, 1e-6))

    def point(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_min) / max(x_max - x_min, 1.0) * (width - left - right))
        py = top + int((y_max - y) / max(y_max - y_min, 1e-12) * (height - top - bottom))
        return px, py

    draw.rectangle((left, top, width - right, height - bottom), outline=(50, 50, 50), width=2)
    for fraction in np.linspace(0.0, 1.0, 6):
        value = fraction * y_max
        _, py = point(x_min, value)
        draw.line((left, py, width - right, py), fill=(225, 225, 225))
        draw.text((8, py - 7), f"{value:.4f}", fill=(30, 30, 30))
    points = [point(float(x), float(y)) for x, y in zip(xs, ys)]
    draw.line(points, fill=(35, 115, 220), width=3)
    for px, py in points:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(35, 115, 220))
    draw.text((left, 12), title, fill=(20, 20, 20))
    draw.text((width // 2 - 40, height - 28), "Frame index", fill=(20, 20, 20))
    image.save(path)


def write_report(output_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 2D-C-7-4 连续几何独立 GT 评价",
        "",
        "所有冻结 prediction 文件均在首次读取 GT 前验证。没有重新运行 SAM2 或几何 prediction，也没有使用 GT 修改 gate。",
        "",
        "| 序列 | 水位绝对误差中位数（cm） | 3 cm 内 | 可见面积相对误差中位数 | 全局面积相对误差中位数 | 可见体积相对误差中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sample_id, item in summary["sequences"].items():
        lines.append(
            f"| {sample_id} | {item['water_level_absolute_error_cm']['median']:.4f} | "
            f"{item['water_level_within_3cm_count']}/{item['frame_count']} | "
            f"{item['visible_area_relative_error']['median']:.2%} | "
            f"{item['global_area_relative_error']['median']:.2%} | "
            f"{item['visible_volume_relative_error']['median']:.2%} |"
        )
    lines.extend([
        "",
        "逐栅格 DEM mask IoU 和 depth MAE/RMSE/bias 标记为 unavailable：C7-3 只冻结了每帧标量，未保存每帧 raster；本评价没有重新运行 prediction 来补造数组。",
        "",
        "结果仅用于离线研究，不是正式 quality gate，也不允许进入 S5-S8。",
        "",
    ])
    (output_root / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    geometry_root = args.geometry_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C7-4 evaluation output: {output_root}")
    config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))["phase2d_c7_video_pilot"]
    samples = config["samples"]
    window_start = int(config["window_start"])
    window_end = int(config["window_end"])

    # Protocol boundary: all prediction files are verified before the first GT loader call.
    frozen = verify_frozen_geometry_outputs(
        geometry_root, samples, window_start, window_end
    )

    sensors = load_yaml(root / args.sensors_config)
    output_root.mkdir(parents=True)
    all_flat: list[dict[str, Any]] = []
    sequence_summaries: dict[str, Any] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        sequence_relative = str(Path(sample["frames_dir"]).parent)
        gt = load_ground_truth_evaluation_inputs(root, sample["case_id"], sequence_relative)
        basin_truth = derive_camera_visible_basin_ground_truth(gt, sensors)
        evaluated = [
            evaluate_frozen_scalar_frame(row, gt, basin_truth)
            for row in frozen["samples"][sample_id]["rows"]
        ]
        flat = [flatten_row(sample_id, row) for row in evaluated]
        summary = summarize_scalar_evaluation(evaluated)
        summary.update({
            "sample_id": sample_id,
            "role": sample["role"],
            "ground_truth_validation": gt["validation"],
            "basin_ground_truth": basin_truth,
            "frozen_prediction_verification": {
                key: value for key, value in frozen["samples"][sample_id].items() if key != "rows"
            },
        })
        sequence_summaries[sample_id] = summary
        all_flat.extend(flat)
        sample_dir = output_root / sample_id
        sample_dir.mkdir()
        write_json(sample_dir / "per_frame_scalar_evaluation.json", evaluated)
        write_json(sample_dir / "sequence_evaluation_summary.json", summary)
        with (sample_dir / "per_frame_scalar_evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        save_error_chart(flat, "water_level_absolute_error_cm", sample_dir / "water_level_error_cm.png", "Water-level absolute error (cm)")
        save_error_chart(flat, "visible_area_relative_error", sample_dir / "visible_area_relative_error.png", "Camera-visible main-basin area relative error")
        save_error_chart(flat, "visible_volume_relative_error", sample_dir / "visible_volume_relative_error.png", "Camera-visible main-basin volume relative error")

    summary = {
        "protocol_version": "phase2d_c7_video_geometry_scalar_gt_evaluation_v1",
        "sample_count": len(samples),
        "frame_count": len(all_flat),
        "all_frozen_predictions_verified_before_first_ground_truth_read": True,
        "frozen_dataset_summary_sha256": frozen["dataset_summary_sha256"],
        "sam2_rerun_count": 0,
        "geometry_prediction_recomputed_count": 0,
        "prompt_modified": False,
        "gate_thresholds_modified": False,
        "ground_truth_used_for_prediction": False,
        "per_cell_metrics_available": False,
        "per_cell_metrics_unavailable_reason": "per-frame prediction rasters were not frozen by C7-3",
        "authoritative": False,
        "eligible_for_downstream": False,
        "sequences": sequence_summaries,
    }
    write_json(output_root / "evaluation_summary.json", summary)
    write_json(output_root / "per_frame_scalar_evaluation.json", all_flat)
    with (output_root / "per_frame_scalar_evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_flat[0]))
        writer.writeheader()
        writer.writerows(all_flat)
    write_report(output_root, summary)
    (output_root / "run_log.txt").write_text(
        json.dumps({
            "status": "completed",
            "frame_count": len(all_flat),
            "sam2_rerun_count": 0,
            "geometry_prediction_recomputed_count": 0,
            "ground_truth_used_for_prediction": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "frame_count": len(all_flat),
        "within_3cm": {
            sample_id: item["water_level_within_3cm_count"]
            for sample_id, item in sequence_summaries.items()
        },
        "output_root": str(output_root),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
