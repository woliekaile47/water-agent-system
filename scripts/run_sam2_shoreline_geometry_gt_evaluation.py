#!/usr/bin/env python3
"""Run independent GT evaluation of saved Phase 2D-C-3C prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_sam2_shoreline_geometry_gt import (
    area_volume_metrics,
    binary_mask_metrics,
    boundary_metrics,
    depth_metrics,
    error_source_analysis,
    gt_shoreline_geometry_counterfactual,
    load_ground_truth_evaluation_inputs,
    shoreline_membership_analysis,
    water_level_metrics,
)
from src.fusion.sam2_shoreline_geometry_adapter import load_yaml, outer_boundary_mask


NEAREST_RESAMPLE = getattr(Image, "Resampling", Image).NEAREST


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/sam2_shoreline_geometry/frame_000059"))
    parser.add_argument("--sam-input-dir", type=Path, default=Path("data/sam2_geometry_inputs/frame_000059"))
    parser.add_argument("--case-id", default="sim_water_5cm_001")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("data/simulation_dynamic/sim_water_5cm_001/heavy/seed_43"),
    )
    parser.add_argument("--sensors-config", type=Path, default=Path("simulation/config/sensors.yaml"))
    parser.add_argument("--mapping-config", type=Path, default=Path("configs/water_surface_aware_mapping.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam2_shoreline_geometry_gt_evaluation/frame_000059"),
    )
    return parser


def _absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _mask_panel(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.where(mask, 255, 0).astype(np.uint8)[..., None], 3, axis=2)


def save_mask_comparison(path: Path, predicted: np.ndarray, truth: np.ndarray) -> None:
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    overlay = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    overlay[prediction & target] = [0, 210, 0]
    overlay[prediction & ~target] = [255, 0, 0]
    overlay[~prediction & target] = [0, 0, 255]
    separator = np.full((prediction.shape[0], 4, 3), 100, dtype=np.uint8)
    Image.fromarray(
        np.concatenate((_mask_panel(prediction), separator, _mask_panel(target), separator, overlay), axis=1),
        mode="RGB",
    ).save(path)


def save_boundary_comparison(path: Path, predicted: np.ndarray, truth: np.ndarray) -> None:
    p = outer_boundary_mask(predicted)
    t = outer_boundary_mask(truth)
    rgb = np.full((*p.shape, 3), 245, dtype=np.uint8)
    rgb[p & ~t] = [255, 0, 0]
    rgb[~p & t] = [0, 200, 0]
    rgb[p & t] = [255, 210, 0]
    Image.fromarray(rgb, mode="RGB").save(path)


def save_dem_comparison(path: Path, predicted: np.ndarray, truth: np.ndarray) -> None:
    p = np.asarray(predicted, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    rgb = np.full((*p.shape, 3), 245, dtype=np.uint8)
    rgb[p & t] = [0, 190, 0]
    rgb[p & ~t] = [255, 40, 40]
    rgb[~p & t] = [30, 80, 255]
    Image.fromarray(rgb[::-1], mode="RGB").resize(
        (p.shape[1] * 5, p.shape[0] * 5), NEAREST_RESAMPLE
    ).save(path)


def save_depth_error(path: Path, predicted: np.ndarray, truth: np.ndarray) -> None:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(error)
    bound = max(float(np.max(np.abs(error[finite]))) if np.any(finite) else 0.0, 1e-9)
    scaled = np.clip(np.nan_to_num(error, nan=0.0) / bound, -1.0, 1.0)
    rgb = np.full((*error.shape, 3), 255, dtype=np.uint8)
    rgb[..., 0] = np.where(scaled > 0, 255, 255 * (1 + scaled)).astype(np.uint8)
    rgb[..., 1] = (255 * (1 - np.abs(scaled))).astype(np.uint8)
    rgb[..., 2] = np.where(scaled < 0, 255, 255 * (1 - scaled)).astype(np.uint8)
    rgb[~finite] = [80, 80, 80]
    Image.fromarray(rgb[::-1], mode="RGB").resize(
        (error.shape[1] * 5, error.shape[0] * 5), NEAREST_RESAMPLE
    ).save(path)


def save_histogram(path: Path, series: list[tuple[str, np.ndarray, tuple[int, int, int]]], title: str) -> None:
    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    all_values = np.concatenate([values[np.isfinite(values)] for _, values, _ in series if np.any(np.isfinite(values))])
    if all_values.size:
        low, high = float(np.min(all_values)), float(np.max(all_values))
        padding = max((high - low) * 0.05, 1e-4)
        bins = np.linspace(low - padding, high + padding, 31)
        histograms = [np.histogram(values[np.isfinite(values)], bins=bins)[0] for _, values, _ in series]
        maximum = max(max(int(np.max(hist)) for hist in histograms), 1)
        for series_index, ((label, _, color), hist) in enumerate(zip(series, histograms)):
            for index, count in enumerate(hist):
                x = 60 + index * 26 + series_index * 7
                y = 430 - int(340 * count / maximum)
                draw.rectangle((x, y, x + 6, 430), fill=color)
            draw.text((60 + series_index * 220, 455), label, fill=color)
        draw.text((60, 20), title, fill="black")
        draw.text((60, 475), f"range [{bins[0]:.6f}, {bins[-1]:.6f}] m", fill="black")
    image.save(path)


def save_water_level_comparison(
    path: Path,
    heights: np.ndarray,
    predicted_level: float,
    true_level: float,
) -> None:
    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = np.asarray(heights, dtype=np.float64)
    low = min(float(np.min(values)), predicted_level, true_level)
    high = max(float(np.max(values)), predicted_level, true_level)
    padding = max((high - low) * 0.05, 1e-4)
    bins = np.linspace(low - padding, high + padding, 31)
    hist, _ = np.histogram(values, bins=bins)
    maximum = max(int(np.max(hist)), 1)
    for index, count in enumerate(hist):
        x = 60 + index * 26
        y = 430 - int(330 * count / maximum)
        draw.rectangle((x, y, x + 20, 430), fill=(160, 190, 230))
    for level, color, label in ((predicted_level, (220, 30, 30), "prediction"), (true_level, (20, 170, 20), "GT")):
        x = 60 + int((level - bins[0]) / (bins[-1] - bins[0]) * 780)
        draw.line((x, 70, x, 430), fill=color, width=4)
        draw.text((x + 4, 50), f"{label}: {level:.6f}", fill=color)
    draw.text((60, 20), "Predicted shoreline Ground DEM heights vs water levels", fill="black")
    image.save(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.project_root.expanduser().resolve()
    prediction_dir = _absolute(root, args.prediction_dir)
    sam_input_dir = _absolute(root, args.sam_input_dir)
    output_dir = _absolute(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_prediction_files = [
        prediction_dir / "predicted_dem_water_mask.npy",
        prediction_dir / "predicted_depth_map.npy",
        prediction_dir / "sam2_shoreline_geometry_result.json",
        prediction_dir / "water_level_estimation.json",
        prediction_dir / "shoreline_ray_intersections.json",
        prediction_dir / "prediction_side_quality_gate.json",
        sam_input_dir / "selected_component_mask.npy",
    ]
    missing = [str(path) for path in required_prediction_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved prediction artifacts: {missing}")
    hashes_before = _directory_hashes(prediction_dir)

    prediction_camera_mask = np.load(sam_input_dir / "selected_component_mask.npy", allow_pickle=False).astype(bool)
    prediction_dem_mask = np.load(prediction_dir / "predicted_dem_water_mask.npy", allow_pickle=False).astype(bool)
    prediction_depth = np.load(prediction_dir / "predicted_depth_map.npy", allow_pickle=False).astype(np.float32)
    prediction_result = _read_json(prediction_dir / "sam2_shoreline_geometry_result.json")
    prediction_level = _read_json(prediction_dir / "water_level_estimation.json")
    ray_document = _read_json(prediction_dir / "shoreline_ray_intersections.json")
    prediction_gate = _read_json(prediction_dir / "prediction_side_quality_gate.json")

    truth = load_ground_truth_evaluation_inputs(root, args.case_id, args.sequence_dir)
    camera = {
        "data_role": "independent_evaluation",
        **binary_mask_metrics(prediction_camera_mask, truth["camera_mask"]),
        "boundary": boundary_metrics(prediction_camera_mask, truth["camera_mask"]),
    }
    _write_json(output_dir / "camera_mask_metrics.json", camera)

    heights = np.asarray(prediction_level["raw_height_samples_m"], dtype=np.float64)
    level = {
        "data_role": "independent_evaluation",
        **water_level_metrics(
            float(prediction_level["estimated_water_level_m"]),
            truth["water_level_m"],
            heights,
            truth["nominal_depth_m"],
        ),
    }
    membership = shoreline_membership_analysis(
        ray_document["records"], truth["camera_mask"], outer_boundary_mask(truth["camera_mask"]), truth["water_level_m"]
    )
    level["predicted_shoreline_membership"] = membership
    _write_json(output_dir / "water_level_metrics.json", level)

    dem_mask = binary_mask_metrics(prediction_dem_mask, truth["dem_mask"])
    depth = depth_metrics(prediction_depth, truth["depth_map"], prediction_dem_mask, truth["dem_mask"])
    predicted_lowland = np.isfinite(truth["ground_dem"]) & (
        truth["ground_dem"] < float(prediction_level["estimated_water_level_m"])
    )
    reconstruction_at_predicted_level = binary_mask_metrics(prediction_dem_mask, predicted_lowland)
    dem_depth = {
        "data_role": "independent_evaluation",
        "dem_mask": dem_mask,
        "depth": depth,
        "reconstruction_consistency_at_saved_predicted_level": reconstruction_at_predicted_level,
        "evaluation_domains": {
            "full_valid_dem_domain": "all cells finite in prediction and GT",
            "predicted_region_domain": "cells selected by saved predicted DEM mask",
            "gt_region_domain": "cells selected by strict DEM water mask GT",
            "overlap_region_domain": "intersection of saved prediction and GT water masks",
        },
    }
    _write_json(output_dir / "dem_depth_metrics.json", dem_depth)

    area_volume = {
        "data_role": "independent_evaluation",
        **area_volume_metrics(
            prediction_result["water_result"],
            truth["water_area_m2"],
            truth["water_volume_m3"],
            truth["depth_map"],
            truth["dem_mask"],
        ),
    }
    _write_json(output_dir / "area_volume_metrics.json", area_volume)

    sensors = load_yaml(_absolute(root, args.sensors_config))
    mapping = load_yaml(_absolute(root, args.mapping_config), "water_surface_aware_mapping")
    gt_counterfactual = gt_shoreline_geometry_counterfactual(
        truth["camera_mask"], truth["ground_dem"], sensors, mapping, truth["water_level_m"]
    )
    residuals = np.asarray([
        abs(float(record["intersection_residual_m"]))
        for record in ray_document["records"]
        if record.get("intersection_residual_m") is not None
    ])
    residual_p95 = float(np.percentile(residuals, 95)) if residuals.size else float("inf")
    source_analysis = error_source_analysis(
        camera,
        camera["boundary"],
        level,
        dem_mask,
        reconstruction_at_predicted_level,
        residual_p95,
        gt_counterfactual,
        prediction_gate,
    )
    source_analysis.update({
        "data_role": "independent_evaluation",
        "ray_dem_absolute_residual_p95_m": residual_p95,
        "gt_shoreline_geometry_counterfactual": gt_counterfactual,
        "predicted_shoreline_membership": membership,
    })
    _write_json(output_dir / "error_source_analysis.json", source_analysis)

    save_mask_comparison(output_dir / "camera_mask_vs_gt.png", prediction_camera_mask, truth["camera_mask"])
    save_boundary_comparison(output_dir / "camera_boundary_vs_gt.png", prediction_camera_mask, truth["camera_mask"])
    save_dem_comparison(output_dir / "dem_mask_vs_gt.png", prediction_dem_mask, truth["dem_mask"])
    save_depth_error(output_dir / "depth_error_map.png", prediction_depth, truth["depth_map"])
    save_histogram(
        output_dir / "depth_distribution_comparison.png",
        [
            ("prediction", prediction_depth[prediction_dem_mask].astype(np.float64), (220, 50, 50)),
            ("GT", truth["depth_map"][truth["dem_mask"]].astype(np.float64), (20, 160, 20)),
        ],
        "Depth distribution comparison",
    )
    save_water_level_comparison(
        output_dir / "water_level_shoreline_height_comparison.png",
        heights,
        float(prediction_level["estimated_water_level_m"]),
        truth["water_level_m"],
    )

    hashes_after = _directory_hashes(prediction_dir)
    unchanged = hashes_before == hashes_after
    if not unchanged:
        raise RuntimeError("Saved prediction artifacts changed during independent evaluation")
    summary = {
        "data_role": "independent_evaluation",
        "case_id": args.case_id,
        "sequence_id": "sim_water_5cm_001/heavy/seed_43",
        "frame_index": 59,
        "ground_truth_validation": truth["validation"],
        "camera_mask": camera,
        "water_level": level,
        "dem_depth": dem_depth,
        "area_volume": area_volume,
        "error_source_analysis": source_analysis,
        "prediction_side_gate_status": prediction_gate["status"],
        "prediction_artifacts_unchanged": unchanged,
        "prediction_hashes_before": hashes_before,
        "prediction_hashes_after": hashes_after,
        "prediction_rerun": False,
        "prediction_modified": False,
        "evaluation_elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "evaluation_summary.json", summary)

    gt_depth = depth["gt_region_domain"]
    report = f"""# Phase 2D-C-3D SAM 2 几何结果独立 Ground Truth 评价

本报告只评价已保存的 Phase 2D-C-3C prediction。没有重新运行或修改 prediction，没有使用 GT 选择候选、岸线、seed、basin 或水位。

## Camera mask

- IoU：{camera['iou']:.6f}
- precision：{camera['precision']:.6f}
- recall：{camera['recall']:.6f}
- F1：{camera['f1']:.6f}
- predicted / GT pixels：{camera['predicted_pixels']} / {camera['gt_pixels']}
- outer boundary P50/P95：{camera['boundary']['symmetric_outer_boundary']['p50_px']:.6f} / {camera['boundary']['symmetric_outer_boundary']['p95_px']:.6f} px

## Water level

- prediction：{level['estimated_water_level_m']:.9f} m
- GT：{level['gt_water_level_m']:.9f} m
- signed error：{level['signed_water_level_error_m']:.9f} m
- absolute error：{level['absolute_water_level_error_cm']:.4f} cm

## DEM mask and depth

- DEM mask IoU：{dem_mask['iou']:.6f}
- DEM precision / recall：{dem_mask['precision']:.6f} / {dem_mask['recall']:.6f}
- GT-region depth MAE：{gt_depth['mae_m']:.9f} m
- GT-region depth RMSE：{gt_depth['rmse_m']:.9f} m
- GT-region depth bias：{gt_depth['bias_m']:.9f} m

## Area and volume

- area error：{area_volume['area_absolute_error_m2']:.6f} m² ({area_volume['area_relative_error']:.2%})
- volume error：{area_volume['volume_absolute_error_m3']:.9f} m³ ({area_volume['volume_relative_error']:.2%})

## Error source

- dominant_error_source：{source_analysis['dominant_error_source']}
- manual prompt scope：{source_analysis['manual_prompt_scope_interpretation']}
- prediction-side reject consistent with GT：{source_analysis['prediction_side_reject_consistent_with_gt_evaluation']}
- prediction artifacts unchanged：{unchanged}

当前结果仍为离线独立评价，不具备 authoritative 或 downstream 语义。
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    (output_dir / "run_log.txt").write_text(
        json.dumps({
            "status": "completed",
            "prediction_rerun": False,
            "prediction_modified": False,
            "prediction_artifacts_unchanged": unchanged,
            "elapsed_seconds": summary["evaluation_elapsed_seconds"],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps({
        "camera_mask_iou": summary["camera_mask"]["iou"],
        "water_level_absolute_error_m": summary["water_level"]["absolute_water_level_error_m"],
        "dem_mask_iou": summary["dem_depth"]["dem_mask"]["iou"],
        "dominant_error_source": summary["error_source_analysis"]["dominant_error_source"],
        "prediction_artifacts_unchanged": summary["prediction_artifacts_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
