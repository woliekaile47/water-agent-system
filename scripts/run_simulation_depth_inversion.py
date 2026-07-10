#!/usr/bin/env python3
"""Independent Phase 2A simulation Camera-mask/DEM inversion CLI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_simulation_depth import load_ground_truth_evaluation_inputs, evaluate_prediction
from src.evaluation.unified_water_quality_gate import evaluate_quality_gate
from src.fusion.project_camera_mask_to_dem import (
    ALGORITHM_VERSION,
    PREDICTION_SOURCE,
    load_prediction_inputs,
    project_camera_mask_to_dem,
)
from src.hydrology.estimate_water_level_from_boundary import estimate_water_level_from_boundary
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem


DEFAULT_CASES = ["sim_water_5cm_001", "sim_water_10cm_001", "sim_water_20cm_001", "sim_water_40cm_001"]


def load_yaml(path: Path, root_key: str) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if root_key not in document:
        raise ValueError(f"Missing {root_key} in {path}")
    return document[root_key]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def metadata(case_id: str, role: str = "prediction") -> dict[str, Any]:
    return {
        "data_role": role,
        "source": PREDICTION_SOURCE,
        "algorithm_version": ALGORITHM_VERSION,
        "case_id": case_id,
        "ground_truth_used_for_evaluation_only": True,
        "eligible_for_downstream": False,
    }


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def save_comparison_figures(
    output_dir: Path,
    predicted_mask: np.ndarray,
    predicted_depth: np.ndarray,
    gt_mask: np.ndarray,
    gt_depth: np.ndarray,
) -> None:
    """Write deterministic raster comparisons without compiled plotting extensions."""
    gt_panel = np.repeat(np.where(gt_mask, 255, 0).astype(np.uint8)[::-1, :, None], 3, axis=2)
    pred_panel = np.repeat(np.where(predicted_mask, 255, 0).astype(np.uint8)[::-1, :, None], 3, axis=2)
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlay[..., 1] = np.where(gt_mask & predicted_mask, 255, 0)
    overlay[..., 0] = np.where(predicted_mask & ~gt_mask, 255, 0)
    overlay[..., 2] = np.where(gt_mask & ~predicted_mask, 255, 0)
    separator = np.full((gt_mask.shape[0], 3, 3), 96, dtype=np.uint8)
    mask_comparison = np.concatenate((gt_panel, separator, pred_panel, separator, overlay[::-1]), axis=1)
    Image.fromarray(mask_comparison, mode="RGB").save(output_dir / "mask_comparison.png")

    finite_values = np.concatenate((gt_depth[np.isfinite(gt_depth)], predicted_depth[np.isfinite(predicted_depth)]))
    vmax = max(float(np.max(finite_values)) if finite_values.size else 0.0, 1e-9)

    def blue_depth(values: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.nan_to_num(values, nan=0.0) / vmax, 0.0, 1.0)
        panel = np.zeros((*values.shape, 3), dtype=np.uint8)
        panel[..., 0] = (235.0 * (1.0 - normalized)).astype(np.uint8)
        panel[..., 1] = (245.0 * (1.0 - 0.65 * normalized)).astype(np.uint8)
        panel[..., 2] = 255
        return panel[::-1]

    difference = predicted_depth.astype(np.float64) - gt_depth.astype(np.float64)
    bound = max(float(np.nanmax(np.abs(difference))) if np.isfinite(difference).any() else 0.0, 1e-9)
    scaled = np.clip(np.nan_to_num(difference, nan=0.0) / bound, -1.0, 1.0)
    difference_panel = np.full((*difference.shape, 3), 255, dtype=np.uint8)
    difference_panel[..., 0] = np.where(scaled > 0, 255, 255.0 * (1.0 + scaled)).astype(np.uint8)
    difference_panel[..., 1] = (255.0 * (1.0 - np.abs(scaled))).astype(np.uint8)
    difference_panel[..., 2] = np.where(scaled < 0, 255, 255.0 * (1.0 - scaled)).astype(np.uint8)
    depth_comparison = np.concatenate(
        (blue_depth(gt_depth), separator, blue_depth(predicted_depth), separator, difference_panel[::-1]), axis=1
    )
    Image.fromarray(depth_comparison, mode="RGB").save(output_dir / "depth_comparison.png")


def run_case(project_root: Path, case_id: str) -> dict[str, Any]:
    projection_config = load_yaml(project_root / "configs" / "simulation_camera_mask_projection.yaml", "camera_mask_projection")
    boundary_config = load_yaml(project_root / "configs" / "boundary_waterline_inversion.yaml", "boundary_waterline_inversion")
    gate_config = load_yaml(project_root / "configs" / "unified_water_quality_gate.yaml", "unified_water_quality_gate")
    output_dir = project_root / "outputs" / "simulation_evaluation" / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prediction stage: no Ground Truth answer loader is reachable before the
    # quality gate has been written.
    inputs = load_prediction_inputs(project_root, case_id, projection_config)
    ground_dem = inputs["ground_dem"]
    camera_mask = inputs["camera_mask"]
    predicted_mask_raw, projection = project_camera_mask_to_dem(
        ground_dem,
        camera_mask,
        inputs["sensors"],
        int(projection_config.get("mask_threshold", 127)),
    )
    water_level, predicted_mask, boundary = estimate_water_level_from_boundary(
        predicted_mask_raw, ground_dem, boundary_config
    )
    cell_size = float(inputs["sensors"]["road"]["dem_resolution_m"])
    predicted_depth, valid_water_mask, water_values = invert_depth_from_ground_dem(
        ground_dem, predicted_mask, water_level, cell_size
    )

    mask_npy = output_dir / "predicted_dem_mask.npy"
    mask_png = output_dir / "predicted_dem_mask.png"
    level_json = output_dir / "predicted_water_level.json"
    depth_npy = output_dir / "predicted_depth_map_m.npy"
    result_json = output_dir / "predicted_water_result.json"
    projection_json = output_dir / "projection_diagnostics.json"
    boundary_json = output_dir / "boundary_diagnostics.json"
    quality_diagnostics_json = output_dir / "quality_diagnostics.json"
    gate_json = output_dir / "quality_gate.json"

    np.save(mask_npy, predicted_mask)
    save_binary_mask(mask_png, predicted_mask)
    np.save(depth_npy, predicted_depth)
    common = {
        **metadata(case_id),
        "prediction_inputs": inputs["prediction_inputs"],
        "prediction_input_paths": inputs["paths"],
    }
    projection_document = {**common, **projection}
    boundary_document = {**common, **boundary}
    level_document = {
        **common,
        "camera_frame": projection["camera_frame"],
        "transform": projection["transform"],
        "water_level_estimation_method": boundary["method"],
        "predicted_water_level_m": water_level,
        "predicted_water_level_cm": water_level * 100.0,
        "quality_gate_status": "pending",
    }
    water_document = {
        **common,
        "quality_gate_status": "pending",
        **water_values,
    }
    quality_diagnostics = {
        **common,
        "projection": projection_document,
        "boundary": boundary_document,
        "depth": water_values,
        "valid_water_mask_cell_count": int(np.count_nonzero(valid_water_mask)),
    }
    write_json(projection_json, projection_document)
    write_json(boundary_json, boundary_document)
    write_json(level_json, level_document)
    write_json(result_json, water_document)
    write_json(quality_diagnostics_json, quality_diagnostics)

    required_prediction_files = [
        mask_npy,
        mask_png,
        level_json,
        depth_npy,
        result_json,
        projection_json,
        boundary_json,
        quality_diagnostics_json,
    ]
    gate = evaluate_quality_gate(
        camera_mask,
        predicted_mask,
        predicted_depth,
        projection,
        boundary,
        water_values,
        gate_config,
        required_prediction_files,
    )
    gate_document = {**common, **gate, "quality_gate_status": gate["status"]}
    write_json(gate_json, gate_document)
    level_document["quality_gate_status"] = gate["status"]
    water_document["quality_gate_status"] = gate["status"]
    projection_document["quality_gate_status"] = gate["status"]
    boundary_document["quality_gate_status"] = gate["status"]
    quality_diagnostics["quality_gate_status"] = gate["status"]
    write_json(level_json, level_document)
    write_json(result_json, water_document)
    write_json(projection_json, projection_document)
    write_json(boundary_json, boundary_document)
    write_json(quality_diagnostics_json, quality_diagnostics)

    # Evaluation stage starts only after prediction and the GT-independent gate
    # are complete. This is the sole call site for answer data.
    ground_truth = load_ground_truth_evaluation_inputs(project_root, case_id)
    evaluation = evaluate_prediction(
        predicted_mask,
        predicted_depth,
        water_document,
        boundary,
        projection,
        ground_truth,
        gate,
    )
    evaluation_document = {
        **metadata(case_id, "evaluation"),
        **evaluation,
        "ground_truth_input_paths": ground_truth["paths"],
    }
    write_json(output_dir / "evaluation_metrics.json", evaluation_document)
    save_comparison_figures(output_dir, predicted_mask, predicted_depth, ground_truth["dem_mask"], ground_truth["depth_map"])
    return {"case_id": case_id, "quality_gate": gate, "evaluation": evaluation_document, "water_result": water_document}


def write_summary(project_root: Path, results: list[dict[str, Any]]) -> None:
    output_root = project_root / "outputs" / "simulation_evaluation"
    fields = [
        "case_id", "quality_gate", "quality_reasons", "mask_iou", "precision", "recall",
        "boundary_f1", "water_level_abs_error_m", "depth_mae_union_m", "depth_rmse_union_m",
        "area_abs_error_m2", "area_relative_error", "volume_abs_error_m3", "volume_relative_error",
        "projection_coverage", "eligible_for_downstream",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        evaluation = result["evaluation"]
        rows.append({
            "case_id": result["case_id"],
            "quality_gate": result["quality_gate"]["status"],
            "quality_reasons": ";".join(result["quality_gate"]["reasons"]),
            "mask_iou": evaluation["mask"]["iou"],
            "precision": evaluation["mask"]["precision"],
            "recall": evaluation["mask"]["recall"],
            "boundary_f1": evaluation["mask"]["boundary_f1_1cell"],
            "water_level_abs_error_m": evaluation["water_level"]["absolute_error_m"],
            "depth_mae_union_m": evaluation["depth_error_domains"]["gt_prediction_union"]["mae_m"],
            "depth_rmse_union_m": evaluation["depth_error_domains"]["gt_prediction_union"]["rmse_m"],
            "area_abs_error_m2": evaluation["area"]["absolute_error_m2"],
            "area_relative_error": evaluation["area"]["relative_error"],
            "volume_abs_error_m3": evaluation["volume"]["absolute_error_m3"],
            "volume_relative_error": evaluation["volume"]["relative_error"],
            "projection_coverage": evaluation["projection_coverage"],
            "eligible_for_downstream": False,
        })
    with (output_root / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Phase 2A simulation evaluation summary", "", "All results are ineligible for S5-S8.", "", "| case | gate | IoU | level error (m) | depth MAE union (m) | area error (m²) | volume error (m³) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['quality_gate']} | {row['mask_iou']:.6f} | "
            f"{row['water_level_abs_error_m']:.6f} | {row['depth_mae_union_m']:.6f} | "
            f"{row['area_abs_error_m2']:.6f} | {row['volume_abs_error_m3']:.6f} |"
        )
    (output_root / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    case_ids = DEFAULT_CASES if args.all else [args.case]
    results = [run_case(root, case_id) for case_id in case_ids]
    if args.all:
        write_summary(root, results)
    for result in results:
        evaluation = result["evaluation"]
        print(
            f"[phase2a] {result['case_id']}: gate={result['quality_gate']['status']} "
            f"iou={evaluation['mask']['iou']:.6f} "
            f"level_abs_error_m={evaluation['water_level']['absolute_error_m']:.6f} "
            f"eligible_for_downstream=false"
        )


if __name__ == "__main__":
    main()
