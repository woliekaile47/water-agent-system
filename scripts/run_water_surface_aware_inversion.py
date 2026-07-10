#!/usr/bin/env python3
"""Run Phase 2B water-surface-aware Camera-mask/DEM inversion."""

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

from scripts.run_simulation_depth_inversion import save_comparison_figures
from src.evaluation.evaluate_simulation_depth import evaluate_prediction, load_ground_truth_evaluation_inputs
from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate
from src.fusion.project_camera_mask_to_dem import load_prediction_inputs, project_camera_mask_to_dem
from src.fusion.water_surface_aware_mask_to_dem import (
    ALGORITHM_VERSION,
    PREDICTION_SOURCE,
    camera_reprojection_consistency,
    intersect_camera_shoreline,
    reconstruct_connected_lowland,
    reproject_water_surface,
)
from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.hydrology.estimate_water_level_from_boundary import connected_components
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem


CASES = ["sim_water_5cm_001", "sim_water_10cm_001", "sim_water_20cm_001", "sim_water_40cm_001"]


def load_yaml(path: Path, key: str) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if key not in document:
        raise ValueError(f"Missing {key} in {path}")
    return document[key]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
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


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def run_case(project_root: Path, case_id: str) -> dict[str, Any]:
    mapping = load_yaml(project_root / "configs/water_surface_aware_mapping.yaml", "water_surface_aware_mapping")
    gate_config = load_yaml(project_root / "configs/water_surface_aware_quality_gate.yaml", "water_surface_aware_quality_gate")
    output_dir = project_root / "outputs" / "simulation_evaluation_v2" / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prediction stage. This loader is allow-listed and never opens the water
    # case manifest, DEM GT mask, depth GT, or GT metadata.
    inputs = load_prediction_inputs(project_root, case_id, mapping)
    ground_dem = inputs["ground_dem"]
    camera_mask = inputs["camera_mask"]
    sensors = inputs["sensors"]
    initial_seed_mask, initial_projection = project_camera_mask_to_dem(
        ground_dem, camera_mask, sensors, int(mapping.get("mask_threshold", 127))
    )
    intersections, ray_diagnostics = intersect_camera_shoreline(
        camera_mask, ground_dem, sensors, mapping
    )
    water_level, shoreline_diagnostics = estimate_water_level_from_shoreline(
        intersections, mapping["shoreline_water_level"]
    )
    shoreline_diagnostics["estimated_water_level_m"] = water_level
    predicted_mask, reconstruction = reconstruct_connected_lowland(
        ground_dem,
        water_level,
        initial_seed_mask,
        mapping["reconstruction"],
        observed_camera_mask=camera_mask,
        sensors=sensors,
    )
    cell_size = float(sensors["road"]["dem_resolution_m"])
    predicted_depth, _, water_result = invert_depth_from_ground_dem(
        ground_dem, predicted_mask, water_level, cell_size
    )
    reprojected_mask, water_surface_projection = reproject_water_surface(
        predicted_mask, water_level, sensors
    )
    consistency = camera_reprojection_consistency(
        camera_mask, reprojected_mask, water_surface_projection
    )
    consistency.update({
        "shoreline_intersection_success_rate": ray_diagnostics["shoreline_intersection_success_rate"],
        "camera_mask_edge_touch_ratio": ray_diagnostics["camera_mask_edge_touch_ratio"],
        "candidate_basin_count": reconstruction["candidate_basin_count"],
        "seed_valid": reconstruction["seed_valid"],
        "water_level_converged": shoreline_diagnostics["water_level_converged"],
    })

    common = {
        **metadata(case_id),
        "prediction_inputs": [
            "dry_ground_dem",
            "camera_water_mask_gt_as_phase2b_visual_input",
            "camera_intrinsics_and_T_map_camera_optical_from_sensors_config",
            "phase2a_ground_projection_used_only_as_flood_fill_seed",
        ],
        "prediction_input_paths": inputs["paths"],
    }
    paths = {
        "intersections": output_dir / "shoreline_intersections.json",
        "ray": output_dir / "ray_intersection_diagnostics.json",
        "level": output_dir / "predicted_water_level.json",
        "mask_npy": output_dir / "predicted_dem_mask.npy",
        "mask_png": output_dir / "predicted_dem_mask.png",
        "reconstruction": output_dir / "reconstruction_diagnostics.json",
        "depth": output_dir / "predicted_depth_map_m.npy",
        "water": output_dir / "predicted_water_result.json",
        "reprojected": output_dir / "reprojected_camera_mask.png",
        "consistency": output_dir / "self_consistency.json",
        "gate": output_dir / "quality_gate.json",
        "evaluation": output_dir / "evaluation_metrics.json",
    }
    np.save(paths["mask_npy"], predicted_mask)
    np.save(paths["depth"], predicted_depth)
    save_mask(paths["mask_png"], predicted_mask)
    Image.fromarray(reprojected_mask, mode="L").save(paths["reprojected"])
    intersection_document = {**common, "intersections": intersections, "quality_gate_status": "pending"}
    ray_document = {**common, **ray_diagnostics, "quality_gate_status": "pending"}
    level_document = {**common, **shoreline_diagnostics, "predicted_water_level_m": water_level, "quality_gate_status": "pending"}
    reconstruction_document = {
        **common,
        **reconstruction,
        "initial_seed_projection": initial_projection,
        "quality_gate_status": "pending",
    }
    water_document = {**common, **water_result, "quality_gate_status": "pending"}
    consistency_document = {**common, **consistency, "quality_gate_status": "pending"}
    for path, document in (
        (paths["intersections"], intersection_document),
        (paths["ray"], ray_document),
        (paths["level"], level_document),
        (paths["reconstruction"], reconstruction_document),
        (paths["water"], water_document),
        (paths["consistency"], consistency_document),
    ):
        write_json(path, document)

    required_prediction_files = [
        paths["intersections"], paths["ray"], paths["level"], paths["mask_npy"], paths["mask_png"],
        paths["reconstruction"], paths["depth"], paths["water"], paths["reprojected"], paths["consistency"],
    ]
    gate = evaluate_water_surface_aware_quality_gate(
        ray_diagnostics,
        shoreline_diagnostics,
        reconstruction,
        consistency,
        water_result,
        predicted_depth,
        gate_config,
        required_prediction_files,
    )
    gate_document = {**common, **gate, "quality_gate_status": gate["status"]}
    write_json(paths["gate"], gate_document)
    semantic_keys = (
        "observation_scope",
        "global_estimate_status",
        "observable_region_result_valid",
        "unobservable_candidate_basin_count",
        "ambiguous_candidate_basin_count",
        "camera_observable_candidate_basin_count",
        "result_semantics",
        "area_volume_semantics",
        "eligible_for_downstream",
    )
    semantics = {key: gate[key] for key in semantic_keys}
    for path, document in (
        (paths["intersections"], intersection_document),
        (paths["ray"], ray_document),
        (paths["level"], level_document),
        (paths["reconstruction"], reconstruction_document),
        (paths["water"], water_document),
        (paths["consistency"], consistency_document),
    ):
        document["quality_gate_status"] = gate["status"]
        document.update(semantics)
        write_json(path, document)

    # Evaluation begins only after all prediction outputs and the GT-free gate
    # are complete. This is the only answer-data loader call.
    ground_truth = load_ground_truth_evaluation_inputs(project_root, case_id)
    evaluation_boundary = {
        "valid_boundary_sample_count": shoreline_diagnostics["valid_shoreline_sample_count"],
        "boundary_height_mad_m": shoreline_diagnostics["shoreline_height_mad_m"],
        "boundary_height_iqr_m": shoreline_diagnostics["shoreline_height_iqr_m"],
        "boundary_height_std_m": shoreline_diagnostics["shoreline_height_std_m"],
    }
    evaluation = evaluate_prediction(
        predicted_mask,
        predicted_depth,
        water_document,
        evaluation_boundary,
        {"projection_coverage": consistency["water_surface_projection_coverage"]},
        ground_truth,
        gate,
    )
    ground_truth_component_evaluation = []
    for component_index, component in enumerate(connected_components(ground_truth["dem_mask"], 8)):
        component_cells = int(np.count_nonzero(component))
        matched_cells = int(np.count_nonzero(component & predicted_mask))
        ground_truth_component_evaluation.append({
            "component_index": component_index,
            "ground_truth_cell_count": component_cells,
            "predicted_intersection_cell_count": matched_cells,
            "recall": float(matched_cells / max(1, component_cells)),
        })
    evaluation_document = {
        **metadata(case_id, "evaluation"),
        **evaluation,
        "prediction_side_self_consistency": consistency,
        **semantics,
        "ground_truth_component_evaluation": ground_truth_component_evaluation,
        "ground_truth_input_paths": ground_truth["paths"],
    }
    write_json(paths["evaluation"], evaluation_document)
    save_comparison_figures(
        output_dir,
        predicted_mask,
        predicted_depth,
        ground_truth["dem_mask"],
        ground_truth["depth_map"],
    )
    return {"case_id": case_id, "gate": gate, "evaluation": evaluation_document}


def write_comparison(project_root: Path, results: list[dict[str, Any]]) -> None:
    output_root = project_root / "outputs/simulation_evaluation_v2"
    fields = [
        "case_id", "phase2a_iou", "phase2b_iou", "phase2a_recall", "phase2b_recall",
        "phase2a_area_relative_error", "phase2b_area_relative_error",
        "phase2a_volume_relative_error", "phase2b_volume_relative_error",
        "phase2b_reprojection_iou", "phase2b_gate", "phase2b_gate_reasons",
        "eligible_for_downstream",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        case_id = result["case_id"]
        baseline_path = project_root / "outputs/simulation_evaluation" / case_id / "evaluation_metrics.json"
        with baseline_path.open("r", encoding="utf-8") as stream:
            baseline = json.load(stream)
        current = result["evaluation"]
        row = {
            "case_id": case_id,
            "phase2a_iou": baseline["mask"]["iou"],
            "phase2b_iou": current["mask"]["iou"],
            "phase2a_recall": baseline["mask"]["recall"],
            "phase2b_recall": current["mask"]["recall"],
            "phase2a_area_relative_error": baseline["area"]["relative_error"],
            "phase2b_area_relative_error": current["area"]["relative_error"],
            "phase2a_volume_relative_error": baseline["volume"]["relative_error"],
            "phase2b_volume_relative_error": current["volume"]["relative_error"],
            "phase2b_reprojection_iou": current["prediction_side_self_consistency"]["camera_reprojection_iou"],
            "phase2b_gate": result["gate"]["status"],
            "phase2b_gate_reasons": ";".join(result["gate"]["reasons"]),
            "eligible_for_downstream": False,
        }
        rows.append(row)
    with (output_root / "phase2a_vs_phase2b.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 2A baseline vs Phase 2B water-surface-aware mapping",
        "",
        "| case | 2A IoU | 2B IoU | 2A recall | 2B recall | 2A area rel. err | 2B area rel. err | 2B reprojection IoU | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['phase2a_iou']:.6f} | {row['phase2b_iou']:.6f} | "
            f"{row['phase2a_recall']:.6f} | {row['phase2b_recall']:.6f} | "
            f"{row['phase2a_area_relative_error']:.6f} | {row['phase2b_area_relative_error']:.6f} | "
            f"{row['phase2b_reprojection_iou']:.6f} | {row['phase2b_gate']} |"
        )
    (output_root / "phase2a_vs_phase2b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    selected = CASES if args.all else [args.case]
    results = [run_case(root, case_id) for case_id in selected]
    if args.all:
        write_comparison(root, results)
    for result in results:
        evaluation = result["evaluation"]
        consistency = evaluation["prediction_side_self_consistency"]
        print(
            f"[phase2b] {result['case_id']}: gate={result['gate']['status']} "
            f"iou={evaluation['mask']['iou']:.6f} recall={evaluation['mask']['recall']:.6f} "
            f"area_relative_error={evaluation['area']['relative_error']:.6f} "
            f"reprojection_iou={consistency['camera_reprojection_iou']:.6f} "
            f"eligible_for_downstream=false"
        )


if __name__ == "__main__":
    main()
