#!/usr/bin/env python3
"""Prediction-only RGB-to-depth core for Phase 2D-A.

This module intentionally contains no Ground Truth evaluation imports. The
only permitted ``ground_truth`` path is the prebuilt dry baseline DEM named
by the integration configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate
from src.fusion.project_camera_mask_to_dem import project_camera_mask_to_dem
from src.fusion.water_surface_aware_mask_to_dem import reproject_water_surface
from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem
from src.integration.integration_quality_gate import evaluate_integration_quality_gate
from src.integration.unknown_aware_geometry import (
    UNKNOWN_SEMANTICS,
    build_trusted_shoreline,
    camera_reprojection_consistency_unknown_aware,
    intersect_trusted_camera_shoreline,
    reconstruct_connected_lowland_unknown_aware,
)
from src.perception.temporal_water_pipeline import run_temporal_prediction
from src.perception.temporal_water_quality_gate import evaluate_temporal_quality_gate


class GeometryUnavailable(RuntimeError):
    def __init__(self, stage: str, failure_type: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.failure_type = failure_type


def _load_yaml(path: Path, key: str | None = None) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if key is None:
        return document
    if key not in document:
        raise ValueError(f"Missing {key} in {path}")
    return document[key]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _update_existing_json(path: Path, fields: dict[str, Any]) -> None:
    """Add final authority semantics only to artifacts actually produced."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    document.update(fields)
    _write_json(path, document)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _order_sensitivity(full: dict[str, Any], shuffled: dict[str, Any]) -> float:
    first = np.asarray(full["evidence"]["predicted_water_probability"], dtype=np.float64)
    second = np.asarray(shuffled["evidence"]["predicted_water_probability"], dtype=np.float64)
    difference = float(np.mean(np.abs(first - second)))
    reference = max(float(np.mean(first) + np.mean(second)), 0.01)
    return float(np.clip(difference / reference, 0.0, 1.0))


def _unavailable_geometry_gate(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "reject", "reasons": [failure["failure_type"]],
        "global_estimate_status": "unavailable",
        "observable_region_result_valid": False,
        "result_semantics": "unavailable", "area_volume_semantics": "unavailable",
        "eligible_for_downstream": False,
    }


def run_synthetic_visual_to_depth_prediction(
    project_root: str | Path, frames_dir: str | Path, output_dir: str | Path,
    integration_config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    frames = Path(frames_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not frames.is_dir() or frames.name != "frames":
        raise ValueError("frames_dir must be an existing directory named frames")
    output.mkdir(parents=True, exist_ok=True)

    detector_path = root / integration_config["temporal_detector_config"]
    visual_gate_path = root / integration_config["temporal_quality_gate_config"]
    mapping_path = root / integration_config["surface_mapping_config"]
    geometry_gate_path = root / integration_config["surface_quality_gate_config"]
    dry_dem_path = root / integration_config["dry_ground_dem"]
    sensors_path = root / integration_config["sensors_config"]
    detector = _load_yaml(detector_path, "temporal_water_mask_detector")
    visual_gate_config = _load_yaml(visual_gate_path, "temporal_water_quality_gate")
    mapping = _load_yaml(mapping_path, "water_surface_aware_mapping")
    geometry_gate_config = _load_yaml(geometry_gate_path, "water_surface_aware_quality_gate")

    full = run_temporal_prediction(str(frames), detector, "full")
    shuffled = run_temporal_prediction(str(frames), detector, "shuffled")
    order_sensitivity = _order_sensitivity(full, shuffled)
    visual_gate = evaluate_temporal_quality_gate(
        full["loader"], full["preprocessing_diagnostics"], full["candidate_diagnostics"],
        full["tracks"], full["classifications"], full["evidence"],
        full["evidence_diagnostics"], order_sensitivity,
        full["water_mask_time_stability"], full["feature_score_separation"],
        float(detector["fps"]), visual_gate_config,
    )
    water = np.asarray(full["evidence"]["predicted_water_mask"], dtype=bool)
    unknown = np.asarray(full["evidence"]["predicted_unknown_mask"], dtype=bool)
    if np.any(water & unknown):
        raise RuntimeError("temporal prediction produced overlapping water and unknown masks")
    trusted_mask, _, trusted_basic = build_trusted_shoreline(
        water, unknown, int(mapping.get("connectivity", 8)),
    )
    np.save(output / "predicted_water_probability.npy", full["evidence"]["predicted_water_probability"])
    np.save(output / "evidence_count_map.npy", full["evidence"]["evidence_count_map"])
    _save_mask(output / "predicted_camera_water_mask.png", water)
    _save_mask(output / "predicted_camera_unknown_mask.png", unknown)
    _save_mask(output / "trusted_shoreline_mask.png", trusted_mask)
    temporal_diagnostics = {
        "loader": full["loader"], "preprocessing": full["preprocessing_diagnostics"],
        "candidates": full["candidate_diagnostics"], "evidence": full["evidence_diagnostics"],
        "water_mask_time_stability": full["water_mask_time_stability"],
        "feature_score_separation": full["feature_score_separation"],
        "order_sensitivity": order_sensitivity,
        "track_count": len(full["tracks"]), "tracks": full["tracks"],
        "classifications": full["classifications"], "ground_truth_used": False,
    }
    _write_json(output / "temporal_diagnostics.json", temporal_diagnostics)
    _write_json(output / "visual_quality_gate.json", visual_gate)

    geometry_gate: dict[str, Any] | None = None
    geometry_failure: dict[str, Any] | None = None
    geometry_result: dict[str, Any] | None = None
    try:
        if visual_gate["status"] == "reject":
            raise GeometryUnavailable("visual_gate", "visual_quality_gate_reject", "Visual quality gate rejected prediction")
        if not np.any(water):
            raise GeometryUnavailable("camera_mask", "empty_camera_water_mask", "Rule Camera water mask is empty")
        if not np.any(trusted_mask):
            raise GeometryUnavailable("trusted_shoreline", "no_trusted_shoreline", "No water-to-known-nonwater shoreline exists")
        ground_dem = np.load(dry_dem_path).astype(np.float32)
        sensors = _load_yaml(sensors_path)
        seed_mask, seed_diagnostics = project_camera_mask_to_dem(
            ground_dem, np.where(water, 255, 0).astype(np.uint8), sensors,
            int(mapping.get("mask_threshold", 127)),
        )
        if not np.any(seed_mask):
            raise GeometryUnavailable("mask_to_dem", "no_valid_seed", "Camera rule mask produced no DEM seed")
        trusted_mask, intersections, ray_diagnostics = intersect_trusted_camera_shoreline(
            water, unknown, ground_dem, sensors, mapping,
        )
        _save_mask(output / "trusted_shoreline_mask.png", trusted_mask)
        _write_json(output / "shoreline_intersections.json", {
            "data_role": "prediction", "intersections": intersections, "ground_truth_used": False,
        })
        _write_json(output / "ray_intersection_diagnostics.json", ray_diagnostics)
        if not intersections:
            raise GeometryUnavailable("ray_dem_intersection", "no_valid_shoreline_intersections", "Trusted shoreline rays did not intersect the dry DEM")
        try:
            water_level, shoreline_diagnostics = estimate_water_level_from_shoreline(
                intersections, mapping["shoreline_water_level"],
            )
        except ValueError as error:
            raise GeometryUnavailable("water_level", "water_level_estimation_failed", str(error)) from error
        if not np.isfinite(water_level):
            raise GeometryUnavailable("water_level", "nonfinite_water_level", "Estimated water level is non-finite")
        shoreline_diagnostics["estimated_water_level_m"] = float(water_level)
        _write_json(output / "predicted_water_level.json", {
            **shoreline_diagnostics, "predicted_water_level_m": float(water_level),
            "data_role": "prediction", "ground_truth_used": False,
        })
        predicted_mask, reconstruction = reconstruct_connected_lowland_unknown_aware(
            ground_dem, water_level, seed_mask, mapping["reconstruction"], water, unknown, sensors,
        )
        reconstruction["initial_seed_projection"] = seed_diagnostics
        _write_json(output / "reconstruction_diagnostics.json", reconstruction)
        if not reconstruction["seed_valid"]:
            raise GeometryUnavailable("reconstruction", "invalid_reconstruction_seed", "No valid below-level Camera seed was retained")
        if reconstruction["selected_basin_count"] < 1 or not np.any(predicted_mask):
            raise GeometryUnavailable("reconstruction", "no_selected_basin", "No Camera-supported lowland basin was selected")
        cell_size = float(sensors["road"]["dem_resolution_m"])
        depth, predicted_mask, water_result = invert_depth_from_ground_dem(
            ground_dem, predicted_mask, water_level, cell_size,
        )
        reprojected, projection_diagnostics = reproject_water_surface(predicted_mask, water_level, sensors)
        consistency = camera_reprojection_consistency_unknown_aware(
            water, unknown, reprojected, projection_diagnostics,
        )
        np.save(output / "predicted_dem_mask.npy", predicted_mask)
        np.save(output / "predicted_depth_map_m.npy", depth)
        _save_mask(output / "predicted_dem_mask.png", predicted_mask)
        Image.fromarray(reprojected.astype(np.uint8), mode="L").save(output / "reprojected_camera_mask.png")
        _write_json(output / "predicted_water_result.json", {
            **water_result, "data_role": "prediction", "ground_truth_used": False,
        })
        _write_json(output / "self_consistency.json", consistency)
        required = [
            output / "shoreline_intersections.json", output / "ray_intersection_diagnostics.json",
            output / "predicted_water_level.json", output / "predicted_dem_mask.npy",
            output / "predicted_dem_mask.png", output / "reconstruction_diagnostics.json",
            output / "predicted_depth_map_m.npy", output / "predicted_water_result.json",
            output / "reprojected_camera_mask.png", output / "self_consistency.json",
        ]
        geometry_gate = evaluate_water_surface_aware_quality_gate(
            ray_diagnostics, shoreline_diagnostics, reconstruction, consistency,
            water_result, depth, geometry_gate_config, required,
        )
        geometry_result = {
            "geometry_status": "available", "predicted_water_level_m": float(water_level),
            "water_area_m2": water_result["water_area_m2"],
            "water_volume_m3": water_result["water_volume_m3"],
            "max_depth_m": water_result["max_depth_m"],
        }
    except GeometryUnavailable as error:
        geometry_failure = {
            "geometry_status": "unavailable", "failure_stage": error.stage,
            "failure_type": error.failure_type, "failure_message": str(error),
            "eligible_for_downstream": False,
        }
    except (ValueError, RuntimeError, OSError) as error:
        geometry_failure = {
            "geometry_status": "unavailable", "failure_stage": "unexpected_geometry_error",
            "failure_type": type(error).__name__, "failure_message": str(error),
            "eligible_for_downstream": False,
        }
    if geometry_failure is not None:
        geometry_gate = _unavailable_geometry_gate(geometry_failure)
        unavailable = {**geometry_failure, "data_role": "prediction"}
        for name in (
            "shoreline_intersections.json", "ray_intersection_diagnostics.json",
            "reconstruction_diagnostics.json", "self_consistency.json",
        ):
            path = output / name
            if not path.exists():
                _write_json(path, unavailable)
    _write_json(output / "geometry_quality_gate.json", geometry_gate)
    integration_gate = evaluate_integration_quality_gate(visual_gate, geometry_gate, geometry_failure)
    _write_json(output / "integration_quality_gate.json", integration_gate)
    measurement_fields = {
        "quality_gate_status": integration_gate["status"],
        "geometry_quality_gate_status": geometry_gate["status"],
        "integration_quality_gate_status": integration_gate["status"],
        "measurement_status": integration_gate["measurement_status"],
        "candidate_values_available": integration_gate["candidate_values_available"],
        "authoritative_measurement_available": integration_gate["authoritative_measurement_available"],
        "authoritative_area_volume_semantics": integration_gate["authoritative_area_volume_semantics"],
        "eligible_for_downstream": False,
    }
    for name in (
        "predicted_water_result.json", "predicted_water_level.json",
        "reconstruction_diagnostics.json", "self_consistency.json",
    ):
        _update_existing_json(output / name, measurement_fields)
    manifest = {
        "data_role": "prediction", "algorithm_version": integration_config["algorithm_version"],
        "camera_mask_source": "phase2c2a_rule_mask",
        "learned_classifier_used_for_geometry": False,
        "ground_truth_or_metadata_read_during_prediction": False,
        "dry_ground_dem_role": "prebuilt_no_water_spatial_baseline",
        "unknown_region_semantics": UNKNOWN_SEMANTICS,
        "prediction_inputs": {
            "frames": str(frames), "detector_config": str(detector_path),
            "visual_gate_config": str(visual_gate_path), "mapping_config": str(mapping_path),
            "geometry_gate_config": str(geometry_gate_path),
            "dry_ground_dem": str(dry_dem_path), "sensors_config": str(sensors_path),
        },
        "synthetic_domain": True, "real_world_validated": False,
        "measurement_status": integration_gate["measurement_status"],
        "candidate_values_available": integration_gate["candidate_values_available"],
        "authoritative_measurement_available": integration_gate["authoritative_measurement_available"],
        "authoritative_area_volume_semantics": integration_gate["authoritative_area_volume_semantics"],
        "integration_quality_gate_status": integration_gate["status"],
        "eligible_for_downstream": False,
    }
    _write_json(output / "prediction_manifest.json", manifest)
    return {
        "water_mask": water, "unknown_mask": unknown, "trusted_shoreline_mask": trusted_mask,
        "visual_gate": visual_gate, "geometry_gate": geometry_gate,
        "integration_gate": integration_gate, "geometry_failure": geometry_failure,
        "geometry_result": geometry_result, "output_dir": str(output),
    }
