#!/usr/bin/env python3
"""Prepare standard S4 files for the existing S5-S8 pipeline.

This module is an interface adapter, not a separate simulation algorithm.  It
accepts frozen prediction-side S4 artifacts, applies the already-frozen C8
candidate gate, and writes the same depth-map contract consumed by S5.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation.phase2d_c8_candidate_quality_gate import evaluate_candidate_gate


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    key = "phase2d_c11_direct_e2e_20cm"
    if key not in document:
        raise ValueError(f"Configuration must contain top-level {key!r}")
    return document[key]


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required source file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _prediction_config(mean_depth_cm: float) -> dict[str, Any]:
    records = [
        {"minutes_before_now": minute, "mean_depth_cm": float(mean_depth_cm)}
        for minute in (10, 5, 1, 0)
    ]
    return {
        "prediction": {
            "mode": "simulation_static_hold_extension",
            "description": (
                "S7 smoke-test forecast from the frozen 41-frame simulation window. "
                "It is not an observed 10-minute history."
            ),
            "input": {
                "area_volume_result_json": "outputs/json/water_area_volume_result.json",
                "weather_correction_result_json": "outputs/json/weather_correction_result.json",
            },
            "output": {
                "reasoning_dir": "data/reasoning",
                "json_dir": "outputs/json",
                "figure_dir": "outputs/figures",
            },
            "depth_history": {
                "source": "simulation_static_41_frame_window_hold_extension",
                "unit": "cm",
                "note": (
                    "All history points repeat the current 41-frame estimate solely to exercise "
                    "the existing S7 interface without inventing a rising trend."
                ),
                "records": records,
            },
            "slope_windows": {
                "window_1min": {"minutes": 1, "weight": 0.5},
                "window_5min": {"minutes": 5, "weight": 0.3},
                "window_10min": {"minutes": 10, "weight": 0.2},
            },
            "forecast_horizons_min": [5, 15, 30, 60],
            "warning_thresholds_cm": {"blue": 15.0, "yellow": 30.0, "orange": 50.0},
            "constraints": {
                "max_reasonable_depth_cm": 100.0,
                "min_slope_cm_per_min": -5.0,
                "max_slope_cm_per_min": 10.0,
            },
            "mvp_note": (
                "Direct simulation sensor-chain smoke test. Static-hold history is an explicit "
                "interface placeholder, not a measured 10-minute trend or real forecast."
            ),
            "note": "No Ground Truth is read and no formal warning action is permitted.",
        }
    }


def prepare_runtime(
    config_path: str | Path,
    project_root: str | Path,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_file = _resolve(root, config_path).resolve()
    config = _load_config(config_file)
    if config.get("data_domain") != "simulation":
        raise ValueError("Direct C11 runtime preparation requires data_domain=simulation")
    if config.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("Prediction-side runtime preparation must explicitly declare GT=false")

    inputs = config["runtime_inputs"]
    geometry_dir = _resolve(root, inputs["geometry_case_dir"]).resolve()
    dem_dir = _resolve(root, inputs["ground_dem_dir"]).resolve()
    gate_config_path = _resolve(root, inputs["candidate_gate_config"]).resolve()
    case_library_path = _resolve(root, inputs["case_library_json"]).resolve()
    target = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else _resolve(root, config["runtime_root"]).resolve()
    )
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Runtime root is not empty; refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=True)

    frame_rows = _load_json(geometry_dir / "per_frame_geometry_summary.json")
    sequence_metrics = _load_json(geometry_dir / "sequence_geometry_stability.json")
    with gate_config_path.open("r", encoding="utf-8") as handle:
        gate_document = yaml.safe_load(handle) or {}
    gate_config = gate_document["phase2d_c8_candidate_quality_gate"]
    sample = config["samples"][0]
    decisions: list[dict[str, Any]] = []
    for row in frame_rows:
        decision = evaluate_candidate_gate(row, sequence_metrics, gate_config)
        decision.update(
            {
                "sample_id": sample["sample_id"],
                "case_id": sample["case_id"],
                "rain_level": sample["rain_level"],
                "seed": int(sample["seed"]),
            }
        )
        decisions.append(decision)

    anchor_index = int(config["anchor_frame_index"])
    anchor_rows = [row for row in frame_rows if int(row["frame_index"]) == anchor_index]
    anchor_decisions = [row for row in decisions if int(row["frame_index"]) == anchor_index]
    if len(anchor_rows) != 1 or len(anchor_decisions) != 1:
        raise ValueError(f"Expected exactly one anchor frame {anchor_index}")
    anchor = anchor_rows[0]
    anchor_gate = anchor_decisions[0]
    if anchor_gate["camera_visible_status"] != "pass":
        raise ValueError(f"Anchor frame is rejected by frozen candidate gate: {anchor_gate}")
    if anchor_gate["global_scene_status"] != "complete":
        raise ValueError("This full S5-S8 smoke test requires a complete global scene estimate")

    source_depth = geometry_dir / "anchor_predicted_depth_m.npy"
    source_mask = geometry_dir / "anchor_predicted_dem_mask.npy"
    depth = np.load(source_depth)
    mask = np.load(source_mask).astype(bool)
    if depth.shape != mask.shape:
        raise ValueError(f"S4 depth/mask shape mismatch: {depth.shape} vs {mask.shape}")
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("S4 depth map must be finite and non-negative")
    if np.any(depth[~mask] != 0.0):
        raise ValueError("S4 depth outside the predicted DEM mask must be zero")

    hydrology_dir = target / "data" / "hydrology"
    dem_target_dir = target / "data" / "dem"
    json_dir = target / "outputs" / "json"
    quality_dir = target / "data" / "quality"
    config_dir = target / "configs"
    for directory in (hydrology_dir, dem_target_dir, json_dir, quality_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    depth_target = hydrology_dir / "water_depth_map.npy"
    mask_target = hydrology_dir / "water_depth_valid_mask.npy"
    np.save(depth_target, depth.astype(np.float32))
    np.save(mask_target, mask)
    for name in (
        "ground_dem.npy",
        "ground_dem_interpolated.npy",
        "ground_dem_valid_mask.npy",
        "ground_dem_metadata.json",
    ):
        _copy_required(dem_dir / name, dem_target_dir / name)
    _copy_required(case_library_path, target / "data" / "cases" / "mock_historical_cases.json")

    s4_result = {
        "stage": "S4_camera_video_mask_ground_dem_depth_inversion",
        "depth_method": "sam2_video_mask_ray_dem_boundary_water_level",
        "simulation_label": "direct_raw_simulation_sensor_pipeline",
        "configured_depth_cm": None,
        "sample_id": sample["sample_id"],
        "case_id": sample["case_id"],
        "rain_level": sample["rain_level"],
        "seed": int(sample["seed"]),
        "frame_index": anchor_index,
        "estimated_water_level_m": float(anchor["estimated_water_level_m"]),
        "mean_depth_cm": float(anchor["mean_depth_cm"]),
        "median_depth_cm": float(anchor["median_depth_cm"]),
        "max_depth_cm": float(anchor["max_depth_cm"]),
        "water_area_m2": float(anchor["water_area_m2"]),
        "water_volume_m3": float(anchor["water_volume_m3"]),
        "camera_reprojection_iou": float(anchor["camera_reprojection_iou"]),
        "outer_boundary_reprojection_p95_px": float(
            anchor["outer_boundary_reprojection_p95_px"]
        ),
        "candidate_gate": anchor_gate,
        "legacy_gate_status": anchor.get("quality_status"),
        "legacy_gate_reasons": anchor.get("gate_reasons", []),
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "eligible_for_simulation_pipeline": True,
        "warning_action_mode": "simulation_record_only",
        "note": (
            "Standard S4 interface generated from raw simulated LiDAR and RGB prediction. "
            "It is non-authoritative and cannot trigger a real warning."
        ),
        "source_files": {
            "depth_map": str(source_depth),
            "dem_mask": str(source_mask),
            "frame_metrics": str(geometry_dir / "per_frame_geometry_summary.json"),
            "sequence_metrics": str(geometry_dir / "sequence_geometry_stability.json"),
        },
        "output_files": {
            "depth_map": str(depth_target),
            "valid_mask": str(mask_target),
            "result_json": str(json_dir / "water_depth_result.json"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(json_dir / "water_depth_result.json", s4_result)
    _write_json(quality_dir / "candidate_gate_per_frame.json", decisions)
    _write_json(quality_dir / "candidate_gate_anchor.json", anchor_gate)

    prediction_path = config_dir / "runtime_prediction.yaml"
    with prediction_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _prediction_config(float(anchor["mean_depth_cm"])),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    status_counts = Counter(row["camera_visible_status"] for row in decisions)
    global_counts = Counter(row["global_scene_status"] for row in decisions)
    manifest = {
        "protocol_version": config["protocol_version"],
        "stage": "C11_direct_standard_pipeline_runtime_preparation",
        "runtime_root": str(target),
        "data_domain": "simulation",
        "source_pipeline": "raw_simulated_lidar_plus_raw_simulated_rgb",
        "standard_pipeline_entry": "S5_area_volume",
        "anchor_frame_index": anchor_index,
        "frame_count": len(frame_rows),
        "candidate_gate_status_counts": dict(sorted(status_counts.items())),
        "global_scene_status_counts": dict(sorted(global_counts.items())),
        "anchor_candidate_gate": anchor_gate,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "warning_action_mode": "simulation_record_only",
        "external_notification_allowed": False,
        "real_device_action_allowed": False,
        "history_semantics": "simulation_static_41_frame_window_hold_extension",
        "input_hashes": {
            "source_depth_map_sha256": _sha256(source_depth),
            "source_dem_mask_sha256": _sha256(source_mask),
            "candidate_gate_config_sha256": _sha256(gate_config_path),
        },
        "standard_output_hashes": {
            "water_depth_map_sha256": _sha256(depth_target),
            "water_depth_valid_mask_sha256": _sha256(mask_target),
            "water_depth_result_sha256": _sha256(json_dir / "water_depth_result.json"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target / "runtime_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["prediction_config_path"] = str(prediction_path)
    return manifest
