from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.integration.prepare_simulation_e2e_runtime import prepare_runtime


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _project(tmp_path: Path, *, camera_iou: float = 0.95) -> tuple[Path, Path]:
    geometry = tmp_path / "geometry"
    dem = tmp_path / "dem"
    geometry.mkdir()
    dem.mkdir()
    depth = np.array([[0.0, 0.10], [0.0, 0.20]], dtype=np.float32)
    mask = depth > 0.0
    np.save(geometry / "anchor_predicted_depth_m.npy", depth)
    np.save(geometry / "anchor_predicted_dem_mask.npy", mask)
    frame = {
        "frame_index": 149,
        "geometry_available": True,
        "selected_basin_count": 1,
        "estimated_water_level_m": -0.2,
        "max_depth_cm": 20.0,
        "shoreline_intersection_success_rate": 1.0,
        "seed_ray_success_ratio": 1.0,
        "filtered_shoreline_sample_count": 128,
        "filtered_shoreline_mad_m": 0.005,
        "filtered_shoreline_iqr_m": 0.01,
        "camera_reprojection_iou": camera_iou,
        "outer_boundary_reprojection_p95_px": 4.0,
        "unobserved_candidate_basin_count": 0,
        "ambiguous_candidate_basin_count": 0,
        "mean_depth_cm": 15.0,
        "median_depth_cm": 15.0,
        "water_area_m2": 0.02,
        "water_volume_m3": 0.003,
        "quality_status": "reject",
        "gate_reasons": ["boundary_reprojection_error_above_threshold"],
    }
    _write_json(geometry / "per_frame_geometry_summary.json", [frame])
    _write_json(
        geometry / "sequence_geometry_stability.json",
        {
            "estimated_water_level_m": {"std": 0.001},
            "adjacent_absolute_water_level_change_m": {"p95": 0.002},
            "water_area_m2": {"coefficient_of_variation": 0.01},
            "water_volume_m3": {"coefficient_of_variation": 0.02},
        },
    )
    for name in ("ground_dem.npy", "ground_dem_interpolated.npy"):
        np.save(dem / name, np.zeros((2, 2), dtype=np.float32))
    np.save(dem / "ground_dem_valid_mask.npy", np.ones((2, 2), dtype=bool))
    _write_json(dem / "ground_dem_metadata.json", {"grid_size": 0.1, "dem_shape": [2, 2]})
    _write_json(tmp_path / "data" / "cases" / "mock_historical_cases.json", [])
    gate = {
        "phase2d_c8_candidate_quality_gate": {
            "protocol_version": "test_gate",
            "frame_thresholds": {
                "min_shoreline_intersection_success_rate": 0.65,
                "min_seed_ray_success_ratio": 0.65,
                "min_valid_shoreline_samples": 20,
                "max_shoreline_mad_m": 0.02,
                "max_shoreline_iqr_m": 0.06,
                "min_camera_reprojection_iou": 0.90,
                "max_physical_depth_m": 0.60,
                "advisory_outer_boundary_p95_px": 3.0,
            },
            "sequence_thresholds": {
                "max_water_level_window_std_cm": 0.5,
                "max_adjacent_water_level_change_p95_cm": 1.0,
                "max_water_area_coefficient_of_variation": 0.1,
                "max_water_volume_coefficient_of_variation": 0.2,
            },
        }
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "gate.yaml").write_text(yaml.safe_dump(gate), encoding="utf-8")
    config = {
        "phase2d_c11_direct_e2e_20cm": {
            "protocol_version": "test_direct",
            "data_domain": "simulation",
            "ground_truth_used_for_prediction": False,
            "anchor_frame_index": 149,
            "runtime_root": "runtime",
            "runtime_inputs": {
                "geometry_case_dir": "geometry",
                "ground_dem_dir": "dem",
                "candidate_gate_config": "configs/gate.yaml",
                "case_library_json": "data/cases/mock_historical_cases.json",
            },
            "samples": [
                {
                    "sample_id": "sample",
                    "case_id": "sim_case",
                    "rain_level": "moderate",
                    "seed": 303,
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, tmp_path / "runtime"


def test_prepare_writes_standard_s4_contract_without_configured_depth(tmp_path: Path) -> None:
    config, runtime = _project(tmp_path)
    manifest = prepare_runtime(config, tmp_path)
    depth = np.load(runtime / "data" / "hydrology" / "water_depth_map.npy")
    valid = np.load(runtime / "data" / "hydrology" / "water_depth_valid_mask.npy")
    result = json.loads((runtime / "outputs" / "json" / "water_depth_result.json").read_text())
    assert depth.shape == valid.shape == (2, 2)
    assert np.all(depth[~valid] == 0.0)
    assert result["configured_depth_cm"] is None
    assert result["ground_truth_used"] is False
    assert result["eligible_for_simulation_pipeline"] is True
    assert manifest["anchor_candidate_gate"]["camera_visible_status"] == "pass"
    assert manifest["anchor_candidate_gate"]["global_scene_status"] == "complete"


def test_advisory_boundary_does_not_reject_otherwise_valid_anchor(tmp_path: Path) -> None:
    config, _ = _project(tmp_path)
    result = prepare_runtime(config, tmp_path)
    gate = result["anchor_candidate_gate"]
    assert gate["camera_visible_status"] == "pass"
    assert gate["boundary_metric_rejected_by_itself"] is False
    assert "outer_boundary_reprojection_p95_above_advisory_threshold" in gate["warnings"]


def test_rejected_anchor_cannot_enter_standard_s5_s8_runtime(tmp_path: Path) -> None:
    config, _ = _project(tmp_path, camera_iou=0.5)
    with pytest.raises(ValueError, match="rejected by frozen candidate gate"):
        prepare_runtime(config, tmp_path)


def test_runtime_preparation_refuses_overwrite(tmp_path: Path) -> None:
    config, runtime = _project(tmp_path)
    prepare_runtime(config, tmp_path)
    assert runtime.exists()
    with pytest.raises(FileExistsError):
        prepare_runtime(config, tmp_path)


def test_prediction_source_is_explicit_static_hold_not_mock_rising_history(tmp_path: Path) -> None:
    config, runtime = _project(tmp_path)
    prepare_runtime(config, tmp_path)
    prediction = yaml.safe_load((runtime / "configs" / "runtime_prediction.yaml").read_text())
    history = prediction["prediction"]["depth_history"]
    assert history["source"] == "simulation_static_41_frame_window_hold_extension"
    assert {row["mean_depth_cm"] for row in history["records"]} == {15.0}
