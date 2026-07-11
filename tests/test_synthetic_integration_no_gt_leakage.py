import inspect
import json
from pathlib import Path

import numpy as np
import yaml

import src.integration.synthetic_visual_to_depth as integration


def _prediction(frame_count=20):
    water = np.zeros((3, 3), dtype=bool)
    water[1, 1] = True
    unknown = np.zeros_like(water)
    probability = water.astype(np.float32)
    return {"loader": {"frame_count": frame_count, "height": 3, "width": 3},
            "preprocessing_diagnostics": {}, "candidate_diagnostics": {},
            "tracks": [], "classifications": [],
            "evidence": {"predicted_water_mask": water, "predicted_unknown_mask": unknown,
                         "predicted_water_probability": probability,
                         "evidence_count_map": probability},
            "evidence_diagnostics": {}, "water_mask_time_stability": 1.0,
            "feature_score_separation": 1.0}


def test_prediction_imports_and_signature_exclude_gt_loaders():
    source = inspect.getsource(integration)
    for forbidden in ("evaluate_simulation_depth", "evaluate_temporal_water_mask",
                      "load_ground_truth_evaluation_inputs", "load_temporal_evaluation_ground_truth",
                      "load_prediction_inputs"):
        assert forbidden not in source
    parameters = list(inspect.signature(integration.run_synthetic_visual_to_depth_prediction).parameters)
    assert parameters == ["project_root", "frames_dir", "output_dir", "integration_config"]
    assert not any(name in parameters for name in ("gt_mask", "depth_map", "water_level"))


def test_runtime_read_allowlist_and_prediction_manifest(monkeypatch, tmp_path: Path):
    root = tmp_path / "project"
    sequence = root / "data/simulation_dynamic/water/moderate/seed_42"
    frames = sequence / "frames"
    frames.mkdir(parents=True)
    (sequence / "ground_truth").mkdir()
    (sequence / "metadata").mkdir()
    (sequence / "manifest.json").write_text("forbidden", encoding="utf-8")
    (sequence / "ground_truth" / "water.json").write_text("forbidden", encoding="utf-8")
    (sequence / "metadata" / "answers.json").write_text("forbidden", encoding="utf-8")
    config_paths = {
        "detector.yaml": {"temporal_water_mask_detector": {"fps": 20}},
        "visual_gate.yaml": {"temporal_water_quality_gate": {}},
        "mapping.yaml": {"water_surface_aware_mapping": {
            "connectivity": 8, "mask_threshold": 127, "shoreline_water_level": {}, "reconstruction": {}}},
        "geometry_gate.yaml": {"water_surface_aware_quality_gate": {}},
        "sensors.yaml": {"road": {"dem_resolution_m": .1}},
    }
    for name, document in config_paths.items():
        path = root / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    dry_dem = root / "data/simulation/sim_dry_baseline_001/ground_truth/ground_dem_gt.npy"
    dry_dem.parent.mkdir(parents=True)
    np.save(dry_dem, np.zeros((2, 2), dtype=np.float32))
    output = root / "outputs/test"
    reads = []
    original_open = Path.open
    original_load = np.load

    def tracked_open(path, mode="r", *args, **kwargs):
        if "r" in mode:
            reads.append(Path(path).resolve())
        return original_open(path, mode, *args, **kwargs)

    def tracked_load(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(np, "load", tracked_load)
    monkeypatch.setattr(integration, "run_temporal_prediction", lambda *args, **kwargs: _prediction())
    monkeypatch.setattr(integration, "evaluate_temporal_quality_gate", lambda *args, **kwargs: {"status": "pass", "reasons": []})
    monkeypatch.setattr(integration, "project_camera_mask_to_dem", lambda *args, **kwargs: (np.asarray([[True, False], [False, False]]), {}))
    monkeypatch.setattr(integration, "intersect_trusted_camera_shoreline", lambda *args, **kwargs: (
        np.asarray([[False, False, False], [False, True, False], [False, False, False]]),
        [{"component_index": 0, "dem_height_m": .1}], {"shoreline_intersection_success_rate": 1.0, "camera_mask_edge_touch_ratio": 0.0}))
    monkeypatch.setattr(integration, "estimate_water_level_from_shoreline", lambda *args, **kwargs: (.1, {
        "estimated_water_level_m": .1, "valid_shoreline_sample_count": 1,
        "shoreline_height_mad_m": 0.0, "shoreline_height_iqr_m": 0.0,
        "water_level_converged": True}))
    monkeypatch.setattr(integration, "reconstruct_connected_lowland_unknown_aware", lambda *args, **kwargs: (
        np.asarray([[True, False], [False, False]]), {"seed_valid": True, "selected_basin_count": 1,
        "candidate_basin_count": 1, "ambiguous_candidate_basins": False,
        "ambiguous_candidate_basin_count": 0, "unobserved_candidate_basin_count": 0,
        "camera_observable_candidate_basin_count": 1}))
    monkeypatch.setattr(integration, "invert_depth_from_ground_dem", lambda *args, **kwargs: (
        np.asarray([[.1, 0], [0, 0]], dtype=np.float32), np.asarray([[True, False], [False, False]]),
        {"water_area_m2": .01, "water_volume_m3": .001, "max_depth_m": .1,
         "negative_depth_count": 0, "inf_depth_count": 0}))
    monkeypatch.setattr(integration, "reproject_water_surface", lambda *args, **kwargs: (
        np.zeros((3, 3), dtype=np.uint8), {"water_surface_projection_coverage": 1.0}))
    monkeypatch.setattr(integration, "camera_reprojection_consistency_unknown_aware", lambda *args, **kwargs: {
        "camera_reprojection_iou": 1.0, "boundary_reprojection_p95_px": 0.0,
        "water_surface_projection_coverage": 1.0})
    monkeypatch.setattr(integration, "evaluate_water_surface_aware_quality_gate", lambda *args, **kwargs: {
        "status": "pass", "reasons": [], "global_estimate_status": "complete",
        "observable_region_result_valid": True, "result_semantics": "global_estimate",
        "area_volume_semantics": "complete_estimate", "eligible_for_downstream": False})
    config = {"algorithm_version": "test", "temporal_detector_config": "configs/detector.yaml",
              "temporal_quality_gate_config": "configs/visual_gate.yaml",
              "surface_mapping_config": "configs/mapping.yaml",
              "surface_quality_gate_config": "configs/geometry_gate.yaml",
              "dry_ground_dem": "data/simulation/sim_dry_baseline_001/ground_truth/ground_dem_gt.npy",
              "sensors_config": "configs/sensors.yaml"}
    integration.run_synthetic_visual_to_depth_prediction(root, frames, output, config)
    forbidden_roots = [(sequence / "ground_truth").resolve(), (sequence / "metadata").resolve()]
    assert not any(path == (sequence / "manifest.json").resolve() for path in reads)
    assert not any(any(root_path in path.parents or path == root_path for root_path in forbidden_roots) for path in reads)
    assert dry_dem.resolve() in reads
    manifest = json.loads((output / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ground_truth_or_metadata_read_during_prediction"] is False
