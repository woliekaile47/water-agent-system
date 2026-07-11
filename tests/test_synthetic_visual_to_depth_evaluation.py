import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.evaluation.evaluate_synthetic_visual_to_depth as module


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _prediction(output: Path, camera_mask, unknown, measurement="rejected_candidate", candidate=True):
    output.mkdir(parents=True)
    _write_json(output / "prediction_manifest.json", {
        "data_role": "prediction", "ground_truth_or_metadata_read_during_prediction": False})
    _write_json(output / "visual_quality_gate.json", {"status": "pass", "reasons": []})
    _write_json(output / "geometry_quality_gate.json", {"status": "reject", "reasons": ["boundary"]})
    _write_json(output / "integration_quality_gate.json", {
        "status": "reject", "reasons": ["geometry:boundary"], "measurement_status": measurement,
        "candidate_values_available": candidate, "authoritative_measurement_available": False,
        "global_estimate_status": "unavailable", "observable_region_result_valid": False,
        "area_volume_semantics": "observable_lower_bound"})
    Image.fromarray(np.where(camera_mask, 255, 0).astype(np.uint8)).save(output / "predicted_camera_water_mask.png")
    Image.fromarray(np.where(unknown, 255, 0).astype(np.uint8)).save(output / "predicted_camera_unknown_mask.png")


def _sequence(root: Path, case_id: str, rain="moderate", seed=42, water_gt=None):
    sequence = root / "data/simulation_dynamic" / case_id / rain / f"seed_{seed}"
    gt = sequence / "ground_truth"
    gt.mkdir(parents=True)
    Image.fromarray(np.where(water_gt, 255, 0).astype(np.uint8)).save(gt / "water_mask.png")
    return sequence


def test_water_rejected_candidate_metrics_are_diagnostic(monkeypatch, tmp_path: Path):
    gt_mask = np.asarray([[True, True], [False, False]])
    sequence = _sequence(tmp_path, "sim_water_10cm_001", water_gt=gt_mask)
    output = tmp_path / "prediction"
    _prediction(output, gt_mask, np.zeros_like(gt_mask))
    np.save(output / "predicted_dem_mask.npy", gt_mask)
    np.save(output / "predicted_depth_map_m.npy", np.asarray([[.1, .08], [0, 0]], dtype=np.float32))
    _write_json(output / "predicted_water_result.json", {
        "predicted_water_level_m": .1, "water_area_m2": .02, "water_volume_m3": .0018})
    marker = {"prediction_checked": False}
    original = module._completed_prediction

    def checked(path):
        result = original(path)
        marker["prediction_checked"] = True
        return result

    def gt_loader(*args):
        assert marker["prediction_checked"]
        return {"water_level_m": .11, "water_area_m2": .02, "water_volume_m3": .002,
                "dem_mask": gt_mask, "depth_map": np.asarray([[.11, .09], [0, 0]], dtype=np.float32)}

    monkeypatch.setattr(module, "_completed_prediction", checked)
    monkeypatch.setattr(module, "load_ground_truth_evaluation_inputs", gt_loader)
    result = module.evaluate_synthetic_visual_to_depth_case(sequence, output, tmp_path)
    assert result["metric_role"] == "rejected_candidate_diagnostic"
    assert result["gate"]["authoritative_measurement_available"] is False
    assert result["water_level"]["water_level_absolute_error_m"] == pytest.approx(.01)
    assert result["dem_mask"]["iou"] == 1.0


def test_dry_sequence_does_not_invent_water_level(monkeypatch, tmp_path: Path):
    empty = np.zeros((3, 3), dtype=bool)
    sequence = _sequence(tmp_path, "sim_dry_baseline_001", rain="light", seed=201, water_gt=empty)
    output = tmp_path / "prediction"
    _prediction(output, empty, empty, measurement="unavailable", candidate=False)
    monkeypatch.setattr(module, "load_ground_truth_evaluation_inputs", lambda *args: (_ for _ in ()).throw(AssertionError("dry must not load static water GT")))
    result = module.evaluate_synthetic_visual_to_depth_case(sequence, output, tmp_path)
    assert result["water_level"] is None
    assert result["depth"] is None
    assert result["area"] is None
    assert result["volume"] is None
    assert result["dry_false_positive"]["false_positive_area_m2"] == 0.0


def test_40cm_components_keep_unobservable_basin_semantics(monkeypatch, tmp_path: Path):
    gt_mask = np.zeros((3, 5), dtype=bool)
    gt_mask[1, 0:2] = True
    gt_mask[1, 4] = True
    sequence = _sequence(tmp_path, "sim_water_40cm_001", rain="heavy", seed=202, water_gt=np.zeros_like(gt_mask))
    output = tmp_path / "prediction"
    _prediction(output, np.zeros_like(gt_mask), np.zeros_like(gt_mask))
    predicted = np.zeros_like(gt_mask)
    predicted[1, 0:2] = True
    np.save(output / "predicted_dem_mask.npy", predicted)
    np.save(output / "predicted_depth_map_m.npy", predicted.astype(np.float32) * .4)
    _write_json(output / "predicted_water_result.json", {
        "predicted_water_level_m": .4, "water_area_m2": .02, "water_volume_m3": .008})
    monkeypatch.setattr(module, "load_ground_truth_evaluation_inputs", lambda *args: {
        "water_level_m": .4, "water_area_m2": .03, "water_volume_m3": .012,
        "dem_mask": gt_mask, "depth_map": gt_mask.astype(np.float32) * .4})
    monkeypatch.setattr(module, "_load_sensors", lambda *args: {})

    def fake_reproject(component, level, sensors):
        image = np.zeros_like(gt_mask, dtype=np.uint8)
        if np.count_nonzero(component) > 1:
            image[0, 0] = 255
        return image, {}

    monkeypatch.setattr(module, "reproject_water_surface", fake_reproject)
    result = module.evaluate_synthetic_visual_to_depth_case(sequence, output, tmp_path)
    roles = {item["component_role"]: item for item in result["components_40cm"]}
    assert roles["visible_main_basin"]["recall"] == 1.0
    assert roles["unobservable_secondary_basin"]["recall"] == 0.0
    assert roles["unobservable_secondary_basin"]["unobservable_false_negative_is_reported_separately"] is True
