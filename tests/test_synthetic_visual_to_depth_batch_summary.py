import csv
import json

import src.evaluation.synthetic_visual_to_depth_summary as summary_module


def _evaluation(case_id, depth, rain, seed, camera_iou, measurement, authoritative, water_level_error):
    dry = depth is None
    return {"case_id": case_id, "nominal_depth_cm": depth, "rain_level": rain, "seed": seed,
            "is_dry": dry, "metric_role": "dry_false_positive_diagnostic" if dry else "rejected_candidate_diagnostic",
            "camera_mask": {"whole_image_iou": camera_iou, "known_region_iou": camera_iou},
            "gate": {"visual_gate_status": "pass", "geometry_gate_status": "reject",
                     "integration_gate_status": "reject", "measurement_status": measurement,
                     "authoritative_measurement_available": authoritative,
                     "candidate_values_available": not dry, "reject_reasons": ["geometry:boundary"]},
            "water_level": None if water_level_error is None else {"water_level_absolute_error_m": water_level_error},
            "dem_mask": None if dry else {"iou": .5},
            "depth": None if dry else {"ground_truth_water_region": {"mae_m": None}},
            "area": None if dry else {"relative_error": .1}, "volume": None if dry else {"relative_error": .2},
            "dry_false_positive": {"false_water_pixels": 0, "false_water_fraction": 0.0,
                                   "false_water_components": 0, "false_positive_area_m2": 0.0} if dry else None,
            "components_40cm": None}


def test_summary_is_deterministic_and_missing_is_not_zero():
    values = [_evaluation("sim_dry_baseline_001", None, "light", 1, 0.0, "unavailable", False, None),
              _evaluation("sim_water_10cm_001", 10, "moderate", 2, .8, "rejected_candidate", False, None)]
    first = summary_module.build_dataset_summary(values)
    second = summary_module.build_dataset_summary(list(reversed(values)))
    assert first == second
    assert first["metrics_by_depth"]["10"]["water_level_absolute_error_m_mean"] is None
    assert first["measurement_status_distribution"] == {"unavailable": 1, "rejected_candidate": 1}


def test_csv_and_json_sequence_counts_match(monkeypatch, tmp_path):
    values = [_evaluation("sim_dry_baseline_001", None, "light", 1, 0.0, "unavailable", False, None),
              _evaluation("sim_water_10cm_001", 10, "moderate", 2, .8, "rejected_candidate", False, .01)]
    monkeypatch.setattr(summary_module, "write_figures_compatible", lambda *args: None)
    summary_module.write_summary_outputs(tmp_path, values)
    document = json.loads((tmp_path / "dataset_summary.json").read_text())
    with (tmp_path / "dataset_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert document["sequence_count"] == len(rows) == 2
