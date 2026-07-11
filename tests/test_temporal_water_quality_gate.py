import numpy as np

from src.perception.temporal_water_quality_gate import evaluate_temporal_quality_gate


def _gate_config():
    return {
        "high_confidence_water_threshold": .6, "min_input_frames": 20, "min_fps": 5,
        "max_fps": 60, "min_dynamic_tracks": 2, "min_high_confidence_water_tracks": 1,
        "min_evidence_coverage": .01, "max_unknown_fraction_for_pass": .9,
        "max_candidate_pixel_fraction": .4, "max_water_components": 4,
        "min_largest_component_ratio": .3, "max_evidence_concentration": .9,
        "max_exposure_anomaly_fraction": .3, "min_order_sensitivity": .05,
        "min_water_mask_time_stability": .1, "min_feature_score_separation": .05,
        "max_predicted_water_fraction": .5,
    }


def test_quality_gate_is_prediction_only_and_rejects_invalid_input():
    water = np.zeros((20, 20), dtype=bool)
    probability = np.zeros((20, 20), dtype=np.float32)
    probability[5, 5] = np.nan
    gate = evaluate_temporal_quality_gate(
        {"frame_count": 1}, {"exposure_anomaly_fraction": 0},
        {"candidate_pixel_fraction": 0}, [], [],
        {"predicted_water_mask": water, "predicted_water_probability": probability},
        {"evidence_coverage_fraction": 0, "unknown_fraction": 1,
         "predicted_water_fraction": 0, "water_track_count": 0,
         "evidence_concentration": 0},
        0, 0, 0, 20, _gate_config(),
    )
    assert gate["status"] == "reject"
    assert "insufficient_input_frames" in gate["reasons"]
    assert "nonfinite_probability" in gate["reasons"]
    assert gate["ground_truth_used"] is False
    assert gate["eligible_for_downstream"] is False
