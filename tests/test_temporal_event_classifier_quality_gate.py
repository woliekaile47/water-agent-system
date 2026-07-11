import numpy as np

from src.perception.temporal_water_quality_gate import evaluate_model_classifier_quality_gate


def test_model_gate_flags_ood_without_ground_truth():
    item = {"model_water_probability": .5, "classification": "uncertain", "confidence": .5,
            "maximum_absolute_train_z": 10, "baseline_rule_classification": "dry_splash",
            "center_mean": [2, 2], "duration_frames": 2, "maximum_area": 4,
            "radius_growth_slope": 0, "ringness": 1, "temporal_energy": 1,
            "local_dynamic_density": 0, "neighbor_track_count": 0}
    prediction = {
        "model_classifications": [item],
        "baseline": {"candidate_diagnostics": {"candidate_pixel_fraction": 0}},
        "model_evidence": {"predicted_water_probability": np.zeros((4, 4)),
                           "predicted_water_mask": np.zeros((4, 4), dtype=bool)},
        "model_evidence_diagnostics": {"predicted_water_fraction": 0,
            "maximum_single_track_contribution_fraction": 0},
        "model_window_mask_iou": 0,
    }
    config = {"ood_z_threshold": 5, "max_nonfinite_feature_fraction": 0,
              "max_predicted_water_fraction": .5, "max_uncertain_fraction_for_pass": .8,
              "min_confident_track_fraction": .1, "max_ood_track_fraction": .2,
              "max_dynamic_candidate_pixel_fraction": .4, "max_single_track_contribution": .8,
              "min_window_mask_iou": .1, "max_rule_model_disagreement": .8,
              "min_probability_separation": .1}
    gate = evaluate_model_classifier_quality_gate(prediction, config)
    assert gate["status"] == "partial"
    assert gate["out_of_distribution_warning"] is True
    assert gate["ground_truth_used"] is False
    assert gate["eligible_for_downstream"] is False
