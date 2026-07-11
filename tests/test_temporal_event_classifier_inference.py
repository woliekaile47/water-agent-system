import inspect

import numpy as np

from src.perception.temporal_event_classifier_inference import infer_track_classifications
from src.perception.temporal_water_evidence import build_rule_preserving_hybrid_evidence


def test_inference_signature_has_no_gt_and_outputs_uncertain():
    parameters = list(inspect.signature(infer_track_classifications).parameters)
    assert parameters == ["feature_records", "model", "thresholds", "rule_config"]
    assert not any("ground" in name or "mask" in name or "metadata" in name for name in parameters)
    feature = {
        "track_id": "t", "duration_frames": 4, "valid_observation_count": 3,
        "radius_growth_slope": 0, "post_peak_persistence": .5,
        "expansion_monotonicity": .5, "ringness": 1, "center_drift": 1,
        "decay_rate": .1, "spatial_compactness": .5,
    }
    model = {"feature_names": ["duration_frames"], "feature_mean": np.asarray([4.0]),
             "feature_std": np.asarray([1.0]), "weights": np.asarray([0.0]), "bias": 0,
             "model_version": "test", "feature_schema_version": "test"}
    result = infer_track_classifications([feature], model, {"low_threshold": .4, "high_threshold": .6}, {
        "minimum_track_observations": 2, "water_score_threshold": .48,
        "dry_score_threshold": .48, "minimum_score_margin": .1,
    })[0]
    assert result["classification"] == "uncertain"
    assert result["ground_truth_used"] is False


def test_hybrid_evidence_cannot_erase_rule_mask():
    rule = {"predicted_water_mask": np.asarray([[True, False]]),
            "predicted_water_probability": np.asarray([[.6, 0]], dtype=np.float32),
            "predicted_unknown_mask": np.asarray([[False, True]]),
            "evidence_count_map": np.asarray([[1, 0]], dtype=np.float32)}
    model = {"predicted_water_mask": np.asarray([[False, False]]),
             "predicted_water_probability": np.asarray([[.1, .2]], dtype=np.float32),
             "predicted_unknown_mask": np.asarray([[True, False]]),
             "evidence_count_map": np.asarray([[0, 1]], dtype=np.float32)}
    hybrid, diagnostics = build_rule_preserving_hybrid_evidence(rule, model, {})
    assert np.array_equal(hybrid["predicted_water_mask"], rule["predicted_water_mask"])
    assert diagnostics["hybrid_policy"].startswith("rule_mask_preserved")
