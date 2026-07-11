from src.perception.rain_impact_classifier import classify_track


CONFIG = {
    "minimum_track_observations": 2, "water_score_threshold": 0.45,
    "dry_score_threshold": 0.45, "minimum_score_margin": 0.08,
}


def _features(**overrides):
    value = {
        "track_id": "track", "duration_frames": 18, "valid_observation_count": 15,
        "radius_growth_slope": 0.8, "post_peak_persistence": 0.9,
        "expansion_monotonicity": 0.9, "ringness": 1.8, "center_drift": 1.0,
        "decay_rate": 0.02, "spatial_compactness": 0.8,
    }
    value.update(overrides)
    return value


def test_rule_classifier_separates_explainable_examples():
    water = classify_track(_features(), CONFIG)
    dry = classify_track(_features(
        duration_frames=2, valid_observation_count=2, radius_growth_slope=0.0,
        post_peak_persistence=0.0, expansion_monotonicity=0.0, ringness=0.6,
        center_drift=8.0, decay_rate=0.3, spatial_compactness=0.2,
    ), CONFIG)
    assert water["classification"] == "water_ripple"
    assert dry["classification"] == "dry_splash"
    assert not water["ground_truth_used"]
