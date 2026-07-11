from src.perception.rain_impact_feature_extraction import extract_track_features


def test_expanding_track_has_positive_growth_and_required_features():
    observations = [
        {"ringness": 1.2, "compactness": 0.8},
        {"ringness": 1.3, "compactness": 0.75},
        {"ringness": 1.4, "compactness": 0.7},
    ]
    track = {
        "track_id": "track_0", "duration_frames": 3, "valid_observation_count": 3,
        "area_sequence": [5, 15, 30], "equivalent_radius_sequence": [1.2, 2.2, 3.1],
        "intensity_sequence": [5.0, 10.0, 7.0], "polarity_sequence": ["positive"] * 3,
        "center_drift": 0.5, "center_mean": [10.0, 10.0], "bbox_union": [5, 5, 15, 15],
        "start_frame": 0, "end_frame": 2, "observations": observations,
    }
    features = extract_track_features(track)
    assert features["radius_growth_slope"] > 0
    assert features["expansion_monotonicity"] == 1.0
    assert "temporal_asymmetry" in features
    assert "radial_energy_profile" in features
