import numpy as np

from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline


def test_robust_shoreline_level_needs_no_true_water_level_and_is_repeatable():
    values = [0.098, 0.100, 0.101, 0.099, 0.102, 3.0] * 3
    records = [{"component_index": 0, "dem_height_m": value} for value in values]
    config = {"method": "robust_median", "min_samples_per_component": 5, "outlier_method": "mad", "mad_threshold": 3.5}
    first = estimate_water_level_from_shoreline(records, config)
    second = estimate_water_level_from_shoreline(records, config)
    assert first == second
    assert np.isclose(first[0], 0.1, atol=0.002)
    assert first[1]["valid_shoreline_sample_count"] >= 5
    assert "ground_truth" not in first[1]


def test_median_and_trimmed_median_are_supported():
    records = [{"component_index": 0, "dem_height_m": value} for value in np.linspace(0.09, 0.11, 20)]
    for method in ("median", "trimmed_median"):
        level, diagnostics = estimate_water_level_from_shoreline(
            records,
            {"method": method, "min_samples_per_component": 5, "outlier_method": "none", "trim_fraction": 0.1},
        )
        assert np.isclose(level, 0.1)
        assert diagnostics["method"] == method
