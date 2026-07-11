import numpy as np

from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.integration.integration_quality_gate import evaluate_integration_quality_gate
from src.integration.unknown_aware_geometry import build_trusted_shoreline


def test_unknown_geometry_and_joint_gate_are_deterministic():
    water = np.zeros((6, 6), dtype=bool)
    unknown = np.zeros_like(water)
    water[2:4, 2:4] = True
    unknown[2:4, 4] = True
    first_shoreline = build_trusted_shoreline(water, unknown)
    second_shoreline = build_trusted_shoreline(water, unknown)
    assert np.array_equal(first_shoreline[0], second_shoreline[0])
    assert first_shoreline[1:] == second_shoreline[1:]
    intersections = [
        {"component_index": 0, "dem_height_m": value}
        for value in (.10, .11, .09, .105, .095)
    ]
    config = {"method": "robust_median", "min_samples_per_component": 3,
              "outlier_method": "mad", "mad_threshold": 3.5,
              "iqr_multiplier": 1.5, "trim_fraction": .1}
    first_level = estimate_water_level_from_shoreline(intersections, config)
    second_level = estimate_water_level_from_shoreline(intersections, config)
    assert first_level == second_level
    visual = {"status": "pass", "reasons": []}
    geometric = {"status": "pass", "reasons": [], "global_estimate_status": "complete",
                 "observable_region_result_valid": True, "result_semantics": "global_estimate",
                 "area_volume_semantics": "complete_estimate"}
    assert evaluate_integration_quality_gate(visual, geometric) == evaluate_integration_quality_gate(visual, geometric)
