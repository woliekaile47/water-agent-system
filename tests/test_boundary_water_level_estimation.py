import numpy as np
import pytest

from src.hydrology.estimate_water_level_from_boundary import estimate_water_level_from_boundary


def base_config(method):
    return {
        "method": method, "connectivity": 8, "min_component_cells": 5,
        "min_boundary_samples": 5, "outer_ring_cells": 1,
        "outlier_method": "mad", "mad_threshold": 3.5,
        "inner_high_quantile": 0.9, "outer_low_quantile": 0.1,
        "bracket_order_tolerance_m": 0.02, "max_bracket_width_m": 0.2,
    }


@pytest.mark.parametrize("method", [
    "inner_boundary_median", "inner_boundary_high_quantile", "inner_outer_bracket_midpoint"
])
def test_supported_estimators_are_finite_and_do_not_need_true_level(method):
    yy, xx = np.mgrid[-5:6, -5:6]
    radius = np.sqrt(xx * xx + yy * yy)
    dem = (radius * 0.01).astype(np.float32)
    mask = radius <= 3.5
    level, cleaned, diagnostics = estimate_water_level_from_boundary(mask, dem, base_config(method))
    assert np.isfinite(level)
    assert np.array_equal(cleaned, mask)
    assert diagnostics["valid_boundary_sample_count"] >= 5
    assert diagnostics["method"] == method


def test_small_noise_component_is_removed_deterministically():
    dem = np.tile(np.linspace(0.0, 0.1, 20), (20, 1)).astype(np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[0, 0] = True
    args = (mask, dem, base_config("inner_boundary_median"))
    first = estimate_water_level_from_boundary(*args)
    second = estimate_water_level_from_boundary(*args)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert not first[1][0, 0]
    assert first[2]["removed_small_component_count"] == 1
