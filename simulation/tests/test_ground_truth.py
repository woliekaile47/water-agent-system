from pathlib import Path

import numpy as np

from waterlogging_simulation.config import REQUIRED_CASES, load_configuration, resolve_scenario
from waterlogging_simulation.geometry import dem_grid, water_ground_truth


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_ground_truth_invariants_for_all_scenarios():
    sensors, scenarios = load_configuration(CONFIG_DIR)
    _, _, ground_dem = dem_grid(sensors)
    cell_area = float(sensors["road"]["dem_resolution_m"]) ** 2
    for case_id in REQUIRED_CASES:
        scenario = resolve_scenario(scenarios, case_id)
        water_level, mask, depth, area, volume = water_ground_truth(
            ground_dem, scenario, sensors
        )
        assert depth.dtype == np.float32
        assert np.all(depth >= 0.0)
        assert np.all(depth[~mask] == 0.0)
        assert np.isclose(area, np.count_nonzero(mask) * cell_area)
        assert np.isclose(volume, np.sum(depth.astype(np.float64)) * cell_area)
        if scenario["water_depth_cm"] == 0:
            assert water_level is None
            assert not np.any(mask)
        else:
            assert water_level is not None
            assert np.isclose(float(np.max(depth)), scenario["water_depth_cm"] / 100.0)


def test_geometric_truth_is_repeatable():
    sensors, scenarios = load_configuration(CONFIG_DIR)
    _, _, ground_dem_a = dem_grid(sensors)
    _, _, ground_dem_b = dem_grid(sensors)
    scenario = resolve_scenario(scenarios, "sim_water_20cm_001")
    truth_a = water_ground_truth(ground_dem_a, scenario, sensors)
    truth_b = water_ground_truth(ground_dem_b, scenario, sensors)
    assert np.array_equal(ground_dem_a, ground_dem_b)
    assert truth_a[0] == truth_b[0]
    assert np.array_equal(truth_a[1], truth_b[1])
    assert np.array_equal(truth_a[2], truth_b[2])
    assert truth_a[3:] == truth_b[3:]
