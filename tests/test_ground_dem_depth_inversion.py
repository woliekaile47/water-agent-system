import numpy as np

from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem


def test_depth_mask_area_volume_and_invalid_dem_rules():
    dem = np.asarray([[0.0, 0.1, np.nan], [0.2, 0.3, 0.4]], dtype=np.float32)
    mask = np.asarray([[True, True, True], [False, True, False]])
    depth, valid_water, result = invert_depth_from_ground_dem(dem, mask, 0.25, 0.5)
    assert np.allclose(depth[np.isfinite(depth)], [0.25, 0.15, 0.0, 0.0, 0.0])
    assert np.isnan(depth[0, 2])
    assert depth[1, 0] == 0.0 and depth[1, 2] == 0.0
    assert not np.any(depth[np.isfinite(depth)] < 0)
    assert np.count_nonzero(valid_water) == 3
    assert result["water_area_m2"] == 3 * 0.25
    assert np.isclose(result["water_volume_m3"], (0.25 + 0.15) * 0.25)
    assert result["invalid_dem_in_mask_count"] == 1
