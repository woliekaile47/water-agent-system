import numpy as np

from src.integration import unknown_aware_geometry as geometry


def test_unknown_unobserved_seed_and_unsupported_basin_semantics(monkeypatch):
    dem = np.ones((5, 9), dtype=np.float64)
    for row, col in ((1, 1), (1, 4), (1, 7), (3, 7)):
        dem[row, col] = 0.0
    seeds = np.zeros_like(dem, dtype=bool)
    seeds[1, 1] = True
    observed_water = np.zeros((4, 4), dtype=bool)
    observed_unknown = np.zeros_like(observed_water)
    observed_water[1, 1] = True
    observed_unknown[2, 2] = True

    def fake_reproject(component, level, sensors):
        image = np.zeros((4, 4), dtype=np.uint8)
        row, col = np.argwhere(component)[0]
        if (row, col) == (1, 1):
            image[1, 1] = 255
        elif (row, col) == (1, 4):
            image[2, 2] = 255
        elif (row, col) == (3, 7):
            image[0, 0] = 255
        return image, {"water_surface_projection_coverage": 1.0}

    monkeypatch.setattr(geometry, "reproject_water_surface", fake_reproject)
    predicted, diagnostics = geometry.reconstruct_connected_lowland_unknown_aware(
        dem, .5, seeds,
        {"connectivity": 8, "min_seed_cells_per_basin": 1,
         "min_candidate_camera_precision": .9, "min_candidate_camera_overlap_pixels": 1,
         "ambiguous_candidate_camera_precision": .2},
        observed_water, observed_unknown, {},
    )
    assert predicted[1, 1]
    assert np.count_nonzero(predicted) == 1
    assert diagnostics["selected_basin_count"] == 1
    assert diagnostics["unknown_only_candidate_basin_count"] == 1
    assert diagnostics["unobserved_candidate_basin_count"] == 1
    statuses = {item["observation_status"] for item in diagnostics["candidate_camera_support"]}
    assert "unknown_only" in statuses
    assert "unobserved" in statuses
    assert diagnostics["ambiguous_candidate_basin_count"] == 2
    assert diagnostics["seed_valid"] is True
