import numpy as np

from src.fusion.water_surface_aware_mask_to_dem import camera_reprojection_consistency, reconstruct_connected_lowland


def test_flood_fill_does_not_cross_high_ground_barrier():
    dem = np.zeros((5, 7), dtype=np.float32)
    dem[:, 3] = 1.0
    seeds = np.zeros_like(dem, dtype=bool)
    seeds[2, 1] = True
    predicted, diagnostics = reconstruct_connected_lowland(
        dem, 0.5, seeds,
        {"connectivity": 4, "lowland_margin_m": 0.0, "min_seed_cells_per_basin": 1, "max_selected_basins": 1},
    )
    assert np.all(predicted[:, :3])
    assert not np.any(predicted[:, 3:])
    assert np.count_nonzero(predicted) == 15
    assert diagnostics["candidate_basin_count"] == 2
    assert diagnostics["selected_basin_count"] == 1
    assert diagnostics["seed_valid"] is True


def test_reprojection_consistency_is_exact_for_identical_masks():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 8:22] = 255
    metrics = camera_reprojection_consistency(mask, mask.copy(), {"water_surface_projection_coverage": 1.0})
    assert metrics["camera_reprojection_iou"] == 1.0
    assert metrics["camera_reprojection_precision"] == 1.0
    assert metrics["camera_reprojection_recall"] == 1.0
    assert metrics["boundary_reprojection_p95_px"] == 0.0
