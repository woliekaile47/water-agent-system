import numpy as np

from src.evaluation.audit_shoreline_cases import (
    boundary_spatial_diagnostics,
    select_audit_cases,
    temporal_curve_diagnostics,
)


def _record(case, p95, iou, seed=1):
    return {
        "case_id": case, "rain_level": "moderate", "seed": seed,
        "boundary_reprojection_p95_px": p95, "camera_mask_iou": iou,
        "geometry_gate_status": "reject",
    }


def test_selection_groups_overlap_but_union_is_deduplicated():
    records = [
        _record("sim_water_5cm_001", 3.5, 0.9),
        _record("sim_water_10cm_001", 12.0, 0.85),
        _record("sim_water_20cm_001", 6.0, 0.81),
        _record("sim_water_40cm_001", 5.0, 0.7),
    ]
    selection = select_audit_cases(records)
    assert selection["group_counts"] == {
        "boundary_p95_above_10_px": 1,
        "boundary_p95_above_3_through_4_px": 1,
        "camera_iou_at_least_0_8_geometry_reject": 3,
    }
    assert selection["selected_unique_count"] == 3
    overlapping = next(item for item in selection["selected_cases"] if item["case_id"] == "sim_water_5cm_001")
    assert len(overlapping["selection_groups"]) == 2


def test_boundary_audit_detects_unknown_touch_and_components_without_point_matching():
    observed = np.zeros((32, 32), dtype=bool)
    observed[8:24, 8:20] = True
    observed[2:4, 2:4] = True
    reprojected = np.zeros_like(observed)
    reprojected[8:24, 10:22] = True
    unknown = np.zeros_like(observed)
    unknown[7, 8:20] = True
    metrics, arrays = boundary_spatial_diagnostics(observed, unknown, reprojected)
    assert metrics["observed_components"]["component_count"] == 2
    assert metrics["observed_components"]["components_at_most_100px_count"] == 1
    assert metrics["unknown_touching_raw_water_boundary_fraction"] > 0
    assert metrics["boundary_distance_p95_px"] is not None
    assert arrays["forward_distance_map"].shape == observed.shape


def test_temporal_curves_do_not_fake_per_frame_water_mask_or_shoreline():
    temporal = {
        "loader": {"frame_count": 4},
        "candidates": {"candidate_count_by_frame": [1, 2, 3, 4]},
        "classifications": [{"track_id": "water_track", "classification": "water_ripple"}],
        "tracks": [{
            "track_id": "water_track",
            "observations": [{"frame_index": 1, "area": 5}, {"frame_index": 3, "area": 7}],
        }],
        "water_mask_time_stability": 0.25,
    }
    result = temporal_curve_diagnostics(temporal)
    assert result["candidate_count_curve_available"] is True
    assert result["classified_water_track_observation_area_proxy_by_frame"] == [0, 5, 0, 7]
    assert result["temporal_water_mask_area_curve_available"] is False
    assert result["temporal_shoreline_stability_curve_available"] is False
    assert "not_a_water_mask_area" in result["classified_water_track_area_proxy_semantics"]
