import numpy as np
import pytest

from src.evaluation.analyze_geometry_reprojection_error import (
    boundary_distance_diagnostics,
    build_geometry_diagnostics_summary,
    phase_correlation_translation,
)


def test_boundary_distance_recomputes_p50_p95_without_fake_signed_matches():
    observed = np.zeros((32, 32), dtype=bool)
    observed[8:24, 8:20] = True
    predicted = np.zeros_like(observed)
    predicted[8:24, 10:22] = True
    diagnostics = boundary_distance_diagnostics(observed, np.zeros_like(observed), predicted)
    assert diagnostics["boundary_reprojection_p50_px"] is not None
    assert diagnostics["boundary_reprojection_p95_px"] >= diagnostics["boundary_reprojection_p50_px"]
    assert diagnostics["signed_dx_mean_px"] is None
    assert diagnostics["signed_offset_status"].startswith("unavailable_no_reliable")


def test_phase_correlation_reports_shift_to_apply_to_reprojected_mask():
    observed = np.zeros((64, 64), dtype=bool)
    observed[20:30, 25:38] = True
    reprojected = np.roll(np.roll(observed, 3, axis=0), -4, axis=1)
    result = phase_correlation_translation(observed, np.zeros_like(observed), reprojected)
    assert result["phase_alignment_dx_px"] == pytest.approx(4.0, abs=0.3)
    assert result["phase_alignment_dy_px"] == pytest.approx(-3.0, abs=0.3)
    assert result["phase_alignment_response"] > 0.8
    assert result["phase_alignment_semantics"].startswith("exploratory_global_translation")


def _record(p95, depth, rain, camera_iou, dx=1.0, dy=-2.0, response=0.5):
    return {
        "case_id": f"sim_water_{depth}cm_001", "nominal_depth_cm": depth,
        "rain_level": rain, "seed": depth, "boundary_reprojection_p95_px": p95,
        "camera_mask_iou": camera_iou, "camera_reprojection_iou": 0.9,
        "water_level_absolute_error_m": 0.01, "depth_mae_m": 0.02,
        "unknown_fraction": 0.2, "water_mask_time_stability": 0.8,
        "geometry_reject_reasons": ["boundary_reprojection_error_above_threshold"],
        "phase_alignment_dx_px": dx, "phase_alignment_dy_px": dy,
        "phase_alignment_response": response,
        "phase_alignment_reporting_level_met": response >= 0.2,
        "stored_recomputed_p95_absolute_delta_px": 0.0,
        "full_image_boundary_reprojection_p95_px": p95 - 0.5,
        "unknown_aware_minus_full_image_p95_px": 0.5,
    }


def test_summary_counts_threshold_bands_and_high_iou_without_changing_gate():
    records = [
        _record(3.5, 5, "light", 0.81),
        _record(5.0, 10, "moderate", 0.7),
        _record(8.1, 20, "heavy", 0.9),
        _record(11.0, 40, "heavy", 0.6),
    ]
    summary = build_geometry_diagnostics_summary(records)
    assert summary["p95_distribution"] == {
        "minimum_px": 3.5, "median_px": 6.55, "mean_px": 6.9, "maximum_px": 11.0,
        "above_3_through_4_count": 1, "above_5_count": 2,
        "above_8_count": 2, "above_10_count": 1,
    }
    assert summary["high_camera_iou_geometry_reject_count"] == 2
    assert summary["threshold_px"] == 3.0
    assert summary["threshold_modified"] is False
    assert summary["unknown_aware_domain_comparison"]["unknown_aware_p95_greater_count"] == 4
    assert summary["unknown_aware_domain_comparison"]["comparison_role"].endswith("domain_unchanged")
    assert summary["prediction_rerun"] is False
    assert summary["eligible_for_downstream"] is False


def test_empty_phase_correlation_is_safely_unavailable():
    empty = np.zeros((10, 10), dtype=bool)
    result = phase_correlation_translation(empty, empty, empty)
    assert result["phase_alignment_dx_px"] is None
    assert result["phase_alignment_response"] is None
