"""Tests for the fixed Phase 2D-C-6C GT-free prompt safety rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from src.perception.temporal_water_pipeline import build_temporal_support_fraction
from src.vision.generate_temporal_sam2_prompt import generate_temporal_sam2_prompt


def _config() -> dict:
    return {
        "schema_version": "phase2d_c6_prompt_v1",
        "algorithm_version": "phase2d_c6c_test",
        "connectivity": 8,
        "box_margin_px": 3,
        "partial_gate_box_margin_ratio": 0.5,
        "partial_gate_box_margin_cap_px": 20,
        "min_component_area_px": 16,
        "ambiguous_component_probability_mass_ratio": 0.8,
        "max_ambiguous_component_count": 0,
        "target_positive_points": 5,
        "min_positive_points": 3,
        "min_positive_boundary_distance_px": 2.0,
        "min_positive_spacing_px": 4.0,
        "min_positive_probability": 0.5,
        "target_negative_points": 8,
        "min_negative_points": 6,
        "min_negative_direction_sectors": 4,
        "dry_track_min_confidence": 0.5,
        "use_dry_splash_negatives_when_temporal_partial": False,
        "negative_ring_inner_distance_px": 2.0,
        "negative_ring_outer_distance_px": 8.0,
        "max_box_area_fraction": 0.75,
        "max_box_border_touch_ratio": 0.5,
    }


def _inputs() -> tuple:
    probability = np.zeros((60, 80), dtype=np.float32)
    water = np.zeros_like(probability, dtype=bool)
    water[18:42, 25:55] = True
    probability[water] = 0.8
    unknown = np.zeros_like(water)
    gate = {"status": "pass", "ground_truth_used": False}
    return probability, water, unknown, [], gate


def _run(
    inputs: tuple,
    config: dict | None = None,
    temporal_support_fraction: np.ndarray | None = None,
):
    probability, water, unknown, classifications, gate = inputs
    return generate_temporal_sam2_prompt(
        probability,
        water,
        unknown,
        classifications,
        gate,
        config or _config(),
        temporal_support_fraction=temporal_support_fraction,
        image_path="frame.png",
        image_sha256="c" * 64,
        frame_index=149,
    )


def test_confidence_core_excludes_low_probability_positive_regions() -> None:
    inputs = list(_inputs())
    probability, water = inputs[0], inputs[1]
    probability[water] = 0.42
    probability[25:35, 34:46] = 0.75
    prompt, diagnostics = _run(tuple(inputs))
    assert prompt["prompt_quality_status"] == "pass"
    assert diagnostics["positive_probability_floor"] == 0.5
    assert diagnostics["positive_candidate_count_after_confidence_filter"] > 0
    assert all(probability[y, x] >= 0.5 for x, y in prompt["positive_points_xy"])


def test_insufficient_high_confidence_core_fails_closed() -> None:
    inputs = list(_inputs())
    inputs[0][inputs[1]] = 0.42
    inputs[0][28:32, 37:41] = 0.75
    prompt, diagnostics = _run(tuple(inputs))
    assert prompt["prompt_quality_status"] == "reject"
    assert "insufficient_safe_positive_points" in diagnostics["hard_reasons"]


def test_feasibility_first_selection_finds_three_points_when_greedy_center_blocks() -> None:
    inputs = list(_inputs())
    probability, water = inputs[0], inputs[1]
    probability[:] = 0.0
    # The deepest center point is within 10 px of every outer candidate. A
    # center-first greedy choice therefore stops early, while the three outer
    # candidates form a valid pairwise-spaced set.
    center = (40, 30)
    feasible_outer_points = ((40, 23), (34, 34), (46, 34))
    for x, y in (center, *feasible_outer_points):
        probability[y, x] = 0.9
    config = _config()
    config["min_positive_spacing_px"] = 10.0
    config["positive_point_selection_method"] = (
        "feasibility_preserving_fallback_v1"
    )
    before = tuple(deepcopy(item) for item in inputs)

    prompt, diagnostics = _run(tuple(inputs), config)

    assert prompt["prompt_quality_status"] == "pass"
    assert len(prompt["positive_points_xy"]) >= 3
    selected = np.asarray(prompt["positive_points_xy"], dtype=np.float64)
    pairwise = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=2)
    assert np.all(pairwise[np.triu_indices(len(selected), k=1)] >= 10.0)
    assert diagnostics["positive_point_selection_method"] == (
        "feasibility_preserving_fallback_v1"
    )
    assert diagnostics["positive_feasible_required_set_found"] is True
    assert np.array_equal(inputs[0], before[0])
    assert np.array_equal(inputs[1], before[1])
    assert np.array_equal(inputs[2], before[2])


def test_packing_fallback_preserves_successful_legacy_greedy_points() -> None:
    inputs = _inputs()
    legacy_prompt, legacy_diagnostics = _run(
        tuple(deepcopy(item) for item in inputs)
    )
    config = _config()
    config["positive_point_selection_method"] = (
        "feasibility_preserving_fallback_v1"
    )

    fallback_prompt, fallback_diagnostics = _run(
        tuple(deepcopy(item) for item in inputs), config
    )

    assert legacy_prompt["positive_points_xy"] == fallback_prompt["positive_points_xy"]
    assert legacy_diagnostics["positive_point_selection_method"] == "legacy_greedy_v1"
    assert fallback_diagnostics["positive_point_selection_method"] == "legacy_greedy_v1"


def test_packing_fallback_fails_closed_when_no_feasible_three_point_set_exists() -> None:
    inputs = list(_inputs())
    probability, water = inputs[0], inputs[1]
    probability[:] = 0.0
    for x, y in ((38, 30), (40, 30), (42, 30)):
        probability[y, x] = 0.9
    config = _config()
    config["min_positive_spacing_px"] = 10.0
    config["positive_point_selection_method"] = (
        "feasibility_preserving_fallback_v1"
    )

    prompt, diagnostics = _run(tuple(inputs), config)

    assert prompt["prompt_quality_status"] == "reject"
    assert len(prompt["positive_points_xy"]) < 3
    assert "insufficient_safe_positive_points" in diagnostics["hard_reasons"]
    assert diagnostics["positive_point_selection_method"] == (
        "feasibility_preserving_fallback_v1"
    )
    assert diagnostics["positive_feasible_required_set_found"] is False


def test_partial_gate_expands_box_by_component_scale_with_cap() -> None:
    inputs = list(_inputs())
    pass_prompt, pass_diagnostics = _run(tuple(deepcopy(item) for item in inputs))
    inputs[-1]["status"] = "partial"
    partial_prompt, partial_diagnostics = _run(tuple(inputs))
    assert partial_diagnostics["box_expansion_policy"] == "partial_gate_component_scaled"
    assert partial_diagnostics["box_margin_xy_px"] == [15, 12]
    assert partial_prompt["box_xyxy"][0] < pass_prompt["box_xyxy"][0]
    assert partial_prompt["box_xyxy"][2] > pass_prompt["box_xyxy"][2]
    assert partial_prompt["prompt_quality_status"] == "diagnostic_only"


def test_partial_gate_disables_dry_splash_negatives_but_keeps_ring_negatives() -> None:
    inputs = list(_inputs())
    inputs[-1]["status"] = "partial"
    inputs[3] = [
        {
            "classification": "dry_splash",
            "confidence": 0.99,
            "duration_frames": 8,
            "center_mean": [20, 30],
        }
    ]
    prompt, diagnostics = _run(tuple(inputs))
    assert diagnostics["dry_splash_negatives_allowed"] is False
    assert "dry_splash_track" not in diagnostics["negative_point_sources"]
    assert len(prompt["negative_points_xy"]) >= 6


def test_recurrent_prompt_support_can_corroborate_only_the_allowed_partial_reason() -> None:
    inputs = list(_inputs())
    inputs[-1] = {
        "status": "partial",
        "reasons": ["insufficient_high_confidence_water_tracks"],
        "observable_region_result_valid": True,
        "ground_truth_used": False,
    }
    config = _config()
    config["min_positive_temporal_support_fraction"] = 2.0 / 3.0
    config["allow_temporally_corroborated_partial_gate"] = True
    config["corroboratable_partial_gate_reasons"] = [
        "insufficient_high_confidence_water_tracks"
    ]
    support = np.ones(inputs[0].shape, dtype=np.float32)

    prompt, diagnostics = _run(tuple(inputs), config, support)

    assert prompt["prompt_quality_status"] == "pass"
    assert prompt["partial_gate_corroboration_applied"] is True
    assert prompt["upstream_temporal_quality_gate_status"] == "partial"
    assert prompt["authoritative"] is False
    assert prompt["eligible_for_downstream"] is False
    assert diagnostics["partial_gate_corroboration_applied"] is True
    assert diagnostics["temporal_quality_gate_status"] == "partial"
    assert diagnostics["informational_reasons"] == [
        "partial_temporal_gate_corroborated_by_recurrent_prompt_support"
    ]


def test_recurrent_support_does_not_override_an_unapproved_partial_reason() -> None:
    inputs = list(_inputs())
    inputs[-1] = {
        "status": "partial",
        "reasons": ["water_evidence_coverage_too_low"],
        "observable_region_result_valid": True,
        "ground_truth_used": False,
    }
    config = _config()
    config["min_positive_temporal_support_fraction"] = 2.0 / 3.0
    config["allow_temporally_corroborated_partial_gate"] = True
    config["corroboratable_partial_gate_reasons"] = [
        "insufficient_high_confidence_water_tracks"
    ]
    support = np.ones(inputs[0].shape, dtype=np.float32)

    prompt, diagnostics = _run(tuple(inputs), config, support)

    assert prompt["prompt_quality_status"] == "diagnostic_only"
    assert prompt["partial_gate_corroboration_applied"] is False
    assert diagnostics["partial_gate_corroboration_applied"] is False
    assert diagnostics["diagnostic_reasons"] == [
        "temporal_quality_gate_not_pass"
    ]


def test_legacy_config_without_c6c_keys_preserves_baseline_behavior() -> None:
    inputs = _inputs()
    legacy = _config()
    for key in (
        "min_positive_probability",
        "partial_gate_box_margin_ratio",
        "partial_gate_box_margin_cap_px",
        "use_dry_splash_negatives_when_temporal_partial",
    ):
        legacy.pop(key)
    prompt, diagnostics = _run(inputs, legacy)
    assert prompt["prompt_quality_status"] == "pass"
    assert diagnostics["positive_probability_floor"] == 0.0
    assert diagnostics["box_margin_xy_px"] == [3, 3]


def test_c6c_rules_are_deterministic_and_gt_free() -> None:
    inputs = _inputs()
    first = _run(tuple(deepcopy(item) for item in inputs))
    second = _run(tuple(deepcopy(item) for item in inputs))
    assert first == second
    prompt, diagnostics = first
    assert prompt["ground_truth_used"] is False
    assert diagnostics["ground_truth_used"] is False


def test_temporal_support_filter_keeps_only_recurrent_positive_core() -> None:
    inputs = _inputs()
    support = np.full(inputs[0].shape, 1.0 / 3.0, dtype=np.float32)
    support[22:38, 31:49] = 2.0 / 3.0
    config = _config()
    config["min_positive_temporal_support_fraction"] = 2.0 / 3.0
    prompt, diagnostics = _run(inputs, config, support)
    assert prompt["prompt_quality_status"] == "pass"
    assert diagnostics["temporal_support_available"] is True
    assert (
        diagnostics["positive_candidate_count_after_temporal_support_filter"]
        < diagnostics["positive_candidate_count_after_confidence_filter"]
    )
    assert all(
        support[y, x] >= 2.0 / 3.0
        for x, y in prompt["positive_points_xy"]
    )
    assert all(
        value >= 2.0 / 3.0
        for value in diagnostics["positive_point_temporal_support_fractions"]
    )


def test_insufficient_recurrent_positive_core_fails_closed() -> None:
    inputs = _inputs()
    support = np.full(inputs[0].shape, 1.0 / 3.0, dtype=np.float32)
    config = _config()
    config["min_positive_temporal_support_fraction"] = 2.0 / 3.0
    prompt, diagnostics = _run(inputs, config, support)
    assert prompt["prompt_quality_status"] == "reject"
    assert "insufficient_safe_positive_points" in diagnostics["hard_reasons"]
    assert diagnostics["positive_candidate_count_after_temporal_support_filter"] == 0


def test_temporal_support_threshold_fails_closed_when_map_is_unavailable() -> None:
    inputs = _inputs()
    config = _config()
    config["min_positive_temporal_support_fraction"] = 2.0 / 3.0
    prompt, diagnostics = _run(inputs, config)
    assert prompt["prompt_quality_status"] == "reject"
    assert "temporal_support_unavailable" in diagnostics["hard_reasons"]


def test_default_temporal_support_setting_preserves_existing_result() -> None:
    inputs = _inputs()
    without_map = _run(tuple(deepcopy(item) for item in inputs))
    support = np.zeros(inputs[0].shape, dtype=np.float32)
    with_ignored_map = _run(tuple(deepcopy(item) for item in inputs), _config(), support)
    assert without_map[0]["positive_points_xy"] == with_ignored_map[0]["positive_points_xy"]
    assert without_map[0]["prompt_quality_status"] == with_ignored_map[0]["prompt_quality_status"]


def test_three_window_support_distinguishes_recurrent_from_transient_evidence() -> None:
    def ripple(center: list[float], frame: int) -> dict:
        return {
            "classification": "water_ripple",
            "center_mean": center,
            "maximum_area": 25,
            "confidence": 0.9,
            "duration_frames": 5,
            "start_frame": frame,
            "end_frame": frame + 4,
        }

    classifications = [
        ripple([20.0, 20.0], 3),
        ripple([20.0, 20.0], 33),
        ripple([20.0, 20.0], 63),
        ripple([65.0, 40.0], 6),
    ]
    support, diagnostics = build_temporal_support_fraction(
        classifications,
        (60, 90),
        {
            "minimum_kernel_sigma_px": 4.0,
            "maximum_propagation_radius_px": 22.0,
            "probability_scale": 2.0,
            "water_probability_threshold": 0.36,
            "unknown_evidence_threshold": 0.08,
            "morphology_close_kernel": 5,
        },
        frame_count=90,
    )
    assert diagnostics["window_count"] == 3
    assert diagnostics["ground_truth_used"] is False
    assert support[20, 20] == 1.0
    assert np.isclose(support[40, 65], 1.0 / 3.0)


def test_matrix_runner_accepts_explicit_prompt_config_without_changing_legacy_default() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_temporal_sam2_prompt_matrix.py"
    ).read_text(encoding="utf-8")
    assert '"--prompt-config"' in script
    assert '"temporal_sam2_prompt.yaml"' in script
    assert '"--config", str(args.prompt_config.resolve())' in script
