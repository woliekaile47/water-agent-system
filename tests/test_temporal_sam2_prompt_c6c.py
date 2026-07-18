"""Tests for the fixed Phase 2D-C-6C GT-free prompt safety rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

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


def _run(inputs: tuple, config: dict | None = None):
    probability, water, unknown, classifications, gate = inputs
    return generate_temporal_sam2_prompt(
        probability,
        water,
        unknown,
        classifications,
        gate,
        config or _config(),
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


def test_matrix_runner_accepts_explicit_prompt_config_without_changing_legacy_default() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_temporal_sam2_prompt_matrix.py"
    ).read_text(encoding="utf-8")
    assert '"--prompt-config"' in script
    assert '"temporal_sam2_prompt.yaml"' in script
    assert '"--config", str(args.prompt_config.resolve())' in script
