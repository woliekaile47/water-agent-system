from copy import deepcopy

import numpy as np

from src.vision.generate_temporal_sam2_prompt import generate_temporal_sam2_prompt


def _config():
    return {
        "schema_version": "phase2d_c6_prompt_v1",
        "algorithm_version": "test",
        "connectivity": 8,
        "box_margin_px": 3,
        "min_component_area_px": 16,
        "ambiguous_component_probability_mass_ratio": 0.8,
        "max_ambiguous_component_count": 0,
        "target_positive_points": 5,
        "min_positive_points": 3,
        "min_positive_boundary_distance_px": 2.0,
        "min_positive_spacing_px": 4.0,
        "target_negative_points": 8,
        "min_negative_points": 6,
        "min_negative_direction_sectors": 4,
        "dry_track_min_confidence": 0.5,
        "negative_ring_inner_distance_px": 2.0,
        "negative_ring_outer_distance_px": 8.0,
        "max_box_area_fraction": 0.75,
        "max_box_border_touch_ratio": 0.5,
    }


def _inputs():
    probability = np.zeros((48, 64), dtype=np.float32)
    water = np.zeros_like(probability, dtype=bool)
    water[15:34, 20:45] = True
    probability[water] = 0.9
    unknown = np.zeros_like(water)
    gate = {"status": "pass", "ground_truth_used": False}
    return probability, water, unknown, [], gate


def _run(inputs=None, config=None):
    probability, water, unknown, classifications, gate = inputs or _inputs()
    return generate_temporal_sam2_prompt(
        probability,
        water,
        unknown,
        classifications,
        gate,
        config or _config(),
        image_path="frame.png",
        image_sha256="a" * 64,
        frame_index=49,
    )


def test_prompt_is_deterministic_and_points_obey_semantics():
    inputs = _inputs()
    first = _run(inputs)
    second = _run(tuple(deepcopy(item) for item in inputs))
    assert first == second
    prompt, diagnostics = first
    assert prompt["prompt_quality_status"] == "pass"
    assert prompt["semantic_label"] == "unknown_candidate"
    assert prompt["authoritative"] is False
    assert prompt["ground_truth_used"] is False
    assert prompt["eligible_for_downstream"] is False
    _, water, unknown, _, _ = inputs
    assert all(water[y, x] and not unknown[y, x] for x, y in prompt["positive_points_xy"])
    assert all(not water[y, x] and not unknown[y, x] for x, y in prompt["negative_points_xy"])
    assert diagnostics["negative_direction_sector_count"] >= 6


def test_dry_track_negatives_are_directionally_deduplicated_before_fill():
    probability, water, unknown, _, gate = _inputs()
    classifications = [
        {"classification": "dry_splash", "confidence": 0.9 - index * 0.01,
         "duration_frames": 5, "center_mean": [48 + index, 24]}
        for index in range(6)
    ]
    prompt, diagnostics = _run((probability, water, unknown, classifications, gate))
    assert prompt["prompt_quality_status"] == "pass"
    assert diagnostics["negative_direction_sector_count"] >= 4
    assert "known_nonwater_ring" in diagnostics["negative_point_sources"]


def test_dry_empty_mask_rejects_without_positive_points():
    probability, water, unknown, classifications, gate = _inputs()
    water[:] = False
    prompt, diagnostics = _run((probability, water, unknown, classifications, gate))
    assert prompt["prompt_quality_status"] == "reject"
    assert prompt["positive_points_xy"] == []
    assert "predicted_water_mask_empty" in diagnostics["hard_reasons"]


def test_primary_component_uses_probability_mass_then_area():
    probability, water, unknown, classifications, gate = _inputs()
    water[:] = False
    probability[:] = 0
    water[10:30, 4:16] = True
    probability[10:30, 4:16] = 0.2
    water[14:26, 40:54] = True
    probability[14:26, 40:54] = 0.9
    config = _config()
    config["ambiguous_component_probability_mass_ratio"] = 0.99
    prompt, diagnostics = _run((probability, water, unknown, classifications, gate), config)
    assert diagnostics["selected_component_area_pixels"] == 12 * 14
    assert all(x >= 40 for x, _ in prompt["positive_points_xy"])


def test_near_equal_components_trigger_ambiguity_reject():
    probability, water, unknown, classifications, gate = _inputs()
    water[:] = False
    probability[:] = 0
    water[10:25, 5:20] = True
    water[10:25, 40:55] = True
    probability[water] = 0.8
    prompt, diagnostics = _run((probability, water, unknown, classifications, gate))
    assert prompt["prompt_quality_status"] == "reject"
    assert diagnostics["ambiguous_component_count"] == 1
    assert "ambiguous_temporal_water_components" in diagnostics["hard_reasons"]


def test_unknown_pixels_are_never_used_as_negative_points():
    probability, water, unknown, classifications, gate = _inputs()
    unknown[:] = True
    unknown[water] = False
    prompt, diagnostics = _run((probability, water, unknown, classifications, gate))
    assert prompt["negative_points_xy"] == []
    assert prompt["prompt_quality_status"] == "reject"
    assert "insufficient_trusted_negative_points" in diagnostics["hard_reasons"]


def test_enclosed_hole_is_not_used_as_known_nonwater_negative():
    probability, water, unknown, classifications, gate = _inputs()
    water[22:26, 30:34] = False
    probability[22:26, 30:34] = 0
    prompt, _ = _run((probability, water, unknown, classifications, gate))
    assert prompt["prompt_quality_status"] == "pass"
    assert all(not (30 <= x < 34 and 22 <= y < 26) for x, y in prompt["negative_points_xy"])


def test_temporal_reject_is_inherited_and_input_arrays_are_not_modified():
    inputs = _inputs()
    probability_before = inputs[0].copy()
    water_before = inputs[1].copy()
    unknown_before = inputs[2].copy()
    inputs[-1]["status"] = "reject"
    prompt, diagnostics = _run(inputs)
    assert prompt["prompt_quality_status"] == "reject"
    assert "temporal_quality_gate_reject" in diagnostics["hard_reasons"]
    assert np.array_equal(inputs[0], probability_before)
    assert np.array_equal(inputs[1], water_before)
    assert np.array_equal(inputs[2], unknown_before)
