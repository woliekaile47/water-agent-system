"""Tests for GT-free temporal stabilization of frozen SAM2 video masks."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.vision import temporal_sam2_mask_stabilization as stabilization


CONFIG = {
    "window": {
        "mode": "centered",
        "size": 5,
        "support_fraction": 0.8,
        "edge_policy": "available_frames",
    },
    "positive_support": {
        "required": True,
        "minimum_count": 1,
        "failure_policy": "empty_mask_reject",
    },
}


def _mask() -> np.ndarray:
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:8, 4:10] = True
    return mask


def test_fixed_window_removes_one_frame_fragment_without_mutation() -> None:
    masks = [_mask() for _ in range(5)]
    masks[2][0, 0] = True
    before = [mask.copy() for mask in masks]
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[6, 5]], dtype=np.float64), CONFIG
    )
    assert len(results) == 5
    assert results[2]["window_frame_count"] == 5
    assert results[2]["required_support_count"] == 4
    assert not results[2]["mask"][0, 0]
    assert results[2]["supported_positive_count"] == 1
    assert results[2]["eligible_for_geometry"] is True
    assert all(np.array_equal(mask, expected) for mask, expected in zip(masks, before))


def test_edges_use_actual_available_frame_count() -> None:
    masks = [_mask() for _ in range(5)]
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[6, 5]], dtype=np.float64), CONFIG
    )
    assert results[0]["window_frame_count"] == 3
    assert results[0]["required_support_count"] == 3
    assert results[1]["window_frame_count"] == 4
    assert results[1]["required_support_count"] == 4
    assert results[2]["required_support_count"] == 4


def test_empty_sequence_masks_remain_empty() -> None:
    masks = [np.zeros((8, 9), dtype=bool) for _ in range(3)]
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[4, 4]], dtype=np.float64), CONFIG
    )
    assert all(not np.any(result["mask"]) for result in results)
    assert all(result["status"] == "reject_empty_raw_mask" for result in results)
    assert all(result["fail_closed"] is True for result in results)
    assert all(result["eligible_for_geometry"] is False for result in results)


def test_empty_raw_frame_is_not_filled_from_nonempty_neighbors() -> None:
    masks = [_mask() for _ in range(7)]
    masks[3][:] = False
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[6, 5]], dtype=np.float64), CONFIG
    )
    assert not np.any(results[3]["mask"])
    assert results[3]["fail_closed"] is True
    assert results[3]["failure_reason"] == "mask_stabilization_empty_raw_mask"
    assert np.any(results[0]["mask"])
    assert np.any(results[6]["mask"])


def test_component_selection_prefers_positive_support_over_larger_area() -> None:
    masks = []
    for _ in range(5):
        mask = np.zeros((20, 24), dtype=bool)
        mask[2:12, 2:12] = True
        mask[14:18, 16:21] = True
        masks.append(mask)
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[18, 16]], dtype=np.float64), CONFIG
    )
    selected = results[2]["mask"]
    assert results[2]["candidate_component_count"] == 2
    assert results[2]["supported_positive_count"] == 1
    assert results[2]["selected_component_area_pixels"] == 20
    assert results[2]["excluded_candidate_pixels"] == 100
    assert selected[16, 18]
    assert not selected[5, 5]


def test_positive_evidence_loss_fails_closed() -> None:
    masks = [_mask() for _ in range(5)]
    results = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[1, 1]], dtype=np.float64), CONFIG
    )
    assert all(result["fail_closed"] is True for result in results)
    assert all(result["eligible_for_geometry"] is False for result in results)
    assert all(not np.any(result["mask"]) for result in results)
    assert all(result["failure_reason"] == "mask_stabilization_positive_support_lost" for result in results)


def test_result_is_deterministic_and_config_is_strict() -> None:
    masks = [_mask() for _ in range(5)]
    first = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[6, 5]], dtype=np.float64), CONFIG
    )
    second = stabilization.stabilize_frozen_mask_sequence(
        masks, np.asarray([[6, 5]], dtype=np.float64), CONFIG
    )
    assert all(np.array_equal(a["mask"], b["mask"]) for a, b in zip(first, second))
    invalid = {**CONFIG, "window": {**CONFIG["window"], "size": 4}}
    with pytest.raises(ValueError, match="odd"):
        stabilization.validate_stabilization_config(invalid)


def test_mask_content_hash_is_deterministic_and_shape_sensitive() -> None:
    mask = _mask()
    assert stabilization.mask_content_sha256(mask) == stabilization.mask_content_sha256(
        mask.copy()
    )
    changed = mask.copy()
    changed[0, 0] = True
    assert stabilization.mask_content_sha256(mask) != stabilization.mask_content_sha256(
        changed
    )
    reshaped = mask.reshape((12, 10))
    assert stabilization.mask_content_sha256(mask) != stabilization.mask_content_sha256(
        reshaped
    )


def test_prediction_module_has_no_evaluation_input_dependency() -> None:
    source = inspect.getsource(stabilization)
    for forbidden in (
        "camera_water_mask_gt",
        "water_level_gt",
        "depth_map_gt",
        "nominal_depth_cm",
        "load_ground_truth",
    ):
        assert forbidden not in source
