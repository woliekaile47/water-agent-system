"""Tests for GT-free cross-window temporal evidence fusion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import src.vision.temporal_multi_window_evidence_fusion as fusion_module
from src.vision.temporal_multi_window_evidence_fusion import (
    fuse_prediction_candidates,
    load_multi_window_fusion_policy,
    run_temporal_multi_window_fusion,
    sha256_file,
)


def _policy() -> dict:
    return {
        "schema_version": "test_v1",
        "algorithm_version": "test_v1",
        "minimum_eligible_window_count": 3,
        "allowed_partial_gate_reasons": ["insufficient_high_confidence_water_tracks"],
        "minimum_window_component_area_pixels": 4,
        "maximum_window_water_fraction": 0.5,
        "window_ambiguous_component_area_ratio": 0.8,
        "maximum_window_ambiguous_component_count": 0,
        "minimum_component_iou": 0.1,
        "maximum_centroid_distance_pixels": 8.0,
        "cluster_overlap_tolerance_radius_pixels": 1,
        "minimum_cluster_window_count": 3,
        "minimum_cluster_time_span_seconds": 15.0,
        "ambiguous_cluster_window_count_ratio": 0.8,
        "minimum_support_window_count": 3,
        "minimum_support_fraction": 0.5,
        "minimum_raw_support_window_count": 1,
        "minimum_final_core_overlap_fraction": 0.1,
        "support_tolerance_radius_pixels": 0,
        "probability_weight": 0.5,
        "support_weight": 0.5,
        "minimum_fused_confidence": 0.5,
        "morphology_close_kernel": 1,
        "minimum_fused_component_area_pixels": 4,
        "fused_ambiguous_component_area_ratio": 0.8,
        "maximum_fused_ambiguous_component_count": 0,
        "unknown_consensus_fraction": 0.5,
        "required_prompt_status": "pass",
    }


def _candidate(index: int, seconds: float, mask: np.ndarray, **overrides) -> dict:
    probability = np.where(mask, 0.8, 0.0).astype(np.float32)
    value = {
        "candidate_index": index,
        "source_start_index": int(seconds),
        "source_end_index": int(seconds) + 40,
        "source_start_seconds": seconds,
        "source_end_seconds": seconds + 2.0,
        "probability": probability,
        "water_mask": mask.copy(),
        "unknown_mask": np.zeros_like(mask, dtype=bool),
        "gate_status": "pass",
        "gate_reasons": [],
        "observable_region_result_valid": True,
        "water_mask_time_stability": 0.8,
    }
    value.update(overrides)
    return value


def test_recurring_core_passes_and_transient_component_does_not_join() -> None:
    core = np.zeros((40, 60), dtype=bool)
    core[12:24, 20:36] = True
    transient = np.zeros_like(core)
    transient[2:6, 2:6] = True
    candidates = [
        _candidate(0, 0.0, core),
        _candidate(1, 10.0, core),
        _candidate(2, 20.0, core),
        _candidate(3, 30.0, transient),
    ]
    summary, artifacts = fuse_prediction_candidates(candidates, _policy())
    assert summary["fusion_status"] == "pass"
    assert summary["selected_cluster_window_count"] == 3
    assert summary["selected_cluster_time_span_seconds"] == 20.0
    assert np.array_equal(artifacts["fused_water_mask"], core)
    assert not artifacts["fused_water_mask"][3, 3]


def test_unknown_heavy_partial_windows_require_recurring_cross_window_support() -> None:
    core = np.zeros((40, 60), dtype=bool)
    core[12:24, 20:36] = True
    policy = _policy()
    policy["allowed_partial_gate_reasons"] = [
        "insufficient_high_confidence_water_tracks",
        "unknown_fraction_high",
    ]
    candidates = [
        _candidate(
            index,
            seconds,
            core,
            gate_status="partial",
            gate_reasons=[
                "insufficient_high_confidence_water_tracks",
                "unknown_fraction_high",
            ],
            observable_region_result_valid=True,
        )
        for index, seconds in enumerate((0.0, 20.0, 40.0))
    ]

    summary, artifacts = fuse_prediction_candidates(candidates, policy)

    assert summary["fusion_status"] == "pass"
    assert summary["eligible_window_count"] == 3
    assert summary["selected_cluster_time_span_seconds"] == 40.0
    assert np.array_equal(artifacts["fused_water_mask"], core)
    assert all(item["gate_status"] == "partial" for item in summary["candidates"])


def test_fewer_than_three_consistent_windows_fails_closed() -> None:
    first = np.zeros((30, 50), dtype=bool)
    first[5:12, 5:15] = True
    second = np.zeros_like(first)
    second[18:25, 32:42] = True
    candidates = [
        _candidate(0, 0.0, first),
        _candidate(1, 20.0, first),
        _candidate(2, 40.0, second),
    ]
    summary, _ = fuse_prediction_candidates(candidates, _policy())
    assert summary["fusion_status"] == "reject"
    assert "insufficient_cross_window_support" in summary["fusion_reasons"]


def test_two_equally_supported_spatial_clusters_are_ambiguous() -> None:
    left = np.zeros((40, 80), dtype=bool)
    right = np.zeros_like(left)
    left[10:20, 8:20] = True
    right[10:20, 58:70] = True
    candidates = [
        _candidate(0, 0.0, left),
        _candidate(1, 20.0, left),
        _candidate(2, 40.0, left),
        _candidate(3, 0.0, right),
        _candidate(4, 20.0, right),
        _candidate(5, 40.0, right),
    ]
    summary, _ = fuse_prediction_candidates(candidates, _policy())
    assert summary["fusion_status"] == "reject"
    assert "ambiguous_cross_window_components" in summary["fusion_reasons"]


def test_final_core_recomputes_actual_support_and_time_span() -> None:
    masks: list[np.ndarray] = []
    for left in (10, 18, 26, 34):
        mask = np.zeros((30, 60), dtype=bool)
        mask[10:20, left : left + 10] = True
        masks.append(mask)
    policy = _policy()
    policy["minimum_support_window_count"] = 2
    candidates = [
        _candidate(index, float(index * 10), mask)
        for index, mask in enumerate(masks)
    ]

    summary, _ = fuse_prediction_candidates(candidates, policy)

    assert summary["preliminary_cluster_window_count"] == 4
    assert summary["preliminary_cluster_time_span_seconds"] == 30.0
    assert summary["selected_cluster_window_count"] == 2
    assert summary["selected_cluster_time_span_seconds"] == 10.0
    assert summary["fusion_status"] == "reject"
    assert "insufficient_cross_window_support" in summary["fusion_reasons"]
    assert "cross_window_time_span_too_short" in summary["fusion_reasons"]


def test_tolerant_support_never_invents_final_water_pixels() -> None:
    masks: list[np.ndarray] = []
    for left in (10, 14, 18):
        mask = np.zeros((32, 48), dtype=bool)
        mask[10:20, left : left + 2] = True
        masks.append(mask)
    policy = _policy()
    policy["minimum_window_component_area_pixels"] = 2
    policy["minimum_fused_component_area_pixels"] = 2
    policy["minimum_component_iou"] = 0.0
    policy["support_tolerance_radius_pixels"] = 4
    policy["morphology_close_kernel"] = 3
    candidates = [
        _candidate(index, float(index * 10), mask)
        for index, mask in enumerate(masks)
    ]

    _, artifacts = fuse_prediction_candidates(candidates, policy)

    raw_union = np.any(np.stack(masks), axis=0)
    assert not np.any(np.asarray(artifacts["fused_water_mask"]) & ~raw_union)
    assert np.all(np.asarray(artifacts["raw_support_count"])[~raw_union] == 0)


def test_approved_partial_window_can_support_final_core() -> None:
    mask = np.zeros((30, 50), dtype=bool)
    mask[8:20, 18:32] = True
    candidates = [
        _candidate(0, 0.0, mask),
        _candidate(
            1,
            20.0,
            mask,
            gate_status="partial",
            gate_reasons=["insufficient_high_confidence_water_tracks"],
        ),
        _candidate(2, 40.0, mask),
    ]

    summary, _ = fuse_prediction_candidates(candidates, _policy())

    assert summary["eligible_window_count"] == 3
    assert summary["selected_cluster_window_count"] == 3
    assert summary["fusion_status"] == "pass"


def test_reject_and_unapproved_partial_windows_are_excluded() -> None:
    mask = np.zeros((30, 50), dtype=bool)
    mask[8:20, 18:32] = True
    candidates = [
        _candidate(0, 0.0, mask),
        _candidate(1, 20.0, mask, gate_status="reject"),
        _candidate(
            2,
            40.0,
            mask,
            gate_status="partial",
            gate_reasons=["water_mask_time_stability_low"],
        ),
    ]
    summary, _ = fuse_prediction_candidates(candidates, _policy())
    assert summary["eligible_window_count"] == 1
    assert summary["fusion_status"] == "reject"


def test_fusion_is_deterministic_and_does_not_modify_inputs() -> None:
    mask = np.zeros((30, 50), dtype=bool)
    mask[8:20, 18:32] = True
    candidates = [_candidate(index, float(index * 10), mask) for index in range(4)]
    before = deepcopy(candidates)
    first_summary, first_artifacts = fuse_prediction_candidates(candidates, _policy())
    second_summary, second_artifacts = fuse_prediction_candidates(candidates, _policy())
    assert first_summary == second_summary
    assert np.array_equal(first_artifacts["fused_water_mask"], second_artifacts["fused_water_mask"])
    for original, unchanged in zip(candidates, before):
        assert np.array_equal(original["water_mask"], unchanged["water_mask"])
        assert np.array_equal(original["probability"], unchanged["probability"])


def test_config_rejects_ground_truth_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "temporal_multi_window_evidence_fusion:\n"
        "  schema_version: v1\n"
        "  algorithm_version: v1\n"
        "  minimum_cluster_window_count: 3\n"
        "  minimum_cluster_time_span_seconds: 15\n"
        "  minimum_component_iou: 0.1\n"
        "  minimum_support_fraction: 0.5\n"
        "  probability_weight: 0.5\n"
        "  support_weight: 0.5\n"
        "  morphology_close_kernel: 1\n"
        "  required_prompt_status: pass\n"
        "  ground_truth_path: forbidden\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_multi_window_fusion_policy(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("probability_weight", -0.1, "within"),
        ("minimum_fused_confidence", 1.1, "between zero and one"),
        ("minimum_support_window_count", 0, "must be positive"),
        ("minimum_final_core_overlap_fraction", 0.0, "must be in"),
    ],
)
def test_config_rejects_unsafe_ranges(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    policy = _policy()
    policy[field] = value
    if field == "probability_weight":
        policy["support_weight"] = 1.1
    config = tmp_path / "bad_range.yaml"
    config.write_text(
        yaml.safe_dump({"temporal_multi_window_evidence_fusion": policy}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_multi_window_fusion_policy(config)


def test_exported_prompt_is_compatible_and_hash_matches(tmp_path: Path) -> None:
    frames = tmp_path / "source" / "frames"
    frames.mkdir(parents=True)
    for index in range(61):
        Image.new("RGB", (64, 48), color=(index, index, index)).save(
            frames / f"frame_{index:06d}.png"
        )
    mask = np.zeros((48, 64), dtype=bool)
    mask[15:33, 22:44] = True
    probability = np.where(mask, 0.9, 0.0).astype(np.float32)
    call_count = 0

    def fake_runner(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "prediction": {
                "evidence": {
                    "predicted_water_probability": probability.copy(),
                    "predicted_water_mask": mask.copy(),
                    "predicted_unknown_mask": np.zeros_like(mask),
                },
                "classifications": [],
                "water_mask_time_stability": 0.9,
            },
            "temporal_quality_gate": {
                "status": "pass",
                "reasons": [],
                "observable_region_result_valid": True,
            },
        }

    window_policy = {
        "burst_frame_count": 21,
        "anchor_frame_index": 10,
        "candidate_stride_seconds": 20.0,
        "max_candidate_count": 3,
    }
    prompt_config = {
        "schema_version": "prompt_v1",
        "algorithm_version": "prompt_v1",
        "connectivity": 8,
        "allow_temporally_corroborated_partial_gate": False,
        "corroboratable_partial_gate_reasons": [],
        "box_margin_px": 3,
        "partial_gate_box_margin_ratio": 0.5,
        "partial_gate_box_margin_cap_px": 10,
        "min_component_area_px": 4,
        "ambiguous_component_probability_mass_ratio": 0.8,
        "max_ambiguous_component_count": 0,
        "target_positive_points": 3,
        "min_positive_points": 3,
        "min_positive_boundary_distance_px": 2.0,
        "min_positive_spacing_px": 4.0,
        "min_positive_probability": 0.5,
        "min_positive_temporal_support_fraction": 0.66,
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
    policy = _policy()
    policy["minimum_cluster_time_span_seconds"] = 30.0
    output = tmp_path / "output"
    result, _ = run_temporal_multi_window_fusion(
        frames,
        1.0,
        window_policy,
        policy,
        {},
        {},
        prompt_config,
        output,
        runner=fake_runner,
    )
    prompt = result["selected_prompt"]
    assert call_count == 3
    assert result["sam2_run_allowed"] is True
    assert prompt["prompt_quality_status"] == "pass"
    assert prompt["ground_truth_used"] is False
    assert len(prompt["positive_points_xy"]) == 3
    assert len(prompt["negative_points_xy"]) >= 6
    assert Path(prompt["image_path"]).is_file()
    assert sha256_file(prompt["image_path"]) == prompt["image_sha256"]


def test_missing_source_frame_fails_before_prediction(tmp_path: Path) -> None:
    frames = tmp_path / "source" / "frames"
    frames.mkdir(parents=True)
    for index in (0, 1, 3, 4):
        Image.new("RGB", (16, 12), color=(index, index, index)).save(
            frames / f"frame_{index:06d}.png"
        )

    def unexpected_runner(*args, **kwargs):
        raise AssertionError("prediction runner must not run for discontinuous input")

    with pytest.raises(ValueError, match="contiguous"):
        run_temporal_multi_window_fusion(
            frames,
            1.0,
            {
                "burst_frame_count": 3,
                "anchor_frame_index": 1,
                "candidate_stride_seconds": 1.0,
                "max_candidate_count": 4,
            },
            _policy(),
            {},
            {},
            {},
            tmp_path / "output",
            runner=unexpected_runner,
        )


def test_prompt_reject_blocks_sam2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "source" / "frames"
    frames.mkdir(parents=True)
    for index in range(41):
        Image.new("RGB", (32, 24), color=(index, index, index)).save(
            frames / f"frame_{index:06d}.png"
        )
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 10:22] = True
    probability = np.where(mask, 0.9, 0.0).astype(np.float32)

    def fake_runner(*args, **kwargs):
        return {
            "prediction": {
                "evidence": {
                    "predicted_water_probability": probability.copy(),
                    "predicted_water_mask": mask.copy(),
                    "predicted_unknown_mask": np.zeros_like(mask),
                },
                "water_mask_time_stability": 0.9,
            },
            "temporal_quality_gate": {
                "status": "pass",
                "reasons": [],
                "observable_region_result_valid": True,
            },
        }

    def fake_prompt(*args, **kwargs):
        return (
            {
                "prompt_quality_status": "reject",
                "positive_points_xy": [],
                "negative_points_xy": [],
            },
            {"prompt_quality_status": "reject"},
        )

    monkeypatch.setattr(fusion_module, "generate_temporal_sam2_prompt", fake_prompt)
    policy = _policy()
    result, _ = run_temporal_multi_window_fusion(
        frames,
        1.0,
        {
            "burst_frame_count": 11,
            "anchor_frame_index": 5,
            "candidate_stride_seconds": 15.0,
            "max_candidate_count": 3,
        },
        policy,
        {},
        {},
        {},
        tmp_path / "output",
        runner=fake_runner,
    )
    assert result["fusion"]["fusion_status"] == "pass"
    assert result["selected_prompt"]["prompt_quality_status"] == "reject"
    assert result["sam2_run_allowed"] is False
