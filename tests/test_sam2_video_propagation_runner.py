"""Tests for the Phase 2D-C-7 standalone SAM2 video runner."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sam2_video_propagation.py"
SPEC = importlib.util.spec_from_file_location("sam2_video_runner", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_frame_plan_maps_original_indices_to_contiguous_local_indices() -> None:
    plan = RUNNER.frame_plan(79, 119, 99)
    assert len(plan) == 41
    assert plan[0] == {"original_frame_index": 79, "local_frame_index": 0}
    assert plan[20] == {"original_frame_index": 99, "local_frame_index": 20}
    assert plan[-1] == {"original_frame_index": 119, "local_frame_index": 40}
    with pytest.raises(ValueError):
        RUNNER.frame_plan(79, 119, 120)


def test_consecutive_iou_is_exact_and_handles_two_empty_masks() -> None:
    first = np.asarray([[1, 1], [0, 0]], dtype=bool)
    second = np.asarray([[1, 0], [1, 0]], dtype=bool)
    assert RUNNER.consecutive_iou(first, second) == 1 / 3
    empty = np.zeros((2, 2), dtype=bool)
    assert RUNNER.consecutive_iou(empty, empty) == 1.0


def test_video_runner_is_prediction_only_and_has_no_gt_or_evaluation_dependency() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "src.evaluation" not in source
    for token in (
        "camera_water_mask_gt",
        "dem_water_mask_gt",
        "depth_map_gt",
        "water_level_m",
        "nominal_depth_cm",
    ):
        assert token not in source
    assert '"ground_truth_used": False' in source
    assert '"authoritative": False' in source
    assert '"eligible_for_downstream": False' in source


def test_video_pilot_is_small_fixed_and_contains_no_answer_paths() -> None:
    path = ROOT / "configs/phase2d_c7_video_pilot.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)["phase2d_c7_video_pilot"]
    assert config["selection_frozen_before_video_propagation"] is True
    assert config["ground_truth_opened_for_selection"] is False
    assert config["window_start"] == 79
    assert config["window_end"] == 119
    assert config["anchor_frame_index"] == 99
    assert len(config["samples"]) == 3
    assert {sample["role"] for sample in config["samples"]} == {
        "shallow_heavy_failure_case",
        "stable_reference_case",
        "partial_coverage_case",
    }
    assert "ground_truth/" not in text


def test_sam2_import_is_deferred_until_runtime() -> None:
    source = inspect.getsource(RUNNER)
    prefix = source[: source.index("def main")]
    assert "from sam2" not in prefix
