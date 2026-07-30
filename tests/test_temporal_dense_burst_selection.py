"""Tests for prediction-side dense-burst selection."""

from pathlib import Path

import pytest

from src.vision.temporal_dense_burst_selection import (
    choose_candidate,
    enumerate_candidate_starts,
    load_dense_burst_policy,
)


def _candidate(index: int, status: str, border: bool, start: int) -> dict:
    return {
        "candidate_index": index,
        "source_start_index": start,
        "source_end_index": start + 40,
        "prompt_quality_status": status,
        "selection_rank": [
            {"reject": 0, "diagnostic_only": 1, "pass": 2}[status],
            2,
            int(not border),
            1,
            1,
            1.0,
            0.5,
            0.5,
            0.5,
            -float(border),
            -start,
        ],
    }


def test_candidate_starts_are_deterministic_and_include_final_window() -> None:
    assert enumerate_candidate_starts(1201, 41, 100) == [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1160
    ]


def test_pass_candidate_beats_diagnostic_candidate() -> None:
    result = choose_candidate([
        _candidate(0, "diagnostic_only", False, 0),
        _candidate(1, "pass", True, 100),
    ])
    assert result["selection_status"] == "pass"
    assert result["selected_candidate_index"] == 1


def test_border_safe_candidate_wins_with_same_status() -> None:
    result = choose_candidate([
        _candidate(0, "pass", True, 0),
        _candidate(1, "pass", False, 100),
    ])
    assert result["selected_candidate_index"] == 1


def test_no_pass_candidate_fails_closed_without_manual_fallback() -> None:
    result = choose_candidate([_candidate(0, "diagnostic_only", False, 0)])
    assert result["selection_status"] == "reject"
    assert result["manual_selection_used"] is False
    assert result["ground_truth_used"] is False


def test_config_rejects_ground_truth_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "temporal_dense_burst_selection:\n"
        "  schema_version: v1\n"
        "  algorithm_version: v1\n"
        "  burst_frame_count: 41\n"
        "  anchor_frame_index: 20\n"
        "  candidate_stride_seconds: 5\n"
        "  required_prompt_status: pass\n"
        "  ground_truth_path: forbidden\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_dense_burst_policy(config)
