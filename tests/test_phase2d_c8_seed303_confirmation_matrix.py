"""Tests for the frozen C8 seed-303 confirmation input matrix."""

import hashlib
import re
from itertools import product
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "phase2d_c8_seed303_confirmation_matrix.yaml"


def _matrix() -> dict:
    return yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))[
        "phase2d_c8_seed303_confirmation_matrix"
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matrix_is_complete_cartesian_product_with_fixed_window() -> None:
    matrix = _matrix()
    samples = matrix["samples"]
    assert len(samples) == matrix["sample_count"] == 12
    expected = set(product(
        (
            "sim_water_5cm_001",
            "sim_water_10cm_001",
            "sim_water_20cm_001",
            "sim_water_40cm_001",
        ),
        ("light", "moderate", "heavy"),
    ))
    assert {(sample["case_id"], sample["rain_level"]) for sample in samples} == expected
    assert matrix["seed"] == 303
    assert matrix["anchor_frame_index"] == 149
    assert (matrix["window_start"], matrix["window_end"], matrix["window_frame_count"]) == (
        129, 169, 41
    )


def test_selection_was_frozen_before_inspection_or_prediction() -> None:
    matrix = _matrix()
    assert matrix["selection_frozen_before_rgb_inspection"] is True
    assert matrix["rgb_opened_for_subjective_selection"] is False
    assert matrix["ground_truth_opened_for_prediction"] is False
    assert matrix["prediction_started"] is False
    assert matrix["authoritative"] is False
    assert matrix["eligible_for_downstream"] is False


def test_all_frozen_hashes_are_sha256_values() -> None:
    matrix = _matrix()
    pattern = re.compile(r"^[0-9a-f]{64}$")
    for sample in matrix["samples"]:
        for field in ("anchor_sha256", "window_sha256", "manifest_sha256", "sequence_sha256"):
            assert pattern.fullmatch(sample[field])


def test_matrix_contains_no_ground_truth_paths_or_metrics() -> None:
    text = MATRIX_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "camera_water_mask_gt", "dem_water_mask_gt", "depth_map_gt", "water_level_gt",
        "area_gt", "volume_gt", "evaluation_status", "evaluation_metrics",
    ):
        assert forbidden not in text


def test_referenced_source_configs_match_frozen_hashes() -> None:
    matrix = _matrix()
    for field in (
        "dynamic_generator_config", "automatic_prompt_config", "frozen_candidate_gate_config"
    ):
        record = matrix[field]
        assert _sha256(ROOT / record["path"]) == record["sha256"]
