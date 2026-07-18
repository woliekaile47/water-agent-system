"""Freeze and validate the Phase 2D-C-6C-2 seed302/frame99 matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/phase2d_c6c2_heldout_matrix.yaml"


def _matrix() -> dict:
    return yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))["phase2d_c6b2_heldout_matrix"]


def test_c6c2_matrix_is_complete_unique_and_deterministically_selected() -> None:
    matrix = _matrix()
    samples = matrix["samples"]
    assert len(samples) == 12
    assert matrix["selection_frozen_before_rgb_review"] is True
    assert matrix["rgb_decoded_or_subjectively_reviewed_for_selection"] is False
    assert matrix["ground_truth_opened_for_selection"] is False
    assert matrix["prompt_config"] == "configs/temporal_sam2_prompt_c6c.yaml"
    combinations = {(item["case_id"], item["rain_level"]) for item in samples}
    assert combinations == {
        (f"sim_water_{depth}cm_001", rain)
        for depth in (5, 10, 20, 40)
        for rain in ("light", "moderate", "heavy")
    }
    assert {item["seed"] for item in samples} == {302}
    assert {item["frame_index"] for item in samples} == {99}
    assert len({item["sample_id"] for item in samples}) == 12


def test_c6c2_rgb_hashes_match_without_reading_ground_truth() -> None:
    for sample in _matrix()["samples"]:
        image = ROOT / sample["image"]
        assert image.is_file()
        assert image.name == "frame_000099.png"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == sample["image_sha256"]
        assert "ground_truth" not in sample["image"]


def test_c6c2_matrix_contains_no_gt_or_answer_fields() -> None:
    text = MATRIX_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "water_level_m",
        "depth_map",
        "camera_water_mask_gt",
        "dem_water_mask_gt",
        "water_area_m2",
        "water_volume_m3",
        "nominal_depth_cm",
    ):
        assert forbidden not in text
