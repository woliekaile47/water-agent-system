import json
import inspect
from pathlib import Path

import numpy as np
import yaml

from scripts.generate_temporal_sam2_prompt import (
    _load_temporal_support,
    _save_prediction_artifacts,
)
from src.vision.generate_temporal_sam2_prompt import sha256_file
from src.vision.temporal_sam2_prompt_pipeline import run_temporal_sam2_prompt_from_frames


ROOT = Path(__file__).resolve().parents[1]


def test_frames_to_prompt_interface_is_prediction_only():
    parameters = set(inspect.signature(run_temporal_sam2_prompt_from_frames).parameters)
    forbidden = {"gt", "ground_truth", "water_level", "depth", "area", "volume", "nominal_depth_cm"}
    assert not parameters & forbidden
    source = inspect.getsource(run_temporal_sam2_prompt_from_frames)
    assert "src.evaluation" not in source
    assert '"ground_truth_used": False' in source


def test_frozen_matrix_has_twelve_unique_seed301_frame149_samples():
    path = ROOT / "configs" / "phase2d_c6b2_heldout_matrix.yaml"
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))["phase2d_c6b2_heldout_matrix"]
    samples = matrix["samples"]
    assert len(samples) == 12
    assert len({item["sample_id"] for item in samples}) == 12
    assert {(item["case_id"], item["rain_level"]) for item in samples} == {
        (f"sim_water_{depth}cm_001", rain) for depth in (5, 10, 20, 40)
        for rain in ("light", "moderate", "heavy")
    }
    assert all(item["seed"] == 301 and item["frame_index"] == 149 for item in samples)
    assert all(len(item["image_sha256"]) == 64 for item in samples)
    assert matrix["ground_truth_opened_for_selection"] is False
    assert matrix["prediction_uses_nominal_depth"] is False


def test_matrix_runner_has_no_evaluation_or_gt_path_dependency():
    source = (ROOT / "scripts" / "run_temporal_sam2_prompt_matrix.py").read_text(encoding="utf-8")
    assert "src.evaluation" not in source
    assert "ground_truth/" not in source
    assert "--expected-image-sha256" in source
    assert '"sam2_started": False' in source
    assert "refusing to overwrite frozen matrix output" in source


def test_hash_mismatch_stops_before_temporal_prediction(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    image = frames / "frame_000149.png"
    image.write_bytes(b"frozen-rgb-placeholder")
    try:
        run_temporal_sam2_prompt_from_frames(frames, image, 149, {}, {}, {}, expected_image_sha256="0" * 64)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("mismatched frozen image hash was accepted")


def test_frames_runner_persists_temporal_support_with_manifest_hashes(tmp_path: Path):
    shape = (4, 5)
    support = np.full(shape, 2.0 / 3.0, dtype=np.float32)
    result = {
        "prediction": {
            "evidence": {
                "predicted_water_probability": np.full(shape, 0.75, dtype=np.float32),
                "evidence_count_map": np.ones(shape, dtype=np.uint16),
                "predicted_water_mask": np.ones(shape, dtype=bool),
                "predicted_unknown_mask": np.zeros(shape, dtype=bool),
            },
            "classifications": [],
            "loader": {},
            "preprocessing_diagnostics": {},
            "candidate_diagnostics": {},
            "evidence_diagnostics": {},
            "water_mask_time_stability": {},
            "feature_score_separation": {},
            "temporal_support_fraction": support,
            "temporal_support_diagnostics": {
                "window_count": 3,
                "ground_truth_used": False,
            },
        },
        "order_sensitivity": {},
        "temporal_quality_gate": {"status": "pass"},
    }

    _save_prediction_artifacts(tmp_path, result)

    support_path = tmp_path / "temporal_support_fraction.npy"
    diagnostics_path = tmp_path / "temporal_support_diagnostics.json"
    assert np.array_equal(np.load(support_path), support)
    assert json.loads(diagnostics_path.read_text(encoding="utf-8"))["ground_truth_used"] is False
    manifest = json.loads((tmp_path / "prediction_manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["prediction_artifact_sha256"]
    for path in (support_path, diagnostics_path):
        assert hashes[str(path.resolve())] == sha256_file(path)


def test_prediction_dir_support_reuse_is_complete_pair_only(tmp_path: Path):
    support_path = tmp_path / "temporal_support_fraction.npy"
    diagnostics_path = tmp_path / "temporal_support_diagnostics.json"
    np.save(support_path, np.ones((2, 3), dtype=np.float32))

    support, paths = _load_temporal_support(tmp_path)
    assert support is None
    assert paths == []

    diagnostics_path.write_text(
        '{"window_count": 3, "ground_truth_used": false}\n',
        encoding="utf-8",
    )
    support, paths = _load_temporal_support(tmp_path)
    assert np.array_equal(support, np.ones((2, 3), dtype=np.float32))
    assert paths == [support_path, diagnostics_path]


def test_prediction_dir_runner_passes_optional_support_to_fail_closed_generator():
    source = (ROOT / "scripts" / "generate_temporal_sam2_prompt.py").read_text(encoding="utf-8")
    assert "temporal_support, temporal_support_paths = _load_temporal_support(prediction_dir)" in source
    assert "temporal_support_fraction=temporal_support" in source
