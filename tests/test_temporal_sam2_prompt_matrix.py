import inspect
from pathlib import Path

import yaml

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
