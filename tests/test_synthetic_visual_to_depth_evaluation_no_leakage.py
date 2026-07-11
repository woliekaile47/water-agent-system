import inspect
from pathlib import Path

import pytest

import src.evaluation.evaluate_synthetic_visual_to_depth as evaluation
import src.integration.synthetic_visual_to_depth as prediction


def test_prediction_core_still_has_no_gt_loader_imports():
    source = inspect.getsource(prediction)
    for forbidden in ("load_temporal_evaluation_ground_truth", "load_ground_truth_evaluation_inputs"):
        assert forbidden not in source


def test_incomplete_prediction_blocks_gt_read(monkeypatch, tmp_path: Path):
    sequence = tmp_path / "data/simulation_dynamic/sim_water_10cm_001/moderate/seed_42"
    (sequence / "ground_truth").mkdir(parents=True)
    output = tmp_path / "incomplete_prediction"
    output.mkdir()
    called = {"gt": False}

    def forbidden_loader(*args):
        called["gt"] = True
        raise AssertionError("GT loader must not run before prediction completion")

    monkeypatch.setattr(evaluation, "load_ground_truth_evaluation_inputs", forbidden_loader)
    with pytest.raises(FileNotFoundError, match="Prediction is incomplete"):
        evaluation.evaluate_synthetic_visual_to_depth_case(sequence, output, tmp_path)
    assert called["gt"] is False
