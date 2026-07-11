import numpy as np
import pytest

from src.perception.synthetic_rain_visual_generator import generate_sequence
from tests.dynamic_rain_test_utils import base_and_mask, project_root, small_config, source_metadata


def test_mask_size_mismatch_is_rejected(tmp_path):
    base, _ = base_and_mask()
    with pytest.raises(ValueError, match="does not match"):
        generate_sequence(base, np.zeros((10, 10), dtype=bool), small_config(), "case", "moderate", 42, tmp_path / "bad", source_metadata(), project_root(), False)


def test_edge_events_are_safely_clipped(tmp_path):
    base, _ = base_and_mask()
    mask = np.ones(base.shape[:2], dtype=bool)
    config = small_config()
    config["event_model"]["center_margin_px"] = 0
    result = generate_sequence(base, mask, config, "all_water", "heavy", 7, tmp_path / "edge", source_metadata(), project_root(), False)
    assert result["quality"]["status"] == "pass"
    event_map = np.load(tmp_path / "edge/ground_truth/event_map_sequence.npy")
    assert event_map.shape == (8, 32, 48)
    assert np.isin(event_map, [0, 2]).all()
