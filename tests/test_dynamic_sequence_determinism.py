from pathlib import Path

from src.perception.synthetic_rain_visual_generator import generate_sequence
from tests.dynamic_rain_test_utils import base_and_mask, project_root, small_config, source_metadata


def frame_bytes(directory: Path):
    return [path.read_bytes() for path in sorted((directory / "frames").glob("frame_*.png"))]


def test_same_seed_is_frame_exact_and_different_seed_differs(tmp_path):
    base, mask = base_and_mask()
    config = small_config()
    first = tmp_path / "first"
    second = tmp_path / "second"
    other = tmp_path / "other"
    generate_sequence(base, mask, config, "case", "moderate", 42, first, source_metadata(), project_root(), False)
    generate_sequence(base, mask, config, "case", "moderate", 42, second, source_metadata(), project_root(), False)
    generate_sequence(base, mask, config, "case", "moderate", 43, other, source_metadata(), project_root(), False)
    assert frame_bytes(first) == frame_bytes(second)
    assert any(left != right for left, right in zip(frame_bytes(first), frame_bytes(other)))
