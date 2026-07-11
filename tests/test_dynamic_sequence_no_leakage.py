import json

from src.perception.synthetic_rain_visual_generator import generate_sequence
from tests.dynamic_rain_test_utils import base_and_mask, project_root, small_config, source_metadata


def test_output_layout_separates_detector_frames_and_ground_truth(tmp_path):
    base, mask = base_and_mask()
    output = tmp_path / "sequence"
    result = generate_sequence(base, mask, small_config(), "case", "light", 42, output, source_metadata(), project_root(), False)
    frame_entries = list((output / "frames").iterdir())
    assert frame_entries
    assert all(path.is_file() and path.suffix == ".png" and not path.is_symlink() for path in frame_entries)
    assert not any("ground_truth" in path.name for path in frame_entries)
    manifest = json.loads((output / "metadata/sequence_manifest.json").read_text())
    assert manifest["ground_truth_used_for_generation_only"] is True
    assert manifest["detector_input_should_only_use_frames"] is True
    assert manifest["synthetic_physics_validity"] == "visual_abstraction_not_fluid_simulation"
    assert result["quality"]["checks"]["ground_truth_files_complete"] is True
    assert result["quality"]["checks"]["detector_frames_directory_has_no_ground_truth_links"] is True


def test_dry_sequence_contains_no_water_ripple(tmp_path):
    base, _ = base_and_mask()
    output = tmp_path / "dry"
    result = generate_sequence(base, base[..., 0] < 0, small_config(), "dry", "moderate", 42, output, source_metadata(), project_root(), False)
    assert result["manifest"]["water_event_count"] == 0
    annotations = json.loads((output / "ground_truth/event_annotations.json").read_text())
    assert all(event["event_type"] == "dry_splash" for event in annotations["events"])
