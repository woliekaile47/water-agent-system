import inspect
from pathlib import Path

import numpy as np

from src.vision.generate_temporal_sam2_prompt import generate_temporal_sam2_prompt


def test_prediction_interface_has_no_ground_truth_inputs_or_evaluation_imports():
    parameters = set(inspect.signature(generate_temporal_sam2_prompt).parameters)
    forbidden = {"gt", "ground_truth", "water_level", "depth", "area", "volume", "nominal_depth_cm"}
    assert not parameters & forbidden
    source = inspect.getsource(generate_temporal_sam2_prompt)
    assert "src.evaluation" not in source
    assert "ground_truth_used" in source


def test_cli_reads_only_named_prediction_artifacts():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_temporal_sam2_prompt.py"
    ).read_text(encoding="utf-8")
    assert "src.evaluation" not in script
    assert "evaluation_metrics" not in script
    assert "ground_truth/" not in script
    assert "predicted_water_probability.npy" in script
    assert "visual_quality_gate.json" in script


def test_corrupt_unrelated_gt_files_cannot_change_prompt(tmp_path: Path):
    probability = np.zeros((40, 50), dtype=np.float32)
    water = np.zeros_like(probability, dtype=bool)
    water[10:30, 15:36] = True
    probability[water] = 0.8
    unknown = np.zeros_like(water)
    config = {
        "schema_version": "phase2d_c6_prompt_v1", "algorithm_version": "test",
        "connectivity": 8, "box_margin_px": 3, "min_component_area_px": 16,
        "ambiguous_component_probability_mass_ratio": 0.8, "max_ambiguous_component_count": 0,
        "target_positive_points": 4, "min_positive_points": 3,
        "min_positive_boundary_distance_px": 2, "min_positive_spacing_px": 4,
        "target_negative_points": 8, "min_negative_points": 6,
        "min_negative_direction_sectors": 4,
        "dry_track_min_confidence": 0.5, "negative_ring_inner_distance_px": 2,
        "negative_ring_outer_distance_px": 8, "max_box_area_fraction": 0.8,
        "max_box_border_touch_ratio": 0.5,
    }
    kwargs = dict(image_path="frame.png", image_sha256="b" * 64, frame_index=49)
    first = generate_temporal_sam2_prompt(
        probability, water, unknown, [], {"status": "pass"}, config, **kwargs
    )
    gt = tmp_path / "ground_truth"
    gt.mkdir()
    (gt / "camera_water_mask.npy").write_bytes(b"corrupt")
    (gt / "water_level.json").write_text('{"water_level_m": 999}', encoding="utf-8")
    second = generate_temporal_sam2_prompt(
        probability, water, unknown, [], {"status": "pass"}, config, **kwargs
    )
    assert first == second
