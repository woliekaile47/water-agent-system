"""Regression tests for the seed-303 candidate-gate confirmation orchestration."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_confirmation_summary_detects_visible_pass_outside_3cm() -> None:
    module = load_script("evaluate_phase2d_c8_seed303_gate_confirmation.py")
    base = {
        "sample_id": "sample",
        "frame_index": 1,
        "evaluation_available": True,
        "camera_visible_status": "pass",
        "global_scene_status": "complete",
        "water_level_absolute_error_cm": 2.0,
        "water_level_within_3cm": True,
        "camera_mask_iou": 0.9,
        "visible_area_relative_error": 0.1,
        "visible_volume_relative_error": 0.1,
    }
    outside = dict(base, frame_index=2, water_level_absolute_error_cm=3.1, water_level_within_3cm=False)
    summary = module.summarize([base, outside])
    assert summary["visible_pass_count"] == 2
    assert summary["visible_pass_within_3cm_count"] == 1
    assert summary["visible_pass_outside_3cm_count"] == 1


def test_candidate_application_has_no_ground_truth_loader() -> None:
    source = (ROOT / "scripts" / "apply_phase2d_c8_seed303_candidate_gate.py").read_text(encoding="utf-8")
    assert "load_ground_truth" not in source
    assert "evaluate_candidate_gate" in source
    assert '"ground_truth_used": False' in source


def test_generic_video_scripts_keep_c7_default_config_key() -> None:
    for name in (
        "run_sam2_video_geometry_stability.py",
        "run_sam2_video_geometry_gt_evaluation.py",
        "run_sam2_video_propagation_gt_evaluation.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'default="phase2d_c7_video_pilot"' in source
        assert "args.config_key" in source
