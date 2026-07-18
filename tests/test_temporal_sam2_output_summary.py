import inspect
import json
from pathlib import Path

import numpy as np

from src.vision.summarize_temporal_sam2_outputs import summarize_frozen_sam2_outputs


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summary_validates_frozen_highest_score_candidate_without_gt(tmp_path: Path):
    prompt = {
        "sample_count": 1,
        "status_counts": {"pass": 1, "diagnostic_only": 0, "reject": 0},
        "samples": [{"sample_id": "c6b2_001", "case_id": "opaque_case", "rain_level": "light",
                     "seed": 301, "frame_index": 149, "prompt_status": "pass"}],
    }
    prompt_path = tmp_path / "prompt.json"
    _write_json(prompt_path, prompt)
    sample = tmp_path / "outputs" / "c6b2_001"
    sample.mkdir(parents=True)
    mask = np.zeros((360, 640), dtype=bool)
    mask[10:20, 15:25] = True
    np.save(sample / "prompted_mask_raw.npy", mask)
    _write_json(sample / "prompted_mask_summary.json", {
        "semantic_label": "unknown_candidate", "prompt_source": "temporal_water_evidence_v1",
        "authoritative": False, "returned_candidate_count": 3,
        "candidates": [{"candidate_id": 1, "score": 0.9}, {"candidate_id": 2, "score": 0.8},
                       {"candidate_id": 3, "score": 0.7}],
        "selected_candidate_id": 1, "selected_score": 0.9, "raw_mask_area_pixels": 100,
        "raw_mask_area_ratio": 100 / (360 * 640), "connected_component_count": 1,
        "enclosed_hole_count": 0, "inference_time_seconds": 1.0, "total_time_seconds": 2.0,
        "peak_gpu_allocated_mib": 500, "peak_gpu_reserved_mib": 600, "cuda_oom": False,
    })
    result = summarize_frozen_sam2_outputs(prompt_path, tmp_path / "outputs")
    assert result["sample_count"] == 1
    assert result["selected_highest_score_count"] == 1
    assert result["cuda_oom_count"] == 0
    assert result["ground_truth_used"] is False
    assert result["eligible_for_downstream"] is False


def test_summary_source_has_no_evaluation_or_gt_dependency():
    signature = set(inspect.signature(summarize_frozen_sam2_outputs).parameters)
    assert not signature & {"gt", "ground_truth", "water_level", "depth", "area", "volume"}
    source = inspect.getsource(summarize_frozen_sam2_outputs)
    assert "src.evaluation" not in source
    assert "ground_truth/" not in source
