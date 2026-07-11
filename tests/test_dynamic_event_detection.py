import numpy as np

from src.perception.dynamic_event_detection import detect_dynamic_event_candidates


def test_local_dynamic_component_is_detected():
    residual = np.zeros((3, 32, 32), dtype=np.float32)
    residual[1, 14:18, 13:17] = 30.0
    signed = residual.copy()
    candidates, diagnostics = detect_dynamic_event_candidates(residual, signed, {
        "residual_threshold": 5.0, "residual_percentile": 99.0,
        "morphology_close_kernel": 1, "morphology_dilate_iterations": 0,
        "min_area_px": 4, "max_area_px": 100, "max_aspect_ratio": 4.0,
    })
    assert len(candidates[1]) == 1
    assert candidates[1][0]["area"] == 16
    assert candidates[1][0]["center_u"] == 14.5
    assert candidates[1][0]["center_v"] == 15.5
    assert diagnostics["ground_truth_used"] is False
