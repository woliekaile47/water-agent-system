import numpy as np

from src.perception.temporal_water_evidence import build_temporal_water_evidence


def test_water_evidence_is_sparse_and_unknown_is_not_dry():
    classification = {
        "classification": "water_ripple", "center_mean": [20.0, 20.0],
        "maximum_area": 100, "confidence": 0.9, "duration_frames": 15,
    }
    evidence, diagnostics = build_temporal_water_evidence([classification], (48, 48), {
        "minimum_kernel_sigma_px": 3, "maximum_propagation_radius_px": 12,
        "probability_scale": 0.5, "water_probability_threshold": 0.25,
        "unknown_evidence_threshold": 0.08, "morphology_close_kernel": 3,
    })
    assert evidence["predicted_water_mask"][20, 20]
    assert evidence["predicted_unknown_mask"][0, 0]
    assert not evidence["predicted_water_mask"][0, 0]
    assert diagnostics["unknown_region_semantics"] == "no_temporal_evidence_not_confirmed_dry"
    assert np.isfinite(evidence["predicted_water_probability"]).all()
