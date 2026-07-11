import pytest

from src.integration.integration_quality_gate import evaluate_integration_quality_gate


def visual(status):
    return {"status": status, "reasons": []}


def geometry(status="pass", global_status="complete", observable=True, area="complete_estimate"):
    return {"status": status, "reasons": [], "global_estimate_status": global_status,
            "observable_region_result_valid": observable,
            "result_semantics": "global_estimate" if global_status == "complete" else "observable_region_estimate",
            "area_volume_semantics": area}


@pytest.mark.parametrize("visual_status,geometry_gate,expected", [
    ("reject", geometry(), "reject"),
    ("partial", geometry(), "partial"),
    ("pass", geometry(), "pass"),
])
def test_joint_status_rules(visual_status, geometry_gate, expected):
    result = evaluate_integration_quality_gate(visual(visual_status), geometry_gate)
    assert result["status"] == expected
    assert result["eligible_for_downstream"] is False
    if expected == "pass":
        assert result["measurement_status"] == "complete_estimate"
        assert result["candidate_values_available"] is True
        assert result["authoritative_measurement_available"] is True
        assert result["authoritative_area_volume_semantics"] == "complete_estimate"
    elif expected == "partial":
        assert result["measurement_status"] == "observable_region_estimate"
        assert result["candidate_values_available"] is True
        assert result["authoritative_measurement_available"] is True
        assert result["authoritative_area_volume_semantics"] == "observable_lower_bound"
    else:
        assert result["measurement_status"] == "rejected_candidate"
        assert result["candidate_values_available"] is True
        assert result["authoritative_measurement_available"] is False
        assert result["authoritative_area_volume_semantics"] == "unavailable"


def test_geometry_reject_preserves_partial_observable_semantics():
    result = evaluate_integration_quality_gate(
        visual("pass"), geometry("reject", "partial", True, "observable_lower_bound"),
    )
    assert result["status"] == "reject"
    assert result["global_estimate_status"] == "partial"
    assert result["observable_region_result_valid"] is True
    assert result["area_volume_semantics"] == "observable_lower_bound"
    assert result["measurement_status"] == "rejected_candidate"
    assert result["candidate_values_available"] is True
    assert result["authoritative_measurement_available"] is False
    assert result["authoritative_area_volume_semantics"] == "unavailable"


def test_geometry_unavailable_is_global_unavailable():
    result = evaluate_integration_quality_gate(
        visual("pass"), None, {"failure_type": "no_trusted_shoreline"},
    )
    assert result["status"] == "reject"
    assert result["global_estimate_status"] == "unavailable"
    assert result["measurement_status"] == "unavailable"
    assert result["candidate_values_available"] is False
    assert result["authoritative_measurement_available"] is False
    assert result["authoritative_area_volume_semantics"] == "unavailable"
    assert result["eligible_for_downstream"] is False
