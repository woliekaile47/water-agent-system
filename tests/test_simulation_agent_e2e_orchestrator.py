from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from src.integration.simulation_agent_e2e_orchestrator import (
    RUNTIME_CONFIG_KEY,
    assert_prepared_run_matches_config,
    build_runtime_config,
    resolve_run_dir,
    safe_extract_tar,
    assert_expected_outcome,
    summarize_gate_blocked_run,
    validate_config,
    validate_run_id,
)


def valid_config() -> dict:
    return {
        "protocol_version": "test",
        "data_domain": "simulation",
        "runtime_config_key": RUNTIME_CONFIG_KEY,
        "run_root_parent": "outputs/phase2d_c13_one_click_runs",
        "ground_truth_used_for_prediction": False,
        "expected_outcome": {
            "camera_visible_status": "pass",
            "global_scene_status": "complete",
            "agent_should_run": True,
        },
        "sample": {
            "sample_id": "sample",
            "case_id": "case",
            "rain_level": "moderate",
            "seed": 303,
            "role": "test",
            "frames_dir": "frames",
            "fps": 20,
            "window_start": 10,
            "window_end": 12,
            "anchor_frame_index": 11,
        },
        "prediction_configs": {
            "candidate_quality_gate": "configs/gate.yaml",
        },
        "standard_pipeline": {
            "case_library_json": "data/cases.json",
        },
        "safety": {
            "simulation_only": True,
            "authoritative": False,
            "eligible_for_real_warning": False,
            "external_notification_allowed": False,
            "real_device_action_allowed": False,
            "real_api_calls_enabled": False,
        },
    }


def test_config_requires_all_simulation_safety_flags() -> None:
    config = valid_config()
    validate_config(config)
    config["safety"]["external_notification_allowed"] = True
    with pytest.raises(ValueError, match="external_notification_allowed"):
        validate_config(config)


def test_config_prevents_agent_for_partial_global_scene() -> None:
    config = valid_config()
    config["expected_outcome"]["global_scene_status"] = "partial"
    with pytest.raises(ValueError, match="Agent may run only"):
        validate_config(config)


def test_expected_outcome_mismatch_is_rejected() -> None:
    config = valid_config()
    with pytest.raises(RuntimeError, match="outcome mismatch"):
        assert_expected_outcome(
            config,
            {"camera_visible_status": "reject", "global_scene_status": "unavailable"},
        )


def test_expected_quality_reject_is_successful_safe_acceptance(tmp_path: Path) -> None:
    config = valid_config()
    config["expected_outcome"] = {
        "camera_visible_status": "reject",
        "global_scene_status": "unavailable",
        "agent_should_run": False,
    }
    decision = {
        "camera_visible_status": "reject",
        "global_scene_status": "unavailable",
        "visible_reject_reasons": ["camera_reprojection_iou_below_candidate_threshold"],
        "global_scope_reasons": [],
        "warnings": [],
        "ground_truth_used": False,
    }
    summary = summarize_gate_blocked_run(
        tmp_path,
        config,
        {
            "estimated_water_level_m": -0.3,
            "water_area_m2": 1.0,
            "water_volume_m3": 0.1,
        },
        decision,
    )
    assert summary["status"] == "success"
    assert summary["agent_status"] == "blocked_by_quality_gate"
    assert summary["standard_pipeline_completed"] is False
    assert summary["eligible_for_downstream"] is False


def test_resume_requires_same_frozen_config_and_sample(tmp_path: Path) -> None:
    import hashlib

    config = valid_config()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("frozen-config\n", encoding="utf-8")
    summary = {
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "sample_id": "sample",
    }
    assert_prepared_run_matches_config(summary, config_file, config)
    summary["sample_id"] = "different"
    with pytest.raises(ValueError, match="sample"):
        assert_prepared_run_matches_config(summary, config_file, config)


def test_resume_rejects_changed_config_file(tmp_path: Path) -> None:
    import hashlib

    config = valid_config()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("before\n", encoding="utf-8")
    summary = {
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "sample_id": "sample",
    }
    config_file.write_text("after\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration hash"):
        assert_prepared_run_matches_config(summary, config_file, config)


@pytest.mark.parametrize("run_id", ["run_001", "simulation-20260726", "a"])
def test_run_id_validation_accepts_safe_values(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["../escape", "contains space", "", "a/b"])
def test_run_id_validation_rejects_path_like_values(run_id: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_run_directory_is_confined_below_outputs(tmp_path: Path) -> None:
    config = valid_config()
    run_dir = resolve_run_dir(tmp_path, config, "run_001")
    assert run_dir == tmp_path / "outputs" / "phase2d_c13_one_click_runs" / "run_001"


def test_runtime_config_uses_standard_c11_contract_and_no_gt(tmp_path: Path) -> None:
    config = valid_config()
    run_dir = tmp_path / "outputs" / "phase2d_c13_one_click_runs" / "run_001"
    document = build_runtime_config(tmp_path, run_dir, config)
    runtime = document[RUNTIME_CONFIG_KEY]
    assert runtime["ground_truth_used_for_prediction"] is False
    assert runtime["runtime_inputs"]["geometry_case_dir"].endswith(
        "s4_geometry/sample"
    )
    assert runtime["samples"][0]["prompt_path"].endswith(
        "s3_prompt/sample/automatic_prompt.json"
    )
    serialized = str(document).lower()
    assert "ground_truth/" not in serialized
    assert "nominal_depth" not in serialized


def test_runtime_config_keeps_ground_dem_directory_contract(tmp_path: Path) -> None:
    config = valid_config()
    run_dir = tmp_path / "outputs" / "phase2d_c13_one_click_runs" / "run_002"
    runtime = build_runtime_config(tmp_path, run_dir, config)[RUNTIME_CONFIG_KEY]
    assert runtime["runtime_inputs"]["ground_dem_dir"].endswith("s2/data/dem")


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../escape.txt")
        payload = b"unsafe"
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_tar(archive, tmp_path / "target")


def test_safe_extract_accepts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("result/summary.json")
        payload = b"{}"
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    target = tmp_path / "target"
    safe_extract_tar(archive, target)
    assert (target / "result" / "summary.json").read_bytes() == b"{}"
