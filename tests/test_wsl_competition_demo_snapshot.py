from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src.integration.wsl_competition_demo_snapshot import (
    WslCompetitionDemoError,
    build_wsl_competition_demo_snapshot,
    find_latest_completed_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_wsl_competition_demo.sh"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path, matrix_id: str = "matrix_001") -> None:
    labels = ("5cm", "10cm", "20cm", "40cm")
    outcomes = {
        "5cm": ("reject", "unavailable", False),
        "10cm": ("pass", "complete", True),
        "20cm": ("pass", "complete", True),
        "40cm": ("pass", "partial", False),
    }
    matrix_scenarios = []
    summary_scenarios = []
    for index, label in enumerate(labels):
        visible, global_status, agent_should_run = outcomes[label]
        sample_id = f"sample_{label}"
        run_id = f"{matrix_id}_{label}"
        config_path = f"configs/{label}.yaml"
        matrix_scenarios.append(
            {
                "label": label,
                "config": config_path,
                "expected": {
                    "camera_visible_status": visible,
                    "global_scene_status": global_status,
                    "agent_should_run": agent_should_run,
                },
            }
        )
        summary_scenarios.append(
            {
                "label": label,
                "run_id": run_id,
                "status": "success",
                "camera_visible_status": visible,
                "global_scene_status": global_status,
                "agent_status": "success" if agent_should_run else "blocked_by_quality_gate",
                "agent_should_run": agent_should_run,
                "quality_reject_reasons": ["expected_reject"] if label == "5cm" else [],
                "global_scope_reasons": ["outside_camera"] if label == "40cm" else [],
            }
        )
        one_click = {
            "phase2d_c13_one_click": {
                "data_domain": "simulation",
                "run_root_parent": "outputs/phase2d_c13_one_click_runs",
                "ground_truth_used_for_prediction": False,
                "expected_outcome": matrix_scenarios[-1]["expected"],
                "sample": {
                    "sample_id": sample_id,
                    "case_id": f"sim_water_{label}_001",
                    "rain_level": "moderate",
                    "seed": 303,
                    "frames_dir": f"data/simulation_dynamic/{label}/frames",
                    "anchor_frame_index": 149,
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
        }
        config_file = root / config_path
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(one_click), encoding="utf-8")
        input_image = root / f"data/simulation_dynamic/{label}/frames/frame_000149.png"
        input_image.parent.mkdir(parents=True, exist_ok=True)
        input_image.write_bytes(b"rgb")
        run = root / "outputs/phase2d_c13_one_click_runs" / run_id
        completion = {
            "status": "success",
            "standard_pipeline_completed": agent_should_run,
            "agent_status": "success" if agent_should_run else "blocked_by_quality_gate",
            "agent_should_run": agent_should_run,
            "acceptance_outcome": "completed" if agent_should_run else "safe_block",
            "expected_outcome_matched": True,
            "estimated_water_level_m": -0.3 + index * 0.1,
            "mean_depth_cm": float(index + 1),
            "max_depth_cm": float((index + 1) * 5),
            "water_area_m2": float(index + 1),
            "water_volume_m3": float(index + 1) / 10,
            "camera_reprojection_iou": 0.9,
            "outer_boundary_reprojection_p95_px": 3.0,
            "camera_visible_status": visible,
            "global_scene_status": global_status,
            "sqlite_database_exists": agent_should_run,
            "warning_mode": "simulation_record_only",
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_real_warning": False,
            "real_device_started": False,
            "real_api_calls_enabled": False,
            "safety_checks_passed": True,
        }
        _write_json(run / "completion_summary.json", completion)
        mask = run / "result/masks_png/frame_000149.png"
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(b"mask")
        geometry = run / "s4_geometry" / sample_id
        geometry.mkdir(parents=True, exist_ok=True)
        _write_json(
            geometry / "per_frame_geometry_summary.json",
            [
                {
                    "frame_index": 149,
                    "median_depth_cm": 1.0,
                    "candidate_basin_count": 1,
                    "unobserved_candidate_basin_count": int(label == "40cm"),
                    "result_semantics": "camera_visible_estimate",
                    "warnings": [],
                    "ground_truth_used": False,
                }
            ],
        )
        for filename in (
            "anchor_reprojected_camera_mask.png",
            "water_level_over_time.png",
            "max_depth_over_time.png",
            "area_over_time.png",
            "volume_over_time.png",
        ):
            (geometry / filename).write_bytes(filename.encode())

    matrix_config = {
        "phase2d_c15_multiscenario_acceptance": {
            "output_root": "outputs/phase2d_c15_multiscenario_acceptance",
            "scenarios": matrix_scenarios,
            "safety": {
                "simulation_only": True,
                "ground_truth_used": False,
                "authoritative": False,
                "eligible_for_real_warning": False,
            },
        }
    }
    config_path = root / "configs/phase2d_c15_multiscenario_acceptance.yaml"
    config_path.write_text(yaml.safe_dump(matrix_config), encoding="utf-8")
    _write_json(
        root
        / "outputs/phase2d_c15_multiscenario_acceptance"
        / matrix_id
        / "acceptance_summary.json",
        {
            "status": "pass",
            "matrix_id": matrix_id,
            "scenario_count": 4,
            "matched_scenario_count": 4,
            "scenarios": summary_scenarios,
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_real_warning": False,
        },
    )


def test_snapshot_uses_completed_wsl_runs_without_gt(tmp_path: Path) -> None:
    _fixture(tmp_path)
    snapshot = build_wsl_competition_demo_snapshot(tmp_path, "matrix_001")
    assert snapshot["demo_mode"] == "wsl_local_simulation_agent_e2e"
    assert snapshot["case_count"] == 4
    assert snapshot["ground_truth_used"] is False
    assert snapshot["authoritative"] is False
    assert snapshot["eligible_for_downstream"] is False
    assert {case["runtime_environment"] for case in snapshot["cases"]} == {"wsl_local"}
    assert snapshot["cases"][1]["pipeline"]["agent_status"] == "success"
    assert snapshot["cases"][3]["quality"]["global_scene_status"] == "partial"
    assert all(case["ground_truth_used"] is False for case in snapshot["cases"])
    for case in snapshot["cases"]:
        assert all(
            "ground_truth" not in str(path).lower()
            for path in case["assets"].values()
            if path is not None
        )


def test_snapshot_rejects_unsafe_completion(tmp_path: Path) -> None:
    _fixture(tmp_path)
    completion = (
        tmp_path
        / "outputs/phase2d_c13_one_click_runs/matrix_001_10cm/completion_summary.json"
    )
    value = json.loads(completion.read_text())
    value["real_api_calls_enabled"] = True
    _write_json(completion, value)
    with pytest.raises(WslCompetitionDemoError, match="not safe"):
        build_wsl_competition_demo_snapshot(tmp_path, "matrix_001")


def test_latest_matrix_requires_safe_passing_summary(tmp_path: Path) -> None:
    _fixture(tmp_path, "matrix_old")
    _fixture(tmp_path, "matrix_new")
    unsafe = (
        tmp_path
        / "outputs/phase2d_c15_multiscenario_acceptance/matrix_new/acceptance_summary.json"
    )
    value = json.loads(unsafe.read_text())
    value["ground_truth_used"] = True
    _write_json(unsafe, value)
    assert find_latest_completed_matrix(tmp_path) == "matrix_old"


def test_wsl_demo_runner_is_local_only_and_has_refresh_mode() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    text = RUNNER.read_text(encoding="utf-8").lower()
    assert "run_simulation_multiscenario_acceptance_wsl.sh" in text
    assert "build_wsl_competition_demo_snapshot.py" in text
    assert "python3 -m streamlit" in text
    for forbidden in (
        "192.168.",
        "\nscp ",
        "\nssh ",
        "scp -",
        "ssh -",
        "vmhostname",
        "ground_truth/",
    ):
        assert forbidden not in text
