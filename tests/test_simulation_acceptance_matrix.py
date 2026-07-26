from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.integration.simulation_acceptance_matrix import (
    load_matrix_config,
    render_markdown,
    summarize_matrix,
)


def matrix_document() -> dict:
    scenarios = []
    values = [
        ("5cm", "reject", "unavailable", False),
        ("10cm", "pass", "complete", True),
        ("20cm", "pass", "complete", True),
        ("40cm", "pass", "partial", False),
    ]
    for label, visible, global_status, agent in values:
        scenarios.append(
            {
                "label": label,
                "config": f"configs/{label}.yaml",
                "expected": {
                    "camera_visible_status": visible,
                    "global_scene_status": global_status,
                    "agent_should_run": agent,
                },
            }
        )
    return {
        "phase2d_c15_multiscenario_acceptance": {
            "protocol_version": "test",
            "output_root": "outputs/matrix",
            "scenarios": scenarios,
            "safety": {
                "simulation_only": True,
                "ground_truth_used": False,
                "authoritative": False,
                "eligible_for_real_warning": False,
            },
        }
    }


def test_matrix_summary_accepts_expected_pass_and_safe_blocks(tmp_path: Path) -> None:
    document = matrix_document()
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    matrix_id = "matrix_001"
    for scenario in document["phase2d_c15_multiscenario_acceptance"]["scenarios"]:
        expected = scenario["expected"]
        target = (
            tmp_path
            / "outputs"
            / "phase2d_c13_one_click_runs"
            / f"{matrix_id}_{scenario['label']}"
        )
        target.mkdir(parents=True)
        (target / "completion_summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    **expected,
                    "agent_status": (
                        "success"
                        if expected["agent_should_run"]
                        else "blocked_by_quality_gate"
                    ),
                    "ground_truth_used": False,
                    "authoritative": False,
                    "eligible_for_real_warning": False,
                }
            ),
            encoding="utf-8",
        )
    summary = summarize_matrix(tmp_path, config, matrix_id)
    assert summary["status"] == "pass"
    assert summary["matched_scenario_count"] == 4
    assert "安全拒绝" in render_markdown(summary)


def test_matrix_config_rejects_unsafe_ground_truth_flag(tmp_path: Path) -> None:
    document = matrix_document()
    document["phase2d_c15_multiscenario_acceptance"]["safety"][
        "ground_truth_used"
    ] = True
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    try:
        load_matrix_config(path)
    except ValueError as exc:
        assert "ground_truth_used" in str(exc)
    else:
        raise AssertionError("unsafe matrix configuration was accepted")


def test_matrix_summary_does_not_treat_missing_agent_flag_as_false(
    tmp_path: Path,
) -> None:
    document = matrix_document()
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    matrix_id = "matrix_missing_flag"
    for scenario in document["phase2d_c15_multiscenario_acceptance"]["scenarios"]:
        expected = scenario["expected"]
        target = (
            tmp_path
            / "outputs"
            / "phase2d_c13_one_click_runs"
            / f"{matrix_id}_{scenario['label']}"
        )
        target.mkdir(parents=True)
        summary = {
            "status": "success",
            "camera_visible_status": expected["camera_visible_status"],
            "global_scene_status": expected["global_scene_status"],
            "agent_should_run": expected["agent_should_run"],
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_real_warning": False,
        }
        if scenario["label"] == "5cm":
            summary.pop("agent_should_run")
        (target / "completion_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    result = summarize_matrix(tmp_path, config, matrix_id)
    assert result["status"] == "fail"
    assert result["matched_scenario_count"] == 3
