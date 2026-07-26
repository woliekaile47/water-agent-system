"""Summarize independently executed simulation-only acceptance scenarios."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CONFIG_KEY = "phase2d_c15_multiscenario_acceptance"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = document[CONFIG_KEY]
    scenarios = config.get("scenarios", [])
    if len(scenarios) != 4 or len({item["label"] for item in scenarios}) != 4:
        raise ValueError("acceptance matrix must contain four uniquely labelled scenarios")
    safety = config.get("safety", {})
    for key in (
        "simulation_only",
        "ground_truth_used",
        "authoritative",
        "eligible_for_real_warning",
    ):
        required = False if key != "simulation_only" else True
        if safety.get(key) is not required:
            raise ValueError(f"unsafe matrix setting: {key}")
    return config


def summarize_matrix(
    repo_root: str | Path,
    config_path: str | Path,
    matrix_id: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_matrix_config(config_file)
    rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        run_id = f"{matrix_id}_{scenario['label']}"
        summary_path = (
            root
            / "outputs"
            / "phase2d_c13_one_click_runs"
            / run_id
            / "completion_summary.json"
        )
        summary = read_json(summary_path)
        expected = scenario["expected"]
        matched = (
            summary.get("status") == "success"
            and summary.get("camera_visible_status")
            == expected["camera_visible_status"]
            and summary.get("global_scene_status")
            == expected["global_scene_status"]
            and summary.get("agent_should_run") is expected["agent_should_run"]
            and summary.get("ground_truth_used") is False
            and summary.get("authoritative") is False
            and summary.get("eligible_for_real_warning") is False
        )
        rows.append(
            {
                "label": scenario["label"],
                "run_id": run_id,
                "status": summary.get("status"),
                "camera_visible_status": summary.get("camera_visible_status"),
                "global_scene_status": summary.get("global_scene_status"),
                "agent_status": summary.get("agent_status"),
                "agent_should_run": summary.get("agent_should_run"),
                "estimated_water_level_m": summary.get("estimated_water_level_m"),
                "mean_depth_cm": summary.get("mean_depth_cm"),
                "max_depth_cm": summary.get("max_depth_cm"),
                "water_area_m2": summary.get("water_area_m2"),
                "water_volume_m3": summary.get("water_volume_m3"),
                "camera_reprojection_iou": summary.get("camera_reprojection_iou"),
                "quality_reject_reasons": summary.get("quality_reject_reasons", []),
                "global_scope_reasons": summary.get("global_scope_reasons", []),
                "expected_outcome_matched": matched,
            }
        )
    passed = all(row["expected_outcome_matched"] for row in rows)
    return {
        "status": "pass" if passed else "fail",
        "protocol_version": config["protocol_version"],
        "matrix_id": matrix_id,
        "scenario_count": len(rows),
        "matched_scenario_count": sum(
            int(row["expected_outcome_matched"]) for row in rows
        ),
        "scenarios": rows,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2D-C-15 多场景系统验收",
        "",
        f"- 矩阵状态：`{summary['status']}`",
        f"- 场景：{summary['matched_scenario_count']}/{summary['scenario_count']} 符合冻结预期",
        "- 数据域：仿真；Ground Truth 未进入预测；不具备真实预警资格。",
        "",
        "| 水深 | Camera 可见结果 | 全局结果 | Agent | 最大水深/cm | 面积/m² | 体积/m³ | 符合预期 |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["scenarios"]:
        lines.append(
            "| {label} | {camera_visible_status} | {global_scene_status} | "
            "{agent_status} | {max_depth_cm} | {water_area_m2} | "
            "{water_volume_m3} | {expected_outcome_matched} |".format(**row)
        )
    lines.extend(
        [
            "",
            "说明：5 cm 的安全拒绝以及 40 cm 的全局 partial 都属于预期行为；",
            "它们不会被强行送入正式 Agent 链路。",
            "",
        ]
    )
    return "\n".join(lines)
