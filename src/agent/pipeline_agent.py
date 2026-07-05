#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent MVP for orchestrating S4-S8 offline pipeline stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import audit_db


TAIL_CHARS = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def load_agent_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read agent config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "agent" not in data:
        raise ValueError("Agent config must contain a top-level 'agent' field.")
    return data["agent"]


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(project_root: Path, outputs: dict[str, str]) -> None:
    for key in ("agent_dir", "audit_dir", "db_dir", "json_dir", "report_dir", "figure_dir"):
        resolve_project_path(project_root, outputs[key]).mkdir(parents=True, exist_ok=True)


def run_stage(project_root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    stage_name = str(stage["name"])
    command_stage = str(stage["command_stage"])
    config_path = str(stage["config"])
    command = [
        sys.executable,
        "run_offline_pipeline.py",
        "--stage",
        command_stage,
        "--config",
        config_path,
    ]
    start_time = utc_now()
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    end_time = utc_now()
    status = "success" if completed.returncode == 0 else "failed"
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "stage_name": stage_name,
        "command_stage": command_stage,
        "command": " ".join(command),
        "config_path": config_path,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "return_code": int(completed.returncode),
        "stdout_tail": stdout[-TAIL_CHARS:],
        "stderr_tail": stderr[-TAIL_CHARS:],
    }


def collect_metrics(project_root: Path, agent_config: dict[str, Any]) -> dict[str, Any]:
    expected = agent_config["expected_outputs"]
    area_volume = load_json(resolve_project_path(project_root, expected["area_volume_result"]))
    weather = load_json(resolve_project_path(project_root, expected["weather_correction_result"]))
    final_forecast = load_json(resolve_project_path(project_root, expected["final_forecast_result"]))
    physical_constraint = load_json(resolve_project_path(project_root, expected["physical_constraint_result"]))
    warning = load_json(resolve_project_path(project_root, expected["warning_decision_result"]))
    final_by_horizon = {
        int(item["horizon_min"]): float(item["final_forecast_depth_cm"])
        for item in final_forecast.get("final_forecast_results", [])
    }
    return {
        "overall_warning_level": warning.get("overall_warning_level"),
        "current_mean_depth_cm": float(warning["current_mean_depth_cm"]),
        "water_area_m2": float(area_volume["water_area_m2"]),
        "water_volume_m3": float(area_volume["water_volume_m3"]),
        "rainfall_intensity_mm_h": float(weather["current_rainfall_intensity_mm_h"]),
        "weather_correction_factor": float(weather["weather_correction_factor"]),
        "final_forecast_5min_cm": final_by_horizon.get(5),
        "final_forecast_15min_cm": final_by_horizon.get(15),
        "final_forecast_30min_cm": final_by_horizon.get(30),
        "final_forecast_60min_cm": final_by_horizon.get(60),
        "final_forecast_results": final_forecast.get("final_forecast_results", []),
        "physical_confidence_summary": final_forecast.get(
            "physical_confidence_summary",
            physical_constraint.get("physical_confidence_summary"),
        ),
        "forecast_source": warning.get("forecast_source"),
        "s7_pipeline_used": warning.get("s7_pipeline_used", []),
    }


def collect_artifacts(project_root: Path) -> list[dict[str, Any]]:
    artifact_specs = [
        ("area_volume_result", "outputs/json/water_area_volume_result.json"),
        ("weather_correction_result", "outputs/json/weather_correction_result.json"),
        ("deterministic_forecast_result", "outputs/json/deterministic_forecast_result.json"),
        ("case_retrieval_result", "outputs/json/case_retrieval_result.json"),
        ("corrected_forecast_result", "outputs/json/corrected_forecast_result.json"),
        ("physical_constraint_result", "outputs/json/physical_constraint_result.json"),
        ("final_forecast_result", "outputs/json/final_forecast_result.json"),
        ("case_retrieval_correction", "outputs/figures/case_retrieval_correction.png"),
        ("physical_constraint_summary", "outputs/figures/physical_constraint_summary.png"),
        ("warning_decision_result", "outputs/json/warning_decision_result.json"),
        ("warning_report", "outputs/reports/warning_report.md"),
        ("warning_summary", "outputs/figures/warning_summary.png"),
        ("warning_audit_log", "data/audit_logs/warning_audit_log.jsonl"),
    ]
    artifacts: list[dict[str, Any]] = []
    for artifact_type, relative_path in artifact_specs:
        path = resolve_project_path(project_root, relative_path)
        artifacts.append(
            {
                "artifact_type": artifact_type,
                "path": str(path),
                "exists_flag": int(path.exists()),
                "note": "Agent MVP artifact path only; file content is not stored in SQLite.",
            }
        )
    return artifacts


def write_summary(
    project_root: Path,
    outputs: dict[str, str],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    agent_dir = resolve_project_path(project_root, outputs["agent_dir"])
    json_dir = resolve_project_path(project_root, outputs["json_dir"])
    agent_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    data_summary = agent_dir / "agent_run_summary.json"
    output_summary = json_dir / "agent_run_summary.json"
    for path in (data_summary, output_summary):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return data_summary, output_summary


def run_agent(config_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path).expanduser()
    if project_root is None:
        root = Path.cwd().resolve()
    else:
        root = Path(project_root).expanduser().resolve()
    if not config_file.is_absolute():
        config_file = root / config_file
    agent = load_agent_config(config_file)

    configured_project_root = resolve_project_path(root, agent.get("project_root", ".")).resolve()
    outputs = agent["outputs"]
    ensure_output_dirs(configured_project_root, outputs)
    db_path = resolve_project_path(configured_project_root, agent["database"]["sqlite_path"])
    audit_db.init_db(db_path)

    run_id = make_run_id()
    start_time = utc_now()
    agent_name = str(agent.get("name", "water_agent_mvp"))
    mode = str(agent.get("mode", "offline_pipeline_agent"))
    mvp_note = str(agent.get("mvp_note", ""))
    audit_db.insert_pipeline_run_start(db_path, run_id, agent_name, mode, start_time, mvp_note)

    stage_summaries: list[dict[str, Any]] = []
    status = "success"
    metrics: dict[str, Any] = {
        "overall_warning_level": None,
        "current_mean_depth_cm": None,
        "water_area_m2": None,
        "water_volume_m3": None,
        "rainfall_intensity_mm_h": None,
        "weather_correction_factor": None,
        "final_forecast_5min_cm": None,
        "final_forecast_15min_cm": None,
        "final_forecast_30min_cm": None,
        "final_forecast_60min_cm": None,
        "final_forecast_results": [],
        "physical_confidence_summary": None,
        "forecast_source": None,
        "s7_pipeline_used": [],
    }

    for stage in agent["pipeline"]["stages"]:
        stage_result = run_stage(configured_project_root, stage)
        stage_summaries.append(stage_result)
        audit_db.insert_stage_run(
            db_path=db_path,
            run_id=run_id,
            stage_name=stage_result["stage_name"],
            command=stage_result["command"],
            config_path=stage_result["config_path"],
            start_time=stage_result["start_time"],
            end_time=stage_result["end_time"],
            status=stage_result["status"],
            return_code=stage_result["return_code"],
            stdout_tail=stage_result["stdout_tail"],
            stderr_tail=stage_result["stderr_tail"],
        )
        print(f"[agent] stage {stage_result['stage_name']}: {stage_result['status']}")
        if stage_result["status"] != "success":
            status = "failed"
            break

    if status == "success":
        metrics = collect_metrics(configured_project_root, agent)

    end_time = utc_now()
    audit_db.update_pipeline_run_end(
        db_path=db_path,
        run_id=run_id,
        end_time=end_time,
        status=status,
        overall_warning_level=metrics.get("overall_warning_level"),
        current_mean_depth_cm=metrics.get("current_mean_depth_cm"),
        water_area_m2=metrics.get("water_area_m2"),
        water_volume_m3=metrics.get("water_volume_m3"),
        rainfall_intensity_mm_h=metrics.get("rainfall_intensity_mm_h"),
        weather_correction_factor=metrics.get("weather_correction_factor"),
        final_forecast_5min_cm=metrics.get("final_forecast_5min_cm"),
        final_forecast_15min_cm=metrics.get("final_forecast_15min_cm"),
        final_forecast_30min_cm=metrics.get("final_forecast_30min_cm"),
        final_forecast_60min_cm=metrics.get("final_forecast_60min_cm"),
        physical_confidence_summary=json.dumps(
            metrics.get("physical_confidence_summary"),
            ensure_ascii=False,
        )
        if metrics.get("physical_confidence_summary") is not None
        else None,
        forecast_source=metrics.get("forecast_source"),
        s7_pipeline_used=json.dumps(metrics.get("s7_pipeline_used", []), ensure_ascii=False),
    )

    artifacts = collect_artifacts(configured_project_root)
    for artifact in artifacts:
        audit_db.insert_artifact(
            db_path=db_path,
            run_id=run_id,
            artifact_type=artifact["artifact_type"],
            path=artifact["path"],
            exists_flag=artifact["exists_flag"],
            note=artifact["note"],
        )

    summary = {
        "run_id": run_id,
        "agent_name": agent_name,
        "mode": mode,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "stages": stage_summaries,
        "overall_warning_level": metrics.get("overall_warning_level"),
        "current_mean_depth_cm": metrics.get("current_mean_depth_cm"),
        "water_area_m2": metrics.get("water_area_m2"),
        "water_volume_m3": metrics.get("water_volume_m3"),
        "rainfall_intensity_mm_h": metrics.get("rainfall_intensity_mm_h"),
        "weather_correction_factor": metrics.get("weather_correction_factor"),
        "final_forecast_5min_cm": metrics.get("final_forecast_5min_cm"),
        "final_forecast_15min_cm": metrics.get("final_forecast_15min_cm"),
        "final_forecast_30min_cm": metrics.get("final_forecast_30min_cm"),
        "final_forecast_60min_cm": metrics.get("final_forecast_60min_cm"),
        "final_forecast_results": metrics.get("final_forecast_results", []),
        "physical_confidence_summary": metrics.get("physical_confidence_summary"),
        "forecast_source": metrics.get("forecast_source"),
        "s7_pipeline_used": metrics.get("s7_pipeline_used", []),
        "artifacts": artifacts,
        "sqlite_db_path": str(db_path),
        "mvp_note": mvp_note,
    }
    data_summary, output_summary = write_summary(configured_project_root, outputs, summary)

    print(f"[agent] run_id: {run_id}")
    print(f"[agent] status: {status}")
    print(f"[agent] overall warning level: {summary['overall_warning_level']}")
    print(f"[agent] current mean depth cm: {summary['current_mean_depth_cm']}")
    print(f"[agent] water area m2: {summary['water_area_m2']}")
    print(f"[agent] water volume m3: {summary['water_volume_m3']}")
    print(f"[agent] rainfall intensity: {summary['rainfall_intensity_mm_h']}")
    print(f"[agent] weather correction factor: {summary['weather_correction_factor']}")
    print(f"[agent] forecast source: {summary['forecast_source']}")
    print(
        "[agent] final forecast 5/15/30/60 min cm: "
        f"{summary['final_forecast_5min_cm']} / "
        f"{summary['final_forecast_15min_cm']} / "
        f"{summary['final_forecast_30min_cm']} / "
        f"{summary['final_forecast_60min_cm']}"
    )
    print(f"[agent] physical confidence summary: {summary['physical_confidence_summary']}")
    print(f"[agent] agent summary path: {data_summary}")
    print(f"[agent] output summary path: {output_summary}")
    print(f"[agent] sqlite db path: {db_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the water_agent_system offline pipeline Agent MVP.")
    parser.add_argument("--config", required=True, help="Path to configs/agent_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    summary = run_agent(args.config, args.project_root)
    if summary["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
