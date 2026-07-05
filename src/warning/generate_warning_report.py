#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Generate Markdown warning report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.warning.generate_warning_decision import load_warning_config, resolve_project_path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_time(value: Any) -> str:
    if value is None:
        return "not reached in current forecast horizons"
    return f"{float(value):.2f} min"


def _fmt_depth(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f} cm"


def _input_path(root: Path, config: dict[str, Any], key: str) -> Path | None:
    value = config["input"].get(key)
    if not value:
        return None
    return resolve_project_path(root, value)


def _deterministic_lines(forecast: dict[str, Any] | None) -> list[str]:
    if not forecast:
        return ["- S7-A deterministic forecast result is not available."]
    lines = [
        f"- k_forecast: {float(forecast.get('k_forecast_cm_per_min', 0.0)):.4f} cm/min",
        f"- source: {forecast.get('depth_history_source', 'N/A')}",
    ]
    for item in forecast.get("forecast_results", []):
        lines.append(
            "- "
            f"{int(item['horizon_min'])} min: "
            f"{float(item['forecast_depth_cm']):.2f} cm, "
            f"warning {item.get('warning_level', 'N/A')}"
        )
    return lines


def _case_retrieval_lines(
    case_result: dict[str, Any] | None,
    corrected: dict[str, Any] | None,
) -> list[str]:
    if not case_result and not corrected:
        return ["- S7-B case retrieval correction result is not available."]
    lines: list[str] = []
    if case_result:
        lines.append(f"- case source: {case_result.get('mode', 'offline_mock_case_library')}")
        retrieved = case_result.get("retrieved_cases", [])
        if retrieved:
            case_text = []
            for item in retrieved[:3]:
                case_id = item.get("case_id", "unknown_case")
                score = item.get("similarity_score")
                if score is None:
                    case_text.append(case_id)
                else:
                    case_text.append(f"{case_id} ({float(score):.4f})")
            lines.append("- top retrieved cases: " + ", ".join(case_text))
        bias = case_result.get("median_bias_cm_by_horizon", {})
        if bias:
            lines.append(
                "- median bias cm: "
                + ", ".join(f"{h}min={float(v):.2f}" for h, v in sorted(bias.items(), key=lambda x: int(x[0])))
            )
    if corrected:
        for item in corrected.get("corrected_forecast_results", []):
            lines.append(
                "- "
                f"{int(item['horizon_min'])} min corrected: "
                f"{float(item['corrected_forecast_depth_cm']):.2f} cm "
                f"(deterministic {float(item['deterministic_forecast_depth_cm']):.2f} cm), "
                f"warning {item.get('warning_level', 'N/A')}"
            )
    return lines


def _physical_constraint_lines(
    physical: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> list[str]:
    if not physical and not final:
        return ["- S7-C physical constraint result is not available."]
    lines: list[str] = []
    if physical:
        lines.append(f"- method: {physical.get('physical_constraint_mode', 'simplified_water_balance_mvp')}")
        lines.append(f"- tolerance ratio: {float(physical.get('tolerance_ratio', 0.0)):.4f}")
        for item in physical.get("physical_check_results", []):
            lines.append(
                "- "
                f"{int(item['horizon_min'])} min: "
                f"corrected {_fmt_depth(item.get('corrected_depth_cm'))} -> "
                f"adjusted {_fmt_depth(item.get('adjusted_depth_cm'))}, "
                f"check {item.get('physical_check', 'N/A')}, "
                f"confidence {item.get('physical_confidence', 'N/A')}"
            )
    if final:
        summary = final.get("physical_confidence_summary")
        if summary:
            lines.append(f"- physical confidence summary: `{json.dumps(summary, ensure_ascii=False)}`")
        lines.append(f"- S7-C overall warning level: {final.get('overall_warning_level', 'N/A')}")
    return lines


def _final_forecast_lines(decision: dict[str, Any]) -> list[str]:
    lines = []
    for item in decision["forecast_warning_results"]:
        confidence = item.get("physical_confidence")
        confidence_text = f", confidence {confidence}" if confidence else ""
        lines.append(
            "- "
            f"{int(item['horizon_min'])} min: "
            f"{float(item['forecast_depth_cm']):.2f} cm, "
            f"warning {item['warning_level']}{confidence_text}"
        )
    return lines


def generate_warning_report(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    output_config = config["output"]
    json_dir = resolve_project_path(root, output_config["json_dir"])
    report_dir = resolve_project_path(root, output_config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    decision_path = json_dir / "warning_decision_result.json"
    decision = load_json(decision_path)
    deterministic = load_optional_json(_input_path(root, config, "deterministic_forecast_result_json"))
    case_result = load_optional_json(_input_path(root, config, "case_retrieval_result_json"))
    corrected = load_optional_json(_input_path(root, config, "corrected_forecast_result_json"))
    physical = load_optional_json(_input_path(root, config, "physical_constraint_result_json"))
    final = load_optional_json(_input_path(root, config, "final_forecast_result_json"))
    report_path = report_dir / "warning_report.md"
    times = decision.get("time_to_thresholds_min", {})

    input_paths = [
        decision.get("source_final_forecast_json"),
        decision.get("source_physical_constraint_json"),
        decision.get("source_case_retrieval_json"),
        decision.get("source_corrected_forecast_json"),
        decision.get("source_deterministic_forecast_json"),
        decision.get("source_area_volume_json"),
        decision.get("source_weather_correction_json"),
    ]
    output_paths = [
        decision["output_files"]["data_warning_decision"],
        decision["output_files"]["output_warning_decision"],
        str(report_path),
        decision["output_files"]["audit_log"],
        decision["output_files"]["warning_summary"],
    ]

    markdown = f"""# S8 Waterlogging Warning Report - MVP

This report is generated from MVP simulation data and is not final real emergency dispatch advice.

The final warning is based on S7-C final forecast when available.

## Current Status

- Forecast source: **{decision.get('forecast_source', 'N/A')}**
- S7 pipeline used: {', '.join(decision.get('s7_pipeline_used', []))}
- Overall warning level: **{decision['overall_warning_level']}**
- Current mean depth: {decision['current_mean_depth_cm']:.2f} cm
- Current max depth: {decision['current_max_depth_cm']:.2f} cm
- Confidence: {decision['confidence_score']:.2f} ({decision['confidence_mode']})

## S5 Area And Volume

- Water area: {decision['water_area_m2']:.4f} m2
- Water volume: {decision['water_volume_m3']:.4f} m3

## S6 Weather Correction

- Rainfall intensity: {decision['rainfall_intensity_mm_h']:.2f} mm/h
- Rainfall level: {decision['rainfall_level_label']}
- Weather correction factor: {decision['weather_correction_factor']:.2f}

## S7 Three-Layer Hybrid Reasoning Results

### S7-A Deterministic Forecast

{chr(10).join(_deterministic_lines(deterministic))}

### S7-B Case Retrieval Correction

{chr(10).join(_case_retrieval_lines(case_result, corrected))}

### S7-C Physical Constraint Check

{chr(10).join(_physical_constraint_lines(physical, final))}

### Final Forecast Used By S8

{chr(10).join(_final_forecast_lines(decision))}

## S8 Warning Decision

- Overall warning level: **{decision['overall_warning_level']}**
- Time to blue threshold: {_fmt_time(times.get('blue'))}
- Time to yellow threshold: {_fmt_time(times.get('yellow'))}
- Time to orange threshold: {_fmt_time(times.get('orange'))}
- Action suggestion: {decision['action_suggestion']}

## MVP Note

{decision['mvp_note']}

{decision.get('upstream_mvp_note', '')}

## Input Files

{chr(10).join(f'- `{path}`' for path in input_paths if path)}

## Output Files

{chr(10).join(f'- `{path}`' for path in output_paths)}
"""
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[S8][report] report path: {report_path}")
    return {"warning_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="S8 Markdown warning report generation.")
    parser.add_argument("--config", required=True, help="Path to configs/warning_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    generate_warning_report(args.config, args.project_root)


if __name__ == "__main__":
    main()
