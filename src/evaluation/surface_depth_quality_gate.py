#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality gate for S4-real surface DEM depth inversion results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dem.build_surface_dem_from_rosbag import get_case_config, load_yaml, resolve_project_path


LOW_COVERAGE_REJECT_THRESHOLD = 0.15
LOW_COVERAGE_WARNING_THRESHOLD = 0.30
MEAN_ERROR_REJECT_CM = 20.0
MEAN_ERROR_WARNING_CM = 10.0
EXTREME_OUTLIER_MARGIN_CM = 50.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _find_diagnosis_case(diagnosis: dict[str, Any], case_name: str) -> dict[str, Any]:
    for item in diagnosis.get("cases", []):
        if item.get("case_name") == case_name:
            return item
    raise ValueError(f"Case {case_name!r} was not found in surface_depth_quality_diagnosis.json")


def _has_max_depth_outlier_warning(depth_result: dict[str, Any]) -> bool:
    warnings = [str(item) for item in depth_result.get("warning") or []]
    return any("exceeded max_valid_depth_m" in item for item in warnings)


def _evaluate_gate(
    accuracy: dict[str, Any],
    depth_result: dict[str, Any],
    diagnosis_case: dict[str, Any],
) -> tuple[str, list[str], list[str], list[dict[str, Any]], bool]:
    reject_reasons: list[str] = []
    warning_reasons: list[str] = []
    checks: list[dict[str, Any]] = []

    valid_ratio = _safe_float(
        depth_result.get(
            "valid_depth_ratio_in_water_region",
            accuracy.get("valid_depth_ratio_in_water_region"),
        )
    )
    known_depth_cm = _safe_float(accuracy.get("known_depth_cm", depth_result.get("known_depth_cm")))
    mean_error_cm = _safe_float(accuracy.get("mean_error_cm"))
    max_depth_cm = _safe_float(depth_result.get("max_depth_cm", accuracy.get("max_depth_cm")))

    if valid_ratio is None:
        reject_reasons.append("missing_valid_depth_ratio")
        checks.append({"name": "coverage", "status": "reject", "value": None})
    elif valid_ratio < LOW_COVERAGE_REJECT_THRESHOLD:
        reject_reasons.append(
            f"low_coverage: valid_depth_ratio_in_water_region={valid_ratio:.4f} < "
            f"{LOW_COVERAGE_REJECT_THRESHOLD:.2f}"
        )
        checks.append({"name": "coverage", "status": "reject", "value": valid_ratio})
    elif valid_ratio < LOW_COVERAGE_WARNING_THRESHOLD:
        warning_reasons.append(
            f"low_coverage_warning: valid_depth_ratio_in_water_region={valid_ratio:.4f} < "
            f"{LOW_COVERAGE_WARNING_THRESHOLD:.2f}"
        )
        checks.append({"name": "coverage", "status": "warning", "value": valid_ratio})
    else:
        checks.append({"name": "coverage", "status": "pass", "value": valid_ratio})

    if known_depth_cm is None:
        checks.append(
            {
                "name": "accuracy_against_known_depth",
                "status": "skipped",
                "reason": "known_depth_cm is not available; accuracy is not fabricated",
            }
        )
    elif mean_error_cm is None:
        reject_reasons.append("missing_mean_error_cm_for_known_depth_case")
        checks.append({"name": "accuracy_against_known_depth", "status": "reject", "value": None})
    elif abs(mean_error_cm) > MEAN_ERROR_REJECT_CM:
        reject_reasons.append(
            f"high_mean_error: abs(mean_error_cm)={abs(mean_error_cm):.2f} cm > "
            f"{MEAN_ERROR_REJECT_CM:.2f} cm"
        )
        checks.append({"name": "accuracy_against_known_depth", "status": "reject", "value": mean_error_cm})
    elif abs(mean_error_cm) > MEAN_ERROR_WARNING_CM:
        warning_reasons.append(
            f"mean_error_warning: abs(mean_error_cm)={abs(mean_error_cm):.2f} cm > "
            f"{MEAN_ERROR_WARNING_CM:.2f} cm"
        )
        checks.append({"name": "accuracy_against_known_depth", "status": "warning", "value": mean_error_cm})
    else:
        checks.append({"name": "accuracy_against_known_depth", "status": "pass", "value": mean_error_cm})

    extreme_outlier_warning = False
    if known_depth_cm is not None and max_depth_cm is not None:
        threshold = known_depth_cm + EXTREME_OUTLIER_MARGIN_CM
        if max_depth_cm > threshold:
            extreme_outlier_warning = True
            reject_reasons.append(
                f"extreme_max_depth_outlier: max_depth_cm={max_depth_cm:.2f} cm > "
                f"known_depth_cm + {EXTREME_OUTLIER_MARGIN_CM:.2f} cm ({threshold:.2f} cm)"
            )
            checks.append({"name": "extreme_outlier", "status": "reject", "value": max_depth_cm})
        else:
            checks.append({"name": "extreme_outlier", "status": "pass", "value": max_depth_cm})
    elif _has_max_depth_outlier_warning(depth_result):
        extreme_outlier_warning = True
        reject_reasons.append("extreme_outlier_warning: depth result contains max_valid_depth_m exceedance")
        checks.append(
            {
                "name": "extreme_outlier",
                "status": "reject",
                "value": max_depth_cm,
                "reason": "known depth unavailable; using depth-result max_valid_depth_m warning",
            }
        )
    else:
        checks.append(
            {
                "name": "extreme_outlier",
                "status": "skipped" if known_depth_cm is None else "pass",
                "value": max_depth_cm,
            }
        )

    high_error_warning = bool(diagnosis_case.get("high_error_warning", False))
    coverage_warning = bool(
        diagnosis_case.get("coverage_warning", diagnosis_case.get("low_coverage_warning", False))
    )
    if high_error_warning and coverage_warning:
        reject_reasons.append("combined_quality_failure: high_error_warning=True and coverage_warning=True")
        checks.append({"name": "combined_diagnosis_warnings", "status": "reject"})
    else:
        checks.append(
            {
                "name": "combined_diagnosis_warnings",
                "status": "pass",
                "high_error_warning": high_error_warning,
                "coverage_warning": coverage_warning,
            }
        )

    if reject_reasons:
        return "reject", reject_reasons, warning_reasons, checks, extreme_outlier_warning
    if warning_reasons:
        return "warning", reject_reasons, warning_reasons, checks, extreme_outlier_warning
    return "pass", reject_reasons, warning_reasons, checks, extreme_outlier_warning


def write_gate_report(report_path: Path, json_dir: Path) -> None:
    rows = []
    for path in sorted(json_dir.glob("surface_depth_quality_gate_*.json")):
        try:
            rows.append(load_json(path))
        except Exception:
            continue

    lines = [
        "# S4-real Surface Depth Quality Gate Report",
        "",
        "This report records whether S4-real offline LiDAR surface-depth results are allowed to enter downstream S5-S8 warning stages.",
        "",
        "S4-real is still in an experimental stage. Rejected results are preserved as diagnostic artifacts only.",
        "",
        "| Case | Quality status | Known cm | Mean cm | Max cm | Mean error cm | Valid ratio | Can enter S5-S8 | Main reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in rows:
        reasons = item.get("reject_reasons") or item.get("warning_reasons") or ["none"]
        lines.append(
            "| "
            f"{item.get('case_name')} | "
            f"{item.get('quality_status')} | "
            f"{_fmt(item.get('known_depth_cm'))} | "
            f"{_fmt(item.get('mean_depth_cm'))} | "
            f"{_fmt(item.get('max_depth_cm'))} | "
            f"{_fmt(item.get('mean_error_cm'))} | "
            f"{float(item.get('valid_depth_ratio_in_water_region', 0.0)):.4f} | "
            f"{item.get('can_enter_s5_s8_warning_chain')} | "
            f"{'; '.join(str(reason) for reason in reasons)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `playground_pit_water_sim_6cm_001` is judged as `reject` when the current metrics show low coverage, high mean error, and extreme maximum-depth outlier behavior.",
            "- Rejected S4-real results are saved for diagnosis only and must not enter the formal S5-S8 warning chain.",
            "- The dormitory 39cm controlled scene indicates the offline surface DEM chain is feasible, but shallow-water playground scenes need further calibration and quality control.",
            "- The quality gate does not tune parameters, alter rosbags, or force measured depths toward known depths.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def surface_depth_quality_gate(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", selected_case))
    json_dir = resolve_project_path(root, config["output"]["json_dir"])
    report_dir = resolve_project_path(root, config["output"]["report_dir"])
    json_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = json_dir / f"surface_depth_accuracy_{output_name}.json"
    depth_result_path = json_dir / f"surface_water_depth_result_{output_name}.json"
    diagnosis_path = json_dir / "surface_depth_quality_diagnosis.json"
    output_json = json_dir / f"surface_depth_quality_gate_{output_name}.json"
    output_report = report_dir / "surface_depth_quality_gate_report.md"

    accuracy = load_json(accuracy_path)
    depth_result = load_json(depth_result_path)
    diagnosis = load_json(diagnosis_path)
    diagnosis_case = _find_diagnosis_case(diagnosis, output_name)

    quality_status, reject_reasons, warning_reasons, checks, extreme_outlier_warning = _evaluate_gate(
        accuracy,
        depth_result,
        diagnosis_case,
    )

    known_depth_cm = _safe_float(accuracy.get("known_depth_cm", depth_result.get("known_depth_cm")))
    mean_depth_cm = _safe_float(accuracy.get("mean_depth_cm", depth_result.get("mean_depth_cm")))
    median_depth_cm = _safe_float(accuracy.get("median_depth_cm", depth_result.get("median_depth_cm")))
    max_depth_cm = _safe_float(depth_result.get("max_depth_cm", accuracy.get("max_depth_cm")))
    mean_error_cm = _safe_float(accuracy.get("mean_error_cm"))
    valid_ratio = _safe_float(
        depth_result.get(
            "valid_depth_ratio_in_water_region",
            accuracy.get("valid_depth_ratio_in_water_region"),
        )
    )
    coverage_warning = bool(
        diagnosis_case.get("coverage_warning", diagnosis_case.get("low_coverage_warning", False))
    )

    result = {
        "stage": "S4_real_surface_depth_quality_gate",
        "case_name": output_name,
        "quality_status": quality_status,
        "can_enter_s5_s8_warning_chain": quality_status != "reject",
        "source_surface_depth_accuracy_json": str(accuracy_path),
        "source_surface_water_depth_result_json": str(depth_result_path),
        "source_surface_depth_quality_diagnosis_json": str(diagnosis_path),
        "known_depth_cm": known_depth_cm,
        "mean_depth_cm": mean_depth_cm,
        "median_depth_cm": median_depth_cm,
        "max_depth_cm": max_depth_cm,
        "mean_error_cm": mean_error_cm,
        "abs_mean_error_cm": None if mean_error_cm is None else abs(mean_error_cm),
        "valid_depth_ratio_in_water_region": valid_ratio,
        "high_error_warning": bool(diagnosis_case.get("high_error_warning", False)),
        "coverage_warning": coverage_warning,
        "sparse_point_warning": bool(diagnosis_case.get("sparse_point_warning", False)),
        "extreme_outlier_warning": extreme_outlier_warning,
        "reject_reasons": reject_reasons,
        "warning_reasons": warning_reasons,
        "gate_checks": checks,
        "thresholds": {
            "coverage_reject_below": LOW_COVERAGE_REJECT_THRESHOLD,
            "coverage_warning_below": LOW_COVERAGE_WARNING_THRESHOLD,
            "mean_error_reject_abs_gt_cm": MEAN_ERROR_REJECT_CM,
            "mean_error_warning_abs_gt_cm": MEAN_ERROR_WARNING_CM,
            "max_depth_reject_gt_known_plus_cm": EXTREME_OUTLIER_MARGIN_CM,
        },
        "note": (
            "S4-real quality gate is experimental. Rejected results are diagnostic artifacts only "
            "and must not enter the formal S5-S8 warning chain."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "surface_depth_quality_gate_json": str(output_json),
            "surface_depth_quality_gate_report": str(output_report),
        },
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_gate_report(output_report, json_dir)

    print(f"[surface_quality_gate] case_name: {output_name}")
    print(f"[surface_quality_gate] quality_status: {quality_status}")
    print(f"[surface_quality_gate] can_enter_s5_s8_warning_chain: {result['can_enter_s5_s8_warning_chain']}")
    print(f"[surface_quality_gate] valid_depth_ratio_in_water_region: {_fmt(valid_ratio)}")
    print(f"[surface_quality_gate] mean_error_cm: {_fmt(mean_error_cm)}")
    print(f"[surface_quality_gate] max_depth_cm: {_fmt(max_depth_cm)}")
    if reject_reasons:
        print("[surface_quality_gate] reject reasons:")
        for reason in reject_reasons:
            print(f"  - {reason}")
    if warning_reasons:
        print("[surface_quality_gate] warning reasons:")
        for reason in warning_reasons:
            print(f"  - {reason}")
    print("[surface_quality_gate] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S4-real surface depth quality gate.")
    parser.add_argument("--config", default="configs/surface_dem_config.yaml", help="Path to surface DEM config")
    parser.add_argument("--case", required=True, help="Case name, e.g. playground_pit_water_sim_6cm_001")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    surface_depth_quality_gate(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
