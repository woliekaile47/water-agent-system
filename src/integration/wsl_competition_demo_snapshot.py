"""Build a competition Dashboard snapshot from completed WSL-local runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


MATRIX_CONFIG_KEY = "phase2d_c15_multiscenario_acceptance"
ONE_CLICK_CONFIG_KEY = "phase2d_c13_one_click"
MATRIX_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
EXPECTED_LABELS = ("5cm", "10cm", "20cm", "40cm")


class WslCompetitionDemoError(ValueError):
    """Raised when a WSL demo snapshot cannot be built safely."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise WslCompetitionDemoError(f"YAML root must be a mapping: {path}")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WslCompetitionDemoError(f"path must be repository-relative: {value}")
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise WslCompetitionDemoError(f"path escaped repository: {value}")
    return resolved


def _require_file(root: Path, value: str | Path) -> Path:
    path = _resolve_repo_path(root, value)
    if not path.is_file():
        raise WslCompetitionDemoError(f"required WSL demo artifact is missing: {value}")
    if "ground_truth" in path.as_posix().lower():
        raise WslCompetitionDemoError(f"Ground Truth is forbidden in the demo builder: {value}")
    return path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _validate_matrix_id(matrix_id: str) -> str:
    if not MATRIX_ID_PATTERN.fullmatch(matrix_id):
        raise WslCompetitionDemoError("matrix-id contains unsafe characters")
    return matrix_id


def find_latest_completed_matrix(
    project_root: str | Path,
    matrix_output_root: str | Path = "outputs/phase2d_c15_multiscenario_acceptance",
) -> str:
    """Return the newest safe, passing local matrix ID."""

    root = Path(project_root).resolve()
    output_root = _resolve_repo_path(root, matrix_output_root)
    candidates: list[tuple[float, str]] = []
    if output_root.is_dir():
        for summary_path in output_root.glob("*/acceptance_summary.json"):
            matrix_id = summary_path.parent.name
            if not MATRIX_ID_PATTERN.fullmatch(matrix_id):
                continue
            try:
                summary = _read_json(summary_path)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(summary, dict)
                and summary.get("status") == "pass"
                and summary.get("ground_truth_used") is False
                and summary.get("authoritative") is False
                and summary.get("eligible_for_real_warning") is False
            ):
                candidates.append((summary_path.stat().st_mtime, matrix_id))
    if not candidates:
        raise WslCompetitionDemoError("no completed safe WSL acceptance matrix was found")
    return max(candidates)[1]


def _load_matrix_config(root: Path, config_path: str | Path) -> dict[str, Any]:
    path = _require_file(root, config_path)
    document = _read_yaml(path)
    config = document.get(MATRIX_CONFIG_KEY)
    if not isinstance(config, dict):
        raise WslCompetitionDemoError(f"missing {MATRIX_CONFIG_KEY!r}")
    safety = config.get("safety")
    required = {
        "simulation_only": True,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_real_warning": False,
    }
    if not isinstance(safety, dict) or any(safety.get(key) is not value for key, value in required.items()):
        raise WslCompetitionDemoError("matrix safety policy is not fail-closed")
    return config


def _load_one_click_config(root: Path, config_path: str) -> dict[str, Any]:
    path = _require_file(root, config_path)
    document = _read_yaml(path)
    config = document.get(ONE_CLICK_CONFIG_KEY)
    if not isinstance(config, dict):
        raise WslCompetitionDemoError(f"missing {ONE_CLICK_CONFIG_KEY!r}: {config_path}")
    safety = config.get("safety")
    required = {
        "simulation_only": True,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "external_notification_allowed": False,
        "real_device_action_allowed": False,
        "real_api_calls_enabled": False,
    }
    if (
        config.get("data_domain") != "simulation"
        or config.get("ground_truth_used_for_prediction") is not False
        or not isinstance(safety, dict)
        or any(safety.get(key) is not value for key, value in required.items())
    ):
        raise WslCompetitionDemoError(f"one-click safety policy is not fail-closed: {config_path}")
    return config


def _find_anchor(rows: Any, frame_index: int) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise WslCompetitionDemoError("geometry summary must contain a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and int(row.get("frame_index", -1)) == frame_index
    ]
    if len(matches) != 1:
        raise WslCompetitionDemoError(
            f"expected one geometry row for anchor frame {frame_index}"
        )
    if matches[0].get("ground_truth_used") is not False:
        raise WslCompetitionDemoError("anchor geometry reports Ground Truth use")
    return matches[0]


def _assert_completion_safety(completion: dict[str, Any], run_id: str) -> None:
    required = {
        "status": "success",
        "expected_outcome_matched": True,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "real_device_started": False,
        "real_api_calls_enabled": False,
        "safety_checks_passed": True,
    }
    if any(completion.get(key) != value for key, value in required.items()):
        raise WslCompetitionDemoError(f"run completion is not safe and successful: {run_id}")


def build_wsl_competition_demo_snapshot(
    project_root: str | Path,
    matrix_id: str,
    matrix_config_path: str | Path = "configs/phase2d_c15_multiscenario_acceptance.yaml",
) -> dict[str, Any]:
    """Build a display-only snapshot from the completed WSL Agent matrix."""

    root = Path(project_root).resolve()
    matrix_id = _validate_matrix_id(matrix_id)
    matrix_config = _load_matrix_config(root, matrix_config_path)
    matrix_root = _resolve_repo_path(root, matrix_config["output_root"])
    matrix_summary_path = matrix_root / matrix_id / "acceptance_summary.json"
    if not matrix_summary_path.is_file():
        raise WslCompetitionDemoError(f"matrix summary is missing: {matrix_id}")
    matrix_summary = _read_json(matrix_summary_path)
    if not isinstance(matrix_summary, dict):
        raise WslCompetitionDemoError("matrix summary must be a mapping")
    if (
        matrix_summary.get("status") != "pass"
        or matrix_summary.get("matrix_id") != matrix_id
        or matrix_summary.get("ground_truth_used") is not False
        or matrix_summary.get("authoritative") is not False
        or matrix_summary.get("eligible_for_real_warning") is not False
    ):
        raise WslCompetitionDemoError("matrix summary is not a safe passing result")

    configured_scenarios = matrix_config.get("scenarios")
    summarized_scenarios = matrix_summary.get("scenarios")
    if not isinstance(configured_scenarios, list) or not isinstance(summarized_scenarios, list):
        raise WslCompetitionDemoError("matrix scenarios are malformed")
    config_by_label = {str(row.get("label")): row for row in configured_scenarios}
    summary_by_label = {str(row.get("label")): row for row in summarized_scenarios}
    if tuple(config_by_label) != EXPECTED_LABELS or set(summary_by_label) != set(EXPECTED_LABELS):
        raise WslCompetitionDemoError("matrix must contain the frozen 5/10/20/40 cm labels")

    built_cases: list[dict[str, Any]] = []
    for label in EXPECTED_LABELS:
        scenario_config = config_by_label[label]
        scenario_summary = summary_by_label[label]
        one_click = _load_one_click_config(root, str(scenario_config["config"]))
        sample = one_click["sample"]
        run_id = str(scenario_summary.get("run_id", ""))
        _validate_matrix_id(run_id)
        run_dir = _resolve_repo_path(
            root, Path(one_click["run_root_parent"]) / run_id
        )
        completion_path = run_dir / "completion_summary.json"
        if not completion_path.is_file():
            raise WslCompetitionDemoError(f"completion summary is missing: {run_id}")
        completion = _read_json(completion_path)
        if not isinstance(completion, dict):
            raise WslCompetitionDemoError(f"completion summary is malformed: {run_id}")
        _assert_completion_safety(completion, run_id)

        expected = scenario_config["expected"]
        for key in ("camera_visible_status", "global_scene_status", "agent_should_run"):
            if scenario_summary.get(key) != expected[key] or completion.get(key) != expected[key]:
                raise WslCompetitionDemoError(
                    f"matrix outcome differs from frozen expectation for {label}: {key}"
                )

        sample_id = str(sample["sample_id"])
        frame_index = int(sample["anchor_frame_index"])
        geometry_dir = run_dir / "s4_geometry" / sample_id
        geometry_rows_path = geometry_dir / "per_frame_geometry_summary.json"
        if not geometry_rows_path.is_file():
            raise WslCompetitionDemoError(f"geometry summary is missing: {run_id}")
        anchor = _find_anchor(_read_json(geometry_rows_path), frame_index)

        input_image = _require_file(
            root, Path(sample["frames_dir"]) / f"frame_{frame_index:06d}.png"
        )
        predicted_mask = _require_file(
            root,
            run_dir.relative_to(root)
            / "result"
            / "masks_png"
            / f"frame_{frame_index:06d}.png",
        )
        reprojected_mask = _require_file(
            root,
            geometry_dir.relative_to(root) / "anchor_reprojected_camera_mask.png",
        )
        plot_paths = {
            name: _require_file(root, geometry_dir.relative_to(root) / filename)
            for name, filename in {
                "water_level": "water_level_over_time.png",
                "max_depth": "max_depth_over_time.png",
                "area": "area_over_time.png",
                "volume": "volume_over_time.png",
            }.items()
        }
        assets = {
            "input_image": _relative(root, input_image),
            "predicted_mask": _relative(root, predicted_mask),
            "reprojected_mask": _relative(root, reprojected_mask),
            "plots": {name: _relative(root, path) for name, path in plot_paths.items()},
        }
        provenance_paths = {
            "input_image": input_image,
            "predicted_mask": predicted_mask,
            "reprojected_mask": reprojected_mask,
            "geometry_rows": geometry_rows_path,
            "completion_summary": completion_path,
            **{f"plot_{name}": path for name, path in plot_paths.items()},
        }
        built_cases.append(
            {
                "sample_id": sample_id,
                "display_name": f"{label.replace('cm', ' cm')} / moderate rain",
                "case_id": sample["case_id"],
                "rain_level": sample["rain_level"],
                "seed": int(sample["seed"]),
                "anchor_frame_index": frame_index,
                "nominal_depth_cm_display_only": int(label.removesuffix("cm")),
                "run_id": run_id,
                "runtime_environment": "wsl_local",
                "assets": assets,
                "prediction_metrics": {
                    "estimated_water_level_m": completion.get("estimated_water_level_m"),
                    "mean_depth_cm": completion.get("mean_depth_cm"),
                    "median_depth_cm": anchor.get("median_depth_cm"),
                    "max_depth_cm": completion.get("max_depth_cm"),
                    "water_area_m2": completion.get("water_area_m2"),
                    "water_volume_m3": completion.get("water_volume_m3"),
                    "camera_reprojection_iou": completion.get("camera_reprojection_iou"),
                    "outer_boundary_reprojection_p95_px": completion.get(
                        "outer_boundary_reprojection_p95_px"
                    ),
                    "candidate_basin_count": anchor.get("candidate_basin_count"),
                    "unobserved_candidate_basin_count": anchor.get(
                        "unobserved_candidate_basin_count"
                    ),
                },
                "quality": {
                    "camera_visible_status": completion.get("camera_visible_status"),
                    "global_scene_status": completion.get("global_scene_status"),
                    "result_semantics": anchor.get("result_semantics"),
                    "visible_reject_reasons": scenario_summary.get(
                        "quality_reject_reasons", []
                    ),
                    "global_scope_reasons": scenario_summary.get(
                        "global_scope_reasons", []
                    ),
                    "warnings": anchor.get("warnings", []),
                },
                "pipeline": {
                    "standard_pipeline_completed": completion.get(
                        "standard_pipeline_completed"
                    ),
                    "agent_status": completion.get("agent_status"),
                    "agent_should_run": completion.get("agent_should_run"),
                    "acceptance_outcome": completion.get("acceptance_outcome"),
                    "sqlite_database_exists": completion.get("sqlite_database_exists"),
                    "warning_mode": completion.get("warning_mode"),
                },
                "provenance_sha256": {
                    name: _sha256(path) for name, path in provenance_paths.items()
                },
                "ground_truth_used": False,
                "authoritative": False,
                "eligible_for_real_warning": False,
            }
        )

    return {
        "protocol_version": "phase2d_wsl_competition_demo_v1",
        "demo_mode": "wsl_local_simulation_agent_e2e",
        "source_type": "completed_wsl_simulation_agent_matrix",
        "matrix_id": matrix_id,
        "matrix_summary": _relative(root, matrix_summary_path),
        "source_policy": {
            "simulation_only": True,
            "wsl_local_runtime": True,
            "ground_truth_used_by_demo_builder": False,
            "manual_prompt_inputs_allowed": False,
            "dormitory_or_cardboard_inputs_allowed": False,
            "real_devices_started": False,
            "real_api_calls_enabled": False,
            "authoritative": False,
            "eligible_for_real_warning": False,
        },
        "case_count": len(built_cases),
        "cases": built_cases,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "eligible_for_real_warning": False,
    }
