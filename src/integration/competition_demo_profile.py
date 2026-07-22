"""Build a display-only competition snapshot from frozen simulation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class CompetitionDemoProfileError(ValueError):
    """Raised when the demo profile violates the frozen source policy."""


ALLOWED_SOURCE_ROOTS = (
    "data/simulation_dynamic",
    "outputs/phase2d_c8_seed303_video_freeze",
    "outputs/phase2d_c8_seed303_geometry_freeze",
    "outputs/phase2d_c8_seed303_candidate_gate_freeze",
)

FORBIDDEN_PATH_FRAGMENTS = (
    "ground_truth",
    "gt_evaluation",
    "manual_prompt",
    "sam2_yujian_workspace",
    "water_test",
    "blind_",
    "cardboard",
    "dormitory",
    "宿舍",
    "纸箱",
)

REQUIRED_CASE_PATH_FIELDS = (
    "input_image",
    "predicted_mask",
    "reprojected_mask",
    "geometry_rows",
    "candidate_gate_rows",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_source_path(project_root: Path, path_value: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise CompetitionDemoProfileError("demo source path must be a non-empty string")
    normalized = path_value.replace("\\", "/")
    lowered = normalized.lower()
    forbidden = [fragment for fragment in FORBIDDEN_PATH_FRAGMENTS if fragment.lower() in lowered]
    if forbidden:
        raise CompetitionDemoProfileError(
            f"demo source path contains forbidden fragment {forbidden[0]!r}: {path_value}"
        )
    relative_path = Path(normalized)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CompetitionDemoProfileError(f"demo source path must be project-relative: {path_value}")
    resolved = (project_root / relative_path).resolve()
    allowed = any(
        _is_relative_to(resolved, (project_root / root).resolve())
        for root in ALLOWED_SOURCE_ROOTS
    )
    if not allowed:
        raise CompetitionDemoProfileError(f"demo source is outside simulation allowlist: {path_value}")
    if not resolved.is_file():
        raise CompetitionDemoProfileError(f"required demo source is missing: {path_value}")
    return resolved


def _find_frame(rows: Any, frame_index: int, source_name: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise CompetitionDemoProfileError(f"{source_name} must contain a JSON list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("frame_index") == frame_index]
    if len(matches) != 1:
        raise CompetitionDemoProfileError(
            f"{source_name} must contain exactly one frame_index={frame_index} row"
        )
    return matches[0]


def _load_profile(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    profile = loaded.get("phase2d_c10_competition_demo") if isinstance(loaded, dict) else None
    if not isinstance(profile, dict):
        raise CompetitionDemoProfileError("missing phase2d_c10_competition_demo config root")
    policy = profile.get("source_policy")
    required_policy = {
        "simulation_only": True,
        "ground_truth_used_by_demo_builder": False,
        "manual_prompt_inputs_allowed": False,
        "dormitory_or_cardboard_inputs_allowed": False,
        "real_devices_started": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) is not value for key, value in required_policy.items()):
        raise CompetitionDemoProfileError("competition demo source policy is not fail-closed")
    if profile.get("demo_mode") != "offline_simulation_road_only":
        raise CompetitionDemoProfileError("demo_mode must be offline_simulation_road_only")
    if profile.get("source_type") != "gazebo_dynamic_rain_road_simulation":
        raise CompetitionDemoProfileError("source_type must be Gazebo road simulation")
    return profile


def build_competition_demo_snapshot(
    project_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate the frozen profile and return a display-only JSON snapshot."""

    root = Path(project_root).resolve()
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    profile = _load_profile(config.resolve())
    cases = profile.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CompetitionDemoProfileError("competition demo must define at least one case")

    built_cases: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise CompetitionDemoProfileError("each demo case must be a mapping")
        sample_id = str(case.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise CompetitionDemoProfileError(f"invalid or duplicate sample_id: {sample_id!r}")
        seen_sample_ids.add(sample_id)
        if not str(case.get("case_id", "")).startswith("sim_water_"):
            raise CompetitionDemoProfileError(f"case is not a water-road simulation: {sample_id}")
        if int(case.get("seed", -1)) != int(profile.get("source_seed", -2)):
            raise CompetitionDemoProfileError(f"case seed differs from frozen source seed: {sample_id}")

        resolved_paths: dict[str, Path] = {}
        relative_paths: dict[str, str] = {}
        for field in REQUIRED_CASE_PATH_FIELDS:
            relative_paths[field] = str(case.get(field, ""))
            resolved_paths[field] = _validate_source_path(root, relative_paths[field])
        plots = case.get("plots")
        if not isinstance(plots, dict) or set(plots) != {"water_level", "max_depth", "area", "volume"}:
            raise CompetitionDemoProfileError(f"case plots are incomplete: {sample_id}")
        plot_paths: dict[str, str] = {}
        resolved_plot_paths: dict[str, Path] = {}
        for name, value in plots.items():
            plot_paths[name] = str(value)
            resolved_plot_paths[name] = _validate_source_path(root, plot_paths[name])

        frame_index = int(case.get("anchor_frame_index"))
        geometry = _find_frame(_read_json(resolved_paths["geometry_rows"]), frame_index, "geometry_rows")
        gate = _find_frame(_read_json(resolved_paths["candidate_gate_rows"]), frame_index, "candidate_gate_rows")
        if geometry.get("ground_truth_used") is not False or gate.get("ground_truth_used") is not False:
            raise CompetitionDemoProfileError(f"prediction-side artifact reports Ground Truth use: {sample_id}")

        built_cases.append(
            {
                "sample_id": sample_id,
                "display_name": case.get("display_name", sample_id),
                "case_id": case.get("case_id"),
                "rain_level": case.get("rain_level"),
                "seed": case.get("seed"),
                "anchor_frame_index": frame_index,
                "nominal_depth_cm_display_only": case.get("nominal_depth_cm_display_only"),
                "assets": {
                    "input_image": relative_paths["input_image"],
                    "predicted_mask": relative_paths["predicted_mask"],
                    "reprojected_mask": relative_paths["reprojected_mask"],
                    "plots": plot_paths,
                },
                "prediction_metrics": {
                    "estimated_water_level_m": geometry.get("estimated_water_level_m"),
                    "mean_depth_cm": geometry.get("mean_depth_cm"),
                    "median_depth_cm": geometry.get("median_depth_cm"),
                    "max_depth_cm": geometry.get("max_depth_cm"),
                    "water_area_m2": geometry.get("water_area_m2"),
                    "water_volume_m3": geometry.get("water_volume_m3"),
                    "camera_reprojection_iou": geometry.get("camera_reprojection_iou"),
                    "outer_boundary_reprojection_p95_px": geometry.get(
                        "outer_boundary_reprojection_p95_px"
                    ),
                    "candidate_basin_count": geometry.get("candidate_basin_count"),
                    "unobserved_candidate_basin_count": geometry.get(
                        "unobserved_candidate_basin_count"
                    ),
                },
                "quality": {
                    "camera_visible_status": gate.get("camera_visible_status"),
                    "global_scene_status": gate.get("global_scene_status"),
                    "result_semantics": gate.get("result_semantics"),
                    "visible_reject_reasons": gate.get("visible_reject_reasons", []),
                    "warnings": gate.get("warnings", []),
                },
                "provenance_sha256": {
                    **{name: _sha256(path) for name, path in resolved_paths.items()},
                    **{f"plot_{name}": _sha256(path) for name, path in resolved_plot_paths.items()},
                },
                "ground_truth_used": False,
                "authoritative": False,
                "eligible_for_downstream": False,
            }
        )

    return {
        "protocol_version": profile.get("protocol_version"),
        "demo_mode": profile.get("demo_mode"),
        "source_type": profile.get("source_type"),
        "source_policy": profile.get("source_policy"),
        "case_count": len(built_cases),
        "cases": built_cases,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
