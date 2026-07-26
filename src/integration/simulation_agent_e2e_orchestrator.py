#!/usr/bin/env python3
"""Coordinate the VM side of the one-click simulated-sensor pipeline.

This module only orchestrates existing project stages.  It does not implement
an alternative perception, hydrology, warning, or Agent algorithm.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.phase2d_c8_candidate_quality_gate import evaluate_candidate_gate

CONFIG_KEY = "phase2d_c13_one_click"
RUNTIME_CONFIG_KEY = "phase2d_c11_direct_e2e_20cm"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if CONFIG_KEY not in document:
        raise ValueError(f"Configuration must contain top-level {CONFIG_KEY!r}")
    config = document[CONFIG_KEY]
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("data_domain") != "simulation":
        raise ValueError("one-click runner requires data_domain=simulation")
    if config.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("prediction must explicitly declare Ground Truth usage false")
    if config.get("runtime_config_key") != RUNTIME_CONFIG_KEY:
        raise ValueError(f"runtime_config_key must remain {RUNTIME_CONFIG_KEY!r}")
    safety = config.get("safety", {})
    expected = {
        "simulation_only": True,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "external_notification_allowed": False,
        "real_device_action_allowed": False,
        "real_api_calls_enabled": False,
    }
    for key, value in expected.items():
        if safety.get(key) is not value:
            raise ValueError(f"unsafe or missing safety flag: {key} must be {value!r}")
    sample = config.get("sample", {})
    start = int(sample["window_start"])
    end = int(sample["window_end"])
    anchor = int(sample["anchor_frame_index"])
    if start > anchor or anchor > end:
        raise ValueError("anchor frame must be inside the configured frame window")
    if end - start + 1 < 3:
        raise ValueError("video window must contain at least three frames")
    outcome = config.get("expected_outcome", {})
    if outcome.get("camera_visible_status") not in {"pass", "reject"}:
        raise ValueError("expected_outcome.camera_visible_status must be pass or reject")
    if outcome.get("global_scene_status") not in {
        "complete",
        "partial",
        "unavailable",
    }:
        raise ValueError(
            "expected_outcome.global_scene_status must be complete, partial or unavailable"
        )
    if not isinstance(outcome.get("agent_should_run"), bool):
        raise ValueError("expected_outcome.agent_should_run must be boolean")
    if outcome["agent_should_run"] and (
        outcome["camera_visible_status"] != "pass"
        or outcome["global_scene_status"] != "complete"
    ):
        raise ValueError("Agent may run only for a complete camera-visible estimate")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run-id may contain only letters, digits, underscore and hyphen")
    return run_id


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def resolve_run_dir(repo_root: Path, config: dict[str, Any], run_id: str) -> Path:
    validate_run_id(run_id)
    parent = resolve_repo_path(repo_root, config["run_root_parent"])
    expected_parent = (repo_root / "outputs").resolve()
    if parent != expected_parent and expected_parent not in parent.parents:
        raise ValueError("run_root_parent must remain below the repository outputs directory")
    run_dir = (parent / run_id).resolve()
    if run_dir.parent != parent:
        raise ValueError("run-id escaped the configured output parent")
    return run_dir


def run_checked(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-20:])
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def assert_required_inputs(repo_root: Path, config: dict[str, Any]) -> None:
    sample = config["sample"]
    paths = [
        config["sensor_inputs"]["dry_lidar_bag"],
        config["sensor_inputs"]["ground_dem_pipeline_config"],
        sample["frames_dir"],
        config["prediction_configs"]["temporal_prompt"],
        config["prediction_configs"]["temporal_detector"],
        config["prediction_configs"]["temporal_quality_gate"],
        config["prediction_configs"]["sensors"],
        config["prediction_configs"]["mask_to_dem_mapping"],
        config["prediction_configs"]["geometry_quality_gate"],
        config["prediction_configs"]["candidate_quality_gate"],
        config["standard_pipeline"]["agent_config"],
        config["standard_pipeline"]["case_library_json"],
    ]
    for value in paths:
        path = resolve_repo_path(repo_root, value)
        if not path.exists():
            raise FileNotFoundError(f"required pipeline input is missing: {path}")
    frames_dir = resolve_repo_path(repo_root, sample["frames_dir"])
    expected_indices = range(int(sample["window_start"]), int(sample["window_end"]) + 1)
    for frame_index in expected_indices:
        frame = frames_dir / f"frame_{frame_index:06d}.png"
        if not frame.is_file():
            raise FileNotFoundError(f"required RGB frame is missing: {frame}")
    anchor = frames_dir / f"frame_{int(sample['anchor_frame_index']):06d}.png"
    if sha256_file(anchor) != sample["anchor_image_sha256"]:
        raise ValueError("anchor RGB SHA-256 differs from the frozen configuration")


def build_runtime_config(
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample = config["sample"]
    return {
        RUNTIME_CONFIG_KEY: {
            "protocol_version": config["protocol_version"],
            "data_domain": "simulation",
            "seed": int(sample["seed"]),
            "fps": int(sample["fps"]),
            "anchor_frame_index": int(sample["anchor_frame_index"]),
            "window_start": int(sample["window_start"]),
            "window_end": int(sample["window_end"]),
            "ground_truth_used_for_prediction": False,
            "runtime_root": str((run_dir / "runtime").resolve()),
            "runtime_inputs": {
                "geometry_case_dir": str(
                    (run_dir / "s4_geometry" / sample["sample_id"]).resolve()
                ),
                "ground_dem_dir": str((run_dir / "s2" / "data" / "dem").resolve()),
                "candidate_gate_config": str(
                    resolve_repo_path(
                        repo_root, config["prediction_configs"]["candidate_quality_gate"]
                    )
                ),
                "case_library_json": str(
                    resolve_repo_path(
                        repo_root, config["standard_pipeline"]["case_library_json"]
                    )
                ),
            },
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "case_id": sample["case_id"],
                    "rain_level": sample["rain_level"],
                    "seed": int(sample["seed"]),
                    "role": sample["role"],
                    "frames_dir": str(resolve_repo_path(repo_root, sample["frames_dir"])),
                    "prompt_path": str(
                        (
                            run_dir
                            / "s3_prompt"
                            / sample["sample_id"]
                            / "automatic_prompt.json"
                        ).resolve()
                    ),
                }
            ],
        }
    }


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _create_exchange_archive(
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    sample = config["sample"]
    payload = run_dir / "exchange_to_wsl" / "payload"
    frames_target = payload / "frames"
    frames_target.mkdir(parents=True)
    frames_source = resolve_repo_path(repo_root, sample["frames_dir"])
    frame_records: list[dict[str, Any]] = []
    for frame_index in range(int(sample["window_start"]), int(sample["window_end"]) + 1):
        source = frames_source / f"frame_{frame_index:06d}.png"
        target = frames_target / source.name
        shutil.copy2(source, target)
        frame_records.append(
            {
                "frame_index": frame_index,
                "filename": target.name,
                "sha256": sha256_file(target),
            }
        )
    prompt_source = run_dir / "s3_prompt" / sample["sample_id"] / "automatic_prompt.json"
    prompt_target = payload / "automatic_prompt.json"
    shutil.copy2(prompt_source, prompt_target)
    propagation_source = repo_root / "scripts" / "run_sam2_video_propagation.py"
    propagation_target = payload / propagation_source.name
    shutil.copy2(propagation_source, propagation_target)
    manifest = {
        "protocol_version": config["protocol_version"],
        "sample_id": sample["sample_id"],
        "window_start": int(sample["window_start"]),
        "window_end": int(sample["window_end"]),
        "anchor_frame_index": int(sample["anchor_frame_index"]),
        "frame_count": len(frame_records),
        "frames": frame_records,
        "prompt_sha256": sha256_file(prompt_target),
        "propagation_script_sha256": sha256_file(propagation_target),
        "ground_truth_used": False,
        "semantic_label": "unknown_candidate",
        "authoritative": False,
    }
    write_json(payload / "exchange_manifest.json", manifest)
    archive = run_dir / "exchange_to_wsl.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="payload")
    return archive, manifest


def prepare_run(repo_root: str | Path, config_path: str | Path, run_id: str) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    config_file = resolve_repo_path(root, config_path)
    config = load_config(config_file)
    assert_required_inputs(root, config)
    run_dir = resolve_run_dir(root, config, run_id)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    sample = config["sample"]

    run_checked(
        [
            sys.executable,
            "run_offline_pipeline.py",
            "--stage",
            "ground_dem",
            "--dry_bag",
            str(resolve_repo_path(root, config["sensor_inputs"]["dry_lidar_bag"])),
            "--config",
            str(
                resolve_repo_path(
                    root, config["sensor_inputs"]["ground_dem_pipeline_config"]
                )
            ),
            "--project-root",
            str(run_dir / "s2"),
            "--skip-figures",
        ],
        root,
        logs / "s2_ground_dem.log",
    )

    frames_dir = resolve_repo_path(root, sample["frames_dir"])
    prompt_dir = run_dir / "s3_prompt" / sample["sample_id"]
    run_checked(
        [
            sys.executable,
            "scripts/generate_temporal_sam2_prompt.py",
            "--frames-dir",
            str(frames_dir),
            "--image",
            str(frames_dir / f"frame_{int(sample['anchor_frame_index']):06d}.png"),
            "--frame-index",
            str(int(sample["anchor_frame_index"])),
            "--expected-image-sha256",
            sample["anchor_image_sha256"],
            "--config",
            str(resolve_repo_path(root, config["prediction_configs"]["temporal_prompt"])),
            "--detector-config",
            str(
                resolve_repo_path(root, config["prediction_configs"]["temporal_detector"])
            ),
            "--temporal-gate-config",
            str(
                resolve_repo_path(
                    root, config["prediction_configs"]["temporal_quality_gate"]
                )
            ),
            "--output-dir",
            str(prompt_dir),
        ],
        root,
        logs / "s3_automatic_prompt.log",
    )
    prompt = read_json(prompt_dir / "automatic_prompt.json")
    if prompt.get("ground_truth_used") is not False:
        raise ValueError("automatic prompt provenance is not GT-free")
    if prompt.get("prompt_quality_status") == "reject":
        raise ValueError("automatic prompt was rejected and cannot enter SAM2")

    runtime_config = build_runtime_config(root, run_dir, config)
    runtime_config_path = run_dir / "runtime_config.yaml"
    _write_yaml(runtime_config_path, runtime_config)
    archive, exchange_manifest = _create_exchange_archive(root, run_dir, config)
    result = {
        "status": "prepared_for_wsl_sam2",
        "protocol_version": config["protocol_version"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sample_id": sample["sample_id"],
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "exchange_archive": str(archive),
        "exchange_archive_sha256": sha256_file(archive),
        "runtime_config": str(runtime_config_path),
        "frame_count": exchange_manifest["frame_count"],
        "prompt_quality_status": prompt["prompt_quality_status"],
        "ground_truth_used": False,
        "real_device_started": False,
        "real_api_calls_enabled": False,
    }
    write_json(run_dir / "prepare_summary.json", result)
    return result


def safe_extract_tar(archive: str | Path, destination: str | Path) -> None:
    archive_path = Path(archive).resolve()
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = (target / member.name).resolve()
            if target != member_path and target not in member_path.parents:
                raise ValueError(f"unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
        handle.extractall(target)


def _assert_sam2_result(run_dir: Path, config: dict[str, Any], result_dir: Path) -> None:
    sample = config["sample"]
    summary = read_json(result_dir / "video_propagation_summary.json")
    metrics = read_json(result_dir / "frame_metrics.json")
    expected_count = int(sample["window_end"]) - int(sample["window_start"]) + 1
    if summary.get("ground_truth_used") is not False:
        raise ValueError("SAM2 result provenance is not GT-free")
    if summary.get("sam2_video_propagation_completed") is not True:
        raise ValueError("SAM2 video propagation did not complete")
    if int(summary.get("frame_count", -1)) != expected_count or len(metrics) != expected_count:
        raise ValueError("SAM2 result frame count differs from the frozen window")
    prompt_path = run_dir / "s3_prompt" / sample["sample_id"] / "automatic_prompt.json"
    if summary.get("prompt_sha256") != sha256_file(prompt_path):
        raise ValueError("SAM2 result prompt hash differs from the VM-frozen prompt")


def _load_if_exists(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def assert_prepared_run_matches_config(
    prepare_summary: dict[str, Any],
    config_file: Path,
    config: dict[str, Any],
) -> None:
    if prepare_summary.get("config_sha256") != sha256_file(config_file):
        raise ValueError("prepared run configuration hash does not match finalize config")
    if prepare_summary.get("sample_id") != config["sample"]["sample_id"]:
        raise ValueError("prepared run sample does not match finalize config")


def evaluate_anchor_candidate_gate(
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the existing frozen candidate gate to the configured anchor frame."""
    sample = config["sample"]
    geometry_dir = run_dir / "s4_geometry" / sample["sample_id"]
    rows = read_json(geometry_dir / "per_frame_geometry_summary.json")
    sequence = read_json(geometry_dir / "sequence_geometry_stability.json")
    anchor_index = int(sample["anchor_frame_index"])
    anchors = [row for row in rows if int(row["frame_index"]) == anchor_index]
    if len(anchors) != 1:
        raise ValueError(f"Expected exactly one anchor geometry row for {anchor_index}")
    gate_path = resolve_repo_path(
        repo_root, config["prediction_configs"]["candidate_quality_gate"]
    )
    gate_document = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    gate_config = gate_document["phase2d_c8_candidate_quality_gate"]
    decision = evaluate_candidate_gate(anchors[0], sequence, gate_config)
    if decision.get("ground_truth_used") is not False:
        raise ValueError("candidate gate provenance is not GT-free")
    return anchors[0], decision


def assert_expected_outcome(
    config: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    expected = config["expected_outcome"]
    actual = {
        "camera_visible_status": decision.get("camera_visible_status"),
        "global_scene_status": decision.get("global_scene_status"),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(
                f"frozen candidate gate outcome mismatch for {key}: "
                f"expected {expected[key]!r}, got {value!r}"
            )


def summarize_gate_blocked_run(
    run_dir: Path,
    config: dict[str, Any],
    anchor: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    expected = config["expected_outcome"]
    return {
        "status": "success",
        "acceptance_outcome": "safely_blocked_as_expected",
        "protocol_version": config["protocol_version"],
        "run_dir": str(run_dir),
        "standard_pipeline_completed": False,
        "agent_status": "blocked_by_quality_gate",
        "agent_should_run": False,
        "expected_outcome_matched": True,
        "estimated_water_level_m": anchor.get("estimated_water_level_m"),
        "mean_depth_cm": anchor.get("mean_depth_cm"),
        "max_depth_cm": anchor.get("max_depth_cm"),
        "water_area_m2": anchor.get("water_area_m2"),
        "water_volume_m3": anchor.get("water_volume_m3"),
        "camera_reprojection_iou": anchor.get("camera_reprojection_iou"),
        "outer_boundary_reprojection_p95_px": anchor.get(
            "outer_boundary_reprojection_p95_px"
        ),
        "camera_visible_status": decision["camera_visible_status"],
        "global_scene_status": decision["global_scene_status"],
        "quality_reject_reasons": decision.get("visible_reject_reasons", []),
        "global_scope_reasons": decision.get("global_scope_reasons", []),
        "quality_warnings": decision.get("warnings", []),
        "warning_level": None,
        "warning_mode": "suppressed_by_quality_gate",
        "sqlite_database": None,
        "sqlite_database_exists": False,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "eligible_for_real_warning": False,
        "real_device_started": False,
        "real_api_calls_enabled": False,
        "safety_checks_passed": (
            expected["agent_should_run"] is False
            and decision.get("ground_truth_used") is False
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize_completed_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    runtime = run_dir / "runtime"
    manifest = read_json(runtime / "runtime_manifest.json")
    depth = read_json(runtime / "outputs" / "json" / "water_depth_result.json")
    area = read_json(runtime / "outputs" / "json" / "water_area_volume_result.json")
    agent = read_json(runtime / "outputs" / "json" / "agent_run_summary.json")
    warning = read_json(runtime / "outputs" / "json" / "warning_decision_result.json")
    database = runtime / "data" / "db" / "water_agent_audit.db"
    safety_ok = (
        manifest.get("ground_truth_used") is False
        and manifest.get("authoritative") is False
        and manifest.get("eligible_for_real_warning") is False
        and manifest.get("external_notification_allowed") is False
        and manifest.get("real_device_action_allowed") is False
        and warning.get("warning_mode") == "simulation_record_only"
    )
    return {
        "status": "success" if agent.get("status") == "success" and safety_ok else "failed",
        "protocol_version": config["protocol_version"],
        "run_dir": str(run_dir),
        "standard_pipeline_completed": agent.get("status") == "success",
        "agent_status": agent.get("status"),
        "agent_should_run": True,
        "acceptance_outcome": "standard_pipeline_completed",
        "expected_outcome_matched": True,
        "estimated_water_level_m": depth.get("estimated_water_level_m"),
        "mean_depth_cm": depth.get("mean_depth_cm"),
        "max_depth_cm": depth.get("max_depth_cm"),
        "water_area_m2": area.get("area_m2", area.get("water_area_m2")),
        "water_volume_m3": area.get("volume_m3", area.get("water_volume_m3")),
        "camera_reprojection_iou": depth.get("camera_reprojection_iou"),
        "outer_boundary_reprojection_p95_px": depth.get(
            "outer_boundary_reprojection_p95_px"
        ),
        "camera_visible_status": depth.get("candidate_gate", {}).get(
            "camera_visible_status"
        ),
        "global_scene_status": depth.get("candidate_gate", {}).get(
            "global_scene_status"
        ),
        "warning_level": warning.get("overall_warning_level"),
        "warning_mode": warning.get("warning_mode"),
        "sqlite_database": str(database),
        "sqlite_database_exists": database.is_file() and database.stat().st_size > 0,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_real_warning": False,
        "real_device_started": False,
        "real_api_calls_enabled": False,
        "safety_checks_passed": safety_ok,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def finalize_run(
    repo_root: str | Path,
    config_path: str | Path,
    run_id: str,
    sam2_archive: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    config_file = resolve_repo_path(root, config_path)
    config = load_config(config_file)
    run_dir = resolve_run_dir(root, config, run_id)
    prepare_summary = read_json(run_dir / "prepare_summary.json")
    if prepare_summary.get("status") != "prepared_for_wsl_sam2":
        raise ValueError("run was not frozen in prepared_for_wsl_sam2 state")
    assert_prepared_run_matches_config(prepare_summary, config_file, config)
    sample = config["sample"]
    propagation_root = run_dir / "s3_video"
    propagation_result = propagation_root / sample["sample_id"]
    if propagation_result.exists():
        # A failed downstream stage may be resumed without rerunning SAM2.  The
        # frozen hashes are verified again before any geometry work continues.
        _assert_sam2_result(run_dir, config, propagation_result)
    else:
        incoming_dir = run_dir / "incoming_sam2"
        if incoming_dir.exists():
            raise FileExistsError(
                f"incoming SAM2 directory exists without a frozen result: {incoming_dir}"
            )
        safe_extract_tar(sam2_archive, incoming_dir)
        source_result = incoming_dir / "result"
        _assert_sam2_result(run_dir, config, source_result)
        propagation_root.mkdir()
        shutil.copytree(source_result, propagation_result)

    runtime_config_path = run_dir / "runtime_config.yaml"
    logs = run_dir / "logs"
    run_checked(
        [
            sys.executable,
            "scripts/run_sam2_video_geometry_stability.py",
            "--project-root",
            str(root),
            "--pilot-config",
            str(runtime_config_path),
            "--config-key",
            RUNTIME_CONFIG_KEY,
            "--propagation-root",
            str(propagation_root),
            "--ground-dem",
            str(
                run_dir
                / "s2"
                / "data"
                / "dem"
                / "ground_dem_interpolated.npy"
            ),
            "--sensors-config",
            str(resolve_repo_path(root, config["prediction_configs"]["sensors"])),
            "--mapping-config",
            str(
                resolve_repo_path(
                    root, config["prediction_configs"]["mask_to_dem_mapping"]
                )
            ),
            "--gate-config",
            str(
                resolve_repo_path(
                    root, config["prediction_configs"]["geometry_quality_gate"]
                )
            ),
            "--output-root",
            str(run_dir / "s4_geometry"),
        ],
        root,
        logs / "s4_geometry.log",
    )
    anchor, candidate_gate = evaluate_anchor_candidate_gate(root, run_dir, config)
    assert_expected_outcome(config, candidate_gate)
    if not config["expected_outcome"]["agent_should_run"]:
        result = summarize_gate_blocked_run(
            run_dir, config, anchor, candidate_gate
        )
        write_json(run_dir / "candidate_gate_decision.json", candidate_gate)
        write_json(run_dir / "completion_summary.json", result)
        return result

    run_checked(
        [
            sys.executable,
            "scripts/prepare_phase2d_c11_direct_runtime.py",
            "--config",
            str(runtime_config_path),
            "--project-root",
            str(root),
            "--runtime-root",
            str(run_dir / "runtime"),
        ],
        root,
        logs / "prepare_standard_runtime.log",
    )
    run_checked(
        [
            sys.executable,
            "run_offline_pipeline.py",
            "--stage",
            "agent_pipeline",
            "--config",
            str(resolve_repo_path(root, config["standard_pipeline"]["agent_config"])),
            "--project-root",
            str(run_dir / "runtime"),
            "--skip-figures",
        ],
        root,
        logs / "standard_agent_pipeline.log",
    )
    result = summarize_completed_run(run_dir, config)
    write_json(run_dir / "completion_summary.json", result)
    if result["status"] != "success":
        raise RuntimeError(f"one-click pipeline completed with failed summary: {result}")
    return result
