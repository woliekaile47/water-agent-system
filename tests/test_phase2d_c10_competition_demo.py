from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.integration.competition_demo_profile import (
    CompetitionDemoProfileError,
    build_competition_demo_snapshot,
)


def _write(path: Path, content: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _profile(root: Path, *, input_image: str | None = None) -> Path:
    sample = "c8c3_test"
    paths = {
        "input_image": input_image or "data/simulation_dynamic/case/moderate/seed_303/frames/frame_000149.png",
        "predicted_mask": f"outputs/phase2d_c8_seed303_video_freeze/{sample}/masks_png/frame_000149.png",
        "reprojected_mask": f"outputs/phase2d_c8_seed303_geometry_freeze/{sample}/anchor_reprojected_camera_mask.png",
        "geometry_rows": f"outputs/phase2d_c8_seed303_geometry_freeze/{sample}/per_frame_geometry_summary.json",
        "candidate_gate_rows": f"outputs/phase2d_c8_seed303_candidate_gate_freeze/{sample}/candidate_gate_per_frame.json",
    }
    plots = {
        name: f"outputs/phase2d_c8_seed303_geometry_freeze/{sample}/{name}.png"
        for name in ("water_level", "max_depth", "area", "volume")
    }
    for key, value in paths.items():
        if key == "geometry_rows":
            _write(
                root / value,
                json.dumps(
                    [
                        {
                            "frame_index": 149,
                            "estimated_water_level_m": -0.2,
                            "mean_depth_cm": 10.0,
                            "max_depth_cm": 20.0,
                            "water_area_m2": 4.0,
                            "water_volume_m3": 0.4,
                            "camera_reprojection_iou": 0.95,
                            "outer_boundary_reprojection_p95_px": 4.0,
                            "ground_truth_used": False,
                        }
                    ]
                ).encode(),
            )
        elif key == "candidate_gate_rows":
            _write(
                root / value,
                json.dumps(
                    [
                        {
                            "frame_index": 149,
                            "camera_visible_status": "pass",
                            "global_scene_status": "complete",
                            "result_semantics": "global_scene_estimate",
                            "ground_truth_used": False,
                        }
                    ]
                ).encode(),
            )
        else:
            _write(root / value)
    for value in plots.values():
        _write(root / value)

    config = {
        "phase2d_c10_competition_demo": {
            "protocol_version": "test",
            "demo_mode": "offline_simulation_road_only",
            "source_type": "gazebo_dynamic_rain_road_simulation",
            "source_seed": 303,
            "source_policy": {
                "simulation_only": True,
                "ground_truth_used_by_demo_builder": False,
                "manual_prompt_inputs_allowed": False,
                "dormitory_or_cardboard_inputs_allowed": False,
                "real_devices_started": False,
                "authoritative": False,
                "eligible_for_downstream": False,
            },
            "cases": [
                {
                    "sample_id": sample,
                    "case_id": "sim_water_10cm_001",
                    "rain_level": "moderate",
                    "seed": 303,
                    "anchor_frame_index": 149,
                    "nominal_depth_cm_display_only": 10,
                    **paths,
                    "plots": plots,
                }
            ],
        }
    }
    config_path = root / "configs/demo.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_builds_simulation_only_snapshot(tmp_path: Path) -> None:
    snapshot = build_competition_demo_snapshot(tmp_path, _profile(tmp_path))
    assert snapshot["demo_mode"] == "offline_simulation_road_only"
    assert snapshot["case_count"] == 1
    assert snapshot["cases"][0]["quality"]["camera_visible_status"] == "pass"
    assert snapshot["ground_truth_used"] is False
    assert snapshot["authoritative"] is False
    assert snapshot["eligible_for_downstream"] is False


@pytest.mark.parametrize("fragment", ["manual_prompt", "cardboard", "dormitory", "water_test"])
def test_rejects_non_competition_input_fragments(tmp_path: Path, fragment: str) -> None:
    path = f"data/simulation_dynamic/case/{fragment}/frame_000149.png"
    config = _profile(tmp_path, input_image=path)
    with pytest.raises(CompetitionDemoProfileError, match="forbidden fragment"):
        build_competition_demo_snapshot(tmp_path, config)


def test_rejects_ground_truth_path(tmp_path: Path) -> None:
    config = _profile(
        tmp_path,
        input_image="data/simulation_dynamic/case/ground_truth/camera_mask.png",
    )
    with pytest.raises(CompetitionDemoProfileError, match="ground_truth"):
        build_competition_demo_snapshot(tmp_path, config)


def test_rejects_prediction_artifact_that_reports_gt_use(tmp_path: Path) -> None:
    config = _profile(tmp_path)
    geometry = next(tmp_path.glob("outputs/**/per_frame_geometry_summary.json"))
    rows = json.loads(geometry.read_text())
    rows[0]["ground_truth_used"] = True
    geometry.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(CompetitionDemoProfileError, match="Ground Truth use"):
        build_competition_demo_snapshot(tmp_path, config)
