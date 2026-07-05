#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline entrypoint for water_agent_system.

Current stages:
- dem: S2-A dry scene-height DEM construction
- ground_dem: S2-B dry ground DEM construction
- extract_camera: S3 offline camera frame extraction
- manual_mask: S3 manual polygon water mask
- mask_to_dem: S4 region-level mask-to-DEM mapping
- water_depth: S4 configured-depth water depth inversion
- build_surface_dem: S4-real surface DEM from offline LiDAR rosbag
- surface_depth: S4-real depth from surface DEM minus ground DEM
- surface_depth_eval: S4-real accuracy evaluation against known simulated depth
- area_volume: S5 water area and volume calculation
- weather_correction: S6 offline mock weather correction
- deterministic_forecast: S7-A deterministic rule-engine forecast
- case_retrieval: S7-B offline mock case retrieval correction
- physical_constraint: S7-C simplified physical constraint check
- warning_report: S8 warning decision, report, audit, and summary
- agent_pipeline: Agent MVP orchestrating S4-S8 with SQLite audit records
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dem.build_dem import build_dem_from_bag
from src.dem.build_ground_dem import build_ground_dem_from_bag
from src.dem.build_surface_dem_from_rosbag import build_surface_dem
from src.evaluation.evaluate_surface_depth_accuracy import evaluate_surface_depth_accuracy
from src.fusion.map_mask_to_dem import map_mask_to_dem
from src.hydrology.calculate_area_volume import calculate_area_volume
from src.hydrology.invert_water_depth import invert_water_depth
from src.hydrology.invert_surface_depth import invert_surface_depth
from src.hydrology.visualize_area_volume_summary import visualize_area_volume_summary
from src.meteorology.compute_weather_correction import compute_weather_correction
from src.meteorology.visualize_weather_correction import visualize_weather_correction
from src.reasoning.deterministic_forecast import deterministic_forecast
from src.reasoning.visualize_forecast import visualize_forecast
from src.reasoning.case_retrieval_correction import case_retrieval_correction
from src.reasoning.visualize_case_retrieval import visualize_case_retrieval
from src.reasoning.physical_constraint_check import physical_constraint_check
from src.reasoning.visualize_physical_constraint import visualize_physical_constraint
from src.vision.create_manual_mask import create_manual_mask
from src.vision.extract_camera_frame import extract_camera_frame
from src.warning.generate_warning_decision import generate_warning_decision
from src.warning.generate_warning_report import generate_warning_report
from src.warning.visualize_warning_summary import visualize_warning_summary
from src.warning.write_audit_log import write_audit_log
from src.agent.pipeline_agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline water_agent_system pipeline.")
    parser.add_argument(
        "--stage",
        default="dem",
        choices=[
            "dem",
            "ground_dem",
            "extract_camera",
            "manual_mask",
            "mask_to_dem",
            "water_depth",
            "build_surface_dem",
            "surface_depth",
            "surface_depth_eval",
            "area_volume",
            "weather_correction",
            "deterministic_forecast",
            "case_retrieval",
            "physical_constraint",
            "warning_report",
            "agent_pipeline",
        ],
        help="Offline stage to run. Use ground_dem for S2-B ground baseline.",
    )
    parser.add_argument("--dry_bag", help="Path to dry_baseline_001 rosbag")
    parser.add_argument("--bag", help="Generic rosbag path, used by extract_camera stage")
    parser.add_argument("--config", default="configs/system_config.yaml", help="System config YAML path")
    parser.add_argument("--output", help="Output path, used by extract_camera stage")
    parser.add_argument("--frame-index", type=int, default=20, help="1-based camera frame index for extract_camera")
    parser.add_argument("--case", help="Case name for surface DEM stages")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print("[pipeline] water_agent_system offline pipeline")
    print(f"[pipeline] current stage: {args.stage}")
    print(f"[pipeline] config={config_path}")
    if args.stage == "ground_dem":
        if not args.dry_bag:
            parser.error("--stage ground_dem requires --dry_bag")
        print(f"[pipeline] dry_bag={Path(args.dry_bag).expanduser()}")
        metadata = build_ground_dem_from_bag(args.dry_bag, config_path, PROJECT_ROOT)
        print("[pipeline] S2-B ground DEM complete")
        print(f"[pipeline] ground DEM shape: {metadata['dem_shape']}")
        print(f"[pipeline] valid cell count: {metadata['valid_cell_count']}")
        print(f"[pipeline] valid ratio: {metadata['valid_ratio']:.4f}")
    elif args.stage == "dem":
        if not args.dry_bag:
            parser.error("--stage dem requires --dry_bag")
        print(f"[pipeline] dry_bag={Path(args.dry_bag).expanduser()}")
        metadata = build_dem_from_bag(args.dry_bag, config_path, PROJECT_ROOT)
        print("[pipeline] S2-A DEM complete")
        print(f"[pipeline] DEM shape: {metadata['dem_shape']}")
        print(f"[pipeline] valid cell count: {metadata['valid_cell_count']}")
        print(f"[pipeline] DEM grid size: {metadata['grid_size']}")
        print(
            "[pipeline] z_min / z_max / z_median: "
            f"{metadata['z_min']:.4f} / {metadata['z_max']:.4f} / {metadata['z_median']:.4f}"
        )
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "extract_camera":
        if not args.bag:
            parser.error("--stage extract_camera requires --bag")
        if not args.output:
            parser.error("--stage extract_camera requires --output")
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        metadata = extract_camera_frame(args.bag, output_path, args.frame_index)
        print("[pipeline] S3 extract_camera complete")
        print(f"[pipeline] output file: {metadata['output']}")
        return
    elif args.stage == "manual_mask":
        metadata = create_manual_mask(config_path, PROJECT_ROOT)
        print("[pipeline] S3 manual_mask complete")
        print(f"[pipeline] mask pixel count: {metadata['mask_pixel_count']}")
        print(f"[pipeline] mask area ratio: {metadata['mask_area_ratio']:.6f}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "mask_to_dem":
        metadata = map_mask_to_dem(config_path, PROJECT_ROOT)
        print("[pipeline] S4 mask_to_dem complete")
        print(f"[pipeline] water region cell count: {metadata['water_region_cell_count']}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "water_depth":
        metadata = invert_water_depth(config_path, PROJECT_ROOT)
        print("[pipeline] S4 water_depth complete")
        print(f"[pipeline] configured depth cm: {metadata['configured_depth_cm']:.2f}")
        print(f"[pipeline] max depth cm: {metadata['max_depth_cm']:.2f}")
        print(f"[pipeline] mean depth cm: {metadata['mean_depth_cm']:.2f}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "build_surface_dem":
        metadata = build_surface_dem(config_path, PROJECT_ROOT, args.case)
        print("[pipeline] S4-real build_surface_dem complete")
        print(f"[pipeline] case_name: {metadata['case_name']}")
        print(f"[pipeline] rosbag_path: {metadata['rosbag_path']}")
        print(f"[pipeline] frames_read: {metadata['frames_read']}")
        print(f"[pipeline] points_after_filter: {metadata['points_after_filter']}")
        print(f"[pipeline] valid_surface_cell_count: {metadata['valid_surface_cell_count']}")
        print(
            "[pipeline] surface_valid_ratio_in_water_region: "
            f"{metadata['surface_valid_ratio_in_water_region']:.4f}"
        )
        if metadata.get("warning"):
            print(f"[pipeline][WARN] {'; '.join(metadata['warning'])}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "surface_depth":
        metadata = invert_surface_depth(config_path, PROJECT_ROOT, args.case)
        print("[pipeline] S4-real surface_depth complete")
        print(f"[pipeline] case_name: {metadata['case_name']}")
        print(f"[pipeline] valid_depth_cell_count: {metadata['valid_depth_cell_count']}")
        print(
            "[pipeline] valid_depth_ratio_in_water_region: "
            f"{metadata['valid_depth_ratio_in_water_region']:.4f}"
        )
        print(f"[pipeline] mean_depth_cm: {metadata['mean_depth_cm']}")
        print(f"[pipeline] median_depth_cm: {metadata['median_depth_cm']}")
        print(f"[pipeline] max_depth_cm: {metadata['max_depth_cm']}")
        if metadata.get("warning"):
            print(f"[pipeline][WARN] {'; '.join(metadata['warning'])}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "surface_depth_eval":
        metadata = evaluate_surface_depth_accuracy(config_path, PROJECT_ROOT, args.case)
        print("[pipeline] S4-real surface_depth_eval complete")
        print(f"[pipeline] case_name: {metadata['case_name']}")
        print(f"[pipeline] known_depth_cm: {metadata['known_depth_cm']}")
        print(f"[pipeline] mean_depth_cm: {metadata['mean_depth_cm']}")
        print(f"[pipeline] median_depth_cm: {metadata['median_depth_cm']}")
        print(f"[pipeline] mean_error_cm: {metadata['mean_error_cm']}")
        print(f"[pipeline] median_error_cm: {metadata['median_error_cm']}")
        if metadata.get("warning"):
            print(f"[pipeline][WARN] {'; '.join(metadata['warning'])}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        return
    elif args.stage == "area_volume":
        depth_metadata = invert_water_depth(config_path, PROJECT_ROOT)
        metadata = calculate_area_volume(config_path, PROJECT_ROOT)
        summary_files = visualize_area_volume_summary(config_path, PROJECT_ROOT)
        print("[pipeline] S5 area_volume complete")
        print(f"[pipeline] valid depth cell count: {metadata['valid_depth_cell_count']}")
        print(f"[pipeline] water area m2: {metadata['water_area_m2']:.4f}")
        print(f"[pipeline] water volume m3: {metadata['water_volume_m3']:.4f}")
        print(f"[pipeline] water volume liter: {metadata['water_volume_liter']:.2f}")
        print(f"[pipeline] max depth cm: {metadata['max_depth_cm']:.2f}")
        print(f"[pipeline] mean depth cm: {metadata['mean_depth_cm']:.2f}")
        print(f"[pipeline] configured depth cm: {depth_metadata['configured_depth_cm']:.2f}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in summary_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "weather_correction":
        metadata = compute_weather_correction(config_path, PROJECT_ROOT)
        summary_files = visualize_weather_correction(config_path, PROJECT_ROOT)
        print("[pipeline] S6 weather_correction complete")
        print(
            "[pipeline] current rainfall intensity mm/h: "
            f"{metadata['current_rainfall_intensity_mm_h']:.2f}"
        )
        print(f"[pipeline] rainfall level label: {metadata['rainfall_level_label']}")
        print(f"[pipeline] weather correction factor: {metadata['weather_correction_factor']:.2f}")
        print(
            "[pipeline] forecast rainfall 15/30/60 min: "
            f"{metadata['forecast_rainfall_15min_mm']:.2f} / "
            f"{metadata['forecast_rainfall_30min_mm']:.2f} / "
            f"{metadata['forecast_rainfall_60min_mm']:.2f} mm"
        )
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in summary_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "deterministic_forecast":
        metadata = deterministic_forecast(config_path, PROJECT_ROOT)
        figure_files = visualize_forecast(config_path, PROJECT_ROOT)
        print("[pipeline] S7-A deterministic_forecast complete")
        print(f"[pipeline] current mean depth cm: {metadata['current_mean_depth_cm']:.2f}")
        print(f"[pipeline] k_1min cm/min: {metadata['k_1min_cm_per_min']:.4f}")
        print(f"[pipeline] k_5min cm/min: {metadata['k_5min_cm_per_min']:.4f}")
        print(f"[pipeline] k_10min cm/min: {metadata['k_10min_cm_per_min']:.4f}")
        print(f"[pipeline] k_base cm/min: {metadata['k_base_cm_per_min']:.4f}")
        print(f"[pipeline] weather correction factor: {metadata['weather_correction_factor']:.2f}")
        print(f"[pipeline] k_forecast cm/min: {metadata['k_forecast_cm_per_min']:.4f}")
        for item in metadata["forecast_results"]:
            print(
                "[pipeline] "
                f"{item['horizon_min']} min forecast depth cm: {item['forecast_depth_cm']:.2f}, "
                f"warning level: {item['warning_level']}"
            )
        times = metadata["time_to_thresholds_min"]
        print(
            "[pipeline] time to blue/yellow/orange threshold min: "
            f"{times['blue']} / {times['yellow']} / {times['orange']}"
        )
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in figure_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "case_retrieval":
        metadata = case_retrieval_correction(config_path, PROJECT_ROOT)
        figure_files = visualize_case_retrieval(config_path, PROJECT_ROOT)
        print("[pipeline] S7-B case_retrieval complete")
        print("[pipeline] top_k retrieved case ids:")
        for case_id, score in zip(metadata["top_case_ids"], metadata["top_case_similarity_scores"]):
            print(f"[pipeline]   {case_id}: similarity={score:.4f}")
        print("[pipeline] median bias for 5/15/30/60 min:")
        for horizon in metadata["forecast_horizons_min"]:
            print(f"[pipeline]   {horizon} min: {metadata['median_bias_cm_by_horizon'][str(horizon)]:.2f} cm")
        print("[pipeline] deterministic vs corrected forecast depths:")
        for item in metadata["corrected_forecast_results"]:
            print(
                "[pipeline] "
                f"{item['horizon_min']} min: deterministic={item['deterministic_forecast_depth_cm']:.2f} cm, "
                f"corrected={item['corrected_forecast_depth_cm']:.2f} cm, "
                f"warning={item['warning_level']}"
            )
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in figure_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "physical_constraint":
        metadata = physical_constraint_check(config_path, PROJECT_ROOT)
        figure_files = visualize_physical_constraint(config_path, PROJECT_ROOT)
        print("[pipeline] S7-C physical_constraint complete")
        for item in metadata["final_forecast_results"]:
            print(
                "[pipeline] "
                f"{item['horizon_min']} min: final={item['final_forecast_depth_cm']:.2f} cm, "
                f"source_corrected={item['source_corrected_depth_cm']:.2f} cm, "
                f"check={item['physical_check']}, "
                f"confidence={item['physical_confidence']}, "
                f"warning={item['warning_level']}"
            )
        print(f"[pipeline] overall warning level: {metadata['overall_warning_level']}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in figure_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "warning_report":
        metadata = generate_warning_decision(config_path, PROJECT_ROOT)
        report_files = generate_warning_report(config_path, PROJECT_ROOT)
        audit_files = write_audit_log(config_path, PROJECT_ROOT)
        figure_files = visualize_warning_summary(config_path, PROJECT_ROOT)
        print("[pipeline] S8 warning_report complete")
        print(f"[pipeline] forecast source: {metadata.get('forecast_source')}")
        print(f"[pipeline] S7 pipeline used: {metadata.get('s7_pipeline_used')}")
        print(f"[pipeline] overall warning level: {metadata['overall_warning_level']}")
        print(f"[pipeline] current mean depth cm: {metadata['current_mean_depth_cm']:.2f}")
        for item in metadata["forecast_warning_results"]:
            confidence_text = item.get("physical_confidence")
            confidence_suffix = f", physical confidence: {confidence_text}" if confidence_text else ""
            print(
                "[pipeline] "
                f"{item['horizon_min']} min warning level: {item['warning_level']} "
                f"({item['forecast_depth_cm']:.2f} cm{confidence_suffix})"
            )
        print(
            "[pipeline] time to orange threshold min: "
            f"{metadata['time_to_thresholds_min'].get('orange')}"
        )
        print(f"[pipeline] action suggestion: {metadata['action_suggestion']}")
        print("[pipeline] output file paths:")
        for path in metadata["output_files"].values():
            print(f"  - {path}")
        for path in report_files.values():
            print(f"  - {path}")
        for path in audit_files.values():
            print(f"  - {path}")
        for path in figure_files.values():
            print(f"  - {path}")
        return
    elif args.stage == "agent_pipeline":
        metadata = run_agent(config_path, PROJECT_ROOT)
        print("[pipeline] Agent MVP complete")
        print(f"[pipeline] run_id: {metadata['run_id']}")
        print(f"[pipeline] status: {metadata['status']}")
        for stage in metadata["stages"]:
            print(f"[pipeline] stage {stage['stage_name']}: {stage['status']}")
        print(f"[pipeline] overall warning level: {metadata['overall_warning_level']}")
        print(f"[pipeline] current mean depth cm: {metadata['current_mean_depth_cm']}")
        print(f"[pipeline] water area m2: {metadata['water_area_m2']}")
        print(f"[pipeline] water volume m3: {metadata['water_volume_m3']}")
        print(f"[pipeline] rainfall intensity: {metadata['rainfall_intensity_mm_h']}")
        print(f"[pipeline] weather correction factor: {metadata['weather_correction_factor']}")
        print(f"[pipeline] forecast source: {metadata.get('forecast_source')}")
        print(
            "[pipeline] final forecast 5/15/30/60 min cm: "
            f"{metadata.get('final_forecast_5min_cm')} / "
            f"{metadata.get('final_forecast_15min_cm')} / "
            f"{metadata.get('final_forecast_30min_cm')} / "
            f"{metadata.get('final_forecast_60min_cm')}"
        )
        print(f"[pipeline] physical confidence summary: {metadata.get('physical_confidence_summary')}")
        print(f"[pipeline] sqlite db path: {metadata['sqlite_db_path']}")
        return

    print(f"[pipeline] DEM grid size: {metadata['grid_size']}")
    print(
        "[pipeline] z_min / z_max / z_median: "
        f"{metadata['z_min']:.4f} / {metadata['z_max']:.4f} / {metadata['z_median']:.4f}"
    )
    print("[pipeline] output file paths:")
    for path in metadata["output_files"].values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
