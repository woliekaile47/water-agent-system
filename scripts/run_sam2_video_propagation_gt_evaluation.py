#!/usr/bin/env python3
"""Independently evaluate frozen SAM 2 video masks against Camera-only GT."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_sam2_video_propagation_gt import (  # noqa: E402
    evaluate_frozen_frame,
    metric_summary,
    summarize_sequence,
    verify_frozen_video_sample,
)
from src.evaluation.evaluate_temporal_sam2_mask_gt import (  # noqa: E402
    load_camera_mask_ground_truth,
)
from src.fusion.sam2_shoreline_geometry_adapter import outer_boundary_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--config-key", default="phase2d_c7_video_pilot")
    parser.add_argument("--propagation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_comparison(predicted: np.ndarray, truth: np.ndarray, path: Path) -> None:
    image = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    image[predicted & truth] = (40, 190, 80)
    image[predicted & ~truth] = (225, 70, 55)
    image[~predicted & truth] = (50, 120, 235)
    pred_outer = outer_boundary_mask(predicted)
    truth_outer = outer_boundary_mask(truth)
    image[pred_outer] = (255, 210, 20)
    image[truth_outer] = (20, 210, 255)
    image[pred_outer & truth_outer] = (255, 255, 255)
    Image.fromarray(image, mode="RGB").save(path)


_COLORS = [(35, 115, 220), (225, 85, 55), (45, 165, 95), (145, 85, 200)]


def _save_chart(
    series: list[tuple[str, list[float], list[float]]],
    path: Path,
    title: str,
    y_label: str,
    connect: bool = True,
    y_bounds: tuple[float, float] | None = None,
    vertical_x: float | None = None,
) -> None:
    """Render a deterministic diagnostic chart without binary plotting dependencies."""
    width, height = 1100, 620
    left, right, top, bottom = 95, 35, 65, 75
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    all_x = [value for _, xs, _ in series for value in xs]
    all_y = [value for _, _, ys in series for value in ys if np.isfinite(value)]
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_bounds is None:
        y_min, y_max = min(all_y), max(all_y)
        padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min, y_max = y_min - padding, y_max + padding
    else:
        y_min, y_max = y_bounds

    def xy(x_value: float, y_value: float) -> tuple[int, int]:
        x_pixel = left + int((x_value - x_min) / (x_max - x_min) * (width - left - right))
        y_pixel = top + int((y_max - y_value) / (y_max - y_min) * (height - top - bottom))
        return x_pixel, y_pixel

    draw.rectangle((left, top, width - right, height - bottom), outline=(60, 60, 60), width=2)
    for fraction in np.linspace(0.0, 1.0, 6):
        y_value = y_min + fraction * (y_max - y_min)
        _, y_pixel = xy(x_min, y_value)
        draw.line((left, y_pixel, width - right, y_pixel), fill=(225, 225, 225), width=1)
        draw.text((8, y_pixel - 7), f"{y_value:.3f}", fill=(40, 40, 40))
    if vertical_x is not None and x_min <= vertical_x <= x_max:
        x_pixel, _ = xy(vertical_x, y_min)
        draw.line((x_pixel, top, x_pixel, height - bottom), fill=(30, 30, 30), width=1)
    for index, (label, xs, ys) in enumerate(series):
        color = _COLORS[index % len(_COLORS)]
        points = [xy(float(x), float(y)) for x, y in zip(xs, ys)]
        if connect and len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        legend_x = left + index * 260
        draw.rectangle((legend_x, 28, legend_x + 18, 42), fill=color)
        draw.text((legend_x + 25, 27), label, fill=(25, 25, 25))
    draw.text((left, 6), title, fill=(20, 20, 20))
    draw.text((8, top - 25), y_label, fill=(20, 20, 20))
    draw.text((width // 2 - 45, height - 30), "Frame / offset", fill=(20, 20, 20))
    canvas.save(path)


def save_sequence_plots(rows: list[dict[str, Any]], sample_dir: Path) -> None:
    frames = [float(row["frame_index"]) for row in rows]
    anchor = next(float(row["frame_index"]) for row in rows if row["is_anchor_frame"])
    _save_chart(
        [("Camera GT IoU", frames, [float(row["iou"]) for row in rows])],
        sample_dir / "temporal_iou_curve.png",
        "Frozen SAM2 propagation vs independent Camera GT",
        "IoU",
        y_bounds=(0.0, 1.02),
        vertical_x=anchor,
    )
    _save_chart(
        [("Outer boundary P95", frames, [float(row["outer_boundary_p95_px"]) for row in rows])],
        sample_dir / "temporal_outer_boundary_curve.png",
        "Outer shoreline error over time",
        "Pixels",
        vertical_x=anchor,
    )
    _save_chart(
        [
            ("Predicted area", frames, [float(row["predicted_pixels"]) for row in rows]),
            ("GT area", frames, [float(row["gt_pixels"]) for row in rows]),
        ],
        sample_dir / "temporal_area_curve.png",
        "Camera mask area over time",
        "Pixels",
        vertical_x=anchor,
    )


def save_dataset_plots(all_rows: list[dict[str, Any]], output_root: Path) -> None:
    iou_series = []
    boundary_series = []
    for sample_id in sorted({row["sample_id"] for row in all_rows}):
        items = [row for row in all_rows if row["sample_id"] == sample_id]
        distance = [float(int(row["frame_index"]) - int(row["anchor_frame_index"])) for row in items]
        iou_series.append((sample_id, distance, [float(row["iou"]) for row in items]))
        boundary_series.append((sample_id, distance, [float(row["outer_boundary_p95_px"]) for row in items]))
    _save_chart(
        iou_series,
        output_root / "iou_vs_anchor_distance.png",
        "Camera GT IoU vs distance from prompt anchor",
        "IoU",
        y_bounds=(0.0, 1.02),
        vertical_x=0.0,
    )
    _save_chart(
        boundary_series,
        output_root / "outer_boundary_vs_anchor_distance.png",
        "Outer shoreline P95 vs distance from prompt anchor",
        "Pixels",
        vertical_x=0.0,
    )
    stability_series = []
    for sample_id in sorted({row["sample_id"] for row in all_rows}):
        items = [row for row in all_rows if row["sample_id"] == sample_id and row["previous_frame_iou"] is not None]
        stability_series.append((
            sample_id,
            [float(row["previous_frame_iou"]) for row in items],
            [float(row["iou"]) for row in items],
        ))
    _save_chart(
        stability_series,
        output_root / "temporal_stability_vs_accuracy.png",
        "Prediction-side temporal stability vs independent accuracy",
        "Camera GT IoU",
        connect=False,
        y_bounds=(0.0, 1.02),
    )


def _format_metric(value: float | None, digits: int = 4) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def write_report(output_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 2D-C-7-2 独立逐帧 Camera GT 评价",
        "",
        "本报告只评价已冻结的 SAM 2 视频传播结果。所有 RGB 与 mask 哈希均在首次读取 GT 前完成验证；没有重新运行 SAM 2、修改提示或改变 prediction-side gate。",
        "",
        "| 序列 | IoU 中位数 | IoU 最小值 | 外岸线 P95 中位数 | Anchor IoU | 最差帧 | 达到既有离线研究条件 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample_id, item in summary["sequences"].items():
        lines.append(
            f"| {sample_id} | {_format_metric(item['iou']['median'])} | {_format_metric(item['iou']['min'])} | "
            f"{_format_metric(item['outer_boundary_p95_px']['median'], 2)} | {_format_metric(item['anchor']['iou'])} | "
            f"{item['worst_frame']['frame_index']} | {item['offline_research_criteria_met_count']}/{item['frame_count']} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 相邻 mask IoU 高只表示传播结果自身稳定，不能证明语义正确。",
        "- Camera GT 仅用于本独立评价，不进入 prediction、提示生成或 quality gate。",
        "- 本阶段没有评价 DEM、水位、面积或体积，也没有接入 S5-S8。",
        "- 所有结果仍为非 authoritative 的研究候选。",
        "",
    ])
    (output_root / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.pilot_config.expanduser().resolve()
    propagation_root = args.propagation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_root}")
    config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = config_document[args.config_key]
    samples = config["samples"]
    window_start = int(config["window_start"])
    window_end = int(config["window_end"])
    anchor = int(config["anchor_frame_index"])

    # Protocol boundary: verify every frozen input before the first GT loader call.
    frozen = {
        sample["sample_id"]: verify_frozen_video_sample(
            root, propagation_root, sample, window_start, window_end, anchor
        )
        for sample in samples
    }

    output_root.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    sequence_summaries: dict[str, Any] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        verification = frozen[sample_id]
        gt = load_camera_mask_ground_truth(
            root,
            sample["case_id"],
            sample["rain_level"],
            int(sample["seed"]),
            anchor,
        )
        truth = np.asarray(gt["camera_mask"], dtype=bool)
        sample_dir = output_root / sample_id
        sample_dir.mkdir()
        rows: list[dict[str, Any]] = []
        for frame in verification["frames"]:
            predicted = np.load(frame["mask_path"], allow_pickle=False).astype(bool)
            if predicted.shape != truth.shape:
                raise ValueError(f"Camera-mask shape mismatch for {sample_id} frame {frame['frame_index']}")
            result = evaluate_frozen_frame(predicted, truth)
            camera = result["camera_mask_metrics"]
            boundary = result["camera_boundary_metrics"]
            row = {
                "sample_id": sample_id,
                "case_id": sample["case_id"],
                "rain_level": sample["rain_level"],
                "seed": int(sample["seed"]),
                "frame_index": int(frame["frame_index"]),
                "anchor_frame_index": anchor,
                "is_anchor_frame": bool(frame["is_anchor_frame"]),
                "previous_frame_iou": frame["previous_frame_iou"],
                "predicted_pixels": int(camera["predicted_pixels"]),
                "gt_pixels": int(camera["gt_pixels"]),
                "intersection_pixels": int(camera["intersection_pixels"]),
                "union_pixels": int(camera["union_pixels"]),
                "iou": float(camera["iou"]),
                "precision": float(camera["precision"]),
                "recall": float(camera["recall"]),
                "f1": float(camera["f1"]),
                "false_positive_pixels": int(camera["false_positive_pixels"]),
                "false_negative_pixels": int(camera["false_negative_pixels"]),
                "area_absolute_error_pixels": int(abs(camera["predicted_pixels"] - camera["gt_pixels"])),
                "area_relative_error": float(abs(camera["predicted_pixels"] - camera["gt_pixels"]) / camera["gt_pixels"]),
                "outer_boundary_p50_px": float(boundary["symmetric_outer_boundary"]["p50_px"]),
                "outer_boundary_p95_px": float(boundary["symmetric_outer_boundary"]["p95_px"]),
                "predicted_to_gt_outer_p95_px": float(boundary["predicted_shoreline_to_gt_boundary"]["p95_px"]),
                "gt_to_predicted_outer_p95_px": float(boundary["gt_boundary_to_predicted_shoreline"]["p95_px"]),
                "offline_research_criteria_met": bool(result["offline_research_criteria"]["all_met"]),
                "ground_truth_used_for_prediction": False,
                "eligible_for_downstream": False,
            }
            rows.append(row)
            all_rows.append(row)
        sequence_summary = summarize_sequence(rows, anchor)
        sequence_summary.update({
            "sample_id": sample_id,
            "role": sample["role"],
            "case_id": sample["case_id"],
            "rain_level": sample["rain_level"],
            "seed": int(sample["seed"]),
            "offline_research_criteria_met_count": sum(row["offline_research_criteria_met"] for row in rows),
            "frozen_input_verification": verification,
            "ground_truth_validation": gt["validation"],
        })
        sequence_summaries[sample_id] = sequence_summary
        write_json(sample_dir / "per_frame_evaluation.json", rows)
        write_json(sample_dir / "sequence_summary.json", sequence_summary)
        with (sample_dir / "per_frame_evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_sequence_plots(rows, sample_dir)
        for frame_index in (window_start, anchor, window_end):
            predicted = np.load(
                propagation_root / sample_id / "masks_npy" / f"frame_{frame_index:06d}.npy",
                allow_pickle=False,
            ).astype(bool)
            save_comparison(predicted, truth, sample_dir / f"frame_{frame_index:06d}_mask_vs_gt.png")

    overall = {
        "iou": metric_summary([row["iou"] for row in all_rows]),
        "precision": metric_summary([row["precision"] for row in all_rows]),
        "recall": metric_summary([row["recall"] for row in all_rows]),
        "f1": metric_summary([row["f1"] for row in all_rows]),
        "outer_boundary_p95_px": metric_summary([row["outer_boundary_p95_px"] for row in all_rows]),
    }
    summary = {
        "protocol_version": "phase2d_c7_video_propagation_camera_gt_evaluation_v1",
        "sample_count": len(samples),
        "evaluated_frame_count": len(all_rows),
        "window_start": window_start,
        "window_end": window_end,
        "anchor_frame_index": anchor,
        "all_frozen_inputs_verified_before_first_ground_truth_read": True,
        "ground_truth_scope": "camera_water_mask_only",
        "sam2_rerun_count": 0,
        "prediction_recomputed_count": 0,
        "prompt_modified": False,
        "prediction_side_gate_modified": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "overall": overall,
        "offline_research_criteria_met_frame_count": sum(row["offline_research_criteria_met"] for row in all_rows),
        "sequences": sequence_summaries,
    }
    with (output_root / "per_frame_evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output_root / "per_frame_evaluation.json", all_rows)
    write_json(output_root / "evaluation_summary.json", summary)
    save_dataset_plots(all_rows, output_root)
    write_report(output_root, summary)
    (output_root / "run_log.txt").write_text(
        "Evaluation-only run completed. Frozen hashes were verified before GT access.\n"
        "SAM2 rerun: false\nPrediction recomputed: false\nGT scope: Camera mask only\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evaluated_frame_count": len(all_rows),
        "overall_iou": overall["iou"],
        "criteria_met": summary["offline_research_criteria_met_frame_count"],
        "output_root": str(output_root),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
