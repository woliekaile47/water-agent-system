#!/usr/bin/env python3
"""Run one frozen SAM 2 prompt through a deterministic video-frame window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_plan(start: int, end: int, anchor: int) -> list[dict[str, int]]:
    if start < 0 or end < start or not start <= anchor <= end:
        raise ValueError("window must satisfy 0 <= start <= anchor <= end")
    return [
        {"original_frame_index": index, "local_frame_index": index - start}
        for index in range(start, end + 1)
    ]


def consecutive_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes differ")
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def prepare_jpeg_window(
    source_frames: Path,
    output_frames: Path,
    plan: list[dict[str, int]],
) -> list[dict[str, Any]]:
    output_frames.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    expected_size: tuple[int, int] | None = None
    for item in plan:
        original = item["original_frame_index"]
        local = item["local_frame_index"]
        source = source_frames / f"frame_{original:06d}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        if expected_size is None:
            expected_size = image.size
        elif image.size != expected_size:
            raise ValueError("source frame dimensions are not constant")
        target = output_frames / f"{local:05d}.jpg"
        image.save(target, format="JPEG", quality=95, subsampling=0, optimize=False)
        records.append({
            **item,
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "jpeg_path": str(target.resolve()),
            "jpeg_sha256": sha256_file(target),
            "width": int(image.width),
            "height": int(image.height),
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--window-start", required=True, type=int)
    parser.add_argument("--window-end", required=True, type=int)
    parser.add_argument("--anchor-frame-index", required=True, type=int)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir.expanduser().resolve()
    prompt_path = args.prompt_config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen video output: {output_dir}")
    if not frames_dir.is_dir() or not prompt_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("frames, prompt, or checkpoint input is missing")
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    if prompt.get("prompt_quality_status") == "reject":
        raise ValueError("rejected prompt cannot start video propagation")
    if prompt.get("ground_truth_used") is not False:
        raise ValueError("prompt provenance is not GT-free")
    plan = frame_plan(args.window_start, args.window_end, args.anchor_frame_index)
    anchor_source = frames_dir / f"frame_{args.anchor_frame_index:06d}.png"
    if sha256_file(anchor_source) != prompt["image_sha256"]:
        raise ValueError("anchor RGB SHA-256 differs from frozen prompt")

    output_dir.mkdir(parents=True)
    jpeg_dir = output_dir / "jpeg_frames"
    masks_npy = output_dir / "masks_npy"
    masks_png = output_dir / "masks_png"
    masks_npy.mkdir()
    masks_png.mkdir()
    frame_records = prepare_jpeg_window(frames_dir, jpeg_dir, plan)
    started = time.perf_counter()
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        predictor = build_sam2_video_predictor(
            args.model_config,
            str(checkpoint),
            device=args.device,
        )
        inference_state = predictor.init_state(
            video_path=str(jpeg_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=False,
        )
        positive = np.asarray(prompt["positive_points_xy"], dtype=np.float32)
        negative = np.asarray(prompt["negative_points_xy"], dtype=np.float32)
        points = np.concatenate((positive, negative), axis=0)
        labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative)))).astype(np.int32)
        box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
        anchor_local = args.anchor_frame_index - args.window_start
        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=anchor_local,
                obj_id=1,
                points=points,
                labels=labels,
                box=box,
            )
            masks: dict[int, np.ndarray] = {}
            forward_count = args.window_end - args.anchor_frame_index
            for local_index, object_ids, logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=anchor_local,
                max_frame_num_to_track=forward_count,
                reverse=False,
            ):
                if list(object_ids) != [1]:
                    raise RuntimeError("unexpected object IDs from video predictor")
                masks[int(local_index)] = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
            reverse_count = args.anchor_frame_index - args.window_start
            for local_index, object_ids, logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=anchor_local,
                max_frame_num_to_track=reverse_count,
                reverse=True,
            ):
                if list(object_ids) != [1]:
                    raise RuntimeError("unexpected object IDs from video predictor")
                masks[int(local_index)] = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
        if set(masks) != {item["local_frame_index"] for item in plan}:
            raise RuntimeError("video propagation did not return the complete frozen window")

        rows: list[dict[str, Any]] = []
        ordered_masks = []
        for item in plan:
            local = item["local_frame_index"]
            original = item["original_frame_index"]
            mask = masks[local]
            npy_path = masks_npy / f"frame_{original:06d}.npy"
            png_path = masks_png / f"frame_{original:06d}.png"
            np.save(npy_path, mask)
            _save_mask(png_path, mask)
            ordered_masks.append(mask)
            rows.append({
                "original_frame_index": original,
                "local_frame_index": local,
                "mask_area_pixels": int(np.count_nonzero(mask)),
                "mask_area_ratio": float(np.mean(mask)),
                "mask_sha256": sha256_file(npy_path),
                "is_anchor_frame": original == args.anchor_frame_index,
            })
        for index, row in enumerate(rows):
            row["previous_frame_iou"] = None if index == 0 else consecutive_iou(ordered_masks[index - 1], ordered_masks[index])
        with (output_dir / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        areas = np.asarray([row["mask_area_pixels"] for row in rows], dtype=np.float64)
        adjacent = np.asarray([row["previous_frame_iou"] for row in rows[1:]], dtype=np.float64)
        elapsed = float(time.perf_counter() - started)
        summary = {
            "protocol_version": "phase2d_c7_video_propagation_v1",
            "model_config": args.model_config,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": bool(args.offload_state_to_cpu),
            "window_start": args.window_start,
            "window_end": args.window_end,
            "anchor_frame_index": args.anchor_frame_index,
            "frame_count": len(rows),
            "prompt_path": str(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
            "prompt_source": prompt["prompt_source"],
            "prompt_quality_status": prompt["prompt_quality_status"],
            "source_frame_records": frame_records,
            "mask_area_pixels": {
                "minimum": int(np.min(areas)),
                "median": float(np.median(areas)),
                "mean": float(np.mean(areas)),
                "maximum": int(np.max(areas)),
                "coefficient_of_variation": float(np.std(areas) / max(np.mean(areas), 1.0)),
            },
            "adjacent_mask_iou": {
                "minimum": float(np.min(adjacent)),
                "median": float(np.median(adjacent)),
                "mean": float(np.mean(adjacent)),
                "maximum": float(np.max(adjacent)),
            },
            "elapsed_seconds": elapsed,
            "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024**2) if args.device == "cuda" else 0.0,
            "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024**2) if args.device == "cuda" else 0.0,
            "jpeg_conversion": {"quality": 95, "subsampling": 0, "geometry_changed": False},
            "semantic_label": "unknown_candidate",
            "authoritative": False,
            "ground_truth_used": False,
            "eligible_for_downstream": False,
            "sam2_video_propagation_completed": True,
        }
        _write_json(output_dir / "video_propagation_summary.json", summary)
        _write_json(output_dir / "frame_metrics.json", rows)
        print(json.dumps({
            "frame_count": len(rows),
            "adjacent_iou_median": summary["adjacent_mask_iou"]["median"],
            "area_cv": summary["mask_area_pixels"]["coefficient_of_variation"],
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
        }, ensure_ascii=False))
        return 0
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            (output_dir / "cuda_oom.txt").write_text(str(exc) + "\n", encoding="utf-8")
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
