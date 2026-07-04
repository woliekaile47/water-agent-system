#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3: create a manual polygon water-candidate mask."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.vision.visualize_mask import save_mask_overlay


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"缺少 PyYAML，无法读取配置: {exc}") from exc
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "manual_mask" not in data:
        raise ValueError("配置文件缺少 manual_mask 字段")
    return data["manual_mask"]


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def create_polygon_mask(width: int, height: int, polygon_points: list[list[int]]) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法创建 polygon mask: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸异常: width={width}, height={height}")
    if len(polygon_points) < 3:
        raise ValueError("polygon_points 至少需要 3 个点")

    points = np.asarray(polygon_points, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("polygon_points 格式应为 [[x, y], ...]")
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        print("[S3][manual_mask][WARN] polygon_points 存在超出图像范围的点，请人工检查 configs/manual_mask_config.yaml")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def save_png(path: Path, array: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法保存 PNG: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {path}")


def create_manual_mask(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_config(config_path)
    source_image = resolve_project_path(root, config["source_image"])
    if not source_image.exists():
        raise FileNotFoundError(
            f"source_image 不存在: {source_image}。请先运行 --stage extract_camera，"
            "或调整 configs/manual_mask_config.yaml。"
        )

    width = int(config["image_width"])
    height = int(config["image_height"])
    mask_type = str(config.get("mask_type", "polygon"))
    if mask_type != "polygon":
        raise ValueError(f"当前 S3 MVP 只支持 polygon mask，不支持: {mask_type}")

    mask = create_polygon_mask(width, height, config["polygon_points"])
    mask_pixel_count = int(np.count_nonzero(mask))
    mask_area_ratio = float(mask_pixel_count / float(width * height))
    if mask_pixel_count <= 0:
        print("[S3][manual_mask][WARN] mask 像素数为 0，请人工调整 configs/manual_mask_config.yaml")
    if mask_area_ratio > 0.8:
        print("[S3][manual_mask][WARN] mask 面积比例过大，请人工确认 polygon_points 是否合适")

    mask_dir = root / "data" / "masks"
    fig_dir = root / "outputs" / "figures"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    output_mask_png = mask_dir / "manual_water_mask.png"
    output_mask_npy = mask_dir / "manual_water_mask.npy"
    output_metadata = mask_dir / "manual_water_mask_metadata.json"
    output_overlay = fig_dir / "manual_water_mask_overlay.png"

    save_png(output_mask_png, mask)
    np.save(output_mask_npy, mask > 0)
    save_mask_overlay(source_image, output_mask_png, output_overlay)

    metadata = {
        "source_image": str(source_image),
        "image_width": width,
        "image_height": height,
        "mask_type": mask_type,
        "polygon_points": config["polygon_points"],
        "description": str(config.get("description", "")),
        "mask_pixel_count": mask_pixel_count,
        "mask_area_ratio": mask_area_ratio,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "S3_manual_water_mask",
        "note": "Manual polygon mask is used as the minimum implementation of image segmentation.",
        "output_files": {
            "manual_water_mask_png": str(output_mask_png),
            "manual_water_mask_npy": str(output_mask_npy),
            "manual_water_mask_metadata": str(output_metadata),
            "manual_water_mask_overlay": str(output_overlay),
        },
    }
    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[S3][manual_mask] mask pixel count: {mask_pixel_count}")
    print(f"[S3][manual_mask] mask area ratio: {mask_area_ratio:.6f}")
    print("[S3][manual_mask] output files:")
    for path in metadata["output_files"].values():
        print(f"  - {path}")
    print("[S3][manual_mask] 如果 polygon_points 不合适，请人工调整 configs/manual_mask_config.yaml")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create S3 manual polygon water mask.")
    parser.add_argument("--config", required=True, help="Path to configs/manual_mask_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    create_manual_mask(args.config, args.project_root)


if __name__ == "__main__":
    main()
