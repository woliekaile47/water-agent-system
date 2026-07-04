#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization helpers for S3 image masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法读取图像: {exc}") from exc
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"读取图像失败: {image_path}")
    return image


def read_mask(path: str | Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法读取 mask: {exc}") from exc
    mask_path = Path(path).expanduser()
    if not mask_path.exists():
        raise FileNotFoundError(f"mask 不存在: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"读取 mask 失败: {mask_path}")
    return mask


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法生成 overlay: {exc}") from exc
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"图像和 mask 尺寸不一致: image={image.shape[:2]}, mask={mask.shape[:2]}")
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 2] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0)[selected]

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def save_mask_overlay(
    source_image: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法保存 overlay: {exc}") from exc
    image = read_image(source_image)
    mask = read_mask(mask_path)
    overlay = create_overlay(image, mask, alpha=alpha)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), overlay)
    if not ok:
        raise RuntimeError(f"保存 overlay 失败: {output}")
    print(f"[S3][visualize_mask] overlay saved: {output}")
    return output
