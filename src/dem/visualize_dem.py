#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEM visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_dem_heatmap(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str | Path,
    metadata: dict[str, Any],
    title: str = "S2 Dry Baseline DEM",
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"缺少 matplotlib，无法生成 DEM 热力图: {exc}") from exc

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    roi = metadata["dem_roi"]
    masked_dem = np.ma.masked_where(~valid_mask, dem)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    image = ax.imshow(
        masked_dem,
        origin="lower",
        extent=[roi["x_min"], roi["x_max"], roi["y_min"], roi["y_max"]],
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("z median elevation (m)")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
