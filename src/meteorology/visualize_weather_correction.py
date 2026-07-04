#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6: Visualize the offline mock weather correction summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Weather correction result JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_weather_correction_summary(result: dict[str, Any], output_path: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate the S6 summary figure: {exc}") from exc

    current_rainfall = float(result.get("current_rainfall_intensity_mm_h", 0.0))
    correction_factor = float(result.get("weather_correction_factor", 0.0))
    forecast_values = [
        float(result.get("forecast_rainfall_15min_mm", 0.0)),
        float(result.get("forecast_rainfall_30min_mm", 0.0)),
        float(result.get("forecast_rainfall_60min_mm", 0.0)),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.35)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.5,
        0.78,
        "S6 Weather Correction - Offline Mock MVP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.5,
        0.52,
        "offline_mock_weather, not real-time meteorological API data",
        ha="center",
        va="center",
        fontsize=10,
        color="#b45309",
        transform=title_ax.transAxes,
    )
    summary_items = [
        ("Current Rainfall Intensity", f"{current_rainfall:.1f} mm/h"),
        ("Rainfall Level", str(result.get("rainfall_level_label", "unknown"))),
        ("Correction Factor", f"{correction_factor:.2f}"),
    ]
    for index, (label, value) in enumerate(summary_items):
        x = 0.18 + index * 0.32
        title_ax.text(x, 0.24, label, ha="center", va="center", fontsize=10, color="#475569")
        title_ax.text(x, 0.06, value, ha="center", va="center", fontsize=16, fontweight="bold", color="#0369a1")

    forecast_ax = fig.add_subplot(gs[1, :])
    labels = ["15 min", "30 min", "60 min"]
    bars = forecast_ax.bar(labels, forecast_values, color=["#38bdf8", "#0ea5e9", "#0369a1"])
    forecast_ax.set_ylabel("Forecast rainfall (mm)")
    forecast_ax.set_title("15/30/60 min Forecast Rainfall")
    forecast_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ymax = max(forecast_values + [1.0]) * 1.25
    forecast_ax.set_ylim(0, ymax)
    for bar, value in zip(bars, forecast_values):
        forecast_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0f172a",
        )

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_weather_correction(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    del config_path
    root = Path(project_root).expanduser().resolve()
    result_json = root / "outputs" / "json" / "weather_correction_result.json"
    output_figure = root / "outputs" / "figures" / "weather_correction_summary.png"
    result = load_json(result_json)
    save_weather_correction_summary(result, output_figure)
    print(f"[S6][weather_summary] output file: {output_figure}")
    return {"weather_correction_summary": str(output_figure)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S6 weather correction summary figure.")
    parser.add_argument("--config", required=True, help="Path to configs/weather_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_weather_correction(args.config, args.project_root)


if __name__ == "__main__":
    main()
