#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-A: Visualize deterministic forecast curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Forecast result JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_forecast_curve(result: dict[str, Any], output_path: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate the S7-A forecast curve: {exc}") from exc

    history_records = sorted(
        result.get("depth_history_records", []),
        key=lambda item: int(item["minutes_before_now"]),
        reverse=True,
    )
    history_x = [-int(item["minutes_before_now"]) for item in history_records if int(item["minutes_before_now"]) > 0]
    history_y = [float(item["mean_depth_cm"]) for item in history_records if int(item["minutes_before_now"]) > 0]
    current_depth = float(result["current_mean_depth_cm"])
    forecast_x = [int(item["horizon_min"]) for item in result["forecast_results"]]
    forecast_y = [float(item["forecast_depth_cm"]) for item in result["forecast_results"]]
    thresholds = result["warning_thresholds_cm"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    if history_x:
        ax.plot(history_x, history_y, marker="o", linewidth=2, color="#64748b", label="mock mean depth history")
    ax.scatter([0], [current_depth], s=90, color="#0f172a", zorder=4, label="current mean depth")
    ax.plot([0] + forecast_x, [current_depth] + forecast_y, marker="o", linewidth=2.5, color="#0284c7", label="forecast")

    ax.axhline(float(thresholds["blue"]), color="#2563eb", linestyle="--", linewidth=1.2, label="blue threshold 15 cm")
    ax.axhline(float(thresholds["yellow"]), color="#ca8a04", linestyle="--", linewidth=1.2, label="yellow threshold 30 cm")
    ax.axhline(float(thresholds["orange"]), color="#ea580c", linestyle="--", linewidth=1.2, label="orange threshold 50 cm")

    for x_value, y_value in zip(forecast_x, forecast_y):
        ax.text(x_value, y_value + 2.0, f"{y_value:.1f}", ha="center", va="bottom", fontsize=9, color="#0f172a")

    ax.set_title("S7-A Deterministic Forecast - MVP")
    ax.text(
        0.5,
        0.96,
        "offline_mock_depth_history + offline_mock_weather, not final real forecast",
        ha="center",
        va="top",
        fontsize=9,
        color="#b45309",
        transform=ax.transAxes,
    )
    ax.set_xlabel("minutes from now")
    ax.set_ylabel("mean water depth (cm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    all_x = history_x + [0] + forecast_x
    all_y = history_y + [current_depth] + forecast_y + [
        float(thresholds["blue"]),
        float(thresholds["yellow"]),
        float(thresholds["orange"]),
    ]
    ax.set_xlim(min(all_x) - 2, max(all_x) + 3)
    ax.set_ylim(0, max(all_y) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_forecast(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    del config_path
    root = Path(project_root).expanduser().resolve()
    result_json = root / "outputs" / "json" / "deterministic_forecast_result.json"
    output_figure = root / "outputs" / "figures" / "deterministic_forecast_curve.png"
    result = load_json(result_json)
    save_forecast_curve(result, output_figure)
    print(f"[S7-A][forecast_curve] output file: {output_figure}")
    return {"deterministic_forecast_curve": str(output_figure)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S7-A deterministic forecast curve.")
    parser.add_argument("--config", required=True, help="Path to configs/prediction_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_forecast(args.config, args.project_root)


if __name__ == "__main__":
    main()
