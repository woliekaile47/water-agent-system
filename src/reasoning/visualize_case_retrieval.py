#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-B: Visualize case retrieval correction result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_case_retrieval_figure(
    case_result: dict[str, Any],
    corrected_result: dict[str, Any],
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate S7-B figure: {exc}") from exc

    corrected_items = sorted(corrected_result["corrected_forecast_results"], key=lambda item: int(item["horizon_min"]))
    horizons = [int(item["horizon_min"]) for item in corrected_items]
    deterministic_depths = [float(item["deterministic_forecast_depth_cm"]) for item in corrected_items]
    corrected_depths = [float(item["corrected_forecast_depth_cm"]) for item in corrected_items]
    biases = [float(item["median_case_bias_cm"]) for item in corrected_items]
    top_cases = case_result["retrieved_cases"]
    case_labels = [case["case_id"].replace("mock_case_", "case_") for case in top_cases]
    similarities = [float(case["similarity_score"]) for case in top_cases]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.45, 1.3, 1.0])

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.5,
        0.72,
        "S7-B Case Retrieval Correction - MVP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    title_ax.text(
        0.5,
        0.25,
        "offline_mock_case_library, not final real historical-case correction",
        ha="center",
        va="center",
        fontsize=10,
        color="#b45309",
    )

    forecast_ax = fig.add_subplot(grid[1, :])
    forecast_ax.plot(horizons, deterministic_depths, marker="o", linewidth=2.2, color="#64748b", label="S7-A deterministic")
    forecast_ax.plot(horizons, corrected_depths, marker="o", linewidth=2.4, color="#0284c7", label="S7-B corrected")
    forecast_ax.axhline(15, color="#2563eb", linestyle="--", linewidth=1.0, label="blue 15 cm")
    forecast_ax.axhline(30, color="#ca8a04", linestyle="--", linewidth=1.0, label="yellow 30 cm")
    forecast_ax.axhline(50, color="#ea580c", linestyle="--", linewidth=1.0, label="orange 50 cm")
    forecast_ax.set_title("Forecast depth before/after case retrieval correction")
    forecast_ax.set_xlabel("horizon (min)")
    forecast_ax.set_ylabel("forecast depth (cm)")
    forecast_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    forecast_ax.legend(fontsize=8)
    for x_value, y_value in zip(horizons, corrected_depths):
        forecast_ax.text(x_value, y_value + 2, f"{y_value:.1f}", ha="center", va="bottom", fontsize=9)

    bias_ax = fig.add_subplot(grid[2, 0])
    bias_ax.bar([str(h) for h in horizons], biases, color="#0ea5e9")
    bias_ax.axhline(0, color="#334155", linewidth=0.8)
    bias_ax.set_title("Median historical bias")
    bias_ax.set_xlabel("horizon (min)")
    bias_ax.set_ylabel("bias (cm)")
    bias_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    similarity_ax = fig.add_subplot(grid[2, 1])
    similarity_ax.barh(case_labels[::-1], similarities[::-1], color="#22c55e")
    similarity_ax.set_xlim(0, max(similarities + [1.0]) * 1.1)
    similarity_ax.set_title("Top retrieved mock cases")
    similarity_ax.set_xlabel("similarity score")
    similarity_ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_case_retrieval(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    del config_path
    root = Path(project_root).expanduser().resolve()
    case_result_path = root / "outputs" / "json" / "case_retrieval_result.json"
    corrected_result_path = root / "outputs" / "json" / "corrected_forecast_result.json"
    output_path = root / "outputs" / "figures" / "case_retrieval_correction.png"
    case_result = load_json(case_result_path)
    corrected_result = load_json(corrected_result_path)
    save_case_retrieval_figure(case_result, corrected_result, output_path)
    print(f"[S7-B][case_figure] output file: {output_path}")
    return {"case_retrieval_correction": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S7-B case retrieval correction figure.")
    parser.add_argument("--config", required=True, help="Path to configs/case_retrieval_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_case_retrieval(args.config, args.project_root)


if __name__ == "__main__":
    main()
