#!/usr/bin/env python3
"""Small deterministic NumPy logistic model for temporal track features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_regression(
    features: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray, config: dict[str, Any],
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    weights_per_sample = np.asarray(sample_weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.size or y.size == 0:
        raise ValueError("non-empty feature matrix and matching labels are required")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = np.clip((x - mean) / std, -12.0, 12.0)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    positive_rate = float(np.mean(y))
    bias = float(np.log(max(positive_rate, 1e-4) / max(1.0 - positive_rate, 1e-4)))
    learning_rate = float(config["learning_rate"])
    l2 = float(config["l2_regularization"])
    weight_sum = max(float(np.sum(weights_per_sample)), 1e-9)
    losses = []
    for iteration in range(int(config["iterations"])):
        probability = sigmoid(normalized @ coefficients + bias)
        error = (probability - y) * weights_per_sample
        gradient = normalized.T @ error / weight_sum + l2 * coefficients
        bias_gradient = float(np.sum(error) / weight_sum)
        coefficients -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
        if iteration % 100 == 0 or iteration == int(config["iterations"]) - 1:
            clipped = np.clip(probability, 1e-8, 1 - 1e-8)
            loss = -np.sum(weights_per_sample * (y * np.log(clipped) + (1 - y) * np.log(1 - clipped))) / weight_sum
            losses.append(float(loss + 0.5 * l2 * np.sum(coefficients * coefficients)))
    return {
        "weights": coefficients, "bias": bias, "feature_mean": mean, "feature_std": std,
        "training_loss_trace": losses,
    }


def predict_water_probability(feature_matrix: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_matrix, dtype=np.float64)
    normalized = (values - np.asarray(model["feature_mean"])) / np.asarray(model["feature_std"])
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=12.0, neginf=-12.0)
    probability = sigmoid(np.clip(normalized, -12.0, 12.0) @ np.asarray(model["weights"]) + float(model["bias"]))
    return probability, normalized


def save_model(model_dir: str | Path, model: dict[str, Any], metadata: dict[str, Any]) -> None:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "model_weights.npz", weights=model["weights"], bias=np.asarray([model["bias"]]),
    )
    with (directory / "feature_normalization.json").open("w", encoding="utf-8") as stream:
        json.dump({"mean": model["feature_mean"].tolist(), "std": model["feature_std"].tolist()}, stream, indent=2)
        stream.write("\n")
    with (directory / "feature_schema.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "feature_names": metadata["feature_names"],
            "feature_schema_version": metadata["feature_schema_version"],
            "model_version": metadata["model_version"],
        }, stream, indent=2)
        stream.write("\n")


def load_model(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir)
    arrays = np.load(directory / "model_weights.npz")
    normalization = json.loads((directory / "feature_normalization.json").read_text(encoding="utf-8"))
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    return {
        "weights": arrays["weights"], "bias": float(arrays["bias"][0]),
        "feature_mean": np.asarray(normalization["mean"], dtype=np.float64),
        "feature_std": np.asarray(normalization["std"], dtype=np.float64),
        **schema,
    }
