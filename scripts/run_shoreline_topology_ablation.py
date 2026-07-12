#!/usr/bin/env python3
"""Run fixed topology methods on existing masks, then independently evaluate them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_shoreline_topology_ablation import (
    evaluate_candidates_after_prediction,
    evaluate_dry_safety,
    run_fixed_candidates_before_gt,
    write_outputs,
)


def _load_yaml(path: Path, key: str):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)[key]


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--config", default="configs/shoreline_topology_ablation.yaml")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser(); config_path = (root / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    config = _load_yaml(config_path, "shoreline_topology_ablation")
    methods, parameters = list(config["methods"]), dict(config["fixed_parameters"])
    prediction_root = root / config["source_prediction_root"]
    evaluations = json.loads((root / config["source_evaluation_json"]).read_text(encoding="utf-8"))
    geometry_records = json.loads((root / config["source_geometry_diagnostics"]).read_text(encoding="utf-8"))
    geometry_lookup = {(row["case_id"], row["rain_level"], row["seed"]): row for row in geometry_records}
    ground_dem = np.load(root / config["dry_ground_dem"]).astype(np.float32)
    sensors = yaml.safe_load((root / config["sensors_config"]).read_text(encoding="utf-8"))
    mapping = _load_yaml(root / config["mapping_config"], "water_surface_aware_mapping")
    gate_config = _load_yaml(root / config["geometry_gate_config"], "water_surface_aware_quality_gate")
    water_evaluations = sorted((item for item in evaluations if not item["is_dry"]), key=lambda item: (item["case_id"], item["rain_level"], item["seed"]))
    dry_evaluations = sorted((item for item in evaluations if item["is_dry"]), key=lambda item: (item["rain_level"], item["seed"]))
    all_rows, safety_rows = [], []
    for index, evaluation in enumerate(water_evaluations, start=1):
        key = (evaluation["case_id"], evaluation["rain_level"], evaluation["seed"])
        relative = Path(key[0]) / key[1] / f"seed_{key[2]}"; output = prediction_root / relative
        print(f"[phase2d-b2b] water {index}/{len(water_evaluations)} {relative}", flush=True)
        water, unknown = _mask(output / "predicted_camera_water_mask.png"), _mask(output / "predicted_camera_unknown_mask.png")
        # Complete all fixed prediction-side candidates before the evaluation call below can read GT.
        candidates = run_fixed_candidates_before_gt(water, unknown, methods, parameters, ground_dem, sensors, mapping, gate_config)
        sequence = root / "data/simulation_dynamic" / relative
        rows, safety = evaluate_candidates_after_prediction(sequence, key[0], key[1], key[2], candidates, water, geometry_lookup[key], root, sensors)
        all_rows.extend(rows); safety_rows.extend(safety)
    dry_rows = []
    for index, evaluation in enumerate(dry_evaluations, start=1):
        relative = Path(evaluation["case_id"]) / evaluation["rain_level"] / f"seed_{evaluation['seed']}"; output = prediction_root / relative
        print(f"[phase2d-b2b] dry {index}/{len(dry_evaluations)} {relative}", flush=True)
        dry_rows.extend(evaluate_dry_safety(evaluation["case_id"], evaluation["rain_level"], evaluation["seed"], _mask(output / "predicted_camera_water_mask.png"), _mask(output / "predicted_camera_unknown_mask.png"), methods, parameters))
    summary = write_outputs(root / config["output_root"], all_rows, methods, dry_rows, safety_rows, config)
    print(f"[phase2d-b2b] complete water={len(water_evaluations)} dry={len(dry_evaluations)} methods={len(methods)} output={root / config['output_root']}", flush=True)


if __name__ == "__main__":
    main()
