import json
import shutil
from pathlib import Path

import numpy as np

from src.perception.temporal_event_classifier_model import load_model, save_model
from src.perception.temporal_event_classifier_inference import infer_track_classifications


def test_model_artifacts_exclude_case_seed_and_gt_answers(tmp_path: Path):
    model = {"weights": np.asarray([1.0]), "bias": 0.2,
             "feature_mean": np.asarray([0.0]), "feature_std": np.asarray([1.0])}
    metadata = {"feature_names": ["duration_frames"], "feature_schema_version": "v",
                "model_version": "m"}
    save_model(tmp_path, model, metadata)
    loaded = load_model(tmp_path)
    assert loaded["weights"].tolist() == [1.0]
    text = " ".join(path.read_bytes().decode("latin1", errors="ignore") for path in tmp_path.iterdir()).lower()
    for forbidden in ("water_mask", "case_depth", "generator_seed", "random_seed"):
        assert forbidden not in text


def test_frozen_classifier_is_unchanged_when_gt_and_metadata_change_or_disappear(tmp_path: Path):
    model_dir = tmp_path / "model"
    sequence = tmp_path / "sequence"
    (sequence / "ground_truth").mkdir(parents=True)
    (sequence / "metadata").mkdir()
    (sequence / "ground_truth" / "event_annotations.json").write_text(json.dumps({"event_type": "water_ripple"}))
    (sequence / "metadata" / "answers.json").write_text(json.dumps({"seed": 999, "depth": 40}))
    save_model(model_dir, {"weights": np.asarray([1.0]), "bias": 0.0,
               "feature_mean": np.asarray([4.0]), "feature_std": np.asarray([1.0])},
               {"feature_names": ["duration_frames"], "feature_schema_version": "v", "model_version": "m"})
    model = load_model(model_dir)
    feature = {"track_id": "t", "duration_frames": 6, "valid_observation_count": 3,
               "radius_growth_slope": 0, "post_peak_persistence": .5,
               "expansion_monotonicity": .5, "ringness": 1, "center_drift": 1,
               "decay_rate": .1, "spatial_compactness": .5}
    rule = {"minimum_track_observations": 2, "water_score_threshold": .48,
            "dry_score_threshold": .48, "minimum_score_margin": .1}
    thresholds = {"low_threshold": .45, "high_threshold": .8}
    first = infer_track_classifications([feature], model, thresholds, rule)
    (sequence / "ground_truth" / "event_annotations.json").write_text(json.dumps({"event_type": "dry_splash"}))
    (sequence / "metadata" / "answers.json").write_text("corrupt")
    second = infer_track_classifications([feature], model, thresholds, rule)
    shutil.rmtree(sequence / "ground_truth")
    shutil.rmtree(sequence / "metadata")
    third = infer_track_classifications([feature], model, thresholds, rule)
    assert first == second == third
