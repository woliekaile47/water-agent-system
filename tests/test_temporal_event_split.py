from pathlib import Path

import pytest

from src.perception.temporal_event_dataset import discover_sequence_splits


def _sequence(root: Path, case: str, rain: str, seed: int):
    (root / case / rain / f"seed_{seed}" / "frames").mkdir(parents=True)


def test_sequence_seed_split_has_no_overlap(tmp_path: Path):
    _sequence(tmp_path, "dry", "light", 41)
    _sequence(tmp_path, "water", "moderate", 111)
    _sequence(tmp_path, "water", "heavy", 201)
    splits, manifest = discover_sequence_splits(tmp_path, {
        "train_seeds": [41], "validation_seeds": [111], "test_seeds": [201],
    })
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [1, 1, 1]
    assert manifest["seed_overlap_check"] == {"overlap": [], "passed": True}


def test_overlapping_seed_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="seed overlap"):
        discover_sequence_splits(tmp_path, {
            "train_seeds": [41], "validation_seeds": [41], "test_seeds": [201],
        })
