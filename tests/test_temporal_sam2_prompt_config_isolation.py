"""Regression checks for C17/C21, C22 and C23 prompt configuration isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = PROJECT_ROOT / "configs" / "temporal_sam2_prompt_corroborated.yaml"
C22_CONFIG = PROJECT_ROOT / "configs" / "temporal_sam2_prompt_c22_packing_aware.yaml"
C23_CONFIG = PROJECT_ROOT / "configs" / "temporal_sam2_prompt_c23_shallow_recall.yaml"
SEED311_FROZEN_PROMPT_CONFIG_SHA256 = (
    "d90c94266677c1baf7cdb8cd9005f606b316827bf9833d0f0c8fbae5b576cc39"
)


def _load(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["temporal_sam2_prompt"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c22_prompt_config_matches_frozen_seed311_hash() -> None:
    assert _sha256(C22_CONFIG) == SEED311_FROZEN_PROMPT_CONFIG_SHA256


def test_legacy_and_c22_configs_differ_only_by_algorithm_selector() -> None:
    legacy = _load(LEGACY_CONFIG)
    c22 = _load(C22_CONFIG)

    assert legacy["algorithm_version"] == "phase2d_c17_temporal_prompt_corroboration_v1"
    assert "positive_point_selection_method" not in legacy
    assert c22["algorithm_version"] == (
        "phase2d_c22_feasibility_first_positive_points_v1"
    )
    assert c22["positive_point_selection_method"] == (
        "feasibility_preserving_fallback_v1"
    )

    c22_without_selector = dict(c22)
    c22_without_selector.pop("positive_point_selection_method")
    c22_without_selector["algorithm_version"] = legacy["algorithm_version"]
    assert c22_without_selector == legacy


def test_c22_prompt_only_entry_uses_isolated_config_by_default() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "rebuild_prompt_from_frozen_fusion.py"
    ).read_text(encoding="utf-8")
    assert '"temporal_sam2_prompt_c22_packing_aware.yaml"' in script


def test_c23_prompt_envelope_is_opt_in_and_does_not_mutate_c22() -> None:
    c22 = _load(C22_CONFIG)
    c23 = _load(C23_CONFIG)

    assert _sha256(C22_CONFIG) == SEED311_FROZEN_PROMPT_CONFIG_SHA256
    assert "use_temporal_support_prompt_envelope" not in c22
    assert c23["algorithm_version"] == (
        "phase2d_c23_recurrent_support_envelope_v1"
    )
    assert c23["use_temporal_support_prompt_envelope"] is True
    assert c23["min_positive_points"] == c22["min_positive_points"] == 3
    assert c23["min_positive_probability"] == c22["min_positive_probability"]
    assert (
        c23["min_positive_temporal_support_fraction"]
        == c22["min_positive_temporal_support_fraction"]
    )
