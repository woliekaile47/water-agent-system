import numpy as np
import pytest

from src.integration.unknown_aware_geometry import build_trusted_shoreline


def test_only_water_known_nonwater_interfaces_are_trusted():
    water = np.zeros((5, 5), dtype=bool)
    unknown = np.zeros_like(water)
    water[2, 2] = True
    unknown[2, 3] = True
    trusted, points, diagnostics = build_trusted_shoreline(water, unknown)
    assert trusted[2, 2]
    assert len(points) == 3
    assert diagnostics["unknown_adjacent_interface_count"] == 1
    assert all(not (point["pixel_u"] == 2.5 and point["pixel_v"] == 2.0) for point in points)


def test_image_outside_is_not_a_trusted_shoreline():
    water = np.zeros((3, 3), dtype=bool)
    unknown = np.zeros_like(water)
    water[0, 1] = True
    _, points, diagnostics = build_trusted_shoreline(water, unknown)
    assert diagnostics["image_edge_interface_count"] == 1
    assert len(points) == 3
    assert all(point["pixel_v"] >= 0 for point in points)


def test_water_unknown_overlap_is_rejected():
    water = np.zeros((2, 2), dtype=bool)
    unknown = np.zeros_like(water)
    water[0, 0] = unknown[0, 0] = True
    with pytest.raises(ValueError, match="must not overlap"):
        build_trusted_shoreline(water, unknown)
