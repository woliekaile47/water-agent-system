from __future__ import annotations

import pytest

from dashboard.utils import status_badge_level


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("reject", "error"),
        ("unavailable", "error"),
        (False, "error"),
        ("blocked", "error"),
        ("partial", "warning"),
        ("not_ready", "warning"),
        ("warning_suppressed", "warning"),
        ("pass", "success"),
        ("complete", "success"),
        (True, "success"),
        ("healthy", "success"),
    ],
)
def test_status_badge_level(value: object, expected: str) -> None:
    assert status_badge_level(value) == expected
