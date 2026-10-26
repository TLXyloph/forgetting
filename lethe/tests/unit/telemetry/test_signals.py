from __future__ import annotations

import pytest

from lethe.telemetry.signals import is_plateau_claim


@pytest.mark.parametrize(
    "text",
    [
        "I've reached the practical limit of this machine configuration.",
        "Further tweaks show diminishing returns, so I made no change.",
        "The current kernel is already near-optimal.",
        "No further improvements are possible with this approach.",
        "We have hit a plateau.",
        "This cannot be optimized further.",
        "There is nothing more to improve here.",
    ],
)
def test_detects_plateau_claims(text):
    assert is_plateau_claim(text)


@pytest.mark.parametrize(
    "text",
    [
        "I unrolled the inner loop and fused the two hash stages.",
        "Removed the recursion limit check, which was redundant.",
        "Replaced first-fit with best-fit decreasing.",
        "",
    ],
)
def test_ignores_ordinary_change_summaries(text):
    assert not is_plateau_claim(text)
