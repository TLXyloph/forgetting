"""`looks_rate_limited`: each signal on its own, against hand-made summaries."""

from __future__ import annotations

from typing import Any

import pytest

from lethe.agents.stream import StreamSummary, looks_rate_limited
from lethe.domain.types import RateLimitInfo, Usage

# --- helpers --------------------------------------------------------------------------------


def make_summary(**overrides: Any) -> StreamSummary:
    fields: dict[str, Any] = {
        "session_id": None,
        "has_result": True,
        "is_error": False,
        "result_text": "",
        "cost_usd": 0.0,
        "usage": Usage(),
        "tool_calls": (),
        "rate_limit": None,
        "num_api_turns": 0,
        "malformed_lines": 0,
    }
    fields.update(overrides)
    return StreamSummary(**fields)


# --- looks_rate_limited -------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["rejected", "blocked", "exceeded", "limited", "denied"])
def test_a_non_allowed_status_means_rate_limited(status: str) -> None:
    assert looks_rate_limited(make_summary(rate_limit=RateLimitInfo(status))) is True


def test_a_non_allowed_status_counts_even_when_the_turn_did_not_error() -> None:
    summary = make_summary(is_error=False, rate_limit=RateLimitInfo("rejected"))
    assert looks_rate_limited(summary) is True


@pytest.mark.parametrize("status", ["allowed", "allowed_warning", "allowed_overage", "allowed-x"])
def test_allowed_statuses_are_not_rate_limited(status: str) -> None:
    assert looks_rate_limited(make_summary(rate_limit=RateLimitInfo(status))) is False


def test_an_empty_status_is_not_rate_limited() -> None:
    assert looks_rate_limited(make_summary(rate_limit=RateLimitInfo(""))) is False


def test_no_signals_means_not_rate_limited() -> None:
    assert looks_rate_limited(make_summary()) is False


LIMIT_TEXTS = [
    "Claude AI usage limit reached|1793050200",
    "Rate limit exceeded",
    "rate_limit_error",
    "RATE-LIMIT hit",
    "ratelimit",
    "You have hit the Usage Limit",
    "usage_limit",
    "5-hour limit reached",
    "Limit Reached",
]


@pytest.mark.parametrize("message", LIMIT_TEXTS)
def test_error_text_mentioning_a_limit_means_rate_limited(message: str) -> None:
    assert looks_rate_limited(make_summary(is_error=True, result_text=message)) is True


@pytest.mark.parametrize("message", LIMIT_TEXTS)
def test_limit_text_on_a_successful_turn_is_not_a_rate_limit(message: str) -> None:
    # e.g. the agent's own summary says "I added rate limiting"
    assert looks_rate_limited(make_summary(is_error=False, result_text=message)) is False


@pytest.mark.parametrize(
    "message",
    [
        "API Error: 500 Internal server error",
        "overloaded_error",
        "invalid x-api-key",
        "prompt is too long",
        "unlimited credits",
        "",
    ],
)
def test_other_api_errors_are_not_rate_limits(message: str) -> None:
    assert looks_rate_limited(make_summary(is_error=True, result_text=message)) is False


def test_the_text_rule_applies_even_when_the_event_says_allowed() -> None:
    summary = make_summary(
        is_error=True,
        result_text="usage limit reached",
        rate_limit=RateLimitInfo("allowed"),
    )
    assert looks_rate_limited(summary) is True
