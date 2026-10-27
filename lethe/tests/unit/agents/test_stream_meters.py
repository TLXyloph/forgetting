"""`parse_stream`: the billing source, the five-hour reset, and the cumulative session total."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lethe.agents.stream import parse_stream

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude"


def line(**fields: Any) -> str:
    return json.dumps(fields)


def init(**fields: Any) -> str:
    return line(type="system", subtype="init", session_id="s", **fields)


def test_api_key_source_comes_from_the_init_lines_and_a_reported_key_wins():
    assert parse_stream([init(apiKeySource="ANTHROPIC_API_KEY")]).api_key_source == (
        "ANTHROPIC_API_KEY"
    )
    keyed, subscription = init(apiKeySource="apiKeyHelper"), init(apiKeySource="none")
    assert parse_stream([keyed, subscription]).api_key_source == "apiKeyHelper"
    assert parse_stream([subscription, keyed]).api_key_source == "apiKeyHelper"
    assert parse_stream([subscription, subscription]).api_key_source == "none"


def test_api_key_source_is_none_when_missing_or_unusable():
    assert parse_stream([]).api_key_source is None
    assert parse_stream([init()]).api_key_source is None
    assert parse_stream([init(apiKeySource=7)]).api_key_source is None
    other = line(type="system", subtype="hook_response", apiKeySource="ANTHROPIC_API_KEY")
    assert parse_stream([other]).api_key_source is None


def test_recorded_subscription_turns_report_no_api_key():
    for name in ("turn1.jsonl", "turn2.jsonl", "api_error.jsonl", "rate_limited.jsonl"):
        lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        assert parse_stream(lines).api_key_source == "none", name


def test_the_five_hour_reset_is_read_from_its_own_window():
    windows = {"five_hour": {"utilization": 0.97, "resetsAt": 5000}, "seven_day": {"resetsAt": 9}}
    event = line(
        type="rate_limit_event",
        rate_limit_info={"status": "allowed_warning", "resetsAt": 9000, "unifiedWindows": windows},
    )
    info = parse_stream([event]).rate_limit
    assert info is not None
    assert (info.resets_at, info.five_hour_resets_at) == (9000.0, 5000.0)
    bare = line(type="rate_limit_event", rate_limit_info={"status": "allowed", "resetsAt": 1})
    assert parse_stream([bare]).rate_limit.five_hour_resets_at is None


def test_the_cumulative_total_is_reported_and_the_fallback_is_flagged():
    per_turn = {"model_usage_since_last_result": {"m": {"costUSD": 0.25}}}
    with_usage = parse_stream([line(type="result", total_cost_usd=9.0, **per_turn)])
    assert (with_usage.cost_usd, with_usage.total_cost_usd, with_usage.cost_is_total) == (
        0.25,
        9.0,
        False,
    )
    total_only = parse_stream([line(type="result", total_cost_usd=9.0)])
    assert (total_only.cost_usd, total_only.total_cost_usd, total_only.cost_is_total) == (
        9.0,
        9.0,
        True,
    )
    nothing = parse_stream([line(type="result")])
    assert (nothing.total_cost_usd, nothing.cost_is_total) == (None, False)
