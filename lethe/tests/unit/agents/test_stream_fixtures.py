"""`parse_stream` against recorded Claude Code output (lethe/tests/fixtures/claude/).

`turn1/2/3.jsonl` are real captures from Claude Code 2.1.310 (haiku, subscription login) with the
workspace paths rewritten to /WORKSPACE and nothing else changed:
  turn1  new session: Read + Edit + a closing text message
  turn2  `--resume` of that session (total_cost_usd is cumulative, the per-turn cost is not)
  turn3  resume + `--fork-session` from another directory (new session id, thinking tokens)
The remaining fixtures are hand-built to cover failure shapes the probe could not produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lethe.agents.stream import StreamSummary, looks_rate_limited, parse_stream
from lethe.domain.types import RateLimitInfo, ToolCall, Usage

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude"


def fixture_lines(name: str) -> list[str]:
    # split on "\n" only: str.splitlines() would also split on U+2028 and friends inside strings
    return (FIXTURES / name).read_text(encoding="utf-8").split("\n")


def summarize(name: str) -> StreamSummary:
    return parse_stream(fixture_lines(name))


def last_json_line(name: str) -> dict:
    return json.loads([ln for ln in fixture_lines(name) if ln.strip()][-1])


def test_new_session_turn() -> None:
    s = summarize("turn1.jsonl")

    assert s.session_id == "a76c6362-2811-4c08-ad32-d8222b63afe8"
    assert s.has_result is True
    assert s.is_error is False
    assert s.result_text == 'I changed "hello" to "hello there" in hello.txt.'
    assert s.cost_usd == pytest.approx(0.00091024)
    assert s.usage == Usage(
        input_tokens=6,
        output_tokens=342,
        cache_read_tokens=16344,
        cache_creation_tokens=2876,
        thinking_tokens=0,
    )
    assert s.tool_calls == (
        ToolCall("Read", "/WORKSPACE/hello.txt"),
        ToolCall("Edit", "/WORKSPACE/hello.txt"),
    )
    assert s.rate_limit == RateLimitInfo(
        status="allowed",
        five_hour_utilization=0.16,
        seven_day_utilization=0.07,
        resets_at=1793050200.0,
        five_hour_resets_at=1793050200.0,
    )
    assert s.num_api_turns == 3
    assert s.malformed_lines == 0
    assert looks_rate_limited(s) is False


def test_resumed_turn_reports_the_per_turn_cost_not_the_cumulative_total() -> None:
    raw = last_json_line("turn2.jsonl")
    assert raw["total_cost_usd"] == pytest.approx(0.00123078)  # cumulative across --resume

    s = summarize("turn2.jsonl")

    assert s.cost_usd == pytest.approx(0.00032054)
    assert s.cost_usd < raw["total_cost_usd"]
    assert s.session_id == "a76c6362-2811-4c08-ad32-d8222b63afe8"  # --resume keeps the id
    assert (
        s.result_text == 'I changed "there" to "again" in hello.txt, so it now reads "hello again."'
    )
    assert s.usage == Usage(
        input_tokens=4,
        output_tokens=215,
        cache_read_tokens=13444,
        cache_creation_tokens=391,
        thinking_tokens=0,
    )
    assert s.tool_calls == (ToolCall("Edit", "/WORKSPACE/hello.txt"),)
    assert s.num_api_turns == 2
    assert s.has_result and not s.is_error and s.malformed_lines == 0


def test_forked_session_turn_reports_the_new_session_id_and_thinking_tokens() -> None:
    s = summarize("turn3.jsonl")

    assert s.session_id == "3b6c9f0b-5d35-420f-943e-c3e7bd8389e3"
    assert s.cost_usd == pytest.approx(0.00018194)
    assert s.usage == Usage(
        input_tokens=2,
        output_tokens=116,
        cache_read_tokens=7074,
        cache_creation_tokens=265,
        thinking_tokens=113,
    )
    assert s.tool_calls == ()
    assert s.result_text == "again"  # the thinking block never leaks into the text
    assert s.num_api_turns == 1
    assert s.rate_limit is not None
    assert s.rate_limit.five_hour_utilization == 0.17
    assert s.rate_limit.seven_day_utilization == 0.07


def test_usage_limit_error_turn() -> None:
    s = summarize("rate_limited.jsonl")

    assert s.has_result and s.is_error
    assert s.result_text == "Claude AI usage limit reached|1793050200"
    assert s.rate_limit == RateLimitInfo(
        status="rejected",
        five_hour_utilization=1.0,  # the fixture carries the integer 1
        seven_day_utilization=0.62,
        resets_at=1793050200.0,
        five_hour_resets_at=1793050200.0,
    )
    # model_usage_since_last_result is present (and empty): nothing was spent this turn, even
    # though the cumulative total_cost_usd in the same line is 0.0123
    assert last_json_line("rate_limited.jsonl")["total_cost_usd"] == 0.0123
    assert s.cost_usd == 0.0
    assert s.usage == Usage()
    assert s.tool_calls == ()
    assert looks_rate_limited(s) is True


def test_generic_api_error_turn_is_not_rate_limited() -> None:
    s = summarize("api_error.jsonl")

    assert s.has_result and s.is_error
    assert s.result_text == "API Error: 500 Internal server error"
    assert s.rate_limit is not None and s.rate_limit.status == "allowed"
    assert s.cost_usd == pytest.approx(0.00041)
    assert s.usage == Usage(
        input_tokens=3,
        output_tokens=18,
        cache_read_tokens=4100,
        cache_creation_tokens=120,
        thinking_tokens=0,
    )
    assert looks_rate_limited(s) is False


def test_malformed_and_non_object_lines_are_counted_and_ignored() -> None:
    s = summarize("malformed.jsonl")

    # not-json, [1,2,3], 42, "str", null, and a truncated object; the blank line is not counted
    assert s.malformed_lines == 6
    # the valid lines around them still contribute
    assert s.session_id == "5a1d0000-0000-4000-8000-000000000003"
    assert s.tool_calls == (ToolCall("Grep", "/WORKSPACE/src"),)
    assert s.has_result and not s.is_error
    assert s.result_text == "Searched the sources."
    assert s.cost_usd == pytest.approx(0.002)  # no model_usage_since_last_result: total_cost_usd
    assert s.usage == Usage(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=30,
        cache_creation_tokens=40,
        thinking_tokens=5,
    )
    assert s.num_api_turns == 2


def test_stream_without_a_result_line() -> None:
    s = summarize("no_result.jsonl")

    assert s.has_result is False
    assert s.is_error is False
    assert s.session_id == "5a1d0000-0000-4000-8000-000000000004"  # from system/init
    assert s.result_text == "I am about to edit the solver."  # concatenated assistant text
    assert s.cost_usd == 0.0
    assert s.usage == Usage()
    assert s.num_api_turns == 0
    assert s.rate_limit is None
    assert s.malformed_lines == 0
    assert s.tool_calls == (
        ToolCall("Read", "/WORKSPACE/solver.py"),
        ToolCall("Edit", "/WORKSPACE/solver.py"),
    )
    assert looks_rate_limited(s) is False


def test_summaries_are_identical_for_lists_and_lazy_iterators() -> None:
    lines = fixture_lines("turn1.jsonl")
    assert parse_stream(iter(lines)) == parse_stream(lines)
    assert parse_stream(ln for ln in lines) == parse_stream(lines)


# --- rate-limit detection, each signal on its own, against realistic data --------------------


def _drop_rate_limit_events(lines: list[str]) -> list[str]:
    return [ln for ln in lines if '"type":"rate_limit_event"' not in ln]


def _without_error_flag(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        obj = json.loads(ln) if ln.strip() else None
        if obj and obj.get("type") == "result":
            obj["is_error"] = False
            ln = json.dumps(obj)
        out.append(ln)
    return out


def test_usage_limit_is_detected_from_the_error_text_alone() -> None:
    lines = _drop_rate_limit_events(fixture_lines("rate_limited.jsonl"))
    s = parse_stream(lines)

    assert s.rate_limit is None
    assert s.is_error and "usage limit reached" in s.result_text
    assert looks_rate_limited(s) is True


def test_usage_limit_is_detected_from_the_rejected_event_alone() -> None:
    s = parse_stream(_without_error_flag(fixture_lines("rate_limited.jsonl")))

    assert s.is_error is False  # error text is ignored unless the result is an error ...
    assert s.rate_limit is not None and s.rate_limit.status == "rejected"
    assert looks_rate_limited(s) is True  # ... but a non-"allowed" status is enough
