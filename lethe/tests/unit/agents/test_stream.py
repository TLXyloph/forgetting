"""Summary-field rules of `parse_stream`, using tiny synthetic streams.

Each binding rule from the module docstring gets its own test; the recorded CLI output is covered
in test_stream_fixtures.py, tool calls and rate-limit events in test_stream_events.py, and the
rate-limit heuristic in test_looks_rate_limited.py.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from lethe.agents.stream import StreamSummary, parse_stream
from lethe.domain.types import RateLimitInfo, Usage

# --- helpers: build stream-json lines -------------------------------------------------------


def line(**fields: Any) -> str:
    return json.dumps(fields)


def result_line(**fields: Any) -> str:
    return line(**{"type": "result", "subtype": "success", "is_error": False, **fields})


def assistant(*blocks: dict[str, Any], **extra: Any) -> str:
    return line(type="assistant", message={"role": "assistant", "content": list(blocks)}, **extra)


def tool_use(name: str, **tool_input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def rate_event(**info: Any) -> str:
    return line(type="rate_limit_event", rate_limit_info=info)


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


# --- empty / malformed input ------------------------------------------------------------------


def test_empty_stream_gives_the_neutral_summary() -> None:
    assert parse_stream([]) == make_summary(has_result=False)


def test_blank_lines_are_skipped_without_counting_as_malformed() -> None:
    assert parse_stream(["", "   ", "\n", "\t\r\n"]).malformed_lines == 0


def test_line_terminators_are_tolerated() -> None:
    s = parse_stream([result_line(result="one") + "\n", result_line(result="two") + "\r\n"])
    assert s.malformed_lines == 0
    assert s.result_text == "two"


def test_every_kind_of_non_object_line_is_counted() -> None:
    lines = ["nope", "[1]", "7", "2.5", '"s"', "null", "true", "{", '{"type":"result"']
    assert parse_stream(lines).malformed_lines == len(lines)


def test_hostile_input_never_raises() -> None:
    hostile: list[Any] = [
        "[" * 200_000,  # RecursionError inside the json module
        '{"a":' * 100_000,  # nesting that is also never closed
        "1" * 5_000,  # beyond the int-digits limit: a plain ValueError (or a bare int)
        "\x00\x01\x02",
        None,  # not a string at all
        b'{"type":"result"}',  # bytes are not accepted as lines
        12,
    ]
    s = parse_stream(hostile)
    assert s.malformed_lines == len(hostile)
    assert s.has_result is False


def test_structurally_odd_objects_are_ignored_without_raising() -> None:
    odd = [
        line(type="assistant"),  # no message
        line(type="assistant", message="oops"),
        line(type="assistant", message={"content": "plain string"}),
        line(
            type="assistant",
            message={"content": [1, None, "x", {"type": "tool_use"}, {"type": "text"}]},
        ),
        line(type="rate_limit_event"),
        line(type="rate_limit_event", rate_limit_info=[1]),
        line(type="system", subtype="init"),
        line(type=["not", "a", "string"]),
        line(type="future_event", payload={"x": 1}),
        line(),
        line(type="result", usage="nope", model_usage_since_last_result=5, num_turns="3"),
    ]
    s = parse_stream(odd)

    assert s.malformed_lines == 0
    assert s.tool_calls == ()  # a tool_use block without a name is not a call
    assert s.rate_limit is None
    assert s.has_result is True
    assert s.usage == Usage()
    assert s.cost_usd == 0.0
    assert s.num_api_turns == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_never_reach_the_summary(bad: float) -> None:
    # events.jsonl is written with allow_nan=False, so a NaN here would crash the run later
    s = parse_stream(
        [
            rate_event(
                status="allowed",
                resetsAt=bad,
                unifiedWindows={
                    "five_hour": {"utilization": bad},
                    "seven_day": {"utilization": bad},
                },
            ),
            result_line(total_cost_usd=bad),
        ]
    )
    assert s.cost_usd == 0.0
    assert s.rate_limit == RateLimitInfo(status="allowed")

    s = parse_stream([result_line(model_usage_since_last_result={"m": {"costUSD": bad}})])
    assert s.cost_usd == 0.0 and math.isfinite(s.cost_usd)


# --- session id ---------------------------------------------------------------------------------


def test_session_id_comes_from_result_then_init_then_first_line_carrying_one() -> None:
    first = line(type="assistant", message={"content": []}, session_id="from-first-line")
    init = line(type="system", subtype="init", session_id="from-init")
    final = result_line(session_id="from-result")

    assert parse_stream([first, init, final]).session_id == "from-result"
    assert parse_stream([first, init]).session_id == "from-init"
    assert parse_stream([first]).session_id == "from-first-line"


def test_session_id_ignores_lines_without_a_usable_one() -> None:
    bare = line(type="assistant", message={"content": []})
    wrong_type = line(type="assistant", message={"content": []}, session_id=7)
    empty = line(type="user", session_id="")
    carrying = line(type="user", session_id="second")

    assert parse_stream([bare, wrong_type, empty, carrying]).session_id == "second"
    assert parse_stream([bare, wrong_type, empty]).session_id is None


def test_only_system_init_counts_as_init() -> None:
    # `system/thinking_tokens` lines also carry session ids (see the forked-session capture)
    other = line(type="system", subtype="thinking_tokens", session_id="from-other-system-line")
    init = line(type="system", subtype="init", session_id="from-init")
    assert parse_stream([other, init]).session_id == "from-init"


# --- per-turn cost ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param(
            {
                "model_usage_since_last_result": {"a": {"costUSD": 0.25}, "b": {"costUSD": 0.5}},
                "total_cost_usd": 9.0,
            },
            0.75,
            id="sums-all-models-and-ignores-the-cumulative-total",
        ),
        pytest.param(
            {"model_usage_since_last_result": {"a": {"costUSD": 0}}, "total_cost_usd": 9.0},
            0.0,
            id="present-but-zero-still-wins-over-total",
        ),
        pytest.param(
            {"model_usage_since_last_result": {}, "total_cost_usd": 9.0},
            0.0,
            id="present-but-empty-still-wins-over-total",
        ),
        pytest.param({"total_cost_usd": 9.0}, 9.0, id="falls-back-to-total"),
        pytest.param({"total_cost_usd": 3}, 3.0, id="integer-total"),
        pytest.param({}, 0.0, id="neither-field"),
        pytest.param(
            {"model_usage_since_last_result": None, "total_cost_usd": 9.0},
            9.0,
            id="null-usage-counts-as-absent",
        ),
        pytest.param(
            {"model_usage_since_last_result": [1], "total_cost_usd": 9.0},
            9.0,
            id="non-mapping-usage-counts-as-absent",
        ),
        pytest.param(
            {
                "model_usage_since_last_result": {
                    "str": {"costUSD": "1.5"},
                    "junk": "junk",
                    "bool": {"costUSD": True},
                    "none": {"costUSD": None},
                    "missing": {},
                    "ok": {"costUSD": 0.1},
                }
            },
            0.1,
            id="unusable-entries-contribute-nothing",
        ),
        pytest.param({"total_cost_usd": "3"}, 0.0, id="string-total-is-ignored"),
        pytest.param({"total_cost_usd": True}, 0.0, id="bool-total-is-ignored"),
    ],
)
def test_cost_rules(fields: dict[str, Any], expected: float) -> None:
    assert parse_stream([result_line(**fields)]).cost_usd == pytest.approx(expected)


def test_cost_is_zero_without_a_result_line() -> None:
    assert parse_stream([assistant(text("hi"))]).cost_usd == 0.0


# --- usage --------------------------------------------------------------------------------------


def test_usage_field_mapping() -> None:
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4,
        "output_tokens_details": {"thinking_tokens": 5},
    }
    assert parse_stream([result_line(usage=usage)]).usage == Usage(
        input_tokens=1,
        output_tokens=2,
        cache_read_tokens=3,
        cache_creation_tokens=4,
        thinking_tokens=5,
    )


def test_usage_missing_or_non_int_fields_are_zero() -> None:
    usage = {
        "input_tokens": "7",
        "output_tokens": 3.5,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": True,  # bool is an int subclass but not a token count
        "output_tokens_details": "n/a",
    }
    assert parse_stream([result_line(usage=usage)]).usage == Usage()
    assert parse_stream([result_line()]).usage == Usage()  # no usage at all
    assert parse_stream([result_line(usage=[1, 2])]).usage == Usage()


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param(
            {
                "usage": {"output_tokens_details": {"thinking_tokens": 7}},
                "model_usage_since_last_result": {"m": {"thinkingTokens": 99}},
            },
            7,
            id="usage-detail-wins",
        ),
        pytest.param(
            {
                "usage": {"output_tokens_details": {}},
                "model_usage_since_last_result": {
                    "a": {"thinkingTokens": 10},
                    "b": {"thinkingTokens": 5},
                    "c": {"thinkingTokens": "x"},
                    "d": "junk",
                },
            },
            15,
            id="falls-back-to-the-per-model-sum",
        ),
        pytest.param(
            {
                "usage": {"output_tokens_details": {"thinking_tokens": "many"}},
                "model_usage_since_last_result": {"a": {"thinkingTokens": 4}},
            },
            4,
            id="non-int-detail-falls-back",
        ),
        pytest.param({"usage": {}}, 0, id="nothing-to-fall-back-on"),
        pytest.param(
            {"usage": {}, "model_usage_since_last_result": {"a": {}}}, 0, id="entries-without-field"
        ),
    ],
)
def test_thinking_tokens_rules(fields: dict[str, Any], expected: int) -> None:
    assert parse_stream([result_line(**fields)]).usage.thinking_tokens == expected


# --- result text, error flag, turns --------------------------------------------------------------


def test_result_text_prefers_the_result_line() -> None:
    s = parse_stream([assistant(text("from assistant")), result_line(result="from result")])
    assert s.result_text == "from result"


def test_result_text_falls_back_to_concatenated_assistant_text_blocks() -> None:
    s = parse_stream(
        [
            assistant(text("Hello "), {"type": "thinking", "thinking": "hmm"}),
            assistant(tool_use("Read", file_path="/a")),
            assistant({"type": "text", "text": "world"}, text("!")),
            result_line(),  # no `result` field
        ]
    )
    assert s.result_text == "Hello world!"


def test_non_string_result_field_falls_back_to_assistant_text() -> None:
    s = parse_stream([assistant(text("fallback")), result_line(result=None)])
    assert s.result_text == "fallback"


def test_is_error_flag() -> None:
    assert parse_stream([result_line(is_error=True)]).is_error is True
    assert parse_stream([result_line(is_error=False)]).is_error is False
    assert parse_stream([line(type="result")]).is_error is False  # missing
    assert parse_stream([assistant(text("only text"))]).is_error is False  # no result line
    assert parse_stream([result_line(is_error=True)]).has_result is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(4, 4), (0, 0), ("3", 0), (2.0, 0), (True, 0), (None, 0)],
)
def test_num_api_turns(value: Any, expected: int) -> None:
    assert parse_stream([result_line(num_turns=value)]).num_api_turns == expected
    assert parse_stream([result_line()]).num_api_turns == 0


def test_the_last_result_line_wins() -> None:
    s = parse_stream(
        [
            result_line(result="first", is_error=True, num_turns=1, total_cost_usd=1.0),
            result_line(result="second", is_error=False, num_turns=2, total_cost_usd=2.0),
        ]
    )
    assert (s.result_text, s.is_error, s.num_api_turns, s.cost_usd) == ("second", False, 2, 2.0)
