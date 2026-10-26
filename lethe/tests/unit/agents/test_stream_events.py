"""`parse_stream`: tool calls and `rate_limit_event`s, using tiny synthetic streams."""

from __future__ import annotations

import json
from typing import Any

import pytest

from lethe.agents.stream import parse_stream
from lethe.domain.types import RateLimitInfo, ToolCall

# --- helpers: build stream-json lines -------------------------------------------------------


def line(**fields: Any) -> str:
    return json.dumps(fields)


def assistant(*blocks: dict[str, Any], **extra: Any) -> str:
    return line(type="assistant", message={"role": "assistant", "content": list(blocks)}, **extra)


def tool_use(name: str, **tool_input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def rate_event(**info: Any) -> str:
    return line(type="rate_limit_event", rate_limit_info=info)


# --- tool calls ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_input", "path"),
    [
        pytest.param(
            {"file_path": "/f", "notebook_path": "/n", "path": "/p"}, "/f", id="file_path"
        ),
        pytest.param({"notebook_path": "/n", "path": "/p"}, "/n", id="notebook_path"),
        pytest.param({"path": "/p"}, "/p", id="path"),
        pytest.param({"file_path": 5, "path": "/p"}, "/p", id="first-present-string"),
        pytest.param({"file_path": "", "path": "/p"}, "", id="empty-string-is-still-a-string"),
        pytest.param({"command": "ls"}, None, id="no-path-key"),
        pytest.param({}, None, id="empty-input"),
    ],
)
def test_tool_call_path_extraction(tool_input: dict[str, Any], path: str | None) -> None:
    s = parse_stream([assistant(tool_use("Whatever", **tool_input))])
    assert s.tool_calls == (ToolCall("Whatever", path),)


def test_tool_call_with_a_non_object_input_has_no_path() -> None:
    block = {"type": "tool_use", "id": "t", "name": "Weird", "input": "not-a-dict"}
    assert parse_stream([assistant(block)]).tool_calls == (ToolCall("Weird", None),)


def test_tool_calls_keep_stream_order_across_and_within_messages() -> None:
    s = parse_stream(
        [
            assistant(
                tool_use("Read", file_path="/a"), text("between"), tool_use("Grep", path="/b")
            ),
            line(type="user", message={"role": "user", "content": [{"type": "tool_result"}]}),
            assistant(tool_use("Edit", file_path="/a"), tool_use("Glob", pattern="*.py")),
        ]
    )
    assert s.tool_calls == (
        ToolCall("Read", "/a"),
        ToolCall("Grep", "/b"),
        ToolCall("Edit", "/a"),
        ToolCall("Glob", None),
    )


def test_only_assistant_messages_contribute_tool_calls_and_text() -> None:
    user_msg = line(
        type="user",
        message={"role": "user", "content": [tool_use("Edit", file_path="/x"), text("not me")]},
    )
    s = parse_stream([user_msg])
    assert s.tool_calls == ()
    assert s.result_text == ""


# --- rate limit ---------------------------------------------------------------------------------


def test_rate_limit_fields() -> None:
    info = {
        "status": "allowed_warning",
        "resetsAt": 1793050200,
        "unifiedWindows": {
            "five_hour": {"utilization": 0.9, "resetsAt": 1793050200},
            "seven_day": {"utilization": 1, "resetsAt": 1793631600},
        },
    }
    assert parse_stream([rate_event(**info)]).rate_limit == RateLimitInfo(
        status="allowed_warning",
        five_hour_utilization=0.9,
        seven_day_utilization=1.0,
        resets_at=1793050200.0,
    )


def test_the_last_rate_limit_event_wins() -> None:
    s = parse_stream([rate_event(status="rejected"), rate_event(status="allowed")])
    assert s.rate_limit == RateLimitInfo(status="allowed")


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        pytest.param({"status": "allowed"}, RateLimitInfo("allowed"), id="status-only"),
        pytest.param(
            {"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.5}}},
            RateLimitInfo("allowed", five_hour_utilization=0.5),
            id="only-the-five-hour-window",
        ),
        pytest.param(
            {"status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.25}}},
            RateLimitInfo("allowed", seven_day_utilization=0.25),
            id="only-the-seven-day-window",
        ),
        pytest.param(
            {"status": "allowed", "unifiedWindows": {"five_hour": {}, "seven_day": "x"}},
            RateLimitInfo("allowed"),
            id="windows-without-utilization",
        ),
        pytest.param(
            {"status": "allowed", "unifiedWindows": [1]}, RateLimitInfo("allowed"), id="bad-windows"
        ),
        pytest.param(
            {"status": "allowed", "resetsAt": "soon"},
            RateLimitInfo("allowed"),
            id="non-numeric-resetsAt",
        ),
        pytest.param({"status": "allowed", "resetsAt": True}, RateLimitInfo("allowed"), id="bool"),
        pytest.param({"resetsAt": 5}, RateLimitInfo("", resets_at=5.0), id="missing-status"),
        pytest.param({"status": 3}, RateLimitInfo(""), id="non-string-status"),
    ],
)
def test_rate_limit_missing_pieces_are_none(info: dict[str, Any], expected: RateLimitInfo) -> None:
    assert parse_stream([rate_event(**info)]).rate_limit == expected


def test_an_unusable_rate_limit_event_does_not_clobber_an_earlier_good_one() -> None:
    s = parse_stream(
        [
            rate_event(status="rejected"),
            line(type="rate_limit_event"),
            line(type="rate_limit_event", rate_limit_info="oops"),
        ]
    )
    assert s.rate_limit == RateLimitInfo(status="rejected")
