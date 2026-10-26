"""Parser for Claude Code's `--output-format stream-json --verbose` output.

Observed line types (Claude Code 2.1.x): `system/init`, `assistant` (message.content blocks of
type text / thinking / tool_use), `user` (tool results), `rate_limit_event`, and a final `result`.

Facts the parser relies on (verified against a live CLI):
  * `result.total_cost_usd` is CUMULATIVE across `--resume`; the per-turn cost is the sum of
    `result.model_usage_since_last_result[<model>].costUSD` (fall back to `total_cost_usd` only
    when that field is absent).
  * `result.usage` has input_tokens, output_tokens, cache_read_input_tokens,
    cache_creation_input_tokens, and output_tokens_details.thinking_tokens for this turn.
  * `rate_limit_event.rate_limit_info` has `status`, `resetsAt`, and
    `unifiedWindows.{five_hour,seven_day}.{utilization,resetsAt}`.
  * tool_use blocks carry `name` and `input`; file tools use `input.file_path`
    (NotebookEdit: `notebook_path`; Glob/Grep: `path` when given).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lethe.domain.types import RateLimitInfo, ToolCall, Usage

# Input keys that name the file a tool acts on, in priority order.
_PATH_KEYS = ("file_path", "notebook_path", "path")

# An error result whose text matches this is a usage/rate limit, e.g.
# "Claude AI usage limit reached|1793050200".
_LIMIT_TEXT = re.compile(r"(?i)(usage|rate)[ _-]?limit|limit reached")


@dataclass(frozen=True)
class StreamSummary:
    session_id: str | None
    has_result: bool  # a `result` line was seen
    is_error: bool  # result.is_error (False when no result line)
    result_text: str  # result.result, or the concatenated assistant text if absent
    cost_usd: float  # per-turn cost, see module docstring
    usage: Usage
    tool_calls: tuple[ToolCall, ...]
    rate_limit: RateLimitInfo | None  # from the last rate_limit_event, if any
    num_api_turns: int  # result.num_turns
    malformed_lines: int  # lines that were not valid JSON objects (ignored)


def parse_stream(lines: Iterable[str]) -> StreamSummary:
    """Summarize one turn's stream-json lines. Never raises on malformed input.

    Blank lines are skipped; any other line that is not a JSON object (including a non-string
    item) is counted in `malformed_lines` and ignored. If several `result` lines appear, the
    last one wins.
    """
    malformed = 0
    first_sid: str | None = None
    init_sid: str | None = None
    result: Mapping[str, Any] | None = None
    rate_limit: RateLimitInfo | None = None
    tool_calls: list[ToolCall] = []
    assistant_text: list[str] = []

    for raw in lines:
        if not isinstance(raw, str):
            malformed += 1
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except (ValueError, RecursionError):  # JSONDecodeError is a ValueError
            malformed += 1
            continue
        if not isinstance(obj, dict):
            malformed += 1
            continue

        sid = _non_empty_str(obj.get("session_id"))
        if first_sid is None:
            first_sid = sid
        kind = obj.get("type")
        if kind == "system":
            if init_sid is None and obj.get("subtype") == "init":
                init_sid = sid
        elif kind == "assistant":
            for block in _content_blocks(obj):
                _collect_block(block, tool_calls, assistant_text)
        elif kind == "rate_limit_event":
            info = _parse_rate_limit(obj.get("rate_limit_info"))
            if info is not None:
                rate_limit = info
        elif kind == "result":
            result = obj

    session_id = first_sid
    if init_sid is not None:
        session_id = init_sid
    if result is None:
        return StreamSummary(
            session_id=session_id,
            has_result=False,
            is_error=False,
            result_text="".join(assistant_text),
            cost_usd=0.0,
            usage=Usage(),
            tool_calls=tuple(tool_calls),
            rate_limit=rate_limit,
            num_api_turns=0,
            malformed_lines=malformed,
        )

    result_sid = _non_empty_str(result.get("session_id"))
    result_text = result.get("result")
    return StreamSummary(
        session_id=result_sid or session_id,
        has_result=True,
        is_error=result.get("is_error") is True,
        result_text=result_text if isinstance(result_text, str) else "".join(assistant_text),
        cost_usd=_turn_cost(result),
        usage=_usage(result),
        tool_calls=tuple(tool_calls),
        rate_limit=rate_limit,
        num_api_turns=_as_int(result.get("num_turns")) or 0,
        malformed_lines=malformed,
    )


def looks_rate_limited(summary: StreamSummary) -> bool:
    """True if the turn failed because the subscription/rate window is exhausted.

    Signals: a rate_limit_event whose status is not an "allowed*" status, or an error result
    whose text mentions a usage/rate limit (case-insensitive).
    """
    info = summary.rate_limit
    if info is not None and info.status and not info.status.startswith("allowed"):
        return True
    return summary.is_error and _LIMIT_TEXT.search(summary.result_text) is not None


# --- helpers ------------------------------------------------------------------------------------


def _non_empty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> int | None:
    """An int, or None for anything else (bool is deliberately not a count)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    """A finite number as a float, or None (NaN/inf would break strict JSON event logs)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _content_blocks(message_line: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = message_line.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _collect_block(
    block: Mapping[str, Any], tool_calls: list[ToolCall], assistant_text: list[str]
) -> None:
    block_type = block.get("type")
    if block_type == "tool_use":
        name = block.get("name")
        if isinstance(name, str):
            tool_calls.append(ToolCall(name=name, path=_tool_path(block.get("input"))))
    elif block_type == "text":
        value = block.get("text")
        if isinstance(value, str):
            assistant_text.append(value)


def _tool_path(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return None


def _parse_rate_limit(info: Any) -> RateLimitInfo | None:
    if not isinstance(info, dict):
        return None
    status = info.get("status")
    windows = info.get("unifiedWindows")
    if not isinstance(windows, dict):
        windows = {}
    return RateLimitInfo(
        status=status if isinstance(status, str) else "",
        five_hour_utilization=_utilization(windows.get("five_hour")),
        seven_day_utilization=_utilization(windows.get("seven_day")),
        resets_at=_as_float(info.get("resetsAt")),
    )


def _utilization(window: Any) -> float | None:
    return _as_float(window.get("utilization")) if isinstance(window, dict) else None


def _since_last_result(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = result.get("model_usage_since_last_result")
    return usage if isinstance(usage, dict) else None


def _turn_cost(result: Mapping[str, Any]) -> float:
    """Sum of per-model costs since the last result; `total_cost_usd` is cumulative on resume."""
    per_model = _since_last_result(result)
    if per_model is not None:  # present counts even when it sums to zero
        return math.fsum(
            _as_float(entry.get("costUSD")) or 0.0
            for entry in per_model.values()
            if isinstance(entry, dict)
        )
    return _as_float(result.get("total_cost_usd")) or 0.0


def _usage(result: Mapping[str, Any]) -> Usage:
    raw = result.get("usage")
    usage: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
    details = usage.get("output_tokens_details")
    thinking = _as_int(details.get("thinking_tokens")) if isinstance(details, dict) else None
    if thinking is None:
        per_model = _since_last_result(result) or {}
        thinking = sum(
            _as_int(entry.get("thinkingTokens")) or 0
            for entry in per_model.values()
            if isinstance(entry, dict)
        )
    return Usage(
        input_tokens=_as_int(usage.get("input_tokens")) or 0,
        output_tokens=_as_int(usage.get("output_tokens")) or 0,
        cache_read_tokens=_as_int(usage.get("cache_read_input_tokens")) or 0,
        cache_creation_tokens=_as_int(usage.get("cache_creation_input_tokens")) or 0,
        thinking_tokens=thinking,
    )
