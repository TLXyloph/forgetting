"""The judge output contract: how a user's scoring command reports a result.

A judge command reports through exactly one channel, the one its task declares in
`judge.output`:
  1. "stdout" (the default): the last non-empty line of stdout, either a JSON object or a bare
     number. No file is ever read, so code the judge runs cannot plant a verdict.
  2. "file": `$LETHE_OUT/result.json`, a JSON object (LETHE_OUT is set only in this mode, to a
     private directory). The runner passes its text as `result_json`, which then takes
     precedence over stdout; exiting 0 without writing it makes the candidate invalid.

JSON fields: `score` (finite number; required when valid), `valid` (bool, default true),
`feedback` (str, shown to agents), `metrics` (object of scalars), `cases` (object mapping
case name -> finite number, e.g. per-test-case scores), `private` (anything; ignored here).

Exit codes: 0 = use the reported result; 75 (EX_TEMPFAIL) = the judge itself broke, raise
`JudgeInfraError` so the runner retries; any other nonzero = the candidate is invalid, with the
tail of stderr/stdout as feedback (plus the parsed feedback, if any).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any

from lethe.domain.types import JudgeResult

EX_TEMPFAIL = 75

# Every feedback is at most `max_feedback_chars + len(_CUT_AFTER)`: a head-keeping cut appends
# `_CUT_AFTER`, a tail-keeping cut prepends `_CUT_BEFORE` (same length), and only one such
# marker can be added beyond the budget (see `_failure_feedback`).
_CUT_AFTER = "\n…[truncated]"
_CUT_BEFORE = "…[truncated]\n"
_OUTPUT_HEADER = "--- judge output (tail) ---\n"
_NO_OUTPUT = "(no output)"
_INFRA_MESSAGE_CHARS = 500
_ECHOED_LINE_CHARS = 200
# A bare score: plain decimal / scientific notation. Stricter than `float()`, which also takes
# "nan", "inf", "infinity", underscores ("1_000") and non-ASCII digits.
_BARE_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class JudgeInfraError(RuntimeError):
    """The judge reported its own failure (exit 75); the candidate must not be charged."""


class _ContractError(Exception):
    """The judge's output broke the contract; the message becomes the candidate's feedback."""


def parse_judge_output(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    result_json: str | None,
    max_feedback_chars: int = 4000,
) -> JudgeResult:
    """Turn raw judge process output into a `JudgeResult` (duration_s left at 0.0).

    Never raises for malformed output (that is an invalid candidate) except `JudgeInfraError`
    for exit code 75. Rejects NaN/inf scores and non-numeric `cases` values as invalid.
    Feedback is truncated to `max_feedback_chars` (keeping the tail for error output).

    Feedback stays within `max_feedback_chars` plus one truncation marker: reported feedback
    keeps its head (marker appended), error output keeps its tail (marker prepended).
    """
    limit = max(0, max_feedback_chars)
    if exit_code == EX_TEMPFAIL:
        raise JudgeInfraError(_infra_message(stdout, stderr))

    reported: JudgeResult | None = None
    problem = ""
    try:
        reported = _result_from_fields(_load_fields(stdout, result_json))
    except _ContractError as exc:
        problem = str(exc)

    if exit_code != 0:
        # A crashed judge's own verdict is not trusted; only its feedback text is kept.
        feedback = _failure_feedback(
            reported.feedback if reported is not None else "", _output_of(stdout, stderr), limit
        )
        return JudgeResult(
            valid=False,
            score=None,
            feedback=feedback,
            error=f"judge exited with code {exit_code}",
        )
    if reported is None:
        return JudgeResult(valid=False, score=None, feedback=_head(problem, limit))
    return replace(reported, feedback=_head(reported.feedback, limit))


# --- reading what the judge reported ---------------------------------------------------------


def _load_fields(stdout: str, result_json: str | None) -> dict[str, Any]:
    """The judge's reported fields: result.json if given, else the last stdout line."""
    if result_json is not None:
        try:
            data = json.loads(result_json)
        except (ValueError, RecursionError) as exc:
            raise _ContractError(f"result.json is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise _ContractError("result.json must contain a JSON object")
        return data

    line = _last_line(stdout)
    if line is None:
        raise _ContractError("judge printed nothing")
    if line.startswith("{"):
        try:
            data = json.loads(line)
        except (ValueError, RecursionError):
            raise _ContractError(_not_understood(line)) from None
        return data  # a document that starts with "{" can only be an object
    score = _bare_number(line)
    if score is None:
        raise _ContractError(_not_understood(line))
    return {"score": score}


def _last_line(text: str) -> str | None:
    # Split on "\n" only: str.splitlines() would also cut at U+2028, U+0085, ... which are
    # legal inside a JSON string.
    for line in reversed(text.split("\n")):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _not_understood(line: str) -> str:
    return f"judge output not understood; last line: {line[:_ECHOED_LINE_CHARS]}"


def _bare_number(line: str) -> float | None:
    if not _BARE_NUMBER.fullmatch(line):
        return None
    value = float(line)
    return value if math.isfinite(value) else None  # "1e999" overflows to inf


def _result_from_fields(fields: dict[str, Any]) -> JudgeResult:
    """Validate the reported fields. Feedback is returned untruncated."""
    valid = fields.get("valid", True)
    if not isinstance(valid, bool):
        raise _ContractError("valid must be a boolean")
    score: float | None = None
    if valid:
        score = _finite_number(fields.get("score"))
        if score is None:
            raise _ContractError("judge reported no finite score")
    try:
        cases = _cases(fields.get("cases"))
    except _ContractError:
        if valid:
            raise
        cases = None  # already invalid: do not let a cases typo hide the judge's own feedback
    return JudgeResult(
        valid=valid,
        score=score,
        feedback=_feedback_text(fields.get("feedback")),
        metrics=_scalar_metrics(fields.get("metrics")),
        cases=cases,
    )


def _finite_number(value: object) -> float | None:
    """`value` as a float if it is a real, finite JSON number; bools are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        return None
    return number if math.isfinite(number) else None


def _cases(raw: object) -> dict[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _ContractError("cases must map names to finite numbers")
    cases: dict[str, float] = {}
    for name, value in raw.items():
        number = _finite_number(value)
        if number is None:
            raise _ContractError("cases must map names to finite numbers")
        cases[name] = number
    return cases


def _scalar_metrics(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {name: value for name, value in raw.items() if _is_scalar(value)}


def _is_scalar(value: object) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _feedback_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw)
    except (TypeError, ValueError, RecursionError):
        return ""


# --- turning raw process output into feedback ----------------------------------------------------


def _output_of(stdout: str, stderr: str) -> str:
    """The text worth showing after a failure: stderr, or stdout when stderr is blank."""
    return stderr.strip() or stdout.strip()


def _output_tail(stdout: str, stderr: str, limit: int) -> str:
    """Failure output for feedback, tail-kept to `limit` chars (shared with the runner)."""
    return _tail(_output_of(stdout, stderr) or _NO_OUTPUT, limit)


def _infra_message(stdout: str, stderr: str) -> str:
    text = _output_of(stdout, stderr)[-_INFRA_MESSAGE_CHARS:].strip()
    return text or f"judge exited with code {EX_TEMPFAIL}"


def _head(text: str, limit: int) -> str:
    """The first `limit` chars, with a marker appended when something was cut."""
    return text if len(text) <= limit else text[:limit] + _CUT_AFTER


def _tail(text: str, limit: int) -> str:
    """The last `limit` chars, with a marker prepended when something was cut."""
    return text if len(text) <= limit else _CUT_BEFORE + text[len(text) - limit :]


def _failure_feedback(reported: str, output: str, limit: int) -> str:
    """`reported` (head kept) + the output header + `output` (tail kept), at most `limit` chars
    plus one marker. The reported text may use up to half the room, or more if the output is
    short; the output's tail takes the rest.
    """
    body = output or _NO_OUTPUT
    header = f"\n{_OUTPUT_HEADER}" if reported else _OUTPUT_HEADER
    whole = reported + header + body
    if len(whole) <= limit:
        return whole

    room = limit - len(header)
    if room >= 1:
        keep = min(len(reported), max(room // 2, room - len(_CUT_AFTER) - len(body)))
        first = _head(reported, keep)
        left = limit - len(first) - len(header)
        if left >= 1:
            return first + header + _tail(body, left)
    # Too small a budget for the header: just the most useful text.
    return _tail(output, limit) if output else _head(reported, limit)
