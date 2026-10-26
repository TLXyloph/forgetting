"""Unit tests for `parse_judge_output`: exit codes, infrastructure errors, feedback truncation."""

from __future__ import annotations

import json
import random

import pytest

from lethe.domain.types import JudgeResult
from lethe.judge.contract import EX_TEMPFAIL, JudgeInfraError, parse_judge_output

MARKER = "\n…[truncated]"  # appended where the head of a text was kept
HEADER = "--- judge output (tail) ---\n"


def parse(
    stdout: str = "",
    *,
    stderr: str = "",
    exit_code: int = 0,
    result_json: str | None = None,
    **kwargs: int,
) -> JudgeResult:
    return parse_judge_output(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        result_json=result_json,
        **kwargs,
    )


def assert_invalid(result: JudgeResult, feedback: str | None = None) -> None:
    assert result.valid is False
    assert result.score is None
    if feedback is not None:
        assert result.feedback == feedback


# --- exit codes -----------------------------------------------------------------------------


def test_exit_75_raises_infra_error_with_the_stderr_tail():
    stderr = "x" * 1000 + "scratch disk full\n"
    with pytest.raises(JudgeInfraError) as info:
        parse('{"score": 1}', stderr=stderr, exit_code=EX_TEMPFAIL, result_json='{"score": 2}')
    message = str(info.value)
    assert message.endswith("scratch disk full")
    assert len(message) <= 500


def test_exit_75_without_stderr_still_has_a_message():
    with pytest.raises(JudgeInfraError, match="75"):
        parse(exit_code=75)
    with pytest.raises(JudgeInfraError, match="gpu lost"):
        parse("gpu lost\n", exit_code=75)


def test_infra_error_is_a_runtime_error():
    assert issubclass(JudgeInfraError, RuntimeError)


def test_nonzero_exit_is_invalid_with_error_and_the_output_tail():
    result = parse(
        '{"score": 3, "feedback": "partial credit"}',
        stderr="Traceback (most recent call last):\nValueError: boom\n",
        exit_code=2,
    )
    assert result.valid is False
    assert result.score is None
    assert result.error == "judge exited with code 2"
    assert result.feedback == (
        "partial credit\n" + HEADER + "Traceback (most recent call last):\nValueError: boom"
    )


def test_nonzero_exit_without_parsed_feedback_has_just_the_tail():
    result = parse("progress...\n", stderr="fatal: boom\n", exit_code=1)
    assert result.error == "judge exited with code 1"
    assert result.feedback == HEADER + "fatal: boom"


def test_nonzero_exit_falls_back_to_stdout_when_stderr_is_blank():
    result = parse("oops from stdout\n", stderr="  \n", exit_code=3)
    assert result.feedback == HEADER + "oops from stdout"


def test_nonzero_exit_ignores_contract_problems_in_the_output():
    result = parse("garbage", stderr="real problem", exit_code=1, result_json="{broken")
    assert result.feedback == HEADER + "real problem"


def test_nonzero_exit_with_no_output_at_all():
    result = parse(exit_code=9)
    assert result.valid is False
    assert result.error == "judge exited with code 9"
    assert result.feedback == HEADER + "(no output)"


def test_negative_exit_code_means_killed_by_a_signal():
    result = parse(stderr="Killed", exit_code=-9)
    assert result.error == "judge exited with code -9"
    assert result.valid is False


def test_nonzero_exit_overrides_a_valid_looking_result_json():
    result = parse(stderr="crash", exit_code=1, result_json='{"score": 100}')
    assert_invalid(result)
    assert result.error == "judge exited with code 1"


# --- truncation -----------------------------------------------------------------------------------


def test_long_feedback_keeps_the_head_and_marks_the_cut():
    feedback = "A" * 50 + "Z" * 50
    result = parse(json.dumps({"score": 1, "feedback": feedback}), max_feedback_chars=50)
    assert result.feedback == "A" * 50 + MARKER
    assert len(result.feedback) == 50 + len(MARKER)


def test_feedback_at_the_limit_is_untouched():
    result = parse(json.dumps({"score": 1, "feedback": "B" * 50}), max_feedback_chars=50)
    assert result.feedback == "B" * 50


def test_default_limit_is_4000_chars():
    result = parse(json.dumps({"score": 1, "feedback": "C" * 5000}))
    assert result.feedback == "C" * 4000 + MARKER


def test_contract_problem_messages_are_bounded_too():
    result = parse("y" * 1000, max_feedback_chars=30)
    assert len(result.feedback) <= 30 + len(MARKER)


def test_error_output_keeps_the_tail():
    result = parse(stderr="x" * 5000 + "FINAL ERROR LINE", exit_code=1, max_feedback_chars=100)
    assert result.feedback.endswith("FINAL ERROR LINE")
    assert "x" * 100 not in result.feedback
    assert len(result.feedback) <= 100 + len(MARKER)
    assert result.feedback.startswith(HEADER)


def test_parsed_feedback_keeps_its_head_and_the_output_keeps_its_tail():
    reported = "HEAD-" + "r" * 3000 + "-LOST"
    output = "-LOST-TOO" + "o" * 3000 + "-TAIL"
    stdout = json.dumps({"valid": False, "feedback": reported})
    result = parse(stdout, stderr=output, exit_code=1, max_feedback_chars=400)
    assert result.feedback.startswith("HEAD-")
    assert result.feedback.endswith("-TAIL")
    assert "-LOST" not in result.feedback
    assert HEADER in result.feedback
    assert len(result.feedback) <= 400 + len(MARKER)


def test_short_output_is_kept_whole_when_the_parsed_feedback_is_long():
    reported = "R" * 3000
    result = parse(
        json.dumps({"valid": False, "feedback": reported}),
        stderr="the whole stack trace",
        exit_code=1,
        max_feedback_chars=500,
    )
    assert result.feedback.endswith(HEADER + "the whole stack trace")
    assert len(result.feedback) <= 500 + len(MARKER)


def test_failure_feedback_never_exceeds_limit_plus_marker():
    rng = random.Random(1234)
    for _ in range(400):
        limit = rng.choice([0, 1, 2, 5, 13, 20, 28, 29, 40, 60, 100, 500])
        reported = "F" * rng.choice([0, 1, 3, 50, 700])
        output = "".join(rng.choice("abc \n") for _ in range(rng.choice([0, 1, 10, 400, 3000])))
        stdout = json.dumps({"valid": False, "feedback": reported})
        stderr = output if rng.random() < 0.7 else ""
        result = parse(
            stdout if rng.random() < 0.7 else "",
            stderr=stderr,
            exit_code=rng.choice([1, 2, -9]),
            max_feedback_chars=limit,
        )
        assert len(result.feedback) <= limit + len(MARKER), (limit, reported, output)
        assert result.valid is False
        assert result.error is not None


def test_success_feedback_never_exceeds_limit_plus_marker():
    rng = random.Random(99)
    for _ in range(200):
        limit = rng.choice([0, 1, 7, 50])
        feedback = "q" * rng.choice([0, 1, 7, 8, 500])
        result = parse(json.dumps({"score": 1, "feedback": feedback}), max_feedback_chars=limit)
        assert len(result.feedback) <= limit + len(MARKER)
        assert result.feedback.startswith(feedback[:limit])


def test_negative_limit_is_treated_as_zero():
    result = parse(json.dumps({"score": 1, "feedback": "abcdefghij"}), max_feedback_chars=-3)
    assert result.feedback == MARKER
    failed = parse(stderr="abcdefghij" * 10, exit_code=1, max_feedback_chars=-3)
    assert len(failed.feedback) <= len(MARKER)
