"""Unit tests for `parse_judge_output`: how stdout / result.json are interpreted."""

from __future__ import annotations

import json

import pytest

from lethe.domain.types import JudgeResult
from lethe.judge.contract import EX_TEMPFAIL, parse_judge_output

NO_SCORE = "judge reported no finite score"
BAD_CASES = "cases must map names to finite numbers"


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


# --- the happy paths -------------------------------------------------------------------------


def test_exit_tempfail_constant_is_75():
    assert EX_TEMPFAIL == 75


def test_json_line_with_every_field():
    payload = {
        "score": 0.5,
        "valid": True,
        "feedback": "nice work",
        "metrics": {"ms": 12, "label": "x", "ok": True, "gap": None, "ratio": 0.25},
        "cases": {"a": 1, "b": 2.5},
        "private": {"anything": [1, 2, 3]},
    }
    result = parse("compiling...\nrunning...\n" + json.dumps(payload) + "\n")
    assert result == JudgeResult(
        valid=True,
        score=0.5,
        feedback="nice work",
        metrics={"ms": 12, "label": "x", "ok": True, "gap": None, "ratio": 0.25},
        cases={"a": 1.0, "b": 2.5},
    )
    assert result.duration_s == 0.0
    assert result.error is None
    assert result.cache_hit is False


def test_integer_score_is_returned_as_float():
    result = parse('{"score": 3}')
    assert result.score == 3.0
    assert isinstance(result.score, float)
    assert result.valid is True


def test_valid_defaults_to_true_and_other_fields_default_to_empty():
    result = parse('{"score": 1.5}')
    assert result == JudgeResult(valid=True, score=1.5, feedback="", metrics={}, cases=None)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("0.75", 0.75),
        ("42", 42.0),
        ("-3.5e2", -350.0),
        ("+7", 7.0),
        ("5.", 5.0),
        (".5", 0.5),
        ("1E3", 1000.0),
        ("0", 0.0),
        ("  12.5  ", 12.5),
    ],
)
def test_bare_number_on_last_line(line, expected):
    result = parse(f"building\nlots of logs\n{line}\n")
    assert result.valid is True
    assert result.score == expected
    assert isinstance(result.score, float)
    assert result.feedback == ""
    assert result.metrics == {}
    assert result.cases is None


def test_only_the_last_non_empty_line_counts():
    stdout = '{"score": 1}\n{"score": 2}\n\n   \n\t\n'
    assert parse(stdout).score == 2.0
    assert parse('{"score": 1}\nnot a result\n').valid is False


def test_windows_line_endings_are_accepted():
    assert parse('log line\r\n{"score": 4}\r\n\r\n').score == 4.0
    assert parse("log line\r\n9.5\r\n").score == 9.5


def test_unicode_line_separators_inside_json_do_not_split_the_line():
    line = json.dumps({"score": 1, "feedback": "a b\x85c"}, ensure_ascii=False)
    result = parse("noise\n" + line)
    assert result.valid is True
    assert result.feedback == "a b\x85c"


# --- bare numbers that must be rejected ---------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["nan", "NaN", "inf", "-inf", "Infinity", "true", "false", "True", "1e999", "0x10", "1_000"]
    + ["１２", "5%", "1,5", "score: 3", "--1", "1e", "e5", "."],
)
def test_bare_non_numbers_are_not_understood(line):
    result = parse(f"log\n{line}\n")
    assert_invalid(result, f"judge output not understood; last line: {line}")


def test_unparseable_last_line_echoes_at_most_200_chars():
    line = "x" * 500
    result = parse(line)
    assert_invalid(result, "judge output not understood; last line: " + "x" * 200)


def test_json_line_that_is_broken_is_not_understood():
    result = parse('{"score": ')
    assert_invalid(result, 'judge output not understood; last line: {"score":')


def test_pretty_printed_json_is_not_understood_only_the_last_line_counts():
    result = parse('{\n  "score": 1\n}\n')
    assert_invalid(result, "judge output not understood; last line: }")


@pytest.mark.parametrize("stdout", ["", "\n", "  \n\t\n  "])
def test_empty_output_is_invalid(stdout):
    assert_invalid(parse(stdout), "judge printed nothing")


# --- result.json ---------------------------------------------------------------------------------


def test_result_json_takes_precedence_over_stdout():
    result = parse(
        '{"score": 1, "feedback": "stdout"}', result_json='{"score": 2, "feedback": "file"}'
    )
    assert result.score == 2.0
    assert result.feedback == "file"


def test_result_json_is_used_even_when_stdout_is_garbage():
    result = parse("total garbage", result_json='{"score": 8}')
    assert result.valid is True
    assert result.score == 8.0


def test_malformed_result_json_is_invalid_and_does_not_fall_back_to_stdout():
    result = parse('{"score": 99}', result_json="{not json")
    assert result.valid is False
    assert result.score is None
    assert result.feedback.startswith("result.json is not valid JSON: ")
    assert len(result.feedback) > len("result.json is not valid JSON: ")


def test_empty_result_json_is_malformed_not_absent():
    result = parse('{"score": 99}', result_json="")
    assert result.valid is False
    assert result.feedback.startswith("result.json is not valid JSON: ")


@pytest.mark.parametrize("text", ["[1, 2]", "0.5", '"score"', "null", "true"])
def test_result_json_must_be_an_object(text):
    assert_invalid(parse(result_json=text), "result.json must contain a JSON object")


# --- score validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"score": -Infinity}',
        '{"score": 1e999}',
        '{"score": -1e999}',
    ],
)
def test_non_finite_score_is_invalid_via_stdout_and_result_json(raw):
    assert_invalid(parse(raw), NO_SCORE)
    assert_invalid(parse(result_json=raw), NO_SCORE)


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": true}',
        '{"score": false}',
        '{"score": "0.5"}',
        '{"score": null}',
        '{"score": [1]}',
        '{"score": {"a": 1}}',
        "{}",
        '{"feedback": "forgot the score"}',
    ],
)
def test_non_numeric_or_missing_score_is_invalid(raw):
    assert_invalid(parse(raw), NO_SCORE)


def test_huge_integer_score_does_not_raise():
    assert_invalid(parse('{"score": ' + "9" * 400 + "}"), NO_SCORE)


# --- valid: false ------------------------------------------------------------------------------


def test_valid_false_without_score_is_a_normal_invalid_result():
    result = parse('{"valid": false, "feedback": "does not compile"}')
    assert result == JudgeResult(valid=False, score=None, feedback="does not compile")


def test_valid_false_forces_the_score_to_none():
    result = parse('{"valid": false, "score": 0.9, "feedback": "cheated"}')
    assert_invalid(result, "cheated")


def test_valid_false_does_not_require_a_finite_score():
    assert_invalid(parse('{"valid": false, "score": NaN}'))


def test_valid_false_keeps_metrics_and_cases():
    result = parse('{"valid": false, "metrics": {"n": 3}, "cases": {"a": 1}}')
    assert result.metrics == {"n": 3}
    assert result.cases == {"a": 1.0}


@pytest.mark.parametrize("bad", ['"yes"', '"true"', "1", "0", "null", "[]", "{}"])
def test_valid_must_be_a_boolean(bad):
    assert_invalid(parse('{"score": 1, "valid": ' + bad + "}"), "valid must be a boolean")


# --- cases -----------------------------------------------------------------------------------


def test_cases_values_are_converted_to_floats():
    result = parse('{"score": 1, "cases": {"easy": 1, "hard": 0.25}}')
    assert result.cases == {"easy": 1.0, "hard": 0.25}
    assert all(isinstance(v, float) for v in result.cases.values())


@pytest.mark.parametrize(
    "bad",
    [
        "[1, 2]",
        '"abc"',
        "5",
        "true",
        '{"a": "x"}',
        '{"a": true}',
        '{"a": false}',
        '{"a": null}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": [1]}',
        '{"a": {"b": 1}}',
        '{"a": 1, "b": "2"}',
    ],
)
def test_malformed_cases_make_the_result_invalid(bad):
    assert_invalid(parse('{"score": 1, "cases": ' + bad + "}"), BAD_CASES)


def test_empty_cases_object_is_kept_and_null_cases_means_absent():
    assert parse('{"score": 1, "cases": {}}').cases == {}
    assert parse('{"score": 1, "cases": null}').cases is None


def test_malformed_cases_do_not_hide_the_judges_feedback_when_already_invalid():
    result = parse('{"valid": false, "feedback": "syntax error on line 3", "cases": "n/a"}')
    assert_invalid(result, "syntax error on line 3")
    assert result.cases is None


# --- metrics -----------------------------------------------------------------------------------


def test_metrics_keep_only_scalar_values():
    metrics = {
        "s": "text",
        "t": True,
        "f": False,
        "i": 3,
        "x": 0.5,
        "n": None,
        "nested": {"a": 1},
        "listy": [1, 2],
        "nan": float("nan"),
        "inf": float("inf"),
    }
    raw = json.dumps({"score": 1, "metrics": metrics})  # json.dumps emits NaN / Infinity
    assert "NaN" in raw
    result = parse(raw)
    assert result.metrics == {"s": "text", "t": True, "f": False, "i": 3, "x": 0.5, "n": None}


@pytest.mark.parametrize("bad", ["[1]", '"text"', "5", "null", "true"])
def test_non_object_metrics_become_empty(bad):
    result = parse('{"score": 1, "metrics": ' + bad + "}")
    assert result.valid is True
    assert result.metrics == {}


# --- feedback --------------------------------------------------------------------------------


def test_non_string_feedback_is_json_encoded():
    assert parse('{"score": 1, "feedback": [1, "a"]}').feedback == '[1, "a"]'
    assert parse('{"score": 1, "feedback": {"k": 2}}').feedback == '{"k": 2}'
    assert parse('{"score": 1, "feedback": 42}').feedback == "42"
    assert parse('{"score": 1, "feedback": true}').feedback == "true"


def test_null_feedback_means_no_feedback():
    assert parse('{"score": 1, "feedback": null}').feedback == ""


# --- never raises on garbage ------------------------------------------------------------------


def test_pathological_json_is_invalid_not_an_exception():
    deep = '{"score": 1, "x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    assert parse(deep).valid is False
    assert parse(result_json=deep).valid is False
    long_int = '{"score": 1, "m": ' + "7" * 10_000 + "}"
    assert parse(long_int).valid is False
    assert parse(result_json=long_int).valid is False
    assert parse("\x00\x01\x02 \ud800 binary\xff").valid is False
