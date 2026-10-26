"""Tests for research.summary: unreadable and unfinished logs, best_at and load_summaries."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lethe.domain.events import Envelope, RateLimited
from research.summary import (
    RunSummary,
    best_at,
    load_summaries,
    summarize_events,
    summarize_run,
)

# ---------------------------------------------------------------------------------------------
# unfinished and malformed logs
# ---------------------------------------------------------------------------------------------


def test_an_unfinished_run_is_summarized_as_far_as_it_got(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)

    s = summarize_run(tmp_path / "r", arm="x", rep=1)

    assert s.finished is False
    assert s.final_best == 110.0
    assert s.spend_usd == 0.5


def test_the_finish_reason_is_recorded(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    s = summarize_run(b.finish("usd budget exhausted ($3.02 of $3.00)"), arm="x", rep=1)

    assert s.finished is True
    assert s.finish_reason == "usd budget exhausted ($3.02 of $3.00)"


@pytest.mark.parametrize("reason", ["interrupted", "all agents stopped after repeated errors"])
def test_a_run_that_ended_abnormally_is_not_finished(
    tmp_path: Path, run_builder: type, reason: str
) -> None:
    """The harness writes run_finished on Ctrl-C too; such a run did not use its budget."""
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    s = summarize_run(b.finish(reason), arm="x", rep=1)

    assert s.finished is False
    assert s.finish_reason == reason
    assert s.final_best == 110.0  # the partial result is still summarized


def test_a_run_that_is_still_going_has_no_finish_reason(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    s = summarize_run(b.run_dir, arm="x", rep=1)
    assert (s.finished, s.finish_reason) == (False, None)


def test_a_log_without_run_started_is_rejected(tmp_path: Path) -> None:
    from lethe.store import EventStore

    store = EventStore(tmp_path / "r" / "events.jsonl")
    store.append(RateLimited(agent="a1", resets_at=None, message="x"))
    with pytest.raises(ValueError, match="run_started"):
        summarize_run(tmp_path / "r", arm="x", rep=1)


def test_an_empty_log_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "events.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="run_started"):
        summarize_run(tmp_path / "r", arm="x", rep=1)


def test_a_missing_log_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_run(tmp_path / "nothing", arm="x", rep=1)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"budget": {"usd": 3.0}}, 3.0),
        ({"budget": {"usd": 5}}, 5.0),
        ({"budget": {"usd": 0, "proposals": 160, "primary": "proposals"}}, None),
        ({"budget": {"proposals": 160}}, None),
        ({"budget": "oops"}, None),
        ({"budget": {"usd": "3"}}, None),
        ({"budget": {"usd": True}}, None),
        ({}, None),
    ],
)
def test_usd_budget_is_read_defensively_from_the_run_config(
    tmp_path: Path, run_builder: type, config: dict, expected: float | None
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",), config=config).start(100.0)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.budget_usd == expected


def test_run_summary_is_immutable(tmp_path: Path, run_builder: type) -> None:
    import dataclasses

    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert isinstance(s, RunSummary)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.final_best = 1.0  # type: ignore[misc]


def test_summarize_events_requires_a_run_started_event() -> None:
    envelope = Envelope(seq=0, ts=1.0, event=RateLimited(agent="a", resets_at=None, message="m"))
    with pytest.raises(ValueError, match="run_started"):
        summarize_events([envelope], arm="x", rep=1, run_dir=Path("."))


# ---------------------------------------------------------------------------------------------
# best_at
# ---------------------------------------------------------------------------------------------

STEP = ((0.0, 100.0), (1.0, 120.0), (3.0, 150.0))


@pytest.mark.parametrize(
    ("spend", "expected"),
    [
        (-1.0, None),
        (0.0, 100.0),
        (0.99, 100.0),
        (1.0, 120.0),
        (2.5, 120.0),
        (3.0, 150.0),
        (99.0, 150.0),
    ],
)
def test_best_at_reads_the_step_function(spend: float, expected: float | None) -> None:
    assert best_at(STEP, spend) == expected


def test_best_at_of_an_empty_curve_is_none() -> None:
    assert best_at((), 1.0) is None


def test_best_at_before_the_first_valid_score_is_none() -> None:
    assert best_at(((0.5, 50.0),), 0.25) is None


# ---------------------------------------------------------------------------------------------
# load_summaries
# ---------------------------------------------------------------------------------------------


def log_run(
    run_builder: type, exp: Path, arm: str, rep: int, *, t0: float, finish: bool = True
) -> Path:
    b = run_builder(exp / arm / f"rep{rep}", agents=("a1",), t0=t0).start(100.0)
    b.candidate("a1", 100.0 + rep, cost=0.5)
    return b.finish() if finish else b.run_dir


def test_load_summaries_orders_arms_by_when_they_first_ran(
    tmp_path: Path, run_builder: type
) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "zeta", 1, t0=100.0)
    log_run(run_builder, exp, "alpha", 1, t0=900.0)
    log_run(run_builder, exp, "alpha", 2, t0=50.0)  # alpha's earliest run predates zeta's
    log_run(run_builder, exp, "mid", 1, t0=300.0)
    log_run(run_builder, exp, "zeta", 2, t0=400.0)

    summaries = load_summaries(exp)

    assert [(s.arm, s.rep) for s in summaries] == [
        ("alpha", 1),
        ("alpha", 2),
        ("zeta", 1),
        ("zeta", 2),
        ("mid", 1),
    ]


def test_load_summaries_breaks_start_time_ties_alphabetically(
    tmp_path: Path, run_builder: type
) -> None:
    exp = tmp_path / "exp"
    for arm in ("b", "a", "c"):
        log_run(run_builder, exp, arm, 1, t0=10.0)
    assert [s.arm for s in load_summaries(exp)] == ["a", "b", "c"]


def test_load_summaries_excludes_unfinished_runs_and_says_so(
    tmp_path: Path, run_builder: type, caplog: pytest.LogCaptureFixture
) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "solo", 1, t0=1.0)
    log_run(run_builder, exp, "solo", 2, t0=2.0, finish=False)
    log_run(run_builder, exp, "team", 1, t0=3.0, finish=False)

    with caplog.at_level(logging.WARNING, logger="research.summary"):
        summaries = load_summaries(exp)

    assert [(s.arm, s.rep) for s in summaries] == [("solo", 1)]
    assert "solo/rep2" in caplog.text
    assert "team/rep1" in caplog.text
    assert "include-unfinished" in caplog.text


def test_load_summaries_says_why_a_run_was_excluded(
    tmp_path: Path, run_builder: type, caplog: pytest.LogCaptureFixture
) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "solo", 1, t0=1.0)
    b = run_builder(exp / "solo" / "rep2", agents=("a1",), t0=2.0).start(100.0)
    b.candidate("a1", 150.0, cost=0.5)
    b.finish("interrupted")  # Ctrl-C: the log ends with run_finished, yet the run is incomplete

    with caplog.at_level(logging.WARNING, logger="research.summary"):
        summaries = load_summaries(exp)

    assert [(s.arm, s.rep) for s in summaries] == [("solo", 1)]
    assert "solo/rep2 (interrupted)" in caplog.text


def test_load_summaries_can_include_unfinished_runs(tmp_path: Path, run_builder: type) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "solo", 1, t0=1.0)
    log_run(run_builder, exp, "solo", 2, t0=2.0, finish=False)

    summaries = load_summaries(exp, include_unfinished=True)

    assert [(s.rep, s.finished) for s in summaries] == [(1, True), (2, False)]


def test_load_summaries_skips_unreadable_logs_with_a_warning(
    tmp_path: Path, run_builder: type, caplog: pytest.LogCaptureFixture
) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "solo", 1, t0=1.0)
    bad = exp / "solo" / "rep2"
    bad.mkdir()
    (bad / "events.jsonl").write_text('{"not": "an event"}\n{"also": "bad"}\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="research.summary"):
        summaries = load_summaries(exp)

    assert [(s.arm, s.rep) for s in summaries] == [("solo", 1)]
    assert "solo/rep2" in caplog.text


def test_load_summaries_ignores_parked_runs(tmp_path: Path, run_builder: type) -> None:
    exp = tmp_path / "exp"
    log_run(run_builder, exp, "solo", 1, t0=1.0)
    parked = exp / "solo" / "rep1.incomplete-20260101T000000Z"
    run_builder(parked, agents=("a1",)).start(100.0)

    assert [(s.arm, s.rep) for s in load_summaries(exp, include_unfinished=True)] == [("solo", 1)]


def test_load_summaries_of_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_summaries(tmp_path / "nope")


def test_load_summaries_of_an_empty_experiment_is_empty(tmp_path: Path) -> None:
    (tmp_path / "exp").mkdir()
    assert load_summaries(tmp_path / "exp") == []
