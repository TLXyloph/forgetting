"""Tests for research.matrix: runner failures, interruptions and abnormal ends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research.matrix import run_matrix
from research.runs import EVENTS_FILE, is_run_complete

# ---------------------------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------------------------


def test_a_runner_exception_is_recorded_and_the_matrix_continues(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder, fail={("solo", 1)})

    records = run_matrix(experiment_file, runs_root=runs_root, runner=runner)

    assert [(r.arm, r.rep, r.status) for r in records] == [
        ("solo", 1, "failed"),
        ("solo", 2, "completed"),
        ("anchored", 1, "completed"),
        ("anchored", 2, "completed"),
    ]
    assert "boom in solo/rep1" in records[0].message
    assert "RuntimeError" in records[0].message
    assert len(runner.calls) == 4  # the failure did not stop the later runs


def test_a_failed_run_is_retried_by_the_next_invocation(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_matrix(
        experiment_file,
        runs_root=runs_root,
        replicates=1,
        runner=FakeRunner(run_builder, fail={("solo", 1)}),
    )

    retry = FakeRunner(run_builder)
    records = run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=retry)

    assert [(r.arm, r.status) for r in records] == [("solo", "completed"), ("anchored", "skipped")]
    assert retry.order == [("solo", "rep1")]
    assert any(".incomplete-" in p.name for p in (runs_root / "demo" / "solo").iterdir())


def test_an_exception_without_text_still_gets_a_message(
    experiment_file: Path, runs_root: Path
) -> None:
    def not_built_yet(*args: Any, **kwargs: Any) -> Path:
        raise NotImplementedError

    records = run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=not_built_yet)

    assert {r.status for r in records} == {"failed"}
    assert {r.message for r in records} == {"NotImplementedError"}


def test_a_runner_that_returns_without_finishing_counts_as_failed(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder, partial={("solo", 1)})

    records = run_matrix(
        experiment_file, runs_root=runs_root, replicates=1, arms=["solo"], runner=runner
    )

    assert records[0].status == "failed"
    assert "run_finished" in records[0].message


def test_a_runner_that_writes_no_log_counts_as_failed(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder, silent={("solo", 1)})

    records = run_matrix(
        experiment_file, runs_root=runs_root, replicates=1, arms=["solo"], runner=runner
    )

    assert records[0].status == "failed"
    assert EVENTS_FILE in records[0].message


@pytest.mark.parametrize("how", ["interrupt", "kill"])
def test_ctrl_c_stops_the_matrix_and_the_replicate_is_rerun_next_time(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type, how: str
) -> None:
    """`interrupt` is what the real harness does (a run_finished reading "interrupted");
    `kill` leaves a log that just stops."""
    runner = FakeRunner(run_builder, **{how: {("anchored", 1)}})

    with pytest.raises(KeyboardInterrupt):
        run_matrix(experiment_file, runs_root=runs_root, runner=runner)

    assert runner.order == [("solo", "rep1"), ("solo", "rep2"), ("anchored", "rep1")]
    # the next invocation resumes exactly where it stopped
    resumed = FakeRunner(run_builder)
    records = run_matrix(experiment_file, runs_root=runs_root, runner=resumed)
    assert resumed.order == [("anchored", "rep1"), ("anchored", "rep2")]
    assert [r.status for r in records] == ["skipped", "skipped", "completed", "completed"]
    parked = [p for p in (runs_root / "demo" / "anchored").iterdir() if ".incomplete-" in p.name]
    assert len(parked) == 1


def test_a_run_that_ended_as_interrupted_is_parked_and_rerun_never_skipped(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_dir = runs_root / "demo" / "solo" / "rep1"
    b = run_builder(run_dir, agents=("a1",)).start(100.0)
    b.candidate("a1", 150.0, cost=0.5)
    b.finish("interrupted")
    runner = FakeRunner(run_builder)

    records = run_matrix(experiment_file, runs_root=runs_root, arms=["solo"], runner=runner)

    assert runner.order[0] == ("solo", "rep1")
    assert records[0].status == "completed"
    assert is_run_complete(run_dir / EVENTS_FILE)  # the fresh run replaced it
    assert len([p for p in run_dir.parent.iterdir() if ".incomplete-" in p.name]) == 1


def test_a_run_that_gave_up_after_repeated_errors_counts_as_failed_with_the_reason(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    reason = "all agents stopped after repeated errors"
    runner = FakeRunner(run_builder, end_reason={("solo", 1): reason})

    records = run_matrix(
        experiment_file, runs_root=runs_root, replicates=1, arms=["solo"], runner=runner
    )

    assert records[0].status == "failed"
    assert reason in records[0].message


def test_a_run_that_exhausted_its_budget_is_completed(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(
        run_builder, end_reason={("solo", 1): "proposal budget exhausted (160 of 160)"}
    )
    records = run_matrix(
        experiment_file, runs_root=runs_root, replicates=1, arms=["solo"], runner=runner
    )
    assert records[0].status == "completed"
