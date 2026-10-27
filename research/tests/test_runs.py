"""Tests for research.runs: where runs live and which of them are finished."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.runs import (
    EVENTS_FILE,
    RunLocation,
    arm_dir_for,
    discover_runs,
    ending_is_complete,
    find_rep_dirs,
    finish_reason,
    is_run_complete,
    next_free_rep,
    rep_name,
    run_dir_for,
)


def finished_log(builder: type, run_dir: Path, reason: str = "usd budget exhausted") -> Path:
    b = builder(run_dir).start(100.0)
    b.candidate("a1", 110.0)
    return b.finish(reason)


def partial_log(builder: type, run_dir: Path) -> Path:
    b = builder(run_dir).start(100.0)
    b.candidate("a1", 110.0)
    return run_dir


# ---------------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------------


def test_rep_name() -> None:
    assert rep_name(1) == "rep1"
    assert rep_name(12) == "rep12"


def test_run_dir_layout(tmp_path: Path) -> None:
    assert run_dir_for(tmp_path, "exp", "solo", 2) == tmp_path / "exp" / "solo" / "rep2"


@pytest.mark.parametrize(
    ("experiment", "arm", "rep"),
    [
        ("..", "solo", 1),
        ("exp", "..", 1),
        ("exp", "a/b", 1),
        ("exp", "", 1),
        ("../escape", "solo", 1),
        ("exp", "solo", 0),
        ("exp", "solo", -3),
    ],
)
def test_run_dir_rejects_unsafe_components(
    tmp_path: Path, experiment: str, arm: str, rep: int
) -> None:
    with pytest.raises(ValueError):
        run_dir_for(tmp_path, experiment, arm, rep)


# ---------------------------------------------------------------------------------------------
# is_run_complete / finish_reason
# ---------------------------------------------------------------------------------------------


def test_finished_log_is_complete(tmp_path: Path, run_builder: type) -> None:
    run_dir = finished_log(run_builder, tmp_path / "r")
    assert is_run_complete(run_dir / EVENTS_FILE)


def test_log_without_run_finished_is_not_complete(tmp_path: Path, run_builder: type) -> None:
    run_dir = partial_log(run_builder, tmp_path / "r")
    assert not is_run_complete(run_dir / EVENTS_FILE)
    assert finish_reason(run_dir / EVENTS_FILE) is None


def test_missing_log_is_not_complete(tmp_path: Path) -> None:
    assert not is_run_complete(tmp_path / "nowhere" / EVENTS_FILE)
    assert finish_reason(tmp_path / "nowhere" / EVENTS_FILE) is None


def test_empty_log_is_not_complete(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILE
    path.write_text("", encoding="utf-8")
    assert not is_run_complete(path)


def test_the_finish_reason_is_read_from_the_log(tmp_path: Path, run_builder: type) -> None:
    reason = "proposal budget exhausted (160 of 160)"
    run_dir = finished_log(run_builder, tmp_path / "r", reason)
    assert finish_reason(run_dir / EVENTS_FILE) == reason


@pytest.mark.parametrize(
    ("reason", "complete"),
    [
        ("usd budget exhausted ($3.02 of $3.00)", True),
        ("proposal budget exhausted (160 of 160)", True),
        ("wall-clock budget exhausted (3600s of 3600s)", True),
        ("finished", True),
        ("budget", True),
        ("something the harness may say in the future", True),
        # the harness writes run_finished on Ctrl-C and when every agent gave up: neither
        # used its budget, so neither is a result to compare or to skip on resume
        ("interrupted", False),
        ("Interrupted by user", False),
        ("all agents stopped after repeated errors", False),
    ],
)
def test_runs_that_ended_abnormally_are_not_complete(
    tmp_path: Path, run_builder: type, reason: str, complete: bool
) -> None:
    run_dir = finished_log(run_builder, tmp_path / "r", reason)
    assert is_run_complete(run_dir / EVENTS_FILE) is complete
    assert finish_reason(run_dir / EVENTS_FILE) == reason  # still reported either way


@pytest.mark.parametrize(
    ("status", "complete"),
    [("completed", True), ("interrupted", False), ("crashed", False), ("stopped", False)],
)
def test_the_logged_status_decides_whether_a_run_ended_normally(
    tmp_path: Path, run_builder: type, status: str, complete: bool
) -> None:
    run_dir = run_builder(tmp_path / "r").start(100.0).finish("budget exhausted", status=status)
    assert is_run_complete(run_dir / EVENTS_FILE) is complete
    assert ending_is_complete(status, "budget exhausted") is complete


def test_logs_without_a_status_fall_back_to_the_reason(tmp_path: Path, run_builder: type) -> None:
    # logs written before RunFinished had a status decode it as "completed"
    run_dir = run_builder(tmp_path / "r").start(100.0).finish("interrupted")
    assert is_run_complete(run_dir / EVENTS_FILE) is False
    assert ending_is_complete("completed", "all agents stopped after repeated errors") is False
    assert ending_is_complete("crashed", "crashed: RuntimeError: boom") is False


def test_torn_final_line_means_the_run_did_not_finish(tmp_path: Path, run_builder: type) -> None:
    run_dir = finished_log(run_builder, tmp_path / "r")
    path = run_dir / EVENTS_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-10], encoding="utf-8")  # cut into the final run_finished line
    assert not is_run_complete(path)


def test_corrupt_log_counts_as_unfinished_instead_of_raising(
    tmp_path: Path, run_builder: type
) -> None:
    run_dir = finished_log(run_builder, tmp_path / "r")
    path = run_dir / EVENTS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = "{this is not json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not is_run_complete(path)
    assert finish_reason(path) is None


# ---------------------------------------------------------------------------------------------
# rep directories
# ---------------------------------------------------------------------------------------------


def test_find_rep_dirs_ignores_parked_and_foreign_entries(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    for name in ("rep1", "rep3", "rep10", "rep2.incomplete-20260101T000000Z", "repx", "rep", "x"):
        (arm / name).mkdir(parents=True)
    (arm / "rep4").write_text("a file, not a run", encoding="utf-8")

    found = find_rep_dirs(arm)

    assert found == {1: arm / "rep1", 3: arm / "rep3", 10: arm / "rep10"}


def test_find_rep_dirs_of_a_missing_arm_is_empty(tmp_path: Path) -> None:
    assert find_rep_dirs(tmp_path / "missing") == {}


def test_next_free_rep_starts_at_one(tmp_path: Path) -> None:
    assert next_free_rep(tmp_path / "missing") == 1
    (tmp_path / "empty").mkdir()
    assert next_free_rep(tmp_path / "empty") == 1


def test_next_free_rep_follows_the_existing_ones(tmp_path: Path) -> None:
    for k in (1, 2, 3):
        (tmp_path / f"rep{k}").mkdir()
    assert next_free_rep(tmp_path) == 4


def test_next_free_rep_fills_the_smallest_gap(tmp_path: Path) -> None:
    for k in (1, 3):
        (tmp_path / f"rep{k}").mkdir()
    assert next_free_rep(tmp_path) == 2


def test_a_parked_directory_does_not_occupy_its_slot(tmp_path: Path) -> None:
    (tmp_path / "rep1.incomplete-20260101T000000Z").mkdir()
    assert next_free_rep(tmp_path) == 1


def test_partial_runs_still_occupy_their_slot(tmp_path: Path, run_builder: type) -> None:
    partial_log(run_builder, tmp_path / "rep1")
    assert next_free_rep(tmp_path) == 2


# ---------------------------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------------------------


def test_discover_runs_finds_every_run_with_a_log(tmp_path: Path, run_builder: type) -> None:
    exp = tmp_path / "exp"
    finished_log(run_builder, exp / "solo" / "rep1")
    finished_log(run_builder, exp / "solo" / "rep10")
    finished_log(run_builder, exp / "solo" / "rep2")
    partial_log(run_builder, exp / "team" / "rep1")

    runs = discover_runs(exp)

    assert runs == [
        RunLocation("solo", 1, exp / "solo" / "rep1"),
        RunLocation("solo", 2, exp / "solo" / "rep2"),
        RunLocation("solo", 10, exp / "solo" / "rep10"),
        RunLocation("team", 1, exp / "team" / "rep1"),
    ]


def test_discover_runs_skips_noise(tmp_path: Path, run_builder: type) -> None:
    exp = tmp_path / "exp"
    finished_log(run_builder, exp / "solo" / "rep1")
    finished_log(run_builder, exp / "solo" / "rep1.incomplete-20260101T000000Z")
    (exp / "solo" / "rep2").mkdir()  # a directory without a log is not a run
    (exp / "analysis").mkdir()  # an output directory
    (exp / "analysis" / "table1.csv").write_text("x", encoding="utf-8")
    (exp / "notes.txt").write_text("hello", encoding="utf-8")

    assert [(r.arm, r.rep) for r in discover_runs(exp)] == [("solo", 1)]


def test_discover_runs_of_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_runs(tmp_path / "nope")


def test_arm_dir_layout_and_safety(tmp_path: Path) -> None:
    assert arm_dir_for(tmp_path, "exp", "solo") == tmp_path / "exp" / "solo"
    with pytest.raises(ValueError):
        arm_dir_for(tmp_path, "exp", "..")
    with pytest.raises(ValueError):
        arm_dir_for(tmp_path, "a/b", "solo")
