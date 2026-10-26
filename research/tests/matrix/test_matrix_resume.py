"""Tests for research.matrix: skipping finished runs and parking unfinished ones."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from research import matrix
from research.matrix import park_incomplete, run_matrix
from research.runs import EVENTS_FILE, is_run_complete


def finished(builder: type, run_dir: Path) -> Path:
    b = builder(run_dir, agents=("a1",)).start(100.0)
    b.candidate("a1", 105.0)
    return b.finish()


def partial(builder: type, run_dir: Path) -> Path:
    b = builder(run_dir, agents=("a1",)).start(100.0)
    b.candidate("a1", 105.0)
    return run_dir


# ---------------------------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------------------------


def test_finished_runs_are_skipped(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    done = finished(run_builder, runs_root / "demo" / "solo" / "rep1")
    before = (done / EVENTS_FILE).read_text(encoding="utf-8")
    runner = FakeRunner(run_builder)

    records = run_matrix(experiment_file, runs_root=runs_root, runner=runner)

    assert [(r.arm, r.rep, r.status) for r in records] == [
        ("solo", 1, "skipped"),
        ("solo", 2, "completed"),
        ("anchored", 1, "completed"),
        ("anchored", 2, "completed"),
    ]
    assert ("solo", "rep1") not in runner.order
    assert (done / EVENTS_FILE).read_text(encoding="utf-8") == before  # untouched


def test_a_second_invocation_does_nothing(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_matrix(experiment_file, runs_root=runs_root, runner=FakeRunner(run_builder))

    again = FakeRunner(run_builder)
    records = run_matrix(experiment_file, runs_root=runs_root, runner=again)

    assert again.calls == []
    assert {r.status for r in records} == {"skipped"}


def test_raising_replicates_runs_only_the_new_ones(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_matrix(experiment_file, runs_root=runs_root, runner=FakeRunner(run_builder))

    more = FakeRunner(run_builder)
    records = run_matrix(experiment_file, runs_root=runs_root, replicates=3, runner=more)

    assert more.order == [("solo", "rep3"), ("anchored", "rep3")]
    assert [r.status for r in records].count("skipped") == 4


def test_a_partial_run_is_renamed_not_deleted_and_rerun(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_dir = partial(run_builder, runs_root / "demo" / "solo" / "rep1")
    (run_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    old_events = (run_dir / EVENTS_FILE).read_text(encoding="utf-8")
    runner = FakeRunner(run_builder)

    records = run_matrix(experiment_file, runs_root=runs_root, arms=["solo"], runner=runner)

    assert records[0].status == "completed"
    assert is_run_complete(run_dir / EVENTS_FILE)  # the fresh run lives at rep1
    parked = [p for p in run_dir.parent.iterdir() if p.name.startswith("rep1.incomplete-")]
    assert len(parked) == 1
    assert re.fullmatch(r"rep1\.incomplete-\d{8}T\d{6}Z", parked[0].name)
    assert (parked[0] / EVENTS_FILE).read_text(encoding="utf-8") == old_events
    assert (parked[0] / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_the_parked_name_uses_the_utc_timestamp(
    FakeRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matrix, "_utc_stamp", lambda: "20261026T163412Z")
    partial(run_builder, runs_root / "demo" / "solo" / "rep1")

    run_matrix(experiment_file, runs_root=runs_root, arms=["solo"], runner=FakeRunner(run_builder))

    assert (
        runs_root / "demo" / "solo" / "rep1.incomplete-20261026T163412Z" / EVENTS_FILE
    ).is_file()


def test_parking_never_overwrites_an_earlier_parked_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_builder: type
) -> None:
    monkeypatch.setattr(matrix, "_utc_stamp", lambda: "20261026T163412Z")
    arm_dir = tmp_path / "arm"
    for marker in ("first", "second", "third"):
        run_dir = arm_dir / "rep1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "marker.txt").write_text(marker, encoding="utf-8")
        park_incomplete(run_dir)

    parked = {p.name: (p / "marker.txt").read_text(encoding="utf-8") for p in arm_dir.iterdir()}
    assert parked == {
        "rep1.incomplete-20261026T163412Z": "first",
        "rep1.incomplete-20261026T163412Z-2": "second",
        "rep1.incomplete-20261026T163412Z-3": "third",
    }


def test_a_directory_with_leftovers_but_no_events_is_parked_too(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    leftovers = runs_root / "demo" / "solo" / "rep1"
    (leftovers / "workspaces").mkdir(parents=True)
    (leftovers / "workspaces" / "a1.txt").write_text("stale", encoding="utf-8")

    run_matrix(
        experiment_file,
        runs_root=runs_root,
        arms=["solo"],
        replicates=1,
        runner=FakeRunner(run_builder),
    )

    parked = [p for p in leftovers.parent.iterdir() if ".incomplete-" in p.name]
    assert len(parked) == 1
    assert (parked[0] / "workspaces" / "a1.txt").read_text(encoding="utf-8") == "stale"
    assert not (leftovers / "workspaces").exists()  # the rerun starts from a clean directory


def test_an_empty_run_directory_is_reused_not_parked(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    empty = runs_root / "demo" / "solo" / "rep1"
    empty.mkdir(parents=True)

    run_matrix(
        experiment_file,
        runs_root=runs_root,
        arms=["solo"],
        replicates=1,
        runner=FakeRunner(run_builder),
    )

    assert [p.name for p in empty.parent.iterdir()] == ["rep1"]
    assert is_run_complete(empty / EVENTS_FILE)


def test_a_finished_log_beats_leftover_files_so_the_run_is_skipped(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    done = finished(run_builder, runs_root / "demo" / "solo" / "rep1")
    (done / "extra.bin").write_bytes(b"\0")
    runner = FakeRunner(run_builder)

    records = run_matrix(
        experiment_file, runs_root=runs_root, arms=["solo"], replicates=1, runner=runner
    )

    assert records[0].status == "skipped"
    assert runner.calls == []


def test_parking_is_logged_with_both_paths(
    FakeRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = partial(run_builder, runs_root / "demo" / "solo" / "rep1")

    with caplog.at_level(logging.INFO, logger="research.matrix"):
        run_matrix(
            experiment_file,
            runs_root=runs_root,
            arms=["solo"],
            replicates=1,
            runner=FakeRunner(run_builder),
        )

    assert str(run_dir) in caplog.text
    assert "incomplete" in caplog.text
