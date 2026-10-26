"""Tests for research.matrix: running the matrix, what the runner is called with, selection."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import pytest

from lethe.api import run_task
from research.experiment import ExperimentError, load_experiment
from research.matrix import RunRecord, park_incomplete, run_matrix
from research.runs import EVENTS_FILE, is_run_complete

# ---------------------------------------------------------------------------------------------
# the plain run
# ---------------------------------------------------------------------------------------------


def test_runs_every_arm_and_replicate_in_file_order(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder)

    records = run_matrix(experiment_file, runs_root=runs_root, runner=runner)

    assert runner.order == [
        ("solo", "rep1"),
        ("solo", "rep2"),
        ("anchored", "rep1"),
        ("anchored", "rep2"),
    ]
    assert [(r.arm, r.rep, r.status) for r in records] == [
        ("solo", 1, "completed"),
        ("solo", 2, "completed"),
        ("anchored", 1, "completed"),
        ("anchored", 2, "completed"),
    ]
    for record in records:
        assert record.run_dir == runs_root / "demo" / record.arm / f"rep{record.rep}"
        assert is_run_complete(record.run_dir / EVENTS_FILE)


def test_runner_is_called_as_documented(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder)
    run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=runner)

    solo, anchored = runner.calls
    task = load_experiment(experiment_file).task

    assert solo["config_path"] == task
    assert solo["preset"] == "solo"
    assert solo["runs_dir"] == runs_root / "demo" / "solo"
    assert solo["run_name"] == "rep1"
    assert solo["overrides"] == {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}
    assert anchored["preset"] == "team"
    assert anchored["overrides"] == {
        "budget": {"usd": 3.0},
        "agent": {"model": "haiku"},
        "forget": {"when": [{"type": "anchored"}]},
    }


def test_runner_arguments_are_exactly_the_documented_ones(
    experiment_file: Path, runs_root: Path
) -> None:
    seen: list[dict[str, Any]] = []

    def strict_runner(config_path: Path, *, preset, overrides, runs_dir, run_name) -> Path:  # type: ignore[no-untyped-def]
        seen.append({"preset": preset, "run_name": run_name})
        raise RuntimeError("stop")

    # a runner accepting only these keywords must work: the matrix may not pass anything else
    records = run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=strict_runner)
    assert [r.message for r in records] == ["RuntimeError: stop"] * 2
    assert len(seen) == 2


def test_an_arm_without_a_preset_passes_none(
    FakeRunner: type, tmp_path: Path, runs_root: Path, run_builder: type
) -> None:
    path = tmp_path / "e.toml"
    path.write_text('name = "e"\ntask = "t.toml"\n[[arms]]\nname = "plain"\n', encoding="utf-8")
    runner = FakeRunner(run_builder)
    run_matrix(path, runs_root=runs_root, runner=runner)
    assert runner.calls[0]["preset"] is None
    assert runner.calls[0]["overrides"] == {}


def test_each_run_gets_its_own_copy_of_the_overrides(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    seen: list[Any] = []

    class Careless(FakeRunner):
        def __call__(self, config_path: Path, **kwargs: Any) -> Path:
            seen.append(copy.deepcopy(kwargs["overrides"]))  # what the runner was handed
            kwargs["overrides"]["budget"]["usd"] = 99.0  # then it scribbles on it
            kwargs["overrides"]["forget"] = {"when": []}
            return super().__call__(config_path, **kwargs)

    run_matrix(experiment_file, runs_root=runs_root, replicates=2, runner=Careless(run_builder))

    assert len(seen) == 4
    assert [o["budget"]["usd"] for o in seen] == [3.0, 3.0, 3.0, 3.0]  # no leakage between runs
    assert "forget" not in seen[0] and "forget" not in seen[1]  # solo arm
    assert seen[2]["forget"] == {"when": [{"type": "anchored"}]}  # anchored arm, untouched


def test_replicates_come_from_the_file_unless_overridden(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    default = FakeRunner(run_builder)
    run_matrix(experiment_file, runs_root=runs_root / "a", runner=default)
    assert len(default.calls) == 4  # 2 arms x 2 replicates from the file

    overridden = FakeRunner(run_builder)
    run_matrix(experiment_file, runs_root=runs_root / "b", replicates=3, runner=overridden)
    assert len(overridden.calls) == 6


def test_arm_selection_keeps_file_order(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder)
    records = run_matrix(
        experiment_file, runs_root=runs_root, arms=["anchored", "solo"], replicates=1, runner=runner
    )
    assert [r.arm for r in records] == ["solo", "anchored"]

    only = FakeRunner(run_builder)
    run_matrix(experiment_file, runs_root=runs_root / "x", arms=["anchored"], runner=only)
    assert {arm for arm, _ in only.order} == {"anchored"}


def test_unknown_arms_fail_before_anything_runs(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder)
    with pytest.raises(ExperimentError, match="nope"):
        run_matrix(experiment_file, runs_root=runs_root, arms=["solo", "nope"], runner=runner)
    assert runner.calls == []
    assert not runs_root.exists()


@pytest.mark.parametrize("bad", [0, -1])
def test_replicates_must_be_positive(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type, bad: int
) -> None:
    with pytest.raises(ValueError, match="replicates"):
        run_matrix(
            experiment_file, runs_root=runs_root, replicates=bad, runner=FakeRunner(run_builder)
        )


def test_runs_root_is_created_as_needed(
    FakeRunner: type, experiment_file: Path, tmp_path: Path, run_builder: type
) -> None:
    deep = tmp_path / "does" / "not" / "exist"
    run_matrix(experiment_file, runs_root=deep, replicates=1, runner=FakeRunner(run_builder))
    assert (deep / "demo" / "solo" / "rep1" / EVENTS_FILE).is_file()


def test_the_default_runner_is_lethe_run_task() -> None:
    assert inspect.signature(run_matrix).parameters["runner"].default is run_task


def test_run_matrix_accepts_string_paths(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    records = run_matrix(
        str(experiment_file), runs_root=str(runs_root), replicates=1, runner=FakeRunner(run_builder)
    )
    assert len(records) == 2


# ---------------------------------------------------------------------------------------------
# park_incomplete
# ---------------------------------------------------------------------------------------------


def test_park_incomplete_returns_the_new_location(tmp_path: Path) -> None:
    run_dir = tmp_path / "rep4"
    run_dir.mkdir()
    (run_dir / "x").write_text("y", encoding="utf-8")

    parked = park_incomplete(run_dir)

    assert not run_dir.exists()
    assert parked.parent == tmp_path
    assert parked.name.startswith("rep4.incomplete-")
    assert (parked / "x").read_text(encoding="utf-8") == "y"


def test_run_record_is_a_frozen_value() -> None:
    record = RunRecord("solo", 1, Path("/x"), "completed")
    assert record.message == ""
    with pytest.raises(AttributeError):
        record.status = "failed"  # type: ignore[misc]
