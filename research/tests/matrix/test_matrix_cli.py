"""Tests for the `python -m research.matrix` command line."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.matrix import main

# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def with_task(experiment_file: Path) -> Path:
    task = experiment_file.parent / "tasks" / "toy" / "lethe.toml"
    task.parent.mkdir(parents=True)
    task.write_text("# a task", encoding="utf-8")
    return experiment_file


def test_cli_runs_the_matrix_and_prints_a_summary(
    FakeRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakeRunner(run_builder)

    code = main([str(with_task), "--runs-root", str(runs_root), "--replicates", "1"], runner=runner)

    out = capsys.readouterr().out
    assert code == 0
    assert len(runner.calls) == 2
    assert "solo" in out and "anchored" in out
    assert "completed" in out


def test_cli_arms_option(
    FakeRunner: type, with_task: Path, runs_root: Path, run_builder: type
) -> None:
    runner = FakeRunner(run_builder)
    main(
        [str(with_task), "--runs-root", str(runs_root), "--arms", "anchored", "--replicates", "1"],
        runner=runner,
    )
    assert runner.order == [("anchored", "rep1")]


def test_cli_exit_code_is_1_when_any_run_failed(
    FakeRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakeRunner(run_builder, fail={("solo", 1)})

    code = main([str(with_task), "--runs-root", str(runs_root), "--replicates", "1"], runner=runner)

    out = capsys.readouterr().out
    assert code == 1
    assert "failed" in out
    assert "boom in solo/rep1" in out


def test_cli_skipped_runs_are_reported_and_exit_code_is_0(
    FakeRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [str(with_task), "--runs-root", str(runs_root), "--replicates", "1"]
    main(args, runner=FakeRunner(run_builder))
    capsys.readouterr()

    code = main(args, runner=FakeRunner(run_builder))

    assert code == 0
    assert "skipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [["--arms", "nope"], ["--replicates", "0"]],
)
def test_cli_rejects_bad_options(
    FakeRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    extra: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakeRunner(run_builder)
    code = main([str(with_task), "--runs-root", str(runs_root), *extra], runner=runner)
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()
    assert runner.calls == []


def test_cli_rejects_an_invalid_experiment_file(
    FakeRunner: type,
    tmp_path: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n', encoding="utf-8")
    assert main([str(bad), "--runs-root", str(runs_root)], runner=FakeRunner(run_builder)) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_missing_experiment_file(
    FakeRunner: type,
    tmp_path: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [str(tmp_path / "nope.toml"), "--runs-root", str(runs_root)], runner=FakeRunner(run_builder)
    )
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_checks_that_the_task_file_exists_before_starting(
    FakeRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakeRunner(run_builder)  # the task file was never created

    code = main([str(experiment_file), "--runs-root", str(runs_root)], runner=runner)

    assert code == 2
    assert "task" in capsys.readouterr().err.lower()
    assert runner.calls == []
