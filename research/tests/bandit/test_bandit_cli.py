"""Tests for the `python -m research.bandit` command line."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research.bandit import main

# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def with_task(experiment_file: Path) -> Path:
    task = experiment_file.parent / "tasks" / "toy" / "lethe.toml"
    task.parent.mkdir(parents=True)
    task.write_text("# a task", encoding="utf-8")
    return experiment_file


def test_cli_runs_the_pulls_and_prints_the_arm_table(
    BanditRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = BanditRunner(run_builder)

    code = main([str(with_task), "--pulls", "5", "--runs-root", str(runs_root)], runner=runner)

    out = capsys.readouterr().out
    assert code == 0
    assert len(runner.calls) == 5
    for arm in "abc":
        assert arm in out
    assert "pulls" in out and "mean reward" in out
    assert "UCB" in out


def test_cli_table_includes_runs_from_earlier_sessions(
    BanditRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [str(with_task), "--pulls", "3", "--runs-root", str(runs_root)]
    main(args, runner=BanditRunner(run_builder))
    capsys.readouterr()

    main(args, runner=BanditRunner(run_builder))

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    arm_rows = [row for row in rows if row and row[0] in ("a", "b", "c") and row[1].isdigit()]
    assert len(arm_rows) == 3
    assert sum(int(row[1]) for row in arm_rows) == 6  # 3 earlier + 3 new pulls


def test_cli_exit_code_is_1_when_a_pull_failed(
    BanditRunner: type, with_task: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder, fail=["b"])
    assert main([str(with_task), "--pulls", "3", "--runs-root", str(runs_root)], runner=runner) == 1


def test_cli_c_option(
    BanditRunner: type,
    expected_ucb_sequence: Any,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
) -> None:
    runner = BanditRunner(run_builder)
    main([str(with_task), "--pulls", "8", "--c", "0", "--runs-root", str(runs_root)], runner=runner)
    assert runner.arms == expected_ucb_sequence({"a": 0.5, "b": 0.05, "c": 0.01}, 8, 0.0)


def test_cli_requires_pulls(with_task: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([str(with_task)])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("extra", [["--pulls", "-2"], ["--pulls", "1", "--c", "-1"]])
def test_cli_rejects_invalid_numbers(
    BanditRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    extra: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = BanditRunner(run_builder)
    assert main([str(with_task), "--runs-root", str(runs_root), *extra], runner=runner) == 2
    assert "error" in capsys.readouterr().err.lower()
    assert runner.calls == []


def test_cli_checks_the_task_file_exists(
    BanditRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = BanditRunner(run_builder)  # the task file was never created
    code = main(
        [str(experiment_file), "--pulls", "1", "--runs-root", str(runs_root)], runner=runner
    )
    assert code == 2
    assert "task" in capsys.readouterr().err.lower()
    assert runner.calls == []
