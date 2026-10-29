"""Tests for research.matrix: one run at a time on a runs root (the `.matrix.lock` file lock)."""

from __future__ import annotations

import fcntl
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from research.experiment import load_experiment
from research.matrix import LOCK_FILE, RunRecord, execute_run, main, run_matrix

# A process that holds the lock on argv[1] until its stdin closes or argv[2] seconds pass.
HOLDER = (
    "import fcntl, select, sys\n"
    "fh = open(sys.argv[1], 'a')\n"
    "fcntl.flock(fh, fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "select.select([sys.stdin], [], [], float(sys.argv[2]))\n"
)
HOLD_S = 20.0


@pytest.fixture
def holder(runs_root: Path) -> Iterator[subprocess.Popen[str]]:
    """Another process holding the runs root's lock; close its stdin to release it."""
    runs_root.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(runs_root / LOCK_FILE), str(HOLD_S)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdout.readline().strip() == "locked"
        yield process
    finally:
        process.kill()
        process.wait(10)


def lock_is_free(path: Path) -> bool:
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
    return True


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def in_thread(target: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
    """Start `target` in a daemon thread; its return value lands in the returned list."""
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(target()), daemon=True)
    thread.start()
    return thread, result


def test_a_run_holds_the_lock_file_in_the_runs_root(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    fake = FakeRunner(run_builder)
    held: list[bool] = []

    def runner(*args: Any, **kwargs: Any) -> Path:
        held.append(not lock_is_free(runs_root / LOCK_FILE))
        return fake(*args, **kwargs)

    records = run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=runner)

    assert held == [True, True]
    assert [r.status for r in records] == ["completed", "completed"]
    assert lock_is_free(runs_root / LOCK_FILE)


def test_a_second_process_waits_until_the_first_run_finishes(
    FakeRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    caplog: pytest.LogCaptureFixture,
) -> None:
    experiment = load_experiment(experiment_file)
    solo, anchored = experiment.arms
    fake = FakeRunner(run_builder)
    started, release = threading.Event(), threading.Event()
    spans: dict[str, list[float]] = {}

    def runner(*args: Any, **kwargs: Any) -> Path:
        arm = kwargs["runs_dir"].name
        spans[arm] = [time.monotonic()]
        if arm == "solo":
            started.set()
            assert release.wait(10)
        run_dir = fake(*args, **kwargs)
        spans[arm].append(time.monotonic())
        return run_dir

    def run(arm: Any) -> Callable[[], RunRecord]:
        return lambda: execute_run(experiment, arm, 1, runs_root=runs_root, runner=runner)

    with caplog.at_level(logging.INFO, logger="research.matrix"):
        first, first_result = in_thread(run(solo))
        assert started.wait(10)
        second, second_result = in_thread(run(anchored))
        assert wait_for(lambda: "waiting" in caplog.text)
        assert "anchored" not in spans  # blocked while solo/rep1 runs
        release.set()
        first.join(10)
        second.join(10)

    assert [r.status for r in first_result + second_result] == ["completed", "completed"]
    assert spans["anchored"][0] >= spans["solo"][1]
    assert caplog.text.count("waiting") == 1  # said once, not once per poll


def test_the_lock_is_shared_with_other_processes(
    FakeRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    holder: subprocess.Popen[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = FakeRunner(run_builder)
    with caplog.at_level(logging.INFO, logger="research.matrix"):
        thread, result = in_thread(
            lambda: run_matrix(
                experiment_file, runs_root=runs_root, replicates=1, arms=["solo"], runner=runner
            )
        )
        assert wait_for(lambda: "waiting" in caplog.text)
        assert runner.calls == []
        assert holder.stdin is not None
        holder.stdin.close()
        thread.join(10)

    assert [r.status for r in result[0]] == ["completed"]


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_the_lock_is_released_when_a_run_raises(
    experiment_file: Path, runs_root: Path, error: BaseException
) -> None:
    def runner(*args: Any, **kwargs: Any) -> Path:
        raise error

    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=runner)
    else:
        records = run_matrix(experiment_file, runs_root=runs_root, replicates=1, runner=runner)
        assert {r.status for r in records} == {"failed"}

    assert lock_is_free(runs_root / LOCK_FILE)


def test_no_lock_runs_even_while_another_process_holds_it(
    FakeRunner: type,
    with_task: Path,
    runs_root: Path,
    run_builder: type,
    holder: subprocess.Popen[str],
) -> None:
    runner = FakeRunner(run_builder)
    args = [str(with_task), "--runs-root", str(runs_root), "--replicates", "1", "--no-lock"]

    began = time.monotonic()
    code = main(args, runner=runner)  # in the main thread: main installs signal handlers

    assert code == 0
    assert len(runner.calls) == 2
    assert time.monotonic() - began < HOLD_S / 2  # it did not wait for the holder to give up
    assert holder.poll() is None  # and the lock was held all along


def test_no_lock_never_creates_the_lock_file(
    FakeRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    run_matrix(
        experiment_file,
        runs_root=runs_root,
        replicates=1,
        runner=FakeRunner(run_builder),
        lock=False,
    )
    assert not (runs_root / LOCK_FILE).exists()
