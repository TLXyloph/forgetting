"""Fixtures for the judge tests: tiny real judge scripts run through real subprocesses."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lethe.domain.types import JudgeResult
from lethe.judge.runner import JudgeRunner, JudgeSpec


class Materializer:
    """A `materialize` callback that writes small files and remembers the trees it was given."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = {name: textwrap.dedent(text) for name, text in files.items()}
        self.trees: list[Path] = []

    def __call__(self, tree: Path) -> None:
        self.trees.append(tree)
        for name, text in self.files.items():
            (tree / name).write_text(text)


@pytest.fixture
def py() -> Callable[[str], str]:
    """`py("judge.py")` is a shell command that runs that script with this interpreter."""
    return lambda script: f'"{sys.executable}" {script}'


@pytest.fixture
def judge_tree() -> Callable[..., Materializer]:
    """`judge_tree(source, **extra_files)` builds a materializer whose tree holds `judge.py`."""

    def make(source: str, **extra: str) -> Materializer:
        return Materializer({"judge.py": source, **extra})

    return make


@pytest.fixture
def make_runner(tmp_path: Path, py: Callable[[str], str]) -> Callable[..., JudgeRunner]:
    """`make_runner(run=None, **spec_overrides)`; defaults to running `judge.py`."""

    def make(run: str | None = None, **spec: Any) -> JudgeRunner:
        params: dict[str, Any] = {"direction": "max", "timeout_s": 30.0, **spec}
        judge = JudgeSpec(run=run if run is not None else py("judge.py"), **params)
        return JudgeRunner(judge, task_dir=tmp_path / "task", work_root=tmp_path / "work")

    return make


@pytest.fixture
def evaluate() -> Callable[..., JudgeResult]:
    """Synchronous wrapper: one `runner.evaluate(...)` inside its own `asyncio.run`."""

    def run(
        runner: JudgeRunner,
        materialize: Callable[[Path], None],
        *,
        cand_id: str = "c0001",
        key: str = "k1",
    ) -> JudgeResult:
        return asyncio.run(
            runner.evaluate(cand_id=cand_id, judged_key=key, materialize=materialize)
        )

    return run


def _process_alive(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")  # a zombie awaiting its reaper is dead


@pytest.fixture
def wait_until_dead() -> Callable[..., bool]:
    """`wait_until_dead(pid)` polls until the process is gone; False if it outlives `timeout`."""

    def wait(pid: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while _process_alive(pid):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    return wait


@pytest.fixture
def count_runs() -> Callable[[Path], int]:
    """`count_runs(path)` is the number of lines a judge appended to its counter file."""

    def count(path: Path) -> int:
        return len(path.read_text().splitlines()) if path.exists() else 0

    return count
