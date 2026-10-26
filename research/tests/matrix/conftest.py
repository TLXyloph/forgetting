"""Fixtures for the matrix tests: a stand-in for `lethe.api.run_task` and a small experiment."""

from __future__ import annotations

import textwrap
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT = """
name = "demo"
task = "tasks/toy/lethe.toml"
replicates = 2
[overrides]
budget = { usd = 3.0 }
agent = { model = "haiku" }
[[arms]]
name = "solo"
preset = "solo"
[[arms]]
name = "anchored"
preset = "team"
overrides = { forget = { when = [ { type = "anchored" } ] } }
"""


class FakeRunner:
    """Stands in for `lethe.api.run_task`: records calls and writes a small events.jsonl.

    All options are collections of (arm, rep) pairs. `fail`: raise after writing a partial log.
    `partial`: return after writing events but without run_finished. `silent`: return without
    writing anything. `kill`: raise KeyboardInterrupt after writing a partial log (the process
    was stopped hard). `interrupt`: what the real harness does on Ctrl-C: write
    run_finished("interrupted") and then let KeyboardInterrupt propagate. `end_reason` maps a pair
    to the reason its run_finished carries when the runner returns normally.
    """

    def __init__(
        self,
        builder: type,
        *,
        fail: Collection[tuple[str, int]] = (),
        partial: Collection[tuple[str, int]] = (),
        silent: Collection[tuple[str, int]] = (),
        kill: Collection[tuple[str, int]] = (),
        interrupt: Collection[tuple[str, int]] = (),
        end_reason: Mapping[tuple[str, int], str] | None = None,
    ) -> None:
        self.builder = builder
        self.fail, self.partial, self.silent = set(fail), set(partial), set(silent)
        self.kill, self.interrupt = set(kill), set(interrupt)
        self.end_reason = dict(end_reason or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        config_path: Path,
        *,
        preset: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        runs_dir: Path | None = None,
        run_name: str | None = None,
        backend: Any = None,
    ) -> Path:
        assert runs_dir is not None and run_name is not None
        self.calls.append(
            {
                "config_path": config_path,
                "preset": preset,
                "overrides": overrides,
                "runs_dir": runs_dir,
                "run_name": run_name,
            }
        )
        arm, rep = runs_dir.name, int(run_name.removeprefix("rep"))
        run_dir = runs_dir / run_name
        if (arm, rep) in self.silent:
            return run_dir
        b = self.builder(run_dir, agents=("a1",)).start(100.0)
        b.candidate("a1", 100.0 + 10 * rep, cost=0.5)
        if (arm, rep) in self.fail:
            raise RuntimeError(f"boom in {arm}/rep{rep}")
        if (arm, rep) in self.kill:
            raise KeyboardInterrupt
        if (arm, rep) in self.interrupt:
            b.finish("interrupted")
            raise KeyboardInterrupt
        if (arm, rep) in self.partial:
            return run_dir
        return b.finish(self.end_reason.get((arm, rep), "usd budget exhausted"))

    @property
    def order(self) -> list[tuple[str, str]]:
        return [(c["runs_dir"].name, c["run_name"]) for c in self.calls]


@pytest.fixture(name="FakeRunner")
def fake_runner_class() -> type[FakeRunner]:
    """The `FakeRunner` class: `FakeRunner(run_builder, fail={("solo", 1)})`."""
    return FakeRunner


@pytest.fixture
def experiment_file(tmp_path: Path) -> Path:
    path = tmp_path / "config" / "demo.toml"
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent(EXPERIMENT), encoding="utf-8")
    return path


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def with_task(experiment_file: Path) -> Path:
    task = experiment_file.parent / "tasks" / "toy" / "lethe.toml"
    task.parent.mkdir(parents=True)
    task.write_text("# a task", encoding="utf-8")
    return experiment_file
