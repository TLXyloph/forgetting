"""Fixtures for the bandit tests: an arm-driven runner and a UCB1 oracle."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT = """
name = "bets"
task = "tasks/toy/lethe.toml"
replicates = 3
[overrides]
budget = { usd = 3.0 }
[[arms]]
name = "a"
preset = "solo"
[[arms]]
name = "b"
preset = "team"
overrides = { forget = { when = [ { type = "anchored" } ] } }
[[arms]]
name = "c"
preset = "team-forget"
"""


FINALS = {"a": 150.0, "b": 105.0, "c": 101.0}


class BanditRunner:
    """A stand-in for `lethe.api.run_task` whose result depends only on the arm."""

    def __init__(
        self,
        builder: type,
        finals: Mapping[str, float | None] = FINALS,
        *,
        direction: str = "max",
        seed_score: float | None = 100.0,
        fail: Sequence[str] = (),
        end_reason: Mapping[str, str] | None = None,
    ) -> None:
        self.builder = builder
        self.finals = dict(finals)
        self.direction = direction
        self.seed_score = seed_score
        self.fail = set(fail)
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
        arm = runs_dir.name
        b = self.builder(runs_dir / run_name, direction=self.direction, agents=("a1",))
        b.start(self.seed_score)
        final = self.finals[arm]
        if final is not None:
            b.candidate("a1", final, cost=0.5)
        if arm in self.fail:
            raise RuntimeError(f"boom in {arm}")
        return b.finish(self.end_reason.get(arm, "usd budget exhausted"))

    @property
    def arms(self) -> list[str]:
        return [c["runs_dir"].name for c in self.calls]


def _expected_ucb_sequence(
    reward_per_arm: Mapping[str, float],
    pulls: int,
    c: float,
    initial: Mapping[str, Sequence[float]] | None = None,
) -> list[str]:
    """UCB1 re-derived from its definition, independently of research.bandit."""
    names = list(reward_per_arm)
    history = {n: list((initial or {}).get(n, [])) for n in names}
    order: list[str] = []
    for _ in range(pulls):
        unpulled = [n for n in names if not history[n]]
        if unpulled:
            pick = unpulled[0]
        else:
            total = sum(len(h) for h in history.values())

            def index(n: str, total: int = total) -> float:
                h = history[n]
                return sum(h) / len(h) + c * math.sqrt(2 * math.log(total) / len(h))

            pick = max(names, key=lambda n: (index(n), -names.index(n)))
        history[pick].append(reward_per_arm[pick])
        order.append(pick)
    return order


@pytest.fixture(name="BanditRunner")
def bandit_runner_class() -> type[BanditRunner]:
    """The `BanditRunner` class: `BanditRunner(run_builder, finals, fail=["b"])`."""
    return BanditRunner


@pytest.fixture(name="expected_ucb_sequence")
def expected_ucb_sequence_function():
    """`expected_ucb_sequence(reward_per_arm, pulls, c, initial=None)`: UCB1 from its definition."""
    return _expected_ucb_sequence


@pytest.fixture
def experiment_file(tmp_path: Path) -> Path:
    path = tmp_path / "config" / "bets.toml"
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
