"""`agent_options`: the Claude Code options a run's agents get."""

from __future__ import annotations

from pathlib import Path

from lethe.api import agent_options
from lethe.config import load_config

TASK = """
[task]
name = "toy"
description = "Maximize x."
seed = "seed"
editable = ["solver.py"]

[judge]
run = "python3 judge.py"
direction = "max"
"""


def config(tmp_path: Path, **agent: object):
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed" / "solver.py").write_text("x = 1\n")
    (tmp_path / "lethe.toml").write_text(TASK)
    return load_config(tmp_path / "lethe.toml", overrides={"agent": dict(agent)} if agent else None)


def test_reads_are_isolated_up_to_the_runs_directory_in_use_by_default(tmp_path: Path) -> None:
    runs = tmp_path / "elsewhere" / "runs"
    assert agent_options(config(tmp_path), runs).read_root == str(runs.resolve())


def test_a_configured_read_root_wins(tmp_path: Path) -> None:
    options = agent_options(config(tmp_path, read_root="."), tmp_path / "runs")
    assert options.read_root == str(tmp_path.resolve())
