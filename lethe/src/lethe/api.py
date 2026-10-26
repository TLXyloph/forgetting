"""Programmatic entry points, used by the CLI and by the research package."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import statistics
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lethe.agents.base import AgentBackend
from lethe.agents.claude_code import ClaudeCodeBackend
from lethe.config import ALWAYS_EXCLUDE, HARNESS_FILES, RunConfig, load_config
from lethe.domain.state import RunState
from lethe.domain.types import JudgeResult
from lethe.judge.runner import JudgeRunner
from lethe.runtime.orchestrator import EventHook, Orchestrator
from lethe.store import EventStore, read_events
from lethe.workspace.archive import Archive
from lethe.workspace.materializer import Materializer
from lethe.workspace.paths import PathFilter, safe_join


def new_run_id(config: RunConfig) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{config.preset}-{secrets.token_hex(2)}"


def run_task(
    config_path: Path,
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    runs_dir: Path | None = None,
    run_name: str | None = None,
    backend: AgentBackend | None = None,
    on_event: EventHook | None = None,
) -> Path:
    """Run the task described by `config_path` to completion and return its run directory.

    Precedence of settings: built-in defaults < preset < config file < `overrides` (a nested
    dict shaped like the TOML file, e.g. {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}).
    The run directory is `<runs_dir>/<run_name>` (a generated id when `run_name` is None);
    it must not already contain an `events.jsonl`. `backend` replaces the configured agent
    backend (tests and dry runs pass a ScriptedBackend).
    """
    config = load_config(Path(config_path), preset=preset, overrides=overrides)
    run_dir = Path(runs_dir or config.runs_dir) / (run_name or new_run_id(config))
    if (run_dir / "events.jsonl").exists():
        raise FileExistsError(f"run directory already holds a run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = Archive(
        run_dir / "archive.git", judged=config.task.judged, exclude=config.task.archive_exclude
    )
    orchestrator = Orchestrator(
        config,
        run_id=run_dir.name,
        run_dir=run_dir,
        archive=archive,
        materializer=Materializer(archive, run_dir / "ws"),
        judge=JudgeRunner(
            config.judge, task_dir=config.task.config_dir, work_root=run_dir / "judge"
        ),
        backend=backend or ClaudeCodeBackend(config.agent.claude_options()),
        store=EventStore(run_dir / "events.jsonl"),
        on_event=on_event,
    )
    asyncio.run(orchestrator.run())
    return run_dir


def load_run(run_dir: Path) -> RunState:
    """Rebuild the state of a finished or in-progress run from its `events.jsonl`."""
    state = RunState()
    for envelope in read_events(Path(run_dir) / "events.jsonl"):
        state.apply(envelope)
    return state


def run_filters(state: RunState) -> tuple[PathFilter, PathFilter]:
    """(judged, archive-exclude) filters from the config recorded in a run's RunStarted event."""
    task = state.config.get("task", {})
    exclude = (*ALWAYS_EXCLUDE, *task.get("exclude", []))
    return PathFilter(task.get("editable", ["**"]), exclude), PathFilter((*HARNESS_FILES, *exclude))


def run_archive(run_dir: Path, state: RunState) -> Archive:
    """Open a run's candidate archive."""
    judged, exclude = run_filters(state)
    return Archive(Path(run_dir) / "archive.git", judged=judged, exclude=exclude)


@dataclass(frozen=True)
class CheckReport:
    config: RunConfig
    seed_files: tuple[str, ...]
    editable_matches: tuple[str, ...]
    results: tuple[JudgeResult, ...]
    claude_path: str | None
    claude_version: str | None
    warnings: tuple[str, ...]


def seed_files(config: RunConfig) -> list[str]:
    """Relative paths of the regular files an agent would start from."""
    root = config.task.seed_dir
    exclude = config.task.archive_exclude
    found = []
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if not full.is_symlink() and full.is_file() and not exclude(rel):
                found.append(rel)
    return sorted(found)


def check_task(
    config_path: Path, *, repeats: int = 3, overrides: Mapping[str, Any] | None = None
) -> CheckReport:
    """Validate a task before spending budget: config, seed files, judge noise, claude CLI."""
    config = load_config(Path(config_path), overrides=overrides)
    files = seed_files(config)
    editable = tuple(p for p in files if config.task.judged(p))
    warnings: list[str] = []
    if not editable:
        warnings.append("no seed file matches task.editable: agents must create them from scratch")

    def materialize(dest: Path) -> None:
        for rel in files:
            target = safe_join(dest, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.task.seed_dir / rel, target)

    spec = replace(config.judge, cache=False, concurrency=1)
    with tempfile.TemporaryDirectory(prefix="lethe-check-") as tmp:
        runner = JudgeRunner(spec, task_dir=config.task.config_dir, work_root=Path(tmp))

        async def judge_all() -> list[JudgeResult]:
            return [
                await runner.evaluate(
                    cand_id=f"check{i}", judged_key=f"check{i}", materialize=materialize
                )
                for i in range(repeats)
            ]

        results = asyncio.run(judge_all())
    scores = [r.score for r in results if r.valid and r.score is not None]
    if len(scores) < len(results):
        bad = next(r for r in results if not r.valid)
        warnings.append(f"the seed was judged invalid: {(bad.error or bad.feedback)[:300]}")
    if len(scores) >= 2 and statistics.mean(scores) != 0:
        noise = statistics.stdev(scores) / abs(statistics.mean(scores))
        if noise > 0.005:
            warnings.append(
                f"judge noise is about {noise:.1%} of the score; small 'improvements' "
                "may be luck (consider averaging repeats inside the judge)"
            )
    if config.agent.shell:
        warnings.append("agent.shell = true lets agents run commands on this machine")
    if config.agent.isolation == "none":
        warnings.append("agent.isolation = 'none': your CLAUDE.md, hooks and plugins reach agents")
    claude = shutil.which("claude")
    version = None
    if claude is None:
        warnings.append("`claude` (Claude Code CLI) is not on PATH")
    else:
        try:
            version = (
                subprocess.run(
                    [claude, "--version"], capture_output=True, text=True, timeout=20
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            warnings.append("could not run `claude --version`")
    return CheckReport(
        config, tuple(files), editable, tuple(results), claude, version, tuple(warnings)
    )
