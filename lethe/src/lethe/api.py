"""Programmatic entry points, used by the CLI and by the research package."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import signal
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lethe.agents.base import AgentBackend
from lethe.agents.claude_code import (
    API_BILLING_ENV,
    ClaudeCodeBackend,
    ClaudeCodeOptions,
    pinned,
)
from lethe.config import ALWAYS_EXCLUDE, HARNESS_FILES, RunConfig, load_config
from lethe.domain.state import RunState
from lethe.domain.types import JudgeResult
from lethe.judge.runner import JudgeRunner
from lethe.runtime.orchestrator import EventHook, Orchestrator
from lethe.store import EventStore, read_events
from lethe.workspace.archive import Archive
from lethe.workspace.materializer import Materializer
from lethe.workspace.paths import PathFilter, safe_join

TRUST_MODEL = (
    "trust model: forgetting is enforced on the agents' tools, not on code the judge executes.\n"
    "Candidate code runs with your permissions (no OS sandbox): a candidate written to do so\n"
    "could read old workspaces and transcripts."
)
_REACH = "they can reach old epochs, transcripts, the archive and hidden judges"


def new_run_id(config: RunConfig) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{config.preset}-{secrets.token_hex(2)}"


def config_warnings(config: RunConfig, environ: Mapping[str, str] | None = None) -> list[str]:
    """What `lethe check` and `lethe run` say before any agent starts: the trust model first,
    then risky agent settings and API-billing variables found in `environ` (os.environ)."""
    env = os.environ if environ is None else environ
    agent = config.agent
    warnings = [TRUST_MODEL]
    if agent.shell:
        warnings.append(f"agent.shell = true lets agents run commands on this machine: {_REACH}")
    if agent.isolation == "none":
        warnings.append(
            "agent.isolation = 'none': your CLAUDE.md, hooks and plugins reach agents, and their "
            f"file tools are not confined to the workspace: {_REACH}"
        )
    if agent.extra_args:
        warnings.append(
            f"agent.extra_args ({' '.join(agent.extra_args)}) go to claude as-is and may undo "
            "safe mode (e.g. --add-dir)"
        )
    present = [name for name in API_BILLING_ENV if name in env]
    if present:
        names = ", ".join(present)
        if agent.isolation == "safe" and not agent.allow_api_billing:
            warnings.append(
                f"{names} set in your environment: removed from the agents' environment (safe "
                "isolation); a turn that still bills an API key stops the run"
            )
        else:
            setting = (
                "agent.allow_api_billing = true"
                if agent.allow_api_billing
                else f"agent.isolation = '{agent.isolation}'"
            )
            warnings.append(
                f"{names} set in your environment: passed to the agents ({setting}), so their "
                "turns may be billed to that account"
            )
    return warnings


def interrupt_on_termination() -> None:
    """Make SIGTERM and Ctrl-C stop a run cleanly: the run is cancelled, ends "interrupted" and
    kills its agents' process groups. Call it from the main thread before `run_task`.

    A shell starts a background command with SIGINT ignored, and Python then installs no Ctrl-C
    handler, so Ctrl-C did nothing; a plain SIGTERM killed lethe outright and left the agents,
    which run in their own process groups, to finish their turns unsupervised.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda _sig, _frame: signal.raise_signal(signal.SIGINT))


def agent_options(config: RunConfig, runs_dir: Path) -> ClaudeCodeOptions:
    """The Claude Code options for a run's agents: the configured ones, on the CLI binary current
    now, with reads isolated up to `agent.read_root` (by default `runs_dir`)."""
    options = pinned(config.agent.claude_options())
    if options.read_root is None:
        options = replace(options, read_root=str(Path(runs_dir).resolve()))
    return options


def run_task(
    config_path: Path,
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    runs_dir: Path | None = None,
    run_name: str | None = None,
    backend: AgentBackend | None = None,
    on_event: EventHook | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> Path:
    """Run the task described by `config_path` to completion and return its run directory.

    Precedence of settings: built-in defaults < preset < config file < `overrides` (a nested
    dict shaped like the TOML file, e.g. {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}).
    The run directory is `<runs_dir>/<run_name>` (a generated id when `run_name` is None);
    it must not already contain an `events.jsonl`. `backend` replaces the configured agent
    backend (tests and dry runs pass a ScriptedBackend). `on_warning` receives each of
    `config_warnings(config)` before anything runs. Judge trees are made in the system temp
    directory, never inside the run (or task) directory.
    """
    config = load_config(Path(config_path), preset=preset, overrides=overrides)
    if on_warning is not None:
        for warning in config_warnings(config):
            on_warning(warning)
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
        judge=JudgeRunner(config.judge, task_dir=config.task.config_dir),
        backend=backend or ClaudeCodeBackend(agent_options(config, run_dir.parent)),
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
    """Relative paths of the regular files an agent would start from (what the archive keeps:
    names with a line break or a backslash are skipped there too)."""
    root = config.task.seed_dir
    exclude = config.task.archive_exclude
    found = []
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if any(c in rel for c in "\n\r\\"):
                continue
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

    runner = JudgeRunner(
        replace(config.judge, cache=False, concurrency=1), task_dir=config.task.config_dir
    )

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
    warnings += config_warnings(config)
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
