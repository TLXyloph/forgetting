"""Run an experiment's matrix of arms x replicates: sequential, resumable, never destructive.

    python -m research.matrix research/config/experiments/paper-toy-mis.toml \
        [--runs-root research/runs] [--replicates N] [--arms solo,team-forget] [--no-lock]

Every run lives in `<runs_root>/<experiment>/<arm>/rep<k>`. Arms run in file order and
replicates 1..R in order, one at a time (subscription rate windows make parallel runs useless).
Re-invoking the command resumes: a complete replicate (`run_finished` in its log, see
`research.runs.is_run_complete`) is skipped; the remains of any other one, including a run the
harness wrote "interrupted" into, are *renamed* to `rep<k>.incomplete-<UTC timestamp>` (nothing is
ever deleted) and the replicate runs again. A runner exception is recorded and the matrix moves
on; Ctrl-C stops it, leaving the interrupted replicate to be parked and rerun next time.

One at a time holds across processes too: each run, with its skip check and parking, holds an
exclusive advisory lock on `<runs_root>/.matrix.lock`, so a second matrix (or bandit) on the same
runs root waits until the current run has finished. Runs that overlap share the sandbox's temp
directory, through which an agent once read another run's kernel (see `research.leaks`).
`--no-lock` opts out.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from lethe.api import interrupt_on_termination, run_task
from research.experiment import (
    Arm,
    Experiment,
    ExperimentError,
    arm_overrides,
    load_experiment,
    select_arms,
)
from research.runs import (
    EVENTS_FILE,
    ending_is_complete,
    finish_event,
    is_run_complete,
    rep_name,
    run_dir_for,
)
from research.textfmt import format_table

__all__ = [
    "LOCK_FILE",
    "RunRecord",
    "RunStatus",
    "Runner",
    "execute_run",
    "main",
    "park_incomplete",
    "run_lock",
    "run_matrix",
]

LOCK_FILE = ".matrix.lock"  # directly in the runs root (research/runs/ is never committed)

RunStatus = Literal["completed", "skipped", "failed"]
Runner = Callable[..., Path]  # lethe.api.run_task, or a stand-in with the same keywords

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunRecord:
    """What happened to one (arm, replicate): ran to the end, was already done, or failed."""

    arm: str
    rep: int
    run_dir: Path
    status: RunStatus
    message: str = ""


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def park_incomplete(run_dir: Path) -> Path:
    """Move a partial run aside as `<name>.incomplete-<UTC timestamp>`; returns the new path.

    A numeric suffix keeps earlier parked runs from ever being overwritten.
    """
    base = f"{run_dir.name}.incomplete-{_utc_stamp()}"
    target = run_dir.with_name(base)
    suffix = 1
    while target.exists():
        suffix += 1
        target = run_dir.with_name(f"{base}-{suffix}")
    run_dir.rename(target)
    return target


def _has_leftovers(run_dir: Path) -> bool:
    """True when something other than an empty directory sits where the run should go."""
    return run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir()))


@contextlib.contextmanager
def run_lock(runs_root: Path) -> Iterator[None]:
    """Hold an exclusive `flock` on `<runs_root>/LOCK_FILE`, creating both when missing.

    Waits while another process (or thread) holds it, saying so once. The lock is released when
    the block exits, however it exits, and by the OS when the process dies.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    path = runs_root / LOCK_FILE
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for the run in progress under %s to finish (%s)", runs_root, path)
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def execute_run(
    experiment: Experiment,
    arm: Arm,
    rep: int,
    *,
    runs_root: Path,
    runner: Runner,
    lock: bool = True,
) -> RunRecord:
    """Run replicate `rep` of `arm` unless it is already finished; never raises for run failures.

    Skips a finished run, parks the remains of an unfinished one, calls
    `runner(task, preset=..., overrides=..., runs_dir=<arm dir>, run_name="rep<k>")`, and checks
    that the run actually wrote a finished log. KeyboardInterrupt and SystemExit propagate.
    With `lock`, all of it happens under `run_lock(runs_root)`.
    """
    if not lock:
        return _execute_run(experiment, arm, rep, runs_root=runs_root, runner=runner)
    with run_lock(runs_root):
        return _execute_run(experiment, arm, rep, runs_root=runs_root, runner=runner)


def _execute_run(
    experiment: Experiment, arm: Arm, rep: int, *, runs_root: Path, runner: Runner
) -> RunRecord:
    run_dir = run_dir_for(runs_root, experiment.name, arm.name, rep)
    events = run_dir / EVENTS_FILE
    label = f"{arm.name}/rep{rep}"
    if events.exists() and is_run_complete(events):
        log.info("%s: already finished, skipping", label)
        return RunRecord(arm.name, rep, run_dir, "skipped")
    if _has_leftovers(run_dir):
        parked = park_incomplete(run_dir)
        log.warning("%s: %s holds an incomplete run; moved it to %s", label, run_dir, parked)
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    def failed(message: str) -> RunRecord:
        log.error("%s: failed: %s", label, message)
        return RunRecord(arm.name, rep, run_dir, "failed", message)

    try:
        runner(
            experiment.task,
            preset=arm.preset,
            overrides=arm_overrides(experiment, arm),
            runs_dir=run_dir.parent,
            run_name=rep_name(rep),
        )
    except Exception as exc:
        return failed(f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)
    if not events.exists():
        return failed(f"runner returned but wrote no {EVENTS_FILE}")
    finished = finish_event(events)
    if finished is None:
        return failed("runner returned but the log has no run_finished event")
    if not ending_is_complete(finished.status, finished.reason):
        return failed(f"run ended early ({finished.status}): {finished.reason}")
    log.info("%s: completed", label)
    return RunRecord(arm.name, rep, run_dir, "completed")


def run_matrix(
    experiment_path: Path | str,
    *,
    runs_root: Path | str,
    replicates: int | None = None,
    arms: Sequence[str] | None = None,
    runner: Runner = run_task,
    lock: bool = True,
) -> list[RunRecord]:
    """Run every selected arm (file order) x replicate 1..R (default: the file's `replicates`).

    Returns one record per (arm, replicate), in run order. `lock`: see `execute_run`.
    """
    experiment = load_experiment(experiment_path)
    count = experiment.replicates if replicates is None else replicates
    if isinstance(count, bool) or count < 1:
        raise ValueError(f"replicates must be an integer >= 1, got {count!r}")
    chosen = select_arms(experiment, arms)
    root = Path(runs_root)
    total = len(chosen) * count
    records: list[RunRecord] = []
    for arm in chosen:
        for rep in range(1, count + 1):
            log.info("[%d/%d] %s/rep%d", len(records) + 1, total, arm.name, rep)
            records.append(
                execute_run(experiment, arm, rep, runs_root=root, runner=runner, lock=lock)
            )
    return records


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def format_records(records: Sequence[RunRecord]) -> str:
    """A table with one row per run, plus a tally."""
    rows = [[f"{r.arm}/rep{r.rep}", r.status, r.message] for r in records]
    tally = ", ".join(
        f"{sum(r.status == status for r in records)} {status}"
        for status in ("completed", "skipped", "failed")
    )
    return format_table(["run", "status", "message"], rows, align="lll") + "\n" + tally


def main(argv: Sequence[str] | None = None, *, runner: Runner = run_task) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research.matrix",
        description="Run (or resume) an experiment: every arm x replicate, one run at a time.",
    )
    parser.add_argument("experiment", type=Path, help="experiment TOML file")
    parser.add_argument("--runs-root", type=Path, default=Path("research/runs"))
    parser.add_argument("--replicates", type=int, default=None, help="override the file's count")
    parser.add_argument("--arms", default=None, help="comma-separated arm names (default: all)")
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help=f"do not wait for runs of other processes (the lock is <runs-root>/{LOCK_FILE})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    interrupt_on_termination()
    try:
        experiment = load_experiment(args.experiment)
        if not experiment.task.is_file():  # fail now, not as N identical failed runs
            raise ExperimentError(f"task file not found: {experiment.task}")
        names = (
            None if args.arms is None else [n.strip() for n in args.arms.split(",") if n.strip()]
        )
        records = run_matrix(
            args.experiment,
            runs_root=args.runs_root,
            replicates=args.replicates,
            arms=names,
            runner=runner,
            lock=not args.no_lock,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_records(records))
    return 1 if any(r.status == "failed" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
