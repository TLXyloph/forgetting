"""Where experiment runs live on disk, and which of them are finished.

Layout: `<runs_root>/<experiment>/<arm>/rep<k>/events.jsonl`, with k counting from 1. Directories
named `rep<k>.incomplete-<timestamp>` are partial runs the matrix runner parked; they never match
the `rep<k>` pattern, so every function here ignores them.

A run is *complete* when its log holds `run_finished` that is not an abnormal end. The harness
also writes `run_finished` when it is interrupted (Ctrl-C), when it crashed, and when it stopped
(every agent gave up after repeated errors, or safe mode refused API-key billing); those runs did
not use their budget, so they are neither results to compare nor runs to skip when resuming.
The event's `status` says which ("completed" is the only normal one). Logs written before the
status existed decode as "completed", so for those the reason text decides: it must not contain
one of ABNORMAL_ENDS.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

from lethe.domain.events import RunFinished
from lethe.store import read_events
from lethe.workspace.paths import safe_join
from research.experiment import is_valid_name

__all__ = [
    "ABNORMAL_ENDS",
    "ABNORMAL_STATUSES",
    "EVENTS_FILE",
    "RunLocation",
    "arm_dir_for",
    "discover_runs",
    "ending_is_complete",
    "find_rep_dirs",
    "finish_event",
    "finish_reason",
    "is_run_complete",
    "next_free_rep",
    "reason_is_complete",
    "rep_name",
    "run_dir_for",
]

EVENTS_FILE = "events.jsonl"
# `run_finished` statuses of runs that were cut short (see module docs)
ABNORMAL_STATUSES: tuple[str, ...] = ("interrupted", "crashed", "stopped")
# lower-case fragments of `run_finished` reasons that mark a run as cut short, for logs written
# before RunFinished had a status
ABNORMAL_ENDS: tuple[str, ...] = ("interrupted", "repeated errors", "crashed:")
_REP_PATTERN = re.compile(r"rep([1-9][0-9]*)")

log = logging.getLogger(__name__)


class RunLocation(NamedTuple):
    arm: str
    rep: int
    run_dir: Path


def rep_name(rep: int) -> str:
    """Directory name of replicate `rep` (1-based)."""
    return f"rep{rep}"


def arm_dir_for(runs_root: Path, experiment: str, arm: str) -> Path:
    """`<runs_root>/<experiment>/<arm>`, refusing names that could leave `runs_root`."""
    for label, name in (("experiment", experiment), ("arm", arm)):
        if not is_valid_name(name):
            raise ValueError(f"unsafe {label} name for a directory: {name!r}")
    return safe_join(Path(runs_root), f"{experiment}/{arm}")


def run_dir_for(runs_root: Path, experiment: str, arm: str, rep: int) -> Path:
    """`<runs_root>/<experiment>/<arm>/rep<rep>`, refusing names that could leave `runs_root`."""
    if rep < 1:
        raise ValueError(f"replicates are numbered from 1, got {rep}")
    return arm_dir_for(runs_root, experiment, arm) / rep_name(rep)


def finish_event(events_path: Path) -> RunFinished | None:
    """The log's `run_finished` event; None if there is none.

    A missing, empty, torn or corrupt log has none (corruption is logged): the safe consequence
    is that a matrix reruns the replicate, after parking the old directory.
    """
    try:
        for envelope in read_events(events_path):
            if isinstance(envelope.event, RunFinished):
                return envelope.event
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.warning("cannot read %s (%s); treating the run as unfinished", events_path, exc)
    return None


def finish_reason(events_path: Path) -> str | None:
    """The reason in the log's `run_finished` event; None if there is none."""
    event = finish_event(events_path)
    return None if event is None else event.reason


def reason_is_complete(reason: str) -> bool:
    """False for the `run_finished` reasons that mean the run was cut short (old logs)."""
    lowered = reason.lower()
    return not any(marker in lowered for marker in ABNORMAL_ENDS)


def ending_is_complete(status: str, reason: str) -> bool:
    """True for a normal end: status "completed" (also what logs without a status decode to)
    and, as the fallback for such old logs, a reason that is not an abnormal one."""
    return status not in ABNORMAL_STATUSES and reason_is_complete(reason)


def is_run_complete(events_path: Path) -> bool:
    """True when the run ended by itself: `run_finished` that is not an abnormal end."""
    event = finish_event(events_path)
    return event is not None and ending_is_complete(event.status, event.reason)


def find_rep_dirs(arm_dir: Path) -> dict[int, Path]:
    """The `rep<k>` directories directly under `arm_dir`, keyed by k."""
    if not arm_dir.is_dir():
        return {}
    found: dict[int, Path] = {}
    for entry in arm_dir.iterdir():
        match = _REP_PATTERN.fullmatch(entry.name)
        if match and entry.is_dir():
            found[int(match.group(1))] = entry
    return found


def next_free_rep(arm_dir: Path) -> int:
    """Smallest replicate number whose `rep<k>` slot is unused (partial runs still use theirs)."""
    rep = 1
    while (arm_dir / rep_name(rep)).exists():
        rep += 1
    return rep


def discover_runs(experiment_dir: Path) -> list[RunLocation]:
    """Every run with an event log under `experiment_dir`, sorted by arm name then replicate."""
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"no such experiment directory: {experiment_dir}")
    runs: list[RunLocation] = []
    for arm_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        for rep, run_dir in sorted(find_rep_dirs(arm_dir).items()):
            if (run_dir / EVENTS_FILE).is_file():
                runs.append(RunLocation(arm_dir.name, rep, run_dir))
    return runs
