"""Leaks between runs, agents and forgotten epochs, found in the runs' Claude Code transcripts.

    python -m research.leaks research/runs/<experiment>[/<arm>/rep<k>] \
        [--runs-root DIR] [--all-experiments]

Agents run in Claude Code's sandbox, which confines their writes but not every read. Its temp
directory (`$TMPDIR`, `/tmp/claude-<uid>`) is shared by every sandboxed session of the user, so one
run can find what another left there; other workspaces, other runs and Claude Code's own logs may
be readable too. lethe denies what it can; this is the second layer, detection after the fact.

A finding is a successful tool call whose input names a path outside the agent's own workspace
`<run_dir>/ws/<agent>/e<epoch>`; relative paths are resolved against that workspace. Paths are
found as `research.leak_paths` describes, best effort: no finding is no proof of isolation.
Kinds, `LeakFinding.kind`:

- other-workspace: another agent's workspace, or the agent's own of another epoch;
- run-state: the rest of the run directory (events.jsonl, archive.git, transcripts/, index/);
- other-run: anything else under the runs root (the nearest directory above named `runs`, the
  experiment directory when there is none, or the one given);
- claude-logs: Claude Code's directory `~/.claude`, except claude-output;
- claude-output: Claude Code's own files for the agent's workspace (named by its slug, the path
  with `/` replaced by `-`): persisted tool output (`~/.claude/projects/<slug>/.../tool-results/`)
  and background-task output (`$TMPDIR/<slug>/...`);
- the shared temp directory: temp-touch (the directory itself, or a file only listed or named),
  temp-write (a file written there), and reads of a file, classified by whoever last wrote it in
  timestamp order over every scanned run: own-temp (the same agent in the same epoch),
  cross-forget (the same agent before a forget), cross-agent (another agent of the run; the
  detail says so when the run's agents are meant to be isolated, `team.peers = "none"`),
  cross-run (another run) and shared-temp (no scanned run wrote it; a file the agent's own
  program wrote is invisible here, so this one can be a false alarm).

temp-touch, temp-write, own-temp and claude-output are benign (BENIGN_KINDS); the other kinds are
leaks (LEAK_KINDS). The command prints a count per run and kind, then every leak, and exits 1
when there is one. `research.analyze` shows the leaks per run as a validity column.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from research.leak_scan import (
    BENIGN_KINDS,
    KINDS,
    LEAK_KINDS,
    TEMP_PREFIX,
    LeakFinding,
    LeakKind,
    RunScan,
    run_dirs,
)
from research.runs import EVENTS_FILE
from research.textfmt import format_table

__all__ = [
    "BENIGN_KINDS",
    "KINDS",
    "LEAK_KINDS",
    "LeakFinding",
    "LeakKind",
    "leak_counts",
    "main",
    "render_findings",
    "scan_experiment",
    "scan_run",
]

_GLOB = re.compile(r"[*?\[]")
_RANK: dict[LeakKind, int] = {"own-temp": 0, "cross-forget": 1, "cross-agent": 2, "cross-run": 3}
_COMMAND_WIDTH = 120


# ---------------------------------------------------------------------------------------------
# provenance of shared temp files
# ---------------------------------------------------------------------------------------------


def _relation(reader: RunScan, read: LeakFinding, writer: RunScan, write: LeakFinding) -> LeakKind:
    if writer is not reader:
        return "cross-run"
    if write.agent != read.agent:
        return "cross-agent"
    return "cross-forget" if write.epoch != read.epoch else "own-temp"


def _attribute(
    reader: RunScan, read: LeakFinding, writers: Sequence[tuple[RunScan, LeakFinding]]
) -> LeakFinding:
    """`read` classified by the last writers of the file(s) it names; the farthest one counts."""
    if not writers:
        return replace(read, detail="no scanned run wrote it")
    kind, writer, write = max(
        ((_relation(reader, read, run, write), run, write) for run, write in writers),
        key=lambda candidate: _RANK[candidate[0]],
    )
    if kind == "own-temp":
        return replace(read, kind=kind)
    who = "" if writer is reader else f"{writer.label} "
    detail = f"last written by {who}{write.agent} e{write.epoch}"
    if kind == "cross-agent" and reader.peers == "none":
        detail += "; the run's agents are meant to be isolated (team.peers = none)"
    return replace(read, kind=kind, detail=detail)


def _trace(runs: Sequence[RunScan], reported: int) -> dict[str, list[LeakFinding]]:
    """Findings of the first `reported` runs, keyed by name, with their reads of shared temp
    files attributed to the files' last writers among all `runs`, in timestamp order."""
    timeline = sorted(
        (finding.ts, i, j)
        for i, run in enumerate(runs)
        for j, finding in enumerate(run.findings)
        if finding.kind in ("temp-write", "shared-temp")
    )
    last: dict[str, tuple[RunScan, LeakFinding]] = {}  # file -> its last writer so far
    traced = [list(run.findings) for run in runs[:reported]]
    for _, i, j in timeline:
        run, finding = runs[i], runs[i].findings[j]
        key = finding.path.removeprefix(TEMP_PREFIX)
        if finding.kind == "temp-write":
            last[key] = (run, finding)
        elif i < reported:
            if _GLOB.search(key):
                writers = [w for name, w in last.items() if fnmatch.fnmatchcase(name, key)]
            else:
                writers = [last[key]] if key in last else []
            traced[i][j] = _attribute(run, finding, writers)
    return {run.name: findings for run, findings in zip(runs, traced, strict=False)}


# ---------------------------------------------------------------------------------------------
# scanning runs and experiments
# ---------------------------------------------------------------------------------------------


def _same_dir(a: Path, b: Path) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def _label(run_dir: Path, experiment_dir: Path) -> str:
    """A run's name: `<arm>/<run>` in `experiment_dir`, `<experiment>/<arm>/<run>` elsewhere."""
    parts = run_dir.absolute().parts
    return "/".join(parts[-2:] if _same_dir(run_dir.parent.parent, experiment_dir) else parts[-3:])


def _scans(
    paths: Iterable[Path], experiment_dir: Path, runs_root: Path | None, skip: Sequence[Path] = ()
) -> list[RunScan]:
    """A RunScan per distinct run directory in `paths`, leaving out those in `skip`."""
    seen = {os.path.realpath(path) for path in skip}
    scans: list[RunScan] = []
    for path in paths:
        if os.path.realpath(path) not in seen:
            seen.add(os.path.realpath(path))
            label = _label(path, experiment_dir)
            scans.append(RunScan(path, name=label, label=label, runs_root=runs_root))
    return scans


def _all_runs(experiment_dir: Path, runs_root: Path | None) -> list[Path]:
    """Every run directory of every experiment under the runs root of `experiment_dir`."""
    named = (p for p in [experiment_dir.absolute(), *experiment_dir.absolute().parents])
    root = runs_root or next((p for p in named if p.name == "runs"), experiment_dir.parent)
    return [run for exp in sorted(root.iterdir()) if exp.is_dir() for run in run_dirs(exp)]


def scan_run(
    run_dir: Path | str,
    *,
    runs_root: Path | str | None = None,
    writers: Iterable[Path | str] = (),
) -> list[LeakFinding]:
    """Findings of one run, in time order. Its reads of shared temp files are traced over its own
    writes and those of the runs in `writers` (run directories)."""
    run_dir = Path(run_dir)
    root = None if runs_root is None else Path(runs_root)
    experiment_dir = run_dir.parent.parent
    (run,) = _scans([run_dir], experiment_dir, root)
    others = _scans(map(Path, writers), experiment_dir, root, skip=[run_dir])
    return _trace([run, *others], reported=1)[run.name]


def scan_experiment(
    experiment_dir: Path | str,
    *,
    runs_root: Path | str | None = None,
    all_experiments: bool = False,
) -> dict[str, list[LeakFinding]]:
    """Findings of every run under `experiment_dir` (parked incomplete runs included), keyed by
    "<arm>/<run>". Reads of shared temp files are traced over the writes of all those runs and,
    with `all_experiments`, of every run under the runs root."""
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"no such experiment directory: {experiment_dir}")
    root = None if runs_root is None else Path(runs_root)
    own = run_dirs(experiment_dir)
    runs = _scans(own, experiment_dir, root)
    if all_experiments:
        runs += _scans(_all_runs(experiment_dir, root), experiment_dir, root, skip=own)
    return _trace(runs, reported=len(own))


def leak_counts(experiment_dir: Path | str) -> dict[str, int]:
    """Leaks (findings that are not benign) per run of `experiment_dir`, keyed "<arm>/<run>"."""
    found = scan_experiment(experiment_dir)
    return {name: sum(not f.benign for f in findings) for name, findings in found.items()}


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _short(command: str) -> str:
    lines = command.strip().splitlines() or [""]
    text = lines[0]
    if len(text) > _COMMAND_WIDTH or len(lines) > 1:
        text = text[:_COMMAND_WIDTH] + " ..."
    return text


def render_findings(found: Mapping[str, Sequence[LeakFinding]], *, title: str) -> str:
    """Counts per run and kind (kinds that occur), then each leak with its command."""
    kinds = [k for k in KINDS if any(f.kind == k for fs in found.values() for f in fs)]
    rows = []
    for name, findings in found.items():
        counts = Counter(f.kind for f in findings)
        leaks = sum(not f.benign for f in findings)
        rows.append([name, str(leaks), *(str(counts[k]) for k in kinds)])
    benign = ", ".join(k for k in KINDS if k in BENIGN_KINDS)
    sections = [f"{title}\n{format_table(['run', 'leaks', *kinds], rows)}\nbenign: {benign}"]
    leaks = [f for findings in found.values() for f in findings if not f.benign]
    if leaks:
        lines = ["Leaks"]
        for f in leaks:
            when = datetime.fromtimestamp(f.ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            who = f"{f.run} {f.agent} e{f.epoch} {when}"
            detail = f" ({f.detail})" if f.detail else ""
            lines.append(f"{who} {f.kind}: {f.access} {f.path}{detail}")
            lines.append(f"    {_short(f.command)}  [{f.transcript}]")
        sections.append("\n".join(lines))
        leaky = sum(any(not f.benign for f in fs) for fs in found.values())
        tally = f"{_plural(len(leaks), 'leak')} in {leaky} of {_plural(len(found), 'run')}"
    else:
        tally = f"no leaks in {_plural(len(found), 'run')}"
    return "\n\n".join([*sections, tally]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research.leaks",
        description="Tool calls in the transcripts that crossed a boundary: another run, agent or"
        " forgotten epoch, the run's own state, or Claude Code's logs. Exits 1 when there is one.",
    )
    parser.add_argument("path", type=Path, help="runs_root/<experiment>, or one run in it")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="where all runs live (default: the nearest directory above named 'runs')",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="trace shared temp files to the runs of every experiment under the runs root",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path: Path = args.path
    try:
        if (path / EVENTS_FILE).is_file() or (path / "transcripts").is_dir():
            experiment_dir = path.parent.parent
            writers = (
                _all_runs(experiment_dir, args.runs_root)
                if args.all_experiments
                else run_dirs(experiment_dir)
            )
            findings = scan_run(path, runs_root=args.runs_root, writers=writers)
            found = {_label(path, experiment_dir): findings}
        else:
            found = scan_experiment(
                path, runs_root=args.runs_root, all_experiments=args.all_experiments
            )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    title = f"Leaks in {path}: tool calls that named a path outside the agent's own workspace"
    print(render_findings(found, title=title), end="")
    return 1 if any(not f.benign for findings in found.values() for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
