"""What one run's Claude Code transcripts show: tool calls that named a path outside the agent's
own workspace, classified by where the path is.

`research.leaks` documents the kinds, attributes reads of shared temp files to their writers (a
read is "shared-temp" here until then) and is the entry point. Transcripts are the stream-json
files `<run_dir>/transcripts/<agent>/e<epoch>/<session>-t<turn>.jsonl`; a call counts once its
`tool_result` arrives without `is_error`. Only lines that mention a tool call, or the result of a
call that named an outside path, are parsed, which keeps runs of thousands of transcripts cheap.
Malformed lines are skipped.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from research.leak_paths import Access, canonical, is_within, temp_key, tool_paths
from research.runs import EVENTS_FILE

__all__ = [
    "BENIGN_KINDS",
    "KINDS",
    "LEAK_KINDS",
    "TEMP_PREFIX",
    "LeakFinding",
    "LeakKind",
    "RunScan",
    "natural_key",
    "run_dirs",
]

LeakKind = Literal[
    "cross-run", "cross-agent", "cross-forget", "shared-temp", "other-workspace", "other-run",
    "run-state", "claude-logs", "temp-write", "temp-touch", "own-temp", "claude-output",
]  # fmt: skip
KINDS: tuple[LeakKind, ...] = (
    "cross-run", "cross-agent", "cross-forget", "shared-temp", "other-workspace", "other-run",
    "run-state", "claude-logs", "temp-write", "temp-touch", "own-temp", "claude-output",
)  # fmt: skip
BENIGN_KINDS: frozenset[LeakKind] = frozenset(KINDS[-4:])
LEAK_KINDS: frozenset[LeakKind] = frozenset(KINDS[:-4])

TEMP_PREFIX = "$TMPDIR/"  # how findings show files in the shared temp directory
_CLAUDE_DIR = re.compile(r"(?:/(?:Users|home)/[^/]+|/root)/\.claude(?:/|$)")
_WORKSPACE = re.compile(r"(.+)/ws/[^/]+/e[0-9]+/*")
_EPOCH_DIR = re.compile(r"e([0-9]+)")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeakFinding:
    """A successful tool call that named a path outside the agent's workspace (research.leaks)."""

    run: str  # "<arm>/<run directory>", e.g. "solo/rep1"
    agent: str
    epoch: int
    ts: float  # when the call was made, seconds since the epoch
    kind: LeakKind
    path: str  # canonical and absolute; files in the shared temp directory as "$TMPDIR/<name>"
    command: str  # the Bash command, or "<tool> <path...>"
    access: Access = "read"
    detail: str = ""  # reads of shared temp files: who last wrote the file
    transcript: str = ""  # the transcript file, relative to the run directory

    @property
    def benign(self) -> bool:
        return self.kind in BENIGN_KINDS


def natural_key(name: str) -> list[tuple[str, int]]:
    """Sort key that puts rep2 before rep10."""
    return [(text, int(digits or 0)) for text, digits in re.findall(r"([^0-9]*)([0-9]*)", name)]


def run_dirs(experiment_dir: Path) -> list[Path]:
    """`<arm>/<run>` directories with an event log or transcripts, parked runs included."""
    return [
        run_dir
        for arm_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir())
        for run_dir in sorted(arm_dir.iterdir(), key=lambda p: natural_key(p.name))
        if (run_dir / EVENTS_FILE).is_file() or (run_dir / "transcripts").is_dir()
    ]


def _record(line: str | bytes) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except ValueError:  # UnicodeDecodeError included
        return None
    return record if isinstance(record, dict) else None


def _first_line(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return _record(fh.readline()) or {}
    except OSError:
        return {}


def _blocks(record: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        yield from (block for block in content if isinstance(block, Mapping))


def _timestamp(record: Mapping[str, Any]) -> float | None:
    value = record.get("timestamp")
    try:
        return datetime.fromisoformat(value).timestamp() if isinstance(value, str) else None
    except ValueError:
        return None


def _transcripts(run_dir: Path) -> Iterator[tuple[str, int, Path]]:
    """(agent, epoch, path) of each `transcripts/<agent>/e<epoch>/*.jsonl`."""
    root = run_dir / "transcripts"
    if not root.is_dir():
        return
    for agent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for epoch_dir in sorted(agent_dir.iterdir(), key=lambda p: natural_key(p.name)):
            match = _EPOCH_DIR.fullmatch(epoch_dir.name)
            if match and epoch_dir.is_dir():
                for path in sorted(epoch_dir.glob("*.jsonl")):
                    yield agent_dir.name, int(match.group(1)), path


def _peers(run_dir: Path) -> str | None:
    """`team.peers` in the run's config (the run_started event on the log's first line)."""
    config = _first_line(run_dir / EVENTS_FILE).get("config")
    team = config.get("team") if isinstance(config, Mapping) else None
    peers = team.get("peers") if isinstance(team, Mapping) else None
    return peers if isinstance(peers, str) else None


def _init_cwd(path: Path) -> str | None:
    """The working directory a transcript's `system/init` line records."""
    record = _first_line(path)
    cwd = record.get("cwd") if record.get("subtype") == "init" else None
    return cwd if isinstance(cwd, str) else None


def _spellings(path: Path | str) -> list[str]:
    """Absolute spellings of `path`: as given, and with symlinks resolved."""
    return list(dict.fromkeys([os.path.abspath(path), os.path.realpath(path)]))


def _runs_roots(roots: Sequence[str], runs_root: Path | None) -> list[str]:
    """The runs root: given, else the nearest directory named `runs` above each spelling of the
    run, else the experiment directory."""
    if runs_root is not None:
        return list(dict.fromkeys(canonical(s) for s in _spellings(runs_root)))
    found: list[str] = []
    for root in roots:
        parts = root.split("/")
        named = [i for i in range(len(parts) - 1) if parts[i] == "runs"]
        found.append("/".join(parts[: named[-1] + 1]) if named else "/".join(parts[:-2]) or "/")
    return list(dict.fromkeys(found))


class RunScan:
    """One run directory, under every spelling of its path, and the findings of its transcripts.

    `name` keys the run in results, `label` names it in the details of other runs' findings.
    """

    def __init__(
        self, run_dir: Path, *, name: str, label: str, runs_root: Path | None = None
    ) -> None:
        self.dir = run_dir
        self.name = name
        self.label = label
        self.peers = _peers(run_dir)
        self.home = str(Path.home())
        self.claude_dir = canonical(self.home) + "/.claude"
        spellings = _spellings(run_dir)
        cwd = next((_init_cwd(path) for _, _, path in _transcripts(run_dir)), None)
        moved = _WORKSPACE.fullmatch(cwd) if cwd else None
        if moved:  # where the run was when its agents ran, should it have moved since
            spellings.append(moved.group(1))
        self.roots = list(dict.fromkeys(canonical(s) for s in spellings))
        self.slug_bases = list(dict.fromkeys([*spellings, *self.roots]))
        self.runs_roots = _runs_roots(self.roots, runs_root)
        self._slugs: dict[tuple[str, int], frozenset[str]] = {}
        self.findings = self._scan()

    def slugs(self, agent: str, epoch: int) -> frozenset[str]:
        """Claude Code's names for the agent's workspace: `/`, or every non-alphanumeric, as `-`."""
        if (agent, epoch) not in self._slugs:
            spaces = [f"{base}/ws/{agent}/e{epoch}" for base in self.slug_bases]
            self._slugs[agent, epoch] = frozenset(
                slug
                for ws in spaces
                for slug in (ws.replace("/", "-"), re.sub(r"[^A-Za-z0-9]", "-", ws))
            )
        return self._slugs[agent, epoch]

    def classify(
        self, raw: str, access: Access, agent: str, epoch: int
    ) -> tuple[LeakKind, str] | None:
        """(kind, path shown) of a path the agent named; None inside its own workspace and for
        paths in no category (system files, say)."""
        own = f"ws/{agent}/e{epoch}"
        path = canonical(raw if raw.startswith("/") else f"{self.roots[0]}/{own}/{raw}")
        for root in self.roots:
            if is_within(path, root):
                rel = path[len(root) :].strip("/")
                if is_within(rel, own):
                    return None
                return ("other-workspace" if is_within(rel, "ws") else "run-state"), path
        if any(is_within(path, root) for root in self.runs_roots):
            return "other-run", path
        parts = set(path.split("/"))
        own_slug = not parts.isdisjoint(self.slugs(agent, epoch))
        if _CLAUDE_DIR.match(path) or is_within(path, self.claude_dir):
            persisted = own_slug and "tool-results" in parts  # output the CLI pointed it to
            return ("claude-output" if persisted else "claude-logs"), path
        key = temp_key(path)
        if key is None:
            return None
        shown = TEMP_PREFIX + key if key else TEMP_PREFIX.rstrip("/")
        if own_slug:
            return "claude-output", shown
        if not key or access == "touch":
            return "temp-touch", shown
        return ("temp-write" if access == "write" else "shared-temp"), shown

    def _scan(self) -> list[LeakFinding]:
        findings: list[LeakFinding] = []
        for agent, epoch, path in _transcripts(self.dir):
            findings += self._scan_file(agent, epoch, path)
        findings.sort(key=lambda f: f.ts)  # stable: a call's findings keep their order
        return findings

    def _scan_file(self, agent: str, epoch: int, path: Path) -> list[LeakFinding]:
        transcript = path.relative_to(self.dir).as_posix()
        pending: dict[str, list[LeakFinding]] = {}  # tool_use id -> findings awaiting its result
        found: list[LeakFinding] = []
        try:
            with path.open("rb") as fh:
                for line in fh:
                    if b'"tool_use"' not in line and not (
                        pending
                        and b'"tool_result"' in line
                        and any(key.encode() in line for key in pending)
                    ):
                        continue
                    record = _record(line) or {}
                    for block in _blocks(record):
                        kind = record.get("type"), block.get("type")
                        if kind == ("assistant", "tool_use") and isinstance(block.get("id"), str):
                            ts = _timestamp(record)
                            ts = path.stat().st_mtime if ts is None else ts
                            calls = self._call(block, agent, epoch, ts, transcript)
                            if calls:
                                pending[block["id"]] = calls
                        elif kind == ("user", "tool_result"):
                            call_id = block.get("tool_use_id")
                            calls = pending.pop(call_id, []) if isinstance(call_id, str) else []
                            if block.get("is_error") is not True:
                                found += calls
        except OSError as exc:
            log.warning("cannot read %s: %s", path, exc)
        return found

    def _call(
        self, block: Mapping[str, Any], agent: str, epoch: int, ts: float, transcript: str
    ) -> list[LeakFinding]:
        """The findings of one tool_use block: one per distinct (kind, access, path)."""
        name, tool_input = block.get("name"), block.get("input")
        if not isinstance(name, str) or not isinstance(tool_input, Mapping):
            return []
        hits: dict[tuple[LeakKind, Access, str], None] = {}  # in order, without repeats
        for raw, access in tool_paths(name, tool_input, self.home):
            hit = self.classify(raw, access, agent, epoch)
            if hit is not None:
                hits[hit[0], access, hit[1]] = None
        if not hits:
            return []
        command = tool_input.get("command")
        if name != "Bash" or not isinstance(command, str):
            keys = ("file_path", "notebook_path", "path", "pattern")
            named = [v for v in (tool_input.get(k) for k in keys) if isinstance(v, str)]
            command = " ".join([name, *named])
        return [
            LeakFinding(self.name, agent, epoch, ts, kind, shown, command, access, "", transcript)
            for kind, access, shown in hits
        ]
