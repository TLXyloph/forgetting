"""Fixtures for the leak-scan tests: runs with synthetic Claude Code transcripts.

Runs live under `<tmp>/runs/<experiment>/<arm>/rep<k>`, so the scanner finds the runs root by its
name. Each run has a schema-valid events.jsonl (written by `run_builder`) whose config carries
`team.peers`, and transcripts written by `TranscriptBuilder`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

T0 = 1_800_000_000.0  # 2027-01-15, seconds since the epoch


def iso(ts: float) -> str:
    """A timestamp the way Claude Code writes it: "2027-01-15T08:00:00.000Z"."""
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TranscriptBuilder:
    """Writes stream-json transcripts like lethe's, one file per agent turn:
    `<run_dir>/transcripts/<agent>/e<epoch>/<session>-t<turn>.jsonl`.

    A new file starts with the `system/init` line (its `cwd` is the agent's workspace). Each call
    appends an assistant `tool_use` line and the user `tool_result` line answering it; `ts` is
    seconds after T0.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._ids = 0

    def workspace(self, agent: str, epoch: int) -> Path:
        return self.run_dir / "ws" / agent / f"e{epoch}"

    def path(self, agent: str, epoch: int, session: str = "s1", turn: int = 1) -> Path:
        return self.run_dir / "transcripts" / agent / f"e{epoch}" / f"{session}-t{turn}.jsonl"

    def lines(
        self, agent: str, epoch: int, *lines: str, session: str = "s1", turn: int = 1
    ) -> Path:
        """Append raw lines (the init line first, if the file is new)."""
        path = self.path(agent, epoch, session, turn)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            init = {
                "type": "system",
                "subtype": "init",
                "cwd": str(self.workspace(agent, epoch)),
                "session_id": session,
                "tools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            }
            path.write_text(json.dumps(init) + "\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in lines)
        return path

    def call(
        self,
        agent: str,
        epoch: int,
        tool: str,
        tool_input: Mapping[str, Any],
        *,
        ts: float,
        is_error: bool = False,
        result: Any = "ok",
        session: str = "s1",
        turn: int = 1,
    ) -> str:
        """One tool call and its result; returns the tool_use id."""
        self._ids += 1
        call_id = f"toolu_{self._ids:04d}"
        use = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": call_id, "name": tool, "input": dict(tool_input)}
                ],
            },
            "session_id": session,
            "timestamp": iso(T0 + ts),
        }
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": result}
        if is_error:
            block["is_error"] = True
        answer = {
            "type": "user",
            "message": {"role": "user", "content": [block]},
            "session_id": session,
            "timestamp": iso(T0 + ts + 0.5),
        }
        self.lines(agent, epoch, json.dumps(use), json.dumps(answer), session=session, turn=turn)
        return call_id

    def bash(self, agent: str, epoch: int, command: str, *, ts: float, **kwargs: Any) -> str:
        return self.call(agent, epoch, "Bash", {"command": command}, ts=ts, **kwargs)

    def read(self, agent: str, epoch: int, file_path: Path | str, *, ts: float, **kw: Any) -> str:
        return self.call(agent, epoch, "Read", {"file_path": str(file_path)}, ts=ts, **kw)


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def make_run(run_builder: type, runs_root: Path) -> Callable[..., TranscriptBuilder]:
    """`make_run("solo", 1, peers="none")`: a finished run, without transcripts yet, at
    `<runs_root>/<experiment>/<arm>/rep<k>`; returns its TranscriptBuilder."""

    def make(
        arm: str,
        rep: int,
        *,
        peers: str = "all",
        agents: Sequence[str] = ("a1", "a2"),
        experiment: str = "exp",
    ) -> TranscriptBuilder:
        run_dir = runs_root / experiment / arm / f"rep{rep}"
        config = {
            "budget": {"usd": 3.0, "proposals": 0, "primary": "usd"},
            "team": {"agents": len(agents), "peers": peers, "session": "continue"},
        }
        b = run_builder(run_dir, agents=tuple(agents), config=config).start(100.0)
        b.candidate(agents[0], 110.0)
        b.finish()
        return TranscriptBuilder(run_dir)

    return make
