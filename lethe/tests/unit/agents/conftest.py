"""Shared fixtures for the agents tests: recorded streams and a fake `claude` executable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lethe.agents.claude_code import ClaudeCodeBackend, ClaudeCodeOptions

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude"
FAKE_SOURCE = Path(__file__).with_name("fake_claude.py")


class FakeClaude:
    """A real executable that records how it was called and replays a recorded stream."""

    def __init__(self, path: Path, record_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = path
        self.record_dir = record_dir
        self._monkeypatch = monkeypatch
        monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(record_dir))

    def script(
        self,
        *,
        stream: str | Path | None = None,
        exit_code: int = 0,
        stderr: str | None = None,
        stderr_file: Path | None = None,
        hang_s: float | None = None,
        child: bool = False,
        big_line: int | None = None,
        strip_newline: bool = False,
        skip_stdin: bool = False,
        self_kill: bool = False,
    ) -> FakeClaude:
        """Choose what the next invocations do.

        `stream` is a file name in fixtures/claude, or an absolute Path to an ad-hoc stream.
        """
        env = {
            "FAKE_CLAUDE_STREAM": str(FIXTURES / stream) if stream else None,
            "FAKE_CLAUDE_EXIT": str(exit_code),
            "FAKE_CLAUDE_STDERR": stderr,
            "FAKE_CLAUDE_STDERR_FILE": None if stderr_file is None else str(stderr_file),
            "FAKE_CLAUDE_HANG": None if hang_s is None else str(hang_s),
            "FAKE_CLAUDE_CHILD": "group" if child else None,
            "FAKE_CLAUDE_BIG_LINE": None if big_line is None else str(big_line),
            "FAKE_CLAUDE_STRIP_NEWLINE": "1" if strip_newline else None,
            "FAKE_CLAUDE_SKIP_STDIN": "1" if skip_stdin else None,
            "FAKE_CLAUDE_SELF_KILL": "1" if self_kill else None,
        }
        for name, value in env.items():
            if value is None:
                self._monkeypatch.delenv(name, raising=False)
            else:
                self._monkeypatch.setenv(name, value)
        return self

    def options(self, **overrides: object) -> ClaudeCodeOptions:
        return ClaudeCodeOptions(executable=str(self.path), **overrides)  # type: ignore[arg-type]

    def backend(self, **overrides: object) -> ClaudeCodeBackend:
        return ClaudeCodeBackend(self.options(**overrides))

    # what the fake saw ---------------------------------------------------------------------

    @property
    def called(self) -> bool:
        return (self.record_dir / "argv.json").exists()

    @property
    def argv(self) -> list[str]:
        return json.loads((self.record_dir / "argv.json").read_text(encoding="utf-8"))

    @property
    def cwd(self) -> Path:
        return Path((self.record_dir / "cwd.txt").read_text(encoding="utf-8"))

    @property
    def stdin(self) -> bytes:
        return (self.record_dir / "stdin.bin").read_bytes()

    @property
    def child_pid_file(self) -> Path:
        return self.record_dir / "child.pid"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    # neither the file nor its parent directories exist yet
    return tmp_path / "runs" / "agent-1" / "transcript.jsonl"


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(f"#!{sys.executable}\n" + FAKE_SOURCE.read_text(encoding="utf-8"), "utf-8")
    exe.chmod(0o755)
    return FakeClaude(exe, tmp_path / "record", monkeypatch)
