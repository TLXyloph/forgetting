"""`ClaudeCodeBackend.run_turn` lifecycle: timeouts, cancellation, process groups, concurrency.

Uses the same fake `claude` executable as test_claude_code_run.py (see conftest.py and
fake_claude.py). The real CLI is never started.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from lethe.agents.claude_code import ClaudeCodeBackend
from lethe.domain.types import SessionChoice, TurnResult

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude"
SID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
NEW = SessionChoice(SID, resume=False)


# --- helpers --------------------------------------------------------------------------------


def run(
    backend: ClaudeCodeBackend,
    workspace: Path,
    transcript: Path,
    *,
    prompt: str = "Improve the solver.",
    session: SessionChoice = NEW,
) -> TurnResult:
    return asyncio.run(
        backend.run_turn(
            agent="agent-1",
            workspace=workspace,
            prompt=prompt,
            session=session,
            transcript_path=transcript,
        )
    )


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A killed child that nobody has reaped yet is a zombie: dead for our purposes.
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():  # Linux
        with contextlib.suppress(OSError, IndexError):
            return proc_stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    if shutil.which("ps"):  # macOS and friends
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        return bool(state) and not state.startswith("Z")
    return True  # cannot tell: assume it is still running


def wait_until_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


def kill_leftover_child(fake_claude: Any) -> None:
    """Never leave the fake's sleeping grandchild behind, even when an assertion failed."""
    pid_file: Path = fake_claude.child_pid_file
    if pid_file.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


# --- timeout and cancellation -----------------------------------------------------------------


def test_timeout_kills_the_turn_returns_promptly_and_beats_every_other_error_class(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    # rate_limited.jsonl is a complete usage-limit stream and the exit status is nonzero, yet
    # the fake then sleeps 30 s: the timeout must win over rate_limit/api/process.
    fake_claude.script(stream="rate_limited.jsonl", hang_s=30, exit_code=1)
    backend = fake_claude.backend(turn_timeout_s=1.0)

    started = time.monotonic()
    result = run(backend, workspace, transcript)
    elapsed = time.monotonic() - started

    assert elapsed < 10
    assert (result.ok, result.error_kind) == (False, "timeout")
    assert "1" in (result.error or "")
    assert 0.9 <= result.duration_s < 10


def test_timeout_keeps_the_partial_stream_and_kills_the_whole_process_group(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="no_result.jsonl", hang_s=30, child=True)
    backend = fake_claude.backend(turn_timeout_s=2.0)  # roomy: the fake must print and spawn first

    try:
        result = run(backend, workspace, transcript)

        assert result.error_kind == "timeout"
        # what the stream showed before the kill is kept ...
        assert result.session_id == "5a1d0000-0000-4000-8000-000000000004"
        assert result.final_text == "I am about to edit the solver."
        assert [c.name for c in result.tool_calls] == ["Read", "Edit"]
        assert transcript.read_bytes() == (FIXTURES / "no_result.jsonl").read_bytes()
        # ... and nothing the agent started outlives the turn
        child = int(fake_claude.child_pid_file.read_text())
        assert wait_until_dead(child), "the grandchild survived the timeout kill"
    finally:
        kill_leftover_child(fake_claude)


def test_cancellation_kills_the_process_group_and_reraises(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="no_result.jsonl", hang_s=30, child=True)
    backend = fake_claude.backend(turn_timeout_s=120.0)

    async def scenario() -> int:
        task = asyncio.ensure_future(
            backend.run_turn(
                agent="agent-1",
                workspace=workspace,
                prompt="go",
                session=NEW,
                transcript_path=transcript,
            )
        )
        deadline = time.monotonic() + 20
        while not fake_claude.child_pid_file.exists():
            assert time.monotonic() < deadline, "the fake never got as far as spawning its child"
            await asyncio.sleep(0.02)
        child = int(fake_claude.child_pid_file.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return child

    try:
        started = time.monotonic()
        child = asyncio.run(scenario())
        assert time.monotonic() - started < 20
        assert wait_until_dead(child), "the grandchild survived cancellation"
    finally:
        kill_leftover_child(fake_claude)


# --- concurrency ------------------------------------------------------------------------------


def test_concurrent_turns_do_not_interfere(
    fake_claude: Any, workspace: Path, tmp_path: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl")
    backend = fake_claude.backend()
    transcripts = [tmp_path / "t" / f"{i}.jsonl" for i in range(4)]

    async def scenario() -> list[TurnResult]:
        return list(
            await asyncio.gather(
                *(
                    backend.run_turn(
                        agent=f"agent-{i}",
                        workspace=workspace,
                        prompt=f"prompt {i}",
                        session=SessionChoice(f"sess-{i}", resume=False),
                        transcript_path=path,
                    )
                    for i, path in enumerate(transcripts)
                )
            )
        )

    results = asyncio.run(scenario())

    assert all(r.ok and r.cost_usd == pytest.approx(0.00091024) for r in results)
    expected = (FIXTURES / "turn1.jsonl").read_bytes()
    assert all(p.read_bytes() == expected for p in transcripts)
