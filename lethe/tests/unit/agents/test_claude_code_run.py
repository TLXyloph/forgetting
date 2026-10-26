"""`ClaudeCodeBackend.run_turn` against a fake `claude` executable (see conftest.py/fake_claude.py).

The fake is a real subprocess that records argv/cwd/stdin and replays a recorded stream, so these
tests cover the adapter itself: spawning, stdin, line streaming, timeouts and process groups.
The real CLI is never started.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from lethe.agents import claude_code
from lethe.agents.claude_code import ClaudeCodeBackend, ClaudeCodeOptions
from lethe.domain.types import RateLimitInfo, SessionChoice, ToolCall, TurnResult, Usage

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude"
SID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
NEW = SessionChoice(SID, resume=False)
RESUME = SessionChoice(SID, resume=True)
TURN1_SID = "a76c6362-2811-4c08-ad32-d8222b63afe8"


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


# --- success --------------------------------------------------------------------------------


def test_success_maps_the_stream_onto_the_turn_result(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl")

    result = run(fake_claude.backend(), workspace, transcript)

    assert result.ok is True
    assert result.error_kind is None and result.error is None
    assert result.session_id == TURN1_SID  # the stream's id wins over the requested one
    assert result.final_text == 'I changed "hello" to "hello there" in hello.txt.'
    assert result.cost_usd == pytest.approx(0.00091024)
    assert result.usage == Usage(
        input_tokens=6,
        output_tokens=342,
        cache_read_tokens=16344,
        cache_creation_tokens=2876,
        thinking_tokens=0,
    )
    assert result.tool_calls == (
        ToolCall("Read", "/WORKSPACE/hello.txt"),
        ToolCall("Edit", "/WORKSPACE/hello.txt"),
    )
    assert result.rate_limit == RateLimitInfo("allowed", 0.16, 0.07, 1793050200.0)
    assert result.num_api_turns == 3
    assert (result.files_read, result.edit_calls) == (1, 1)
    assert 0 < result.duration_s < 30


def test_resumed_turn_cost_is_the_per_turn_cost(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn2.jsonl")
    result = run(fake_claude.backend(), workspace, transcript, session=RESUME)
    assert result.ok
    assert result.cost_usd == pytest.approx(0.00032054)  # not the cumulative 0.00123078


def test_session_id_falls_back_to_the_requested_one(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream=None, exit_code=1, stderr="nothing on stdout")
    assert run(fake_claude.backend(), workspace, transcript).session_id == SID


# --- how the process is launched -------------------------------------------------------------


def test_prompt_arrives_on_stdin_and_never_in_argv(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    prompt = 'Fix solver.py -- it\'s "quoted"\nsecond line with $VARS and `ticks` and 世界 --flag'
    fake_claude.script(stream="turn1.jsonl")

    run(fake_claude.backend(), workspace, transcript, prompt=prompt)

    assert fake_claude.stdin == prompt.encode("utf-8")
    assert all(prompt not in arg and "solver.py" not in arg for arg in fake_claude.argv)


@pytest.mark.parametrize("session", [NEW, RESUME], ids=["new-session", "resume"])
def test_the_process_gets_exactly_build_argv(
    fake_claude: Any, workspace: Path, transcript: Path, session: SessionChoice
) -> None:
    fake_claude.script(stream="turn1.jsonl")
    backend = fake_claude.backend(model="haiku", effort="low", extra_args=("--name", "t"))

    run(backend, workspace, transcript, session=session)

    assert fake_claude.argv == backend.build_argv(session)[1:]
    flag = "--resume" if session.resume else "--session-id"
    assert fake_claude.argv[fake_claude.argv.index(flag) + 1] == SID


def test_the_process_runs_in_the_workspace(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl")
    run(fake_claude.backend(), workspace, transcript)
    assert fake_claude.cwd.resolve() == workspace.resolve()


def test_a_prompt_larger_than_the_pipe_buffers_is_delivered_intact(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    prompt = ("0123456789abcdef" * 64 + "\n") * 1024  # ~1 MiB
    fake_claude.script(stream="turn1.jsonl")

    result = run(fake_claude.backend(), workspace, transcript, prompt=prompt)

    assert result.ok
    assert fake_claude.stdin == prompt.encode("utf-8")


def test_a_child_that_never_reads_stdin_neither_hangs_nor_crashes_the_backend(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream=None, exit_code=1, stderr="refusing input", skip_stdin=True)
    started = time.monotonic()

    result = run(fake_claude.backend(), workspace, transcript, prompt="x" * (4 * 1024 * 1024))

    assert time.monotonic() - started < 15
    assert (result.ok, result.error_kind) == (False, "process")
    assert "refusing input" in (result.error or "")


def test_the_executable_is_resolved_like_a_normal_command(
    fake_claude: Any, workspace: Path, transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `executable="claude"` (the default) means "look it up on PATH"
    fake_claude.script(stream="turn1.jsonl")
    monkeypatch.setenv("PATH", f"{fake_claude.path.parent}{os.pathsep}{os.environ['PATH']}")
    assert run(ClaudeCodeBackend(), workspace, transcript).ok


def test_missing_executable_raises_file_not_found(
    tmp_path: Path, workspace: Path, transcript: Path
) -> None:
    backend = ClaudeCodeBackend(ClaudeCodeOptions(executable=str(tmp_path / "nope" / "claude")))
    with pytest.raises(FileNotFoundError):
        run(backend, workspace, transcript)

    on_path = ClaudeCodeBackend(ClaudeCodeOptions(executable="lethe-no-such-claude-binary"))
    with pytest.raises(FileNotFoundError):
        run(on_path, workspace, transcript)


def test_the_environment_is_never_logged(
    fake_claude: Any,
    workspace: Path,
    transcript: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret = "hunter2-do-not-log"
    monkeypatch.setenv("LETHE_TEST_SECRET", secret)
    caplog.set_level(logging.DEBUG)
    seen: list[str] = []
    for script in (
        {"stream": "turn1.jsonl"},
        {"stream": "api_error.jsonl", "exit_code": 1},
        {"stream": None, "exit_code": 1, "stderr": "dead"},
    ):
        fake_claude.script(**script)
        seen.append(repr(run(fake_claude.backend(), workspace, transcript)))

    out, err = capfd.readouterr()
    assert secret not in caplog.text and secret not in out and secret not in err
    assert not any(secret in text for text in seen)


# --- transcript ------------------------------------------------------------------------------


def test_transcript_holds_the_raw_stdout_lines_and_its_parents_are_created(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl")
    assert not transcript.parent.exists()

    run(fake_claude.backend(), workspace, transcript)

    assert transcript.read_bytes() == (FIXTURES / "turn1.jsonl").read_bytes()


def test_transcript_is_appended_to_not_truncated(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    backend = fake_claude.backend()
    fake_claude.script(stream="turn1.jsonl")
    run(backend, workspace, transcript)
    fake_claude.script(stream="turn2.jsonl")
    run(backend, workspace, transcript, session=RESUME)

    assert transcript.read_bytes() == (
        (FIXTURES / "turn1.jsonl").read_bytes() + (FIXTURES / "turn2.jsonl").read_bytes()
    )


def test_a_cut_off_final_line_still_ends_with_a_newline_in_the_transcript(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    # otherwise the next turn's first line would be glued onto it
    fake_claude.script(stream="no_result.jsonl", strip_newline=True, exit_code=1)
    run(fake_claude.backend(), workspace, transcript)
    assert transcript.read_bytes() == (FIXTURES / "no_result.jsonl").read_bytes()


def test_stdout_lines_larger_than_asyncio_default_limit_are_not_lost(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl", big_line=2_000_000)  # asyncio's default is 64 KiB

    result = run(fake_claude.backend(), workspace, transcript)

    assert result.ok
    assert result.cost_usd == pytest.approx(0.00091024)
    first_line = transcript.read_bytes().split(b"\n", 1)[0]
    assert len(first_line) > 2_000_000


def test_a_line_beyond_the_reader_limit_is_dropped_but_the_turn_survives(
    fake_claude: Any,
    workspace: Path,
    transcript: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # shrink the limit so a 300 KB line overruns it (the real limit is 64 MiB)
    monkeypatch.setattr(claude_code, "_STDOUT_LIMIT", 100_000)
    fake_claude.script(stream="turn1.jsonl", big_line=300_000)

    with caplog.at_level(logging.WARNING):
        result = run(fake_claude.backend(), workspace, transcript)

    assert result.ok  # the lines after the oversized one still arrive
    assert result.cost_usd == pytest.approx(0.00091024)
    assert result.edit_calls == 1
    assert any("exceeded" in record.getMessage() for record in caplog.records)
    assert transcript.read_bytes().endswith((FIXTURES / "turn1.jsonl").read_bytes())


def test_an_unwritable_transcript_fails_before_any_process_is_started(
    fake_claude: Any, workspace: Path, tmp_path: Path
) -> None:
    blocked = tmp_path / "is-a-directory"
    blocked.mkdir()
    fake_claude.script(stream="turn1.jsonl")

    with pytest.raises(OSError):
        run(fake_claude.backend(), workspace, blocked)

    assert not fake_claude.called


class _FlakyFile:
    """Wraps a real file; the second write fails like a full disk."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self._writes = 0

    def write(self, data: bytes) -> int:
        self._writes += 1
        if self._writes == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return self._real.write(data)

    def flush(self) -> None:
        self._real.flush()

    def close(self) -> None:
        self._real.close()

    def __enter__(self) -> _FlakyFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _FlakyPath(type(Path())):  # type: ignore[misc]
    def open(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return _FlakyFile(super().open(*args, **kwargs))


def test_a_transcript_write_failure_costs_the_transcript_but_not_the_turn(
    fake_claude: Any, workspace: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake_claude.script(stream="turn1.jsonl")
    flaky = _FlakyPath(tmp_path / "flaky" / "transcript.jsonl")

    with caplog.at_level(logging.WARNING):
        result = run(fake_claude.backend(), workspace, flaky)

    # the stream kept being read and parsed: the turn's cost and edits are not lost
    assert result.ok
    assert result.cost_usd == pytest.approx(0.00091024)
    assert result.edit_calls == 1
    assert any("transcript" in record.getMessage().lower() for record in caplog.records)


# --- failure classification -----------------------------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 1])
def test_usage_limit_is_a_rate_limit_error(
    fake_claude: Any, workspace: Path, transcript: Path, exit_code: int
) -> None:
    fake_claude.script(stream="rate_limited.jsonl", exit_code=exit_code)

    result = run(fake_claude.backend(), workspace, transcript)

    assert result.ok is False
    assert result.error_kind == "rate_limit"  # wins over the process/api classes
    assert result.rate_limit == RateLimitInfo("rejected", 1.0, 0.62, 1793050200.0)
    assert "usage limit reached" in (result.error or "")
    assert result.cost_usd == 0.0
    assert result.session_id == "5a1d0000-0000-4000-8000-000000000001"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_api_error(fake_claude: Any, workspace: Path, transcript: Path, exit_code: int) -> None:
    fake_claude.script(stream="api_error.jsonl", exit_code=exit_code)

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (False, "api")
    assert result.error == "API Error: 500 Internal server error"
    assert result.final_text == "API Error: 500 Internal server error"
    assert result.cost_usd == pytest.approx(0.00041)  # money spent on a failed turn still counts
    assert result.rate_limit is not None and result.rate_limit.status == "allowed"


def test_process_error_when_the_process_dies_without_a_result(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream=None, exit_code=1, stderr="boom: cannot open the session store\n")

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (False, "process")
    assert "boom: cannot open the session store" in (result.error or "")
    assert result.final_text == "" and result.tool_calls == () and result.cost_usd == 0.0
    assert result.rate_limit is None


def test_the_error_carries_only_the_tail_of_a_long_stderr(
    fake_claude: Any, workspace: Path, transcript: Path, tmp_path: Path
) -> None:
    noise = "".join(f"noise line {i}\n" for i in range(20_000))  # ~300 KB, past the 64 KiB kept
    noise_file = tmp_path / "noise.txt"
    noise_file.write_text(noise + "the final complaint\n", encoding="utf-8")
    fake_claude.script(stream=None, exit_code=1, stderr_file=noise_file)

    result = run(fake_claude.backend(), workspace, transcript)

    assert result.error_kind == "process"
    assert result.error is not None
    assert "the final complaint" in result.error
    assert "noise line 0\n" not in result.error
    assert len(result.error) < 5_000  # a short message, not the whole log


def test_a_crash_keeps_whatever_the_partial_stream_showed(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="no_result.jsonl", exit_code=1, stderr="killed by signal\n")

    result = run(fake_claude.backend(), workspace, transcript)

    assert result.error_kind == "process"
    assert result.session_id == "5a1d0000-0000-4000-8000-000000000004"
    assert result.final_text == "I am about to edit the solver."
    assert result.tool_calls == (
        ToolCall("Read", "/WORKSPACE/solver.py"),
        ToolCall("Edit", "/WORKSPACE/solver.py"),
    )


def test_nonzero_exit_after_a_successful_result_is_still_a_process_error(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl", exit_code=3, stderr="crashed during shutdown\n")

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (False, "process")
    assert "3" in (result.error or "") and "crashed during shutdown" in (result.error or "")
    assert result.cost_usd == pytest.approx(0.00091024)  # what the stream said is kept
    assert result.edit_calls == 1


def test_exit_zero_without_a_result_line_is_a_process_error(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="no_result.jsonl", exit_code=0)

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (False, "process")
    assert "result" in (result.error or "").lower()


def test_a_process_killed_by_a_signal_is_a_process_error(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="no_result.jsonl", self_kill=True)  # like the OOM killer

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (False, "process")
    assert "signal 9" in (result.error or "")
    assert [c.name for c in result.tool_calls] == ["Read", "Edit"]  # what it did before dying


@pytest.mark.parametrize(
    ("stream_line", "kind", "message"),
    [
        pytest.param(
            {"type": "result", "is_error": True, "result": ""},
            "api",
            "the API returned an error without a message",
            id="api",
        ),
        pytest.param(
            {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}},
            "rate_limit",
            "rate limited (status: rejected)",
            id="rate-limit",
        ),
    ],
)
def test_failures_without_any_text_still_get_a_message(
    fake_claude: Any,
    workspace: Path,
    transcript: Path,
    tmp_path: Path,
    stream_line: dict[str, Any],
    kind: str,
    message: str,
) -> None:
    stream = tmp_path / "ad-hoc.jsonl"
    stream.write_text(json.dumps(stream_line) + "\n", encoding="utf-8")
    fake_claude.script(stream=stream, exit_code=1)

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.error_kind, result.error) == (kind, message)
