"""Claude Code backend: runs one agent turn as a headless `claude -p` process.

Works with a Claude subscription login (no API key needed) in the default `safe` isolation:
`--safe-mode` disables CLAUDE.md, skills, plugins, hooks and MCP servers; `--restricted` removes
command-running tools and confines file tools to the working directory;
`--disable-slash-commands` removes skills/commands from the context. `bare` isolation uses
`--bare` instead of `--safe-mode` and requires ANTHROPIC_API_KEY in the environment.

Safe isolation also keeps the subscription promise: unless `allow_api_billing` is set, `claude`
runs without the variables that would bill an API key or a cloud provider (API_BILLING_ENV), and
a turn whose init line still reports an API key (`apiKeySource` other than "none") is refused
with error_kind "billing", which stops the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from lethe.agents.stream import StreamSummary, looks_rate_limited, parse_stream
from lethe.domain.types import SessionChoice, TurnErrorKind, TurnResult

Isolation = Literal["safe", "bare", "none"]

DEFAULT_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")

# Variables that make `claude` bill an API key or a cloud provider instead of the subscription
# login. Safe isolation launches it without them unless `allow_api_billing` is set.
API_BILLING_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

_LOG = logging.getLogger(__name__)

_ISOLATION_FLAGS: dict[str, tuple[str, ...]] = {
    "safe": ("--safe-mode", "--restricted", "--disable-slash-commands"),
    "bare": ("--bare", "--restricted", "--disable-slash-commands"),
    "none": (),
}

# Session ids go straight into argv: refuse anything that could pass for a flag or split an arg.
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")

# One stdout line can hold a whole file (tool results echo file contents); asyncio's default
# StreamReader limit of 64 KiB would drop such lines.
_STDOUT_LIMIT = 64 * 1024 * 1024
_READ_CHUNK = 64 * 1024
_STDERR_KEEP = 64 * 1024  # bytes of stderr remembered for error messages
_ERROR_TEXT_MAX = 2000  # characters of stderr/result text copied into TurnResult.error
_REAP_TIMEOUT_S = 5.0  # how long to wait for a killed process to be collected


@dataclass(frozen=True)
class ClaudeCodeOptions:
    executable: str = "claude"
    model: str | None = "sonnet"
    effort: str | None = None
    tools: tuple[str, ...] = DEFAULT_TOOLS
    shell: bool = False  # adds "Bash" to tools (runs commands on the host; opt-in)
    isolation: Isolation = "safe"
    permission_mode: str = "acceptEdits"
    turn_timeout_s: float = 1800.0
    extra_args: tuple[str, ...] = ()
    allow_api_billing: bool = False  # safe isolation: keep the API-key variables (see run_turn)


class ClaudeCodeBackend:
    """`AgentBackend` implementation. The prompt is written to the process's stdin."""

    def __init__(self, options: ClaudeCodeOptions | None = None) -> None:
        opts = options if options is not None else ClaudeCodeOptions()
        if not isinstance(opts.isolation, str) or opts.isolation not in _ISOLATION_FLAGS:
            raise ValueError(
                f"unknown isolation {opts.isolation!r}; expected one of {sorted(_ISOLATION_FLAGS)}"
            )
        timeout = opts.turn_timeout_s
        # `not timeout > 0` also rejects NaN; +inf means "no timeout"
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not timeout > 0:
            raise ValueError(f"turn_timeout_s must be a positive number, got {timeout!r}")
        self.options = opts
        # session id -> its latest cumulative total_cost_usd (for streams without per-turn usage)
        self._session_totals: dict[str, float] = {}

    @property
    def refuses_api_billing(self) -> bool:
        """True in safe isolation unless `allow_api_billing`: subscription login only."""
        return self.options.isolation == "safe" and not self.options.allow_api_billing

    def child_env(self) -> dict[str, str] | None:
        """The environment `claude` runs with: lethe's own (None), minus API_BILLING_ENV when
        `refuses_api_billing`."""
        if not self.refuses_api_billing:
            return None
        return {k: v for k, v in os.environ.items() if k not in API_BILLING_ENV}

    def build_argv(self, session: SessionChoice) -> list[str]:
        """Exact argv (without the prompt), in this order:
        [executable, "-p", "--output-format", "stream-json", "--verbose",
         "--permission-mode", permission_mode, "--tools", ",".join(tools [+ "Bash" if shell])]
        + isolation flags (safe: --safe-mode --restricted --disable-slash-commands;
                          bare: --bare --restricted --disable-slash-commands; none: nothing)
        + ["--model", model] if model + ["--effort", effort] if effort
        + (["--resume", id] if session.resume else ["--session-id", id])
        + list(extra_args)
        """
        opts = self.options
        session_id = session.session_id
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError(f"invalid session id for the claude command line: {session_id!r}")

        tools = list(opts.tools)
        if opts.shell and "Bash" not in tools:
            tools.append("Bash")

        argv = [
            opts.executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            opts.permission_mode,
            "--tools",
            ",".join(tools),
            *_ISOLATION_FLAGS[opts.isolation],
        ]
        if opts.model:
            argv += ["--model", opts.model]
        if opts.effort:
            argv += ["--effort", opts.effort]
        argv += ["--resume" if session.resume else "--session-id", session_id]
        argv += opts.extra_args
        return argv

    async def run_turn(
        self,
        *,
        agent: str,
        workspace: Path,
        prompt: str,
        session: SessionChoice,
        transcript_path: Path,
    ) -> TurnResult:
        """Run `build_argv(session)` with cwd=workspace, prompt on stdin, in a new process group,
        with `child_env()` as its environment.

        Streams stdout lines into `transcript_path` (parent dirs created) and parses them with
        `parse_stream`. Outcomes, in this order:
          * `refuses_api_billing` and the init line's apiKeySource is not "none"
            -> ok=False, error_kind="billing" (the turn ran on an API key: the run must stop)
          * timeout (options.turn_timeout_s): kill the process group
            -> ok=False, error_kind="timeout"
          * `looks_rate_limited` with no result or an error result
            -> ok=False, error_kind="rate_limit" (a successful result wins; rate_limit info is
            attached either way, for the pacer)
          * result.is_error -> ok=False, error_kind="api"
          * nonzero exit or no result line -> ok=False, error_kind="process" (stderr tail in error)
          * otherwise ok=True.
        `cost_usd` is the stream's per-turn cost; when the CLI reports only the cumulative
        session total, it is that total minus the last total seen for the session (by resume
        id, so a session that comes back under a new id is charged correctly).
        `session_id` in the result is the one reported by the stream, else `session.session_id`.
        Never raises for agent-side failures; a missing executable raises FileNotFoundError.
        """
        argv = self.build_argv(session)
        timeout_s = self.options.turn_timeout_s
        stdin_bytes = prompt.encode("utf-8", errors="replace")

        # Open the transcript before spawning: if it cannot be written we fail without ever
        # launching (and paying for) an agent turn.
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with transcript_path.open("ab") as sink:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workspace,
                env=self.child_env(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group, so a kill reaches every descendant
                limit=_STDOUT_LIMIT,
            )
            log = _TurnLog(sink)
            timed_out = await _drive(proc, log, stdin_bytes, timeout_s)
        duration_s = time.monotonic() - started

        summary = parse_stream(log.lines)
        error_kind, error = _classify(
            summary,
            timed_out=timed_out,
            returncode=proc.returncode,
            timeout_s=timeout_s,
            stderr_text=log.stderr_text(),
            refuse_api_key=self.refuses_api_billing,
        )
        return TurnResult(
            session_id=summary.session_id or session.session_id,
            ok=error_kind is None,
            final_text=summary.result_text,
            cost_usd=self._turn_cost(summary, session),
            usage=summary.usage,
            tool_calls=summary.tool_calls,
            duration_s=duration_s,
            error_kind=error_kind,
            error=error,
            rate_limit=summary.rate_limit,
            num_api_turns=summary.num_api_turns,
        )

    def _turn_cost(self, summary: StreamSummary, session: SessionChoice) -> float:
        """The stream's per-turn cost, or the growth of the session's cumulative total."""
        total, cost = summary.total_cost_usd, summary.cost_usd
        if summary.cost_is_total and total is not None:
            before = self._session_totals.get(session.session_id, 0.0) if session.resume else 0.0
            cost = total - before if total >= before else total  # a reset total is all new
        if total is not None:
            self._session_totals[session.session_id] = total
            if summary.session_id:
                self._session_totals[summary.session_id] = total
        return cost


# --- process supervision ------------------------------------------------------------------------


class _TurnLog:
    """One turn's output: stdout lines (mirrored into the transcript) and the stderr tail."""

    def __init__(self, sink: BinaryIO) -> None:
        self.lines: list[str] = []
        self._stderr = bytearray()
        self._sink: BinaryIO | None = sink

    async def pump_stdout(self, stream: asyncio.StreamReader) -> None:
        while True:
            try:
                raw = await stream.readline()
            except ValueError:
                # a single line beyond _STDOUT_LIMIT: asyncio discarded it, keep reading the rest
                _LOG.warning("claude stdout line exceeded %d bytes and was dropped", _STDOUT_LIMIT)
                continue
            if not raw:
                return
            self._add_line(raw)

    async def pump_stderr(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(_READ_CHUNK):
            self._stderr += chunk
            if len(self._stderr) > _STDERR_KEEP:
                del self._stderr[: len(self._stderr) - _STDERR_KEEP]

    def stderr_text(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    def _add_line(self, raw: bytes) -> None:
        self.lines.append(raw.decode("utf-8", errors="replace"))
        if self._sink is None:
            return
        try:
            # a line cut off by a kill still gets its newline, so later appends start cleanly
            self._sink.write(raw if raw.endswith(b"\n") else raw + b"\n")
            self._sink.flush()
        except OSError as exc:
            # the transcript is a debugging aid: losing it must not lose the turn's cost/edits
            self._sink = None
            _LOG.warning("transcript write failed (%s); continuing without the transcript", exc)


def _start_stdin(stdin: asyncio.StreamWriter, data: bytes) -> None:
    """Hand the prompt to the pipe now, before the event loop runs anything else. The CLI waits
    only 3 s for stdin before running without a prompt, and other agents' synchronous work can
    hold the loop that long; bytes already in the pipe do not depend on the loop."""
    with contextlib.suppress(OSError):  # BrokenPipeError, ConnectionResetError, ...
        stdin.write(data)


async def _finish_stdin(stdin: asyncio.StreamWriter) -> None:
    """Flush what the pipe did not take at once (a prompt larger than its buffer) and close
    stdin. A child that exits early is not an error here: its exit status and missing result
    line are reported by the caller."""
    with contextlib.suppress(OSError):
        await stdin.drain()
        stdin.close()
        await stdin.wait_closed()


async def _drive(
    proc: asyncio.subprocess.Process, log: _TurnLog, stdin_bytes: bytes, timeout_s: float
) -> bool:
    """Run the pipes until the process is done. Returns True when the timeout killed it.

    Whatever happens, no task is left running and the child has been collected on return;
    cancellation (or any unexpected error) kills the process group before propagating.
    """
    tasks: list[asyncio.Future[None]] = []
    timed_out = False
    try:
        stdin, stdout, stderr = proc.stdin, proc.stdout, proc.stderr
        if stdin is None or stdout is None or stderr is None:
            raise RuntimeError("subprocess pipes were not created")
        _start_stdin(stdin, stdin_bytes)
        tasks = [
            asyncio.ensure_future(_finish_stdin(stdin)),
            asyncio.ensure_future(log.pump_stdout(stdout)),
            asyncio.ensure_future(log.pump_stderr(stderr)),
        ]
        deadline = asyncio.timeout(None if timeout_s == math.inf else timeout_s)
        try:
            async with deadline:
                await asyncio.gather(*tasks)  # EOF on both pipes ...
                await proc.wait()  # ... and the exit status
        except TimeoutError:
            if not deadline.expired():
                raise
            timed_out = True
            _kill_group(proc)
    except BaseException:  # incl. CancelledError: never leave the agent running behind us
        _kill_group(proc)
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _reap(proc)
    return timed_out


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group (it leads one: see start_new_session).

    Skipped once the child has been collected: its pid may already belong to someone else.
    """
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT_S)


# --- result classification ----------------------------------------------------------------------


def _classify(
    summary: StreamSummary,
    *,
    timed_out: bool,
    returncode: int | None,
    timeout_s: float,
    stderr_text: str,
    refuse_api_key: bool = False,
) -> tuple[TurnErrorKind | None, str | None]:
    """billing > timeout > rate_limit > api > process > ok, with a short message for each failure.

    A rate-limit signal only fails a turn that has no result or an error result.
    """
    source = summary.api_key_source
    if refuse_api_key and source is not None and source != "none":
        return "billing", (
            f"safe mode refused API-key billing (apiKeySource={source}); "
            "set agent.allow_api_billing = true to allow it"
        )
    if timed_out:
        return "timeout", f"turn exceeded the {timeout_s:g}s limit and its process group was killed"
    if looks_rate_limited(summary) and (not summary.has_result or summary.is_error):
        status = summary.rate_limit.status if summary.rate_limit else "unknown"
        return "rate_limit", _head(summary.result_text) or f"rate limited (status: {status})"
    if summary.is_error:
        return "api", _head(summary.result_text) or "the API returned an error without a message"
    if returncode != 0 or not summary.has_result:
        return "process", _process_message(returncode, stderr_text)
    return None, None


def _process_message(returncode: int | None, stderr_text: str) -> str:
    if returncode is None or returncode == 0:
        head = "claude exited without producing a result line"
    elif returncode < 0:
        head = f"claude was killed by signal {-returncode}"
    else:
        head = f"claude exited with code {returncode}"
    tail = stderr_text.strip()[-_ERROR_TEXT_MAX:].strip()
    return f"{head}: {tail}" if tail else head


def _head(text: str) -> str:
    return text.strip()[:_ERROR_TEXT_MAX]
