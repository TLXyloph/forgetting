"""Scripted backend: a deterministic stand-in for a coding agent (tests, dry runs, demos).

A script is a plain function that receives the turn (including the workspace path), edits files
in the workspace however it likes, and returns a `ScriptReply` describing the turn.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lethe.domain.types import (
    RateLimitInfo,
    SessionChoice,
    ToolCall,
    TurnErrorKind,
    TurnResult,
    Usage,
)


@dataclass(frozen=True)
class ScriptCall:
    agent: str
    workspace: Path
    prompt: str
    session: SessionChoice
    turn_in_session: int  # 1 for the first turn of a session, then 2, 3, ...


@dataclass(frozen=True)
class ScriptReply:
    final_text: str = "done"
    cost_usd: float = 0.01
    output_tokens: int = 100
    thinking_tokens: int = 0
    tool_calls: tuple[ToolCall, ...] = ()
    error_kind: TurnErrorKind | None = None
    rate_limit: RateLimitInfo | None = None


Script = Callable[[ScriptCall], ScriptReply]


class ScriptedBackend:
    """`AgentBackend` that delegates each turn to `script`.

    Tracks turns per session id: a new session (resume=False) starts at turn 1; resuming an
    unknown session id is an error result (error_kind="process"). The transcript file receives
    one JSON line per turn ({"agent", "session_id", "turn_in_session", "prompt", "final_text"}).
    `ok` is False exactly when the reply sets `error_kind`.
    """

    def __init__(self, script: Script) -> None:
        self._script = script
        self._turns: dict[str, int] = {}  # session id -> turns run so far

    async def run_turn(
        self,
        *,
        agent: str,
        workspace: Path,
        prompt: str,
        session: SessionChoice,
        transcript_path: Path,
    ) -> TurnResult:
        started = time.monotonic()
        session_id = session.session_id
        if session.resume:
            if session_id not in self._turns:
                return TurnResult(
                    session_id=session_id,
                    ok=False,
                    final_text="",
                    cost_usd=0.0,
                    usage=Usage(),
                    tool_calls=(),
                    duration_s=time.monotonic() - started,
                    error_kind="process",
                    error=f"unknown session {session_id}: it was never started, cannot resume",
                )
            turn = self._turns[session_id] + 1
        else:
            turn = 1  # a new session (even under a reused id) starts over
        self._turns[session_id] = turn

        # Script exceptions propagate on purpose: they are bugs in the test, not agent failures.
        reply = self._script(
            ScriptCall(
                agent=agent,
                workspace=workspace,
                prompt=prompt,
                session=session,
                turn_in_session=turn,
            )
        )
        if not isinstance(reply, ScriptReply):
            raise TypeError(f"a script must return a ScriptReply, got {type(reply).__name__}")

        _append_transcript(
            transcript_path,
            {
                "agent": agent,
                "session_id": session_id,
                "turn_in_session": turn,
                "prompt": prompt,
                "final_text": reply.final_text,
            },
        )
        return TurnResult(
            session_id=session_id,
            ok=reply.error_kind is None,
            final_text=reply.final_text,
            cost_usd=reply.cost_usd,
            usage=Usage(output_tokens=reply.output_tokens, thinking_tokens=reply.thinking_tokens),
            tool_calls=reply.tool_calls,
            duration_s=time.monotonic() - started,
            error_kind=reply.error_kind,
            error=None if reply.error_kind is None else f"scripted {reply.error_kind} error",
            rate_limit=reply.rate_limit,
        )


def _append_transcript(path: Path, record: dict[str, object]) -> None:
    """Append one JSON line (pure ASCII, so lone surrogates and control characters survive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as transcript:
        transcript.write(json.dumps(record) + "\n")
