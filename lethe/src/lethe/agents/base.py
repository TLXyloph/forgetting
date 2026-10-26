"""The agent-backend port: anything that can run one turn of a coding agent in a workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from lethe.domain.types import SessionChoice, TurnResult


class AgentBackend(Protocol):
    async def run_turn(
        self,
        *,
        agent: str,
        workspace: Path,
        prompt: str,
        session: SessionChoice,
        transcript_path: Path,
    ) -> TurnResult:
        """Run one turn with cwd=`workspace`, sending `prompt` in `session`.

        Implementations write the raw transcript of the turn to `transcript_path` and never
        raise for agent-side failures: those come back as `TurnResult(ok=False, error_kind=...)`.
        """
        ...
