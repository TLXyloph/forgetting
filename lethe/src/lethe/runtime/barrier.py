"""The team forget barrier.

When a team trigger fires, agents finish their in-flight turn (including its judging) and wait
here. The last active agent to arrive runs the commit (new epoch for everyone) and releases the
others, so a forget never interrupts a turn and never loses a score.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from lethe.policy.triggers import ForgetDecision

Commit = Callable[[ForgetDecision], Awaitable[None]]


class ForgetBarrier:
    def __init__(self, commit: Commit, *, active: Iterable[str]) -> None:
        self._commit = commit
        self._active = set(active)
        self._arrived: set[str] = set()
        self._generation = 0
        self._cond: asyncio.Condition | None = None
        self.pending: ForgetDecision | None = None

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def request(self, decision: ForgetDecision) -> bool:
        """Schedule a team forget; False if one is already pending."""
        if self.pending is not None:
            return False
        self.pending = decision
        return True

    async def arrive(self, agent: str) -> None:
        """Wait at the barrier if a forget is pending (returns immediately otherwise)."""
        if self.pending is None:
            return
        cond = self._condition()
        async with cond:
            if self.pending is None:
                return
            generation = self._generation
            self._arrived.add(agent)
            await self._maybe_commit()
            await cond.wait_for(lambda: self._generation != generation)

    async def depart(self, agent: str) -> None:
        """An agent stopped for good; it no longer holds up a pending forget."""
        cond = self._condition()
        async with cond:
            self._active.discard(agent)
            self._arrived.discard(agent)
            await self._maybe_commit()

    async def _maybe_commit(self) -> None:
        if self.pending is None or not self._active <= self._arrived:
            return
        decision, self.pending = self.pending, None
        try:
            if self._active:
                await self._commit(decision)
        finally:  # a failed commit raises in the agent that ran it, but releases everyone else
            self._arrived.clear()
            self._generation += 1
            self._condition().notify_all()
