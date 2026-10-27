"""Budgets and subscription pacing.

Spend is metered in API-equivalent USD (Claude Code reports each turn's cost at list price even
on a subscription), in proposals (agent turns), and in wall-clock seconds. On a subscription the
binding constraint is usually the 5-hour usage window, so the `Pacer` pauses every agent when the
window is nearly full or exhausted and resumes after it resets. An optional weekly threshold does
the same for the 7-day window, to leave part of a week's allowance for other use during long runs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from lethe.domain.types import RateLimitInfo

Primary = Literal["usd", "proposals", "wall"]


@dataclass(frozen=True)
class BudgetSpec:
    usd: float = 0.0
    proposals: int = 0
    wall_s: float = 0.0
    primary: Primary = "usd"
    pause_at_utilization: float = 0.95
    pause_at_weekly_utilization: float = 1.0


def spend_fraction(
    spec: BudgetSpec, *, spent_usd: float, proposals: int, elapsed_s: float
) -> float:
    """Fraction of the primary budget used (the unit forget schedules are expressed in)."""
    if spec.primary == "usd" and spec.usd > 0:
        return spent_usd / spec.usd
    if spec.primary == "proposals" and spec.proposals > 0:
        return proposals / spec.proposals
    if spec.primary == "wall" and spec.wall_s > 0:
        return elapsed_s / spec.wall_s
    return 0.0


def exhausted(
    spec: BudgetSpec, *, spent_usd: float, proposals: int, elapsed_s: float
) -> str | None:
    """Why the run must stop, or None while every cap still has room."""
    if spec.usd > 0 and spent_usd >= spec.usd:
        return f"usd budget exhausted (${spent_usd:.2f} of ${spec.usd:.2f})"
    if spec.proposals > 0 and proposals >= spec.proposals:
        return f"proposal budget exhausted ({proposals} of {spec.proposals})"
    if spec.wall_s > 0 and elapsed_s >= spec.wall_s:
        return f"wall-clock budget exhausted ({elapsed_s:.0f}s of {spec.wall_s:.0f}s)"
    return None


class Pacer:
    """Holds every agent while the subscription window is (nearly) exhausted."""

    GRACE_S = 15.0
    MAX_SLEEP_S = 30.0

    def __init__(
        self,
        *,
        pause_at: float,
        weekly_pause_at: float = 1.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        base_backoff_s: float = 60.0,
        max_backoff_s: float = 900.0,
    ) -> None:
        self.pause_at = pause_at
        self.weekly_pause_at = weekly_pause_at
        self._clock = clock
        self._sleep = sleep
        self._base = base_backoff_s
        self._max = max_backoff_s
        self._backoff = base_backoff_s
        self.paused_until = 0.0

    def observe(self, info: RateLimitInfo | None) -> float | None:
        """Record a healthy turn's meter; pause if the 5-hour window crossed `pause_at` or the
        7-day window crossed `weekly_pause_at`, until that window resets (its own reset time; the
        event's `resets_at` when absent). When both crossed, the later reset wins."""
        if info is None:
            return None
        self._backoff = self._base
        until = None
        windows = (
            (info.five_hour_utilization, self.pause_at, info.five_hour_resets_at),
            (info.seven_day_utilization, self.weekly_pause_at, info.seven_day_resets_at),
        )
        for util, threshold, resets in windows:
            if util is not None and util >= threshold:
                until = self._pause_until(info.resets_at if resets is None else resets)
        return until

    def note_rate_limited(self, info: RateLimitInfo | None) -> float:
        """A turn was refused for rate limits: pause until the window resets (or back off)."""
        return self._pause_until(info.resets_at if info is not None else None)

    def _pause_until(self, resets_at: float | None) -> float:
        now = self._clock()
        if resets_at is not None and resets_at > now:
            until = resets_at + self.GRACE_S
        else:
            until = now + self._backoff
            self._backoff = min(self._backoff * 2, self._max)
        self.paused_until = max(self.paused_until, until)
        return self.paused_until

    @property
    def paused(self) -> bool:
        return self._clock() < self.paused_until

    async def wait(self) -> None:
        while (remaining := self.paused_until - self._clock()) > 0:
            await self._sleep(min(remaining, self.MAX_SLEEP_S))
