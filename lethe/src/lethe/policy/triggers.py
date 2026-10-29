"""Forget triggers: when to forget.

Every trigger is a pure function of `RunState` (plus the current budget fraction), so a run's
forgets can be re-derived from its log. Team triggers make every agent forget at the next
barrier; agent triggers reset one agent without stopping the others.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal

from lethe.domain.state import RunState, TurnInfo
from lethe.domain.types import Direction


@dataclass(frozen=True)
class ForgetDecision:
    scope: Literal["team", "agent"]
    agents: tuple[str, ...]
    reason: str


def relative_gain(new: float, old: float, direction: Direction) -> float:
    delta = new - old if direction == "max" else old - new
    return delta / max(abs(old), 1e-12)


def improved_since(state: RunState, seq: int, min_gain: float = 0.0) -> bool:
    """True if the team best improved by at least `min_gain` (relative) after event `seq`.

    Every later best is measured against the best at `seq` (the window start), so several small
    steps that add up to `min_gain` count as an improvement.
    """
    start = state.best_at(seq)
    for entry in state.best_history:
        if entry.seq <= seq:
            continue
        if start is None or relative_gain(entry.score, start.score, state.direction) >= min_gain:
            return True
    return False


def _window_start(proposals: list[TurnInfo], window: int) -> int:
    return proposals[-window].seq if len(proposals) >= window else -1


@dataclass(frozen=True)
class BudgetFractions:
    """Forget each time spend crosses one of `fractions` (the paper: 0.25, 0.5, 0.75)."""

    fractions: tuple[float, ...]
    scope: ClassVar[str] = "team"
    TAG: ClassVar[str] = "budget_fractions@"

    def check(self, state: RunState, *, spend_fraction: float) -> ForgetDecision | None:
        consumed = 0
        for reason in state.forget_reasons("team"):
            if reason.startswith(self.TAG):
                consumed = max(consumed, int(reason[len(self.TAG) :].split()[0]))
        crossed = sum(1 for f in self.fractions if spend_fraction >= f)
        if crossed <= consumed:
            return None
        reason = f"{self.TAG}{crossed} (spend at {spend_fraction:.0%} of budget)"
        return ForgetDecision("team", state.agents, reason)


@dataclass(frozen=True)
class Plateau:
    """Forget when the team best has not improved by `min_gain` over the last `window` proposals."""

    window: int = 8
    min_gain: float = 0.0
    min_proposals: int = 8
    scope: ClassVar[str] = "team"

    def check(self, state: RunState, *, spend_fraction: float) -> ForgetDecision | None:
        proposals = state.round_proposals()
        if len(proposals) < max(self.min_proposals, self.window):
            return None
        if improved_since(state, _window_start(proposals, self.window), self.min_gain):
            return None
        reason = f"plateau: no gain >= {self.min_gain:g} in the last {self.window} proposals"
        return ForgetDecision("team", state.agents, reason)


@dataclass(frozen=True)
class Anchored:
    """Forget when sessions narrow: recent edit sizes fall to `ratio` of each session's start.

    An agent is anchored when the median lines changed over its last `window` turns is at most
    `ratio` x the median over its session's first `warmup` turns (empty turns count as 0 lines).
    Fires when at least `quorum` of the agents are anchored and, with `require_stall`, the team
    best did not improve during the last `window` proposals of the round.
    """

    ratio: float = 0.3
    warmup: int = 4
    window: int = 4
    quorum: float = 0.5
    require_stall: bool = True
    min_proposals: int = 8
    scope: ClassVar[str] = "team"

    def is_anchored(self, state: RunState, agent: str) -> bool:
        turns = state.session_turn_list(agent)
        if len(turns) < self.warmup + self.window:
            return False
        baseline = statistics.median(t.lines_changed for t in turns[: self.warmup])
        recent = statistics.median(t.lines_changed for t in turns[-self.window :])
        return baseline > 0 and recent <= self.ratio * baseline

    def check(self, state: RunState, *, spend_fraction: float) -> ForgetDecision | None:
        proposals = state.round_proposals()
        if len(proposals) < self.min_proposals:
            return None
        anchored = [a for a in state.agents if self.is_anchored(state, a)]
        needed = max(1, math.ceil(self.quorum * len(state.agents)))
        if len(anchored) < needed:
            return None
        if self.require_stall and improved_since(state, _window_start(proposals, self.window)):
            return None
        reason = (
            f"anchored: {len(anchored)}/{len(state.agents)} sessions narrowed to <= "
            f"{self.ratio:g}x their opening edit size"
        )
        return ForgetDecision("team", state.agents, reason)


@dataclass(frozen=True)
class Stalled:
    """Reset one agent when `n` of its last `of` turns were empty or claimed a plateau.

    With `passive_only`, such a turn counts only if it made no tool call at all: the agent
    stopped acting. A turn that explored or ran experiments but has not submitted yet never
    counts, and neither does one logged before tool calls were counted. The window is still the
    agent's last `of` completed turns, so active turns in it dilute the passive ones.
    """

    n: int = 3
    of: int = 5
    passive_only: bool = False
    scope: ClassVar[str] = "agent"

    def counts(self, turn: TurnInfo) -> bool:
        idle = turn.status == "empty" or turn.plateau_claim
        return idle and (not self.passive_only or turn.tool_calls == 0)

    def check_agent(
        self, state: RunState, agent: str, *, spend_fraction: float
    ) -> ForgetDecision | None:
        turns = [t for t in state.epoch_turns(agent) if t.ok]
        if len(turns) < self.of:
            return None
        stalled = sum(1 for t in turns[-self.of :] if self.counts(t))
        if stalled < self.n:
            return None
        what = "empty or claimed a plateau"
        if self.passive_only:
            what = f"passive ({what}, with no tool call)"
        reason = f"stalled: {stalled} of the last {self.of} turns were {what}"
        return ForgetDecision("agent", (agent,), reason)


TeamTrigger = BudgetFractions | Plateau | Anchored
AgentTrigger = Stalled
Trigger = TeamTrigger | AgentTrigger

_TYPES: dict[str, type] = {
    "budget_fractions": BudgetFractions,
    "plateau": Plateau,
    "anchored": Anchored,
    "stalled": Stalled,
}


def trigger_from_dict(spec: Mapping[str, Any]) -> Trigger:
    """Build a trigger from its config table, e.g. {"type": "plateau", "window": 8}."""
    params = dict(spec)
    kind = params.pop("type", None)
    cls = _TYPES.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown trigger type {kind!r}; expected one of {sorted(_TYPES)}")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unknown parameter(s) for {kind}: {', '.join(unknown)}")
    if cls is BudgetFractions:
        fractions = params.get("fractions")
        if (
            not isinstance(fractions, list | tuple)
            or not fractions
            or not all(_is_number(f) and 0 < f < 1 for f in fractions)
        ):
            raise ValueError(
                "budget_fractions: fractions must be a non-empty list of numbers in (0, 1)"
            )
        return BudgetFractions(tuple(sorted(float(f) for f in fractions)))
    flags = {f.name for f in fields(cls) if f.type in ("bool", bool)}
    for name, value in params.items():
        if isinstance(value, bool) != (name in flags) or not (
            isinstance(value, bool) or _is_number(value)
        ):
            raise ValueError(f"{kind}: parameter {name} has an invalid value {value!r}")
        if not isinstance(value, bool) and value < 0:
            raise ValueError(f"{kind}: parameter {name} must be >= 0")
    return cls(**params)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
