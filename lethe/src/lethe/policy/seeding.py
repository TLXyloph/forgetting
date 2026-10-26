"""Seed policies: what each agent remembers (starts from) after a forget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lethe.domain.state import SEED_ID, CandidateInfo, RunState
from lethe.domain.types import is_better
from lethe.policy.triggers import relative_gain


class SeedPolicy(Protocol):
    def seeds(self, state: RunState, agents: Sequence[str]) -> dict[str, str]: ...


@dataclass(frozen=True)
class BestSeed:
    """The paper's policy: every agent restarts from the single best candidate."""

    def seeds(self, state: RunState, agents: Sequence[str]) -> dict[str, str]:
        best = state.best()
        seed = best.cand_id if best is not None else SEED_ID
        return {agent: seed for agent in agents}


@dataclass(frozen=True)
class HedgedSeed:
    """Experimental: part of the team restarts from near-best elites that win different test cases.

    Elites are valid candidates within `epsilon` (relative) of the best score that beat the best
    on at least one judge-reported case (`cases`). Without per-case scores this is `BestSeed`.
    """

    elite_fraction: float = 0.5
    epsilon: float = 0.02

    def seeds(self, state: RunState, agents: Sequence[str]) -> dict[str, str]:
        best = state.best()
        if best is None or not best.cases:
            return BestSeed().seeds(state, agents)
        elites = self._elites(state, best)[: int(len(agents) * self.elite_fraction)]
        n_best = len(agents) - len(elites)
        return {
            agent: best.cand_id if i < n_best else elites[i - n_best]
            for i, agent in enumerate(agents)
        }

    def _elites(self, state: RunState, best: CandidateInfo) -> list[str]:
        assert best.cases is not None and best.score is not None
        direction = state.direction
        seen = {best.judged_key}
        rows: list[tuple[int, float, str]] = []
        for cand in state.candidates.values():
            if not cand.valid or cand.score is None or not cand.cases or cand.judged_key in seen:
                continue
            if relative_gain(best.score, cand.score, direction) > self.epsilon:
                continue
            wins = sum(
                1
                for case, value in cand.cases.items()
                if case in best.cases and is_better(value, best.cases[case], direction)
            )
            if wins:
                seen.add(cand.judged_key)
                rank = cand.score if direction == "max" else -cand.score
                rows.append((wins, rank, cand.cand_id))
        rows.sort(key=lambda row: (-row[0], -row[1]))
        return [cand_id for _, _, cand_id in rows]


def seed_policy_from_name(name: str) -> SeedPolicy:
    if name == "best":
        return BestSeed()
    if name == "hedged":
        return HedgedSeed()
    raise ValueError(f"unknown seed policy {name!r}; expected 'best' or 'hedged'")
