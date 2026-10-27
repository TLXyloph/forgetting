"""Outcome tables, computed from run summaries: what the team forgets led to, how much of the
spend went to empty turns, and what it cost to reach a score.

Like `research.tables`, everything here is pure: `RunSummary` objects in, plain row dataclasses
out. Text and CSV rendering lives in `research.analyze`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from lethe.domain.types import Direction, is_better
from research import stats
from research.summary import RunSummary
from research.tables import arm_names, direction_of

__all__ = [
    "EmptyTurnRow",
    "ForgetOutcomeRow",
    "ThresholdRow",
    "empty_turn_table",
    "forget_outcome_table",
    "spend_to_reach",
    "threshold_table",
]


def _share(part: float, whole: float) -> float | None:
    return part / whole if whole else None


# ---------------------------------------------------------------------------------------------
# forget outcomes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgetOutcomeRow:
    """The team forgets of an arm's `runs` forgetting runs, and how many of them were followed
    by a new best within the next round."""

    arm: str
    runs: int  # the arm's runs with at least one team forget
    forgets: int
    new_bests: int
    new_best_rate: float | None  # None without forgets


def _new_best(before: float | None, after: float | None, direction: Direction) -> bool:
    if after is None:
        return False
    return before is None or is_better(after, before, direction)


def forget_outcome_table(summaries: Sequence[RunSummary]) -> list[ForgetOutcomeRow]:
    """Per arm whose runs forgot: team forgets, and those followed by a new best in the next round.

    The forget between rounds k and k + 1 led to a new best when round k + 1 ended on a best
    strictly better than the one round k ended on (round bests are the run's best so far, so a
    tie means the round found nothing new). A round that ends without any valid score is no
    improvement; the first valid score of a run is one. Runs without team forgets are left out,
    as in Table 2, and arms without any forgetting run produce no row.
    """
    direction = direction_of(summaries)
    rows: list[ForgetOutcomeRow] = []
    for arm in arm_names(summaries):
        forgetting = [s for s in summaries if s.arm == arm and s.team_forgets > 0]
        if not forgetting:
            continue
        outcomes = [
            _new_best(before.best, after.best, direction)
            for run in forgetting
            for before, after in itertools.pairwise(run.rounds)
        ]
        new_bests = sum(outcomes)
        rate = _share(new_bests, len(outcomes))
        rows.append(ForgetOutcomeRow(arm, len(forgetting), len(outcomes), new_bests, rate))
    return rows


# ---------------------------------------------------------------------------------------------
# empty turns and their spend
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EmptyTurnRow:
    """An arm's turns, pooled over its `runs` runs: how many ended without a candidate and what
    share of the spend they consumed. Error turns count in `turns` and in the spend (a failed
    turn still costs) but are never empty turns."""

    arm: str
    runs: int
    turns: int
    empty_turns: int
    error_turns: int
    empty_share: float | None  # of all turns; None without turns
    spend_usd: float  # the cost of every turn
    empty_spend_usd: float
    empty_spend_share: float | None  # None without spend


def empty_turn_table(summaries: Sequence[RunSummary]) -> list[EmptyTurnRow]:
    """Per arm: turns, empty turns, and the shares of turns and spend that went to empty ones."""
    rows: list[EmptyTurnRow] = []
    for arm in arm_names(summaries):
        runs = [s for s in summaries if s.arm == arm]
        turns = [t for s in runs for t in s.turns]
        empty = [t for t in turns if t.status == "empty"]
        spend = math.fsum(t.cost for t in turns)
        empty_spend = math.fsum(t.cost for t in empty)
        rows.append(
            EmptyTurnRow(
                arm=arm,
                runs=len(runs),
                turns=len(turns),
                empty_turns=len(empty),
                error_turns=sum(t.status == "error" for t in turns),
                empty_share=_share(len(empty), len(turns)),
                spend_usd=spend,
                empty_spend_usd=empty_spend,
                empty_spend_share=_share(empty_spend, spend),
            )
        )
    return rows


# ---------------------------------------------------------------------------------------------
# cost to reach a score
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdRow:
    """Cumulative spend at which an arm's runs first reached `threshold` (a score at least as
    good), over the `reached` of its `runs` runs that did; the spend columns are None when none
    did."""

    arm: str
    threshold: float
    runs: int
    reached: int
    mean_spend: float | None
    min_spend: float | None
    max_spend: float | None


def spend_to_reach(
    curve: Sequence[tuple[float, float]], threshold: float, direction: Direction
) -> float | None:
    """Spend at which a best-vs-spend curve first reaches `threshold` (None if it never does)."""
    for spend, best in curve:
        if best == threshold or is_better(best, threshold, direction):
            return spend
    return None


def threshold_table(
    summaries: Sequence[RunSummary], thresholds: Sequence[float]
) -> list[ThresholdRow]:
    """Per arm and threshold (in the order given): runs that reached it, and at what spend.

    Spend is read from each run's best-vs-spend curve, so a candidate counts at the spend at the
    end of the turn that proposed it; a seed that already meets the threshold counts at 0.
    """
    direction = direction_of(summaries)
    rows: list[ThresholdRow] = []
    for arm in arm_names(summaries):
        runs = [s for s in summaries if s.arm == arm]
        for threshold in thresholds:
            spends = [
                spend
                for run in runs
                if (spend := spend_to_reach(run.curve, threshold, direction)) is not None
            ]
            rows.append(
                ThresholdRow(
                    arm=arm,
                    threshold=threshold,
                    runs=len(runs),
                    reached=len(spends),
                    mean_spend=stats.mean(spends) if spends else None,
                    min_spend=min(spends, default=None),
                    max_spend=max(spends, default=None),
                )
            )
    return rows
