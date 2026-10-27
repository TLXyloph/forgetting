"""The analysis tables, computed from run summaries.

Everything here is pure: `RunSummary` objects in, plain row dataclasses out. Text and CSV
rendering lives in `research.analyze`.

"Change" columns are signed percentages in the better direction: positive always means better,
so on a task where lower scores win (cycle counts) a drop in the score is a positive change.
The paper prints the raw sign instead (negative cycles = green).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from lethe.domain.types import Direction
from research import stats
from research.summary import PROPOSAL_STATUSES, RunSummary, TurnRecord

__all__ = [
    "ArmRow",
    "Comparison",
    "RoundRow",
    "TraceRow",
    "anchoring_traces",
    "arm_names",
    "arm_table",
    "compare_arms",
    "direction_of",
    "improvement_pct",
    "round_table",
]


def arm_names(summaries: Sequence[RunSummary]) -> list[str]:
    """Arm names in the order they first appear in `summaries`."""
    return list(dict.fromkeys(s.arm for s in summaries))


def direction_of(summaries: Sequence[RunSummary]) -> Direction:
    """The one direction all `summaries` share ("max" for none); mixed directions are an error."""
    directions = {s.direction for s in summaries}
    if len(directions) > 1:
        raise ValueError(f"runs disagree on direction ({sorted(directions)}): mixed tasks?")
    return next(iter(directions), "max")


def improvement_pct(value: float, reference: float, direction: Direction) -> float | None:
    """`value` against `reference` as a signed % of |reference|, positive when better."""
    if reference == 0 or not math.isfinite(reference):
        return None
    delta = value - reference if direction == "max" else reference - value
    return 100.0 * delta / abs(reference)


def _finals_by_arm(summaries: Sequence[RunSummary]) -> dict[str, list[float]]:
    """Final best scores per arm, skipping runs that never produced a valid candidate."""
    finals: dict[str, list[float]] = {arm: [] for arm in arm_names(summaries)}
    for summary in summaries:
        if summary.final_best is not None:
            finals[summary.arm].append(summary.final_best)
    return finals


# ---------------------------------------------------------------------------------------------
# Table 1: final best per arm
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRow:
    """One arm of Table 1. `n` counts the runs that produced a valid result."""

    arm: str
    n: int
    best_of_runs: float | None
    mean: float | None
    sd: float | None  # None below two runs
    best_change_pct: float | None  # best-of-runs against the reference arm's
    mean_change_pct: float | None  # mean against the reference arm's
    is_reference: bool


def arm_table(summaries: Sequence[RunSummary], *, ref: str | None = "solo") -> list[ArmRow]:
    """Table 1: per arm, n, best of runs, mean and sd of the final best, change vs `ref`.

    Without a usable reference (None, an unknown arm, or one without valid runs) the change
    columns are None.
    """
    direction = direction_of(summaries)
    pick_best = max if direction == "max" else min
    finals = _finals_by_arm(summaries)

    def best_of(values: list[float]) -> float | None:
        return pick_best(values) if values else None

    ref_best = best_of(finals[ref]) if ref in finals else None
    ref_mean = stats.mean(finals[ref]) if ref in finals and finals[ref] else None

    rows: list[ArmRow] = []
    for arm, values in finals.items():
        best = best_of(values)
        mean = stats.mean(values) if values else None
        is_reference = arm == ref
        best_change = mean_change = None
        if not is_reference and values:
            if ref_best is not None and best is not None:
                best_change = improvement_pct(best, ref_best, direction)
            if ref_mean is not None and mean is not None:
                mean_change = improvement_pct(mean, ref_mean, direction)
        rows.append(
            ArmRow(
                arm=arm,
                n=len(values),
                best_of_runs=best,
                mean=mean,
                sd=stats.sample_sd(values) if len(values) >= 2 else None,
                best_change_pct=best_change,
                mean_change_pct=mean_change,
                is_reference=is_reference,
            )
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Table 2: mean best at the end of each round
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundRow:
    """Mean best-so-far when round `round` (1-based) ended, over the `n` of the arm's `runs`
    forgetting runs that reached it with a valid score."""

    arm: str
    round: int
    n: int
    mean_best: float
    change_pct: float | None  # against the previous round's mean; None for round 1
    runs: int = 0  # the arm's runs with at least one team forget: n < runs flags a partial round


def round_table(summaries: Sequence[RunSummary]) -> list[RoundRow]:
    """Table 2: for arms whose runs forgot, the mean best at the end of each round.

    Only runs with at least one team forget take part (a run whose trigger never fired is one
    long round, not a round of a forgetting team). Runs may differ in their number of rounds,
    which is why each row reports how many of the arm's forgetting runs reached it (`n` of
    `runs`): a mean over fewer runs than the previous row's is not a like-for-like change.
    """
    direction = direction_of(summaries)
    rows: list[RoundRow] = []
    for arm in arm_names(summaries):
        forgetting = [s for s in summaries if s.arm == arm and s.team_forgets > 0]
        previous: float | None = None
        for index in range(max((len(s.rounds) for s in forgetting), default=0)):
            ends: list[float] = []
            for run in forgetting:
                best = run.rounds[index].best if index < len(run.rounds) else None
                if best is not None:
                    ends.append(best)
            if not ends:
                continue
            mean = stats.mean(ends)
            change = None if previous is None else improvement_pct(mean, previous, direction)
            rows.append(RoundRow(arm, index + 1, len(ends), mean, change, len(forgetting)))
            previous = mean
    return rows


# ---------------------------------------------------------------------------------------------
# anchoring traces
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRow:
    """Behaviour of an arm's proposals whose turn ended in the spend bin [bin_start, bin_end)."""

    arm: str
    bin_start: float
    bin_end: float
    n_turns: int
    median_lines_changed: float
    median_thinking_tokens: float
    median_output_tokens: float
    empty_rate: float  # share of proposals that made no candidate
    plateau_rate: float  # share of proposals that claimed a plateau


def _total_budget(summaries: Sequence[RunSummary]) -> float:
    budgets = [s.budget_usd for s in summaries if s.budget_usd]
    if budgets:
        return max(budgets)
    return max((s.spend_usd for s in summaries), default=0.0)


def _bin_index(spend: float, width: float, n_bins: int) -> int:
    # round first: 0.3 / 0.1 is 2.9999999999999996, yet 0.3 opens the bin [0.3, 0.4)
    return min(max(math.floor(round(spend / width, 9)), 0), n_bins - 1)


def anchoring_traces(
    summaries: Sequence[RunSummary], *, bin_width: float | None = None, bins: int = 10
) -> list[TraceRow]:
    """Per arm and spend bin: median edit size and token use, empty-turn and plateau rates.

    Proposals (status candidate or empty; error turns are infrastructure noise) are binned by
    the cumulative spend at the end of their turn and pooled across the arm's replicates, as in
    the paper's Figure 5. The default bin width is the total budget over `bins` (10); without
    a USD budget the largest observed spend stands in for it. Spend past the budget lands in
    the last bin. Bins without proposals produce no row.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if bin_width is not None and not bin_width > 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    total = _total_budget(summaries)
    if total <= 0:
        return []
    width = bin_width if bin_width is not None else total / bins
    n_bins = max(1, math.ceil(round(total / width, 9)))

    rows: list[TraceRow] = []
    for arm in arm_names(summaries):
        pooled: defaultdict[int, list[TurnRecord]] = defaultdict(list)
        for summary in summaries:
            if summary.arm != arm:
                continue
            for turn in summary.turns:
                if turn.status in PROPOSAL_STATUSES:
                    pooled[_bin_index(turn.spend, width, n_bins)].append(turn)
        for index in sorted(pooled):
            turns = pooled[index]
            rows.append(
                TraceRow(
                    arm=arm,
                    bin_start=index * width,
                    bin_end=(index + 1) * width,
                    n_turns=len(turns),
                    median_lines_changed=float(statistics.median(t.lines_changed for t in turns)),
                    median_thinking_tokens=float(
                        statistics.median(t.thinking_tokens for t in turns)
                    ),
                    median_output_tokens=float(statistics.median(t.output_tokens for t in turns)),
                    empty_rate=sum(t.status == "empty" for t in turns) / len(turns),
                    plateau_rate=sum(t.plateau_claim for t in turns) / len(turns),
                )
            )
    return rows


# ---------------------------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """Arm A against arm B on the final best score. `diff` is mean(A) - mean(B).

    t, df, p (Welch) and the bootstrap interval of `diff` need two valid runs in each arm
    and are None otherwise. The one-sided p-values test "A is better than B" in the run's
    direction: `p_one_sided` from Welch's t, `mw_p` from the exact Mann-Whitney distribution
    of `wins` (the (A, B) run pairs where A's final best is the better one, ties counting 1/2).
    `wins` needs a valid run in each arm, `mw_p` also at most 20 of them in all.
    """

    arm_a: str
    arm_b: str
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    diff: float | None
    change_pct: float | None  # A against B in the better direction
    t: float | None
    df: float | None
    p: float | None
    p_one_sided: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    wins: float | None
    mw_p: float | None


def _one_sided_p(t: float, df: float, direction: Direction) -> float:
    """Welch p-value of "A is better than B" from the t of mean(A) - mean(B)."""
    t_better = t if direction == "max" else -t  # large when A is the better arm
    if math.isnan(df):  # both samples constant: t is 0 (same constant) or +/-inf
        if t_better == 0:
            return 0.5
        return 0.0 if t_better > 0 else 1.0
    return stats.student_t_cdf(-t_better, df)  # P(T >= t_better), accurate in the far tail


def compare_arms(
    summaries: Sequence[RunSummary],
    arm_a: str,
    arm_b: str,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Welch's t-test, a percentile bootstrap CI and an exact Mann-Whitney test between two
    arms' final best scores."""
    finals = _finals_by_arm(summaries)
    for arm in (arm_a, arm_b):
        if arm not in finals:
            raise ValueError(f"unknown arm {arm!r} (arms: {', '.join(finals)})")
    direction = direction_of(summaries)
    a, b = finals[arm_a], finals[arm_b]
    mean_a = stats.mean(a) if a else None
    mean_b = stats.mean(b) if b else None
    diff = change = None
    if mean_a is not None and mean_b is not None:
        diff = mean_a - mean_b
        change = improvement_pct(mean_a, mean_b, direction)
    t = df = p = p_one_sided = ci_low = ci_high = None
    if len(a) >= 2 and len(b) >= 2:
        welch = stats.welch_t_test(a, b)
        t, df, p = welch.t, welch.df, welch.p
        p_one_sided = _one_sided_p(t, df, direction)
        ci_low, ci_high = stats.bootstrap_diff_ci(
            a, b, n_boot=n_boot, confidence=confidence, seed=seed
        )
    wins = mw_p = None
    if a and b:
        # Mann-Whitney counts pairs where the first sample is larger: negate lower-is-better scores
        sign = 1.0 if direction == "max" else -1.0
        better_a, better_b = [sign * v for v in a], [sign * v for v in b]
        wins = stats.mann_whitney_u(better_a, better_b)
        mw_p = stats.mann_whitney_exact_p(better_a, better_b)
    return Comparison(
        arm_a=arm_a,
        arm_b=arm_b,
        n_a=len(a),
        n_b=len(b),
        mean_a=mean_a,
        mean_b=mean_b,
        diff=diff,
        change_pct=change,
        t=t,
        df=df,
        p=p,
        p_one_sided=p_one_sided,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        wins=wins,
        mw_p=mw_p,
    )
