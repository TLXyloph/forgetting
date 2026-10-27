"""Plain-text sections of the analysis report for the tables of `research.outcomes`.

`research.analyze.render_report` places them after the per-run listing.
"""

from __future__ import annotations

from collections.abc import Sequence

from lethe.domain.types import Direction
from research.outcomes import EmptyTurnRow, ForgetOutcomeRow, ThresholdRow
from research.textfmt import fmt_num, fmt_share, format_table

__all__ = ["render_empty_turns", "render_forget_outcomes", "render_thresholds"]


def _usd(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_forget_outcomes(rows: Sequence[ForgetOutcomeRow]) -> str:
    """One line per forgetting arm: "<arm>: k of n forgets led to a new best ..."."""
    title = "Forget outcomes: team forgets followed by a new best within the next round"
    if not rows:
        return f"{title}\n(no arm has team forgets)"
    lines = [
        f"{r.arm}: {r.new_bests} of {r.forgets} forgets led to a new best within the next round"
        f" ({fmt_share(r.new_best_rate)}; {r.runs} forgetting runs)"
        for r in rows
    ]
    return "\n".join([title, *lines])


def render_empty_turns(rows: Sequence[EmptyTurnRow]) -> str:
    """Per arm: turns, empty and error turns, and the shares of turns and spend on empty ones."""
    cells = [
        [
            r.arm,
            str(r.runs),
            str(r.turns),
            str(r.empty_turns),
            str(r.error_turns),
            fmt_share(r.empty_share),
            _usd(r.spend_usd),
            _usd(r.empty_spend_usd),
            fmt_share(r.empty_spend_share),
        ]
        for r in rows
    ]
    headers = [
        "arm", "runs", "turns", "empty", "error", "empty %", "spend $", "empty $", "of spend",
    ]  # fmt: skip
    return (
        "Empty turns: turns that ended without a candidate, and the spend they took\n"
        + format_table(headers, cells)
        + "\nshares are of all turns and all spend, failed (error) turns included"
    )


def render_thresholds(rows: Sequence[ThresholdRow], direction: Direction) -> str:
    """Per arm and threshold: runs that reached the score, and the spend at which they did."""
    better = "higher" if direction == "max" else "lower"
    cells = [
        [
            r.arm,
            fmt_num(r.threshold),
            f"{r.reached}/{r.runs}",
            _usd(r.mean_spend),
            _usd(r.min_spend),
            _usd(r.max_spend),
        ]
        for r in rows
    ]
    headers = ["arm", "score", "reached", "mean $", "min $", "max $"]
    return (
        f"Cost to reach a score: spend at which a run's best first reached it (or {better})\n"
        + format_table(headers, cells)
        + "\nspend columns are over the runs that reached the score"
    )
