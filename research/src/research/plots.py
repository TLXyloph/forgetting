"""Optional PNG plots. matplotlib is not a dependency: install it with the `plots` extra.

    uv sync --package lethe-research --extra plots

`load_pyplot()` returns None when matplotlib is missing, and callers print a one-line notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any

from research.summary import RunSummary, best_at
from research.tables import TraceRow, arm_names

__all__ = ["PALETTE", "WARMUP", "load_pyplot", "score_window", "write_plots"]

# Okabe-Ito: distinguishable under the common forms of colour blindness
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
WARMUP = 0.05  # share of the budget whose opening drop the score axis may leave off-scale


def score_window(summaries: Sequence[RunSummary], *, warmup: float) -> tuple[float, float] | None:
    """Score-axis limits that show where the arms compete: from every run's best so far once
    `warmup` of the budget is spent to every run's final best, padded by 5%. The seed and the
    first improvements (147,734 cycles on VLIW, against finals near 1,200) would otherwise
    flatten every difference that matters. None when no run has a curve."""
    curves = [s.curve for s in summaries if s.curve]
    if not curves:
        return None
    budget = max((s.budget_usd or s.spend_usd) for s in summaries if s.curve)
    values = []
    for curve in curves:
        early = best_at(curve, warmup * budget)
        values += [curve[0][1] if early is None else early, curve[-1][1]]
    low, high = min(values), max(values)
    pad = 0.05 * (high - low) or 0.01 * abs(high) or 1.0
    return low - pad, high + pad


def load_pyplot() -> Any | None:
    """`matplotlib.pyplot` on the headless Agg backend, or None if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def write_plots(
    summaries: Sequence[RunSummary],
    traces: Sequence[TraceRow],
    out_dir: Path,
    plt: Any,
) -> list[Path]:
    """Write `best_vs_spend.png` and `anchoring_traces.png` into `out_dir`; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = arm_names(summaries)
    colors = {arm: PALETTE[i % len(PALETTE)] for i, arm in enumerate(arms)}
    written = [_best_vs_spend(plt, summaries, colors, out_dir / "best_vs_spend.png")]
    if traces:
        written.append(_anchoring(plt, traces, colors, out_dir / "anchoring_traces.png"))
    return written


def _best_vs_spend(
    plt: Any, summaries: Sequence[RunSummary], colors: Mapping[str, str], path: Path
) -> Path:
    """Best score so far against total spend: every replicate thin, each arm's best one bold."""
    higher = summaries[0].direction == "max"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arm in colors:
        scored = [
            (s.final_best, s)
            for s in summaries
            if s.arm == arm and s.curve and s.final_best is not None
        ]
        if not scored:
            continue
        _, best_run = (max if higher else min)(scored, key=itemgetter(0))
        for _, run in scored:
            spend, best = zip(*run.curve, strict=True)
            is_best = run is best_run
            ax.step(
                spend,
                best,
                where="post",
                color=colors[arm],
                linewidth=2.4 if is_best else 1.0,
                alpha=1.0 if is_best else 0.35,
                label=arm if is_best else None,
            )
    window = score_window(summaries, warmup=WARMUP)
    if window is not None:
        ax.set_ylim(*window)
    ax.set_xlabel("Total dollars spent (API-equivalent USD)")
    ax.set_ylabel(f"Best score so far (first {WARMUP:.0%} of spend may be off-scale)")
    ax.set_title(f"Best score vs spend ({'higher' if higher else 'lower'} is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    return _save(plt, fig, path)


def _anchoring(plt: Any, traces: Sequence[TraceRow], colors: Mapping[str, str], path: Path) -> Path:
    """Median edit size and token use per spend bin, plus empty-turn and plateau-claim rates."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    medians = (
        (axes[0][0], "Median lines changed per proposal", "median_lines_changed"),
        (axes[0][1], "Median thinking tokens per proposal", "median_thinking_tokens"),
        (axes[1][0], "Median output tokens per proposal", "median_output_tokens"),
    )
    rates = axes[1][1]
    for arm, color in colors.items():
        rows = [r for r in traces if r.arm == arm]
        if not rows:
            continue
        centres = [(r.bin_start + r.bin_end) / 2 for r in rows]
        for ax, _title, field in medians:
            ax.plot(centres, [getattr(r, field) for r in rows], marker="o", color=color, label=arm)
        rates.plot(centres, [r.empty_rate for r in rows], marker="o", color=color, linestyle="-")
        rates.plot(centres, [r.plateau_rate for r in rows], marker="s", color=color, linestyle="--")
    for ax, title, _field in medians:
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    rates.set_title("Empty turns (solid), plateau claims (dashed)")
    rates.set_ylim(0.0, 1.0)
    rates.grid(True, alpha=0.3)
    axes[1][0].set_xlabel("Total dollars spent (bin centre)")
    axes[1][1].set_xlabel("Total dollars spent (bin centre)")
    axes[0][0].legend(loc="best", frameon=False)
    return _save(plt, fig, path)


def _save(plt: Any, fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
