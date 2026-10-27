"""Analysis of an experiment directory, from the event logs alone.

    python -m research.analyze research/runs/paper-toy-mis \
        --ref solo --compare team-forget:parallel --out out/paper-toy-mis --plots

Prints Table 1 (final best per arm), Table 2 (mean best at the end of each round), anchoring
traces, optional arm comparisons and a per-run listing. `--out` also writes each table as CSV;
`--plots` adds PNG plots when matplotlib is installed. Definitions of the quantities are in
`research.summary` and `research.tables`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lethe.domain.types import Direction
from research.plots import load_pyplot, write_plots
from research.summary import (
    RoundScores,
    RunSummary,
    TurnRecord,
    load_summaries,
    summarize_events,
    summarize_run,
)
from research.tables import (
    ArmRow,
    Comparison,
    RoundRow,
    TraceRow,
    anchoring_traces,
    arm_names,
    arm_table,
    compare_arms,
    direction_of,
    round_table,
)
from research.textfmt import fmt_num, fmt_pct, format_table

__all__ = [
    "Report",
    "RoundScores",
    "RunSummary",
    "TurnRecord",
    "build_report",
    "fmt_num",
    "fmt_pct",
    "format_table",
    "load_summaries",
    "main",
    "render_report",
    "summarize_events",
    "summarize_run",
    "write_csvs",
]

DEFAULT_REF = "solo"
PLOTS_HINT = "uv sync --package lethe-research --extra plots"


# ---------------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    experiment: str
    direction: Direction
    summaries: tuple[RunSummary, ...]
    ref: str | None  # the reference arm actually used for the change columns
    arms: tuple[ArmRow, ...]
    rounds: tuple[RoundRow, ...]
    traces: tuple[TraceRow, ...]
    comparisons: tuple[Comparison, ...]


def build_report(
    summaries: Sequence[RunSummary],
    *,
    experiment: str,
    ref: str | None = DEFAULT_REF,
    strict_ref: bool = False,
    compares: Sequence[tuple[str, str]] = (),
    bin_width: float | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Report:
    """Compute every table. An unknown `ref` is dropped, or an error when `strict_ref`."""
    names = arm_names(summaries)
    if ref is not None and ref not in names:
        if strict_ref:
            raise ValueError(f"unknown reference arm {ref!r} (arms: {', '.join(names)})")
        ref = None
    comparisons = tuple(
        compare_arms(summaries, a, b, n_boot=n_boot, confidence=confidence, seed=seed)
        for a, b in compares
    )
    return Report(
        experiment=experiment,
        direction=direction_of(summaries),
        summaries=tuple(summaries),
        ref=ref,
        arms=tuple(arm_table(summaries, ref=ref)),
        rounds=tuple(round_table(summaries)),
        traces=tuple(anchoring_traces(summaries, bin_width=bin_width)),
        comparisons=comparisons,
    )


def _render_arms(report: Report) -> str:
    change = f"; change vs {report.ref}, positive = better" if report.ref else ""
    rows = []
    for row in report.arms:
        spread = fmt_num(row.mean)
        if row.sd is not None:
            spread += f" ± {fmt_num(row.sd)}"
        if row.is_reference:
            best_change = mean_change = "ref"
        else:
            best_change, mean_change = fmt_pct(row.best_change_pct), fmt_pct(row.mean_change_pct)
        rows.append(
            [row.arm, str(row.n), fmt_num(row.best_of_runs), spread, best_change, mean_change]
        )
    headers = ["arm", "n", "best of runs", "mean ± sd", "best Δ", "mean Δ"]
    return f"Table 1: final best score by arm{change}\n" + format_table(headers, rows)


def _render_rounds(report: Report) -> str:
    title = "Table 2: mean best at the end of each round (change vs previous round)"
    if not report.rounds:
        return f"{title}\n(no arm has team forgets)"
    rows = [
        [r.arm, str(r.round), f"{r.n}/{r.runs}", fmt_num(r.mean_best), fmt_pct(r.change_pct)]
        for r in report.rounds
    ]
    table = format_table(["arm", "round", "n", "mean best", "change"], rows)
    return f"{title}\n{table}\nn = runs that reached the round, of the arm's forgetting runs"


def _render_traces(report: Report) -> str:
    if not report.traces:
        return "Anchoring traces\n(no proposals with spend)"
    first = report.traces[0]
    width = first.bin_end - first.bin_start
    title = f"Anchoring traces: medians per spend bin of ${width:.2f} (proposals only)"
    rows = [
        [
            r.arm,
            f"{r.bin_start:.2f}-{r.bin_end:.2f}",
            str(r.n_turns),
            fmt_num(r.median_lines_changed),
            fmt_num(r.median_thinking_tokens),
            fmt_num(r.median_output_tokens),
            f"{100 * r.empty_rate:.0f}%",
            f"{100 * r.plateau_rate:.0f}%",
        ]
        for r in report.traces
    ]
    headers = ["arm", "spend $", "n", "lines", "think tok", "output tok", "empty", "plateau"]
    return f"{title}\n" + format_table(headers, rows, align="llrrrrrr")


def _render_comparison(c: Comparison) -> str:
    line = f"{c.arm_a} vs {c.arm_b}: n={c.n_a} vs {c.n_b}"
    if c.diff is None:
        return line + "; no valid runs in one of the arms"
    line += f"; mean {fmt_num(c.mean_a)} vs {fmt_num(c.mean_b)}, diff {c.diff:+,.4g}"
    if c.change_pct is not None:
        line += f" ({fmt_pct(c.change_pct)})"
    if c.p is None or c.ci_low is None or c.ci_high is None or c.t is None or c.df is None:
        return line + "; needs at least 2 valid runs per arm for the Welch test and bootstrap CI"
    return (
        f"{line}; Welch t={c.t:.3f} df={c.df:.1f} p={c.p:.4f}; "
        f"{c.confidence:.0%} CI [{c.ci_low:+,.4g}, {c.ci_high:+,.4g}] (bootstrap)"
    )


def _render_runs(report: Report) -> str:
    rows = [
        [
            f"{s.arm}/rep{s.rep}",
            fmt_num(s.final_best),
            fmt_num(s.seed_score),
            f"{s.spend_usd:.2f}",
            str(s.proposals),
            str(s.team_forgets),
            str(s.agent_resets),
            str(s.unplanned_forgets),
            "complete" if s.finished else (s.finish_reason or "no run_finished"),
        ]
        for s in report.summaries
    ]
    headers = [
        "run", "final best", "seed", "spend $", "proposals", "forgets", "resets", "unplanned",
        "ended",
    ]  # fmt: skip
    text = "Runs\n" + format_table(headers, rows, align="lrrrrrrrl")
    if any(s.unplanned_forgets for s in report.summaries):
        text += (
            "\nunplanned: sessions an agent lost without a forget (e.g. after a failed turn);"
            " such a run is not a clean replicate"
        )
    return text


def render_report(report: Report) -> str:
    """The whole report as plain text."""
    better = "higher" if report.direction == "max" else "lower"
    arms = len(report.arms)
    sections = [
        f"Experiment {report.experiment}: {len(report.summaries)} runs in {arms} arms "
        f"({better} is better)",
        _render_arms(report),
        _render_rounds(report),
        _render_traces(report),
    ]
    if report.comparisons:
        lines = "\n".join(_render_comparison(c) for c in report.comparisons)
        sections.append("Comparisons (difference of mean final best, A - B)\n" + lines)
    sections.append(_render_runs(report))
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------------


_RUN_COLUMNS = (
    "arm",
    "rep",
    "direction",
    "finished",
    "finish_reason",
    "started_at",
    "seed_score",
    "final_best",
    "spend_usd",
    "proposals",
    "team_forgets",
    "agent_resets",
    "unplanned_forgets",
)


def _columns(row_type: type) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(row_type))


def _write_csv(path: Path, rows: Iterable[Any], columns: Sequence[str]) -> Path:
    """One CSV row per object, one column per attribute name; None becomes an empty cell."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            values = (getattr(row, column) for column in columns)
            writer.writerow("" if value is None else value for value in values)
    return path


def write_csvs(report: Report, out_dir: Path) -> list[Path]:
    """Write one CSV per table into `out_dir` (created if needed); returns the paths.

    Columns are the attribute names of the row objects (`ArmRow`, `RoundRow`, ...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: list[tuple[str, Sequence[Any], Sequence[str]]] = [
        ("runs.csv", report.summaries, _RUN_COLUMNS),
        ("table1.csv", report.arms, _columns(ArmRow)),
        ("table2.csv", report.rounds, _columns(RoundRow)),
        ("anchoring_traces.csv", report.traces, _columns(TraceRow)),
    ]
    if report.comparisons:
        tables.append(("comparisons.csv", report.comparisons, _columns(Comparison)))
    return [_write_csv(out_dir / name, rows, columns) for name, rows, columns in tables]


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def _parse_compare(text: str) -> tuple[str, str]:
    arm_a, sep, arm_b = text.partition(":")
    if not sep or not arm_a or not arm_b or ":" in arm_b:
        raise ValueError(f"--compare expects A:B (two arm names), got {text!r}")
    return arm_a, arm_b


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research.analyze",
        description="Tables, anchoring traces and arm comparisons for one experiment directory.",
    )
    parser.add_argument("experiment_dir", type=Path, help="runs_root/<experiment>")
    parser.add_argument(
        "--ref",
        default=None,
        metavar="ARM",
        help=f"reference arm for the change columns (default: {DEFAULT_REF}, if present)",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="A:B",
        help="Welch test and bootstrap CI of A against B (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write CSV files here")
    parser.add_argument("--plots", action="store_true", help="write PNG plots into --out")
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        metavar="USD",
        help="spend bin width for the anchoring traces (default: total budget / 10)",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="include runs without a run_finished event (their 'final best' is best so far)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.plots and args.out is None:
        return _fail("--plots needs --out DIR to write the PNG files into")
    try:
        compares = [_parse_compare(text) for text in args.compare]
        summaries = load_summaries(args.experiment_dir, include_unfinished=args.include_unfinished)
        if not summaries:
            return _fail(
                f"no finished runs under {args.experiment_dir} "
                "(--include-unfinished shows runs in progress)"
            )
        report = build_report(
            summaries,
            experiment=args.experiment_dir.resolve().name,
            ref=args.ref or DEFAULT_REF,
            strict_ref=args.ref is not None,
            compares=compares,
            bin_width=args.bin_width,
        )
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    print(render_report(report), end="")
    written: list[Path] = []
    if args.out is not None:
        written += write_csvs(report, args.out)
    if args.plots:
        plt = load_pyplot()
        if plt is None:
            print(f"matplotlib is not installed; skipping plots ({PLOTS_HINT})", file=sys.stderr)
        else:
            written += write_plots(report.summaries, report.traces, args.out, plt)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
