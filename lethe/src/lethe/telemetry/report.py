"""Run reports: summary tables (text) and CSV exports computed from `RunState`."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from lethe.domain.state import SEED_ID, RunState
from lethe.policy.prompts import format_score
from lethe.policy.triggers import relative_gain

EDGE_TURNS = 3  # turns at each end of a session used for opening/closing edit sizes


@dataclass(frozen=True)
class RoundRow:
    round: int
    seed_id: str
    seed_score: float | None
    best_id: str | None
    best_score: float | None
    gain_pct: float | None
    proposals: int
    spent_usd: float
    reason: str


@dataclass(frozen=True)
class AgentRow:
    agent: str
    turns: int
    candidates: int
    empty: int
    errors: int
    plateau_claims: int
    resets: int
    opening_lines: float | None  # median lines changed over the first turns of its sessions
    closing_lines: float | None  # ... and over the last turns of its sessions


@dataclass(frozen=True)
class RunReport:
    run_id: str | None
    task: str | None
    direction: str
    finished: str | None
    seed_score: float | None
    best_id: str | None
    best_score: float | None
    best_agent: str | None
    improvement_pct: float | None
    spent_usd: float
    proposals: int
    rounds: tuple[RoundRow, ...]
    agents: tuple[AgentRow, ...]
    rate_limited: int
    five_hour_utilization: float | None


def _gain_pct(new: float | None, old: float | None, direction: str) -> float | None:
    if new is None or old is None:
        return None
    return 100.0 * relative_gain(new, old, direction)  # type: ignore[arg-type]


def build_report(state: RunState) -> RunReport:
    seed = state.candidates.get(SEED_ID)
    seed_score = seed.score if seed is not None and seed.valid else None
    best = state.best()
    return RunReport(
        run_id=state.run_id,
        task=state.task_name,
        direction=state.direction,
        finished=state.finished.reason if state.finished else None,
        seed_score=seed_score,
        best_id=best.cand_id if best else None,
        best_score=best.score if best else None,
        best_agent=best.agent if best else None,
        improvement_pct=_gain_pct(best.score if best else None, seed_score, state.direction),
        spent_usd=state.spent_usd,
        proposals=state.proposals,
        rounds=tuple(_rounds(state)),
        agents=tuple(_agents(state)),
        rate_limited=len(state.rate_limits),
        five_hour_utilization=state.five_hour_utilization,
    )


def _rounds(state: RunState) -> list[RoundRow]:
    starts = [r for r in state.epoch_history if r.scope in ("start", "team")]
    rows: list[RoundRow] = []
    for index, record in enumerate(starts):
        end_seq = starts[index + 1].seq if index + 1 < len(starts) else state.last_seq + 1
        seeds = [state.candidates[c] for c in dict.fromkeys(record.seeds.values())]
        valid_seeds = [s for s in seeds if s.valid and s.score is not None]
        lead = max(
            valid_seeds,
            key=lambda s: s.score if state.direction == "max" else -s.score,
            default=seeds[0] if seeds else None,
        )
        best = state.best_at(end_seq - 1)
        turns = [t for t in state.turns if record.seq < t.seq < end_seq]
        seed_score = lead.score if lead is not None and lead.valid else None
        rows.append(
            RoundRow(
                round=index + 1,
                seed_id=lead.cand_id if lead else SEED_ID,
                seed_score=seed_score,
                best_id=best.cand_id if best else None,
                best_score=best.score if best else None,
                gain_pct=_gain_pct(best.score if best else None, seed_score, state.direction),
                proposals=sum(1 for t in turns if t.ok),
                spent_usd=sum(t.cost_usd for t in turns),
                reason=record.reason,
            )
        )
    return rows


def _agents(state: RunState) -> list[AgentRow]:
    rows = []
    for agent in state.agents:
        turns = [t for t in state.turns if t.agent == agent]
        sessions: dict[str, list[int]] = {}
        for t in turns:
            if t.ok:
                sessions.setdefault(t.session_id, []).append(t.lines_changed)
        opening = [x for lines in sessions.values() for x in lines[:EDGE_TURNS]]
        closing = [
            x for lines in sessions.values() if len(lines) > EDGE_TURNS for x in lines[-EDGE_TURNS:]
        ]
        rows.append(
            AgentRow(
                agent=agent,
                turns=sum(1 for t in turns if t.ok),
                candidates=sum(1 for t in turns if t.status == "candidate"),
                empty=sum(1 for t in turns if t.status == "empty"),
                errors=sum(1 for t in turns if t.status == "error"),
                plateau_claims=sum(1 for t in turns if t.plateau_claim),
                resets=sum(
                    1 for r in state.epoch_history if r.scope == "agent" and agent in r.epochs
                ),
                opening_lines=statistics.median(opening) if opening else None,
                closing_lines=statistics.median(closing) if closing else None,
            )
        )
    return rows


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def format_report(report: RunReport) -> str:
    better = "higher" if report.direction == "max" else "lower"
    lines = [
        f"Run {report.run_id} · task {report.task} · {better} is better",
        f"Finished: {report.finished or 'still running'}",
        f"Best: {report.best_id} = {format_score(report.best_score)} (by {report.best_agent}); "
        f"seed {format_score(report.seed_score)}; change {_pct(report.improvement_pct)}",
        f"Spend: ${report.spent_usd:.2f} API-equivalent · {report.proposals} proposals"
        + (
            f" · 5-hour window {report.five_hour_utilization:.0%}"
            if report.five_hour_utilization is not None
            else ""
        )
        + (f" · rate-limited {report.rate_limited}x" if report.rate_limited else ""),
        "",
        "Round  start-from        best              gain     proposals  spend    reason",
    ]
    for r in report.rounds:
        lines.append(
            f"{r.round:<6} {r.seed_id + ' ' + format_score(r.seed_score):<17} "
            f"{(r.best_id or '-') + ' ' + format_score(r.best_score):<17} "
            f"{_pct(r.gain_pct):<8} {r.proposals:<10} ${r.spent_usd:<7.2f} {r.reason}"
        )
    lines += ["", "Agent  turns  cands  empty  errors  plateau-claims  resets  lines open->close"]
    for a in report.agents:
        span = (
            f"{format_score(a.opening_lines)} -> {format_score(a.closing_lines)}"
            if a.opening_lines is not None
            else "n/a"
        )
        lines.append(
            f"{a.agent:<6} {a.turns:<6} {a.candidates:<6} {a.empty:<6} {a.errors:<7} "
            f"{a.plateau_claims:<15} {a.resets:<7} {span}"
        )
    return "\n".join(lines)


def write_csvs(state: RunState, out_dir: Path) -> list[Path]:
    """Write turns.csv, best_curve.csv and rounds.csv for analysis; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = out_dir / "turns.csv"
    with turns_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "seq",
                "agent",
                "epoch",
                "session_id",
                "turn_in_session",
                "status",
                "cand_id",
                "score",
                "cost_usd",
                "spent_after",
                "output_tokens",
                "thinking_tokens",
                "files_read",
                "edit_calls",
                "lines_changed",
                "plateau_claim",
                "duration_s",
            ]
        )
        for t in state.turns:
            score = state.candidates[t.cand_id].score if t.cand_id in state.candidates else None
            writer.writerow(
                [
                    t.seq,
                    t.agent,
                    t.epoch,
                    t.session_id,
                    t.turn_in_session,
                    t.status,
                    t.cand_id or "",
                    "" if score is None else score,
                    t.cost_usd,
                    round(t.spent_after, 6),
                    t.output_tokens,
                    t.thinking_tokens,
                    t.files_read,
                    t.edit_calls,
                    t.lines_changed,
                    int(t.plateau_claim),
                    t.duration_s,
                ]
            )
    curve_path = out_dir / "best_curve.csv"
    with curve_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["seq", "cand_id", "score", "spent_usd", "proposals"])
        for b in state.best_history:
            writer.writerow([b.seq, b.cand_id, b.score, round(b.spent_usd, 6), b.proposals])
    rounds_path = out_dir / "rounds.csv"
    with rounds_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "round",
                "seed_id",
                "seed_score",
                "best_id",
                "best_score",
                "gain_pct",
                "proposals",
                "spent_usd",
                "reason",
            ]
        )
        for r in _rounds(state):
            writer.writerow(
                [
                    r.round,
                    r.seed_id,
                    r.seed_score,
                    r.best_id,
                    r.best_score,
                    r.gain_pct,
                    r.proposals,
                    round(r.spent_usd, 6),
                    r.reason,
                ]
            )
    return [turns_path, curve_path, rounds_path]
