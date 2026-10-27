"""Per-run summaries, folded from nothing but a run's `events.jsonl`.

Definitions used throughout the analysis:

- seed score: the judge score of candidate c0000 (agent "seed"), None if it was invalid.
- final best: the best valid score of any candidate, seed included, under the run's direction.
  It is also defined for runs that have not finished: it is then the best so far.
- finished: the run ended by itself (`run_finished` whose status is "completed" and whose reason
  is not an abnormal one, see `research.runs`). `finish_reason` keeps the raw reason.
- spend: cumulative `cost_usd` over every turn, errors included (a failed turn still costs).
- proposal: a turn with status "candidate" or "empty"; "error" turns are not proposals.
- round: the stretch between team forgets (`EpochStarted(scope="team")`). A round's seed score is
  the best score among the candidates its agents started from; its "best at end" is the best score
  of the whole run so far when the round ended (cumulative, like the paper's Table 2).
  Agent-scope resets (the `stalled` trigger) do not end a round.
- curve: the best-so-far step function over spend, as (spend, best) points: one per improvement,
  plus a closing point at the total spend so the last plateau has a length. A candidate is
  credited with the spend at the end of the turn that proposed it, not at the (later, because
  judging is asynchronous) moment its judge result arrived, so the curve answers "what had
  been reached once this much was spent".
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

from lethe.domain.events import (
    CandidateCommitted,
    Envelope,
    EpochStarted,
    JudgeFinished,
    RunFinished,
    RunStarted,
    TurnFinished,
    TurnStatus,
)
from lethe.domain.types import Direction, is_better
from lethe.store import read_events
from research.runs import EVENTS_FILE, discover_runs, ending_is_complete

__all__ = [
    "PROPOSAL_STATUSES",
    "RoundScores",
    "RunSummary",
    "TurnRecord",
    "best_at",
    "load_summaries",
    "summarize_events",
    "summarize_run",
]

PROPOSAL_STATUSES = frozenset({"candidate", "empty"})
_SEED_CAND_ID = "c0000"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnRecord:
    """One agent turn. `spend` is the run's cumulative spend (USD) at the end of the turn,
    `cost` what the turn itself cost."""

    spend: float
    lines_changed: int
    output_tokens: int
    thinking_tokens: int
    status: TurnStatus
    plateau_claim: bool
    cost: float = 0.0


class RoundScores(NamedTuple):
    """Scores bracketing one round: where it started and the best so far when it ended."""

    seed: float | None
    best: float | None


@dataclass(frozen=True)
class RunSummary:
    arm: str
    rep: int
    run_dir: Path
    direction: Direction
    finished: bool
    finish_reason: str | None  # the run_finished reason; None while there is no run_finished
    started_at: float  # `ts` of the first event, seconds since the epoch
    seed_score: float | None
    final_best: float | None
    spend_usd: float
    proposals: int
    team_forgets: int
    agent_resets: int
    rounds: tuple[RoundScores, ...]
    curve: tuple[tuple[float, float], ...]
    turns: tuple[TurnRecord, ...]
    budget_usd: float | None  # the run's USD cap, None when it was capped in another unit
    # sessions an agent lost without a forget (e.g. a failed turn the harness did not resume): a
    # treatment the arm was not assigned, so the run is not a clean replicate
    unplanned_forgets: int = 0


def _usd_budget(config: object) -> float | None:
    budget = config.get("budget") if isinstance(config, Mapping) else None
    usd = budget.get("usd") if isinstance(budget, Mapping) else None
    if isinstance(usd, bool) or not isinstance(usd, int | float):
        return None
    return float(usd) if usd > 0 else None


def _fresh_sessions(config: object) -> bool:
    """team.session = "fresh": every turn starts a new session by design."""
    team = config.get("team") if isinstance(config, Mapping) else None
    return isinstance(team, Mapping) and team.get("session") == "fresh"


def _valid_score(event: JudgeFinished) -> float | None:
    if event.valid and event.score is not None and math.isfinite(event.score):
        return float(event.score)
    return None


class _Fold:
    """Accumulates a summary one envelope at a time."""

    def __init__(self) -> None:
        self.started = False
        self.direction: Direction = "max"  # replaced by run_started
        self.budget_usd: float | None = None
        self.started_at = 0.0
        self.seed_ids = {_SEED_CAND_ID}
        self.scores: dict[str, float] = {}
        self.seed_score: float | None = None
        self.best: float | None = None
        self.spend = 0.0
        self.proposals = 0
        self.team_forgets = 0
        self.agent_resets = 0
        self.fresh_sessions = False
        self.held: dict[str, str | None] = {}  # agent -> session of its last completed turn
        self.unplanned_forgets = 0
        self.finish_reason: str | None = None
        self.finish_status = "completed"
        self.turn_spend: dict[str, float] = {}  # candidate -> spend at the end of its turn
        self.scored: list[tuple[float, float]] = []  # (spend credited, score) per valid candidate
        self.round_bests: list[float | None] = []  # best so far at the end of each closed round
        self.round_seeds: list[float | None] = []  # seeds of rounds 2.. (round 1 starts from c0000)
        self.turns: list[TurnRecord] = []

    def feed(self, envelope: Envelope) -> None:
        event = envelope.event
        if isinstance(event, RunStarted):
            self.started = True
            self.direction = event.direction
            self.budget_usd = _usd_budget(event.config)
            self.fresh_sessions = _fresh_sessions(event.config)
            self.started_at = envelope.ts
        elif not self.started:
            raise ValueError("event log does not begin with a run_started event")
        elif isinstance(event, CandidateCommitted):
            if event.agent == "seed":
                self.seed_ids.add(event.cand_id)
        elif isinstance(event, JudgeFinished):
            self._judged(event)
        elif isinstance(event, TurnFinished):
            self._turn(event)
        elif isinstance(event, EpochStarted):
            self._epoch(event)
        elif isinstance(event, RunFinished):
            self.finish_reason, self.finish_status = event.reason, event.status

    def _judged(self, event: JudgeFinished) -> None:
        score = _valid_score(event)
        if score is None:
            return
        self.scores[event.cand_id] = score
        if event.cand_id in self.seed_ids:
            self.seed_score = score
            self.scored.append((0.0, score))
        else:
            self.scored.append((self.turn_spend.get(event.cand_id, self.spend), score))
        if self.best is None or is_better(score, self.best, self.direction):
            self.best = score

    def _turn(self, event: TurnFinished) -> None:
        self.spend += event.cost_usd
        held = self.held.get(event.agent)
        if held is not None and event.session_id != held and not self.fresh_sessions:
            self.unplanned_forgets += 1  # counted once: until a turn completes, nothing is held
            self.held[event.agent] = None
        if event.status != "error":
            self.held[event.agent] = event.session_id
        if event.cand_id is not None:
            self.turn_spend[event.cand_id] = self.spend
        if event.status in PROPOSAL_STATUSES:
            self.proposals += 1
        self.turns.append(
            TurnRecord(
                spend=self.spend,
                lines_changed=event.lines_changed,
                output_tokens=event.output_tokens,
                thinking_tokens=event.thinking_tokens,
                status=event.status,
                plateau_claim=event.plateau_claim,
                cost=event.cost_usd,
            )
        )

    def _epoch(self, event: EpochStarted) -> None:
        for agent in event.epochs:  # a planned forget: the next session is a new one by design
            self.held[agent] = None
        if event.scope == "team":
            self.team_forgets += 1
            self.round_bests.append(self.best)
            self.round_seeds.append(self._best_of(event.seeds.values()))
        elif event.scope == "agent":
            self.agent_resets += 1

    def _best_of(self, cand_ids: Iterable[str]) -> float | None:
        known = [self.scores[c] for c in cand_ids if c in self.scores]
        if not known:
            return None
        return max(known) if self.direction == "max" else min(known)

    def result(self, arm: str, rep: int, run_dir: Path) -> RunSummary:
        if not self.started:
            raise ValueError("event log has no run_started event")
        curve: list[tuple[float, float]] = []
        for spend, score in sorted(self.scored, key=itemgetter(0)):  # stable: ties keep log order
            if not curve or is_better(score, curve[-1][1], self.direction):
                curve.append((spend, score))
        if curve and curve[-1][0] < self.spend:
            curve.append((self.spend, curve[-1][1]))
        seeds = [self.seed_score, *self.round_seeds]
        bests = [*self.round_bests, self.best]
        return RunSummary(
            arm=arm,
            rep=rep,
            run_dir=run_dir,
            direction=self.direction,
            finished=self.finish_reason is not None
            and ending_is_complete(self.finish_status, self.finish_reason),
            finish_reason=self.finish_reason,
            started_at=self.started_at,
            seed_score=self.seed_score,
            final_best=self.best,
            spend_usd=self.spend,
            proposals=self.proposals,
            team_forgets=self.team_forgets,
            agent_resets=self.agent_resets,
            rounds=tuple(RoundScores(s, b) for s, b in zip(seeds, bests, strict=True)),
            curve=tuple(curve),
            turns=tuple(self.turns),
            budget_usd=self.budget_usd,
            unplanned_forgets=self.unplanned_forgets,
        )


def summarize_events(
    envelopes: Iterable[Envelope], *, arm: str, rep: int, run_dir: Path
) -> RunSummary:
    """Fold a stream of envelopes into a `RunSummary`. Raises ValueError without `run_started`."""
    fold = _Fold()
    for envelope in envelopes:
        fold.feed(envelope)
    return fold.result(arm, rep, run_dir)


def summarize_run(run_dir: Path, *, arm: str, rep: int) -> RunSummary:
    """Summarize `<run_dir>/events.jsonl`.

    Raises FileNotFoundError when there is no log and ValueError when the log is unreadable.
    """
    run_dir = Path(run_dir)
    path = run_dir / EVENTS_FILE
    try:
        envelopes = list(read_events(path))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: unreadable event log ({type(exc).__name__}: {exc})") from exc
    return summarize_events(envelopes, arm=arm, rep=rep, run_dir=run_dir)


def best_at(curve: Sequence[tuple[float, float]], spend: float) -> float | None:
    """Value of a best-vs-spend step function at `spend` (None before its first point)."""
    index = bisect.bisect_right(curve, spend, key=itemgetter(0)) - 1
    return curve[index][1] if index >= 0 else None


def load_summaries(experiment_dir: Path, *, include_unfinished: bool = False) -> list[RunSummary]:
    """Summaries of every run under `experiment_dir` (`<arm>/rep<k>/events.jsonl`).

    Arms come in the order they first ran, replicates in numeric order. Runs without a
    `run_finished` event are excluded unless `include_unfinished` (their "final best" is only
    the best so far); unreadable logs are skipped. Both are reported through the log.
    """
    summaries: list[RunSummary] = []
    unfinished: list[str] = []  # "arm/rep<k> (why)"
    for location in discover_runs(Path(experiment_dir)):
        label = f"{location.arm}/rep{location.rep}"
        try:
            summary = summarize_run(location.run_dir, arm=location.arm, rep=location.rep)
        except (OSError, ValueError) as exc:
            log.warning("skipping %s: %s", label, exc)
            continue
        if summary.finished or include_unfinished:
            summaries.append(summary)
        else:
            unfinished.append(f"{label} ({summary.finish_reason or 'no run_finished event'})")
    if unfinished:
        log.warning(
            "excluding %d unfinished run(s): %s (pass --include-unfinished to keep them)",
            len(unfinished),
            ", ".join(unfinished),
        )
    first_ran: dict[str, float] = {}
    for summary in summaries:
        first_ran[summary.arm] = min(first_ran.get(summary.arm, math.inf), summary.started_at)
    summaries.sort(key=lambda s: (first_ran[s.arm], s.arm, s.rep))
    return summaries
