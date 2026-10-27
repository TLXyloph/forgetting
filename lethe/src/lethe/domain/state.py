"""RunState: a pure fold over a run's event log.

The orchestrator and every analysis read the run through this projection, so live runs and
replays from `events.jsonl` see exactly the same state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lethe.domain.events import (
    CandidateCommitted,
    Envelope,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RateLimited,
    RunFinished,
    RunStarted,
    TurnFinished,
)
from lethe.domain.types import Direction, is_better

SEED_ID = "c0000"
OK_STATUSES = ("candidate", "empty")
# Error kinds after which the session survives: the next turn resumes the last OK session.
# Anything else ("process", or no kind in logs written before kinds were logged) starts fresh.
RESUMABLE_ERRORS = ("api", "timeout", "harness")
MAX_RESUMED_ERRORS = 2  # consecutive transient errors after which a session is abandoned


@dataclass
class CandidateInfo:
    cand_id: str
    sha: str
    parent_id: str | None
    agent: str
    epoch: int
    session_id: str | None
    judged_key: str
    lines_added: int
    lines_deleted: int
    files_changed: int
    summary: str
    seq: int
    judged: bool = False
    valid: bool | None = None
    score: float | None = None
    feedback: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    cases: dict[str, float] | None = None

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted


@dataclass(frozen=True)
class TurnInfo:
    seq: int
    agent: str
    epoch: int
    session_id: str
    turn_in_session: int
    status: str
    cand_id: str | None
    cost_usd: float
    output_tokens: int
    thinking_tokens: int
    files_read: int
    edit_calls: int
    lines_changed: int
    plateau_claim: bool
    duration_s: float
    spent_after: float  # cumulative run spend once this turn is counted
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES


@dataclass(frozen=True)
class BestEntry:
    seq: int
    cand_id: str
    score: float
    spent_usd: float
    proposals: int


@dataclass(frozen=True)
class EpochRecord:
    seq: int
    scope: str
    seeds: dict[str, str]
    epochs: dict[str, int]
    reason: str


@dataclass
class _AgentEpoch:
    epoch: int
    seed_id: str
    start_seq: int


class RunState:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.task_name: str | None = None
        self.direction: Direction = "max"
        self.agents: tuple[str, ...] = ()
        self.config: dict[str, Any] = {}
        self.candidates: dict[str, CandidateInfo] = {}
        self.turns: list[TurnInfo] = []
        self.best_history: list[BestEntry] = []
        self.epoch_history: list[EpochRecord] = []
        self.forgets: list[ForgetDecided] = []
        self.rate_limits: list[RateLimited] = []
        self.finished: RunFinished | None = None
        self.spent_usd = 0.0
        self.proposals = 0
        self.five_hour_utilization: float | None = None  # latest subscription meter readings
        self.seven_day_utilization: float | None = None
        self.started_ts: float | None = None
        self.last_ts: float | None = None
        self.last_seq = -1
        self._epochs: dict[str, _AgentEpoch] = {}
        self._best_id: str | None = None
        self._next_cand = 0

    # ---------------------------------------------------------------- fold
    def apply(self, envelope: Envelope) -> None:
        event, seq = envelope.event, envelope.seq
        self.last_seq = seq
        self.last_ts = envelope.ts
        if self.started_ts is None:
            self.started_ts = envelope.ts
        if isinstance(event, RunStarted):
            self.run_id, self.task_name = event.run_id, event.task_name
            self.direction, self.agents = event.direction, tuple(event.agents)
            self.config = dict(event.config)
        elif isinstance(event, CandidateCommitted):
            self._apply_candidate(event, seq)
        elif isinstance(event, JudgeFinished):
            self._apply_judge(event, seq)
        elif isinstance(event, TurnFinished):
            self._apply_turn(event, seq)
        elif isinstance(event, EpochStarted):
            for agent, epoch in event.epochs.items():
                self._epochs[agent] = _AgentEpoch(epoch, event.seeds[agent], seq)
            self.epoch_history.append(
                EpochRecord(seq, event.scope, dict(event.seeds), dict(event.epochs), event.reason)
            )
        elif isinstance(event, ForgetDecided):
            self.forgets.append(event)
        elif isinstance(event, RateLimited):
            self.rate_limits.append(event)
        elif isinstance(event, RunFinished):
            self.finished = event

    def _apply_candidate(self, event: CandidateCommitted, seq: int) -> None:
        self.candidates[event.cand_id] = CandidateInfo(
            cand_id=event.cand_id,
            sha=event.sha,
            parent_id=event.parent_id,
            agent=event.agent,
            epoch=event.epoch,
            session_id=event.session_id,
            judged_key=event.judged_key,
            lines_added=event.lines_added,
            lines_deleted=event.lines_deleted,
            files_changed=event.files_changed,
            summary=event.summary,
            seq=seq,
        )
        number = _cand_number(event.cand_id)
        if number is not None:
            self._next_cand = max(self._next_cand, number + 1)

    def _apply_judge(self, event: JudgeFinished, seq: int) -> None:
        cand = self.candidates.get(event.cand_id)
        if cand is None:
            return
        cand.judged, cand.valid, cand.score = True, event.valid, event.score
        cand.feedback, cand.metrics = event.feedback, dict(event.metrics)
        cand.cases = dict(event.cases) if event.cases is not None else None
        if not event.valid or event.score is None:
            return
        best = self.best()
        if best is None or is_better(event.score, best.score, self.direction):  # type: ignore[arg-type]
            self._best_id = cand.cand_id
            self.best_history.append(
                BestEntry(seq, cand.cand_id, event.score, self.spent_usd, self.proposals)
            )

    def _apply_turn(self, event: TurnFinished, seq: int) -> None:
        self.spent_usd += event.cost_usd
        if event.status in OK_STATUSES:
            self.proposals += 1
        if event.five_hour_utilization is not None:
            self.five_hour_utilization = event.five_hour_utilization
        if event.seven_day_utilization is not None:
            self.seven_day_utilization = event.seven_day_utilization
        self.turns.append(
            TurnInfo(
                seq=seq,
                agent=event.agent,
                epoch=event.epoch,
                session_id=event.session_id,
                turn_in_session=event.turn_in_session,
                status=event.status,
                cand_id=event.cand_id,
                cost_usd=event.cost_usd,
                output_tokens=event.output_tokens,
                thinking_tokens=event.thinking_tokens,
                files_read=event.files_read,
                edit_calls=event.edit_calls,
                lines_changed=event.lines_changed,
                plateau_claim=event.plateau_claim,
                duration_s=event.duration_s,
                spent_after=self.spent_usd,
                error_kind=event.error_kind,
            )
        )

    # ------------------------------------------------------------- queries
    def best(self) -> CandidateInfo | None:
        return self.candidates.get(self._best_id) if self._best_id else None

    def next_cand_id(self) -> str:
        return f"c{self._next_cand:04d}"

    def epoch(self, agent: str) -> int:
        record = self._epochs.get(agent)
        return record.epoch if record else 0

    def epoch_start_seq(self, agent: str) -> int:
        record = self._epochs.get(agent)
        return record.start_seq if record else -1

    def epoch_seed(self, agent: str) -> CandidateInfo:
        record = self._epochs.get(agent)
        return self.candidates[record.seed_id if record else SEED_ID]

    def own_candidates(self, agent: str) -> list[CandidateInfo]:
        """The agent's candidates in its current epoch, oldest first."""
        epoch = self.epoch(agent)
        return [c for c in self.candidates.values() if c.agent == agent and c.epoch == epoch]

    def candidates_of(self, agent: str, *, since_seq: int) -> list[CandidateInfo]:
        return [c for c in self.candidates.values() if c.agent == agent and c.seq > since_seq]

    def current_candidate(self, agent: str) -> CandidateInfo:
        own = self.own_candidates(agent)
        return own[-1] if own else self.epoch_seed(agent)

    def epoch_turns(self, agent: str) -> list[TurnInfo]:
        epoch = self.epoch(agent)
        return [t for t in self.turns if t.agent == agent and t.epoch == epoch]

    def current_session(self, agent: str) -> str | None:
        """Session to continue in this epoch: the last OK turn's session, unless a turn after it
        failed with an error that does not survive (see RESUMABLE_ERRORS). None = start fresh.

        Transient failures (an API error, a timeout, a harness-side failure) keep the session,
        so a run without forgetting never forgets by accident; a process failure starts over,
        and so do MAX_RESUMED_ERRORS consecutive transient failures (a poisoned session, e.g. a
        history the API keeps rejecting, must not stall the agent until it is stopped).
        """
        for errors, turn in enumerate(reversed(self.epoch_turns(agent)), start=1):
            if turn.ok:
                return turn.session_id
            if turn.error_kind not in RESUMABLE_ERRORS or errors >= MAX_RESUMED_ERRORS:
                return None
        return None

    def session_turn_list(self, agent: str) -> list[TurnInfo]:
        """Completed (non-error) turns of the agent's current session, oldest first."""
        session = self.current_session(agent)
        if session is None:
            return []
        return [t for t in self.epoch_turns(agent) if t.session_id == session and t.ok]

    def session_turns(self, agent: str) -> int:
        return len(self.session_turn_list(agent))

    def forget_reasons(self, scope: str) -> list[str]:
        return [f.reason for f in self.forgets if f.scope == scope]

    def team_round_start_seq(self) -> int:
        """Seq of the last round boundary (run start or team forget)."""
        starts = [r.seq for r in self.epoch_history if r.scope in ("start", "team")]
        return starts[-1] if starts else -1

    def team_round(self) -> int:
        return sum(1 for r in self.epoch_history if r.scope in ("start", "team"))

    def round_proposals(self) -> list[TurnInfo]:
        start = self.team_round_start_seq()
        return [t for t in self.turns if t.seq > start and t.ok]

    def best_at(self, seq: int) -> BestEntry | None:
        entries = [b for b in self.best_history if b.seq <= seq]
        return entries[-1] if entries else None


def _cand_number(cand_id: str) -> int | None:
    if cand_id.startswith("c") and cand_id[1:].isdigit():
        return int(cand_id[1:])
    return None
