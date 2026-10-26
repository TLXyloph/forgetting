"""The event log schema. Every state change in a run is one of these events.

Events are appended to `events.jsonl` (one JSON object per line) wrapped in an `Envelope`
that adds a monotonic `seq` and a wall-clock `ts`. `RunState` is a pure fold over envelopes,
so a run can be replayed, analyzed, or forked from any prefix of its log.

Forgetting never deletes anything here: it starts a new epoch (`EpochStarted`) and the
reducer stops showing older material to the agents.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from lethe.domain.types import Direction

TurnStatus = Literal["candidate", "empty", "error"]
ForgetScope = Literal["team", "agent"]
EpochScope = Literal["start", "team", "agent"]


@dataclass(frozen=True)
class RunStarted:
    kind: ClassVar[str] = "run_started"
    run_id: str
    task_name: str
    agents: tuple[str, ...]
    direction: Direction
    config: Mapping[str, Any]
    lethe_version: str


@dataclass(frozen=True)
class CandidateCommitted:
    """A turn produced a new candidate (its editable files changed). `agent == "seed"` for c0000."""

    kind: ClassVar[str] = "candidate_committed"
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


@dataclass(frozen=True)
class JudgeFinished:
    kind: ClassVar[str] = "judge_finished"
    cand_id: str
    valid: bool
    score: float | None
    feedback: str
    metrics: Mapping[str, Any]
    cases: Mapping[str, float] | None
    duration_s: float
    cache_hit: bool
    error: str | None


@dataclass(frozen=True)
class TurnFinished:
    """One agent turn (= one proposal) ended. Emitted after the snapshot, before judging."""

    kind: ClassVar[str] = "turn_finished"
    agent: str
    epoch: int
    session_id: str
    turn_in_session: int
    status: TurnStatus
    cand_id: str | None
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    thinking_tokens: int
    files_read: int
    edit_calls: int
    lines_changed: int
    plateau_claim: bool
    duration_s: float
    error: str | None = None
    five_hour_utilization: float | None = None
    seven_day_utilization: float | None = None


@dataclass(frozen=True)
class ForgetDecided:
    """A trigger fired; the forget happens at `EpochStarted`, after in-flight turns drain."""

    kind: ClassVar[str] = "forget_decided"
    scope: ForgetScope
    agents: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class EpochStarted:
    """Agents in `epochs` start a new epoch: fresh workspace, fresh session, given seed."""

    kind: ClassVar[str] = "epoch_started"
    scope: EpochScope
    seeds: Mapping[str, str]  # agent -> cand_id it starts from
    epochs: Mapping[str, int]  # agent -> new epoch number (1-based)
    reason: str


@dataclass(frozen=True)
class RateLimited:
    """The subscription window is exhausted (or nearly); agents pause until `resets_at`."""

    kind: ClassVar[str] = "rate_limited"
    agent: str
    resets_at: float | None
    message: str


@dataclass(frozen=True)
class RunFinished:
    kind: ClassVar[str] = "run_finished"
    reason: str
    best_cand_id: str | None
    best_score: float | None


Event = (
    RunStarted
    | CandidateCommitted
    | JudgeFinished
    | TurnFinished
    | ForgetDecided
    | EpochStarted
    | RateLimited
    | RunFinished
)

EVENT_TYPES: dict[str, type] = {
    cls.kind: cls
    for cls in (
        RunStarted,
        CandidateCommitted,
        JudgeFinished,
        TurnFinished,
        ForgetDecided,
        EpochStarted,
        RateLimited,
        RunFinished,
    )
}


@dataclass(frozen=True)
class Envelope:
    seq: int
    ts: float
    event: Event = field()


def encode(envelope: Envelope) -> str:
    """Serialize an envelope to one JSON line (no trailing newline)."""
    body = dataclasses.asdict(envelope.event)
    record = {"seq": envelope.seq, "ts": envelope.ts, "kind": envelope.event.kind, **body}
    return json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))


def decode(line: str) -> Envelope:
    """Parse one JSON line produced by `encode`. Raises ValueError on unknown kinds or fields."""
    record = json.loads(line)
    kind = record.pop("kind")
    seq = record.pop("seq")
    ts = record.pop("ts")
    cls = EVENT_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"unknown event kind: {kind!r}")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in record:
            continue
        value = record[f.name]
        if isinstance(value, list) and str(f.type).startswith("tuple"):
            value = tuple(value)
        kwargs[f.name] = value
    unknown = set(record) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown fields for {kind}: {sorted(unknown)}")
    return Envelope(seq=seq, ts=ts, event=cls(**kwargs))
