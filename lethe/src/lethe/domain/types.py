"""Value types shared across lethe's bounded contexts.

Everything here is an immutable value. Contexts (judge, workspace, agents, policy, runtime)
exchange these types instead of reaching into each other's internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["max", "min"]
TurnErrorKind = Literal["rate_limit", "timeout", "process", "api", "billing"]

# Tool names that modify files, as reported by Claude Code's stream-json output.
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


def is_better(a: float, b: float, direction: Direction) -> bool:
    """True when score `a` strictly improves on score `b` under `direction`."""
    return a > b if direction == "max" else a < b


def format_score(score: float | None) -> str:
    """A score as agents and people see it: compact, but exact to 10 significant digits."""
    if score is None:
        return "n/a"
    return f"{score:.10g}"


@dataclass(frozen=True)
class JudgeResult:
    """Outcome of judging one candidate. `score` is None when the candidate is invalid."""

    valid: bool
    score: float | None
    feedback: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    cases: Mapping[str, float] | None = None
    duration_s: float = 0.0
    error: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True)
class Usage:
    """Token usage of one agent turn (one `claude -p` invocation)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0


@dataclass(frozen=True)
class RateLimitInfo:
    """Subscription meter reported by Claude Code (`rate_limit_event`).

    Utilizations are fractions in [0, 1] of the 5-hour and 7-day windows; `resets_at` is an
    epoch timestamp (seconds) for the window named by the event, and `five_hour_resets_at` the
    reset of the 5-hour window itself (None when the event does not report it).
    """

    status: str
    five_hour_utilization: float | None = None
    seven_day_utilization: float | None = None
    resets_at: float | None = None
    five_hour_resets_at: float | None = None


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation made by the agent during a turn."""

    name: str
    path: str | None = None


@dataclass(frozen=True)
class SessionChoice:
    """Which agent session a turn runs in.

    `resume=False` starts a new session whose id is `session_id`;
    `resume=True` continues the existing session `session_id`.
    """

    session_id: str
    resume: bool


@dataclass(frozen=True)
class TurnResult:
    """Everything the harness learns from one agent turn."""

    session_id: str
    ok: bool
    final_text: str
    cost_usd: float
    usage: Usage
    tool_calls: tuple[ToolCall, ...]
    duration_s: float
    error_kind: TurnErrorKind | None = None
    error: str | None = None
    rate_limit: RateLimitInfo | None = None
    num_api_turns: int = 0

    @property
    def files_read(self) -> int:
        """Number of distinct files the agent opened with the Read tool."""
        return len({c.path for c in self.tool_calls if c.name == "Read" and c.path})

    @property
    def edit_calls(self) -> int:
        """Number of file-modifying tool calls."""
        return sum(1 for c in self.tool_calls if c.name in EDIT_TOOLS)


@dataclass(frozen=True)
class CandidateCard:
    """What an agent is shown about one candidate inside `.context/`."""

    cand_id: str
    parent_id: str | None
    score: float | None
    valid: bool | None  # None: not judged yet
    lines_changed: int
    summary: str
    diff: str  # unified diff against the parent, editable files only


@dataclass(frozen=True)
class PeerBranch:
    """One peer agent as seen by another agent (branch numbers are 1-based)."""

    branch: int
    best: CandidateCard | None
    best_files: Mapping[str, bytes]
    pool: tuple[CandidateCard, ...]


@dataclass(frozen=True)
class ContextBundle:
    """Everything rendered into an agent's workspace before a turn."""

    task_md: str
    current: CandidateCard
    feedback: str
    metrics: Mapping[str, Any]
    lineage: tuple[CandidateCard, ...]  # own candidates this epoch, newest first
    peers: tuple[PeerBranch, ...]
