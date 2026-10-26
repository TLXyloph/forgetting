"""The reducer: decides, before every turn, what an agent sees and which session it continues.

As in the paper, every configuration is a setting of the reducer: who an agent's peers are
(`TeamPolicy.peers`), whether sessions continue (`TeamPolicy.session`), and what forgetting
hides. Forgetting itself is just a new epoch: the agent's horizon moves forward, so its old
session and everything created before the horizon disappear from its view (never from the log).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from lethe.domain.state import CandidateInfo, RunState
from lethe.domain.types import CandidateCard, ContextBundle, PeerBranch, SessionChoice, is_better
from lethe.policy.prompts import (
    PeerStatus,
    continue_prompt,
    first_turn_prompt,
    status_block,
    task_md,
)
from lethe.workspace.paths import PathFilter

MAX_CARDS = 30
SUMMARY_CHARS = 300

Peers = Literal["all", "none", "ring"]
SessionMode = Literal["continue", "fresh"]


@dataclass(frozen=True)
class TeamPolicy:
    agents: tuple[str, ...]
    peers: Peers
    session: SessionMode


@dataclass(frozen=True)
class TurnPlan:
    agent: str
    epoch: int
    session: SessionChoice
    turn_in_session: int
    prompt: str
    bundle: ContextBundle
    current: CandidateInfo


class ArchiveReader(Protocol):
    def files(self, sha: str) -> dict[str, str]: ...

    def read_file(self, sha: str, path: str) -> bytes: ...

    def diff(self, base_sha: str | None, sha: str, *, judged_only: bool = True) -> str: ...


def peers_of(agent: str, team: TeamPolicy) -> list[str]:
    if team.peers == "none" or len(team.agents) < 2:
        return []
    if team.peers == "ring":
        index = team.agents.index(agent)
        return [team.agents[(index + 1) % len(team.agents)]]
    return [a for a in team.agents if a != agent]


def plan_turn(
    state: RunState,
    agent: str,
    *,
    team: TeamPolicy,
    task_description: str,
    archive: ArchiveReader,
    judged: PathFilter,
    new_session_id: Callable[[], str],
) -> TurnPlan:
    current = state.current_candidate(agent)
    continuing = state.current_session(agent) if team.session == "continue" else None
    if continuing is None:
        session, turn_no = SessionChoice(new_session_id(), resume=False), 1
    else:
        session, turn_no = SessionChoice(continuing, resume=True), state.session_turns(agent) + 1

    horizon = state.epoch_start_seq(agent)
    own = state.own_candidates(agent)
    lineage = tuple(_card(state, archive, c) for c in reversed(own[-MAX_CARDS:]))

    branches: list[PeerBranch] = []
    statuses: list[PeerStatus] = []
    for peer in peers_of(agent, team):
        number = team.agents.index(peer) + 1
        pool = state.candidates_of(peer, since_seq=horizon)
        best = _best_of(state, [state.epoch_seed(peer), *pool])
        best_files = _judged_files(archive, judged, best.sha) if best is not None else {}
        branches.append(
            PeerBranch(
                branch=number,
                best=_card(state, archive, best, with_diff=False) if best is not None else None,
                best_files=best_files,
                pool=tuple(_card(state, archive, c) for c in reversed(pool[-MAX_CARDS:])),
            )
        )
        statuses.append(
            PeerStatus(number, best.cand_id if best else None, best.score if best else None)
        )

    status = status_block(
        round_no=state.epoch(agent),
        current_id=current.cand_id,
        current_score=current.score,
        current_valid=current.valid,
        direction=state.direction,
        peers=statuses,
    )
    if session.resume:
        prompt = continue_prompt(status=status)
    else:
        prompt = first_turn_prompt(task_description=task_description, status=status)
    bundle = ContextBundle(
        task_md=task_md(task_description=task_description),
        current=_card(state, archive, current, with_diff=False),
        feedback=current.feedback,
        metrics=dict(current.metrics),
        lineage=lineage,
        peers=tuple(branches),
    )
    return TurnPlan(agent, state.epoch(agent), session, turn_no, prompt, bundle, current)


def _best_of(state: RunState, cands: Sequence[CandidateInfo]) -> CandidateInfo | None:
    best: CandidateInfo | None = None
    for cand in cands:
        if not cand.valid or cand.score is None:
            continue
        if best is None or is_better(cand.score, best.score, state.direction):  # type: ignore[arg-type]
            best = cand
    return best


def _card(
    state: RunState, archive: ArchiveReader, cand: CandidateInfo, *, with_diff: bool = True
) -> CandidateCard:
    parent = state.candidates.get(cand.parent_id) if cand.parent_id else None
    diff = archive.diff(parent.sha, cand.sha) if (with_diff and parent is not None) else ""
    return CandidateCard(
        cand_id=cand.cand_id,
        parent_id=cand.parent_id,
        score=cand.score,
        valid=cand.valid if cand.judged else None,
        lines_changed=cand.lines_changed,
        summary=cand.summary[:SUMMARY_CHARS],
        diff=diff,
    )


def _judged_files(archive: ArchiveReader, judged: PathFilter, sha: str) -> dict[str, bytes]:
    return {path: archive.read_file(sha, path) for path in archive.files(sha) if judged(path)}
