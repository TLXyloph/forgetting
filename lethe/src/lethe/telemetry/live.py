"""One-line progress messages for a running experiment (used by `lethe run`)."""

from __future__ import annotations

import time

from lethe.domain.events import (
    Envelope,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RateLimited,
    RunFinished,
    TurnFinished,
)
from lethe.domain.state import SEED_ID, RunState
from lethe.policy.prompts import format_score


def _turn_of(state: RunState, cand_id: str):
    return next((t for t in reversed(state.turns) if t.cand_id == cand_id), None)


def format_event(envelope: Envelope, state: RunState) -> str | None:
    """A human-readable line for events worth showing live, else None. Call after `apply`."""
    event = envelope.event
    spend = f"${state.spent_usd:.2f}"
    if isinstance(event, JudgeFinished):
        cand = state.candidates.get(event.cand_id)
        if cand is None or event.cand_id == SEED_ID:
            return f"seed {event.cand_id} scored {format_score(event.score)}" + (
                "" if event.valid else " (invalid!)"
            )
        turn = _turn_of(state, event.cand_id)
        where = f"[{cand.agent} r{cand.epoch} t{turn.turn_in_session if turn else '?'}]"
        if not event.valid:
            return f"{where} {event.cand_id} rejected by the judge · {spend}"
        best = state.best()
        tag = " · new best" if best is not None and best.cand_id == event.cand_id else ""
        return (
            f"{where} {event.cand_id} score {format_score(event.score)} "
            f"({cand.lines_changed} lines){tag} · {spend}"
        )
    if isinstance(event, TurnFinished) and event.status != "candidate":
        where = f"[{event.agent} r{event.epoch} t{event.turn_in_session}]"
        if event.status == "error":
            return f"{where} turn failed: {event.error} · {spend}"
        claim = " (claimed a plateau)" if event.plateau_claim else ""
        return f"{where} no change{claim} · {spend}"
    if isinstance(event, ForgetDecided):
        who = "team" if event.scope == "team" else ", ".join(event.agents)
        return f"forget ({who}): {event.reason}"
    if isinstance(event, EpochStarted) and event.scope != "start":
        seeds = sorted(set(event.seeds.values()))
        scores = ", ".join(f"{s} {format_score(state.candidates[s].score)}" for s in seeds)
        if event.scope == "team":
            return f"round {state.team_round()} starts fresh from {scores}"
        return f"{', '.join(event.epochs)} restarted fresh from {scores}"
    if isinstance(event, RateLimited):
        when = (
            time.strftime("%H:%M", time.localtime(event.resets_at)) if event.resets_at else "later"
        )
        return f"rate limited (subscription window); pausing all agents until ~{when}"
    if isinstance(event, RunFinished):
        return (
            f"run finished: {event.reason}; best {event.best_cand_id} = "
            f"{format_score(event.best_score)} · {spend}"
        )
    return None
