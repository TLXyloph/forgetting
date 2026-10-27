"""How a run ends: every way out writes `run_finished` with a status, and nothing starts after.

Covers the fatal billing stop, crashes and interruptions, a misbehaving progress hook, forgets
that would only start empty epochs once the budget is spent, in-flight reservations, and a team
forget that becomes pending while an agent waits out a rate-limit pause.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeTime, Reply, build  # noqa: E402

from lethe.domain.events import (  # noqa: E402
    EpochStarted,
    ForgetDecided,
    RunFinished,
    TurnFinished,
)
from lethe.domain.types import RateLimitInfo, TurnResult, Usage  # noqa: E402
from lethe.runtime.budget import Pacer  # noqa: E402

REFUSED = (
    "safe mode refused API-key billing (apiKeySource=ANTHROPIC_API_KEY); "
    "set agent.allow_api_billing = true to allow it"
)


def climbing(agent, index, prompt, session):
    return Reply(value=str(11 + index))


def proposals(n, **extra):
    return {"budget": {"usd": 0, "proposals": n, "primary": "proposals"}, **extra}


# --- statuses -----------------------------------------------------------------------------------


def test_a_run_that_spends_its_budget_is_completed(tmp_path):
    orch, store, *_ = build(tmp_path, climbing, preset="solo", **proposals(2))
    asyncio.run(orch.run())
    (finished,) = store.of(RunFinished)
    assert finished.status == "completed" and "proposal budget" in finished.reason


def test_repeated_errors_stop_the_run(tmp_path):
    orch, store, *_ = build(tmp_path, lambda *a: Reply(error_kind="api"), preset="solo")
    asyncio.run(orch.run())
    (finished,) = store.of(RunFinished)
    assert (finished.status, finished.reason) == (
        "stopped",
        "all agents stopped after repeated errors",
    )


class BillingBackend:
    """a2 reports API-key billing at once; a1's turn is still running and must be killed."""

    def __init__(self):
        self.cancelled: list[str] = []

    async def run_turn(self, *, agent, workspace, prompt, session, transcript_path):
        if agent == "a1":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled.append(agent)
                raise
        error = REFUSED
        return TurnResult(session.session_id, False, "", 0.2, Usage(), (), 0.01, "billing", error)


def test_billing_stops_the_whole_run_at_once(tmp_path):
    backend = BillingBackend()
    orch, store, *_ = build(tmp_path, None, backend=backend, team={"agents": 2})

    state = asyncio.run(asyncio.wait_for(orch.run(), 10))

    (finished,) = store.of(RunFinished)
    assert (finished.status, finished.reason) == ("stopped", REFUSED)
    assert backend.cancelled == ["a1"]
    (turn,) = store.of(TurnFinished)  # the refused turn is logged with what it cost
    assert (turn.agent, turn.status, turn.error_kind, turn.cost_usd) == (
        "a2",
        "error",
        "billing",
        0.2,
    )
    assert store.events[-1] is finished and state.finished is finished


def test_a_crash_is_logged_before_it_propagates_and_nothing_runs_after(tmp_path):
    def script(agent, index, prompt, session):
        if agent == "a1" and index == 1:
            raise RuntimeError("boom")
        return Reply(value=str(11 + index), delay=0.05 if agent == "a2" else 0.0)

    orch, store, *_ = build(tmp_path, script, team={"agents": 2}, **proposals(40))

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(orch.run())

    finished = store.events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "crashed"
    assert finished.reason == "crashed: RuntimeError: boom"


def test_an_interrupted_run_is_logged_as_interrupted(tmp_path):
    orch, store, *_ = build(tmp_path, lambda *a: Reply(value="12", delay=30), preset="solo")

    async def main():
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    (finished,) = store.of(RunFinished)
    assert (finished.status, finished.reason) == ("interrupted", "interrupted")


def test_a_failing_progress_hook_is_dropped_instead_of_killing_the_run(tmp_path):
    calls = []

    def hook(envelope, state):
        calls.append(envelope.event.kind)
        raise BrokenPipeError("lethe run | head")

    orch, store, *_ = build(tmp_path, climbing, preset="solo", **proposals(3))
    orch._on_event = hook

    state = asyncio.run(orch.run())

    assert calls == ["run_started"] and state.proposals == 3
    assert store.of(RunFinished)[0].status == "completed"


# --- nothing starts once the budget is spent ------------------------------------------------


def test_a_team_forget_decided_near_the_end_starts_no_empty_round(tmp_path):
    def script(agent, index, prompt, session):  # a1 is fast, a2 slow (the verified scenario)
        return Reply(value=str(20 + index), delay=0.01 if agent == "a1" else 0.3)

    overrides = {"forget": {"when": [{"type": "budget_fractions", "fractions": [0.5]}]}}
    orch, store, _, materializer, _ = build(
        tmp_path, script, team={"agents": 2}, **proposals(3, **overrides)
    )

    state = asyncio.run(orch.run())

    assert len(store.of(ForgetDecided)) == 1  # decided, but the budget ran out before it began
    assert [e.scope for e in store.of(EpochStarted)] == ["start"]
    assert sorted((a, e) for a, e, *_ in materializer.prepared) == [("a1", 1), ("a2", 1)]
    assert state.proposals == 3
    assert store.of(RunFinished)[0].reason == "proposal budget exhausted (3 of 3)"


def test_an_agent_reset_only_happens_right_before_a_turn(tmp_path):
    def script(agent, index, prompt, session):  # every turn is empty: a1 is always stalled
        return Reply(value=None, text="nothing", delay=0.01 if agent == "a1" else 0.3)

    overrides = {"forget": {"when": [{"type": "stalled", "n": 1, "of": 1}]}}
    orch, store, *_ = build(tmp_path, script, team={"agents": 2}, **proposals(3, **overrides))

    state = asyncio.run(orch.run())

    for record in state.epoch_history:
        for agent, epoch in record.epochs.items():
            assert any(t.agent == agent and t.epoch == epoch for t in state.turns), record


def test_in_flight_turns_that_fail_give_their_budget_back(tmp_path):
    def script(agent, index, prompt, session):
        if agent == "a2":
            return Reply(error_kind="process", cost=0.0, delay=0.05)
        return Reply(value=str(20 + index), delay=0.01)

    orch, store, *_ = build(tmp_path, script, team={"agents": 2}, **proposals(4))

    state = asyncio.run(orch.run())

    assert state.proposals == 4
    assert store.of(RunFinished)[0].reason == "proposal budget exhausted (4 of 4)"


def test_a_forget_that_became_pending_during_a_pause_is_joined_before_the_next_turn(tmp_path):
    t = FakeTime(tick=0.01)  # each 30 s step of the pause takes 10 ms of real time
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    limited = RateLimitInfo("rejected", 1.0, 0.3, t.now + 600)

    def script(agent, index, prompt, session):
        if agent == "a1" and index == 0:
            return Reply(cost=0.0, error_kind="rate_limit", rate_limit=limited)
        return Reply(value=str(20 + index), delay=0.05 if agent == "a2" else 0.0)

    overrides = {"forget": {"when": [{"type": "budget_fractions", "fractions": [0.5]}]}}
    orch, store, *_ = build(
        tmp_path, script, pacer=pacer, team={"agents": 2}, **proposals(2, **overrides)
    )

    asyncio.run(orch.run())

    assert [e.scope for e in store.of(EpochStarted)] == ["start", "team"]
    a1_turns = [turn.epoch for turn in store.of(TurnFinished) if turn.agent == "a1"]
    assert a1_turns == [2]  # never a turn in the old epoch once the forget was pending
