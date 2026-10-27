from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeJudge, FakeTime, Reply, build  # noqa: E402

from lethe.domain.events import (  # noqa: E402
    CandidateCommitted,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RateLimited,
    RunFinished,
    TurnFinished,
)
from lethe.domain.types import RateLimitInfo  # noqa: E402
from lethe.runtime.budget import Pacer  # noqa: E402
from lethe.runtime.orchestrator import Orchestrator  # noqa: E402


def climbing(agent, index, prompt, session):
    return Reply(value=str(11 + index))


SOLO = {"preset": "solo", "budget": {"usd": 0, "proposals": 3, "primary": "proposals"}}


def test_solo_run_records_seed_turns_and_finish(tmp_path):
    seen = []
    orch, store, backend, _, judge = build(tmp_path, climbing, events=seen, **SOLO)
    state = asyncio.run(orch.run())
    assert store.kinds()[:4] == [
        "run_started",
        "candidate_committed",
        "judge_finished",
        "epoch_started",
    ]
    assert [e.cand_id for e in store.of(CandidateCommitted)] == ["c0000", "c0001", "c0002", "c0003"]
    assert judge.calls == ["c0000", "c0001", "c0002", "c0003"]
    assert state.best().cand_id == "c0003" and state.best().score == 13.0
    finished = store.of(RunFinished)[0]
    assert "proposal budget" in finished.reason and finished.best_cand_id == "c0003"
    assert seen == store.kinds()


def test_sessions_continue_and_prompts_shrink_after_first_turn(tmp_path):
    orch, _, backend, _, _ = build(tmp_path, climbing, **SOLO)
    asyncio.run(orch.run())
    sessions = [c["session"] for c in backend.calls]
    assert [s.resume for s in sessions] == [False, True, True]
    assert len({s.session_id for s in sessions}) == 1
    assert backend.calls[0]["prompt"].startswith("# Task")
    assert backend.calls[1]["prompt"].startswith("# Feedbacks and Updates")
    assert str(backend.calls[0]["transcript"]).endswith("transcripts/a1/e1/sess-1-t1.jsonl")


def test_turn_without_judged_change_is_empty_and_not_judged(tmp_path):
    orch, store, _, _, judge = build(
        tmp_path, lambda *a: Reply(value=None, text="no change"), **SOLO
    )
    asyncio.run(orch.run())
    turns = store.of(TurnFinished)
    assert [t.status for t in turns] == ["empty", "empty", "empty"]
    assert judge.calls == ["c0000"]
    assert all(t.cand_id is None and t.lines_changed == 0 for t in turns)


def test_plateau_claims_are_recorded(tmp_path):
    reply = Reply(value=None, text="I've reached the practical limit of this machine.")
    orch, store, _, _, _ = build(tmp_path, lambda *a: reply, **SOLO)
    asyncio.run(orch.run())
    assert all(t.plateau_claim for t in store.of(TurnFinished))


def test_team_forget_starts_fresh_epochs_from_the_best(tmp_path):
    overrides = {
        "team": {"agents": 2},
        "forget": {"when": [{"type": "budget_fractions", "fractions": [0.5]}]},
        "budget": {"usd": 0, "proposals": 8, "primary": "proposals"},
    }
    orch, store, backend, materializer, _ = build(tmp_path, climbing, **overrides)
    state = asyncio.run(orch.run())
    forgets = store.of(ForgetDecided)
    assert len(forgets) == 1 and forgets[0].scope == "team"
    team_epochs = [e for e in store.of(EpochStarted) if e.scope == "team"]
    assert len(team_epochs) == 1
    epoch = team_epochs[0]
    assert set(epoch.seeds.values()) == {_best_at_forget(store)}
    assert epoch.epochs == {"a1": 2, "a2": 2}
    assert [(a, e) for a, e, *_ in materializer.prepared] == [
        ("a1", 1),
        ("a2", 1),
        ("a1", 2),
        ("a2", 2),
    ]
    for agent in ("a1", "a2"):
        calls = [c for c in backend.calls if c["agent"] == agent]
        epoch2 = [c for c in calls if "/e2" in str(c["workspace"])]
        assert epoch2 and epoch2[0]["session"].resume is False
        assert epoch2[0]["prompt"].startswith("# Task") and "Round 2." in epoch2[0]["prompt"]
    assert state.proposals == 8


def _best_at_forget(store):
    best_id, best = "c0000", float("-inf")
    for event in store.events:
        if isinstance(event, JudgeFinished) and event.valid and event.score > best:
            best_id, best = event.cand_id, event.score
        if isinstance(event, EpochStarted) and event.scope == "team":
            return best_id
    raise AssertionError("no team epoch")


def test_stalled_agent_is_reset_alone(tmp_path):
    def script(agent, index, prompt, session):
        return (
            Reply(value=None, text="nothing to do")
            if agent == "a1"
            else Reply(value=str(20 + index))
        )

    overrides = {
        "team": {"agents": 2},
        "forget": {"when": [{"type": "stalled", "n": 2, "of": 2}]},
        "budget": {"usd": 0, "proposals": 8, "primary": "proposals"},
    }
    orch, store, _, _, _ = build(tmp_path, script, **overrides)
    asyncio.run(orch.run())
    agent_epochs = [e for e in store.of(EpochStarted) if e.scope == "agent"]
    assert agent_epochs and all(set(e.epochs) == {"a1"} for e in agent_epochs)


def test_rate_limited_turns_pause_and_are_not_proposals(tmp_path):
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    limited = RateLimitInfo("rejected", 1.0, 0.3, t.now + 600)

    def script(agent, index, prompt, session):
        if index == 0:
            return Reply(value=None, cost=0.0, error_kind="rate_limit", rate_limit=limited)
        return Reply(value=str(11 + index))

    orch, store, _, _, _ = build(tmp_path, script, pacer=pacer, **SOLO)
    state = asyncio.run(orch.run())
    assert len(store.of(RateLimited)) == 1
    assert t.now >= limited.resets_at
    assert state.proposals == 3 and all(t.status != "error" for t in store.of(TurnFinished))


def test_the_default_pacer_takes_both_window_thresholds_from_the_budget(tmp_path):
    budget = {"usd": 1.0, "pause_at_utilization": 0.9, "pause_at_weekly_utilization": 0.8}
    orch, _, _, _, _ = build(tmp_path, lambda *a: Reply(value="1"), budget=budget)
    assert (orch.pacer.pause_at, orch.pacer.weekly_pause_at) == (0.9, 0.8)


def test_repeated_errors_stop_the_agent(tmp_path):
    orch, store, _, _, _ = build(tmp_path, lambda *a: Reply(error_kind="api"), **SOLO)
    state = asyncio.run(orch.run())
    assert [t.status for t in store.of(TurnFinished)] == [
        "error"
    ] * Orchestrator.MAX_CONSECUTIVE_ERRORS
    assert "errors" in store.of(RunFinished)[0].reason
    assert state.proposals == 0


def test_clean_forget_restarts_from_the_original_seed_overlaid_with_the_best(tmp_path):
    overrides = {
        "team": {"agents": 1, "peers": "none"},
        "forget": {
            "carry_scratch": False,
            "when": [{"type": "budget_fractions", "fractions": [0.5]}],
        },
        "budget": {"usd": 0, "proposals": 4, "primary": "proposals"},
    }
    orch, store, _, materializer, _ = build(tmp_path, climbing, **overrides)
    state = asyncio.run(orch.run())
    seed_sha = state.candidates["c0000"].sha
    assert [(e, base) for _, e, _, _, base in materializer.prepared] == [
        (1, seed_sha),
        (2, seed_sha),
    ]
    assert [e.scope for e in store.of(EpochStarted)] == ["start", "team"]


def test_carrying_scratch_copies_the_best_candidate_whole(tmp_path):
    orch, _, _, materializer, _ = build(tmp_path, climbing, **SOLO)
    asyncio.run(orch.run())
    assert [base for *_, base in materializer.prepared] == [None]


def test_usd_budget_is_metered_from_turn_costs(tmp_path):
    overrides = {"preset": "solo", "budget": {"usd": 0.25}}
    orch, store, _, _, _ = build(tmp_path, climbing, **overrides)
    state = asyncio.run(orch.run())
    assert len(store.of(TurnFinished)) == 3  # 0.1 + 0.1 + 0.1 crosses 0.25
    assert "usd budget" in store.of(RunFinished)[0].reason
    assert abs(state.spent_usd - 0.3) < 1e-9


class SlowJudge(FakeJudge):
    """Judging takes a while, so other agents check the budget while a turn is being judged."""

    async def evaluate(self, *, cand_id, judged_key, materialize):
        await asyncio.sleep(0.05)
        return await super().evaluate(
            cand_id=cand_id, judged_key=judged_key, materialize=materialize
        )


def test_turns_being_judged_are_not_counted_twice_against_the_budget(tmp_path):
    overrides = {
        "team": {"agents": 2},
        "budget": {"usd": 0, "proposals": 6, "primary": "proposals"},
    }
    orch, store, _, _, _ = build(tmp_path, climbing, **overrides)
    orch.judge = SlowJudge()
    state = asyncio.run(orch.run())
    assert state.proposals == 6
    assert store.of(RunFinished)[0].reason == "proposal budget exhausted (6 of 6)"
