from __future__ import annotations

import dataclasses

import pytest

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RunStarted,
    TurnFinished,
)
from lethe.policy.triggers import (
    Anchored,
    BudgetFractions,
    Plateau,
    Stalled,
    improved_since,
    trigger_from_dict,
)


class Run:
    """Tiny DSL for building run histories."""

    def __init__(self, feeder, agents=("a1", "a2"), direction="max"):
        self.state, self.feed = feeder
        self.agents = agents
        self.n = 0
        self.turn_no = {a: 0 for a in agents}
        self.feed(RunStarted("r", "t", tuple(agents), direction, {}, "0"))
        self.feed(CandidateCommitted("c0000", "s0", None, "seed", 0, None, "k0", 0, 0, 0, "seed"))
        self.feed(JudgeFinished("c0000", True, 100.0, "", {}, None, 0.0, False, None))
        self.feed(
            EpochStarted("start", {a: "c0000" for a in agents}, {a: 1 for a in agents}, "start")
        )

    def turn(self, agent, *, lines=10, score=None, status=None, claim=False, session="s", tools=2):
        self.turn_no[agent] += 1
        epoch = self.state.epoch(agent)
        cid = None
        if status is None:
            status = "candidate" if score is not None else "empty"
        if status == "candidate":
            self.n += 1
            cid = f"c{self.n:04d}"
            self.feed(
                CandidateCommitted(
                    cid, cid, None, agent, epoch, f"{session}{agent}{epoch}", cid, lines, 0, 1, ""
                )
            )
        self.feed(
            TurnFinished(
                agent,
                epoch,
                f"{session}{agent}{epoch}",
                self.turn_no[agent],
                status,
                cid,
                0.1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                lines,
                claim,
                1.0,
                tool_calls=tools,
            )
        )
        if cid is not None:
            self.feed(JudgeFinished(cid, True, score, "", {}, None, 0.0, False, None))

    def team_forget(self, reason):
        self.feed(ForgetDecided("team", self.agents, reason))
        best = self.state.best().cand_id
        self.feed(
            EpochStarted(
                "team",
                {a: best for a in self.agents},
                {a: self.state.epoch(a) + 1 for a in self.agents},
                reason,
            )
        )
        self.turn_no = {a: 0 for a in self.agents}


def test_budget_fractions_fire_once_per_crossing(feeder):
    run = Run(feeder)
    trig = BudgetFractions((0.25, 0.5, 0.75))
    assert trig.check(run.state, spend_fraction=0.1) is None
    decision = trig.check(run.state, spend_fraction=0.3)
    assert decision.scope == "team" and decision.agents == ("a1", "a2")
    assert decision.reason.startswith("budget_fractions@1")
    run.team_forget(decision.reason)
    assert trig.check(run.state, spend_fraction=0.4) is None


def test_budget_fractions_consume_all_crossed_fractions_at_once(feeder):
    run = Run(feeder)
    trig = BudgetFractions((0.25, 0.5, 0.75))
    decision = trig.check(run.state, spend_fraction=0.6)
    assert decision.reason.startswith("budget_fractions@2")
    run.team_forget(decision.reason)
    assert trig.check(run.state, spend_fraction=0.7) is None
    assert trig.check(run.state, spend_fraction=0.8).reason.startswith("budget_fractions@3")


def test_plateau_fires_after_window_without_gain(feeder):
    run = Run(feeder)
    trig = Plateau(window=4, min_gain=0.0, min_proposals=4)
    run.turn("a1", score=110.0)
    for _ in range(3):
        run.turn("a1", score=105.0)
    assert trig.check(run.state, spend_fraction=0.5) is None  # improvement inside the window
    run.turn("a2", score=101.0)
    assert trig.check(run.state, spend_fraction=0.5).reason.startswith("plateau")


def test_plateau_respects_min_gain_and_min_proposals(feeder):
    run = Run(feeder)
    trig = Plateau(window=3, min_gain=0.05, min_proposals=6)
    for score in (100.5, 101.0, 101.2):
        run.turn("a1", score=score)
    assert trig.check(run.state, spend_fraction=0.5) is None  # fewer than min_proposals
    for score in (101.3, 101.4, 101.5):
        run.turn("a2", score=score)
    assert trig.check(run.state, spend_fraction=0.5) is not None  # gains below 5% don't count


def test_anchored_fires_when_quorum_of_sessions_narrow(feeder):
    run = Run(feeder)
    trig = Anchored(ratio=0.3, warmup=2, window=2, quorum=0.5, require_stall=False, min_proposals=0)
    for lines in (100, 120, 20, 10):
        run.turn("a1", lines=lines, score=90.0)
    assert trig.check(run.state, spend_fraction=0.5).reason.startswith("anchored")
    trig_all = Anchored(
        ratio=0.3, warmup=2, window=2, quorum=1.0, require_stall=False, min_proposals=0
    )
    assert trig_all.check(run.state, spend_fraction=0.5) is None  # a2 has no session yet


def test_anchored_requires_stall_by_default(feeder):
    run = Run(feeder)
    trig = Anchored(ratio=0.3, warmup=2, window=2, quorum=0.5, min_proposals=0)
    for lines, score in ((100, 101.0), (100, 102.0), (5, 103.0), (5, 104.0)):
        run.turn("a1", lines=lines, score=score)
    assert trig.check(run.state, spend_fraction=0.5) is None  # still improving


def test_anchored_counts_empty_turns_as_zero_lines(feeder):
    run = Run(feeder)
    trig = Anchored(ratio=0.3, warmup=2, window=2, quorum=0.5, min_proposals=0)
    run.turn("a1", lines=80, score=90.0)
    run.turn("a1", lines=60, score=90.0)
    run.turn("a1", lines=0, status="empty")
    run.turn("a1", lines=0, status="empty")
    assert trig.check(run.state, spend_fraction=0.5) is not None


def test_stalled_resets_only_that_agent(feeder):
    run = Run(feeder)
    trig = Stalled(n=2, of=3)
    run.turn("a1", score=90.0)
    run.turn("a1", status="empty", lines=0)
    assert trig.check_agent(run.state, "a1", spend_fraction=0.5) is None
    run.turn("a1", score=91.0, claim=True)
    decision = trig.check_agent(run.state, "a1", spend_fraction=0.5)
    assert decision.scope == "agent" and decision.agents == ("a1",)
    assert trig.check_agent(run.state, "a2", spend_fraction=0.5) is None


def _fires_after_each_turn(run, trig, turns):
    """Feed a1 `turns` (kwargs for `Run.turn`) and record, after each, whether `trig` fires."""
    fired = []
    for kwargs in turns:
        run.turn("a1", **kwargs)
        fired.append(trig.check_agent(run.state, "a1", spend_fraction=0.5) is not None)
    return fired


ACTIVE_EMPTY = {"status": "empty", "lines": 0, "tools": 6}  # explored, has not submitted yet
ACTIVE_CLAIM = {"status": "empty", "lines": 0, "tools": 3, "claim": True}  # tested, then gave up
PASSIVE_EMPTY = {"status": "empty", "lines": 0, "tools": 0}  # no tool call at all
PASSIVE_CLAIM = {"status": "empty", "lines": 0, "tools": 0, "claim": True}
SUBMITTED = {"score": 90.0}


def test_stalled_passive_only_never_fires_while_the_agent_keeps_acting(feeder):
    trig = Stalled(n=3, of=5, passive_only=True)
    fired = _fires_after_each_turn(Run(feeder), trig, [ACTIVE_EMPTY, ACTIVE_CLAIM] * 6)
    assert not any(fired)


def test_stalled_passive_only_fires_once_the_agent_stops_acting(feeder):
    run = Run(feeder)
    trig = Stalled(n=3, of=5, passive_only=True)
    fired = _fires_after_each_turn(
        run, trig, [SUBMITTED, ACTIVE_EMPTY, PASSIVE_EMPTY, PASSIVE_CLAIM, PASSIVE_EMPTY]
    )
    assert fired == [False, False, False, False, True]
    decision = trig.check_agent(run.state, "a1", spend_fraction=0.5)
    assert decision.scope == "agent" and decision.agents == ("a1",)
    assert decision.reason == (
        "stalled: 3 of the last 5 turns were passive (empty or claimed a plateau, "
        "with no tool call)"
    )


def test_stalled_passive_only_counts_passive_turns_among_active_ones(feeder):
    trig = Stalled(n=3, of=5, passive_only=True)
    turns = [PASSIVE_EMPTY, ACTIVE_EMPTY, PASSIVE_CLAIM, ACTIVE_EMPTY, SUBMITTED]  # 2 of 5
    turns += [PASSIVE_EMPTY]  # the window drops a passive turn and gains one: still 2
    turns += [PASSIVE_CLAIM]  # it drops an active turn and gains a passive one: 3 of 5
    fired = _fires_after_each_turn(Run(feeder), trig, turns)
    assert fired == [False] * 6 + [True]


def test_stalled_passive_only_skips_turns_logged_without_a_tool_call_count(feeder):
    trig = Stalled(n=1, of=1, passive_only=True)
    unknown = {**PASSIVE_CLAIM, "tools": None}  # a log written before tool calls were counted
    assert _fires_after_each_turn(Run(feeder), trig, [unknown, unknown]) == [False, False]


def test_stalled_default_still_counts_every_empty_turn(feeder):
    run = Run(feeder)
    trig = Stalled(n=3, of=5)
    assert trig == Stalled(n=3, of=5, passive_only=False)
    turns = [SUBMITTED, ACTIVE_EMPTY, SUBMITTED, ACTIVE_EMPTY, ACTIVE_CLAIM]
    assert _fires_after_each_turn(run, trig, turns) == [False] * 4 + [True]
    assert trig.check_agent(run.state, "a1", spend_fraction=0.5).reason == (
        "stalled: 3 of the last 5 turns were empty or claimed a plateau"
    )


def test_stalled_passive_only_round_trips_through_its_config_table():
    spec = {"type": "stalled", "n": 3, "of": 5, "passive_only": True}
    trig = trigger_from_dict(spec)
    assert trig == Stalled(n=3, of=5, passive_only=True)
    assert {"type": "stalled", **dataclasses.asdict(trig)} == spec
    assert trigger_from_dict({"type": "stalled"}).passive_only is False
    assert trigger_from_dict({**spec, "passive_only": False}) == Stalled(n=3, of=5)


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "stalled", "passive_only": 1},
        {"type": "stalled", "passive_only": "yes"},
        {"type": "stalled", "n": True},
        {"type": "anchored", "require_stall": 0},
    ],
)
def test_trigger_flags_take_booleans_and_counts_take_numbers(spec):
    with pytest.raises(ValueError, match="invalid value"):
        trigger_from_dict(spec)


def test_passive_only_belongs_to_stalled_alone():
    with pytest.raises(ValueError, match="unknown parameter"):
        trigger_from_dict({"type": "plateau", "passive_only": True})


def test_trigger_from_dict_builds_and_validates():
    assert trigger_from_dict({"type": "budget_fractions", "fractions": [0.5]}) == BudgetFractions(
        (0.5,)
    )
    assert isinstance(trigger_from_dict({"type": "anchored"}), Anchored)
    with pytest.raises(ValueError, match="unknown trigger type"):
        trigger_from_dict({"type": "nope"})
    with pytest.raises(ValueError, match="unknown parameter"):
        trigger_from_dict({"type": "plateau", "windw": 3})
    with pytest.raises(ValueError, match="fractions"):
        trigger_from_dict({"type": "budget_fractions", "fractions": [1.5]})


def test_improved_since_measures_every_gain_from_the_window_start(feeder):
    run = Run(feeder)  # the seed scores 100
    window_start = run.state.last_seq
    run.turn("a1", score=100.6)
    run.turn("a1", score=101.2)  # +0.6% per step, but +1.2% over the window
    assert improved_since(run.state, window_start, 0.01) is True
    assert improved_since(run.state, window_start, 0.02) is False
