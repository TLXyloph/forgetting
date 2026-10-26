from __future__ import annotations

from lethe.domain.events import CandidateCommitted, EpochStarted, JudgeFinished, RunStarted
from lethe.policy.seeding import BestSeed, HedgedSeed, seed_policy_from_name

AGENTS = ("a1", "a2", "a3", "a4")


def setup(feed, direction="max", seed_valid=True):
    feed(RunStarted("r", "t", AGENTS, direction, {}, "0"))
    feed(CandidateCommitted("c0000", "s0", None, "seed", 0, None, "k0", 0, 0, 0, ""))
    feed(
        JudgeFinished(
            "c0000", seed_valid, 50.0 if seed_valid else None, "", {}, None, 0, False, None
        )
    )
    feed(EpochStarted("start", {a: "c0000" for a in AGENTS}, {a: 1 for a in AGENTS}, "start"))


def add(feed, cid, score, cases=None, agent="a1"):
    feed(CandidateCommitted(cid, cid, "c0000", agent, 1, "s", cid, 1, 0, 1, ""))
    feed(JudgeFinished(cid, True, score, "", {}, cases, 0, False, None))


def test_best_seed_gives_everyone_the_best(feeder):
    state, feed = feeder
    setup(feed)
    add(feed, "c0001", 100.0)
    assert BestSeed().seeds(state, AGENTS) == {a: "c0001" for a in AGENTS}


def test_best_seed_falls_back_to_task_seed_when_nothing_is_valid(feeder):
    state, feed = feeder
    setup(feed, seed_valid=False)
    assert BestSeed().seeds(state, ("a1",)) == {"a1": "c0000"}


def test_hedged_seed_sends_half_the_team_to_complementary_elites(feeder):
    state, feed = feeder
    setup(feed)
    add(feed, "c0001", 100.0, {"a": 10, "b": 10, "c": 10})
    add(feed, "c0002", 99.0, {"a": 12, "b": 5, "c": 9})
    add(feed, "c0003", 99.5, {"a": 9, "b": 11, "c": 10})
    add(feed, "c0004", 80.0, {"a": 20, "b": 20, "c": 0})  # beyond epsilon
    seeds = HedgedSeed(elite_fraction=0.5, epsilon=0.02).seeds(state, AGENTS)
    assert seeds == {"a1": "c0001", "a2": "c0001", "a3": "c0003", "a4": "c0002"}


def test_hedged_seed_without_cases_is_best_seed(feeder):
    state, feed = feeder
    setup(feed)
    add(feed, "c0001", 100.0)
    add(feed, "c0002", 99.9)
    assert HedgedSeed().seeds(state, AGENTS) == {a: "c0001" for a in AGENTS}


def test_hedged_seed_handles_min_direction(feeder):
    state, feed = feeder
    setup(feed, direction="min")
    add(feed, "c0001", 40.0, {"x": 20, "y": 20})
    add(feed, "c0002", 40.5, {"x": 10, "y": 30.5})  # 1.25% worse overall, better on x
    seeds = HedgedSeed(epsilon=0.02).seeds(state, ("a1", "a2"))
    assert seeds == {"a1": "c0001", "a2": "c0002"}


def test_seed_policy_from_name():
    assert isinstance(seed_policy_from_name("best"), BestSeed)
    assert isinstance(seed_policy_from_name("hedged"), HedgedSeed)
