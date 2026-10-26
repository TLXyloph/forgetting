"""Tests for research.bandit: the reward, the UCB1 index and arm selection."""

from __future__ import annotations

import math
from typing import Any

import pytest

from research.bandit import (
    arm_stats,
    normalized_improvement,
    reward_of_run,
    select_arm,
    ucb_index,
)

# ---------------------------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "best", "direction", "expected"),
    [
        (100.0, 110.0, "max", 0.1),
        (100.0, 100.0, "max", 0.0),
        (1000.0, 800.0, "min", 0.2),  # lower is better: a drop is an improvement
        (-10.0, -5.0, "max", 0.5),  # relative to |seed|
        (100.0, 1000.0, "max", 1.0),  # clipped above
        (100.0, 1.0, "min", 0.99),
        (100.0, -500.0, "max", -1.0),  # clipped below (cannot happen for a best-so-far)
        (100.0, 500.0, "min", -1.0),
        (0.0, 5.0, "max", 1.0),  # a zero seed would divide by zero: the 1e-9 floor clips it
        (0.0, 0.0, "max", 0.0),
    ],
)
def test_reward_is_the_normalized_improvement_over_the_seed(
    seed: float, best: float, direction: str, expected: float
) -> None:
    assert normalized_improvement(seed, best, direction) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(("seed", "best"), [(None, 5.0), (5.0, None), (None, None)])
def test_a_missing_score_means_no_reward(seed: float | None, best: float | None) -> None:
    assert normalized_improvement(seed, best, "max") == 0.0


def test_reward_of_a_finished_run_comes_from_its_summary(make_summary: Any) -> None:
    assert reward_of_run(make_summary(seed_score=100.0, final_best=120.0)) == pytest.approx(0.2)
    lower = make_summary(direction="min", seed_score=1000.0, final_best=900.0)
    assert reward_of_run(lower) == pytest.approx(0.1)


def test_failed_or_unfinished_runs_earn_zero(make_summary: Any) -> None:
    assert reward_of_run(None) == 0.0
    assert reward_of_run(make_summary(finished=False, final_best=500.0)) == 0.0
    assert reward_of_run(make_summary(seed_score=None, final_best=500.0)) == 0.0
    assert reward_of_run(make_summary(final_best=None)) == 0.0


# ---------------------------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------------------------


def test_ucb_index_formula() -> None:
    # mean + c * sqrt(2 ln N / n)
    assert ucb_index(0.5, 2, 10, 1.0) == pytest.approx(0.5 + math.sqrt(math.log(10)))
    assert ucb_index(0.5, 2, 10, 2.0) == pytest.approx(0.5 + 2 * math.sqrt(math.log(10)))
    assert ucb_index(0.5, 2, 10, 0.0) == 0.5


def test_ucb_index_of_an_unpulled_arm_is_infinite() -> None:
    assert ucb_index(0.0, 0, 5, 1.0) == math.inf


def test_ucb_index_with_a_single_pull_in_total_has_no_bonus() -> None:
    assert ucb_index(0.3, 1, 1, 1.0) == pytest.approx(0.3)  # ln 1 = 0


def test_each_arm_is_pulled_once_before_ucb_applies() -> None:
    names = ["a", "b", "c"]
    assert select_arm(names, {"a": [], "b": [], "c": []}, 1.0) == ("a", math.inf)
    assert select_arm(names, {"a": [0.9], "b": [], "c": []}, 1.0) == ("b", math.inf)
    assert select_arm(names, {"a": [0.9], "b": [0.1], "c": []}, 1.0) == ("c", math.inf)


def test_selection_after_the_first_round_prefers_the_higher_index() -> None:
    names = ["a", "b", "c"]
    rewards = {"a": [0.9], "b": [0.1], "c": [0.5]}
    arm, index = select_arm(names, rewards, 1.0)
    assert arm == "a"
    assert index == pytest.approx(0.9 + math.sqrt(2 * math.log(3)))


def test_exploration_bonus_can_beat_a_better_mean() -> None:
    rewards = {"a": [0.5] * 10, "b": [0.45]}
    assert select_arm(["a", "b"], rewards, 1.0)[0] == "b"
    assert select_arm(["a", "b"], rewards, 0.0)[0] == "a"  # c = 0 is greedy


def test_ties_go_to_the_arm_listed_first() -> None:
    rewards = {"x": [0.4], "y": [0.4], "z": [0.4]}
    assert select_arm(["x", "y", "z"], rewards, 1.0)[0] == "x"
    assert select_arm(["z", "y", "x"], rewards, 1.0)[0] == "z"


def test_a_single_arm_experiment_always_picks_it() -> None:
    assert select_arm(["only"], {"only": [0.2, 0.3]}, 1.0)[0] == "only"


def test_arm_stats_summarize_rewards_and_index() -> None:
    stats = arm_stats(["a", "b", "c"], {"a": [0.5, 0.3], "b": [0.1], "c": []}, 1.0)

    assert [s.arm for s in stats] == ["a", "b", "c"]
    assert [s.pulls for s in stats] == [2, 1, 0]
    assert stats[0].mean_reward == pytest.approx(0.4)
    assert stats[2].mean_reward is None
    assert stats[2].index == math.inf
    assert stats[1].index == pytest.approx(0.1 + math.sqrt(2 * math.log(3)))
