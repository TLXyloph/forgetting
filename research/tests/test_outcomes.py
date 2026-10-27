"""Tests for research.outcomes: forget outcomes, empty-turn spend and the cost to reach a score."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from research.outcomes import (
    empty_turn_table,
    forget_outcome_table,
    spend_to_reach,
    threshold_table,
)
from research.summary import TurnRecord, summarize_run

Factory = Callable[..., Any]


# ---------------------------------------------------------------------------------------------
# forget outcomes
# ---------------------------------------------------------------------------------------------


def forgetting(
    make_summary: Factory,
    arm: str,
    per_run_round_bests: Sequence[Sequence[float | None]],
    **kw: Any,
) -> list:
    return [
        make_summary(
            arm,
            rep,
            team_forgets=len(bests) - 1,
            rounds=[(None, best) for best in bests],
            final_best=bests[-1],
            **kw,
        )
        for rep, bests in enumerate(per_run_round_bests, 1)
    ]


def test_a_forget_counts_when_the_next_round_ends_on_a_new_best(make_summary: Factory) -> None:
    # forgets: 100->120 yes, 120->120 no (a tie), 120->130 yes; 90->95 yes, 95->95 no
    summaries = forgetting(make_summary, "tf", [[100.0, 120.0, 120.0, 130.0], [90.0, 95.0, 95.0]])

    (row,) = forget_outcome_table(summaries)

    assert (row.arm, row.runs, row.forgets, row.new_bests) == ("tf", 2, 5, 3)
    assert row.new_best_rate == pytest.approx(3 / 5)


def test_forget_outcomes_follow_the_direction_of_the_task(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "tf", [[1231.0, 1145.0, 1145.0]], direction="min")
    (row,) = forget_outcome_table(summaries)
    assert (row.forgets, row.new_bests) == (2, 1)


def test_a_round_that_ends_without_a_valid_score_is_not_a_new_best(make_summary: Factory) -> None:
    (row,) = forget_outcome_table(forgetting(make_summary, "tf", [[None, None]]))
    assert (row.forgets, row.new_bests) == (1, 0)


def test_a_first_valid_score_after_a_forget_is_a_new_best(make_summary: Factory) -> None:
    (row,) = forget_outcome_table(forgetting(make_summary, "tf", [[None, 90.0]]))
    assert (row.forgets, row.new_bests) == (1, 1)


def test_forget_outcomes_list_only_arms_that_forgot(make_summary: Factory) -> None:
    summaries = (
        [make_summary("solo", 1, team_forgets=0)]
        + forgetting(make_summary, "z-arm", [[1.0, 2.0]])
        + forgetting(make_summary, "a-arm", [[1.0, 2.0]])
    )
    assert [r.arm for r in forget_outcome_table(summaries)] == ["z-arm", "a-arm"]
    assert forget_outcome_table([make_summary("solo", 1)]) == []


def test_runs_that_never_forgot_do_not_count_as_forgetting_runs(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "tf", [[100.0, 120.0]]) + [
        make_summary("tf", 2, team_forgets=0, rounds=[(None, 500.0)], final_best=500.0)
    ]
    (row,) = forget_outcome_table(summaries)
    assert (row.runs, row.forgets, row.new_bests) == (1, 1, 1)


def test_forget_outcomes_from_an_event_log(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "tf" / "rep1", agents=("a1", "a2")).start(100.0)
    b.candidate("a1", 110.0)
    b.forget()
    b.candidate("a1", 105.0)  # worse than the best: the round ends where it began
    b.forget()
    b.candidate("a2", 130.0)
    summary = summarize_run(b.finish(), arm="tf", rep=1)

    (row,) = forget_outcome_table([summary])

    assert (row.forgets, row.new_bests) == (2, 1)


# ---------------------------------------------------------------------------------------------
# empty turns and their spend
# ---------------------------------------------------------------------------------------------


def turn(status: str, cost: float) -> TurnRecord:
    return TurnRecord(
        spend=0.0,  # cumulative spend plays no part here
        lines_changed=0 if status != "candidate" else 10,
        output_tokens=100,
        thinking_tokens=50,
        status=status,  # type: ignore[arg-type]
        plateau_claim=False,
        cost=cost,
    )


def test_empty_turns_and_their_share_of_turns_and_spend(make_summary: Factory) -> None:
    turns = [turn("candidate", 1.0), turn("empty", 0.5), turn("empty", 1.5), turn("error", 1.0)]

    (row,) = empty_turn_table([make_summary("solo", turns=turns)])

    assert (row.arm, row.runs, row.turns, row.empty_turns, row.error_turns) == ("solo", 1, 4, 2, 1)
    assert row.empty_share == pytest.approx(2 / 4)  # error turns are turns, but never empty
    assert row.spend_usd == pytest.approx(4.0)  # error turns cost money too
    assert row.empty_spend_usd == pytest.approx(2.0)
    assert row.empty_spend_share == pytest.approx(0.5)


def test_empty_turns_are_pooled_over_the_runs_of_an_arm(make_summary: Factory) -> None:
    summaries = [
        make_summary("solo", 1, turns=[turn("empty", 3.0), turn("candidate", 1.0)]),
        make_summary("solo", 2, turns=[turn("candidate", 4.0)]),
        make_summary("team", 1, turns=[turn("empty", 1.0)]),
    ]

    solo, team = empty_turn_table(summaries)

    assert (solo.runs, solo.turns, solo.empty_turns) == (2, 3, 1)
    assert solo.empty_share == pytest.approx(1 / 3)
    assert solo.empty_spend_share == pytest.approx(3 / 8)
    assert (team.arm, team.empty_share, team.empty_spend_share) == ("team", 1.0, 1.0)


def test_shares_are_undefined_without_turns_or_spend(make_summary: Factory) -> None:
    idle = make_summary("idle", turns=[])
    free = make_summary("free", turns=[turn("empty", 0.0)])

    none_at_all, no_spend = empty_turn_table([idle, free])

    assert (none_at_all.turns, none_at_all.empty_share, none_at_all.empty_spend_share) == (
        0,
        None,
        None,
    )
    assert (no_spend.empty_share, no_spend.empty_spend_share) == (1.0, None)


def test_empty_turn_spend_from_an_event_log(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "solo" / "rep1", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.empty("a1", cost=0.25)
    b.error("a1", cost=0.25)
    b.empty("a1", cost=1.0)
    summary = summarize_run(b.finish(), arm="solo", rep=1)

    (row,) = empty_turn_table([summary])

    assert (row.turns, row.empty_turns, row.error_turns) == (4, 2, 1)
    assert row.empty_spend_usd == pytest.approx(1.25)
    assert row.empty_spend_share == pytest.approx(1.25 / 2.0)


# ---------------------------------------------------------------------------------------------
# cost to reach a score
# ---------------------------------------------------------------------------------------------

CURVE = ((0.0, 100.0), (0.5, 110.0), (1.5, 130.0), (2.0, 130.0))


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(90.0, 0.0), (100.0, 0.0), (105.0, 0.5), (110.0, 0.5), (130.0, 1.5), (131.0, None)],
)
def test_spend_to_reach_reads_the_first_curve_point_at_least_as_good(
    threshold: float, expected: float | None
) -> None:
    assert spend_to_reach(CURVE, threshold, "max") == expected


def test_spend_to_reach_on_a_lower_is_better_task() -> None:
    curve = ((0.0, 1500.0), (1.0, 1300.0), (2.5, 1150.0))
    assert spend_to_reach(curve, 1300.0, "min") == 1.0
    assert spend_to_reach(curve, 1200.0, "min") == 2.5
    assert spend_to_reach(curve, 1100.0, "min") is None
    assert spend_to_reach((), 1100.0, "min") is None


def test_threshold_table_summarizes_the_runs_that_reached_each_score(
    make_summary: Factory,
) -> None:
    summaries = [
        make_summary("tf", 1, direction="min", curve=[(0.0, 1500.0), (1.0, 1250.0)]),
        make_summary("tf", 2, direction="min", curve=[(0.0, 1500.0), (3.0, 1100.0)]),
        make_summary("tf", 3, direction="min", curve=[(0.0, 1500.0), (2.0, 1400.0)]),
        make_summary("par", 1, direction="min", curve=[(0.0, 1500.0)]),
    ]

    rows = threshold_table(summaries, [1300.0, 1100.0])

    assert [(r.arm, r.threshold, r.runs, r.reached) for r in rows] == [
        ("tf", 1300.0, 3, 2),
        ("tf", 1100.0, 3, 1),
        ("par", 1300.0, 1, 0),
        ("par", 1100.0, 1, 0),
    ]
    first = rows[0]
    assert (first.mean_spend, first.min_spend, first.max_spend) == (2.0, 1.0, 3.0)
    assert (rows[1].mean_spend, rows[1].min_spend, rows[1].max_spend) == (3.0, 3.0, 3.0)
    assert (rows[2].mean_spend, rows[2].min_spend, rows[2].max_spend) == (None, None, None)


def test_threshold_table_without_thresholds_is_empty(make_summary: Factory) -> None:
    assert threshold_table([make_summary("solo")], []) == []


def test_cost_to_reach_from_an_event_log(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "solo" / "rep1", agents=("a1",), direction="min").start(1500.0)
    b.candidate("a1", 1400.0, cost=0.5)
    b.empty("a1", cost=0.25)
    b.candidate("a1", 1200.0, cost=0.5)
    summary = summarize_run(b.finish(), arm="solo", rep=1)

    rows = threshold_table([summary], [1500.0, 1300.0])

    assert [(r.reached, r.mean_spend) for r in rows] == [(1, 0.0), (1, pytest.approx(1.25))]
