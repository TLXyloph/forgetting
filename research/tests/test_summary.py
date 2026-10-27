"""Tests for research.summary: folding an event log into a RunSummary (the fold itself).

Loading whole experiment directories is covered in test_summary_logs.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lethe.domain.events import RateLimited
from research.summary import (
    RoundScores,
    TurnRecord,
    summarize_events,
    summarize_run,
)

# ---------------------------------------------------------------------------------------------
# a plain solo run
# ---------------------------------------------------------------------------------------------


def solo_run(run_builder: type, run_dir: Path) -> Path:
    """Seed 100; proposals: +10, empty (plateau claim), worse, +20, empty. Spend 2.0 in total."""
    b = run_builder(run_dir, agents=("a1",), budget_usd=3.0, t0=500.0).start(100.0)
    b.candidate("a1", 110.0, cost=0.5, lines=40, out=600, think=300)
    b.empty("a1", cost=0.25, out=100, think=50, plateau=True)
    b.candidate("a1", 105.0, cost=0.25, lines=5, out=200, think=90)
    b.candidate("a1", 130.0, cost=0.5, lines=60, out=700, think=400)
    b.empty("a1", cost=0.5, out=120, think=60)
    return b.finish()


def test_headline_numbers_of_a_solo_run(tmp_path: Path, run_builder: type) -> None:
    run_dir = solo_run(run_builder, tmp_path / "solo" / "rep1")

    s = summarize_run(run_dir, arm="solo", rep=1)

    assert (s.arm, s.rep, s.run_dir) == ("solo", 1, run_dir)
    assert s.direction == "max"
    assert s.finished is True
    assert s.seed_score == 100.0
    assert s.final_best == 130.0
    assert s.spend_usd == pytest.approx(2.0)
    assert s.proposals == 5
    assert s.team_forgets == 0
    assert s.agent_resets == 0
    assert s.budget_usd == 3.0
    assert s.started_at == 501.0  # timestamp of the first event


def test_a_run_without_forgets_has_a_single_round(tmp_path: Path, run_builder: type) -> None:
    s = summarize_run(solo_run(run_builder, tmp_path / "r"), arm="solo", rep=1)
    assert s.rounds == (RoundScores(100.0, 130.0),)


def test_best_vs_spend_curve_steps_at_each_improvement_and_ends_at_total_spend(
    tmp_path: Path, run_builder: type
) -> None:
    s = summarize_run(solo_run(run_builder, tmp_path / "r"), arm="solo", rep=1)

    # seed at spend 0; +10 after the first turn; the "worse" 105 adds nothing; +20 at 1.5;
    # a closing point at the total spend of 2.0
    assert s.curve == ((0.0, 100.0), (0.5, 110.0), (1.5, 130.0), (2.0, 130.0))


def test_per_turn_records_carry_cumulative_spend(tmp_path: Path, run_builder: type) -> None:
    s = summarize_run(solo_run(run_builder, tmp_path / "r"), arm="solo", rep=1)

    assert [t.spend for t in s.turns] == [0.5, 0.75, 1.0, 1.5, 2.0]
    assert s.turns[0] == TurnRecord(
        spend=0.5,
        lines_changed=40,
        output_tokens=600,
        thinking_tokens=300,
        status="candidate",
        plateau_claim=False,
        cost=0.5,
    )
    assert s.turns[1] == TurnRecord(
        spend=0.75,
        lines_changed=0,
        output_tokens=100,
        thinking_tokens=50,
        status="empty",
        plateau_claim=True,
        cost=0.25,
    )


def test_per_turn_records_carry_their_own_cost(tmp_path: Path, run_builder: type) -> None:
    s = summarize_run(solo_run(run_builder, tmp_path / "r"), arm="solo", rep=1)
    assert [t.cost for t in s.turns] == [0.5, 0.25, 0.25, 0.5, 0.5]
    assert sum(t.cost for t in s.turns) == pytest.approx(s.spend_usd)


def test_a_turn_record_built_without_a_cost_costs_nothing() -> None:
    record = TurnRecord(
        spend=1.0,
        lines_changed=0,
        output_tokens=0,
        thinking_tokens=0,
        status="empty",
        plateau_claim=False,
    )
    assert record.cost == 0.0


def test_a_tie_with_the_best_is_not_an_improvement(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 100.0, cost=0.5)  # equals the seed
    b.candidate("a1", 120.0, cost=0.5)
    b.candidate("a1", 120.0, cost=0.5)  # equals the best
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.curve == ((0.0, 100.0), (1.0, 120.0), (1.5, 120.0))


def test_a_late_judge_result_is_credited_to_the_spend_at_the_end_of_its_own_turn(
    tmp_path: Path, run_builder: type
) -> None:
    """The harness judges asynchronously: four agents' turns finish before the first judge result,
    yet the improvement belongs to the spend at which its own proposal was made."""
    b = run_builder(tmp_path / "r").start(100.0)
    ids = [b.candidate(a, 110.0 + 10 * i, cost=0.5, judge=False) for i, a in enumerate(b.agents)]
    for cand_id in ids:
        b.judge_pending(cand_id)  # all four results arrive after all four turns, at spend 2.0
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.curve == ((0.0, 100.0), (0.5, 110.0), (1.0, 120.0), (1.5, 130.0), (2.0, 140.0))
    assert s.final_best == 140.0


def test_judge_results_that_arrive_out_of_order_still_give_a_monotone_curve(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1", "a2")).start(100.0)
    first = b.candidate("a1", 130.0, cost=0.5, judge=False)  # the better one, proposed first
    second = b.candidate("a2", 120.0, cost=0.5, judge=False)
    b.judge_pending(second)  # but judged last
    b.judge_pending(first)
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.curve == ((0.0, 100.0), (0.5, 130.0), (1.0, 130.0))  # 120 at spend 1.0 adds nothing
    assert [spend for spend, _ in s.curve] == sorted(spend for spend, _ in s.curve)
    assert s.final_best == 130.0


def test_a_candidate_that_was_never_judged_does_not_appear_on_the_curve(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 150.0, cost=0.5, judge=False)  # the run ended before it was judged
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.final_best == 100.0
    assert s.curve == ((0.0, 100.0), (0.5, 100.0))


def test_summarize_events_accepts_any_iterable_of_envelopes(
    tmp_path: Path, run_builder: type
) -> None:
    from lethe.store import read_events

    run_dir = solo_run(run_builder, tmp_path / "r")
    from_file = summarize_run(run_dir, arm="a", rep=2)
    from_iter = summarize_events(
        iter(list(read_events(run_dir / "events.jsonl"))), arm="a", rep=2, run_dir=run_dir
    )
    assert from_iter == from_file


# ---------------------------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------------------------


def test_min_direction_prefers_lower_scores(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", direction="min", agents=("a1",)).start(1000.0)
    b.candidate("a1", 900.0, cost=0.5)
    b.candidate("a1", 950.0, cost=0.5)  # worse for a min task
    b.candidate("a1", 800.0, cost=0.5)
    s = summarize_run(b.finish(), arm="vliw", rep=1)

    assert s.direction == "min"
    assert s.seed_score == 1000.0
    assert s.final_best == 800.0
    assert s.curve == ((0.0, 1000.0), (0.5, 900.0), (1.5, 800.0))
    assert s.rounds == (RoundScores(1000.0, 800.0),)


def test_zero_and_negative_scores_are_ordinary_scores(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(-5.0)
    b.candidate("a1", 0.0, cost=0.5)
    b.candidate("a1", -1.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert (s.seed_score, s.final_best) == (-5.0, 0.0)


# ---------------------------------------------------------------------------------------------
# invalid candidates
# ---------------------------------------------------------------------------------------------


def test_invalid_candidates_never_count_as_best(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.invalid("a1", cost=0.5)
    b.candidate("a1", 120.0, cost=0.5)
    b.invalid("a1", cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.final_best == 120.0
    assert s.proposals == 3  # an invalid candidate is still a proposal
    assert s.curve == ((0.0, 100.0), (1.0, 120.0), (1.5, 120.0))


def test_an_invalid_seed_leaves_the_seed_score_unset(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(None)
    b.candidate("a1", 50.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.seed_score is None
    assert s.final_best == 50.0
    assert s.curve == ((0.5, 50.0),)  # no point at spend 0: nothing valid existed yet
    assert s.rounds == (RoundScores(None, 50.0),)


def test_a_run_with_no_valid_candidate_at_all(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(None)
    b.invalid("a1", cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.seed_score is None
    assert s.final_best is None
    assert s.curve == ()
    assert s.rounds == (RoundScores(None, None),)
    assert s.proposals == 1
    assert s.spend_usd == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------
# turn bookkeeping
# ---------------------------------------------------------------------------------------------


def test_error_turns_cost_money_but_are_not_proposals(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.error("a1", cost=0.25)
    b.empty("a1", cost=0.25)
    s = summarize_run(b.finish(), arm="x", rep=1)

    assert s.proposals == 2  # candidate + empty
    assert s.spend_usd == pytest.approx(1.0)
    assert [t.status for t in s.turns] == ["candidate", "error", "empty"]
    assert s.turns[1].spend == 0.75
    assert s.turns[1].cost == 0.25


def test_rate_limit_events_do_not_disturb_the_summary(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.store.append(RateLimited(agent="a1", resets_at=None, message="5h window full"))
    b.candidate("a1", 120.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert (s.final_best, s.proposals) == (120.0, 2)


def test_parallel_agents_share_one_spend_counter(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r").start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.candidate("a2", 105.0, cost=0.5)
    b.candidate("a3", 115.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert [t.spend for t in s.turns] == [0.5, 1.0, 1.5]
    assert s.curve == ((0.0, 100.0), (0.5, 110.0), (1.5, 115.0))


# ---------------------------------------------------------------------------------------------
# rounds and forgetting
# ---------------------------------------------------------------------------------------------


def forgetting_run(run_builder: type, run_dir: Path) -> Path:
    b = run_builder(run_dir, agents=("a1", "a2")).start(100.0)
    b.candidate("a1", 120.0, cost=0.5)
    b.candidate("a2", 110.0, cost=0.5)
    b.forget()  # round 1 ends with best 120
    b.candidate("a1", 140.0, cost=0.5)
    b.candidate("a2", 125.0, cost=0.5)
    b.forget()  # round 2 ends with best 140
    b.empty("a1", cost=0.5)
    b.empty("a2", cost=0.5)
    return b.finish()


def test_team_forgets_split_the_run_into_rounds(tmp_path: Path, run_builder: type) -> None:
    s = summarize_run(forgetting_run(run_builder, tmp_path / "r"), arm="tf", rep=1)

    assert s.team_forgets == 2
    # [seed score, best at round end]; each later round starts from the best so far
    assert s.rounds == (
        RoundScores(100.0, 120.0),
        RoundScores(120.0, 140.0),
        RoundScores(140.0, 140.0),
    )
    assert s.final_best == 140.0


def test_curve_of_a_forgetting_run_is_monotone_and_spans_the_whole_budget(
    tmp_path: Path, run_builder: type
) -> None:
    s = summarize_run(forgetting_run(run_builder, tmp_path / "r"), arm="tf", rep=1)

    assert s.curve == ((0.0, 100.0), (0.5, 120.0), (1.5, 140.0), (3.0, 140.0))
    bests = [best for _, best in s.curve]
    assert bests == sorted(bests)


def test_min_direction_rounds_track_the_lowest_score(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "r", direction="min", agents=("a1", "a2")).start(1000.0)
    b.candidate("a1", 900.0, cost=0.5)
    b.candidate("a2", 950.0, cost=0.5)
    b.forget()
    b.candidate("a1", 850.0, cost=0.5)
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.rounds == (RoundScores(1000.0, 900.0), RoundScores(900.0, 850.0))


def test_a_run_that_forgets_right_before_the_end_still_reports_its_last_round(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 120.0, cost=0.5)
    b.forget()
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.rounds == (RoundScores(100.0, 120.0), RoundScores(120.0, 120.0))


def test_hedged_seeding_uses_the_best_starting_point_as_the_round_seed(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1", "a2", "a3", "a4")).start(100.0)
    best = b.candidate("a1", 130.0, cost=0.5)
    elite = b.candidate("a2", 115.0, cost=0.5)
    b.forget(seeds={"a1": best, "a2": best, "a3": elite, "a4": elite})
    b.candidate("a3", 118.0, cost=0.5)  # a hedged agent improves on its elite but not on the best
    s = summarize_run(b.finish(), arm="hedged", rep=1)

    assert s.rounds == (RoundScores(100.0, 130.0), RoundScores(130.0, 130.0))


def test_hedged_seeding_for_a_min_task_picks_the_lowest_start(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", direction="min", agents=("a1", "a2")).start(1000.0)
    best = b.candidate("a1", 800.0, cost=0.5)
    worse = b.candidate("a2", 900.0, cost=0.5)
    b.forget(seeds={"a1": worse, "a2": best})
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.rounds[1].seed == 800.0


def test_a_forget_seed_that_was_never_scored_gives_no_round_seed(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1",)).start(100.0)
    b.candidate("a1", 120.0, cost=0.5)
    b.forget(seeds={"a1": "c9999"})
    s = summarize_run(b.finish(), arm="x", rep=1)
    assert s.rounds == (RoundScores(100.0, 120.0), RoundScores(None, 120.0))


def test_agent_scope_resets_are_counted_but_do_not_start_a_round(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1", "a2")).start(100.0)
    b.candidate("a1", 120.0, cost=0.5)
    b.reset_agent("a2")
    b.candidate("a2", 125.0, cost=0.5)
    b.reset_agent("a1")
    s = summarize_run(b.finish(), arm="stalled", rep=1)

    assert s.team_forgets == 0
    assert s.agent_resets == 2
    assert s.rounds == (RoundScores(100.0, 125.0),)


# ---------------------------------------------------------------------------------------------
# unplanned forgets
# ---------------------------------------------------------------------------------------------


def test_a_session_restarted_without_a_forget_is_an_unplanned_forget(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1", "a2")).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.candidate("a2", 105.0, cost=0.5)
    b.error("a1")  # a failure the harness does not resume: a1 starts over
    b.new_session("a1")
    b.error("a1")  # still the same restart, however many turns it takes to recover
    b.candidate("a1", 120.0, cost=0.5)
    b.empty("a1", cost=0.5)
    s = summarize_run(b.finish(), arm="solo", rep=1)

    assert s.unplanned_forgets == 1


def test_planned_forgets_resumed_errors_and_sessions_that_never_ran_are_not_unplanned(
    tmp_path: Path, run_builder: type
) -> None:
    b = run_builder(tmp_path / "r", agents=("a1", "a2")).start(100.0)
    b.new_session("a2")
    b.error("a2")  # a2's first session failed before any turn: it held nothing to forget
    b.new_session("a2")
    b.candidate("a2", 105.0, cost=0.5)
    b.candidate("a1", 110.0, cost=0.5)
    b.error("a1")  # resumed: the next turn continues the same session
    b.candidate("a1", 115.0, cost=0.5)
    b.forget()
    b.candidate("a1", 120.0, cost=0.5)
    b.reset_agent("a2")
    b.candidate("a2", 125.0, cost=0.5)
    s = summarize_run(b.finish(), arm="tf", rep=1)

    assert s.unplanned_forgets == 0


def test_an_arm_whose_sessions_never_continue_has_no_unplanned_forgets(
    tmp_path: Path, run_builder: type
) -> None:
    config = {
        "team": {"session": "fresh"},
        "budget": {"usd": 3.0, "proposals": 0, "primary": "usd"},
    }
    b = run_builder(tmp_path / "r", agents=("a1",), config=config).start(100.0)
    b.candidate("a1", 110.0, cost=0.5)
    b.new_session("a1")
    b.candidate("a1", 120.0, cost=0.5)
    s = summarize_run(b.finish(), arm="fresh-every-turn", rep=1)

    assert s.unplanned_forgets == 0
