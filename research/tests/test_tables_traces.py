"""Tests for research.tables: anchoring traces and arm comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from research.summary import RunSummary, TurnRecord
from research.tables import anchoring_traces, compare_arms

Factory = Callable[..., Any]


def runs(make_summary: Factory, arm: str, finals: Sequence[float | None], **kwargs: Any) -> list:
    """One summary per value in `finals`, numbered from rep 1."""
    return [
        make_summary(arm, rep, final_best=value, **kwargs) for rep, value in enumerate(finals, 1)
    ]


# ---------------------------------------------------------------------------------------------
# anchoring traces
# ---------------------------------------------------------------------------------------------


def turn(
    spend: float,
    *,
    lines: int = 10,
    out: int = 100,
    think: int = 50,
    status: str = "candidate",
    plateau: bool = False,
) -> TurnRecord:
    return TurnRecord(
        spend=spend,
        lines_changed=lines,
        output_tokens=out,
        thinking_tokens=think,
        status=status,  # type: ignore[arg-type]
        plateau_claim=plateau,
    )


def test_default_bins_are_a_tenth_of_the_budget(make_summary: Factory) -> None:
    turns = [turn(0.1), turn(0.2), turn(0.35), turn(2.95)]
    rows = anchoring_traces([make_summary("solo", turns=turns, budget_usd=3.0)])

    assert [r.n_turns for r in rows] == [2, 1, 1]
    assert [r.bin_start for r in rows] == pytest.approx([0.0, 0.3, 2.7])
    assert [r.bin_end for r in rows] == pytest.approx([0.3, 0.6, 3.0])


def test_medians_are_pooled_over_all_replicates_of_an_arm(make_summary: Factory) -> None:
    first = make_summary(
        "solo",
        1,
        turns=[turn(0.1, lines=10, out=100, think=10), turn(0.2, lines=20, out=300, think=30)],
    )
    second = make_summary("solo", 2, turns=[turn(0.15, lines=30, out=200, think=20)])

    (row,) = anchoring_traces([first, second])

    assert row.n_turns == 3
    assert row.median_lines_changed == 20.0
    assert row.median_output_tokens == 200.0
    assert row.median_thinking_tokens == 20.0


def test_median_of_an_even_number_of_turns_averages_the_middle_pair(make_summary: Factory) -> None:
    turns = [turn(0.1, lines=10), turn(0.1, lines=40)]
    (row,) = anchoring_traces([make_summary("solo", turns=turns)])
    assert row.median_lines_changed == 25.0


def test_empty_and_plateau_rates_are_fractions_of_the_proposals_in_the_bin(
    make_summary: Factory,
) -> None:
    turns = [
        turn(0.1, status="candidate"),
        turn(0.1, status="empty", lines=0, plateau=True),
        turn(0.2, status="empty", lines=0),
        turn(0.2, status="candidate", plateau=False),
    ]
    (row,) = anchoring_traces([make_summary("solo", turns=turns)])

    assert row.empty_rate == pytest.approx(0.5)
    assert row.plateau_rate == pytest.approx(0.25)


def test_empty_turns_pull_the_median_edit_size_down(make_summary: Factory) -> None:
    turns = [
        turn(0.1, lines=50),
        turn(0.1, status="empty", lines=0),
        turn(0.1, status="empty", lines=0),
    ]
    (row,) = anchoring_traces([make_summary("solo", turns=turns)])
    assert row.median_lines_changed == 0.0


def test_error_turns_are_left_out_of_every_statistic(make_summary: Factory) -> None:
    turns = [turn(0.1, lines=10), turn(0.1, status="error", lines=0, out=0, think=0)]
    (row,) = anchoring_traces([make_summary("solo", turns=turns)])

    assert row.n_turns == 1
    assert row.median_lines_changed == 10.0
    assert row.empty_rate == 0.0


def test_a_bin_with_only_error_turns_has_no_row(make_summary: Factory) -> None:
    turns = [turn(0.1), turn(0.5, status="error")]
    rows = anchoring_traces([make_summary("solo", turns=turns)])
    assert [r.n_turns for r in rows] == [1]


def test_each_arm_gets_its_own_rows_in_first_appearance_order(make_summary: Factory) -> None:
    summaries = [
        make_summary("team", turns=[turn(0.1, lines=100)]),
        make_summary("solo", turns=[turn(0.1, lines=5), turn(1.0, lines=2)]),
    ]
    rows = anchoring_traces(summaries)

    assert [(r.arm, r.median_lines_changed) for r in rows] == [
        ("team", 100.0),
        ("solo", 5.0),
        ("solo", 2.0),
    ]


def test_bin_boundaries_are_not_hurt_by_floating_point_noise(make_summary: Factory) -> None:
    # 0.3 / 0.1 is 2.9999999999999996 in floating point, but 0.3 belongs to the bin starting at 0.3
    turns = [turn(0.3)]
    (row,) = anchoring_traces([make_summary("solo", turns=turns, budget_usd=1.0)])
    assert row.bin_start == pytest.approx(0.3)


def test_spend_past_the_budget_lands_in_the_last_bin(make_summary: Factory) -> None:
    turns = [turn(0.95), turn(1.07)]
    (row,) = anchoring_traces([make_summary("solo", turns=turns, budget_usd=1.0)])
    assert row.n_turns == 2
    assert row.bin_start == pytest.approx(0.9)


def test_an_explicit_bin_width_overrides_the_default(make_summary: Factory) -> None:
    turns = [turn(0.4), turn(2.9)]
    rows = anchoring_traces([make_summary("solo", turns=turns, budget_usd=3.0)], bin_width=0.5)
    assert [(r.bin_start, r.bin_end) for r in rows] == pytest.approx([(0.0, 0.5), (2.5, 3.0)])


def test_without_a_usd_budget_the_largest_spend_defines_the_bins(make_summary: Factory) -> None:
    short = make_summary("solo", 1, turns=[turn(0.1)], budget_usd=None, spend_usd=1.0)
    long = make_summary("solo", 2, turns=[turn(1.9)], budget_usd=None, spend_usd=2.0)

    rows = anchoring_traces([short, long])

    assert [(r.bin_start, r.bin_end) for r in rows] == pytest.approx([(0.0, 0.2), (1.8, 2.0)])


def test_traces_of_runs_with_no_spend_are_empty(make_summary: Factory) -> None:
    assert anchoring_traces([make_summary("solo", turns=[], budget_usd=None, spend_usd=0.0)]) == []
    assert anchoring_traces([]) == []


@pytest.mark.parametrize("kwargs", [{"bin_width": 0.0}, {"bin_width": -1.0}, {"bins": 0}])
def test_traces_reject_invalid_binning(make_summary: Factory, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        anchoring_traces([make_summary("solo", turns=[turn(0.1)])], **kwargs)  # type: ignore[arg-type]


def test_rows_follow_spend_order_within_an_arm(make_summary: Factory) -> None:
    """Edit size shrinking over the spend axis (anchoring) reads left to right in the rows."""
    shrinking = [turn(0.1 + 0.3 * i, lines=100 // (i + 1)) for i in range(5)]

    rows = anchoring_traces([make_summary("solo", turns=shrinking[::-1])])  # fed out of order

    assert [r.median_lines_changed for r in rows] == [100.0, 50.0, 33.0, 25.0, 20.0]
    assert [r.bin_start for r in rows] == sorted(r.bin_start for r in rows)


# ---------------------------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------------------------


def welch_data(make_summary: Factory, **kw: Any) -> list[RunSummary]:
    return runs(make_summary, "team-forget", [12.0, 15.0, 14.0, 10.0, 13.0, 16.0], **kw) + runs(
        make_summary, "parallel", [9.0, 11.0, 10.0, 12.0], **kw
    )


def test_comparison_reports_welch_and_a_bootstrap_interval(make_summary: Factory) -> None:
    c = compare_arms(welch_data(make_summary), "team-forget", "parallel", n_boot=3000)

    assert (c.arm_a, c.arm_b, c.n_a, c.n_b) == ("team-forget", "parallel", 6, 4)
    assert c.mean_a == pytest.approx(80 / 6)
    assert c.mean_b == pytest.approx(10.5)
    assert c.diff == pytest.approx(80 / 6 - 10.5)
    assert c.t == pytest.approx(2.59247569565, abs=1e-9)
    assert c.df == pytest.approx(7.97670405522, abs=1e-9)
    assert c.p == pytest.approx(0.0320668355, abs=1e-8)
    assert c.ci_low is not None and c.ci_high is not None
    assert c.ci_low < c.diff < c.ci_high
    assert c.confidence == 0.95


def test_comparison_change_is_a_percentage_in_the_better_direction(make_summary: Factory) -> None:
    c = compare_arms(welch_data(make_summary), "team-forget", "parallel", n_boot=200)
    assert c.change_pct == pytest.approx(100 * (80 / 6 - 10.5) / 10.5)

    lower = compare_arms(
        welch_data(make_summary, direction="min"), "team-forget", "parallel", n_boot=200
    )
    assert lower.change_pct == pytest.approx(-c.change_pct)


def test_comparison_is_reproducible(make_summary: Factory) -> None:
    summaries = welch_data(make_summary)
    first = compare_arms(summaries, "team-forget", "parallel", n_boot=1000, seed=3)
    second = compare_arms(summaries, "team-forget", "parallel", n_boot=1000, seed=3)
    assert first == second


def test_comparison_confidence_level_is_passed_through(make_summary: Factory) -> None:
    summaries = welch_data(make_summary)
    narrow = compare_arms(summaries, "team-forget", "parallel", n_boot=2000, confidence=0.5)
    wide = compare_arms(summaries, "team-forget", "parallel", n_boot=2000, confidence=0.99)
    assert narrow.confidence == 0.5
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)  # type: ignore[operator]


def test_comparison_swapping_arms_negates_the_difference(make_summary: Factory) -> None:
    summaries = welch_data(make_summary)
    forward = compare_arms(summaries, "team-forget", "parallel", n_boot=200)
    backward = compare_arms(summaries, "parallel", "team-forget", n_boot=200)
    assert backward.diff == pytest.approx(-forward.diff)  # type: ignore[operator]
    assert backward.t == pytest.approx(-forward.t)  # type: ignore[operator]
    assert backward.p == pytest.approx(forward.p)  # type: ignore[arg-type]


def test_comparison_needs_two_valid_runs_per_arm_for_inference(make_summary: Factory) -> None:
    summaries = runs(make_summary, "a", [10.0]) + runs(make_summary, "b", [8.0, 9.0, 10.0])

    c = compare_arms(summaries, "a", "b", n_boot=100)

    assert (c.n_a, c.n_b) == (1, 3)
    assert c.mean_a == 10.0
    assert c.diff == pytest.approx(1.0)
    assert (c.t, c.df, c.p, c.ci_low, c.ci_high) == (None, None, None, None, None)


def test_comparison_skips_runs_without_a_valid_result(make_summary: Factory) -> None:
    summaries = runs(make_summary, "a", [10.0, None, 12.0]) + runs(make_summary, "b", [1.0, 2.0])
    c = compare_arms(summaries, "a", "b", n_boot=100)
    assert c.n_a == 2


def test_comparison_with_an_arm_that_has_no_valid_run(make_summary: Factory) -> None:
    summaries = runs(make_summary, "a", [10.0, 11.0]) + runs(make_summary, "b", [None])
    c = compare_arms(summaries, "a", "b", n_boot=100)
    assert (c.n_b, c.mean_b, c.diff, c.change_pct, c.p) == (0, None, None, None, None)


def test_comparison_rejects_unknown_arms(make_summary: Factory) -> None:
    with pytest.raises(ValueError, match=r"nope.*team-forget, parallel"):
        compare_arms(welch_data(make_summary), "nope", "parallel")
