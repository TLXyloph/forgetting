"""Tests for research.tables: Table 1 and Table 2.

Anchoring traces and arm comparisons are covered in test_tables_traces.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from research import stats
from research.analyze import build_report, render_report
from research.summary import RunSummary
from research.tables import (
    arm_names,
    arm_table,
    improvement_pct,
    round_table,
)

Factory = Callable[..., Any]


def runs(make_summary: Factory, arm: str, finals: Sequence[float | None], **kwargs: Any) -> list:
    """One summary per value in `finals`, numbered from rep 1."""
    return [
        make_summary(arm, rep, final_best=value, **kwargs) for rep, value in enumerate(finals, 1)
    ]


def paper_like(make_summary: Factory) -> list[RunSummary]:
    return (
        runs(make_summary, "solo", [100.0, 110.0, 90.0])
        + runs(make_summary, "parallel", [120.0, 130.0, 110.0])
        + runs(make_summary, "team-forget", [140.0, 150.0, 130.0])
    )


# ---------------------------------------------------------------------------------------------
# improvement_pct
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "reference", "direction", "expected"),
    [
        (110.0, 100.0, "max", 10.0),
        (90.0, 100.0, "max", -10.0),
        (90.0, 100.0, "min", 10.0),  # fewer cycles is better
        (110.0, 100.0, "min", -10.0),
        (-5.0, -10.0, "max", 50.0),  # a negative reference: relative to its magnitude
        (-15.0, -10.0, "min", 50.0),
        (7.0, 7.0, "max", 0.0),
    ],
)
def test_improvement_is_signed_in_the_better_direction(
    value: float, reference: float, direction: str, expected: float
) -> None:
    assert improvement_pct(value, reference, direction) == pytest.approx(expected)  # type: ignore[arg-type]


def test_improvement_over_a_zero_reference_is_undefined() -> None:
    assert improvement_pct(5.0, 0.0, "max") is None


# ---------------------------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------------------------


def test_table_1_rows_in_order_with_summary_statistics(make_summary: Factory) -> None:
    rows = arm_table(paper_like(make_summary))

    assert [r.arm for r in rows] == ["solo", "parallel", "team-forget"]
    assert [r.n for r in rows] == [3, 3, 3]
    assert [r.best_of_runs for r in rows] == [110.0, 130.0, 150.0]
    assert [r.mean for r in rows] == [100.0, 120.0, 140.0]
    assert [r.sd for r in rows] == [pytest.approx(10.0)] * 3


def test_table_1_change_versus_the_default_reference_arm_solo(make_summary: Factory) -> None:
    solo, parallel, team = arm_table(paper_like(make_summary))

    assert solo.is_reference and solo.mean_change_pct is None and solo.best_change_pct is None
    assert not parallel.is_reference
    assert parallel.mean_change_pct == pytest.approx(20.0)  # (120 - 100) / 100
    assert parallel.best_change_pct == pytest.approx(100 * 20 / 110)  # best-of-runs 130 vs 110
    assert team.mean_change_pct == pytest.approx(40.0)
    assert team.best_change_pct == pytest.approx(100 * 40 / 110)


def test_table_1_min_task_improvement_is_positive_when_the_score_drops(
    make_summary: Factory,
) -> None:
    summaries = runs(make_summary, "solo", [1200.0, 1400.0, 1600.0], direction="min") + runs(
        make_summary, "parallel", [1100.0, 1200.0, 1300.0], direction="min"
    )

    solo, parallel = arm_table(summaries)

    assert solo.best_of_runs == 1200.0  # lowest is best
    assert parallel.best_of_runs == 1100.0
    assert parallel.mean_change_pct == pytest.approx(100 * 200 / 1400)
    assert parallel.best_change_pct == pytest.approx(100 * 100 / 1200)


def test_table_1_custom_reference_arm(make_summary: Factory) -> None:
    rows = {r.arm: r for r in arm_table(paper_like(make_summary), ref="parallel")}

    assert rows["parallel"].is_reference
    assert rows["solo"].mean_change_pct == pytest.approx(100 * (100 - 120) / 120)
    assert rows["team-forget"].mean_change_pct == pytest.approx(100 * 20 / 120)


@pytest.mark.parametrize("ref", [None, "missing"])
def test_table_1_without_a_usable_reference_has_no_change_columns(
    make_summary: Factory, ref: str | None
) -> None:
    rows = arm_table(paper_like(make_summary), ref=ref)

    assert all(r.mean_change_pct is None and r.best_change_pct is None for r in rows)
    assert not any(r.is_reference for r in rows)


def test_table_1_counts_only_runs_with_a_valid_result(make_summary: Factory) -> None:
    summaries = runs(make_summary, "solo", [100.0, None, 120.0])

    (row,) = arm_table(summaries)

    assert row.n == 2
    assert row.mean == pytest.approx(110.0)
    assert row.sd == pytest.approx(stats.sample_sd([100.0, 120.0]))


def test_table_1_single_run_has_no_standard_deviation(make_summary: Factory) -> None:
    (row,) = arm_table(runs(make_summary, "solo", [100.0]))
    assert (row.n, row.mean, row.sd, row.best_of_runs) == (1, 100.0, None, 100.0)


def test_table_1_arm_without_any_valid_run_still_gets_a_row(make_summary: Factory) -> None:
    summaries = runs(make_summary, "solo", [100.0]) + runs(make_summary, "broken", [None, None])

    rows = {r.arm: r for r in arm_table(summaries)}

    assert rows["broken"].n == 0
    assert rows["broken"].mean is None
    assert rows["broken"].best_of_runs is None
    assert rows["broken"].mean_change_pct is None


def test_table_1_rejects_runs_that_disagree_on_direction(make_summary: Factory) -> None:
    summaries = runs(make_summary, "solo", [1.0]) + runs(make_summary, "x", [2.0], direction="min")
    with pytest.raises(ValueError, match="direction"):
        arm_table(summaries)


def test_table_1_of_nothing_is_empty() -> None:
    assert arm_table([]) == []


def test_arm_names_keep_first_appearance_order(make_summary: Factory) -> None:
    summaries = (
        runs(make_summary, "z", [1.0, 2.0])
        + runs(make_summary, "a", [1.0])
        + runs(make_summary, "m", [3.0])
    )
    assert arm_names(summaries) == ["z", "a", "m"]


# ---------------------------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------------------------


def forgetting(
    make_summary: Factory, arm: str, per_run_round_bests: Sequence[Sequence[float]], **kw: Any
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


def test_table_2_is_the_mean_best_at_the_end_of_each_round(make_summary: Factory) -> None:
    summaries = forgetting(
        make_summary,
        "team-forget",
        [[120.0, 140.0, 150.0], [110.0, 130.0, 160.0], [130.0, 150.0, 170.0]],
    )

    rows = round_table(summaries)

    assert [(r.arm, r.round, r.n) for r in rows] == [("team-forget", k, 3) for k in (1, 2, 3)]
    assert [r.mean_best for r in rows] == [120.0, 140.0, 160.0]
    assert rows[0].change_pct is None  # nothing to compare the first round with
    assert rows[1].change_pct == pytest.approx(100 * 20 / 120)
    assert rows[2].change_pct == pytest.approx(100 * 20 / 140)


def test_table_2_last_round_mean_equals_the_arms_mean_final_best(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "tf", [[100.0, 120.0], [110.0, 150.0]])
    (_, last) = round_table(summaries)
    (row,) = arm_table(summaries)
    assert last.mean_best == pytest.approx(row.mean)


def test_table_2_only_lists_arms_that_forgot(make_summary: Factory) -> None:
    summaries = runs(make_summary, "solo", [100.0, 110.0]) + forgetting(
        make_summary, "tf", [[100.0, 120.0]]
    )
    assert {r.arm for r in round_table(summaries)} == {"tf"}


def test_table_2_of_a_run_without_forgets_is_empty(make_summary: Factory) -> None:
    assert round_table(runs(make_summary, "solo", [100.0, 110.0])) == []


def test_table_2_ignores_runs_that_never_forgot_within_a_forgetting_arm(
    make_summary: Factory,
) -> None:
    summaries = forgetting(make_summary, "plateau", [[100.0, 130.0, 140.0]]) + [
        make_summary("plateau", 2, team_forgets=0, rounds=[(None, 500.0)], final_best=500.0)
    ]
    rows = round_table(summaries)
    assert [r.mean_best for r in rows] == [100.0, 130.0, 140.0]
    assert all(r.n == 1 for r in rows)


def test_table_2_handles_runs_with_different_numbers_of_rounds(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "anchored", [[100.0, 120.0, 130.0], [110.0, 150.0]])

    rows = round_table(summaries)

    assert [(r.round, r.n, r.mean_best) for r in rows] == [
        (1, 2, 105.0),
        (2, 2, 135.0),
        (3, 1, 130.0),
    ]


def test_table_2_skips_rounds_with_no_valid_score(make_summary: Factory) -> None:
    summaries = [
        make_summary("tf", 1, team_forgets=1, rounds=[(None, None), (None, 90.0)], final_best=90.0),
        make_summary("tf", 2, team_forgets=1, rounds=[(None, 80.0), (None, 95.0)], final_best=95.0),
    ]
    rows = round_table(summaries)
    assert [(r.round, r.n, r.mean_best) for r in rows] == [(1, 1, 80.0), (2, 2, 92.5)]


def test_table_2_min_task_improvement_is_positive_when_cycles_drop(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "tf", [[1231.0, 1145.0]], direction="min")
    first, second = round_table(summaries)
    assert second.change_pct == pytest.approx(100 * (1231 - 1145) / 1231)
    assert first.change_pct is None


def test_table_2_keeps_arms_in_first_appearance_order(make_summary: Factory) -> None:
    summaries = forgetting(make_summary, "z-arm", [[1.0, 2.0]]) + forgetting(
        make_summary, "a-arm", [[1.0, 2.0]]
    )
    assert [r.arm for r in round_table(summaries)] == ["z-arm", "z-arm", "a-arm", "a-arm"]


def test_table_2_shows_how_many_of_the_arms_runs_reached_each_round(make_summary: Factory) -> None:
    """A trailing round only some replicates reached is visible as such (n of the arm's runs)."""
    summaries = forgetting(
        make_summary, "tf", [[100.0, 120.0, 130.0], [110.0, 150.0], [90.0, 95.0]]
    )

    rows = round_table(summaries)

    assert [(r.round, r.n, r.runs) for r in rows] == [(1, 3, 3), (2, 3, 3), (3, 1, 3)]
    text = render_report(build_report(summaries, experiment="x"))
    table2 = text[text.index("Table 2") :].split("\n\n")[0]
    assert "1/3" in table2 and "3/3" in table2
