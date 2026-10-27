"""Tests for research.analyze: formatting, the rendered report and the command line.

CSV and plot output are covered in test_analyze_csv_plots.py.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research import analyze
from research.analyze import (
    build_report,
    fmt_num,
    fmt_pct,
    format_table,
    main,
    render_report,
)
from research.summary import load_summaries


@pytest.fixture
def experiment(tmp_path: Path, build_analysis_experiment: Callable[..., Path]) -> Path:
    return build_analysis_experiment(tmp_path / "paper-toy-mis")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        (1_145_000.0, "1,145,000"),
        (1077.0, "1,077"),
        (123.456, "123.5"),
        (12.345, "12.35"),
        (0.96123, "0.9612"),
        (0.0, "0"),
        (90.0, "90"),
        (-250.0, "-250"),
        (111.1, "111.1"),
    ],
)
def test_fmt_num(value: float | None, expected: str) -> None:
    assert fmt_num(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "-"), (20.0, "+20.0%"), (-3.14159, "-3.1%"), (0.0, "+0.0%")],
)
def test_fmt_pct(value: float | None, expected: str) -> None:
    assert fmt_pct(value) == expected


def test_format_table_aligns_columns() -> None:
    text = format_table(
        ["arm", "n", "mean"], [["solo", "3", "100.0"], ["team-forget", "12", "9.5"]]
    )
    lines = text.splitlines()

    assert lines[0].split() == ["arm", "n", "mean"]
    assert set(lines[1].replace(" ", "")) == {"-"}
    assert lines[2].startswith("solo ")  # first column left-aligned
    assert lines[2].endswith("100.0")  # the rest right-aligned
    assert lines[3].endswith("  9.5")
    assert len({len(line) for line in lines}) == 1  # rectangular


def test_format_table_without_rows_keeps_the_header() -> None:
    assert format_table(["a", "b"], []).splitlines()[0].split() == ["a", "b"]


# ---------------------------------------------------------------------------------------------
# report content
# ---------------------------------------------------------------------------------------------


def report_text(experiment: Path, **kwargs: Any) -> str:
    summaries = load_summaries(experiment)
    return render_report(build_report(summaries, experiment=experiment.name, **kwargs))


def test_report_has_all_sections_and_the_arms(experiment: Path) -> None:
    text = report_text(experiment, compares=[("team-forget", "parallel")])

    assert "paper-toy-mis" in text
    assert "9 runs" in text
    assert "Table 1" in text
    assert "Table 2" in text
    assert "Anchoring traces" in text
    assert "Comparisons" in text
    for arm in ("solo", "parallel", "team-forget"):
        assert arm in text


def test_table_1_in_the_report_shows_means_and_changes(experiment: Path) -> None:
    text = report_text(experiment)

    assert "110 ± 10" in text  # solo: mean 110, sd 10
    assert "ref" in text  # the reference row is marked
    assert "+18.2%" in text  # parallel: mean 130 vs 110
    assert "+36.4%" in text  # team-forget: mean 150 vs 110


def test_report_states_which_direction_is_better(
    tmp_path: Path, build_analysis_experiment: Callable[..., Path]
) -> None:
    exp = build_analysis_experiment(tmp_path / "vliw", direction="min")
    assert "lower is better" in report_text(exp)


def test_table_2_lists_only_forgetting_arms(experiment: Path) -> None:
    report = build_report(load_summaries(experiment), experiment="x")
    assert {row.arm for row in report.rounds} == {"team-forget"}
    assert [row.round for row in report.rounds] == [1, 2]


def test_report_without_forgetting_arms_says_so(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "e" / "solo" / "rep1", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0)
    b.finish()
    text = render_report(build_report(load_summaries(tmp_path / "e"), experiment="e"))
    assert "no arm has team forgets" in text.lower()


def test_the_report_flags_a_run_with_an_unplanned_forget(tmp_path: Path, run_builder: type) -> None:
    b = run_builder(tmp_path / "e" / "solo" / "rep1", agents=("a1",)).start(100.0)
    b.candidate("a1", 110.0)
    b.error("a1")
    b.new_session("a1")
    b.candidate("a1", 120.0)
    b.finish()
    text = render_report(build_report(load_summaries(tmp_path / "e"), experiment="e"))
    listing = text[text.index("Runs") :]
    row = next(line for line in listing.splitlines() if line.startswith("solo/rep1"))
    assert "unplanned" in listing and row.split()[-2] == "1"  # the count, before "ended"


def test_report_includes_a_per_run_listing(experiment: Path) -> None:
    text = report_text(experiment)
    assert "solo/rep1" in text
    assert "team-forget/rep3" in text


def test_comparison_line_in_the_report(experiment: Path) -> None:
    text = report_text(experiment, compares=[("team-forget", "parallel")])
    assert "team-forget vs parallel" in text
    assert "Welch" in text
    assert "p=" in text
    assert "95% CI" in text


def test_comparison_with_too_few_runs_explains_itself(tmp_path: Path, run_builder: type) -> None:
    for arm in ("a", "b"):
        b = run_builder(tmp_path / "e" / arm / "rep1", agents=("a1",)).start(100.0)
        b.candidate("a1", 110.0)
        b.finish()
    text = render_report(
        build_report(load_summaries(tmp_path / "e"), experiment="e", compares=[("a", "b")])
    )
    assert "at least 2" in text


def test_build_report_validates_the_reference_and_comparisons(experiment: Path) -> None:
    summaries = load_summaries(experiment)
    with pytest.raises(ValueError, match="nope"):
        build_report(summaries, experiment="x", ref="nope", strict_ref=True)
    with pytest.raises(ValueError, match="nope"):
        build_report(summaries, experiment="x", compares=[("nope", "solo")])


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def test_cli_prints_the_tables(experiment: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(experiment)])

    out = capsys.readouterr().out
    assert code == 0
    assert "Table 1" in out
    assert "team-forget" in out
    assert "Comparisons" not in out  # none requested


def test_cli_compare_and_out(
    experiment: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "analysis"

    code = main(
        [
            str(experiment),
            "--compare",
            "team-forget:parallel",
            "--compare",
            "parallel:solo",
            "--out",
            str(out_dir),
        ]
    )

    text = capsys.readouterr().out
    assert code == 0
    assert text.count(" vs ") >= 2
    assert (out_dir / "table1.csv").is_file()
    assert len(read_csv(out_dir / "comparisons.csv")) == 2
    assert "table1.csv" in text  # the files written are listed


def test_cli_ref_changes_the_baseline(experiment: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(experiment), "--ref", "parallel"])
    text = capsys.readouterr().out
    assert code == 0
    assert "vs parallel" in text


def test_cli_without_a_solo_arm_just_omits_the_change(
    tmp_path: Path, run_builder: type, capsys: pytest.CaptureFixture[str]
) -> None:
    for arm in ("x", "y"):
        b = run_builder(tmp_path / "e" / arm / "rep1", agents=("a1",)).start(100.0)
        b.candidate("a1", 110.0)
        b.finish()
    assert main([str(tmp_path / "e")]) == 0
    assert "Table 1" in capsys.readouterr().out


def test_cli_bin_width(experiment: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "o"
    assert main([str(experiment), "--bin-width", "0.5", "--out", str(out_dir)]) == 0
    widths = {
        round(float(r["bin_end"]) - float(r["bin_start"]), 6)
        for r in read_csv(out_dir / "anchoring_traces.csv")
    }
    assert widths == {0.5}


def test_cli_excludes_unfinished_runs_unless_asked(
    experiment: Path, run_builder: type, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    b = run_builder(experiment / "solo" / "rep4", agents=("a1",)).start(100.0)
    b.candidate("a1", 999.0, cost=0.5)  # unfinished, and would dominate the mean

    out_dir = tmp_path / "default"
    with caplog.at_level(logging.WARNING):
        assert main([str(experiment), "--out", str(out_dir)]) == 0
    assert "solo/rep4" in caplog.text
    solo = next(r for r in read_csv(out_dir / "table1.csv") if r["arm"] == "solo")
    assert solo["n"] == "3"

    out_all = tmp_path / "all"
    assert main([str(experiment), "--include-unfinished", "--out", str(out_all)]) == 0
    solo = next(r for r in read_csv(out_all / "table1.csv") if r["arm"] == "solo")
    assert solo["n"] == "4"


@pytest.mark.parametrize(
    "extra",
    [
        ["--ref", "nope"],
        ["--compare", "team-forget"],
        ["--compare", "team-forget:nope"],
        ["--compare", "a:b:c"],
    ],
)
def test_cli_rejects_bad_arm_arguments(
    experiment: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(experiment), *extra]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope")]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_directory_without_finished_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()
    assert main([str(tmp_path / "empty")]) == 2
    assert "no finished runs" in capsys.readouterr().err.lower()


def test_cli_plots_need_an_output_directory(
    experiment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(experiment), "--plots"]) == 2
    assert "--out" in capsys.readouterr().err


def test_module_exposes_the_summary_api() -> None:
    # analyze is the documented entry point for RunSummary and friends
    for name in ("RunSummary", "summarize_run", "load_summaries", "RoundScores", "TurnRecord"):
        assert hasattr(analyze, name)
