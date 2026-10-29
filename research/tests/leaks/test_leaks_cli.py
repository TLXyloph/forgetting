"""Tests for `python -m research.leaks` and the leaks column of `python -m research.analyze`."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research import analyze
from research.analyze import build_report, render_report
from research.leaks import leak_counts, main

MakeRun = Callable[..., Any]


@pytest.fixture
def leaky(make_run: MakeRun, runs_root: Path) -> Path:
    """solo/rep1 writes $TMPDIR/new.py; solo/rep2 reads it and Claude Code's history (2 leaks)."""
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("solo", 2, agents=("a1",))
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=1)
    x.bash("a1", 1, "echo $TMPDIR", ts=2)
    y.bash("a1", 1, "cat $TMPDIR/new.py", ts=3)
    y.bash("a1", 1, "tail ~/.claude/history.jsonl", ts=4)
    return runs_root / "exp"


def runs_listing(text: str) -> dict[str, list[str]]:
    """The cells of each row of the report's per-run listing, by run."""
    listing = text[text.index("\nRuns\n") + 1 :].split("\n\n")[0]
    return {line.split()[0]: line.split() for line in listing.splitlines()[3:] if line.strip()}


# ---------------------------------------------------------------------------------------------
# python -m research.leaks
# ---------------------------------------------------------------------------------------------


def test_cli_prints_counts_by_kind_and_the_leaks(
    leaky: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(leaky)])

    out = capsys.readouterr().out
    assert code == 1
    header, _, rep1, rep2 = out.splitlines()[1:5]
    assert header.split() == [
        "run",
        "leaks",
        "cross-run",
        "claude-logs",
        "temp-write",
        "temp-touch",
    ]
    assert rep1.split() == ["solo/rep1", "0", "0", "0", "1", "1"]
    assert rep2.split() == ["solo/rep2", "2", "1", "1", "0", "0"]
    assert "cross-run: read $TMPDIR/new.py (last written by solo/rep1 a1 e1)" in out
    assert "cat $TMPDIR/new.py" in out
    assert "claude-logs: read" in out
    assert "2 leaks in 1 of 2 runs" in out


def test_cli_exits_0_without_leaks(make_run: MakeRun, capsys: pytest.CaptureFixture[str]) -> None:
    t = make_run("solo", 1)
    t.read("a1", 1, t.workspace("a1", 1) / "kernel.py", ts=1)
    t.bash("a1", 1, "ls $TMPDIR", ts=2)

    code = main([str(t.run_dir.parent.parent)])

    out = capsys.readouterr().out
    assert code == 0
    assert "no leaks in 1 run" in out


def test_cli_scans_one_run_against_the_writes_of_its_experiment(
    leaky: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(leaky / "solo" / "rep2")])

    out = capsys.readouterr().out
    assert code == 1
    assert "solo/rep2" in out
    assert "solo/rep1 " not in out.split("\n\n")[0]  # the table lists the scanned run only
    assert "last written by solo/rep1 a1 e1" in out


def test_cli_all_experiments(make_run: MakeRun, runs_root: Path, capsys: Any) -> None:
    make_run("solo", 1, experiment="e1").bash("a1", 1, "cp k.py $TMPDIR/new.py", ts=1)
    make_run("solo", 1, experiment="e2").bash("a1", 1, "cat $TMPDIR/new.py", ts=2)

    assert main([str(runs_root / "e2"), "--all-experiments"]) == 1
    assert "last written by e1/solo/rep1 a1 e1" in capsys.readouterr().out


def test_cli_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope")]) == 2
    assert "error" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------------------------


def test_leak_counts_per_run(leaky: Path) -> None:
    assert leak_counts(leaky) == {"solo/rep1": 0, "solo/rep2": 2}


def test_the_analysis_lists_leaks_per_run_and_explains_them(
    leaky: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"

    assert analyze.main([str(leaky), "--out", str(out_dir)]) == 0

    text = capsys.readouterr().out
    rows = runs_listing(text)
    assert rows["solo/rep1"][-3:] == ["0", "0", "complete"]  # leaks, unplanned, ended
    assert rows["solo/rep2"][-3:] == ["2", "0", "complete"]
    assert "\nleaks: " in text and "research.leaks" in text
    with (out_dir / "runs.csv").open(newline="", encoding="utf-8") as fh:
        leaks = {(r["arm"], r["rep"]): r["leaks"] for r in csv.DictReader(fh)}
    assert leaks == {("solo", "1"): "0", ("solo", "2"): "2"}


def test_the_analysis_has_no_leaks_note_without_leaks(
    make_run: MakeRun, runs_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run("solo", 1).bash("a1", 1, "echo $TMPDIR", ts=1)

    assert analyze.main([str(runs_root / "exp")]) == 0

    text = capsys.readouterr().out
    assert runs_listing(text)["solo/rep1"][-3] == "0"
    assert "\nleaks: " not in text


def test_a_report_built_without_a_scan_shows_no_count(make_summary: Callable[..., Any]) -> None:
    text = render_report(build_report([make_summary()], experiment="x"))
    assert runs_listing(text)["solo/rep1"][-3] == "-"
    assert "\nleaks: " not in text
