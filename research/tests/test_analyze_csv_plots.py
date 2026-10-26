"""Tests for research.analyze: the CSV files and the optional plots."""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from research.analyze import build_report, main, write_csvs
from research.summary import load_summaries


@pytest.fixture
def experiment(tmp_path: Path, build_analysis_experiment: Callable[..., Path]) -> Path:
    return build_analysis_experiment(tmp_path / "paper-toy-mis")


# ---------------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_write_csvs_creates_one_file_per_table(experiment: Path, tmp_path: Path) -> None:
    report = build_report(
        load_summaries(experiment), experiment="x", compares=[("team-forget", "parallel")]
    )
    out = tmp_path / "out"

    written = write_csvs(report, out)

    assert {p.name for p in written} == {
        "runs.csv",
        "table1.csv",
        "table2.csv",
        "anchoring_traces.csv",
        "comparisons.csv",
    }
    assert all(p.parent == out and p.is_file() for p in written)


def test_table_1_csv_contents(experiment: Path, tmp_path: Path) -> None:
    report = build_report(load_summaries(experiment), experiment="x")
    write_csvs(report, tmp_path)

    rows = {r["arm"]: r for r in read_csv(tmp_path / "table1.csv")}

    assert float(rows["solo"]["mean"]) == pytest.approx(110.0)
    assert float(rows["solo"]["sd"]) == pytest.approx(10.0)
    assert rows["solo"]["is_reference"] == "True"
    assert rows["solo"]["mean_change_pct"] == ""  # None is a blank cell
    assert float(rows["team-forget"]["mean_change_pct"]) == pytest.approx(100 * 40 / 110)
    assert float(rows["team-forget"]["best_of_runs"]) == 160.0
    assert rows["parallel"]["n"] == "3"


def test_table_2_csv_contents(experiment: Path, tmp_path: Path) -> None:
    write_csvs(build_report(load_summaries(experiment), experiment="x"), tmp_path)

    rows = read_csv(tmp_path / "table2.csv")

    assert [(r["arm"], r["round"], r["n"]) for r in rows] == [
        ("team-forget", "1", "3"),
        ("team-forget", "2", "3"),
    ]
    assert float(rows[0]["mean_best"]) == pytest.approx(125.0)  # half-way points 125, 130, 120


def test_runs_csv_has_one_row_per_run(experiment: Path, tmp_path: Path) -> None:
    write_csvs(build_report(load_summaries(experiment), experiment="x"), tmp_path)

    rows = read_csv(tmp_path / "runs.csv")

    assert len(rows) == 9
    first = next(r for r in rows if r["arm"] == "solo" and r["rep"] == "1")
    assert float(first["final_best"]) == 110.0
    assert float(first["seed_score"]) == 100.0
    assert float(first["spend_usd"]) == pytest.approx(1.5)
    assert first["proposals"] == "4"
    assert first["team_forgets"] == "0"


def test_traces_csv_contents(experiment: Path, tmp_path: Path) -> None:
    write_csvs(build_report(load_summaries(experiment), experiment="x"), tmp_path)

    rows = read_csv(tmp_path / "anchoring_traces.csv")

    assert {r["arm"] for r in rows} == {"solo", "parallel", "team-forget"}
    assert set(rows[0]) >= {
        "arm",
        "bin_start",
        "bin_end",
        "n_turns",
        "median_lines_changed",
        "median_thinking_tokens",
        "median_output_tokens",
        "empty_rate",
        "plateau_rate",
    }
    # solo turns end at spend 0.5, 0.75, 1.25, 1.5; the default bin is 0.3 wide (budget 3.0 / 10)
    first, second, *_ = (r for r in rows if r["arm"] == "solo")
    assert float(first["bin_start"]) == pytest.approx(0.3)
    assert (first["n_turns"], first["median_lines_changed"]) == ("3", "80.0")  # 3 pooled reps
    assert float(first["median_thinking_tokens"]) == 300.0
    assert float(first["empty_rate"]) == 0.0
    assert float(second["bin_start"]) == pytest.approx(0.6)  # the plateau-claiming empty turn
    assert float(second["empty_rate"]) == 1.0
    assert float(second["plateau_rate"]) == 1.0


def test_comparisons_csv_contents(experiment: Path, tmp_path: Path) -> None:
    report = build_report(
        load_summaries(experiment),
        experiment="x",
        compares=[("team-forget", "parallel")],
        n_boot=500,
    )
    write_csvs(report, tmp_path)

    (row,) = read_csv(tmp_path / "comparisons.csv")

    assert (row["arm_a"], row["arm_b"], row["n_a"], row["n_b"]) == (
        "team-forget",
        "parallel",
        "3",
        "3",
    )
    assert float(row["diff"]) == pytest.approx(150.0 - 130.0)
    assert 0.0 <= float(row["p"]) <= 1.0
    assert float(row["ci_low"]) < float(row["diff"]) < float(row["ci_high"])


def test_no_comparisons_csv_without_comparisons(experiment: Path, tmp_path: Path) -> None:
    written = write_csvs(build_report(load_summaries(experiment), experiment="x"), tmp_path)
    assert "comparisons.csv" not in {p.name for p in written}


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------


def test_missing_matplotlib_is_a_one_line_notice_not_an_error(
    experiment: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` fail
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    out_dir = tmp_path / "o"

    code = main([str(experiment), "--plots", "--out", str(out_dir)])

    captured = capsys.readouterr()
    assert code == 0
    lines = (captured.out + captured.err).splitlines()
    # (the "wrote <path>" lines may mention matplotlib too: pytest puts the test name in tmp paths)
    notice = [line for line in lines if "matplotlib" in line and not line.startswith("wrote ")]
    assert len(notice) == 1
    assert "skipping plots" in notice[0]
    assert (out_dir / "table1.csv").is_file()  # the tables are still written
    assert not list(out_dir.glob("*.png"))


def fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a stand-in `matplotlib.pyplot` whose figures 'save' a stub PNG."""
    plt = MagicMock(name="pyplot")

    def make_figure(*args: Any, **kwargs: Any) -> tuple[MagicMock, Any]:
        fig = MagicMock(name="figure")
        fig.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"\x89PNG stub")
        axes = MagicMock(name="axes")
        return fig, axes

    plt.subplots.side_effect = make_figure
    mpl = MagicMock(name="matplotlib")
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return plt


def test_plots_are_written_when_matplotlib_is_available(
    experiment: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plt = fake_matplotlib(monkeypatch)
    out_dir = tmp_path / "o"

    code = main([str(experiment), "--plots", "--out", str(out_dir)])

    text = capsys.readouterr().out
    assert code == 0
    assert (out_dir / "best_vs_spend.png").is_file()
    assert (out_dir / "anchoring_traces.png").is_file()
    assert "best_vs_spend.png" in text
    plt.close.assert_called()  # figures are released


def test_headless_backend_is_selected_before_plotting(
    experiment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_matplotlib(monkeypatch)
    main([str(experiment), "--plots", "--out", str(tmp_path / "o")])
    sys.modules["matplotlib"].use.assert_called_with("Agg")  # type: ignore[attr-defined]


def test_real_matplotlib_writes_valid_png_files(experiment: Path, tmp_path: Path) -> None:
    """Runs only where the `plots` extra is installed (skipped otherwise)."""
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "o"

    assert main([str(experiment), "--plots", "--out", str(out_dir)]) == 0

    for name in ("best_vs_spend.png", "anchoring_traces.png"):
        data = (out_dir / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 5_000  # an actual drawing, not an empty canvas
