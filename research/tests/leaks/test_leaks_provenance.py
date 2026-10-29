"""Tests for research.leaks: tracing reads of shared temp files to whoever last wrote them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from research.leaks import LeakFinding, scan_experiment, scan_run

MakeRun = Callable[..., Any]


def leaks(findings: Sequence[LeakFinding]) -> list[LeakFinding]:
    return [f for f in findings if not f.benign]


def test_a_read_of_a_file_another_run_wrote_is_cross_run(
    make_run: MakeRun, runs_root: Path
) -> None:
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("parallel", 1, peers="none")
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=10)
    y.bash("a2", 1, "cat $TMPDIR/new.py", ts=20)

    found = scan_experiment(runs_root / "exp")

    assert list(found) == ["parallel/rep1", "solo/rep1"]
    (leak,) = leaks(found["parallel/rep1"])
    assert (leak.kind, leak.agent, leak.path, leak.access) == (
        "cross-run",
        "a2",
        "$TMPDIR/new.py",
        "read",
    )
    assert "solo/rep1 a1 e1" in leak.detail
    assert [f.kind for f in found["solo/rep1"]] == ["temp-write"]


def test_a_read_before_any_write_is_not_attributed_to_the_later_writer(
    make_run: MakeRun, runs_root: Path
) -> None:
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("solo", 2, agents=("a1",))
    y.bash("a1", 1, "cat $TMPDIR/new.py", ts=5)
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=10)

    (leak,) = leaks(scan_experiment(runs_root / "exp")["solo/rep2"])

    assert leak.kind == "shared-temp"


def test_the_last_writer_decides(make_run: MakeRun, runs_root: Path) -> None:
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("solo", 2, agents=("a1",))
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=1)
    y.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=2)
    y.bash("a1", 1, "python $TMPDIR/new.py", ts=3)  # its own file: benign
    x.bash("a1", 1, "echo '# x' >> $TMPDIR/new.py", ts=4)
    y.bash("a1", 1, "python $TMPDIR/new.py", ts=5)  # x changed it since

    reads = [f for f in scan_experiment(runs_root / "exp")["solo/rep2"] if f.access == "read"]

    assert [f.kind for f in reads] == ["own-temp", "cross-run"]


def test_a_read_of_a_file_written_before_a_forget_is_cross_forget(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 1, "cp kernel.py /tmp/claude-501/keep.py", ts=1)
    t.call("a1", 2, "Read", {"file_path": "/private/tmp/claude-501/keep.py"}, ts=50)

    findings = scan_run(t.run_dir)

    assert [(f.epoch, f.kind) for f in findings] == [(1, "temp-write"), (2, "cross-forget")]
    assert "a1 e1" in findings[1].detail


def test_a_read_of_a_file_another_agent_wrote_is_cross_agent_flagged_when_isolated(
    make_run: MakeRun, runs_root: Path
) -> None:
    isolated = make_run("parallel", 1, peers="none")
    team = make_run("team", 1, peers="all")
    isolated.bash("a1", 1, "python gen.py > $TMPDIR/k.py", ts=1)
    isolated.bash("a2", 1, "python $TMPDIR/k.py", ts=2)
    team.bash("a1", 1, "python gen.py > $TMPDIR/k.py", ts=3)
    team.bash("a2", 1, "python $TMPDIR/k.py", ts=4)

    found = scan_experiment(runs_root / "exp")

    (alone,) = leaks(found["parallel/rep1"])
    (shared,) = leaks(found["team/rep1"])
    assert alone.kind == shared.kind == "cross-agent"
    assert "a1 e1" in alone.detail
    assert "peers = none" in alone.detail
    assert "peers" not in shared.detail


def test_a_glob_read_is_traced_to_the_files_it_matches(make_run: MakeRun, runs_root: Path) -> None:
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("solo", 2, agents=("a1",))
    x.bash("a1", 1, "cp a.py $TMPDIR/cand_a.py", ts=1)
    y.bash("a1", 1, "cp b.py $TMPDIR/cand_b.py", ts=2)
    y.bash("a1", 1, "head $TMPDIR/cand_b*.py", ts=3)
    y.bash("a1", 1, "head $TMPDIR/cand_*.py", ts=4)

    reads = [f for f in scan_experiment(runs_root / "exp")["solo/rep2"] if f.access == "read"]

    assert [(f.kind, f.path) for f in reads] == [
        ("own-temp", "$TMPDIR/cand_b*.py"),
        ("cross-run", "$TMPDIR/cand_*.py"),
    ]


def test_scan_run_traces_the_writes_of_the_runs_it_is_given(make_run: MakeRun) -> None:
    x = make_run("solo", 1, agents=("a1",))
    y = make_run("solo", 2, agents=("a1",))
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=1)
    y.bash("a1", 1, "cat $TMPDIR/new.py", ts=2)

    assert [f.kind for f in scan_run(y.run_dir)] == ["shared-temp"]
    assert [f.kind for f in scan_run(y.run_dir, writers=[x.run_dir, y.run_dir])] == ["cross-run"]


def test_writers_in_other_experiments_count_when_asked(make_run: MakeRun, runs_root: Path) -> None:
    x = make_run("solo", 1, agents=("a1",), experiment="e1")
    y = make_run("solo", 1, agents=("a1",), experiment="e2")
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=1)
    y.bash("a1", 1, "cat $TMPDIR/new.py", ts=2)

    alone = scan_experiment(runs_root / "e2")
    across = scan_experiment(runs_root / "e2", all_experiments=True)

    assert [f.kind for f in alone["solo/rep1"]] == ["shared-temp"]
    assert list(across) == ["solo/rep1"]  # only the experiment's own runs are reported
    (leak,) = across["solo/rep1"]
    assert leak.kind == "cross-run"
    assert "e1/solo/rep1 a1 e1" in leak.detail


def test_parked_incomplete_runs_are_scanned_too(make_run: MakeRun, runs_root: Path) -> None:
    x = make_run("solo", 1, agents=("a1",))
    x.bash("a1", 1, "cp kernel.py $TMPDIR/new.py", ts=1)
    x.run_dir.rename(x.run_dir.with_name("rep1.incomplete-20270101T000000Z"))
    y = make_run("solo", 1, agents=("a1",))
    y.bash("a1", 1, "cat $TMPDIR/new.py", ts=2)
    make_run("solo", 10, agents=("a1",))

    found = scan_experiment(runs_root / "exp")

    assert list(found) == ["solo/rep1", "solo/rep1.incomplete-20270101T000000Z", "solo/rep10"]
    assert [f.kind for f in found["solo/rep1"]] == ["cross-run"]
