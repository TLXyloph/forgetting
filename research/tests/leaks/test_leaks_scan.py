"""Tests for research.leaks: which tool calls of one run name a path outside the agent's own
workspace, and how they are classified.

Provenance across runs (cross-run, cross-agent, cross-forget) is in test_leaks_provenance.py.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research.leaks import BENIGN_KINDS, KINDS, LEAK_KINDS, UNVERIFIED_KINDS, scan_run

MakeRun = Callable[..., Any]


def slug(path: Path, *, strict: bool = False) -> str:
    """Claude Code's project directory name for a working directory."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path)) if strict else str(path).replace("/", "-")


def test_leak_unverified_and_benign_kinds_partition_the_kinds() -> None:
    assert BENIGN_KINDS.isdisjoint(LEAK_KINDS) and UNVERIFIED_KINDS.isdisjoint(LEAK_KINDS)
    assert BENIGN_KINDS.isdisjoint(UNVERIFIED_KINDS)
    assert set(KINDS) == BENIGN_KINDS | LEAK_KINDS | UNVERIFIED_KINDS
    assert {"temp-write", "temp-touch", "own-temp", "claude-output"} == BENIGN_KINDS
    # a temp file no scanned agent was seen writing: no evidence that data crossed
    assert {"shared-temp"} == UNVERIFIED_KINDS
    assert {"cross-run", "cross-agent", "cross-forget"} < LEAK_KINDS
    assert {"other-workspace", "run-state", "other-run", "claude-logs"} < LEAK_KINDS


def test_an_unverified_read_is_reported_but_not_counted(make_run: MakeRun) -> None:
    t = make_run("parallel", 1, peers="none")
    t.bash("a1", 1, "mkdir -p sw && cp $TMPDIR/sw.py sw/sw.py 2>/dev/null; ls sw", ts=1, result="")

    (finding,) = scan_run(t.run_dir)

    assert finding.kind == "shared-temp" and not finding.counts and not finding.benign


def test_shell_keywords_are_not_commands_or_paths(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash(
        "a1", 1,
        "cd /tmp/claude-501/s/u/tasks; until grep -q done b1x.output; do sleep 5; done",
        ts=1,
    )  # fmt: skip

    assert all(f.benign for f in scan_run(t.run_dir))


# ---------------------------------------------------------------------------------------------
# inside and outside the workspace
# ---------------------------------------------------------------------------------------------


def test_calls_inside_the_own_workspace_are_not_findings(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    ws = t.workspace("a1", 1)
    t.read("a1", 1, ws / "kernel.py", ts=1)
    t.call("a1", 1, "Write", {"file_path": str(ws / "new.py"), "content": "x = 1"}, ts=2)
    t.call("a1", 1, "Edit", {"file_path": str(ws / "new.py"), "old_string": "1"}, ts=3)
    t.call("a1", 1, "Glob", {"pattern": "**/*.py", "path": str(ws)}, ts=4)
    t.call("a1", 1, "Grep", {"pattern": "def ", "path": "."}, ts=5)
    t.bash("a1", 1, f"cd {ws} && python bench.py > out.txt 2>/dev/null && cat ./out.txt", ts=6)
    t.bash("a1", 1, "ls sub/../other.py; python3 -c 'import sys' /usr/lib/x.py", ts=7)

    assert scan_run(t.run_dir) == []


def test_a_relative_path_into_another_workspace_is_other_workspace(make_run: MakeRun) -> None:
    t = make_run("parallel", 1, peers="none")
    t.bash("a1", 1, "cat ../../a2/e1/x.py", ts=1)
    t.bash("a1", 2, "diff kernel.py ./../e1/kernel.py", ts=2)  # its own forgotten epoch
    t.read("a2", 1, t.workspace("a1", 1) / "kernel.py", ts=3)

    findings = scan_run(t.run_dir)

    assert [(f.agent, f.epoch, f.kind, f.access) for f in findings] == [
        ("a1", 1, "other-workspace", "read"),
        ("a1", 2, "other-workspace", "read"),
        ("a2", 1, "other-workspace", "read"),
    ]
    assert findings[0].path.endswith("/runs/exp/parallel/rep1/ws/a2/e1/x.py")
    assert findings[1].path.endswith("/runs/exp/parallel/rep1/ws/a1/e1/kernel.py")
    assert not any(f.benign for f in findings)


def test_the_runs_log_archive_transcripts_and_index_are_run_state(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    t.read("a1", 1, t.run_dir / "events.jsonl", ts=1)
    t.bash("a1", 1, f"git --git-dir={t.run_dir}/archive.git log --oneline", ts=2)
    t.call("a1", 1, "Grep", {"pattern": "score", "path": str(t.run_dir / "transcripts")}, ts=3)
    t.bash("a1", 1, "ls ../../../index", ts=4)  # a listing: names, no data, so no finding

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.access) for f in findings] == [("run-state", "read")] * 3


def test_paths_in_other_runs_under_the_runs_root_are_other_run(
    make_run: MakeRun, runs_root: Path
) -> None:
    x = make_run("solo", 1)
    y = make_run("solo", 2)
    y.read("a1", 1, x.workspace("a1", 1) / "kernel.py", ts=1)
    y.bash("a1", 1, f"ls {runs_root}/other-experiment", ts=2)  # a listing: no finding
    y.bash("a1", 1, "cat ../../../../rep1/events.jsonl", ts=3)

    assert [f.kind for f in scan_run(y.run_dir)] == ["other-run"] * 2


def test_the_runs_root_can_be_given(make_run: MakeRun, tmp_path: Path) -> None:
    t = make_run("solo", 1)
    t.bash("a1", 1, f"cat {tmp_path}/archive/old-run/kernel.py", ts=1)

    assert scan_run(t.run_dir) == []  # outside the runs root found by name: not a category
    found = scan_run(t.run_dir, runs_root=tmp_path)
    assert [f.kind for f in found] == ["other-run"]


def test_claude_codes_logs_in_the_home_directory_are_claude_logs(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    t.bash("a1", 1, "tail -5 ~/.claude/history.jsonl", ts=1)
    t.read("a1", 1, Path.home() / ".claude" / "projects" / "-Users-x-other" / "s.jsonl", ts=2)
    t.bash("a1", 1, "ls ${HOME}/.claude/projects", ts=3)

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.access) for f in findings] == [
        ("claude-logs", "read"),
        ("claude-logs", "read"),
        ("claude-logs", "touch"),
    ]
    assert findings[0].path == str(Path.home() / ".claude" / "history.jsonl")


@pytest.mark.parametrize("strict", [False, True])
def test_persisted_output_of_the_agents_own_session_is_benign(
    make_run: MakeRun, strict: bool
) -> None:
    t = make_run("solo", 1)
    projects = Path.home() / ".claude" / "projects"
    own = projects / slug(t.workspace("a1", 1), strict=strict) / "s1" / "tool-results" / "t1.txt"
    other = projects / slug(t.workspace("a2", 1), strict=strict) / "s9" / "tool-results" / "t9.txt"
    t.read("a1", 1, own, ts=1)
    t.read("a1", 1, other, ts=2)

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.benign) for f in findings] == [
        ("claude-output", True),
        ("claude-logs", False),
    ]


def test_claude_codes_task_output_in_the_temp_dir_is_benign(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    own = slug(t.workspace("a1", 1), strict=True)
    t.bash("a1", 1, f"tail /private/tmp/claude-501/{own}/s1/tasks/b1.output", ts=1)
    t.read("a1", 1, f"/tmp/claude-501/{own}/s1/tasks/b2.output", ts=2)

    assert [(f.kind, f.benign) for f in scan_run(t.run_dir)] == [("claude-output", True)] * 2


# ---------------------------------------------------------------------------------------------
# the shared temp directory
# ---------------------------------------------------------------------------------------------


def test_shared_temp_touches_writes_and_reads(make_run: MakeRun) -> None:
    t = make_run("solo", 1, agents=("a1",))
    t.bash("a1", 1, "echo $TMPDIR && ls -la ${TMPDIR}/", ts=1)
    t.bash("a1", 1, "cp kernel.py $TMPDIR/best.py", ts=2)
    t.bash("a1", 1, 'python bench.py > "$TMPDIR/out.txt" 2>&1 | tee -a $TMPDIR/log.txt', ts=3)
    t.bash("a1", 1, "cat $TMPDIR/best.py", ts=4)
    t.bash("a1", 1, "python3 /private/tmp/claude-501/stranger.py", ts=5)

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.access, f.path) for f in findings] == [
        ("temp-touch", "touch", "$TMPDIR"),
        ("temp-write", "write", "$TMPDIR/best.py"),
        ("temp-write", "write", "$TMPDIR/out.txt"),
        ("temp-write", "write", "$TMPDIR/log.txt"),
        ("own-temp", "read", "$TMPDIR/best.py"),
        ("shared-temp", "read", "$TMPDIR/stranger.py"),
    ]
    assert [f.benign for f in findings] == [True] * 5 + [False]
    assert "no scanned run wrote it" in findings[-1].detail


def test_copies_and_file_tools_in_the_temp_dir(make_run: MakeRun) -> None:
    t = make_run("solo", 1, agents=("a1",))
    t.bash("a1", 1, "cp kernel.py notes.md $TMPDIR/", ts=1)
    t.call("a1", 1, "Write", {"file_path": "/tmp/claude-501/w.py", "content": "x"}, ts=2)
    t.bash("a1", 1, "cp ${TMPDIR:-/tmp}/kernel.py kernel.py && ls $TMPDIR/kernel.py", ts=3)
    t.call("a1", 1, "Glob", {"pattern": "*.py", "path": "/tmp/claude-501"}, ts=4)

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.path) for f in findings] == [
        ("temp-write", "$TMPDIR/kernel.py"),
        ("temp-write", "$TMPDIR/notes.md"),
        ("temp-write", "$TMPDIR/w.py"),
        ("own-temp", "$TMPDIR/kernel.py"),
        ("temp-touch", "$TMPDIR/kernel.py"),
        ("temp-touch", "$TMPDIR"),
    ]


# ---------------------------------------------------------------------------------------------
# what is not counted, and what a finding says
# ---------------------------------------------------------------------------------------------


def test_failed_and_unanswered_calls_are_ignored(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    other = t.workspace("a2", 1) / "kernel.py"
    t.read("a1", 1, other, ts=1, is_error=True, result="Operation not permitted")
    t.bash("a1", 1, "cat ~/.claude/history.jsonl", ts=2, is_error=True)
    t.lines(
        "a1",
        1,
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"x9",'
        f'"name":"Read","input":{{"file_path":"{other}"}}}}]}}}}',
    )  # never answered

    assert scan_run(t.run_dir) == []


def test_malformed_lines_are_skipped(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    t.lines(
        "a1",
        1,
        '{"type":"assistant","message":{"content":[{"type":"tool_use"',
        "tool_use, but not json",
        '["tool_use"]',
        '{"type":"assistant","message":{"content":"tool_use"}}',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":["x"],"input":"no"}]}}',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"y","name":"Bash",'
        '"input":{"command":["cat","/tmp/claude-501/x"]}}]}}',
        '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":{"a":1}}]}}',
    )
    t.read("a1", 1, t.workspace("a2", 1) / "kernel.py", ts=1)
    with t.path("a1", 1).open("ab") as fh:
        fh.write(b'\xff\xfe{"tool_use" \x00 tool_result}\n')

    assert [f.kind for f in scan_run(t.run_dir)] == ["other-workspace"]


def test_a_finding_says_who_when_what_and_where(make_run: MakeRun) -> None:
    t = make_run("parallel", 2, peers="none")
    t.bash("a2", 3, "cat ../../a1/e3/kernel.py | head", ts=7, session="abc", turn=4)

    (finding,) = scan_run(t.run_dir)

    assert (finding.run, finding.agent, finding.epoch) == ("parallel/rep2", "a2", 3)
    assert finding.ts == pytest.approx(1_800_000_007.0)  # the tool_use line's timestamp
    assert finding.command == "cat ../../a1/e3/kernel.py | head"
    assert finding.transcript == "transcripts/a2/e3/abc-t4.jsonl"


def test_a_run_without_transcripts_has_no_findings(make_run: MakeRun) -> None:
    t = make_run("solo", 1)
    assert scan_run(t.run_dir) == []


# --- false alarms seen on E1's real transcripts ------------------------------------------------


def test_cd_within_a_command_is_followed(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 1, "cd .context/peers/branch3; diff best/k.py ../../../k.py", ts=1)
    t.bash("a1", 1, "cd .context && wc -l ../k.py", ts=2)

    assert scan_run(t.run_dir) == []


def test_the_shells_directory_carries_over_between_calls_in_a_turn(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 1, "cd .context/peers", ts=1)
    t.bash("a1", 1, "diff branch1/best/k.py ../../k.py", ts=2)
    t.bash("a1", 1, "cd ..", ts=3)  # leaves the workspace: Claude Code resets the directory
    t.bash("a1", 1, "cat k.py", ts=4)

    assert scan_run(t.run_dir) == []


def test_a_cd_out_of_the_workspace_that_reads_something_is_still_found(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 2, "cd ..; cat e1/notes.md", ts=1, result="tuned VK=12 ...")

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.access) for f in findings] == [("other-workspace", "read")]
    assert findings[0].path.endswith("/ws/a1/e1/notes.md")


def test_a_copy_into_the_temp_dir_is_a_write_despite_a_redirection(make_run: MakeRun) -> None:
    t = make_run("parallel", 1, peers="none")
    t.bash("a1", 1, "cp perf_takehome.py $TMPDIR/bak.py 2>/dev/null; echo ok", ts=1)

    assert [(f.kind, f.access) for f in scan_run(t.run_dir)] == [("temp-write", "write")]


def test_an_access_whose_result_says_it_failed_is_not_a_finding(make_run: MakeRun) -> None:
    t = make_run("parallel", 1, peers="none")
    failed = "python3: can't open file '/tmp/claude-501/ed.py': [Errno 2] No such file or directory"
    t.bash("a1", 1, "python3 /tmp/claude-501/ed.py; python3 tests/t.py", ts=1, result=failed)
    t.bash(
        "a1",
        1,
        "cat ../../a2/e1/x.py",
        ts=2,
        result="cat: ../../a2/e1/x.py: Operation not permitted",
    )

    assert scan_run(t.run_dir) == []


def test_a_bare_climb_names_a_directory_and_listing_outside_is_no_leak(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 1, "python - <<'EOF'\nimport sys; sys.path.insert(0, '..')\nEOF", ts=1)
    t.bash("a1", 1, "ls ..; ls ../..", ts=2, result="e1\ne2")
    t.bash("a1", 1, "grep -rn VK ..", ts=3, result="../e0/k.py: VK = 12")

    findings = scan_run(t.run_dir)

    assert [(f.kind, f.access) for f in findings] == [("other-workspace", "read")]


def test_reading_claude_codes_own_task_output_through_a_glob_is_benign(make_run: MakeRun) -> None:
    t = make_run("team-forget", 1)
    t.bash("a1", 1, "cat /private/tmp/claude-501/*/*/tasks/b9a49fcyi.output", ts=1)

    assert [f.kind for f in scan_run(t.run_dir)] == ["claude-output"]


def test_a_cd_into_the_own_workspace_by_absolute_path_stays_inside(make_run: MakeRun) -> None:
    t = make_run("clean-forget", 1)
    ws = t.workspace("a2", 1)
    t.bash("a2", 1, f"cd {ws} && python3 - <<'EOF'\nn = 3  # nodes 0 .. n-1\nEOF", ts=1)

    assert scan_run(t.run_dir) == []
