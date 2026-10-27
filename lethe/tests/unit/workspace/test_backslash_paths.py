"""Backslashes in file names are never paths: archives skip them and path helpers refuse them.

Regressions for two verified scenarios: a workspace file named `tests\\check.py` overwrote the
seed's `tests/check.py` in the judge tree (overlay judging bypassed for any wildcard editable
glob), and scratch files `notes` and `notes\\x` made the next epoch's checkout raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lethe.workspace.archive import Archive
from lethe.workspace.materializer import Materializer
from lethe.workspace.paths import PathFilter, UnsafePathError, safe_join, safe_relpath


@pytest.mark.parametrize("rel", ["a\\b", "tests\\check.py", "\\abs", "..\\x", "a/b\\c"])
def test_safe_relpath_refuses_backslashes_instead_of_rewriting_them(rel, tmp_path):
    with pytest.raises(UnsafePathError):
        safe_relpath(rel)
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, rel)


def test_a_backslash_name_cannot_overwrite_a_non_editable_judge_tree_file(tmp_path):
    archive = Archive(
        tmp_path / "archive.git", judged=PathFilter(["*.py"]), exclude=PathFilter([".context/**"])
    )
    seed = tmp_path / "seed"
    (seed / "tests").mkdir(parents=True)
    (seed / "solver.py").write_text("x = 1\n")
    (seed / "tests" / "check.py").write_text("REAL HIDDEN CHECK\n")
    base = archive.snapshot(seed, parent_sha=None, message="seed", index_name="seed")
    workspace = tmp_path / "ws"
    archive.checkout(base.sha, workspace)
    (workspace / "tests\\check.py").write_text("AGENT-CONTROLLED CHECK\n")

    cand = archive.snapshot(workspace, parent_sha=base.sha, message="t1", index_name="a1")

    assert sorted(archive.files(cand.sha)) == ["solver.py", "tests/check.py"]
    assert cand.changed_judged is False
    archive.checkout_overlay(base.sha, cand.sha, tmp_path / "tree")
    assert (tmp_path / "tree" / "tests" / "check.py").read_text() == "REAL HIDDEN CHECK\n"


def test_scratch_names_that_alias_each_other_do_not_break_the_next_epoch(tmp_path):
    archive = Archive(
        tmp_path / "archive.git",
        judged=PathFilter(["solver.py"]),
        exclude=PathFilter([".context/**", "TASK.md"]),
    )
    workspace = tmp_path / "ws1"
    workspace.mkdir()
    (workspace / "solver.py").write_text("x = 2\n")
    (workspace / "notes").write_text("scratch\n")
    (workspace / "notes\\x").write_text("more scratch\n")
    snap = archive.snapshot(workspace, parent_sha=None, message="t", index_name="a1")
    assert sorted(archive.files(snap.sha)) == ["notes", "solver.py"]

    epoch2 = Materializer(archive, tmp_path / "wsroot").prepare_epoch("a1", 2, snap.sha)

    assert (epoch2 / "notes").read_text() == "scratch\n"
    assert not any("\\" in p.name for p in Path(epoch2).iterdir())
