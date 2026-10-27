"""Workspaces get an empty git repository of their own, and a clean forget restarts from the seed.

Claude Code puts the enclosing git repository's branch, status and recent commit messages into an
agent's context. An empty repository at the workspace root stops that search, so the agent sees
only its own (untracked) files. lethe never commits to it, never renders into it and never
archives it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lethe.config import ALWAYS_EXCLUDE, TaskConfig
from lethe.domain.types import CandidateCard, ContextBundle
from lethe.workspace.archive import Archive
from lethe.workspace.materializer import Materializer


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t")
    env.update(GIT_COMMITTER_EMAIL="t@x")
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env, check=False
    )


def files_of(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def a_bundle() -> ContextBundle:
    current = CandidateCard("c0000", None, 1.0, True, 0, "seed", "")
    return ContextBundle("# Task\n", current, "fine", {}, (current,), ())


def test_a_new_workspace_is_an_empty_repository_that_hides_an_enclosing_one(
    archive, make_tree, take, tmp_path, seed_files
):
    outer = tmp_path / "outer"
    make_tree(outer, {"README.md": "the user's project\n"})
    assert git(outer, "init", "-q").returncode == 0
    git(outer, "add", "README.md")
    assert git(outer, "commit", "-q", "-m", "secret commit message").returncode == 0
    seed = take(make_tree(tmp_path / "seed", seed_files), index_name="seed")

    ws = Materializer(archive, outer / "runs" / "ws").prepare_epoch("a1", 1, seed.sha)

    assert (ws / ".git").is_dir()
    toplevel = git(ws, "rev-parse", "--show-toplevel").stdout.strip()
    assert Path(toplevel).resolve() == ws.resolve()
    assert git(ws, "rev-parse", "--verify", "-q", "HEAD").returncode != 0  # no commits at all
    assert "secret commit message" not in git(ws, "log", "--all").stdout
    status = git(ws, "status", "--porcelain", "--untracked-files=all").stdout
    assert "?? solver.py" in status and "README.md" not in status


def test_render_never_touches_the_workspace_repository(archive, make_tree, take, tmp_path):
    seed = take(make_tree(tmp_path / "seed", {"solver.py": "x = 1\n"}), index_name="seed")
    materializer = Materializer(archive, tmp_path / "ws")
    ws = materializer.prepare_epoch("a1", 1, seed.sha)
    before = files_of(ws / ".git")

    materializer.render(ws, a_bundle())
    materializer.render(ws, a_bundle())

    assert files_of(ws / ".git") == before and before


def test_snapshots_never_archive_the_workspace_repository(make_tree, take, tmp_path):
    task = TaskConfig("t", "d", tmp_path, tmp_path / "seed", ("*.py",), ALWAYS_EXCLUDE)
    assert task.archive_exclude(".git/HEAD") and task.archive_exclude(".git/objects/info/x")
    archive = Archive(tmp_path / "a.git", judged=task.judged, exclude=task.archive_exclude)
    make_tree(tmp_path / "seed", {"solver.py": "x = 1\n"})
    seed = archive.snapshot(tmp_path / "seed", parent_sha=None, message="s", index_name="seed")
    ws = Materializer(archive, tmp_path / "ws").prepare_epoch("a1", 1, seed.sha)
    (ws / "solver.py").write_text("x = 2\n")

    snap = archive.snapshot(ws, parent_sha=seed.sha, message="t", index_name="a1")

    assert sorted(archive.files(snap.sha)) == ["solver.py"]


def test_a_clean_forget_restarts_from_the_seed_with_the_best_judged_files(
    archive, make_tree, take, tmp_path, seed_files
):
    seed = take(make_tree(tmp_path / "seed", seed_files), index_name="seed")
    materializer = Materializer(archive, tmp_path / "ws")
    ws1 = materializer.prepare_epoch("a1", 1, seed.sha)
    (ws1 / "solver.py").write_text("def solve(x):\n    return 2 * x\n")
    (ws1 / "evaluate.py").write_text("print('score: 1')\n# idea 7 works\n")  # a seed file
    (ws1 / "notes.txt").write_text("pre-forget note\n")
    best = archive.snapshot(ws1, parent_sha=seed.sha, message="t1", index_name="a1")

    ws2 = materializer.prepare_epoch("a1", 2, best.sha, base_sha=seed.sha)

    assert (ws2 / "solver.py").read_text() == "def solve(x):\n    return 2 * x\n"
    assert (ws2 / "evaluate.py").read_text() == seed_files["evaluate.py"]
    assert not (ws2 / "notes.txt").exists()
    assert (ws2 / ".git").is_dir()


def test_prepare_epoch_with_a_base_uses_the_overlay_checkout(tmp_path):
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def checkout(self, sha, dest, *, keep=None):
            raise AssertionError("a clean epoch must not copy the candidate's whole tree")

        def checkout_overlay(self, base_sha, cand_sha, dest):
            self.calls.append((base_sha, cand_sha, Path(dest)))
            Path(dest).mkdir(parents=True)

    recorder = Recorder()
    path = Materializer(recorder, tmp_path / "ws").prepare_epoch(  # type: ignore[arg-type]
        "a1", 2, "b" * 40, base_sha="a" * 40
    )
    assert recorder.calls == [("a" * 40, "b" * 40, path)]
