"""Materializer.prepare_epoch: per-agent, per-epoch workspaces seeded from the archive."""

from __future__ import annotations

from pathlib import Path

import pytest

from lethe.workspace.materializer import CONTEXT_DIR, TASK_FILE, Materializer


class FakeArchive:
    """Stands in for `Archive`: records `checkout` calls and drops a marker file."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, object]] = []

    def checkout(self, sha, dest, *, keep=None):
        self.calls.append((sha, Path(dest), keep))
        Path(dest).mkdir(parents=True)
        (Path(dest) / "seed.txt").write_text(sha)


def test_module_constants_are_the_documented_names():
    assert CONTEXT_DIR == ".context"
    assert TASK_FILE == "TASK.md"


def test_constructing_a_materializer_touches_no_filesystem(tmp_path):
    ws_root = tmp_path / "workspaces"

    Materializer(FakeArchive(), ws_root)

    assert not ws_root.exists()


def test_prepare_epoch_checks_the_seed_out_into_agent_epoch_dir(tmp_path):
    archive = FakeArchive()
    materializer = Materializer(archive, tmp_path / "ws")

    path = materializer.prepare_epoch("agent-1", 3, "a" * 40)

    assert path == tmp_path / "ws" / "agent-1" / "e3"
    assert archive.calls == [("a" * 40, path, None)]
    assert (path / "seed.txt").read_text() == "a" * 40


def test_prepare_epoch_forwards_the_keep_filter(tmp_path):
    archive = FakeArchive()
    keep = lambda p: p.endswith(".py")  # noqa: E731

    Materializer(archive, tmp_path / "ws").prepare_epoch("a1", 1, "b" * 40, keep=keep)

    assert archive.calls[0][2] is keep


def test_prepare_epoch_refuses_to_reuse_a_workspace_without_calling_the_archive(tmp_path):
    archive = FakeArchive()
    materializer = Materializer(archive, tmp_path / "ws")
    path = materializer.prepare_epoch("a1", 1, "c" * 40)
    (path / "agent_work.py").write_text("my edits")

    with pytest.raises(FileExistsError):
        materializer.prepare_epoch("a1", 1, "c" * 40)

    assert len(archive.calls) == 1
    assert (path / "agent_work.py").read_text() == "my edits"


def test_prepare_epoch_refuses_a_preexisting_empty_directory(tmp_path):
    archive = FakeArchive()
    (tmp_path / "ws" / "a1" / "e1").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        Materializer(archive, tmp_path / "ws").prepare_epoch("a1", 1, "d" * 40)

    assert archive.calls == []


def test_agents_and_epochs_get_separate_workspaces(tmp_path):
    materializer = Materializer(FakeArchive(), tmp_path / "ws")

    paths = {
        materializer.prepare_epoch("a1", 1, "e" * 40),
        materializer.prepare_epoch("a1", 2, "e" * 40),
        materializer.prepare_epoch("a2", 1, "e" * 40),
    }

    assert len(paths) == 3
    assert all(p.is_dir() for p in paths)


@pytest.mark.parametrize(
    "agent", ["", ".", "..", "a/b", "../x", "/abs", "x y", "x\n", "café", "a\x00b"]
)
def test_prepare_epoch_rejects_unsafe_agent_names(tmp_path, agent):
    archive = FakeArchive()

    with pytest.raises(ValueError, match="agent"):
        Materializer(archive, tmp_path / "ws").prepare_epoch(agent, 1, "f" * 40)

    assert archive.calls == []
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("epoch", [0, -1, "1", "../x", 1.0, True])
def test_prepare_epoch_requires_a_positive_integer_epoch(tmp_path, epoch):
    archive = FakeArchive()

    with pytest.raises(ValueError, match="epoch"):
        Materializer(archive, tmp_path / "ws").prepare_epoch("a1", epoch, "f" * 40)

    assert archive.calls == []


def test_prepare_epoch_seeds_the_workspace_from_a_real_archive(
    archive, make_tree, take, tree_contents, tmp_path, seed_files
):
    seed = take(make_tree(tmp_path / "seed_wd", seed_files))
    materializer = Materializer(archive, tmp_path / "ws")

    path = materializer.prepare_epoch(
        "agent-1", 2, seed.sha, keep=lambda p: not p.startswith("data/")
    )

    assert path == tmp_path / "ws" / "agent-1" / "e2"
    assert tree_contents(path) == {
        k: v.encode() for k, v in seed_files.items() if not k.startswith("data/")
    }
