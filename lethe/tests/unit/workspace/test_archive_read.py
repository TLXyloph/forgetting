"""Archive: reading committed trees back (`files`, `read_file`, `diff`)."""

from __future__ import annotations

import unicodedata

import pytest

from lethe.workspace.archive import Archive
from lethe.workspace.paths import PathFilter

# --- files / read_file -------------------------------------------------------------------------


def test_files_maps_every_path_to_its_git_blob_id(
    archive, make_tree, take, tmp_path, seed_files, git
):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    files = archive.files(snap.sha)

    assert files == {
        line.split("\t", 1)[1]: line.split()[2]
        for line in git("ls-tree", "-r", snap.sha).splitlines()
    }
    assert len(files) == len(seed_files)


def test_read_file_returns_the_committed_bytes(archive, make_tree, take, tmp_path, seed_files):
    first = take(make_tree(tmp_path / "wd", seed_files))
    (tmp_path / "wd" / "solver.py").write_text("changed\n")
    second = take(tmp_path / "wd", parent=first)

    assert archive.read_file(first.sha, "solver.py") == seed_files["solver.py"].encode()
    assert archive.read_file(second.sha, "solver.py") == b"changed\n"
    assert archive.read_file(second.sha, "data/config.json") == b'{"n": 3}\n'


def test_read_file_round_trips_large_binary_content(archive, make_tree, take, tmp_path):
    big = bytes(range(256)) * 12_000  # about 3 MB
    snap = take(make_tree(tmp_path / "wd", {"big.bin": big, "small.txt": "x"}))

    assert archive.read_file(snap.sha, "big.bin") == big


def test_read_file_finds_decomposed_unicode_names(archive, make_tree, take, tmp_path):
    name = unicodedata.normalize("NFD", "src/café.py")  # e + combining acute accent
    snap = take(make_tree(tmp_path / "wd", {name: "X = 1\n"}))
    stored = next(iter(archive.files(snap.sha)))

    assert archive.read_file(snap.sha, stored) == b"X = 1\n"
    assert "X = 1" in archive.diff(None, snap.sha)  # the pathspec must find it too


def test_read_file_of_a_missing_path_is_a_runtime_error(
    archive, make_tree, take, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    with pytest.raises(RuntimeError, match="cat-file"):
        archive.read_file(snap.sha, "nope.py")


# --- diff --------------------------------------------------------------------------------------


def edited_pair(make_tree, take, tmp_path, seed_files):
    """Two snapshots: the second edits one judged and one non-judged file."""
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "solver.py").write_text("def solve(x):\n    return x + 1\n")
    (wd / "evaluate.py").write_text("print('score: 2')\n")
    return first, take(wd, parent=first)


def test_diff_shows_judged_files_only_by_default(make_tree, take, archive, tmp_path, seed_files):
    first, second = edited_pair(make_tree, take, tmp_path, seed_files)

    text = archive.diff(first.sha, second.sha)

    assert "diff --git a/solver.py b/solver.py" in text
    assert "-    return x\n+    return x + 1\n" in text
    assert "evaluate.py" not in text
    assert "\x1b[" not in text  # no colour codes


def test_diff_can_include_every_file(make_tree, take, archive, tmp_path, seed_files):
    first, second = edited_pair(make_tree, take, tmp_path, seed_files)

    text = archive.diff(first.sha, second.sha, judged_only=False)

    assert "a/solver.py" in text
    assert "a/evaluate.py" in text


def test_diff_without_a_base_is_against_the_empty_tree(
    make_tree, take, archive, tmp_path, seed_files
):
    _, second = edited_pair(make_tree, take, tmp_path, seed_files)

    judged_text = archive.diff(None, second.sha)
    full_text = archive.diff(None, second.sha, judged_only=False)

    assert judged_text.count("new file mode") == 2  # solver.py and src/util.py
    assert "b/src/util.py" in judged_text
    assert "evaluate.py" not in judged_text
    assert full_text.count("new file mode") == len(seed_files)
    assert "b/data/config.json" in full_text


def test_diff_spans_non_adjacent_commits(make_tree, take, archive, tmp_path, seed_files):
    first, second = edited_pair(make_tree, take, tmp_path, seed_files)
    (tmp_path / "wd" / "src/util.py").write_text("VALUE = 2\n")
    third = take(tmp_path / "wd", parent=second)

    text = archive.diff(first.sha, third.sha)

    assert "return x + 1" in text
    assert "+VALUE = 2" in text


def test_diff_is_empty_when_nothing_judged_changed(make_tree, take, archive, tmp_path, seed_files):
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "evaluate.py").write_text("print('only a non-judged file changed')\n")
    second = take(wd, parent=first)

    assert archive.diff(first.sha, second.sha) == ""
    assert archive.diff(first.sha, first.sha, judged_only=False) == ""
    assert "evaluate.py" in archive.diff(first.sha, second.sha, judged_only=False)


def test_diff_is_empty_when_no_file_is_judged(make_tree, take, archive, tmp_path):
    snap = take(make_tree(tmp_path / "wd", {"notes.txt": "a\n", "data.csv": "1,2\n"}))

    assert archive.diff(None, snap.sha) == ""
    assert "notes.txt" in archive.diff(None, snap.sha, judged_only=False)


def test_diff_handles_paths_with_spaces_and_unicode(make_tree, take, archive, tmp_path):
    snap = take(make_tree(tmp_path / "wd", {"src/my file.py": "A = 1\n", "src/ünï.py": "B = 2\n"}))

    text = archive.diff(None, snap.sha)

    assert "+A = 1" in text
    assert "+B = 2" in text


def test_diff_matches_judged_paths_literally_not_as_globs(git_dir, make_tree, tmp_path):
    archive = Archive(git_dir, judged=PathFilter(["docs[1].md"]), exclude=PathFilter([]))
    wd = make_tree(tmp_path / "wd", {"docs[1].md": "old\n", "docs1.md": "old\n"})
    first = archive.snapshot(wd, parent_sha=None, message="a", index_name="a")
    (wd / "docs[1].md").write_text("new\n")
    (wd / "docs1.md").write_text("new\n")  # a git glob `docs[1].md` would also match this
    second = archive.snapshot(wd, parent_sha=first.sha, message="b", index_name="a")

    text = archive.diff(first.sha, second.sha)

    assert "docs[1].md" in text
    assert "docs1.md" not in text


def test_diff_decodes_invalid_utf8_with_replacement(archive, make_tree, take, tmp_path):
    wd = make_tree(tmp_path / "wd", {"solver.py": b"# caf\xe9\nx = 1\n"})
    first = take(wd)
    (wd / "solver.py").write_bytes(b"# caf\xe9\nx = 2\n")
    second = take(wd, parent=first)

    text = archive.diff(first.sha, second.sha)

    assert "caf�" in text
    assert "+x = 2" in text


def test_diff_reports_binary_changes_without_dumping_them(
    archive, make_tree, take, tmp_path, seed_files
):
    wd = make_tree(tmp_path / "wd", {**seed_files, "weights.bin": b"\x00\x01\x02"})
    first = take(wd)
    (wd / "weights.bin").write_bytes(b"\x00\x09\x09")
    second = take(wd, parent=first)

    text = archive.diff(first.sha, second.sha)

    assert "Binary files" in text
    assert "weights.bin" in text


# --- memoized reads ----------------------------------------------------------------------------


def count_git_calls(archive, monkeypatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    real = archive._git

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(archive, "_git", counting)
    return calls


def test_repeated_reads_of_a_commit_run_git_once(
    make_tree, take, archive, tmp_path, seed_files, monkeypatch
):
    # Commits are immutable, and every agent turn re-reads the same lineage and peer cards: a
    # reader that shells out each time holds the orchestrator's event loop for seconds.
    first, second = edited_pair(make_tree, take, tmp_path, seed_files)
    calls = count_git_calls(archive, monkeypatch)

    once = (archive.diff(first.sha, second.sha), archive.files(second.sha))
    after_first = len(calls)
    again = (archive.diff(first.sha, second.sha), archive.files(second.sha))

    assert after_first > 0 and len(calls) == after_first
    assert again == once
    assert archive.diff(first.sha, second.sha, judged_only=False) != once[0]  # its own entry


def test_a_caller_cannot_change_what_later_reads_return(
    make_tree, take, archive, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    archive.files(snap.sha).clear()

    assert len(archive.files(snap.sha)) == len(seed_files)
