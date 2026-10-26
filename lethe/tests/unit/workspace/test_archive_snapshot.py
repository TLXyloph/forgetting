"""Archive: opening the repo and snapshotting working trees (real git, tmp_path)."""

from __future__ import annotations

import hashlib
import os
import unicodedata

import pytest

from lethe.workspace.archive import Archive

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def tree_of(git, sha: str) -> str:
    return git("rev-parse", f"{sha}^{{tree}}").strip()


# --- opening the archive -----------------------------------------------------------------------


def test_init_creates_a_bare_repo_and_the_index_dir(git_dir, git, judged, exclude):
    assert not git_dir.exists()

    Archive(git_dir, judged=judged, exclude=exclude)

    assert (git_dir / "HEAD").is_file()
    assert (git_dir.parent / "index").is_dir()
    assert git("rev-parse", "--is-bare-repository").strip() == "true"


def test_init_reopens_an_existing_archive_without_resetting_it(
    git_dir, judged, exclude, make_tree, tmp_path, seed_files
):
    first = Archive(git_dir, judged=judged, exclude=exclude)
    wd = make_tree(tmp_path / "wd", seed_files)
    snap = first.snapshot(wd, parent_sha=None, message="seed", index_name="a")

    second = Archive(git_dir, judged=judged, exclude=exclude)

    assert second.files(snap.sha) == first.files(snap.sha)


def test_init_makes_git_byte_exact_whatever_the_host_filesystem(archive, git):
    # `git init` on macOS turns on case folding and unicode precomposition; an archive needs
    # neither.
    assert git("config", "core.ignorecase").strip() == "false"
    assert git("config", "core.precomposeunicode").strip() == "false"


# --- what gets committed -----------------------------------------------------------------------


def test_snapshot_commits_every_regular_file_with_posix_relative_paths(
    archive, make_tree, take, tmp_path, seed_files, git
):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    files = archive.files(snap.sha)

    assert set(files) == set(seed_files)  # includes nested "src/util.py" and "data/config.json"
    for path, blob in files.items():
        assert git("hash-object", "--stdin", input=seed_files[path].encode()).strip() == blob


def test_snapshot_applies_the_exclude_filter(archive, make_tree, take, tmp_path, seed_files, git):
    excluded = {
        ".context/feedback.md": "rendered feedback",
        ".context/candidates/c1.diff": "rendered diff",
        "TASK.md": "rendered task",
        "__pycache__/solver.cpython-312.pyc": "top-level cache",
        "src/__pycache__/util.cpython-312.pyc": "nested cache",
    }
    wd = make_tree(
        tmp_path / "wd",
        {**seed_files, **excluded, "docs/TASK.md": "only the top-level TASK.md is excluded"},
    )

    snap = take(wd)

    assert set(archive.files(snap.sha)) == {*seed_files, "docs/TASK.md"}
    for content in excluded.values():  # excluded files are not even hashed into the store
        assert not git.has_blob(content)


def test_snapshot_skips_symlinks_and_special_files(
    archive, make_tree, take, tmp_path, seed_files, git
):
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the workdir")
    wd = make_tree(tmp_path / "wd", seed_files)
    (wd / "link_to_file.py").symlink_to(wd / "solver.py")
    (wd / "link_to_dir").symlink_to(wd / "src", target_is_directory=True)
    (wd / "link_outside").symlink_to(secret)
    (wd / "dangling").symlink_to(tmp_path / "does-not-exist")
    os.mkfifo(wd / "pipe")  # reading a FIFO would block forever

    snap = take(wd)

    assert set(archive.files(snap.sha)) == set(seed_files)
    assert not git.has_blob("outside the workdir")  # a link must never leak its target


def test_snapshot_never_archives_nested_git_metadata(
    archive, make_tree, take, tmp_path, seed_files, git
):
    metadata = {
        ".git/config": "[remote]\n\turl = https://user:token@example.com/x.git\n",
        ".git/objects/ab/cdef": "loose object",
        "vendored/.git": "gitdir: ../elsewhere\n",
    }
    wd = make_tree(tmp_path / "wd", {**seed_files, **metadata, "vendored/lib.py": "kept\n"})

    snap = take(wd)

    assert set(archive.files(snap.sha)) == {*seed_files, "vendored/lib.py"}
    for content in metadata.values():  # credentials in a nested repo must not land in the store
        assert not git.has_blob(content)


def test_snapshot_records_the_executable_bit(archive, make_tree, take, tmp_path, seed_files, git):
    wd = make_tree(tmp_path / "wd", {**seed_files, "run.sh": "#!/bin/sh\n"}, executable=["run.sh"])

    snap = take(wd)

    modes = {}
    for line in git("ls-tree", "-r", snap.sha).splitlines():
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]
    assert modes["run.sh"] == "100755"
    assert modes["solver.py"] == "100644"


def test_snapshot_keeps_file_bytes_exactly(archive, make_tree, take, tmp_path):
    payload = {
        "crlf.txt": b"a\r\nb\r\n",
        "binary.dat": bytes(range(256)) * 3,
        "empty.txt": b"",
        "no_newline.txt": b"last line without newline",
        "latin1.txt": b"caf\xe9\n",
    }
    snap = take(make_tree(tmp_path / "wd", payload))

    for path, content in payload.items():
        assert archive.read_file(snap.sha, path) == content


def test_snapshot_handles_awkward_file_names(archive, make_tree, take, tmp_path):
    names = [
        "with space.txt",
        'quo"te.txt',
        "tab\there.txt",
        '"leading-quote.txt',
        "-leading-dash.txt",
        "glob[1]*?.txt",
        "café.txt",
        unicodedata.normalize("NFD", "naïve.txt"),
        "dir with space/inner file.txt",
    ]
    snap = take(make_tree(tmp_path / "wd", {name: f"content of {name}" for name in names}))

    stored = archive.files(snap.sha)

    def fold(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    assert {fold(p) for p in stored} == {fold(n) for n in names}
    for path in stored:  # whatever bytes the filesystem kept, the same name reads back
        assert archive.read_file(snap.sha, path).startswith(b"content of ")


@pytest.mark.parametrize("name", ["bad\nname.py", "bad\rname.py"])
def test_snapshot_skips_names_it_cannot_hand_to_git(
    archive, make_tree, take, tmp_path, seed_files, name
):
    wd = make_tree(tmp_path / "wd", seed_files)
    try:
        (wd / name).write_text("x")
    except OSError:
        pytest.skip("filesystem refuses this file name")

    snap = take(wd)

    assert set(archive.files(snap.sha)) == set(seed_files)


def test_snapshot_of_an_empty_workdir_is_the_empty_tree(archive, take, tmp_path, git):
    empty = tmp_path / "empty"
    empty.mkdir()

    snap = take(empty)

    assert archive.files(snap.sha) == {}
    assert tree_of(git, snap.sha) == EMPTY_TREE
    assert snap.judged_key == hashlib.sha256(b"").hexdigest()
    assert (snap.lines_added, snap.lines_deleted, snap.files_changed) == (0, 0, 0)
    assert snap.changed_judged is True


def test_snapshot_rejects_a_workdir_that_does_not_exist(archive, take, tmp_path):
    with pytest.raises(FileNotFoundError):
        take(tmp_path / "missing")


def test_snapshot_rejects_a_workdir_that_is_a_file(archive, take, tmp_path):
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")

    with pytest.raises(NotADirectoryError):
        take(not_a_dir)


def test_snapshot_leaves_the_workdir_untouched(archive, make_tree, take, tmp_path, seed_files):
    wd = make_tree(tmp_path / "wd", seed_files)

    def state():
        return sorted(
            (str(p.relative_to(wd)), p.stat().st_mtime_ns, p.stat().st_mode)
            for p in [wd, *wd.rglob("*")]
        )

    before = state()
    take(wd)

    assert state() == before
    assert not (wd / ".git").exists()


# --- commit metadata ---------------------------------------------------------------------------


def test_snapshot_commit_uses_the_lethe_identity_and_a_private_ref(
    archive, make_tree, take, tmp_path, seed_files, git
):
    snap = take(make_tree(tmp_path / "wd", seed_files), message="first line\n\nbody text")

    identity = git("log", "-1", "--format=%an|%ae|%cn|%ce", snap.sha).strip()
    assert identity == "lethe|lethe@localhost|lethe|lethe@localhost"
    assert git("log", "-1", "--format=%B", snap.sha).strip() == "first line\n\nbody text"
    # reachable only under refs/lethe/<sha>; no branch is created and HEAD stays unborn
    assert git("for-each-ref", "--format=%(refname)").split() == [f"refs/lethe/{snap.sha}"]
    assert git.raw("rev-parse", "--verify", "-q", "HEAD", check=False).returncode != 0


def test_snapshot_accepts_long_and_nul_containing_messages(
    archive, make_tree, take, tmp_path, seed_files, git
):
    message = "summary\x00after nul\n" + "x" * 300_000

    snap = take(make_tree(tmp_path / "wd", seed_files), message=message)

    logged = git("log", "-1", "--format=%B", snap.sha)
    assert logged.startswith("summaryafter nul\n")
    assert len(logged.strip()) > 300_000


def test_snapshot_records_the_parent_chain(archive, make_tree, take, tmp_path, seed_files, git):
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd, message="one")
    (wd / "solver.py").write_text("v2\n")
    second = take(wd, parent=first, message="two")
    (wd / "solver.py").write_text("v3\n")
    third = take(wd, parent=second, message="three")
    fork = take(wd, parent=first, message="fork from the first")

    assert git("rev-list", third.sha).split() == [third.sha, second.sha, first.sha]
    assert git("rev-list", "--parents", "-n", "1", first.sha).split() == [first.sha]  # root
    assert git("rev-parse", f"{fork.sha}^").strip() == first.sha
    assert set(git("for-each-ref", "--format=%(refname)").split()) == {
        f"refs/lethe/{s.sha}" for s in (first, second, third, fork)
    }


# --- stats and the judged key ------------------------------------------------------------------


def test_snapshot_without_a_parent_has_no_stats_and_counts_as_changed(
    archive, make_tree, take, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    assert (snap.lines_added, snap.lines_deleted, snap.files_changed) == (0, 0, 0)
    assert snap.changed_judged is True
    assert snap.judged_key == archive.judged_key(snap.sha)


def test_identical_content_gives_the_same_tree_and_judged_key(
    archive, make_tree, take, tmp_path, seed_files, git
):
    first = take(make_tree(tmp_path / "one", seed_files), index_name="one")
    second = take(make_tree(tmp_path / "two", seed_files), index_name="two")

    assert tree_of(git, first.sha) == tree_of(git, second.sha)
    assert first.judged_key == second.judged_key


def test_judged_key_is_sha256_over_sorted_judged_path_blob_lines(
    archive, make_tree, take, tmp_path, seed_files, judged
):
    snap = take(make_tree(tmp_path / "wd", seed_files))
    files = archive.files(snap.sha)
    lines = [f"{path}\0{files[path]}" for path in sorted(files) if judged(path)]
    expected = hashlib.sha256("\n".join(lines).encode()).hexdigest()

    assert lines == [f"solver.py\0{files['solver.py']}", f"src/util.py\0{files['src/util.py']}"]
    assert snap.judged_key == archive.judged_key(snap.sha) == expected


def test_editing_only_non_judged_files_leaves_the_judged_key_alone(
    archive, make_tree, take, tmp_path, seed_files, git
):
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "evaluate.py").write_text("print('tampered')\n" * 40)
    (wd / "data/config.json").write_text('{"n": 99}\n')

    second = take(wd, parent=first)

    assert tree_of(git, second.sha) != tree_of(git, first.sha)
    assert second.judged_key == first.judged_key
    assert second.changed_judged is False
    assert (second.lines_added, second.lines_deleted, second.files_changed) == (0, 0, 0)


def test_editing_a_judged_file_counts_its_lines_and_changes_the_key(
    archive, make_tree, take, tmp_path, seed_files
):
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "solver.py").write_text("def solve(x):\n    y = x * 2\n    return y + 1\n")
    (wd / "evaluate.py").write_text("noise\n" * 50)  # not judged: must not be counted

    second = take(wd, parent=first)

    assert second.judged_key != first.judged_key
    assert second.changed_judged is True
    assert (second.lines_added, second.lines_deleted, second.files_changed) == (2, 1, 1)


def test_added_and_deleted_judged_files_are_counted(archive, make_tree, take, tmp_path, seed_files):
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "src/new.py").write_text("a\nb\nc\n")
    (wd / "src/util.py").unlink()

    second = take(wd, parent=first)

    assert (second.lines_added, second.lines_deleted, second.files_changed) == (3, 1, 2)
    assert second.changed_judged is True
    assert "src/util.py" not in archive.files(second.sha)


def test_a_binary_judged_change_is_a_changed_file_with_zero_lines(
    archive, make_tree, take, tmp_path, seed_files
):
    wd = make_tree(tmp_path / "wd", {**seed_files, "weights.bin": b"\x00\x01\x02"})
    first = take(wd)
    (wd / "weights.bin").write_bytes(b"\x00\x09\x09\x09")

    second = take(wd, parent=first)

    assert (second.lines_added, second.lines_deleted, second.files_changed) == (0, 0, 1)
    assert second.changed_judged is True


def test_the_judged_key_covers_path_and_content_but_not_the_file_mode(
    archive, make_tree, take, tmp_path, seed_files
):
    # Mirrors the brief's formula: "path\0blob" lines only. A chmod-only edit is not "changed".
    wd = make_tree(tmp_path / "wd", seed_files)
    first = take(wd)
    (wd / "solver.py").chmod(0o755)

    second = take(wd, parent=first)

    assert second.judged_key == first.judged_key
    assert second.changed_judged is False
    assert (second.lines_added, second.lines_deleted, second.files_changed) == (0, 0, 1)
