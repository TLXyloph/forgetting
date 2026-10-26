"""Archive: `checkout` and `checkout_overlay` (real git, tmp_path)."""

from __future__ import annotations

import stat

import pytest

from lethe.workspace.paths import UnsafePathError


def is_executable(path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)


# --- checkout ----------------------------------------------------------------------------------


def test_checkout_recreates_the_committed_tree_exactly(
    archive, make_tree, take, tree_contents, tmp_path
):
    payload = {
        "solver.py": "def solve(x):\n    return x\n",
        "src/deep/er/util.py": "VALUE = 1\n",
        "crlf.txt": b"a\r\nb\r\n",
        "binary.dat": bytes(range(256)),
        "empty.txt": b"",
    }
    snap = take(make_tree(tmp_path / "wd", payload, executable=["solver.py"]))
    dest = tmp_path / "out" / "nested" / "checkout"  # parents are created as needed

    archive.checkout(snap.sha, dest)

    expected = {k: v.encode() if isinstance(v, str) else v for k, v in payload.items()}
    assert tree_contents(dest) == expected
    assert is_executable(dest / "solver.py")
    assert not is_executable(dest / "src/deep/er/util.py")


def test_checkout_keep_filter_selects_paths(
    archive, make_tree, take, tree_contents, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))
    seen: list[str] = []

    def keep(path: str) -> bool:
        seen.append(path)
        return path.endswith(".py")

    archive.checkout(snap.sha, tmp_path / "out", keep=keep)

    assert set(tree_contents(tmp_path / "out")) == {"solver.py", "src/util.py", "evaluate.py"}
    assert sorted(seen) == sorted(seed_files)  # called once per file with posix relative paths


def test_checkout_accepts_an_existing_empty_directory(
    archive, make_tree, take, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))
    dest = tmp_path / "out"
    dest.mkdir()

    archive.checkout(snap.sha, dest)

    assert (dest / "solver.py").is_file()


def test_checkout_of_the_empty_tree_creates_an_empty_directory(archive, take, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    snap = take(empty)

    archive.checkout(snap.sha, tmp_path / "out")

    assert (tmp_path / "out").is_dir()
    assert list((tmp_path / "out").iterdir()) == []


def test_checkout_into_a_non_empty_directory_raises_and_changes_nothing(
    archive, make_tree, take, tmp_path, seed_files
):
    snap = take(make_tree(tmp_path / "wd", seed_files))
    dest = make_tree(tmp_path / "out", {"mine.txt": "do not touch"})

    with pytest.raises(FileExistsError):
        archive.checkout(snap.sha, dest)

    assert sorted(p.name for p in dest.iterdir()) == ["mine.txt"]
    assert (dest / "mine.txt").read_text() == "do not touch"


def test_checkout_onto_a_regular_file_raises(archive, make_tree, take, tmp_path, seed_files):
    snap = take(make_tree(tmp_path / "wd", seed_files))
    target = tmp_path / "a-file"
    target.write_text("x")

    with pytest.raises(FileExistsError):
        archive.checkout(snap.sha, target)


@pytest.mark.parametrize(
    "bad_name", ["../escape.txt", "/abs-escape.txt", "a/../../escape.txt", ".."]
)
def test_checkout_refuses_tree_entries_that_escape_the_destination(
    archive, git, tmp_path, bad_name
):
    # Forge a commit whose tree has a hostile entry (git itself refuses to build one normally).
    blob = git("hash-object", "-w", "--stdin", input=b"pwned\n").strip()
    entries = [("ok.txt", blob), (bad_name, blob)]
    raw = b"".join(b"100644 " + name.encode() + b"\0" + bytes.fromhex(sha) for name, sha in entries)
    tree = git("hash-object", "-t", "tree", "-w", "--literally", "--stdin", input=raw).strip()
    commit = git("commit-tree", tree, "-m", "hostile").strip()
    dest = tmp_path / "sandbox" / "out"

    with pytest.raises(UnsafePathError):
        archive.checkout(commit, dest)

    assert not dest.exists()  # nothing at all is written
    assert not (tmp_path / "sandbox" / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


# --- checkout_overlay --------------------------------------------------------------------------


def overlay_setup(make_tree, take, tmp_path):
    """A base snapshot and a candidate that edits, adds, deletes judged and non-judged files."""
    base_files = {
        "solver.py": "def solve(x):\n    return x\n",
        "src/util.py": "VALUE = 1\n",
        "src/old.py": "OLD = True\n",
        "weights.bin": b"\x00base",
        "evaluate.py": "print('score')\n",
        "data/config.json": '{"n": 3}\n',
        "check.sh": "#!/bin/sh\n",
        "notes.txt": "base notes\n",
    }
    base = take(make_tree(tmp_path / "base", base_files, executable=["check.sh"]), index_name="b")
    cand_files = {
        "solver.py": "def solve(x):\n    return x + 1\n",  # judged, edited
        "src/util.py": "VALUE = 1\n",  # judged, unchanged
        "src/new.py": "NEW = True\n",  # judged, added
        # src/old.py: judged, deleted        weights.bin: judged, deleted
        "evaluate.py": "print('always 100')\n",  # non-judged, tampered
        "data/config.json": '{"n": 999}\n',  # non-judged, tampered
        "extra.txt": "candidate-only, not judged\n",  # non-judged, added
        # check.sh and notes.txt: non-judged, deleted
    }
    cand = take(make_tree(tmp_path / "cand", cand_files), parent=base, index_name="c")
    return base, cand, base_files, cand_files


def test_overlay_is_the_bases_non_judged_files_plus_the_candidates_judged_files(
    archive, make_tree, take, tree_contents, tmp_path
):
    base, cand, base_files, cand_files = overlay_setup(make_tree, take, tmp_path)

    archive.checkout_overlay(base.sha, cand.sha, tmp_path / "judge")

    assert tree_contents(tmp_path / "judge") == {
        # judged files come from the candidate; judged deletions propagate
        "solver.py": cand_files["solver.py"].encode(),
        "src/util.py": cand_files["src/util.py"].encode(),
        "src/new.py": cand_files["src/new.py"].encode(),
        # everything else comes from the base: edits dropped, deletions undone, additions absent
        "evaluate.py": base_files["evaluate.py"].encode(),
        "data/config.json": base_files["data/config.json"].encode(),
        "check.sh": base_files["check.sh"].encode(),
        "notes.txt": base_files["notes.txt"].encode(),
    }


def test_overlay_keeps_executable_bits_from_the_side_each_file_comes_from(
    archive, make_tree, take, tmp_path
):
    base, cand, _, _ = overlay_setup(make_tree, take, tmp_path)
    (tmp_path / "cand" / "src/new.py").chmod(0o755)
    cand = take(tmp_path / "cand", parent=base, index_name="c")

    archive.checkout_overlay(base.sha, cand.sha, tmp_path / "judge")

    assert is_executable(tmp_path / "judge" / "check.sh")  # non-judged, from the base
    assert is_executable(tmp_path / "judge" / "src/new.py")  # judged, from the candidate
    assert not is_executable(tmp_path / "judge" / "solver.py")


def test_overlay_of_the_base_with_itself_is_the_full_base_tree(
    archive, make_tree, take, tree_contents, tmp_path, seed_files
):
    base = take(make_tree(tmp_path / "wd", seed_files))

    archive.checkout_overlay(base.sha, base.sha, tmp_path / "judge")

    assert tree_contents(tmp_path / "judge") == {k: v.encode() for k, v in seed_files.items()}


def test_overlay_requires_an_empty_destination(archive, make_tree, take, tmp_path, seed_files):
    base = take(make_tree(tmp_path / "wd", seed_files))
    dest = make_tree(tmp_path / "judge", {"stale.txt": "x"})

    with pytest.raises(FileExistsError):
        archive.checkout_overlay(base.sha, base.sha, dest)

    assert sorted(p.name for p in dest.iterdir()) == ["stale.txt"]
