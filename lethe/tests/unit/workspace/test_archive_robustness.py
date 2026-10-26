"""Archive: input validation, environment isolation, error reporting, index files, threads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

BAD_INDEX_NAMES = ["", ".", "..", "a/b", "../escape", "a b", "x\n", "café", "a\x00b", "a;b"]
BAD_SHAS = ["", "HEAD", "--output=/tmp/pwned", "-p", "abc def", "zz" * 20, "0" * 65, "refs/heads/x"]

# --- validation at the boundary ----------------------------------------------------------------


@pytest.mark.parametrize("name", BAD_INDEX_NAMES)
def test_snapshot_rejects_unsafe_index_names_before_touching_anything(
    archive, make_tree, tmp_path, seed_files, git, git_dir, name
):
    wd = make_tree(tmp_path / "wd", seed_files)

    with pytest.raises(ValueError, match="index_name"):
        archive.snapshot(wd, parent_sha=None, message="m", index_name=name)

    assert list((git_dir.parent / "index").iterdir()) == []
    assert git("for-each-ref").strip() == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run", "wd"]  # nothing escaped


def test_snapshot_accepts_ordinary_index_names(archive, make_tree, take, tmp_path, seed_files):
    wd = make_tree(tmp_path / "wd", seed_files)

    snap = take(wd, index_name="agent-1.v2_x")

    assert set(archive.files(snap.sha)) == set(seed_files)


@pytest.mark.parametrize("bad", BAD_SHAS)
def test_malformed_object_ids_are_rejected_before_git_runs(
    archive, make_tree, take, tmp_path, seed_files, git, bad
):
    wd = make_tree(tmp_path / "wd", seed_files)
    good = take(wd).sha
    refs_before = git("for-each-ref").strip()
    out = tmp_path / "out"

    calls = {
        "files": lambda: archive.files(bad),
        "read_file": lambda: archive.read_file(bad, "solver.py"),
        "judged_key": lambda: archive.judged_key(bad),
        "diff sha": lambda: archive.diff(good, bad),
        "diff base": lambda: archive.diff(bad, good),
        "checkout": lambda: archive.checkout(bad, out),
        "overlay base": lambda: archive.checkout_overlay(bad, good, out),
        "overlay cand": lambda: archive.checkout_overlay(good, bad, out),
        "snapshot parent": lambda: archive.snapshot(
            wd, parent_sha=bad, message="m", index_name="agent1"
        ),
    }
    for label, call in calls.items():
        with pytest.raises(ValueError, match="object id"):
            call()
        assert not out.exists(), label

    assert git("for-each-ref").strip() == refs_before  # the rejected snapshot made no commit


def test_abbreviated_object_ids_are_accepted(archive, make_tree, take, tmp_path, seed_files):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    assert archive.files(snap.sha[:12]) == archive.files(snap.sha)


# --- git failures ------------------------------------------------------------------------------


def test_git_failures_become_runtime_errors_carrying_gits_stderr(archive, git):
    missing = "0" * 40
    complaint = git.raw("ls-tree", missing, check=False).stderr.decode().strip()
    assert complaint  # sanity: git does say something

    with pytest.raises(RuntimeError) as excinfo:
        archive.files(missing)

    assert complaint in str(excinfo.value)
    assert "ls-tree" in str(excinfo.value)


def test_an_unknown_parent_fails_cleanly_and_leaves_no_trace(
    archive, make_tree, tmp_path, seed_files, git, git_dir
):
    wd = make_tree(tmp_path / "wd", seed_files)

    with pytest.raises(RuntimeError, match="commit-tree"):
        archive.snapshot(wd, parent_sha="0" * 40, message="m", index_name="agent1")

    assert git("for-each-ref").strip() == ""
    assert list((git_dir.parent / "index").iterdir()) == []


# --- environment isolation ---------------------------------------------------------------------


def test_user_git_configuration_cannot_change_what_is_archived(
    archive_factory, tmp_path, monkeypatch, make_tree, git
):
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".gitconfig").write_text(
        "[user]\n\tname = Someone Else\n\temail = else@example.com\n"
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = /nonexistent/gpg\n"
        "[core]\n\tautocrlf = true\n\thooksPath = /nonexistent/hooks\n"
        "[init]\n\ttemplateDir = /nonexistent/templates\n"
    )
    (home / ".config" / "git" / "attributes").write_text("* text eol=crlf\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Someone Else")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "else@example.com")

    archive = archive_factory()
    wd = make_tree(tmp_path / "wd", {"solver.py": b"a\r\nb\r\n", "lf.txt": b"a\nb\n"})
    snap = archive.snapshot(wd, parent_sha=None, message="m", index_name="agent1")

    assert archive.read_file(snap.sha, "solver.py") == b"a\r\nb\r\n"  # no EOL conversion
    assert archive.read_file(snap.sha, "lf.txt") == b"a\nb\n"
    assert git("log", "-1", "--format=%an|%ae|%cn|%ce", snap.sha).strip() == (
        "lethe|lethe@localhost|lethe|lethe@localhost"
    )


def test_inherited_repo_selection_variables_do_not_leak_into_git_calls(
    archive_factory, tmp_path, monkeypatch, make_tree, git, seed_files
):
    # e.g. the harness launched from inside a git hook, which exports GIT_DIR / GIT_INDEX_FILE
    outer = {
        "GIT_DIR": tmp_path / "outer.git",
        "GIT_WORK_TREE": tmp_path / "outer-worktree",
        "GIT_INDEX_FILE": tmp_path / "outer-index",
        "GIT_OBJECT_DIRECTORY": tmp_path / "outer-objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": tmp_path / "outer-alternates",
        "GIT_COMMON_DIR": tmp_path / "outer-common",
        "GIT_NAMESPACE": tmp_path / "outer-namespace",
    }
    for var, value in outer.items():
        monkeypatch.setenv(var, str(value))

    archive = archive_factory()
    wd = make_tree(tmp_path / "wd", seed_files)
    snap = archive.snapshot(wd, parent_sha=None, message="m", index_name="agent1")
    archive.checkout(snap.sha, tmp_path / "out")

    assert (tmp_path / "out" / "solver.py").is_file()
    assert git("cat-file", "-t", snap.sha).strip() == "commit"  # stored in the archive itself
    assert not any(path.exists() for path in outer.values())


# --- index files -------------------------------------------------------------------------------


def test_a_stale_index_file_never_leaks_into_a_snapshot(
    archive, git, git_dir, make_tree, take, tmp_path, seed_files
):
    ghost = git("hash-object", "-w", "--stdin", input=b"ghost\n").strip()
    stale = git_dir.parent / "index" / "agent1"
    git.raw(
        "update-index",
        "--add",
        "--index-info",
        input=f"100644 {ghost}\tghost.py\n".encode(),
        env={"GIT_INDEX_FILE": str(stale)},
    )
    assert stale.exists()

    snap = take(make_tree(tmp_path / "wd", seed_files), index_name="agent1")

    assert set(archive.files(snap.sha)) == set(seed_files)


def test_snapshot_removes_its_index_file_afterwards(
    archive, git_dir, make_tree, take, tmp_path, seed_files
):
    take(make_tree(tmp_path / "wd", seed_files), index_name="agent1")

    assert list((git_dir.parent / "index").iterdir()) == []


# --- concurrency -------------------------------------------------------------------------------


def test_concurrent_snapshots_with_distinct_index_names(archive, git, git_dir, make_tree, tmp_path):
    agents, rounds = 6, 3
    workdirs = [
        make_tree(tmp_path / f"wd{i}", {"solver.py": "v\n", f"only_{i}.txt": str(i)})
        for i in range(agents)
    ]

    def run_agent(i: int):
        parent = None
        snaps = []
        for r in range(rounds):
            (workdirs[i] / "solver.py").write_text(f"agent {i} round {r}\n")
            snap = archive.snapshot(
                workdirs[i], parent_sha=parent, message=f"a{i} r{r}", index_name=f"agent{i}"
            )
            snaps.append(snap)
            parent = snap.sha
        return snaps

    with ThreadPoolExecutor(max_workers=agents) as pool:
        results = list(pool.map(run_agent, range(agents)))

    for i, snaps in enumerate(results):
        contents = [archive.read_file(s.sha, "solver.py") for s in snaps]
        assert contents == [f"agent {i} round {r}\n".encode() for r in range(rounds)]
        assert set(archive.files(snaps[-1].sha)) == {"solver.py", f"only_{i}.txt"}
        assert git("rev-list", snaps[-1].sha).split() == [s.sha for s in reversed(snaps)]
    refs = git("for-each-ref", "--format=%(refname)").split()
    assert len(refs) == agents * rounds
    assert list((git_dir.parent / "index").iterdir()) == []


def test_reads_work_from_many_threads_at_once(archive, make_tree, take, tmp_path, seed_files):
    snap = take(make_tree(tmp_path / "wd", seed_files))

    def read(i: int):
        return archive.files(snap.sha), archive.read_file(snap.sha, "solver.py")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read, range(16)))

    assert all(r == results[0] for r in results)
