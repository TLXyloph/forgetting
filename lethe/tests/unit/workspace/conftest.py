"""Fixtures for the workspace-context tests. Everything runs against real git in `tmp_path`."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from lethe.workspace.archive import Archive, Snapshot
from lethe.workspace.paths import PathFilter

# The toy task: `solver.py` and anything under `src/` are the editable (judged) files.
JUDGED_GLOBS = ("solver.py", "src/**/*.py", "weights.bin")
# What the harness renders or caches itself: these never enter the archive.
EXCLUDE_GLOBS = (".context/**", "TASK.md", "**/__pycache__/**")

SEED_FILES: dict[str, str] = {
    "solver.py": "def solve(x):\n    return x\n",
    "src/util.py": "VALUE = 1\n",
    "evaluate.py": "print('score: 1')\n",
    "data/config.json": '{"n": 3}\n',
}


class GitCli:
    """Runs git against a bare repo in a clean environment, independently of the code under test."""

    def __init__(self, git_dir: Path) -> None:
        self.git_dir = git_dir

    def raw(
        self,
        *args: str,
        input: bytes | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        clean.update(
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="test",
            GIT_AUTHOR_EMAIL="test@localhost",
            GIT_COMMITTER_NAME="test",
            GIT_COMMITTER_EMAIL="test@localhost",
        )
        clean.update(env or {})
        return subprocess.run(
            ["git", "--git-dir", str(self.git_dir), *args],
            input=input,
            capture_output=True,
            env=clean,
            check=check,
        )

    def __call__(self, *args: str, input: bytes | None = None) -> str:
        return self.raw(*args, input=input).stdout.decode()

    def has_blob(self, content: str | bytes) -> bool:
        """True if an object with exactly this content is in the object store (even unreachable)."""
        data = content.encode() if isinstance(content, str) else content
        blob = self.raw("hash-object", "--stdin", input=data).stdout.decode().strip()
        return self.raw("cat-file", "-e", blob, check=False).returncode == 0


def write_tree(
    root: Path, files: Mapping[str, str | bytes], *, executable: Iterable[str] = ()
) -> Path:
    """Create `files` (relative path -> content) under `root`; chmod +x the `executable` ones."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)
    for rel in executable:
        target = root / rel
        target.chmod(target.stat().st_mode | 0o111)
    return root


def read_tree(root: Path) -> dict[str, bytes]:
    """Every regular file under `root` as {posix relative path: bytes} (empty dirs are ignored).

    A `.git` directory is left out: every workspace is an empty repository on purpose
    (test_workspace_git.py covers it).
    """
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink() and ".git" not in p.relative_to(root).parts
    }


@pytest.fixture
def judged() -> PathFilter:
    return PathFilter(JUDGED_GLOBS)


@pytest.fixture
def exclude() -> PathFilter:
    return PathFilter(EXCLUDE_GLOBS)


@pytest.fixture
def git_dir(tmp_path: Path) -> Path:
    return tmp_path / "run" / "archive.git"


@pytest.fixture
def archive(git_dir: Path, judged: PathFilter, exclude: PathFilter) -> Archive:
    return Archive(git_dir, judged=judged, exclude=exclude)


@pytest.fixture
def archive_factory(
    git_dir: Path, judged: PathFilter, exclude: PathFilter
) -> Callable[[], Archive]:
    """Build the archive lazily, so a test can change the environment first."""
    return lambda: Archive(git_dir, judged=judged, exclude=exclude)


@pytest.fixture
def git(git_dir: Path) -> GitCli:
    return GitCli(git_dir)


@pytest.fixture
def seed_files() -> dict[str, str]:
    return dict(SEED_FILES)


@pytest.fixture
def make_tree() -> Callable[..., Path]:
    return write_tree


@pytest.fixture
def tree_contents() -> Callable[[Path], dict[str, bytes]]:
    return read_tree


@pytest.fixture
def take(archive: Archive) -> Callable[..., Snapshot]:
    """Snapshot `workdir` into the archive; `parent` is the previous `Snapshot`, if any."""

    def _take(
        workdir: Path,
        *,
        parent: Snapshot | None = None,
        message: str = "msg",
        index_name: str = "agent1",
    ) -> Snapshot:
        return archive.snapshot(
            workdir,
            parent_sha=parent.sha if parent else None,
            message=message,
            index_name=index_name,
        )

    return _take
