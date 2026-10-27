"""The candidate archive: a bare git repository holding every candidate as a commit.

The archive is the harness's long-term memory. Agents never see it directly: they only get
what the materializer copies into their workspace. Forgetting never deletes from it.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import NamedTuple

from lethe.workspace.paths import PathFilter, safe_join, safe_relpath

_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_OID_RE = re.compile(r"[0-9a-fA-F]{4,64}")
_REF_PREFIX = "refs/lethe/"
_FILE_MODE = "100644"
_EXEC_MODE = "100755"
_SYMLINK_MODE = "120000"

# Every git call runs with this identity and without any user or system config, so results never
# depend on who runs the harness (signing, EOL conversion, hooks, templates...).
_FIXED_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "lethe",
    "GIT_COMMITTER_NAME": "lethe",
    "GIT_AUTHOR_EMAIL": "lethe@localhost",
    "GIT_COMMITTER_EMAIL": "lethe@localhost",
}
# An enclosing git process (a hook, `rebase --exec`) exports these to select a repository or to
# inject config. Inherited, they would send our plumbing calls somewhere else.
_SCRUBBED_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIFF_OPTS",
        "GIT_EXTERNAL_DIFF",
    }
)

# path -> (file mode, blob sha) for the regular files of one commit
_Entries = dict[str, tuple[str, str]]


class _Found(NamedTuple):
    """A file `snapshot` will archive."""

    rel: str  # relative posix path
    full: str  # absolute path on disk
    executable: bool


@dataclass(frozen=True)
class Snapshot:
    """A commit of a working tree, with stats of its editable ("judged") files vs. its parent."""

    sha: str
    judged_key: str  # sha256 hex over sorted "path\0blob_sha" lines of judged files
    lines_added: int  # judged files only, vs parent (0 when there is no parent)
    lines_deleted: int
    files_changed: int
    changed_judged: bool  # judged_key differs from the parent's (always True with no parent)


class Archive:
    """Git-backed store of candidate trees.

    `judged` selects the editable files that are scored (and diffed/counted); `exclude` selects
    paths that never enter the archive (harness-rendered `.context/**`, `TASK.md`, caches...).
    Symlinks and non-regular files in a working tree are skipped, never followed.

    Concurrency: methods may be called from several threads at once. Each `snapshot` call uses
    its own index file (`<index_dir>/<index_name>`), so agents never share a git index.
    Commits use a fixed identity ("lethe <lethe@localhost>") and are kept reachable under
    `refs/lethe/<sha>`.
    """

    def __init__(self, git_dir: Path, *, judged: PathFilter, exclude: PathFilter) -> None:
        """Open the bare repo at `git_dir`, creating it (and `git_dir.parent/index/`) if missing."""
        self.git_dir = Path(git_dir).absolute()
        self.index_dir = self.git_dir.parent / "index"
        self.judged = judged
        self.exclude = exclude
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.mkdir(parents=True, exist_ok=True)
            self._git("init", "--bare", "-q")
            # `git init` bakes host traits into the new repo (macOS: case folding, unicode
            # precomposition). An archive must treat paths as plain bytes, like our own filters.
            self._git("config", "core.ignorecase", "false")
            self._git("config", "core.precomposeunicode", "false")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.empty_tree = (
            self._git("hash-object", "-t", "tree", "--stdin", input=b"").decode().strip()
        )

    # --- writing -------------------------------------------------------------------------------

    def snapshot(
        self, workdir: Path, *, parent_sha: str | None, message: str, index_name: str
    ) -> Snapshot:
        """Commit every regular file under `workdir` not matched by `exclude`.

        `index_name` names this call's private git index; use one distinct name per agent.
        Names git cannot be handed safely (a line break in the path), names with a backslash (which
        `safe_relpath` refuses, so they could never be checked out) and `.git` entries are skipped.
        """
        _check_name(index_name, "index_name")
        if parent_sha is not None:
            _check_oid(parent_sha)
        entries = self._scan(Path(workdir))
        index_file = self.index_dir / index_name
        lock_file = self.index_dir / f"{index_name}.lock"
        for stale in (index_file, lock_file):  # leftovers of a crashed snapshot
            stale.unlink(missing_ok=True)
        try:
            blobs = self._hash_blobs(entries)
            records = b"".join(
                f"{_EXEC_MODE if found.executable else _FILE_MODE} {blob}\t".encode("ascii")
                + os.fsencode(found.rel)
                + b"\0"
                for found, blob in zip(entries, blobs, strict=True)
            )
            env = {"GIT_INDEX_FILE": str(index_file)}
            self._git("update-index", "-z", "--add", "--index-info", input=records, env=env)
            tree = self._git("write-tree", env=env).decode("ascii").strip()
        finally:
            for leftover in (index_file, lock_file):
                leftover.unlink(missing_ok=True)
        sha = self._commit(tree, parent_sha, message)
        self._git("update-ref", f"{_REF_PREFIX}{sha}", sha)
        return self._stats(sha, parent_sha)

    # --- reading -------------------------------------------------------------------------------

    def files(self, sha: str) -> dict[str, str]:
        """Map of relative path -> blob sha for every file in commit `sha`."""
        return {path: blob for path, (_, blob) in self._ls_tree(sha).items()}

    def read_file(self, sha: str, path: str) -> bytes:
        """Contents of `path` in commit `sha`; a path that is not in the commit is an error."""
        _check_oid(sha)
        return self._git("cat-file", "blob", f"{sha}:{path}")

    def judged_key(self, sha: str) -> str:
        """Content address of the judged files of `sha` (paths and blobs; not file modes)."""
        judged = sorted(
            (path, blob) for path, (_, blob) in self._ls_tree(sha).items() if self.judged(path)
        )
        payload = "\n".join(f"{path}\0{blob}" for path, blob in judged)
        return hashlib.sha256(payload.encode("utf-8", errors="surrogateescape")).hexdigest()

    def diff(self, base_sha: str | None, sha: str, *, judged_only: bool = True) -> str:
        """Unified diff from `base_sha` (empty tree if None) to `sha`."""
        _check_oid(sha)
        if base_sha is not None:
            _check_oid(base_sha)
        args = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames"]
        args += [self.empty_tree if base_sha is None else base_sha, sha]
        if judged_only:
            before = {} if base_sha is None else self._ls_tree(base_sha)
            after = self._ls_tree(sha)
            changed = sorted(
                path
                for path in before.keys() | after.keys()
                if self.judged(path) and before.get(path) != after.get(path)
            )
            if not changed:
                return ""
            args += ["--", *(f":(literal){path}" for path in changed)]
        return self._git(*args).decode("utf-8", errors="replace")

    def checkout(self, sha: str, dest: Path, *, keep: Callable[[str], bool] | None = None) -> None:
        """Write the files of `sha` into `dest` (created if missing; must be empty if it exists).

        If `keep` is given, only paths for which `keep(path)` is True are written.
        Raises FileExistsError for a non-empty `dest` and UnsafePathError if the commit holds a
        path that would escape `dest`; in both cases nothing is written.
        """
        dest = Path(dest)
        _require_free(dest)
        entries = self._ls_tree(sha)
        chosen = {path: entry for path, entry in entries.items() if keep is None or keep(path)}
        self._write_files(chosen, dest)

    def checkout_overlay(self, base_sha: str, cand_sha: str, dest: Path) -> None:
        """Write the judge tree: files of `base_sha` that are NOT judged, plus the judged files of
        `cand_sha`. Judged files deleted in the candidate are therefore absent; edits the candidate
        made outside judged paths are dropped."""
        dest = Path(dest)
        _require_free(dest)
        merged = {p: e for p, e in self._ls_tree(base_sha).items() if not self.judged(p)}
        merged |= {p: e for p, e in self._ls_tree(cand_sha).items() if self.judged(p)}
        self._write_files(merged, dest)

    # --- internals -----------------------------------------------------------------------------

    def _git(
        self, *args: str, input: bytes | None = None, env: Mapping[str, str] | None = None
    ) -> bytes:
        """Run `git --git-dir <archive> <args>`; failures raise RuntimeError carrying stderr."""
        command = ["git", "--git-dir", str(self.git_dir), *args]
        stdin = subprocess.DEVNULL if input is None else None
        try:
            done = subprocess.run(
                command,
                input=input,
                stdin=stdin,
                capture_output=True,
                check=True,
                env=git_env(env),
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(_failure_message(args, exc.returncode, exc.stderr)) from exc
        return done.stdout

    def _scan(self, workdir: Path) -> list[_Found]:
        """Archivable files of `workdir`: regular files only, `exclude` and `.git` dropped."""
        root = workdir.absolute()
        if not root.exists():
            raise FileNotFoundError(f"workdir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"workdir is not a directory: {root}")
        if "\n" in str(root) or "\r" in str(root):
            raise ValueError(f"workdir path must not contain a line break: {str(root)!r}")
        found: list[_Found] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if name != ".git")
            relative_dir = os.path.relpath(dirpath, root)
            prefix = "" if relative_dir == "." else relative_dir.replace(os.sep, "/") + "/"
            for name in sorted(filenames):
                rel = prefix + name
                if name == ".git" or _unsafe_name(rel) or self.exclude(rel):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    mode = os.lstat(full).st_mode
                except OSError:  # vanished since it was listed
                    continue
                if stat.S_ISREG(mode):  # not a symlink, FIFO, socket or device
                    found.append(_Found(rel, full, bool(mode & stat.S_IXUSR)))
        return found

    def _hash_blobs(self, entries: list[_Found]) -> list[str]:
        """Write every file into the object store (one git process); returns blob shas in order."""
        if not entries:
            return []
        paths = b"".join(os.fsencode(found.full) + b"\n" for found in entries)
        out = self._git("hash-object", "-w", "--no-filters", "--stdin-paths", input=paths)
        blobs = out.decode("ascii").split()
        if len(blobs) != len(entries):
            raise RuntimeError(
                f"git hash-object returned {len(blobs)} ids for {len(entries)} files"
            )
        return blobs

    def _commit(self, tree: str, parent_sha: str | None, message: str) -> str:
        args = ["commit-tree", tree]
        if parent_sha is not None:
            args += ["-p", parent_sha]
        text = message.replace("\x00", "")  # git rejects NUL; stdin avoids argv size limits
        if text and not text.endswith("\n"):
            text += "\n"
        return self._git(*args, input=text.encode("utf-8", errors="replace")).decode().strip()

    def _stats(self, sha: str, parent_sha: str | None) -> Snapshot:
        key = self.judged_key(sha)
        if parent_sha is None:
            return Snapshot(sha, key, 0, 0, 0, True)
        out = self._git("diff-tree", "-r", "--no-renames", "--numstat", "-z", parent_sha, sha)
        added = deleted = changed = 0
        for record in out.split(b"\0"):
            if not record:
                continue
            added_text, deleted_text, raw_path = record.split(b"\t", 2)
            if not self.judged(os.fsdecode(raw_path)):
                continue
            changed += 1
            if added_text != b"-":  # binary files report "-" and count as a file with no lines
                added += int(added_text)
                deleted += int(deleted_text)
        return Snapshot(sha, key, added, deleted, changed, key != self.judged_key(parent_sha))

    def _ls_tree(self, sha: str) -> _Entries:
        """Regular files of commit `sha`; symlinks and submodules never come from `snapshot`."""
        _check_oid(sha)
        out = self._git("ls-tree", "-r", "-z", "--full-tree", sha)
        entries: _Entries = {}
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, raw_path = record.split(b"\t", 1)
            mode, kind, blob = meta.decode("ascii").split(" ")
            if kind == "blob" and mode != _SYMLINK_MODE:
                entries[os.fsdecode(raw_path)] = (mode, blob)
        return entries

    def _write_files(self, entries: _Entries, dest: Path) -> None:
        """Write `entries` under `dest`. Every path is validated before anything is created."""
        relatives = {path: safe_relpath(path) for path in entries}
        dest.mkdir(parents=True, exist_ok=True)
        with _BlobReader(self.git_dir) as reader:
            for path, (mode, blob) in entries.items():
                target = safe_join(dest, relatives[path])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(reader.read(blob))
                if mode == _EXEC_MODE:
                    target.chmod(0o755)


class _BlobReader:
    """Serves blob contents from one `git cat-file --batch` process, one request at a time.

    Strict request/response lockstep keeps memory bounded by the largest blob and cannot
    deadlock on full pipes.
    """

    def __init__(self, git_dir: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "--git-dir", str(git_dir), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_env(),
        )
        self._stderr = b""
        self._closed = False

    def __enter__(self) -> _BlobReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._shutdown()

    def read(self, blob: str) -> bytes:
        proc = self._proc
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(f"{blob}\n".encode("ascii"))
            proc.stdin.flush()
            header = proc.stdout.readline().split()
        except BrokenPipeError:
            header = []
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(self._failure(f"no blob reply for {blob}"))
        size = int(header[2])
        data = proc.stdout.read(size)
        if len(data) != size or proc.stdout.read(1) != b"\n":
            raise RuntimeError(self._failure(f"truncated reply for {blob}"))
        return data

    def _failure(self, what: str) -> str:
        self._shutdown(kill=True)
        detail = self._stderr.decode("utf-8", errors="replace").strip() or "no output on stderr"
        return f"git cat-file --batch failed ({what}): {detail}"

    def _shutdown(self, *, kill: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if kill and proc.poll() is None:
            proc.kill()
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.stderr is not None:
            self._stderr = proc.stderr.read()
            proc.stderr.close()


def git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for every git call lethe makes: no user or system config, a fixed
    identity, and none of the variables that would point git at another repository."""
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env.update(_FIXED_ENV)
    if extra:
        env.update(extra)
    return env


def _unsafe_name(rel: str) -> bool:
    """A path git cannot be handed safely (line breaks) or that is not a plain POSIX path."""
    return "\n" in rel or "\r" in rel or "\\" in rel


def _failure_message(args: tuple[str, ...], returncode: int, stderr: bytes) -> str:
    command = " ".join(args)
    if len(command) > 200:
        command = command[:200] + "..."
    detail = stderr.decode("utf-8", errors="replace").strip() or "no output on stderr"
    return f"git {command} failed with exit code {returncode}: {detail}"


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or value in (".", "..") or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{what} must match [A-Za-z0-9_.-]+ and not be '.' or '..': {value!r}")
    return value


def _check_oid(value: str) -> str:
    if not isinstance(value, str) or not _OID_RE.fullmatch(value):
        raise ValueError(f"expected a hex object id (4 to 64 hex digits), got {value!r}")
    return value


def _require_free(dest: Path) -> None:
    """`dest` must not exist, or be an empty directory (a symlink never counts as one)."""
    if os.path.lexists(dest) and (dest.is_symlink() or not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"checkout destination exists and is not an empty directory: {dest}")
