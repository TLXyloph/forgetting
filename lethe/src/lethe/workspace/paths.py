"""Path matching and path safety for workspaces, archives, and judge trees.

Paths are POSIX-style and relative to a workspace/seed root. Glob semantics are explicit:
`**` matches any number of directories (including none), `*` and `?` never cross `/`.
A pattern without a slash therefore matches top-level paths only: use `**/*.py` for any depth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath


class UnsafePathError(ValueError):
    """Raised for absolute paths, parent traversal, or paths that escape their root."""


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern to an anchored regex using the module's semantics."""
    if not pattern or pattern.startswith("/"):
        raise ValueError(f"glob must be a non-empty relative pattern: {pattern!r}")
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


class PathFilter:
    """Matches relative paths that match any `include` glob and no `exclude` glob."""

    def __init__(self, include: Sequence[str], exclude: Sequence[str] = ()) -> None:
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        self._include = [glob_to_regex(p) for p in self.include]
        self._exclude = [glob_to_regex(p) for p in self.exclude]

    def matches(self, rel_path: str) -> bool:
        if any(rx.match(rel_path) for rx in self._exclude):
            return False
        return any(rx.match(rel_path) for rx in self._include)

    def __call__(self, rel_path: str) -> bool:
        return self.matches(rel_path)


def safe_relpath(rel: str) -> str:
    """Normalize a relative POSIX path; reject absolute, empty, NUL-containing, or `..` paths."""
    if not rel or "\x00" in rel:
        raise UnsafePathError(f"invalid path: {rel!r}")
    posix = PurePosixPath(rel.replace("\\", "/"))
    if posix.is_absolute() or any(part == ".." for part in posix.parts):
        raise UnsafePathError(f"path escapes its root: {rel!r}")
    parts = [p for p in posix.parts if p not in ("", ".")]
    if not parts:
        raise UnsafePathError(f"invalid path: {rel!r}")
    return "/".join(parts)


def safe_join(root: Path, rel: str) -> Path:
    """Join `rel` under `root`, guaranteeing the result stays inside `root` (symlinks resolved)."""
    candidate = root / safe_relpath(rel)
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise UnsafePathError(f"path escapes its root: {rel!r}")
    return candidate
