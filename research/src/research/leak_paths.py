"""The paths a Claude Code tool call names, and whether it reads, writes or only touches them.

`research.leaks` decides which of these paths cross a boundary; this module only finds them.

Bash commands are not parsed by a shell. After `$TMPDIR` (also `${TMPDIR}`, `${TMPDIR:-...}`),
`$HOME`, `${HOME}` and a leading `~` are expanded, regular expressions find absolute paths and
relative ones that climb out with `..`, anywhere in the command, quoted code included. Relative
paths without `..` stay in the working directory (the agent's workspace) unless the command
changes directory, which is not followed. How a path is used is read off the simple command
around it (the text between `;`, `&&`, `||`, `|`, `&`, newlines, parentheses and backquotes):

- write: after a redirection (`>`, `>>`), as an argument of `tee` or `touch` or of an output
  option (`-o`, `--output`, `--out`), or as the last argument of `cp`, `mv`, `install`, `ln` or
  `rsync`; a copy into a directory (`cp a b $TMPDIR/`) writes `<directory>/<name>` per source;
- touch: as an argument of a command that lists, names or removes (`ls`, `echo`, `cd`, `find`,
  `stat`, `rm`, ...);
- read: anything else (`cat`, `python`, `diff`, the source of a copy, ...).

File tools: Read and Grep read, Write, Edit, MultiEdit and NotebookEdit write, Glob touches; their
paths are `file_path`, `notebook_path`, `path` and an absolute Glob `pattern`.
"""

from __future__ import annotations

import bisect
import posixpath
import re
import shlex
from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "TEMP_VAR_ROOT",
    "Access",
    "bash_paths",
    "canonical",
    "expand",
    "is_within",
    "temp_key",
    "tool_paths",
]

Access = Literal["touch", "read", "write"]

# what `$TMPDIR` stands for while paths are found: the shape of the real one, /tmp/claude-<uid>
TEMP_VAR_ROOT = "/tmp/claude-0"

_TEMP_VAR = re.compile(r"\$\{TMPDIR(?:[^}A-Za-z0-9_][^}]*)?\}|\$TMPDIR(?![A-Za-z0-9_])")
_HOME_VAR = re.compile(
    r"\$\{HOME\}|\$HOME(?![A-Za-z0-9_])|(?<![^\s'\"=:(<>|;&`])~(?=/|$|[\s'\";|&)`])"
)
_STOP = r"\s'\"`;|&<>(){}$\\,:"  # characters that end a path in a command
_PATH = re.compile(
    r"(?<![\w./~$-])"  # not inside a word or another path
    rf"(/[^{_STOP}]*"  # an absolute path
    rf"|(?:\./)*\.\.(?=/|$|[{_STOP}])(?:/[^{_STOP}]*)?)"  # or a relative one climbing out
)
_BREAK = re.compile(r"&&|\|\||\$\(|[;|&\n(){}`]")  # ends of simple commands
_OUTPUT = re.compile(r"(?:>|(?:^|\s)(?:-o|--output|--out))$")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_PRIVATE = re.compile(r"^/private(?=/(?:tmp|var|etc)(?:/|$))")
_TEMP_ROOT = re.compile(r"/tmp/claude-[0-9]+(?:/(.*))?")

_WRITERS = frozenset({"tee", "touch"})
_COPIERS = frozenset({"cp", "mv", "install", "ln", "rsync"})
_TOUCHERS = frozenset(
    {"basename", "cd", "df", "dirname", "du", "echo", "export", "find", "ls", "mkdir", "printf",
     "pushd", "readlink", "realpath", "rm", "rmdir", "stat", "test", "tree", "[", "[["}
)  # fmt: skip
_WRAPPERS = frozenset({"command", "env", "exec", "nice", "nohup", "sudo", "time", "timeout"})
_TOOL_ACCESS: dict[str, Access] = {
    "Read": "read",
    "Grep": "read",
    "Glob": "touch",
    "Write": "write",
    "Edit": "write",
    "MultiEdit": "write",
    "NotebookEdit": "write",
}
_PATH_KEYS = ("file_path", "notebook_path", "path")


def canonical(path: str) -> str:
    """`path` normalized (no `.`, `..` or doubled slashes), without the `/private` macOS puts in
    front of `/tmp`, `/var` and `/etc`."""
    path = posixpath.normpath(path)
    if path.startswith("//"):  # POSIX lets normpath keep exactly two leading slashes
        path = "/" + path.lstrip("/")
    return _PRIVATE.sub("", path, count=1)


def is_within(path: str, root: str) -> bool:
    """True when `path` is `root` or below it (both canonical, or both relative)."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def temp_key(path: str) -> str | None:
    """The name of a canonical path in the shared temp directory ("" for the directory itself);
    None outside it."""
    match = _TEMP_ROOT.fullmatch(path)
    return None if match is None else match.group(1) or ""


def expand(text: str, home: str) -> str:
    """`text` with `$TMPDIR` forms replaced by TEMP_VAR_ROOT and `$HOME` forms and `~` by `home`."""
    text = _TEMP_VAR.sub(TEMP_VAR_ROOT, text)
    return _HOME_VAR.sub(lambda _: home, text)


def tool_paths(name: str, tool_input: Mapping[str, Any], home: str) -> list[tuple[str, Access]]:
    """(path, access) for each path a tool call names, expanded but not resolved."""
    if name == "Bash":
        command = tool_input.get("command")
        return bash_paths(command, home) if isinstance(command, str) else []
    values = [tool_input.get(key) for key in _PATH_KEYS]
    pattern = tool_input.get("pattern")
    if name == "Glob" and isinstance(pattern, str) and pattern.startswith(("/", "~", "..", "$")):
        values.append(pattern)
    access = _TOOL_ACCESS.get(name, "read")
    return [(expand(value, home), access) for value in values if isinstance(value, str) and value]


def bash_paths(command: str, home: str) -> list[tuple[str, Access]]:
    """(path, access) for each path in a Bash command, in order (see the module docs)."""
    text = expand(command, home)
    breaks = [match.span() for match in _BREAK.finditer(text)]
    starts = [start for start, _ in breaks]
    found: list[tuple[str, Access]] = []
    for match in _PATH.finditer(text):
        index = bisect.bisect_right(starts, match.start())  # breaks never fall inside a path
        begin = breaks[index - 1][1] if index else 0
        end = breaks[index][0] if index < len(breaks) else len(text)
        found += _uses(text, begin, end, match)
    return found


def _uses(text: str, begin: int, end: int, match: re.Match[str]) -> list[tuple[str, Access]]:
    """How the simple command `text[begin:end]` uses the path `match` found in it."""
    path = match.group(1)
    verb = _verb(text[begin:end])
    before = text[begin : match.start()].rstrip(" \t'\"=")
    if _OUTPUT.search(before) or verb in _WRITERS:
        return [(path, "write")]
    if verb in _COPIERS and not text[match.end() : end].strip(" \t'\""):  # the destination
        if path.endswith("/") or temp_key(canonical(path)) == "":
            return _copied_into(text[begin:end], path) or [(path, "touch")]
        return [(path, "write")]
    return [(path, "touch" if verb in _TOUCHERS else "read")]


def _verb(segment: str) -> str:
    """The command name of a simple command, past `VAR=value` prefixes and wrappers like `env`."""
    for word in segment.split():
        word = word.strip("'\"!")
        if not word or word.isdigit() or word.startswith("-") or _ASSIGNMENT.fullmatch(word):
            continue
        name = word.rsplit("/", 1)[-1]
        if name not in _WRAPPERS:
            return name
    return ""


def _copied_into(segment: str, directory: str) -> list[tuple[str, Access]]:
    """The files `cp a b <directory>` writes: `<directory>/<name>` for each source."""
    try:
        words = [word for word in shlex.split(segment) if not word.startswith("-")]
    except ValueError:  # unbalanced quotes
        return []
    names = [word.rsplit("/", 1)[-1] for word in words]
    first = next((i for i, name in enumerate(names) if name in _COPIERS), None)
    if first is None:
        return []
    sources = [word.rstrip("/") for word in words[first + 1 : -1]]
    return [(posixpath.join(directory, posixpath.basename(s)), "write") for s in sources if s]
