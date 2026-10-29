"""The paths a Claude Code tool call names, and whether it reads, writes or only touches them.

`research.leaks` decides which of these paths cross a boundary; this module only finds them.

Bash commands are not parsed by a shell. After `$TMPDIR` (also `${TMPDIR}`, `${TMPDIR:-...}`),
`$HOME`, `${HOME}` and a leading `~` are expanded, regular expressions find absolute paths and
relative ones that climb out with `..`, anywhere in the command, quoted code included. `cd` (and
`pushd`) is followed: relative paths are joined to the directory the shell is in by then, and once
that directory is outside the workspace, plain path-like arguments count too (`cd ..; cat e1/x`).
How a path is used is read off the simple command around it (the text between `;`, `&&`, `||`,
`|`, `&`, newlines, parentheses and backquotes):

- write: after a redirection (`>`, `>>`), as an argument of `tee` or `touch` or of an output
  option (`-o`, `--output`, `--out`), or as the last argument of `cp`, `mv`, `install`, `ln` or
  `rsync` (redirections after it aside); a copy into a directory (`cp a b $TMPDIR/`) writes
  `<directory>/<name>` per source;
- touch: as an argument of a command that lists, names or removes (`ls`, `echo`, `cd`, `find`,
  `stat`, `rm`, ...);
- read: anything else (`cat`, `python`, `diff`, the source of a copy, ...); but a bare climb
  (`..`, `../..`, as in code) names a directory, so it reads only for a command that reads
  directories recursively (`grep -r`, `find`, `cp -r`, `rsync`, `tar`, `du`, `zip`, `rg`) and
  touches otherwise.

File tools: Read and Grep read, Write, Edit, MultiEdit and NotebookEdit write, Glob touches; their
paths are `file_path`, `notebook_path`, `path` and an absolute Glob `pattern`.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, Literal

__all__ = [
    "TEMP_VAR_ROOT",
    "Access",
    "bash_paths",
    "bash_walk",
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
_REDIRECT = re.compile(r"\s*[0-9]*(?:>>?|<)&?\s*\S*")  # `2>/dev/null`, `>&2`, `< in.txt`
_WORD_PATH = re.compile(r"[\w.@+-]+(?:/[\w.@+-]*)*")  # a plain relative path, no spaces
_PRIVATE = re.compile(r"^/private(?=/(?:tmp|var|etc)(?:/|$))")
_TEMP_ROOT = re.compile(r"/tmp/claude-[0-9]+(?:/(.*))?")

_WRITERS = frozenset({"tee", "touch"})
_COPIERS = frozenset({"cp", "mv", "install", "ln", "rsync"})
_TOUCHERS = frozenset(
    {"basename", "cd", "df", "dirname", "du", "echo", "export", "find", "ls", "mkdir", "printf",
     "pushd", "readlink", "realpath", "rm", "rmdir", "stat", "test", "tree", "[", "[["}
)  # fmt: skip
_TREE_READERS = frozenset({"find", "rsync", "tar", "du", "zip", "rg"})
_RECURSIVE_FLAG = re.compile(r"\s-[A-Za-z]*[rRa]")  # grep -rn, cp -a, ...
_WRAPPERS = frozenset(
    {"command", "env", "exec", "nice", "nohup", "sudo", "time", "timeout",
     "if", "then", "else", "elif", "while", "until", "do", "for", "!"}
)  # fmt: skip
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
    """(path, access) for each path in a Bash command run in the workspace (see the module docs)."""
    return bash_walk(command, home)[0]


def bash_walk(
    command: str, home: str, cwd: str = "", workspaces: Sequence[str] = ()
) -> tuple[list[tuple[str, Access]], str]:
    """`bash_paths` for a command run in `cwd` (relative to the workspace, "" for the workspace
    itself, or absolute), with relative paths joined to the directory the shell is in by then;
    also returns the directory the shell ends in. `workspaces` are absolute spellings of the
    workspace, so that a `cd` into it by absolute path stays relative (inside)."""
    text = expand(command, home)
    found: list[tuple[str, Access]] = []
    begin = 0
    for begin, end in _segments(text):
        segment = text[begin:end]
        verb = _verb(segment)
        if verb in ("cd", "pushd"):
            cwd = _relative_to(_changed_dir(segment, cwd, home), workspaces)
            continue
        for match in _PATH.finditer(text, begin, end):
            found += [(_join(cwd, path), access) for path, access in _uses(text, begin, end, match)]
        if cwd.startswith("/") or cwd.split("/")[0] == "..":  # the shell left the workspace
            access: Access = "touch" if verb in _TOUCHERS else "read"
            found += [(_join(cwd, word), access) for word in _plain_paths(segment)]
    return found, cwd


def _segments(text: str) -> list[tuple[int, int]]:
    """(begin, end) of each simple command in `text`, in order."""
    spans, start = [], 0
    for match in _BREAK.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    return [*spans, (start, len(text))]


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:  # unbalanced quotes
        return segment.split()


def _changed_dir(segment: str, cwd: str, home: str) -> str:
    """The shell's directory after `cd <target>` run in `cwd`."""
    words = _words(segment)
    verb = next((i for i, w in enumerate(words) if w.rsplit("/", 1)[-1] in ("cd", "pushd")), 0)
    targets = [w for w in words[verb + 1 :] if not w.startswith("-")]
    if not targets:
        return home if "-" not in words[verb + 1 :] else cwd
    return _join(cwd, targets[0])


def _relative_to(cwd: str, workspaces: Sequence[str]) -> str:
    """`cwd` relative to the workspace when it is an absolute path inside one of its spellings."""
    for ws in workspaces:
        if cwd == ws or cwd.startswith(ws.rstrip("/") + "/"):
            return cwd[len(ws) :].strip("/")
    return cwd


def _join(cwd: str, path: str) -> str:
    if path.startswith("/") or not cwd:
        return path
    joined = posixpath.normpath(posixpath.join(cwd, path))
    return "" if joined == "." else joined


def _plain_paths(segment: str) -> list[str]:
    """The arguments of a simple command that look like files (with a `.` or `/`; `grep -q done`
    has a pattern, not a file), not its name, options or assignments."""
    words = _words(segment)
    verb = _verb(segment)
    after = next((i + 1 for i, w in enumerate(words) if w.rsplit("/", 1)[-1] == verb), len(words))
    return [
        w
        for w in words[after:]
        if not w.startswith("-")
        and _WORD_PATH.fullmatch(w)
        and w.strip("./")  # a bare climb names a directory, not a file
        and any(c in w for c in "./")
    ]


def _uses(text: str, begin: int, end: int, match: re.Match[str]) -> list[tuple[str, Access]]:
    """How the simple command `text[begin:end]` uses the path `match` found in it."""
    path = match.group(1)
    verb = _verb(text[begin:end])
    before = text[begin : match.start()].rstrip(" \t'\"=")
    if _OUTPUT.search(before) or verb in _WRITERS:
        return [(path, "write")]
    rest = _REDIRECT.sub("", text[match.end() : end])
    if verb in _COPIERS and not rest.strip(" \t'\""):  # the destination
        if path.endswith("/") or temp_key(canonical(path)) == "":
            return _copied_into(text[begin:end], path) or [(path, "touch")]
        return [(path, "write")]
    if verb in _TOUCHERS:
        return [(path, "touch")]
    if not path.strip("./"):  # a bare climb: a directory, read only by a recursive command
        segment = text[begin:end]
        recursive = verb in _TREE_READERS or (
            verb in ("grep", "egrep", "cp") and _RECURSIVE_FLAG.search(segment)
        )
        return [(path, "read" if recursive else "touch")]
    return [(path, "read")]


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
