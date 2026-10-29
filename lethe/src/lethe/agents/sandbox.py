"""Claude Code's sandbox for agents with a shell, and what their commands may read.

Sandboxed commands:
- are approved automatically;
- have no network;
- write only inside the workspace and the sandbox's temp directory.

They may read anywhere, though. `sandbox_settings` adds `denyRead` rules, so a command cannot read
what lies beside its agent's own workspace:
- a peer's workspace, or its own forgotten one;
- the run's archive and logs, and other runs: everything beside the path from the workspace up to
  `read_root`;
- the sandbox's temp directory, `/tmp/claude-<uid>`. It is `$TMPDIR` for every sandboxed Claude Code
  session of the user, so other runs leave files there;
- Claude Code's own logs, `~/.claude`.

In the sandbox a deny rule wins over everything, so neither the workspace nor any of its ancestors
may be listed. Denying writes to the temp directory is not possible: Claude Code records each
command's working directory there, and every command would fail. Writes there are therefore blind,
and Claude Code's other temp files go to a private directory inside the workspace, PRIVATE_TMP,
passed as CLAUDE_CODE_TMPDIR. Verified live with CLI 2.1.312.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

SHELL_SANDBOX_SETTINGS = '{"sandbox":{"enabled":true,"autoAllowBashIfSandboxed":true}}'
PRIVATE_TMP = ".lethe-tmp"  # inside the workspace; never archived (config.ALWAYS_EXCLUDE)


def shared_temp_dirs(uid: int) -> tuple[str, str]:
    """The sandbox's per-user temp directory under both of its macOS names."""
    return f"/tmp/claude-{uid}", f"/private/tmp/claude-{uid}"


def sandbox_settings(
    workspace: Path,
    *,
    read_root: Path | str | None,
    uid: int,
    home: Path,
    deny_read: Sequence[str] = (),
    keep: Sequence[str] = (),
) -> str:
    """The `--settings` JSON for one turn: the sandbox, plus reads denied beside `workspace`.

    Everything beside the path from `workspace` up to `read_root` is denied, as listed when the
    turn starts, except what holds a `keep` path: the agent's PATH, which under `uv run` starts in
    the project's own `.venv`. A workspace outside `read_root` gets only the fixed denials and
    `deny_read`.
    """
    kept = [Path(path).resolve() for path in keep if path]
    ws = Path(workspace).resolve()
    root = Path(read_root).resolve() if read_root is not None else None
    deny: list[str] = []
    if root is not None and root in ws.parents:
        node = ws
        while node != root:
            deny += [
                str(entry)
                for entry in sorted(node.parent.iterdir())
                if entry != node and not any(k == entry or entry in k.parents for k in kept)
            ]
            node = node.parent
    deny += [*shared_temp_dirs(uid), str(Path(home) / ".claude"), *deny_read]
    sandbox = {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "filesystem": {"denyRead": list(dict.fromkeys(deny))},
    }
    return json.dumps({"sandbox": sandbox}, separators=(",", ":"))
