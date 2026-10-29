"""Read isolation for agents with a shell: what the sandbox hides beside each agent's workspace.

Claude Code's sandbox lets commands read anywhere, so without these rules an agent could read a
peer's or its own forgotten workspace, other runs, the run's logs, Claude Code's session logs, or
files that other runs left in the sandbox's shared temp directory (`/tmp/claude-<uid>`, which is
every sandboxed session's `$TMPDIR`). Verified live (CLI 2.1.312): with these `denyRead` rules
the reads fail with "Operation not permitted" while ordinary commands run normally.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

from lethe.agents.claude_code import (
    SHELL_SANDBOX_SETTINGS,
    ClaudeCodeBackend,
    ClaudeCodeOptions,
    sandbox_settings,
)
from lethe.domain.types import SessionChoice

SID = "11111111-2222-4333-8444-555555555555"
NEW = SessionChoice(SID, resume=False)


def layout(root: Path) -> Path:
    """A runs root with two runs; returns agent a1's round-2 workspace in the first run."""
    run = root / "exp" / "arm" / "rep1"
    for path in ("ws/a1/e1", "ws/a1/e2", "ws/a2/e1", "archive.git", "transcripts/a1"):
        (run / path).mkdir(parents=True)
    (run / "events.jsonl").write_text("{}\n")
    (root / "exp" / "arm" / "rep2" / "ws").mkdir(parents=True)
    (root / "other-exp").mkdir()
    return run / "ws" / "a1" / "e2"


def denied(settings: str) -> set[str]:
    data = json.loads(settings)
    assert data["sandbox"]["enabled"] is True and data["sandbox"]["autoAllowBashIfSandboxed"]
    return set(data["sandbox"]["filesystem"]["denyRead"])


def test_everything_beside_the_workspace_up_to_the_root_is_hidden(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    ws = layout(root)
    run = ws.parents[2]

    hidden = denied(sandbox_settings(ws, read_root=root, uid=501, home=tmp_path / "home"))

    for path in (
        run / "ws" / "a1" / "e1",  # its own forgotten round
        run / "ws" / "a2",  # a peer
        run / "archive.git",
        run / "events.jsonl",
        run / "transcripts",
        root / "exp" / "arm" / "rep2",  # another run
        root / "other-exp",
    ):
        assert str(path) in hidden, path
    for path in (ws, *ws.parents):
        assert str(path) not in hidden, path  # deny wins in the sandbox: never an ancestor


def test_the_shared_temp_dir_and_claude_codes_logs_are_always_hidden(tmp_path: Path) -> None:
    ws = tmp_path / "elsewhere" / "ws"
    ws.mkdir(parents=True)

    hidden = denied(sandbox_settings(ws, read_root=tmp_path / "runs", uid=777, home=tmp_path))

    assert {"/tmp/claude-777", "/private/tmp/claude-777", str(tmp_path / ".claude")} <= hidden
    assert not any("elsewhere" in path for path in hidden)  # not under the root: no siblings


def test_extra_paths_are_hidden_too(tmp_path: Path) -> None:
    ws = layout(tmp_path / "runs")
    extra = (str(tmp_path / "docs"), str(tmp_path / "notes.md"))

    hidden = denied(
        sandbox_settings(ws, read_root=tmp_path / "runs", uid=1, home=tmp_path, deny_read=extra)
    )

    assert set(extra) <= hidden


def test_build_argv_carries_the_settings_it_is_given() -> None:
    backend = ClaudeCodeBackend(ClaudeCodeOptions(shell=True))
    argv = backend.build_argv(NEW, settings='{"sandbox":{}}')
    assert argv[argv.index("--settings") + 1] == '{"sandbox":{}}'
    default = backend.build_argv(NEW)
    assert default[default.index("--settings") + 1] == SHELL_SANDBOX_SETTINGS


def run(backend: ClaudeCodeBackend, workspace: Path, transcript: Path) -> Any:
    return asyncio.run(
        backend.run_turn(
            agent="a1",
            workspace=workspace,
            prompt="Improve the solver.",
            session=NEW,
            transcript_path=transcript,
        )
    )


def test_a_turn_with_a_shell_runs_isolated_with_a_private_claude_temp_dir(
    fake_claude: Any, tmp_path: Path, transcript: Path
) -> None:
    root = tmp_path / "runs"
    ws = layout(root)
    fake_claude.script(stream="turn1.jsonl")

    result = run(fake_claude.backend(shell=True, read_root=str(root)), ws, transcript)

    assert result.ok, result.error
    argv = fake_claude.argv
    assert str(ws.parent / "e1") in denied(argv[argv.index("--settings") + 1])
    assert "CLAUDE_CODE_TMPDIR" in fake_claude.env_names
    private = ws / ".lethe-tmp"
    assert private.is_dir() and stat.S_IMODE(private.stat().st_mode) == 0o700


def test_a_turn_without_a_shell_has_no_sandbox_settings_or_temp_dir(
    fake_claude: Any, workspace: Path, transcript: Path
) -> None:
    fake_claude.script(stream="turn1.jsonl")

    run(fake_claude.backend(read_root=str(workspace.parent)), workspace, transcript)

    assert "--settings" not in fake_claude.argv
    assert "CLAUDE_CODE_TMPDIR" not in fake_claude.env_names
    assert not (workspace / ".lethe-tmp").exists()


def test_nothing_holding_a_directory_on_path_is_hidden(tmp_path: Path) -> None:
    # `uv run` puts the project's .venv/bin first on PATH: hiding .venv would hide `python3`.
    root = tmp_path / "repo"
    ws = layout(root / "runs")
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (root / "docs").mkdir()

    hidden = denied(
        sandbox_settings(ws, read_root=root, uid=1, home=tmp_path, keep=(str(venv_bin), "/usr/bin"))
    )

    assert str(root / ".venv") not in hidden
    assert str(root / "docs") in hidden
