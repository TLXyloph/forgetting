"""`ClaudeCodeBackend.build_argv` and option validation (no subprocess involved)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lethe.agents.claude_code import (
    DEFAULT_TOOLS,
    SHELL_SANDBOX_SETTINGS,
    ClaudeCodeBackend,
    ClaudeCodeOptions,
    pinned,
)
from lethe.domain.types import SessionChoice

SID = "11111111-2222-4333-8444-555555555555"
NEW = SessionChoice(SID, resume=False)
RESUME = SessionChoice(SID, resume=True)

HEAD = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
SAFE = ["--safe-mode", "--restricted", "--disable-slash-commands"]
BARE = ["--bare", "--restricted", "--disable-slash-commands"]


def argv(session: SessionChoice = NEW, **options: object) -> list[str]:
    return ClaudeCodeBackend(ClaudeCodeOptions(**options)).build_argv(session)  # type: ignore[arg-type]


def test_defaults_are_the_safe_subscription_friendly_set() -> None:
    assert DEFAULT_TOOLS == ("Read", "Edit", "Write", "Glob", "Grep")
    assert ClaudeCodeBackend().build_argv(NEW) == [
        *HEAD,
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read,Edit,Write,Glob,Grep",
        *SAFE,
        "--model",
        "sonnet",
        "--session-id",
        SID,
    ]


def tools_arg(session: SessionChoice = NEW, **options: object) -> str:
    result = argv(session, **options)
    return result[result.index("--tools") + 1]


def test_none_options_are_the_same_as_defaults() -> None:
    assert ClaudeCodeBackend(None).build_argv(NEW) == ClaudeCodeBackend().build_argv(NEW)
    assert ClaudeCodeBackend().options == ClaudeCodeOptions()


def test_bare_isolation_swaps_safe_mode_for_bare() -> None:
    assert argv(isolation="bare") == [
        *HEAD,
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read,Edit,Write,Glob,Grep",
        *BARE,
        "--model",
        "sonnet",
        "--session-id",
        SID,
    ]


def test_no_isolation_adds_no_isolation_flags() -> None:
    assert argv(isolation="none") == [
        *HEAD,
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read,Edit,Write,Glob,Grep",
        "--model",
        "sonnet",
        "--session-id",
        SID,
    ]


def test_shell_appends_a_sandboxed_bash_to_the_tools() -> None:
    assert argv(shell=True) == [
        *HEAD,
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read,Edit,Write,Glob,Grep,Bash",
        *SAFE,
        "--settings",
        SHELL_SANDBOX_SETTINGS,
        "--model",
        "sonnet",
        "--session-id",
        SID,
    ]


def test_the_sandbox_confines_bash_and_approves_it_without_a_person() -> None:
    # Headless, an unsandboxed Bash call needs an approval nobody can give: even `python3` was
    # refused (live check, CLI 2.1.311). Sandboxed, commands run with writes confined to the
    # workspace and no network, and are approved automatically.
    assert json.loads(SHELL_SANDBOX_SETTINGS) == {
        "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True}
    }


def test_any_bash_is_sandboxed_and_no_bash_means_no_sandbox_settings() -> None:
    assert "--settings" in argv(tools=("Read", "Bash"))  # Bash named directly, without shell
    assert "--settings" not in argv()
    assert "--settings" not in argv(tools=())


def test_shell_does_not_duplicate_bash() -> None:
    assert tools_arg(shell=True, tools=("Read", "Bash")) == "Read,Bash"
    assert tools_arg(shell=False, tools=("Read", "Bash")) == "Read,Bash"  # explicit Bash stays


def test_model_none_omits_the_flag() -> None:
    result = argv(model=None)
    assert "--model" not in result
    assert result[-2:] == ["--session-id", SID]


def test_effort_follows_the_model() -> None:
    assert argv(model="opus", effort="high") == [
        *HEAD,
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read,Edit,Write,Glob,Grep",
        *SAFE,
        "--model",
        "opus",
        "--effort",
        "high",
        "--session-id",
        SID,
    ]


def test_effort_without_a_model() -> None:
    result = argv(model=None, effort="low")
    assert result[-4:] == ["--effort", "low", "--session-id", SID]
    assert "--model" not in result


def test_resume_uses_resume_instead_of_session_id() -> None:
    result = argv(RESUME)
    assert result[-2:] == ["--resume", SID]
    assert "--session-id" not in result
    assert result == [*argv(NEW)[:-2], "--resume", SID]  # nothing else differs


def test_extra_args_come_last() -> None:
    result = argv(extra_args=("--max-turns", "5", "--name", "x y"))
    assert result[-6:] == ["--session-id", SID, "--max-turns", "5", "--name", "x y"]
    assert argv(RESUME, extra_args=("--flag",))[-3:] == ["--resume", SID, "--flag"]


def test_custom_executable_permission_mode_and_tools() -> None:
    assert argv(
        executable="/opt/bin/claude",
        permission_mode="plan",
        tools=("Read", "Grep"),
        isolation="none",
        model=None,
    ) == [
        "/opt/bin/claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "plan",
        "--tools",
        "Read,Grep",
        "--session-id",
        SID,
    ]


def test_empty_tools_disable_every_tool() -> None:
    assert tools_arg(tools=()) == ""  # the CLI reads "" as "no tools"
    assert tools_arg(tools=(), shell=True) == "Bash"


def test_build_argv_is_pure() -> None:
    backend = ClaudeCodeBackend()
    assert backend.build_argv(NEW) == backend.build_argv(NEW)
    first = backend.build_argv(NEW)
    first.append("mutated")
    assert backend.build_argv(NEW)[-1] == SID  # callers get a fresh list each time


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize("isolation", ["strict", "", "SAFE", None, "bare "])
def test_unknown_isolation_is_rejected_at_construction(isolation: object) -> None:
    with pytest.raises(ValueError, match="isolation"):
        ClaudeCodeBackend(ClaudeCodeOptions(isolation=isolation))  # type: ignore[arg-type]


@pytest.mark.parametrize("isolation", ["safe", "bare", "none"])
def test_known_isolations_are_accepted(isolation: str) -> None:
    ClaudeCodeBackend(ClaudeCodeOptions(isolation=isolation))  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0, -1.0, math.nan, -math.inf])
def test_a_non_positive_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="turn_timeout_s"):
        ClaudeCodeBackend(ClaudeCodeOptions(turn_timeout_s=timeout))


def test_an_infinite_timeout_means_no_timeout_and_is_accepted() -> None:
    ClaudeCodeBackend(ClaudeCodeOptions(turn_timeout_s=math.inf))


@pytest.mark.parametrize(
    "session_id", ["", "-rf", "--dangerously-skip-permissions", "a b", "a\nb", "a\x00b"]
)
@pytest.mark.parametrize("resume", [False, True])
def test_session_ids_that_could_pass_for_flags_or_break_argv_are_rejected(
    session_id: str, resume: bool
) -> None:
    with pytest.raises(ValueError, match="session id"):
        ClaudeCodeBackend().build_argv(SessionChoice(session_id, resume=resume))


@pytest.mark.parametrize("session_id", [SID, "s1", "sid-1", "agent_1.epoch2:turn3", "0"])
def test_ordinary_session_ids_are_accepted(session_id: str) -> None:
    assert ClaudeCodeBackend().build_argv(SessionChoice(session_id, resume=False))[-1] == session_id


def test_pinned_resolves_the_cli_to_the_versioned_binary_it_links_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude Code installs `claude` as a link to versions/<x.y.z>; an update re-points the link,
    # so a run that looked `claude` up on every turn could change CLI version mid-run.
    binary = tmp_path / "versions" / "2.1.312"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").symlink_to(binary)
    monkeypatch.setenv("PATH", str(bin_dir))

    options = pinned(ClaudeCodeOptions(model="haiku"))

    assert options.executable == str(binary.resolve())
    assert options.model == "haiku"


def test_pinned_leaves_a_cli_it_cannot_find_to_fail_at_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert pinned(ClaudeCodeOptions()).executable == "claude"
