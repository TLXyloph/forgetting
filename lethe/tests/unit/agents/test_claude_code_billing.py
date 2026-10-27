"""Subscription safety and metering of `ClaudeCodeBackend` (against the fake `claude`).

In safe isolation the backend launches `claude` without the variables that switch it to API-key
or cloud billing, and refuses any turn whose init line still reports an API key. A successful
result is never a rate-limit failure, and without per-turn usage the cost of a turn is the growth
of its session's cumulative total.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from lethe.agents import claude_code
from lethe.agents.claude_code import ClaudeCodeBackend
from lethe.domain.types import SessionChoice, TurnResult

BILLING_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
REFUSED = (
    "safe mode refused API-key billing (apiKeySource=ANTHROPIC_API_KEY); "
    "set agent.allow_api_billing = true to allow it"
)
NEW = SessionChoice("sess-1", resume=False)


def run(backend: ClaudeCodeBackend, workspace: Path, transcript: Path, session=NEW) -> TurnResult:
    turn = backend.run_turn(
        agent="a1", workspace=workspace, prompt="go", session=session, transcript_path=transcript
    )
    return asyncio.run(turn)


def stream(
    tmp_path: Path,
    name: str,
    *,
    source: str | None = "none",
    rate_status: str | None = None,
    result: bool = True,
    is_error: bool = False,
    total: float | None = None,
    per_turn: float | None = None,
    session: str = "sess-1",
) -> Path:
    """An ad-hoc stream-json file: init line, optional rate-limit event, optional result."""
    init: dict[str, Any] = {"type": "system", "subtype": "init", "session_id": session}
    if source is not None:
        init["apiKeySource"] = source
    lines: list[dict[str, Any]] = [init]
    if rate_status is not None:
        window = {"utilization": 1.0, "resetsAt": 1793000000, "status": rate_status}
        info = {"status": rate_status, "resetsAt": 1793050200}
        info["unifiedWindows"] = {"five_hour": window}
        lines.append({"type": "rate_limit_event", "rate_limit_info": info})
    if result:
        end: dict[str, Any] = {"type": "result", "subtype": "success", "is_error": is_error}
        end |= {"result": "Usage limit reached" if is_error else "done", "session_id": session}
        if total is not None:
            end["total_cost_usd"] = total
        if per_turn is not None:
            end["model_usage_since_last_result"] = {"m": {"costUSD": per_turn}}
        lines.append(end)
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


# --- B1: which environment `claude` is launched with ----------------------------------------


def test_the_billing_variables_are_the_documented_six():
    assert tuple(claude_code.API_BILLING_ENV) == BILLING_VARS


@pytest.mark.parametrize(
    ("options", "removed"),
    [
        ({}, True),
        ({"allow_api_billing": True}, False),
        ({"isolation": "bare"}, False),
        ({"isolation": "none"}, False),
    ],
    ids=["safe", "safe-allowed", "bare", "none"],
)
def test_safe_mode_launches_claude_without_the_api_billing_variables(
    fake_claude, workspace, transcript, monkeypatch, options, removed
):
    for name in BILLING_VARS:
        monkeypatch.setenv(name, "sk-test-not-a-real-key")
    monkeypatch.setenv("UNRELATED_SETTING", "kept")
    fake_claude.script(stream="turn1.jsonl")

    assert run(fake_claude.backend(**options), workspace, transcript).ok

    seen = fake_claude.env_names
    assert "UNRELATED_SETTING" in seen and "PATH" in seen
    assert not set(BILLING_VARS) & seen if removed else set(BILLING_VARS) <= seen


# --- B2: refusing a turn that reports an API key --------------------------------------------


def test_safe_mode_refuses_a_turn_billed_to_an_api_key(
    fake_claude, workspace, transcript, tmp_path
):
    keyed = stream(tmp_path, "keyed", source="ANTHROPIC_API_KEY", total=0.25, per_turn=0.25)
    fake_claude.script(stream=keyed)

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind, result.error) == (False, "billing", REFUSED)
    assert result.cost_usd == pytest.approx(0.25)  # what it cost is still recorded


def test_billing_beats_every_other_failure_class(fake_claude, workspace, transcript, tmp_path):
    keyed = stream(tmp_path, "keyed", source="ANTHROPIC_API_KEY", rate_status="rejected")
    fake_claude.script(stream=keyed, exit_code=1)
    assert run(fake_claude.backend(), workspace, transcript).error_kind == "billing"


@pytest.mark.parametrize(
    "options", [{"allow_api_billing": True}, {"isolation": "bare"}, {"isolation": "none"}]
)
def test_api_key_billing_is_accepted_when_allowed(
    fake_claude, workspace, transcript, tmp_path, options
):
    fake_claude.script(stream=stream(tmp_path, "keyed", source="ANTHROPIC_API_KEY", total=0.1))
    assert run(fake_claude.backend(**options), workspace, transcript).ok


@pytest.mark.parametrize("source", ["none", None])
def test_a_subscription_turn_is_not_refused(fake_claude, workspace, transcript, tmp_path, source):
    fake_claude.script(stream=stream(tmp_path, "sub", source=source, total=0.1))
    assert run(fake_claude.backend(), workspace, transcript).ok


# --- B4: a successful result is never a rate-limit failure ------------------------------------


def test_a_successful_result_wins_over_a_rejected_rate_limit_event(
    fake_claude, workspace, transcript, tmp_path
):
    fake_claude.script(stream=stream(tmp_path, "s", rate_status="rejected", per_turn=0.1))

    result = run(fake_claude.backend(), workspace, transcript)

    assert (result.ok, result.error_kind) == (True, None)
    assert result.cost_usd == pytest.approx(0.1)
    assert result.rate_limit is not None and result.rate_limit.status == "rejected"  # pacer


def test_a_rejected_event_without_a_result_is_a_rate_limit(
    fake_claude, workspace, transcript, tmp_path
):
    fake_claude.script(stream=stream(tmp_path, "s", rate_status="rejected", result=False))
    assert run(fake_claude.backend(), workspace, transcript).error_kind == "rate_limit"


# --- B6: per-session cost when the CLI gives no per-turn usage --------------------------------


def test_without_per_turn_usage_the_cost_is_the_growth_of_the_session_total(
    fake_claude, workspace, transcript, tmp_path
):
    backend = fake_claude.backend()

    def cost(name: str, session: SessionChoice, reported: str, total: float) -> float:
        fake_claude.script(stream=stream(tmp_path, name, session=reported, total=total))
        return run(backend, workspace, transcript, session=session).cost_usd

    assert cost("t1", SessionChoice("sess-1", False), "sess-1", 0.5) == pytest.approx(0.5)
    assert cost("t2", SessionChoice("sess-1", True), "sess-1", 0.8) == pytest.approx(0.3)
    # a resumed session may come back under a new id, still carrying its parent's total
    assert cost("t3", SessionChoice("sess-1", True), "sess-fork", 1.0) == pytest.approx(0.2)
    assert cost("t4", SessionChoice("sess-fork", True), "sess-fork", 1.5) == pytest.approx(0.5)
    assert cost("t5", SessionChoice("sess-2", False), "sess-2", 0.4) == pytest.approx(0.4)


def test_per_turn_usage_still_wins_when_present(fake_claude, workspace, transcript, tmp_path):
    backend = fake_claude.backend()
    fake_claude.script(stream=stream(tmp_path, "a", total=0.5, per_turn=0.5))
    run(backend, workspace, transcript)
    fake_claude.script(stream=stream(tmp_path, "b", total=0.9, per_turn=0.25))
    resumed = run(backend, workspace, transcript, session=SessionChoice("sess-1", True))
    assert resumed.cost_usd == pytest.approx(0.25)
