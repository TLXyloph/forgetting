"""`ScriptedBackend`: the deterministic stand-in for a coding agent."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

import pytest

from lethe.agents.base import AgentBackend
from lethe.agents.claude_code import ClaudeCodeBackend
from lethe.agents.scripted import ScriptCall, ScriptedBackend, ScriptReply
from lethe.domain.types import (
    RateLimitInfo,
    SessionChoice,
    ToolCall,
    TurnErrorKind,
    TurnResult,
    Usage,
)


class Recorder:
    """A script that remembers every call and returns a fixed reply."""

    def __init__(self, reply: ScriptReply | None = None) -> None:
        self.calls: list[ScriptCall] = []
        self.reply = reply if reply is not None else ScriptReply()

    def __call__(self, call: ScriptCall) -> ScriptReply:
        self.calls.append(call)
        return self.reply


def new(session_id: str) -> SessionChoice:
    return SessionChoice(session_id, resume=False)


def resume(session_id: str) -> SessionChoice:
    return SessionChoice(session_id, resume=True)


def run_turn(
    backend: ScriptedBackend,
    workspace: Path,
    transcript: Path,
    *,
    session: SessionChoice,
    agent: str = "agent-1",
    prompt: str = "improve it",
) -> TurnResult:
    return asyncio.run(
        backend.run_turn(
            agent=agent,
            workspace=workspace,
            prompt=prompt,
            session=session,
            transcript_path=transcript,
        )
    )


# --- what the script sees ---------------------------------------------------------------------


def test_the_script_receives_the_turn(workspace: Path, transcript: Path) -> None:
    script = Recorder()
    session = new("s1")

    run_turn(
        ScriptedBackend(script),
        workspace,
        transcript,
        session=session,
        agent="agent-7",
        prompt="do the thing",
    )

    assert script.calls == [
        ScriptCall(
            agent="agent-7",
            workspace=workspace,
            prompt="do the thing",
            session=session,
            turn_in_session=1,
        )
    ]


def test_turns_are_numbered_per_session_across_calls(workspace: Path, transcript: Path) -> None:
    script = Recorder()
    backend = ScriptedBackend(script)

    sequence = [new("s1"), resume("s1"), new("s2"), resume("s1"), resume("s2"), resume("s1")]
    for session in sequence:
        assert run_turn(backend, workspace, transcript, session=session).ok

    assert [(c.session.session_id, c.turn_in_session) for c in script.calls] == [
        ("s1", 1),
        ("s1", 2),
        ("s2", 1),
        ("s1", 3),
        ("s2", 2),
        ("s1", 4),
    ]


def test_a_new_session_that_reuses_an_id_starts_over_at_turn_one(
    workspace: Path, transcript: Path
) -> None:
    script = Recorder()
    backend = ScriptedBackend(script)
    for session in (new("s1"), resume("s1"), new("s1"), resume("s1")):
        run_turn(backend, workspace, transcript, session=session)
    assert [c.turn_in_session for c in script.calls] == [1, 2, 1, 2]


def test_backends_do_not_share_session_state(workspace: Path, transcript: Path) -> None:
    first, second = ScriptedBackend(Recorder()), ScriptedBackend(Recorder())
    run_turn(first, workspace, transcript, session=new("s1"))
    result = run_turn(second, workspace, transcript, session=resume("s1"))
    assert result.error_kind == "process"


# --- resuming something that never existed ----------------------------------------------------


def test_resuming_an_unknown_session_is_a_process_error_without_calling_the_script(
    workspace: Path, transcript: Path
) -> None:
    script = Recorder()

    result = run_turn(ScriptedBackend(script), workspace, transcript, session=resume("ghost"))

    assert script.calls == []
    assert result.ok is False
    assert result.error_kind == "process"
    assert result.error is not None and result.error.startswith("unknown session")
    assert "ghost" in result.error
    assert result.session_id == "ghost"
    assert result.final_text == ""
    assert result.cost_usd == 0.0
    assert result.usage == Usage()
    assert result.tool_calls == ()
    assert result.rate_limit is None


def test_a_failed_resume_does_not_create_the_session(workspace: Path, transcript: Path) -> None:
    script = Recorder()
    backend = ScriptedBackend(script)
    assert run_turn(backend, workspace, transcript, session=resume("ghost")).error_kind == "process"
    assert run_turn(backend, workspace, transcript, session=resume("ghost")).error_kind == "process"
    assert script.calls == []
    assert not transcript.exists()  # no turn happened, so nothing was recorded


# --- the reply becomes the TurnResult ---------------------------------------------------------


def test_reply_fields_map_onto_the_turn_result(workspace: Path, transcript: Path) -> None:
    reply = ScriptReply(
        final_text="I tightened the loop",
        cost_usd=0.25,
        output_tokens=7,
        thinking_tokens=3,
        tool_calls=(ToolCall("Read", "a.py"), ToolCall("Edit", "a.py"), ToolCall("Edit", "b.py")),
        rate_limit=RateLimitInfo("allowed", 0.4, 0.1, 1793050200.0),
    )

    result = run_turn(ScriptedBackend(Recorder(reply)), workspace, transcript, session=new("s1"))

    assert result.ok is True
    assert result.error_kind is None and result.error is None
    assert result.session_id == "s1"
    assert result.final_text == "I tightened the loop"
    assert result.cost_usd == 0.25
    assert result.usage == Usage(output_tokens=7, thinking_tokens=3)  # everything else is 0
    assert result.tool_calls == reply.tool_calls
    assert result.rate_limit == RateLimitInfo("allowed", 0.4, 0.1, 1793050200.0)
    assert (result.files_read, result.edit_calls) == (1, 2)


def test_default_reply(workspace: Path, transcript: Path) -> None:
    result = run_turn(ScriptedBackend(Recorder()), workspace, transcript, session=new("s1"))
    assert result.ok
    assert result.final_text == "done"
    assert result.cost_usd == 0.01
    assert result.usage == Usage(output_tokens=100)
    assert result.tool_calls == ()
    assert result.rate_limit is None


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "process", "api"])
def test_error_kind_passes_through_and_means_not_ok(
    kind: TurnErrorKind, workspace: Path, transcript: Path
) -> None:
    reply = ScriptReply(
        final_text="partial",
        cost_usd=0.5,
        error_kind=kind,
        rate_limit=RateLimitInfo("rejected", resets_at=1793050200.0),
    )

    result = run_turn(ScriptedBackend(Recorder(reply)), workspace, transcript, session=new("s1"))

    assert result.ok is False
    assert result.error_kind == kind
    assert result.error and kind in result.error  # a short human message
    # the rest of the reply still comes through, so budgets and pacing see it
    assert result.final_text == "partial"
    assert result.cost_usd == 0.5
    assert result.rate_limit == RateLimitInfo("rejected", resets_at=1793050200.0)


def test_a_failed_turn_still_counts_as_a_turn_of_the_session(
    workspace: Path, transcript: Path
) -> None:
    replies = iter([ScriptReply(error_kind="api"), ScriptReply()])
    seen: list[int] = []

    def script(call: ScriptCall) -> ScriptReply:
        seen.append(call.turn_in_session)
        return next(replies)

    backend = ScriptedBackend(script)
    assert run_turn(backend, workspace, transcript, session=new("s1")).error_kind == "api"
    assert run_turn(backend, workspace, transcript, session=resume("s1")).ok
    assert seen == [1, 2]


def test_duration_is_measured(workspace: Path, transcript: Path) -> None:
    def slow(call: ScriptCall) -> ScriptReply:
        time.sleep(0.05)
        return ScriptReply()

    result = run_turn(ScriptedBackend(slow), workspace, transcript, session=new("s1"))
    assert 0.04 <= result.duration_s < 5


def test_script_exceptions_propagate(workspace: Path, transcript: Path) -> None:
    def broken(call: ScriptCall) -> ScriptReply:
        raise RuntimeError("bug in the test script")

    with pytest.raises(RuntimeError, match="bug in the test script"):
        run_turn(ScriptedBackend(broken), workspace, transcript, session=new("s1"))


# --- the workspace ----------------------------------------------------------------------------


def test_edits_the_script_makes_to_the_workspace_persist(workspace: Path, transcript: Path) -> None:
    def script(call: ScriptCall) -> ScriptReply:
        target = call.workspace / "solver.py"
        previous = target.read_text() if target.exists() else ""
        target.write_text(previous + f"# turn {call.turn_in_session}\n")
        return ScriptReply(tool_calls=(ToolCall("Edit", "solver.py"),))

    backend = ScriptedBackend(script)
    run_turn(backend, workspace, transcript, session=new("s1"))
    result = run_turn(backend, workspace, transcript, session=resume("s1"))

    assert (workspace / "solver.py").read_text() == "# turn 1\n# turn 2\n"
    assert result.edit_calls == 1


# --- transcript -------------------------------------------------------------------------------


def test_transcript_gets_one_json_line_per_turn(workspace: Path, transcript: Path) -> None:
    replies = iter(
        [ScriptReply(final_text="first"), ScriptReply(final_text="second — with ünïcode ✓")]
    )
    backend = ScriptedBackend(lambda call: next(replies))
    assert not transcript.parent.exists()

    run_turn(backend, workspace, transcript, session=new("s1"), agent="a1", prompt="p1")
    run_turn(backend, workspace, transcript, session=resume("s1"), agent="a1", prompt="p2\nline")

    lines = transcript.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records == [
        {
            "agent": "a1",
            "session_id": "s1",
            "turn_in_session": 1,
            "prompt": "p1",
            "final_text": "first",
        },
        {
            "agent": "a1",
            "session_id": "s1",
            "turn_in_session": 2,
            "prompt": "p2\nline",
            "final_text": "second — with ünïcode ✓",
        },
    ]


def test_transcript_also_records_failed_turns(workspace: Path, transcript: Path) -> None:
    backend = ScriptedBackend(Recorder(ScriptReply(final_text="oops", error_kind="timeout")))
    run_turn(backend, workspace, transcript, session=new("s1"))
    (record,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert record["final_text"] == "oops" and record["turn_in_session"] == 1


def test_transcript_survives_text_that_json_would_choke_on(
    workspace: Path, transcript: Path
) -> None:
    prompt = "lone surrogate: \ud800 and a NUL: \x00"
    backend = ScriptedBackend(Recorder())
    run_turn(backend, workspace, transcript, session=new("s1"), prompt=prompt)
    (record,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert record["prompt"] == prompt


# --- concurrency and the port -----------------------------------------------------------------


def test_concurrent_agents_keep_independent_counters(workspace: Path, tmp_path: Path) -> None:
    script = Recorder()
    backend = ScriptedBackend(script)

    async def agent_loop(name: str) -> list[TurnResult]:
        results = []
        for turn in range(3):
            results.append(
                await backend.run_turn(
                    agent=name,
                    workspace=workspace,
                    prompt=f"{name} turn {turn}",
                    session=SessionChoice(f"sess-{name}", resume=turn > 0),
                    transcript_path=tmp_path / f"{name}.jsonl",
                )
            )
        return results

    async def main() -> list[list[TurnResult]]:
        return list(await asyncio.gather(agent_loop("a"), agent_loop("b"), agent_loop("c")))

    outcomes = asyncio.run(main())

    assert all(r.ok for results in outcomes for r in results)
    for name in "abc":
        mine = [c.turn_in_session for c in script.calls if c.agent == name]
        assert mine == [1, 2, 3]


def test_both_backends_have_the_port_signature() -> None:
    port = inspect.signature(AgentBackend.run_turn)
    for backend_type in (ScriptedBackend, ClaudeCodeBackend):
        sig = inspect.signature(backend_type.run_turn)
        assert list(sig.parameters) == list(port.parameters)
        assert {n: p.kind for n, p in sig.parameters.items()} == {
            n: p.kind for n, p in port.parameters.items()
        }
        assert inspect.iscoroutinefunction(backend_type.run_turn)
