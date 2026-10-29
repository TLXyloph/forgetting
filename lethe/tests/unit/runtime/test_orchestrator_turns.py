"""Turn-level robustness: harness failures become error turns, error kinds are logged (and
decide whether the session survives), and the logged config is safe to keep and to replay.
Each turn also logs how many tool calls it made, which tells a passive turn from an active one."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeArchive, FakeMaterializer, Reply, build  # noqa: E402

from lethe.domain.events import EpochStarted, RunFinished, RunStarted, TurnFinished  # noqa: E402
from lethe.domain.types import ToolCall  # noqa: E402


def climbing(agent, index, prompt, session):
    return Reply(value=str(11 + index))


SOLO = {"preset": "solo", "budget": {"usd": 0, "proposals": 3, "primary": "proposals"}}


class RenderFailsOnce(FakeMaterializer):
    def render(self, workspace, bundle):
        if not self.rendered:
            self.rendered.append(None)
            raise OSError("no space left on device")
        super().render(workspace, bundle)


class SnapshotFailsOnce(FakeArchive):
    """The second turn's snapshot fails (the first turn and the seed are archived fine)."""

    turns = 0

    def snapshot(self, workdir, *, parent_sha, message, index_name):
        if index_name != "seed":
            self.turns += 1
            if self.turns == 2:
                raise RuntimeError("git hash-object failed")
        return super().snapshot(
            workdir, parent_sha=parent_sha, message=message, index_name=index_name
        )


def test_a_render_failure_is_an_error_turn_not_a_crash(tmp_path):
    orch, store, backend, materializer, _ = build(tmp_path, climbing, **SOLO)
    orch.materializer = RenderFailsOnce(materializer.archive, tmp_path / "ws2")

    state = asyncio.run(orch.run())

    first, *rest = store.of(TurnFinished)
    assert (first.status, first.error_kind, first.cost_usd) == ("error", "harness", 0.0)
    assert "no space left on device" in first.error
    assert len(backend.calls) == 3  # the agent never ran for the failed turn
    assert state.proposals == 3 and store.of(RunFinished)[0].status == "completed"


def test_a_snapshot_failure_is_an_error_turn_that_keeps_what_the_turn_cost(tmp_path):
    orch, store, backend, _, _ = build(tmp_path, climbing, **SOLO)
    archive = SnapshotFailsOnce()
    orch.archive = archive
    orch.materializer = FakeMaterializer(archive, tmp_path / "ws2")

    state = asyncio.run(orch.run())

    failed = store.of(TurnFinished)[1]
    assert (failed.status, failed.error_kind, failed.cost_usd) == ("error", "harness", 0.1)
    assert "git hash-object failed" in failed.error
    assert state.proposals == 3 and store.of(RunFinished)[0].status == "completed"
    # the session survives a harness failure: the next turn resumes it
    sessions = [c["session"] for c in backend.calls]
    assert [s.resume for s in sessions] == [False, True, True, True]
    assert len({s.session_id for s in sessions}) == 1


def test_error_kinds_are_logged_and_transient_errors_keep_the_session(tmp_path):
    def script(agent, index, prompt, session):
        return Reply(error_kind="api", cost=0.0) if index == 1 else Reply(value=str(11 + index))

    orch, store, backend, _, _ = build(tmp_path, script, **SOLO)
    asyncio.run(orch.run())

    assert [t.error_kind for t in store.of(TurnFinished)] == [None, "api", None, None]
    sessions = [c["session"] for c in backend.calls]
    assert [s.resume for s in sessions] == [False, True, True, True]
    assert len({s.session_id for s in sessions}) == 1  # no accidental fresh start


def test_a_process_error_starts_a_fresh_session(tmp_path):
    def script(agent, index, prompt, session):
        return Reply(error_kind="process", cost=0.0) if index == 1 else Reply(value=str(11 + index))

    orch, _, backend, _, _ = build(tmp_path, script, **SOLO)
    asyncio.run(orch.run())

    sessions = [c["session"] for c in backend.calls]
    assert [s.resume for s in sessions] == [False, True, False, True]


def test_the_logged_config_hides_judge_secrets_and_keeps_the_task_text(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "task.txt").write_text("Make the number big, from a file.")
    overrides = {
        **SOLO,
        "task": {"description": None, "description_file": "task.txt"},
        "judge": {"env": {"API_TOKEN": "s3cret-value", "MODE": "fast"}},
    }
    orch, store, *_ = build(tmp_path, climbing, **overrides)

    asyncio.run(orch.run())

    (started,) = store.of(RunStarted)
    assert started.config["judge"]["env"] == {"API_TOKEN": "***", "MODE": "***"}
    assert "s3cret-value" not in json.dumps(started.config)
    assert started.config["task"]["description"] == "Make the number big, from a file."
    assert started.config["task"]["description_file"] == "task.txt"
    assert orch.config.resolved["judge"]["env"]["API_TOKEN"] == "s3cret-value"  # not mutated


EXPLORED = (ToolCall("Bash"), ToolCall("Read", "solver.txt"), ToolCall("Bash"))


def test_each_turn_logs_how_many_tool_calls_it_made(tmp_path):
    def script(agent, index, prompt, session):
        return [
            Reply(value="12"),  # the default reply: a Read and an Edit
            Reply(error_kind="api", cost=0.0, tool_calls=(ToolCall("Bash"),)),
            Reply(value=None, text="I've plateaued.", tool_calls=()),
            Reply(value=None, tool_calls=EXPLORED),
        ][index]

    orch, store, *_ = build(tmp_path, script, **SOLO)
    asyncio.run(orch.run())

    turns = [(t.status, t.tool_calls) for t in store.of(TurnFinished)]
    assert turns == [("candidate", 2), ("error", 1), ("empty", 0), ("empty", 3)]


@pytest.mark.parametrize(
    ("passive_only", "a1_tools", "reset"),
    [
        (False, EXPLORED, True),  # by default every empty turn counts
        (True, EXPLORED, False),  # exploring without submitting yet is not a stall
        (True, (), True),  # not acting at all is
    ],
)
def test_passive_only_resets_only_an_agent_that_stopped_acting(
    tmp_path, passive_only, a1_tools, reset
):
    def script(agent, index, prompt, session):
        if agent == "a1":  # never submits: works with `a1_tools`, or does nothing at all
            return Reply(value=None, text="still measuring", tool_calls=a1_tools)
        return Reply(value=str(20 + index))

    stalled = {"type": "stalled", "n": 2, "of": 2, "passive_only": passive_only}
    overrides = {
        "team": {"agents": 2},
        "forget": {"when": [stalled]},
        "budget": {"usd": 0, "proposals": 8, "primary": "proposals"},
    }
    orch, store, *_ = build(tmp_path, script, **overrides)
    asyncio.run(orch.run())

    agent_epochs = [e for e in store.of(EpochStarted) if e.scope == "agent"]
    assert bool(agent_epochs) is reset
    assert all(set(e.epochs) == {"a1"} for e in agent_epochs)
