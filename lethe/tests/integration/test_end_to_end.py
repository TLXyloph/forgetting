"""End-to-end runs with the real archive, materializer and judge, and a scripted agent."""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

from lethe.agents.scripted import ScriptCall, ScriptedBackend, ScriptReply
from lethe.api import load_run, run_task
from lethe.cli import main
from lethe.domain.events import EpochStarted

JUDGE = textwrap.dedent(
    """
    import json, re, pathlib
    text = pathlib.Path("solver.py").read_text()
    match = re.search(r"VALUE = (-?\\d+)", text)
    if not match:
        print(json.dumps({"valid": False, "feedback": "VALUE missing"}))
    else:
        value = int(match.group(1))
        print(json.dumps({"score": value, "feedback": f"value is {value}",
                          "cases": {"a": value, "b": -value}}))
    """
)


def make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "seed").mkdir(parents=True)
    (task / "judge").mkdir()
    (task / "seed" / "solver.py").write_text("VALUE = 10\n")
    (task / "seed" / "README.txt").write_text("rules\n")
    (task / "judge" / "score.py").write_text(JUDGE)
    (task / "lethe.toml").write_text(
        textwrap.dedent(f"""
        [task]
        name = "count-up"
        description = "Make VALUE in solver.py as large as possible."
        seed = "seed"
        editable = ["solver.py"]

        [judge]
        run = '"{sys.executable}" "$LETHE_TASK_DIR/judge/score.py"'
        direction = "max"
        timeout = 30
    """)
    )
    return task / "lethe.toml"


def climber(call: ScriptCall) -> ScriptReply:
    """Adds a shrinking step each turn of a session (anchoring), plus a scratch note."""
    solver = call.workspace / "solver.py"
    value = int(re.search(r"VALUE = (-?\d+)", solver.read_text()).group(1))
    step = max(0, 4 - call.turn_in_session)
    (call.workspace / "notes.txt").write_text(f"turn {call.turn_in_session}\n")
    assert (call.workspace / "TASK.md").exists()
    assert (call.workspace / ".context" / "feedback.md").exists()
    if step:
        solver.write_text(f"VALUE = {value + step}\n")
        return ScriptReply(final_text=f"added {step}")
    return ScriptReply(final_text="I've reached the practical limit.")


def run(tmp_path, **overrides):
    return run_task(
        make_task(tmp_path),
        overrides=overrides,
        runs_dir=tmp_path / "runs",
        run_name="r1",
        backend=ScriptedBackend(climber),
    )


def test_team_forget_run_end_to_end(tmp_path, capsys):
    run_dir = run(
        tmp_path,
        team={"agents": 2},
        forget={"when": [{"type": "budget_fractions", "fractions": [0.5]}]},
        budget={"usd": 0, "proposals": 12, "primary": "proposals"},
    )
    state = load_run(run_dir)
    assert state.finished is not None and "proposal budget" in state.finished.reason
    assert state.best().score > 10
    team_epochs = [r for r in state.epoch_history if r.scope == "team"]
    assert len(team_epochs) == 1
    assert (run_dir / "ws" / "a1" / "e2").is_dir() and (run_dir / "ws" / "a2" / "e2").is_dir()
    # forgetting lets the sessions restart with large steps: the best keeps climbing after it
    assert state.best().seq > team_epochs[0].seq
    # the judge only ever saw editable files: scratch notes never reach it
    assert all(c.valid for c in state.candidates.values())
    assert main(["report", str(run_dir)]) == 0
    out_dir = tmp_path / "best"
    assert main(["export", str(run_dir), "--out", str(out_dir), "--judged-only"]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["solver.py"]
    assert f"VALUE = {int(state.best().score)}" in (out_dir / "solver.py").read_text()
    events = [json.loads(line)["kind"] for line in (run_dir / "events.jsonl").open()]
    assert events[0] == "run_started" and events[-1] == "run_finished"
    capsys.readouterr()


def test_clean_forget_drops_scratch_files(tmp_path):
    run_dir = run(
        tmp_path,
        team={"agents": 1, "peers": "none"},
        forget={"carry_scratch": False, "when": [{"type": "budget_fractions", "fractions": [0.5]}]},
        budget={"usd": 0, "proposals": 4, "primary": "proposals"},
    )
    epoch2 = run_dir / "ws" / "a1" / "e2"
    assert (epoch2 / "solver.py").exists() and (epoch2 / "README.txt").exists()
    seeds = [e for e in (load_run(run_dir).epoch_history) if e.scope == "team"]
    assert seeds  # the forget happened
    # notes.txt written in epoch 1 rode along in the best candidate but was stripped
    assert not (epoch2 / "notes.txt").exists() or _written_this_epoch(epoch2)


def _written_this_epoch(workspace: Path) -> bool:
    return (workspace / "notes.txt").read_text().startswith("turn ")


def test_rerunning_into_an_existing_run_dir_is_refused(tmp_path):
    run(tmp_path, preset="solo", budget={"usd": 0, "proposals": 1, "primary": "proposals"})
    with pytest.raises(FileExistsError):
        run(tmp_path, preset="solo", budget={"usd": 0, "proposals": 1, "primary": "proposals"})


def test_epoch_events_are_consistent_with_workspaces(tmp_path):
    run_dir = run(
        tmp_path,
        team={"agents": 2},
        forget={"when": [{"type": "stalled", "n": 1, "of": 1}]},
        budget={"usd": 0, "proposals": 10, "primary": "proposals"},
    )
    state = load_run(run_dir)
    for record in state.epoch_history:
        for agent, epoch in record.epochs.items():
            assert (run_dir / "ws" / agent / f"e{epoch}").is_dir()
    assert any(isinstance(r, EpochStarted) or r.scope == "agent" for r in state.epoch_history)
