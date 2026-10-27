"""The warnings `lethe check` and `lethe run` share: the trust model, risky agent settings, and
API-billing variables in the environment (and what happens to them)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lethe import api, cli
from lethe.agents.scripted import ScriptedBackend, ScriptReply
from lethe.api import TRUST_MODEL, config_warnings, run_task, seed_files
from lethe.config import load_config
from lethe.domain.events import RunFinished, RunStarted
from lethe.store import EventStore

TASK = """
[task]
name = "toy"
description = "Maximize x."
seed = "seed"
editable = ["solver.py"]

[judge]
run = "echo 1"
direction = "max"
"""
REACH = "old epochs, transcripts, the archive and hidden judges"


def write_task(tmp_path: Path) -> Path:
    (tmp_path / "seed").mkdir(exist_ok=True)
    (tmp_path / "seed" / "solver.py").write_text("x = 1\n")
    (tmp_path / "lethe.toml").write_text(TASK)
    return tmp_path / "lethe.toml"


def warnings_for(tmp_path: Path, environ=None, **agent) -> list[str]:
    config = load_config(write_task(tmp_path), overrides={"agent": agent} if agent else None)
    return config_warnings(config, environ={} if environ is None else environ)


def test_the_trust_model_comes_first_and_stays_short(tmp_path):
    warnings = warnings_for(tmp_path)
    assert warnings == [TRUST_MODEL]
    assert 2 <= len(TRUST_MODEL.splitlines()) <= 3
    assert "not on code the judge executes" in TRUST_MODEL and "no OS sandbox" in TRUST_MODEL


@pytest.mark.parametrize(
    ("agent", "needles"),
    [
        ({"shell": True}, ["agent.shell = true", REACH]),
        ({"isolation": "none"}, ["agent.isolation = 'none'", REACH, "CLAUDE.md"]),
        ({"extra_args": ["--add-dir", "/"]}, ["--add-dir /", "may undo safe mode"]),
    ],
)
def test_risky_agent_settings_are_spelled_out(tmp_path, agent, needles):
    (warning,) = warnings_for(tmp_path, **agent)[1:]
    for needle in needles:
        assert needle in warning


ENV = {"ANTHROPIC_API_KEY": "sk-x", "CLAUDE_CODE_USE_BEDROCK": "1", "OTHER": "y"}


@pytest.mark.parametrize(
    ("agent", "fate"),
    [({}, "removed"), ({"allow_api_billing": True}, "passed"), ({"isolation": "bare"}, "passed")],
)
def test_api_billing_variables_are_named_with_their_fate(tmp_path, agent, fate):
    (warning,) = warnings_for(tmp_path, environ=ENV, **agent)[1:]
    assert "ANTHROPIC_API_KEY, CLAUDE_CODE_USE_BEDROCK" in warning and "OTHER" not in warning
    assert fate in warning and "sk-x" not in warning


def test_check_prints_every_shared_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(api.shutil, "which", lambda name: None)  # never runs a real `claude`
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    path = write_task(tmp_path)
    assert cli.main(["check", str(path), "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "warning: trust model: forgetting is enforced on the agents' tools" in out
    assert "ANTHROPIC_API_KEY" in out and "removed" in out


def test_run_task_reports_the_warnings_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    seen: list[tuple[str, bool]] = []
    runs = tmp_path / "runs"

    def on_warning(text: str) -> None:
        seen.append((text, (runs / "r" / "events.jsonl").exists()))

    run_task(
        write_task(tmp_path),
        overrides={"preset": "solo", "budget": {"usd": 0, "proposals": 1, "primary": "proposals"}},
        runs_dir=runs,
        run_name="r",
        backend=ScriptedBackend(lambda call: ScriptReply()),
        on_warning=on_warning,
    )

    assert [text for text, _ in seen][0] == TRUST_MODEL
    assert any("ANTHROPIC_AUTH_TOKEN" in text for text, _ in seen)
    assert not any(started for _, started in seen)


def test_lethe_run_prints_the_warnings_on_stderr(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"

    def fake_run_task(config, **kwargs):
        kwargs["on_warning"](TRUST_MODEL)
        store = EventStore(run_dir / "events.jsonl")
        store.append(RunStarted("r", "toy", ("a1",), "max", {}, "0"))
        store.append(RunFinished("finished", None, None))
        return run_dir

    monkeypatch.setattr(cli, "run_task", fake_run_task)
    assert cli.main(["run", "lethe.toml", "--quiet"]) == 0
    err = capsys.readouterr().err
    assert err.startswith("warning: trust model:") and "  Candidate code runs" in err


def test_seed_files_skip_names_the_archive_cannot_hold(tmp_path):
    path = write_task(tmp_path)
    (tmp_path / "seed" / "odd\\name.txt").write_text("x")
    assert seed_files(load_config(path)) == ["solver.py"]
