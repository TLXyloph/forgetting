from __future__ import annotations

from lethe.api import load_run
from lethe.cli import build_parser, main, overrides_from_args
from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    JudgeFinished,
    RunFinished,
    RunStarted,
)
from lethe.store import EventStore


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_run_flags_become_nested_overrides():
    args = parse(
        "run",
        "task.toml",
        "--agents",
        "2",
        "--usd",
        "3.5",
        "--model",
        "haiku",
        "--effort",
        "high",
        "--peers",
        "ring",
        "--session",
        "fresh",
        "--set",
        "judge.timeout=30",
        "--set",
        "forget.carry_scratch=false",
        "--set",
        'task.name="renamed"',
    )
    assert overrides_from_args(args) == {
        "team": {"agents": 2, "peers": "ring", "session": "fresh"},
        "budget": {"usd": 3.5},
        "agent": {"model": "haiku", "effort": "high"},
        "judge": {"timeout": 30},
        "forget": {"carry_scratch": False},
        "task": {"name": "renamed"},
    }


def test_proposals_flag_makes_proposals_the_primary_unit():
    args = parse("run", "task.toml", "--proposals", "40")
    assert overrides_from_args(args) == {"budget": {"proposals": 40, "primary": "proposals"}}
    args = parse("run", "task.toml", "--proposals", "40", "--primary", "usd")
    assert overrides_from_args(args)["budget"]["primary"] == "usd"


def test_presets_command_lists_presets(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in ("solo", "parallel", "team", "team-forget"):
        assert name in out


def write_run(run_dir):
    store = EventStore(run_dir / "events.jsonl")
    store.append(RunStarted("run-1", "toy", ("a1",), "min", {}, "0.1.0"))
    store.append(CandidateCommitted("c0000", "s", None, "seed", 0, None, "k", 0, 0, 0, "seed"))
    store.append(JudgeFinished("c0000", True, 5.0, "", {}, None, 0.0, False, None))
    store.append(EpochStarted("start", {"a1": "c0000"}, {"a1": 1}, "run start"))
    store.append(RunFinished("finished", "c0000", 5.0))


def test_load_run_folds_the_event_log(tmp_path):
    write_run(tmp_path)
    state = load_run(tmp_path)
    assert state.run_id == "run-1" and state.direction == "min"
    assert state.best().cand_id == "c0000" and state.finished.reason == "finished"


def test_report_command_prints_summary_and_writes_csv(tmp_path, capsys):
    write_run(tmp_path)
    assert main(["report", str(tmp_path), "--csv", str(tmp_path / "csv")]) == 0
    out = capsys.readouterr().out
    assert "Run run-1" in out and "lower is better" in out
    assert (tmp_path / "csv" / "turns.csv").exists()


def test_config_errors_exit_with_status_2(tmp_path, capsys):
    bad = tmp_path / "lethe.toml"
    bad.write_text("[task]\nname = 'x'\n")
    assert main(["check", str(bad)]) == 2
    assert "config error" in capsys.readouterr().err
