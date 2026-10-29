from __future__ import annotations

from pathlib import Path

import pytest

from lethe.config import ConfigError, list_presets, load_config


def write_task(tmp_path: Path, body: str, *, seed=True) -> Path:
    if seed:
        (tmp_path / "seed").mkdir(exist_ok=True)
        (tmp_path / "seed" / "solver.py").write_text("x = 1\n")
    path = tmp_path / "lethe.toml"
    path.write_text(body)
    return path


MINIMAL = """
[task]
name = "toy"
description = "Maximize x."
seed = "seed"
editable = ["solver.py"]

[judge]
run = "python3 judge.py"
direction = "max"
"""


def test_minimal_config_gets_paper_defaults(tmp_path):
    cfg = load_config(write_task(tmp_path, MINIMAL))
    assert cfg.preset == "team-forget"
    assert cfg.team.agents == 4 and cfg.team.peers == "all" and cfg.team.session == "continue"
    assert cfg.team.agent_ids == ("a1", "a2", "a3", "a4")
    assert cfg.forget.when == ({"type": "budget_fractions", "fractions": [0.25, 0.5, 0.75]},)
    assert cfg.forget.seed == "best" and cfg.forget.carry_scratch is True
    assert cfg.task.seed_dir == (tmp_path / "seed").resolve()
    assert cfg.task.config_dir == tmp_path.resolve()
    assert ".git/**" in cfg.task.exclude
    assert cfg.judge.run == "python3 judge.py" and cfg.judge.timeout_s == 600
    assert cfg.budget.usd == 5.0 and cfg.budget.primary == "usd"
    assert cfg.budget.pause_at_utilization == 0.95
    assert cfg.budget.pause_at_weekly_utilization == 1.0  # off: pause only on a full week
    assert cfg.agent.isolation == "safe" and cfg.agent.model == "sonnet"
    assert cfg.runs_dir == (tmp_path / ".lethe" / "runs").resolve()
    assert cfg.resolved["team"]["agents"] == 4


def test_preset_argument_and_overrides_take_precedence(tmp_path):
    path = write_task(tmp_path, 'preset = "team"\n' + MINIMAL + "\n[team]\nagents = 2\n")
    cfg = load_config(path)
    assert cfg.preset == "team" and cfg.team.agents == 2 and cfg.forget.when == ()
    cfg = load_config(
        path, preset="solo", overrides={"budget": {"usd": 1.5}, "agent": {"model": ""}}
    )
    assert cfg.preset == "solo" and cfg.team.agents == 2  # file beats preset
    assert cfg.team.peers == "none"
    assert cfg.budget.usd == 1.5 and cfg.agent.model is None


def test_lists_are_replaced_not_merged(tmp_path):
    cfg = load_config(
        write_task(tmp_path, MINIMAL),
        overrides={"forget": {"when": [{"type": "stalled", "n": 2, "of": 4}]}},
    )
    assert cfg.forget.when == ({"type": "stalled", "n": 2, "of": 4},)


def test_description_file(tmp_path):
    (tmp_path / "task.txt").write_text("From a file.")
    body = MINIMAL.replace('description = "Maximize x."', 'description_file = "task.txt"')
    assert load_config(write_task(tmp_path, body)).task.description == "From a file."


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (("[team]", "[team]\nagnets = 3"), "unknown key"),
        (('direction = "max"', 'direction = "up"'), "direction"),
        (('editable = ["solver.py"]', "editable = []"), "editable"),
        (('seed = "seed"', 'seed = "missing"'), "seed"),
        (('name = "toy"', 'name = "bad name"'), "name"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, patch, message):
    body = MINIMAL + "\n[team]\nagents = 4\n"
    old, new = patch
    with pytest.raises(ConfigError, match=message):
        load_config(write_task(tmp_path, body.replace(old, new, 1)))


def test_invalid_overrides_are_rejected(tmp_path):
    path = write_task(tmp_path, MINIMAL)
    with pytest.raises(ConfigError, match="at least one budget cap"):
        load_config(path, overrides={"budget": {"usd": 0}})
    with pytest.raises(ConfigError, match="primary"):
        load_config(path, overrides={"budget": {"primary": "proposals"}})
    for bad in (0, 1.5):
        with pytest.raises(ConfigError, match="pause_at_weekly_utilization"):
            load_config(path, overrides={"budget": {"pause_at_weekly_utilization": bad}})
    with pytest.raises(ConfigError, match="trigger"):
        load_config(path, overrides={"forget": {"when": [{"type": "nope"}]}})
    with pytest.raises(ConfigError, match="effort"):
        load_config(path, overrides={"agent": {"effort": "huge"}})
    with pytest.raises(ConfigError, match="agents"):
        load_config(path, overrides={"team": {"agents": 0}})
    with pytest.raises(ConfigError, match="preset"):
        load_config(path, preset="nonexistent")


def test_missing_required_sections(tmp_path):
    with pytest.raises(ConfigError, match="judge.run"):
        load_config(write_task(tmp_path, MINIMAL.replace('run = "python3 judge.py"\n', "")))


def test_the_weekly_pause_threshold_reaches_the_budget(tmp_path):
    cfg = load_config(
        write_task(tmp_path, MINIMAL), overrides={"budget": {"pause_at_weekly_utilization": 0.8}}
    )
    assert cfg.budget.pause_at_weekly_utilization == 0.8


def test_list_presets():
    assert list_presets() == ["parallel", "solo", "team", "team-forget"]


def test_claude_options_from_agent_config(tmp_path):
    cfg = load_config(
        write_task(tmp_path, MINIMAL),
        overrides={"agent": {"effort": "high", "shell": True, "extra_args": ["--x"]}},
    )
    options = cfg.agent.claude_options()
    assert options.model == "sonnet" and options.effort == "high" and options.shell is True
    assert options.extra_args == ("--x",) and options.isolation == "safe"


def test_judge_output_defaults_to_stdout_and_accepts_file(tmp_path):
    path = write_task(tmp_path, MINIMAL)
    assert load_config(path).judge.output == "stdout"
    assert load_config(path).resolved["judge"]["output"] == "stdout"
    assert load_config(path, overrides={"judge": {"output": "file"}}).judge.output == "file"
    for bad in ("files", "result.json", 1, None):
        with pytest.raises(ConfigError, match="judge.output"):
            load_config(path, overrides={"judge": {"output": bad}})


def test_allow_api_billing_defaults_to_false_and_reaches_the_claude_options(tmp_path):
    path = write_task(tmp_path, MINIMAL)
    cfg = load_config(path)
    assert cfg.agent.allow_api_billing is False
    assert cfg.agent.claude_options().allow_api_billing is False
    cfg = load_config(path, overrides={"agent": {"allow_api_billing": True}})
    assert cfg.agent.claude_options().allow_api_billing is True
    with pytest.raises(ConfigError, match="allow_api_billing"):
        load_config(path, overrides={"agent": {"allow_api_billing": "yes"}})


def test_read_isolation_defaults_and_private_temp_never_archived(tmp_path):
    cfg = load_config(write_task(tmp_path, MINIMAL))
    assert cfg.agent.read_root is None  # run_task then uses the runs directory
    assert cfg.agent.deny_read == ()
    assert ".lethe-tmp/**" in cfg.task.exclude


def test_read_isolation_paths_resolve_against_the_config_dir(tmp_path):
    overrides = {"agent": {"read_root": "..", "deny_read": ["docs", "/abs/notes"]}}
    cfg = load_config(write_task(tmp_path, MINIMAL), overrides=overrides)
    assert cfg.agent.read_root == tmp_path.parent.resolve()
    assert cfg.agent.deny_read == ((tmp_path / "docs").resolve(), Path("/abs/notes"))
    options = cfg.agent.claude_options()
    assert options.read_root == str(tmp_path.parent.resolve())
    assert options.deny_read == (str((tmp_path / "docs").resolve()), "/abs/notes")


def test_invalid_read_isolation_settings_are_rejected(tmp_path):
    path = write_task(tmp_path, MINIMAL)
    with pytest.raises(ConfigError, match="read_root"):
        load_config(path, overrides={"agent": {"read_root": 3}})
    with pytest.raises(ConfigError, match="deny_read"):
        load_config(path, overrides={"agent": {"deny_read": "docs"}})
