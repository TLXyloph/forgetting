"""The shipped experiment files: they match the brief and lethe accepts every arm's settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lethe.policy.triggers import trigger_from_dict
from research.experiment import Experiment, arm_overrides, load_experiment

RESEARCH = Path(__file__).resolve().parents[1]
EXPERIMENTS = RESEARCH / "config" / "experiments"
TASKS = RESEARCH / "tasks"

PAPER_ARMS = [("solo", "solo"), ("parallel", "parallel"), ("team-forget", "team-forget")]

PAPER_FILES = {
    # file: (task directory, overrides shared by every arm)
    "paper-toy-mis": ("toy-mis", {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}),
    "paper-vliw": ("vliw", {"budget": {"proposals": 160, "primary": "proposals", "usd": 200}}),
    "paper-ahc014": ("ahc014", {"budget": {"proposals": 160, "primary": "proposals", "usd": 200}}),
    "replicate-vliw-sonnet": (
        "vliw",
        {
            "agent": {"model": "claude-sonnet-5-5", "shell": True, "read_root": "../../.."},
            "budget": {
                "usd": 40,
                "proposals": 0,
                "primary": "usd",
                "pause_at_weekly_utilization": 0.8,
            },
        },
    ),
}

# arm -> (preset, arm-level overrides), in file order
BETS: dict[str, tuple[str, dict[str, Any]]] = {
    "team": ("team", {}),
    "team-forget": ("team-forget", {}),
    "anchored": (
        "team",
        {
            "forget": {
                "when": [
                    {
                        "type": "anchored",
                        "ratio": 0.3,
                        "warmup": 4,
                        "window": 4,
                        "quorum": 0.5,
                        "require_stall": True,
                        "min_proposals": 8,
                    }
                ]
            }
        },
    ),
    "stalled": ("team", {"forget": {"when": [{"type": "stalled", "n": 3, "of": 5}]}}),
    "hedged": ("team-forget", {"forget": {"seed": "hedged"}}),
    "clean-forget": ("team-forget", {"forget": {"carry_scratch": False}}),
    "fresh-every-turn": ("team", {"team": {"session": "fresh"}}),
    "plateau": (
        "team",
        {
            "forget": {
                "when": [{"type": "plateau", "window": 8, "min_gain": 0.0, "min_proposals": 8}]
            }
        },
    ),
}

ALL_FILES = [*PAPER_FILES, "bets-toy-mis", "bets-vliw-sonnet", "passive-stall-vliw-sonnet"]


def load(name: str) -> Experiment:
    return load_experiment(EXPERIMENTS / f"{name}.toml")


# ---------------------------------------------------------------------------------------------
# what the brief asks for
# ---------------------------------------------------------------------------------------------


def test_all_shipped_experiment_files_exist() -> None:
    assert set(ALL_FILES) <= {p.stem for p in EXPERIMENTS.glob("*.toml")}


@pytest.mark.parametrize("name", ALL_FILES)
def test_name_matches_the_file_name_and_there_are_three_replicates(name: str) -> None:
    experiment = load(name)
    assert experiment.name == name
    assert experiment.replicates == 3


@pytest.mark.parametrize("name", list(PAPER_FILES))
def test_paper_experiments_compare_solo_parallel_and_team_forget(name: str) -> None:
    task_dir, shared = PAPER_FILES[name]
    experiment = load(name)

    assert [(a.name, a.preset) for a in experiment.arms] == PAPER_ARMS
    assert all(dict(a.overrides) == {} for a in experiment.arms)
    assert experiment.overrides == shared
    assert experiment.task == (TASKS / task_dir / "lethe.toml").resolve()


@pytest.mark.parametrize("name", ["paper-vliw", "paper-ahc014"])
def test_subscription_experiments_explain_the_scale_down(name: str) -> None:
    text = (EXPERIMENTS / f"{name}.toml").read_text(encoding="utf-8")
    assert "subscription" in text
    assert "DeepSeek V4.1 Flash" in text
    assert "$20" in text and "$40" in text
    assert "160 proposals" in text
    assert "safety cap" in text and "usd = 0" not in text  # proposals primary, dollars capped


def test_bets_cover_the_listed_policies_in_order() -> None:
    experiment = load("bets-toy-mis")
    assert [a.name for a in experiment.arms] == list(BETS)
    for arm in experiment.arms:
        preset, overrides = BETS[arm.name]
        assert arm.preset == preset, arm.name
        assert dict(arm.overrides) == overrides, arm.name


def test_bets_use_the_same_matched_budget_as_the_toy_paper_run() -> None:
    paper, bets = load("paper-toy-mis"), load("bets-toy-mis")
    assert bets.overrides == paper.overrides
    assert bets.replicates == paper.replicates
    assert bets.task == paper.task


# ---------------------------------------------------------------------------------------------
# lethe accepts what the files say
# ---------------------------------------------------------------------------------------------


def resolved(experiment: Experiment, arm_name: str) -> Any:
    """The RunConfig lethe builds for one arm (skips if its task is not built yet)."""
    config_mod = pytest.importorskip("lethe.config")
    if not experiment.task.is_file():
        pytest.skip(f"task not built yet: {experiment.task}")
    arm = next(a for a in experiment.arms if a.name == arm_name)
    return config_mod.load_config(
        experiment.task, preset=arm.preset, overrides=arm_overrides(experiment, arm)
    )


@pytest.mark.parametrize(
    ("name", "arm"), [(n, a) for n in PAPER_FILES for a, _ in PAPER_ARMS], ids=str
)
def test_lethe_accepts_every_paper_arm(name: str, arm: str) -> None:
    config = resolved(load(name), arm)
    solo = arm == "solo"
    assert config.team.agents == (1 if solo else 4)
    assert config.team.peers == {"solo": "none", "parallel": "none", "team-forget": "all"}[arm]
    if arm == "team-forget":
        assert [w["type"] for w in config.forget.when] == ["budget_fractions"]
        assert list(config.forget.when[0]["fractions"]) == [0.25, 0.5, 0.75]
    else:
        assert config.forget.when == ()


def test_paper_toy_mis_budget_and_model_reach_the_config() -> None:
    for arm, _ in PAPER_ARMS:
        config = resolved(load("paper-toy-mis"), arm)
        assert config.budget.usd == 3.0
        assert config.budget.primary == "usd"
        assert config.agent.model == "haiku"


@pytest.mark.parametrize("name", ["paper-vliw", "paper-ahc014"])
def test_proposal_budget_reaches_the_config(name: str) -> None:
    for arm, _ in PAPER_ARMS:
        config = resolved(load(name), arm)
        assert config.budget.proposals == 160
        assert config.budget.usd == 200  # a safety cap, never the unit forgets are scheduled in
        assert config.budget.primary == "proposals"


def test_vliw_replication_matches_the_paper_in_dollars_with_a_pinned_sonnet() -> None:
    for arm, _ in PAPER_ARMS:
        config = resolved(load("replicate-vliw-sonnet"), arm)
        assert config.budget.usd == 40  # the paper's VLIW budget per run
        assert config.budget.proposals == 0  # no proposal cap: dollars are the only budget
        assert config.budget.primary == "usd"
        assert config.agent.model == "claude-sonnet-5-5"  # a full id, not the moving alias
        assert config.budget.pause_at_weekly_utilization == 0.8  # a fifth of the week stays free
        assert config.agent.shell is True  # the paper's Pi agents had a shell (sandboxed here)
        # agents' commands may not read beside their workspace anywhere in this repository
        assert config.agent.read_root == RESEARCH.parent


@pytest.mark.parametrize("arm", list(BETS))
def test_lethe_accepts_every_bet_arm(arm: str) -> None:
    config = resolved(load("bets-toy-mis"), arm)

    assert config.team.agents == 4
    assert config.team.peers == "all"
    assert config.budget.usd == 3.0
    assert config.agent.model == "haiku"

    forget_types = [w["type"] for w in config.forget.when]
    expected_types = {
        "team": [],
        "team-forget": ["budget_fractions"],
        "anchored": ["anchored"],
        "stalled": ["stalled"],
        "hedged": ["budget_fractions"],  # the preset's schedule survives the seed override
        "clean-forget": ["budget_fractions"],
        "fresh-every-turn": [],
        "plateau": ["plateau"],
    }[arm]
    assert forget_types == expected_types
    assert config.forget.seed == ("hedged" if arm == "hedged" else "best")
    assert config.forget.carry_scratch is (arm != "clean-forget")
    assert config.team.session == ("fresh" if arm == "fresh-every-turn" else "continue")


# ---------------------------------------------------------------------------------------------
# E2: bets on VLIW, against a team-forget baseline run alongside them
# ---------------------------------------------------------------------------------------------

STALLED = {"type": "stalled", "n": 3, "of": 5}
VLIW_BETS: dict[str, tuple[str, dict[str, Any]]] = {
    "team-forget": ("team-forget", {}),
    "stalled": ("team", {"forget": {"when": [STALLED]}}),
    "clean-forget": ("team-forget", {"forget": {"carry_scratch": False}}),
    "stalled-clean": ("team", {"forget": {"carry_scratch": False, "when": [STALLED]}}),
}


def test_vliw_bets_cover_the_arms_in_order_with_e1s_agent_and_budget() -> None:
    experiment = load("bets-vliw-sonnet")
    assert experiment.name == "bets-vliw-sonnet" and experiment.replicates == 3
    assert [a.name for a in experiment.arms] == list(VLIW_BETS)
    for arm in experiment.arms:
        preset, overrides = VLIW_BETS[arm.name]
        assert (arm.preset, dict(arm.overrides)) == (preset, overrides), arm.name
    e1 = load("replicate-vliw-sonnet")
    assert (experiment.task, experiment.overrides) == (e1.task, e1.overrides)


@pytest.mark.parametrize("arm", list(VLIW_BETS))
def test_lethe_accepts_every_vliw_bet_arm(arm: str) -> None:
    config = resolved(load("bets-vliw-sonnet"), arm)
    assert (config.team.agents, config.team.peers) == (4, "all")
    assert config.agent.model == "claude-sonnet-5-5" and config.agent.shell is True
    assert (config.budget.usd, config.budget.primary) == (40, "usd")
    stalled = arm.startswith("stalled")
    assert [w["type"] for w in config.forget.when] == (
        ["stalled"] if stalled else ["budget_fractions"]
    )
    assert config.forget.carry_scratch is ("clean" not in arm)
    assert config.forget.seed == "best"


# ---------------------------------------------------------------------------------------------
# E3: the stalled trigger counting only passive turns, against a team-forget baseline
# ---------------------------------------------------------------------------------------------

PASSIVE_STALLED = {"type": "stalled", "n": 3, "of": 5, "passive_only": True}
PASSIVE_STALL_ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "team-forget": ("team-forget", {}),
    "stalled-passive": ("team", {"forget": {"when": [PASSIVE_STALLED]}}),
}


def test_passive_stall_covers_the_arms_in_order_with_e1s_task_agent_and_budget() -> None:
    experiment = load("passive-stall-vliw-sonnet")
    assert experiment.name == "passive-stall-vliw-sonnet" and experiment.replicates == 3
    assert [a.name for a in experiment.arms] == list(PASSIVE_STALL_ARMS)
    for arm in experiment.arms:
        preset, overrides = PASSIVE_STALL_ARMS[arm.name]
        assert (arm.preset, dict(arm.overrides)) == (preset, overrides), arm.name
    e1 = load("replicate-vliw-sonnet")
    assert (experiment.task, experiment.overrides) == (e1.task, e1.overrides)


@pytest.mark.parametrize("arm", list(PASSIVE_STALL_ARMS))
def test_lethe_accepts_every_passive_stall_arm(arm: str) -> None:
    config = resolved(load("passive-stall-vliw-sonnet"), arm)
    assert (config.team.agents, config.team.peers) == (4, "all")
    assert config.agent.model == "claude-sonnet-5-5" and config.agent.shell is True
    assert (config.budget.usd, config.budget.primary) == (40, "usd")
    assert (config.forget.seed, config.forget.carry_scratch) == ("best", True)
    if arm == "stalled-passive":
        assert [dict(w) for w in config.forget.when] == [PASSIVE_STALLED]
        assert trigger_from_dict(config.forget.when[0]).passive_only is True
    else:
        assert [w["type"] for w in config.forget.when] == ["budget_fractions"]
