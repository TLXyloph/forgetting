"""The shipped experiment files: they match the brief and lethe accepts every arm's settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research.experiment import Experiment, arm_overrides, load_experiment

RESEARCH = Path(__file__).resolve().parents[1]
EXPERIMENTS = RESEARCH / "config" / "experiments"
TASKS = RESEARCH / "tasks"

PAPER_ARMS = [("solo", "solo"), ("parallel", "parallel"), ("team-forget", "team-forget")]

PAPER_FILES = {
    # file: (task directory, overrides shared by every arm)
    "paper-toy-mis": ("toy-mis", {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}),
    "paper-vliw": ("vliw", {"budget": {"proposals": 160, "primary": "proposals", "usd": 0}}),
    "paper-ahc014": ("ahc014", {"budget": {"proposals": 160, "primary": "proposals", "usd": 0}}),
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

ALL_FILES = [*PAPER_FILES, "bets-toy-mis"]


def load(name: str) -> Experiment:
    return load_experiment(EXPERIMENTS / f"{name}.toml")


# ---------------------------------------------------------------------------------------------
# what the brief asks for
# ---------------------------------------------------------------------------------------------


def test_all_four_experiment_files_exist() -> None:
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
        assert config.budget.usd == 0
        assert config.budget.primary == "proposals"


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
