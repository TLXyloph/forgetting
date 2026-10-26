"""Tests for research.experiment: parsing, validation and arm override merging."""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest

from research.experiment import (
    Arm,
    Experiment,
    ExperimentError,
    arm_overrides,
    load_experiment,
    select_arms,
)

BRIEF_EXAMPLE = """
name = "paper-toy-mis"                     # [A-Za-z0-9_.-]+
task = "../../tasks/toy-mis/lethe.toml"    # relative to the experiment file
replicates = 3
[overrides]                                # deep-merged into every arm (lists replace)
budget = { usd = 3.0 }
agent = { model = "haiku" }
[[arms]]
name = "solo"
preset = "solo"
[[arms]]
name = "anchored"
preset = "team"
overrides = { forget = { when = [ { type = "anchored" } ] } }
"""

MINIMAL = """
name = "tiny"
task = "task/lethe.toml"
[[arms]]
name = "only"
"""


def write(tmp_path: Path, text: str, name: str = "exp.toml") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


def test_loads_the_example_from_the_brief(tmp_path: Path) -> None:
    experiment = load_experiment(write(tmp_path, BRIEF_EXAMPLE))

    assert experiment.name == "paper-toy-mis"
    assert experiment.replicates == 3
    assert experiment.overrides == {"budget": {"usd": 3.0}, "agent": {"model": "haiku"}}
    assert [arm.name for arm in experiment.arms] == ["solo", "anchored"]
    assert experiment.arms[0] == Arm(name="solo", preset="solo", overrides={})
    assert experiment.arms[1].preset == "team"
    assert experiment.arms[1].overrides == {"forget": {"when": [{"type": "anchored"}]}}


def test_task_is_resolved_relative_to_the_experiment_file_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, BRIEF_EXAMPLE, "research/config/experiments/paper-toy-mis.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    experiment = load_experiment(path)

    assert experiment.task == (tmp_path / "research/tasks/toy-mis/lethe.toml").resolve()
    assert experiment.task.is_absolute()


def test_accepts_a_string_path(tmp_path: Path) -> None:
    assert load_experiment(str(write(tmp_path, MINIMAL))).name == "tiny"


def test_defaults_for_optional_fields(tmp_path: Path) -> None:
    experiment = load_experiment(write(tmp_path, MINIMAL))

    assert experiment.replicates == 1
    assert experiment.overrides == {}
    assert experiment.arms == (Arm(name="only", preset=None, overrides={}),)


def test_dataclasses_are_frozen_and_arms_are_a_tuple(tmp_path: Path) -> None:
    experiment = load_experiment(write(tmp_path, BRIEF_EXAMPLE))

    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.arms, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        experiment.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        experiment.arms[0].name = "other"  # type: ignore[misc]


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "nope.toml")


def test_invalid_toml_is_reported_with_the_file_name(tmp_path: Path) -> None:
    path = write(tmp_path, "name = = 3", "broken.toml")
    with pytest.raises(ExperimentError, match="broken.toml"):
        load_experiment(path)


def test_experiment_error_is_a_value_error() -> None:
    assert issubclass(ExperimentError, ValueError)


# ---------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------


def broken(text: str) -> str:
    return MINIMAL + textwrap.dedent(text)


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        # unknown keys (top level, then inside an arm)
        (MINIMAL.replace("[[arms]]", "replicate = 3\n[[arms]]"), "replicate"),
        (MINIMAL + "override = {}\n", "override"),
        # name
        (MINIMAL.replace('name = "tiny"', 'name = "bad name"'), "name"),
        (MINIMAL.replace('name = "tiny"', 'name = ".."'), "name"),
        (MINIMAL.replace('name = "tiny"', 'name = "."'), "name"),
        (MINIMAL.replace('name = "tiny"', 'name = "a/b"'), "name"),
        (MINIMAL.replace('name = "tiny"', 'name = ""'), "name"),
        (MINIMAL.replace('name = "tiny"', "name = 3"), "name"),
        (MINIMAL.replace('name = "tiny"\n', ""), "name"),
        # task
        (MINIMAL.replace('task = "task/lethe.toml"\n', ""), "task"),
        (MINIMAL.replace('"task/lethe.toml"', '""'), "task"),
        (MINIMAL.replace('"task/lethe.toml"', "7"), "task"),
        # replicates
        (MINIMAL.replace("[[arms]]", "replicates = 0\n[[arms]]"), "replicates"),
        (MINIMAL.replace("[[arms]]", "replicates = -2\n[[arms]]"), "replicates"),
        (MINIMAL.replace("[[arms]]", 'replicates = "3"\n[[arms]]'), "replicates"),
        (MINIMAL.replace("[[arms]]", "replicates = 2.5\n[[arms]]"), "replicates"),
        (MINIMAL.replace("[[arms]]", "replicates = true\n[[arms]]"), "replicates"),
        # arms
        ('name = "tiny"\ntask = "t.toml"\n', "arms"),
        ('name = "tiny"\ntask = "t.toml"\narms = []\n', "arms"),
        ('name = "tiny"\ntask = "t.toml"\narms = [1, 2]\n', "arms"),
        # arm name
        (MINIMAL.replace('name = "only"', ""), "name"),
        (MINIMAL.replace('name = "only"', 'name = "team/forget"'), "name"),
        (MINIMAL.replace('name = "only"', 'name = ".."'), "name"),
        (MINIMAL.replace('name = "only"', 'name = "a:b"'), "name"),
        # preset
        (MINIMAL + 'preset = "team_forget"\n', "preset"),
        (MINIMAL + "preset = 4\n", "preset"),
        # overrides
        (MINIMAL.replace("[[arms]]", "overrides = 3\n[[arms]]"), "overrides"),
        (MINIMAL + "overrides = 3\n", "overrides"),
        (MINIMAL.replace("[[arms]]", "[overrides]\nbudjet = { usd = 1 }\n[[arms]]"), "budjet"),
        (MINIMAL + "overrides = { budjet = { usd = 1 } }\n", "budjet"),
        (MINIMAL.replace("[[arms]]", '[overrides]\npreset = "solo"\n[[arms]]'), "preset"),
        (MINIMAL + 'overrides = { preset = "solo" }\n', "preset"),
    ],
)
def test_invalid_experiments_are_rejected_with_a_helpful_message(
    tmp_path: Path, text: str, needle: str
) -> None:
    with pytest.raises(ExperimentError, match=needle):
        load_experiment(write(tmp_path, text))


def test_duplicate_arm_names_are_rejected(tmp_path: Path) -> None:
    text = MINIMAL + '[[arms]]\nname = "only"\n'
    with pytest.raises(ExperimentError, match="duplicate.*only"):
        load_experiment(write(tmp_path, text))


def test_arm_names_that_differ_only_in_case_collide_on_case_insensitive_disks(
    tmp_path: Path,
) -> None:
    text = MINIMAL + '[[arms]]\nname = "ONLY"\n'
    with pytest.raises(ExperimentError, match="duplicate"):
        load_experiment(write(tmp_path, text))


def test_the_error_names_the_offending_file(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL + "typo = 1\n", "weird-name.toml")
    with pytest.raises(ExperimentError, match="weird-name.toml"):
        load_experiment(path)


@pytest.mark.parametrize("preset", ["solo", "parallel", "team", "team-forget"])
def test_all_documented_presets_are_accepted(tmp_path: Path, preset: str) -> None:
    text = MINIMAL + f'preset = "{preset}"\n'
    assert load_experiment(write(tmp_path, text)).arms[0].preset == preset


@pytest.mark.parametrize("section", ["task", "judge", "team", "forget", "agent", "budget", "run"])
def test_all_documented_config_sections_are_accepted_as_overrides(
    tmp_path: Path, section: str
) -> None:
    text = MINIMAL.replace("[[arms]]", f"[overrides.{section}]\nx = 1\n[[arms]]")
    assert section in load_experiment(write(tmp_path, text)).overrides


# ---------------------------------------------------------------------------------------------
# arm_overrides
# ---------------------------------------------------------------------------------------------


def load_brief(tmp_path: Path) -> Experiment:
    return load_experiment(write(tmp_path, BRIEF_EXAMPLE))


def test_arm_overrides_merges_experiment_and_arm_levels(tmp_path: Path) -> None:
    experiment = load_brief(tmp_path)

    assert arm_overrides(experiment, experiment.arms[0]) == {
        "budget": {"usd": 3.0},
        "agent": {"model": "haiku"},
    }
    assert arm_overrides(experiment, experiment.arms[1]) == {
        "budget": {"usd": 3.0},
        "agent": {"model": "haiku"},
        "forget": {"when": [{"type": "anchored"}]},
    }


def make_experiment(shared: dict, arm: dict) -> tuple[Experiment, Arm]:
    chosen = Arm(name="a", preset=None, overrides=arm)
    return Experiment(
        name="x", task=Path("/t/lethe.toml"), replicates=1, overrides=shared, arms=(chosen,)
    ), chosen


def test_arm_values_win_over_experiment_values() -> None:
    experiment, arm = make_experiment({"budget": {"usd": 3.0}}, {"budget": {"usd": 1.5}})
    assert arm_overrides(experiment, arm) == {"budget": {"usd": 1.5}}


def test_nested_tables_merge_key_by_key() -> None:
    experiment, arm = make_experiment(
        {"budget": {"usd": 3.0, "primary": "usd"}, "agent": {"model": "haiku"}},
        {"budget": {"proposals": 40}, "agent": {"effort": "low"}},
    )
    assert arm_overrides(experiment, arm) == {
        "budget": {"usd": 3.0, "primary": "usd", "proposals": 40},
        "agent": {"model": "haiku", "effort": "low"},
    }


def test_lists_replace_instead_of_concatenating() -> None:
    experiment, arm = make_experiment(
        {"forget": {"when": [{"type": "plateau"}, {"type": "stalled"}]}},
        {"forget": {"when": [{"type": "anchored"}]}},
    )
    assert arm_overrides(experiment, arm) == {"forget": {"when": [{"type": "anchored"}]}}


def test_an_empty_list_replaces_too_so_an_arm_can_disable_forgetting() -> None:
    experiment, arm = make_experiment(
        {"forget": {"when": [{"type": "plateau"}]}}, {"forget": {"when": []}}
    )
    assert arm_overrides(experiment, arm) == {"forget": {"when": []}}


def test_an_arm_without_overrides_gets_the_experiment_level_ones() -> None:
    experiment, arm = make_experiment({"budget": {"usd": 3.0}}, {})
    assert arm_overrides(experiment, arm) == {"budget": {"usd": 3.0}}


def test_nothing_to_merge_gives_an_empty_dict() -> None:
    experiment, arm = make_experiment({}, {})
    assert arm_overrides(experiment, arm) == {}


def test_arm_overrides_returns_an_independent_copy() -> None:
    experiment, arm = make_experiment(
        {"budget": {"usd": 3.0}}, {"forget": {"when": [{"type": "anchored"}]}}
    )

    merged = arm_overrides(experiment, arm)
    merged["budget"]["usd"] = 99.0
    merged["forget"]["when"][0]["type"] = "mutated"
    merged["forget"]["when"].append({"type": "extra"})

    assert experiment.overrides == {"budget": {"usd": 3.0}}
    assert arm.overrides == {"forget": {"when": [{"type": "anchored"}]}}
    assert arm_overrides(experiment, arm)["budget"] == {"usd": 3.0}


def test_loaded_experiment_does_not_share_state_between_arms(tmp_path: Path) -> None:
    experiment = load_brief(tmp_path)
    first = arm_overrides(experiment, experiment.arms[0])
    first["agent"]["model"] = "opus"
    assert arm_overrides(experiment, experiment.arms[1])["agent"] == {"model": "haiku"}


# ---------------------------------------------------------------------------------------------
# select_arms
# ---------------------------------------------------------------------------------------------


def three_arm_experiment(tmp_path: Path) -> Experiment:
    text = MINIMAL + '[[arms]]\nname = "second"\n[[arms]]\nname = "third"\n'
    return load_experiment(write(tmp_path, text))


def test_select_arms_defaults_to_all_in_file_order(tmp_path: Path) -> None:
    experiment = three_arm_experiment(tmp_path)
    assert [a.name for a in select_arms(experiment, None)] == ["only", "second", "third"]


def test_select_arms_keeps_file_order_whatever_order_was_requested(tmp_path: Path) -> None:
    experiment = three_arm_experiment(tmp_path)
    assert [a.name for a in select_arms(experiment, ["third", "only"])] == ["only", "third"]


def test_select_arms_ignores_repeats(tmp_path: Path) -> None:
    experiment = three_arm_experiment(tmp_path)
    assert [a.name for a in select_arms(experiment, ["second", "second"])] == ["second"]


def test_select_arms_rejects_unknown_names_and_lists_the_known_ones(tmp_path: Path) -> None:
    experiment = three_arm_experiment(tmp_path)
    with pytest.raises(ExperimentError, match=r"nope.*only, second, third"):
        select_arms(experiment, ["second", "nope"])


def test_select_arms_rejects_an_empty_selection(tmp_path: Path) -> None:
    experiment = three_arm_experiment(tmp_path)
    with pytest.raises(ExperimentError, match="no arms"):
        select_arms(experiment, [])
