"""Tests for research.bandit: running pulls, resuming from disk and failed pulls."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import pytest

from lethe.api import run_task
from research.bandit import Pull, load_rewards, run_bandit
from research.experiment import ExperimentError, load_experiment
from research.runs import EVENTS_FILE

# ---------------------------------------------------------------------------------------------
# run_bandit
# ---------------------------------------------------------------------------------------------


def test_every_arm_is_pulled_once_in_file_order_first(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=3, runs_root=runs_root, runner=runner)

    assert [p.arm for p in pulls] == ["a", "b", "c"]
    assert [p.index for p in pulls] == [math.inf] * 3


@pytest.mark.parametrize("c", [1.0, 0.1, 0.0, 3.0])
def test_selection_follows_ucb1(
    BanditRunner: type,
    expected_ucb_sequence: Any,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    c: float,
) -> None:
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=14, runs_root=runs_root, runner=runner, c=c)

    expected = expected_ucb_sequence({"a": 0.5, "b": 0.05, "c": 0.01}, 14, c)
    assert [p.arm for p in pulls] == expected


def test_a_small_c_settles_on_the_best_arm(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    pulls = run_bandit(
        experiment_file, pulls=9, runs_root=runs_root, runner=BanditRunner(run_builder), c=0.05
    )
    assert [p.arm for p in pulls[3:]] == ["a"] * 6


def test_default_exploration_constant_is_one() -> None:
    assert inspect.signature(run_bandit).parameters["c"].default == 1.0
    assert inspect.signature(run_bandit).parameters["runner"].default is run_task


def test_each_pull_is_a_new_replicate_of_the_chosen_arm(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=8, runs_root=runs_root, runner=runner, c=0.05)

    for arm in "abc":
        reps = [p.rep for p in pulls if p.arm == arm]
        assert reps == list(range(1, len(reps) + 1))
    for pull in pulls:
        assert pull.run_dir == runs_root / "bets" / pull.arm / f"rep{pull.rep}"
        assert (pull.run_dir / EVENTS_FILE).is_file()
    assert [c["run_name"] for c in runner.calls if c["runs_dir"].name == "a"][:3] == [
        "rep1",
        "rep2",
        "rep3",
    ]


def test_runner_gets_the_arms_preset_and_merged_overrides(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder)
    run_bandit(experiment_file, pulls=3, runs_root=runs_root, runner=runner)

    a, b, c = runner.calls
    task = load_experiment(experiment_file).task
    assert (a["config_path"], a["preset"], a["overrides"]) == (
        task,
        "solo",
        {"budget": {"usd": 3.0}},
    )
    assert b["preset"] == "team"
    assert b["overrides"] == {
        "budget": {"usd": 3.0},
        "forget": {"when": [{"type": "anchored"}]},
    }
    assert c["preset"] == "team-forget"
    assert a["runs_dir"] == runs_root / "bets" / "a"


def test_pull_records_carry_the_reward(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    pulls = run_bandit(
        experiment_file, pulls=3, runs_root=runs_root, runner=BanditRunner(run_builder)
    )

    assert [p.reward for p in pulls] == pytest.approx([0.5, 0.05, 0.01])
    assert {p.status for p in pulls} == {"completed"}
    assert all(isinstance(p, Pull) and p.message == "" for p in pulls)


def test_min_tasks_reward_a_lower_score(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    finals = {"a": 80.0, "b": 100.0, "c": 120.0}  # seed 100, lower is better
    runner = BanditRunner(run_builder, finals, direction="min")

    pulls = run_bandit(experiment_file, pulls=3, runs_root=runs_root, runner=runner)

    assert [p.reward for p in pulls] == pytest.approx([0.2, 0.0, 0.0])  # best-so-far never < seed


def test_zero_pulls_do_nothing(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder)
    assert run_bandit(experiment_file, pulls=0, runs_root=runs_root, runner=runner) == []
    assert runner.calls == []


@pytest.mark.parametrize("kwargs", [{"pulls": -1}, {"pulls": 1, "c": -0.5}])
def test_invalid_parameters_are_rejected(
    BanditRunner: type,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
    kwargs: dict[str, Any],
) -> None:
    runner = BanditRunner(run_builder)
    with pytest.raises(ValueError):
        run_bandit(experiment_file, runs_root=runs_root, runner=runner, **kwargs)
    assert runner.calls == []


# ---------------------------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------------------------


def finished_run(
    builder: type, run_dir: Path, final: float, *, reason: str = "usd budget exhausted"
) -> None:
    b = builder(run_dir, agents=("a1",)).start(100.0)
    b.candidate("a1", final, cost=0.5)
    b.finish(reason)


def test_existing_finished_runs_seed_the_statistics(
    BanditRunner: type,
    expected_ucb_sequence: Any,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
) -> None:
    root = runs_root / "bets"
    finished_run(run_builder, root / "a" / "rep1", 150.0)  # reward 0.5
    finished_run(run_builder, root / "a" / "rep2", 130.0)  # reward 0.3
    finished_run(run_builder, root / "b" / "rep1", 105.0)  # reward 0.05
    initial = {"a": [0.5, 0.3], "b": [0.05], "c": []}
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=6, runs_root=runs_root, runner=runner, c=1.0)

    assert pulls[0].arm == "c"  # the only arm never pulled goes first
    assert [p.arm for p in pulls] == expected_ucb_sequence(
        {"a": 0.5, "b": 0.05, "c": 0.01}, 6, 1.0, initial
    )


def test_new_replicates_continue_the_numbering(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    root = runs_root / "bets"
    for rep in (1, 2):
        finished_run(run_builder, root / "a" / f"rep{rep}", 150.0)
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=4, runs_root=runs_root, runner=runner, c=0.0)

    a_reps = [p.rep for p in pulls if p.arm == "a"]
    assert a_reps and a_reps[0] == 3
    assert [p.rep for p in pulls if p.arm == "c"] == [1]


def test_resuming_twice_equals_running_straight_through(
    BanditRunner: type,
    expected_ucb_sequence: Any,
    experiment_file: Path,
    runs_root: Path,
    run_builder: type,
) -> None:
    runner = BanditRunner(run_builder)
    first = run_bandit(experiment_file, pulls=5, runs_root=runs_root, runner=runner)
    second = run_bandit(experiment_file, pulls=5, runs_root=runs_root, runner=runner)

    assert [p.arm for p in first + second] == expected_ucb_sequence(
        {"a": 0.5, "b": 0.05, "c": 0.01}, 10, 1.0
    )


def test_unfinished_and_parked_runs_are_not_counted_as_pulls(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    root = runs_root / "bets"
    b = run_builder(root / "a" / "rep1", agents=("a1",)).start(100.0)
    b.candidate("a1", 150.0, cost=0.5)  # no run_finished: still "running" or killed
    finished_run(run_builder, root / "b" / "rep1", 150.0, reason="interrupted")
    finished_run(run_builder, root / "c" / "rep1.incomplete-20260101T000000Z", 150.0)
    runner = BanditRunner(run_builder)

    pulls = run_bandit(experiment_file, pulls=3, runs_root=runs_root, runner=runner)

    assert [p.arm for p in pulls] == ["a", "b", "c"]  # all three still count as unpulled
    # the partial run keeps its slot; the new replicate goes to the next free one
    assert [p.rep for p in pulls] == [2, 2, 1]
    assert (root / "a" / "rep1" / EVENTS_FILE).is_file()


def test_runs_of_arms_that_are_not_in_the_file_are_ignored(
    experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    finished_run(run_builder, runs_root / "bets" / "retired-arm" / "rep1", 150.0)
    assert load_rewards(load_experiment(experiment_file), runs_root) == {"a": [], "b": [], "c": []}


def test_load_rewards_reads_finished_runs_per_arm(
    experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    root = runs_root / "bets"
    finished_run(run_builder, root / "b" / "rep1", 120.0)
    finished_run(run_builder, root / "b" / "rep2", 110.0)

    rewards = load_rewards(load_experiment(experiment_file), runs_root)

    assert rewards["b"] == pytest.approx([0.2, 0.1])
    assert rewards["a"] == [] and rewards["c"] == []


def test_load_rewards_of_a_missing_directory_is_empty(
    experiment_file: Path, tmp_path: Path
) -> None:
    rewards = load_rewards(load_experiment(experiment_file), tmp_path / "nothing")
    assert rewards == {"a": [], "b": [], "c": []}


def test_an_invalid_seed_in_a_finished_run_counts_as_a_zero_reward_pull(
    experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    b = run_builder(runs_root / "bets" / "a" / "rep1", agents=("a1",)).start(None)
    b.candidate("a1", 150.0, cost=0.5)
    b.finish()
    assert load_rewards(load_experiment(experiment_file), runs_root)["a"] == [0.0]


# ---------------------------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------------------------


def test_a_failed_pull_earns_zero_and_still_counts_as_a_pull(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder, fail=["a"])

    pulls = run_bandit(experiment_file, pulls=6, runs_root=runs_root, runner=runner, c=0.0)

    first = pulls[0]
    assert (first.arm, first.status, first.reward) == ("a", "failed", 0.0)
    assert "boom in a" in first.message
    # with a zero reward on record, arm a is no longer the favourite (b beats it greedily)
    assert pulls[3].arm == "b"
    assert pulls[0].rep == 1 and runner.arms[0] == "a"


def test_a_failed_replicate_keeps_its_slot_and_the_next_pull_uses_a_fresh_one(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder, fail=["a"])
    pulls = run_bandit(experiment_file, pulls=8, runs_root=runs_root, runner=runner, c=5.0)

    a_pulls = [p for p in pulls if p.arm == "a"]
    assert len(a_pulls) >= 2
    assert [p.rep for p in a_pulls] == list(range(1, len(a_pulls) + 1))
    for pull in a_pulls:
        assert (runs_root / "bets" / "a" / f"rep{pull.rep}").is_dir()


def test_a_run_that_ends_abnormally_counts_as_a_failed_pull(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    runner = BanditRunner(run_builder, end_reason={"a": "all agents stopped after repeated errors"})

    pulls = run_bandit(experiment_file, pulls=1, runs_root=runs_root, runner=runner)

    assert (pulls[0].status, pulls[0].reward) == ("failed", 0.0)
    assert "repeated errors" in pulls[0].message


def test_a_complete_run_that_cannot_be_summarized_earns_zero(
    BanditRunner: type, experiment_file: Path, runs_root: Path, run_builder: type
) -> None:
    class Truncating(BanditRunner):
        def __call__(self, config_path: Path, **kwargs: Any) -> Path:
            run_dir = super().__call__(config_path, **kwargs)
            lines = (run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
            # keep only run_finished: the run still looks complete, but there is no run_started
            (run_dir / EVENTS_FILE).write_text(lines[-1] + "\n", encoding="utf-8")
            return run_dir

    pulls = run_bandit(
        experiment_file, pulls=1, runs_root=runs_root, runner=Truncating(run_builder)
    )

    assert pulls[0].status == "completed"
    assert pulls[0].reward == 0.0
    assert "run_started" in pulls[0].message  # says why


def test_unknown_experiment_file_raises(
    BanditRunner: type, tmp_path: Path, run_builder: type
) -> None:
    with pytest.raises(FileNotFoundError):
        run_bandit(
            tmp_path / "nope.toml", pulls=1, runs_root=tmp_path, runner=BanditRunner(run_builder)
        )


def test_invalid_experiment_raises(BanditRunner: type, tmp_path: Path, run_builder: type) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n', encoding="utf-8")
    with pytest.raises(ExperimentError):
        run_bandit(bad, pulls=1, runs_root=tmp_path, runner=BanditRunner(run_builder))
