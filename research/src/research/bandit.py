"""UCB1 over an experiment's arms: a first step toward a learned forgetting policy.

Each arm of the experiment file is a forgetting policy (a preset plus overrides). Instead of
giving every arm the same number of replicates, the bandit treats each run as one pull of its
arm and spends a fixed number of pulls where they look most valuable:

    python -m research.bandit research/config/experiments/bets-toy-mis.toml --pulls 12

Reward of a run: its normalized improvement over the seed score,
    (best - seed) / max(|seed|, 1e-9)    when higher scores are better,
    (seed - best) / max(|seed|, 1e-9)    when lower scores are better,
clipped to [-1, 1]. A run that failed, ended abnormally (interrupted, every agent gave up) or has
no valid seed or best earns 0.

Selection is UCB1 (Auer et al., 2002): each arm is pulled once, in file order; after that the arm
maximizing `mean_reward + c * sqrt(2 ln N / n)` is pulled, where N counts all pulls so far and n
the arm's own. Ties go to the arm listed first. c = 1 is the textbook rule. Rewards are usually
small (a few percent to a few tens of percent), so with c = 1 the bonus dominates and a budget of
a dozen pulls comes out nearly round-robin; c around 0.1 to 0.3 exploits sooner. There is no
training here: this only decides which arm to run next.

A pull is one new replicate of the chosen arm, in the next free `rep<k>` slot under
`<runs_root>/<experiment>/<arm>`. The bandit resumes: complete runs already on disk seed the
statistics. A failed pull leaves a partial directory, which keeps its slot but is not counted on
the next session; within the session that failed it counts as a pull with reward 0.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lethe.api import run_task
from lethe.domain.types import Direction
from research.experiment import Experiment, ExperimentError, load_experiment
from research.matrix import Runner, RunRecord, RunStatus, execute_run
from research.runs import (
    EVENTS_FILE,
    arm_dir_for,
    find_rep_dirs,
    is_run_complete,
    next_free_rep,
)
from research.summary import RunSummary, summarize_run
from research.textfmt import format_table

__all__ = [
    "ArmStats",
    "Pull",
    "arm_stats",
    "load_rewards",
    "main",
    "normalized_improvement",
    "reward_of_run",
    "run_bandit",
    "select_arm",
    "ucb_index",
]

log = logging.getLogger(__name__)

_SEED_FLOOR = 1e-9  # keeps a zero seed score from dividing by zero


@dataclass(frozen=True)
class Pull:
    """One pull: a new replicate of `arm`, the reward it earned, and the index that chose it."""

    arm: str
    rep: int
    reward: float
    status: RunStatus  # "completed" or "failed"
    run_dir: Path
    index: float  # the arm's UCB index when chosen; infinite while it had never been pulled
    message: str = ""


@dataclass(frozen=True)
class ArmStats:
    arm: str
    pulls: int
    mean_reward: float | None  # None before the first pull
    index: float  # infinite before the first pull


# ---------------------------------------------------------------------------------------------
# reward and selection
# ---------------------------------------------------------------------------------------------


def normalized_improvement(seed: float | None, best: float | None, direction: Direction) -> float:
    """Improvement of `best` over `seed` relative to |seed|, clipped to [-1, 1] (0 if unknown)."""
    if seed is None or best is None or not (math.isfinite(seed) and math.isfinite(best)):
        return 0.0
    gain = best - seed if direction == "max" else seed - best
    return min(1.0, max(-1.0, gain / max(abs(seed), _SEED_FLOOR)))


def reward_of_run(summary: RunSummary | None) -> float:
    """Reward of a summarized run; a missing or unfinished run earns 0."""
    if summary is None or not summary.finished:
        return 0.0
    return normalized_improvement(summary.seed_score, summary.final_best, summary.direction)


def ucb_index(mean_reward: float, pulls: int, total_pulls: int, c: float) -> float:
    """UCB1 index `mean + c * sqrt(2 ln N / n)`; infinite for an arm that was never pulled."""
    if pulls == 0:
        return math.inf
    return mean_reward + c * math.sqrt(2 * math.log(total_pulls) / pulls)


def select_arm(
    arm_names: Sequence[str], rewards: Mapping[str, Sequence[float]], c: float
) -> tuple[str, float]:
    """The next arm to pull and its index: first any unpulled arm, then the highest UCB index."""
    for name in arm_names:
        if not rewards[name]:
            return name, math.inf
    total = sum(len(rewards[name]) for name in arm_names)
    chosen, chosen_index = arm_names[0], -math.inf
    for name in arm_names:
        history = rewards[name]
        index = ucb_index(sum(history) / len(history), len(history), total, c)
        if index > chosen_index:  # strict: the arm listed first wins ties
            chosen, chosen_index = name, index
    return chosen, chosen_index


def arm_stats(
    arm_names: Sequence[str], rewards: Mapping[str, Sequence[float]], c: float
) -> list[ArmStats]:
    """Pull count, mean reward and current UCB index per arm, in `arm_names` order."""
    total = sum(len(rewards[name]) for name in arm_names)
    stats = []
    for name in arm_names:
        history = rewards[name]
        mean = sum(history) / len(history) if history else None
        stats.append(
            ArmStats(
                name,
                len(history),
                mean,
                math.inf if mean is None else ucb_index(mean, len(history), total, c),
            )
        )
    return stats


# ---------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------


def load_rewards(experiment: Experiment, runs_root: Path) -> dict[str, list[float]]:
    """Rewards of the complete runs already on disk, per arm of `experiment` (file order).

    Unfinished, interrupted and parked runs are not pulls. A complete run whose log cannot be
    summarized counts as a pull with reward 0. Directories of arms not in the file are ignored.
    """
    rewards: dict[str, list[float]] = {}
    for arm in experiment.arms:
        history = rewards[arm.name] = []
        arm_dir = arm_dir_for(Path(runs_root), experiment.name, arm.name)
        for rep, run_dir in sorted(find_rep_dirs(arm_dir).items()):
            if not is_run_complete(run_dir / EVENTS_FILE):
                continue
            try:
                summary: RunSummary | None = summarize_run(run_dir, arm=arm.name, rep=rep)
            except (OSError, ValueError) as exc:
                log.warning(
                    "%s/rep%d: cannot summarize (%s); counting reward 0", arm.name, rep, exc
                )
                summary = None
            history.append(reward_of_run(summary))
    return rewards


def _reward_of_record(record: RunRecord) -> tuple[float, str]:
    """(reward, explanation) of a run that `execute_run` has just handled."""
    if record.status == "failed":
        return 0.0, record.message
    try:
        summary = summarize_run(record.run_dir, arm=record.arm, rep=record.rep)
    except (OSError, ValueError) as exc:
        return 0.0, f"cannot summarize the run: {exc}"
    return reward_of_run(summary), ""


def run_bandit(
    experiment_path: Path | str,
    *,
    pulls: int,
    runs_root: Path | str,
    runner: Runner = run_task,
    c: float = 1.0,
) -> list[Pull]:
    """Make `pulls` new pulls with UCB1, one run at a time; returns them in order.

    `pulls` counts new pulls only, on top of whatever complete runs are already on disk.
    """
    if pulls < 0:
        raise ValueError(f"pulls must be >= 0, got {pulls}")
    if not (math.isfinite(c) and c >= 0):
        raise ValueError(f"c must be a finite number >= 0, got {c}")
    experiment = load_experiment(experiment_path)
    root = Path(runs_root)
    names = [arm.name for arm in experiment.arms]
    by_name = {arm.name: arm for arm in experiment.arms}
    rewards = load_rewards(experiment, root)
    history: list[Pull] = []
    for number in range(1, pulls + 1):
        name, index = select_arm(names, rewards, c)
        rep = next_free_rep(arm_dir_for(root, experiment.name, name))
        log.info("[%d/%d] pulling %s as rep%d (index %s)", number, pulls, name, rep, index)
        record = execute_run(experiment, by_name[name], rep, runs_root=root, runner=runner)
        reward, message = _reward_of_record(record)
        rewards[name].append(reward)
        history.append(Pull(name, rep, reward, record.status, record.run_dir, index, message))
        log.info(
            "[%d/%d] %s/rep%d: %s, reward %+.3f", number, pulls, name, rep, record.status, reward
        )
    return history


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def _fmt_index(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def main(argv: Sequence[str] | None = None, *, runner: Runner = run_task) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research.bandit",
        description="Spend a fixed number of runs on an experiment's arms with UCB1.",
    )
    parser.add_argument("experiment", type=Path, help="experiment TOML file")
    parser.add_argument("--pulls", type=int, required=True, help="new runs to make")
    parser.add_argument("--runs-root", type=Path, default=Path("research/runs"))
    parser.add_argument("--c", type=float, default=1.0, help="exploration constant (default 1)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        experiment = load_experiment(args.experiment)
        if not experiment.task.is_file():  # fail now, not as N identical failed pulls
            raise ExperimentError(f"task file not found: {experiment.task}")
        rewards = load_rewards(experiment, args.runs_root)
        pulls = run_bandit(
            args.experiment, pulls=args.pulls, runs_root=args.runs_root, runner=runner, c=args.c
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for pull in pulls:
        rewards[pull.arm].append(pull.reward)

    pull_rows = [
        [str(n), p.arm, f"rep{p.rep}", f"{p.reward:+.3f}", _fmt_index(p.index), p.status, p.message]
        for n, p in enumerate(pulls, 1)
    ]
    if pull_rows:
        headers = ["#", "arm", "rep", "reward", "UCB index", "status", "message"]
        print("Pulls\n" + format_table(headers, pull_rows, align="rllrrll") + "\n")
    stat_rows = [
        [
            s.arm,
            str(s.pulls),
            "-" if s.mean_reward is None else f"{s.mean_reward:+.3f}",
            _fmt_index(s.index),
        ]
        for s in arm_stats([a.name for a in experiment.arms], rewards, args.c)
    ]
    title = f"Arms (c = {args.c:g}), counting runs from earlier sessions"
    print(title + "\n" + format_table(["arm", "pulls", "mean reward", "UCB index"], stat_rows))
    return 1 if any(p.status == "failed" for p in pulls) else 0


if __name__ == "__main__":
    raise SystemExit(main())
