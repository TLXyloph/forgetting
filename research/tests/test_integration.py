"""End to end against the real harness: `lethe.api.run_task` with a scripted agent (no Claude).

The unit tests build event logs by hand; these runs produce genuine ones, so they guard the
assumptions the tooling makes about the log (event order, asynchronous judging, how a run ends).
The task is a toy: the judge reads `SCORE = <n>` from solver.py, and the scripted agent adds 5 on
each of the first two turns of a session and then stalls with a "practical limit" message, the
anchoring pattern that forgetting is meant to cure.
"""

from __future__ import annotations

import asyncio
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from lethe.agents.scripted import ScriptCall, ScriptedBackend, ScriptReply
from lethe.api import run_task
from lethe.domain.types import ToolCall
from research.analyze import main as analyze_main
from research.matrix import RunRecord, run_matrix
from research.runs import EVENTS_FILE, finish_reason, is_run_complete
from research.summary import RunSummary, load_summaries

SEED_SCORE = 10
SESSION_PRODUCTIVE_TURNS = 2
GAIN_PER_TURN = 5

JUDGE = textwrap.dedent(
    """
    import json, re
    match = re.search(r"^SCORE = (-?\\d+)", open("solver.py").read(), re.M)
    print(json.dumps({"score": int(match.group(1)), "feedback": "ok"}))
    """
)

TASK = textwrap.dedent(
    """
    [task]
    name = "tiny"
    description = "Raise SCORE in solver.py."
    seed = "seed"
    editable = ["solver.py"]

    [judge]
    run = "python3 \\"$LETHE_TASK_DIR/judge/score.py\\""
    direction = "max"
    timeout = 60

    [budget]
    usd = 1.0
    """
)

Script = Callable[[ScriptCall], ScriptReply]


def anchoring_script(call: ScriptCall) -> ScriptReply:
    """Productive for the first turns of a session, then anchored: no edit, a plateau claim."""
    solver = call.workspace / "solver.py"
    current = int(re.search(r"SCORE = (-?\d+)", solver.read_text(encoding="utf-8")).group(1))  # type: ignore[union-attr]
    if call.turn_in_session <= SESSION_PRODUCTIVE_TURNS:
        solver.write_text(f"SCORE = {current + GAIN_PER_TURN}\n", encoding="utf-8")
        return ScriptReply(
            cost_usd=0.1,
            output_tokens=500,
            thinking_tokens=800 // call.turn_in_session,
            tool_calls=(ToolCall("Edit", "solver.py"),),
        )
    return ScriptReply(
        final_text="I've reached the practical limit of this task.",
        cost_usd=0.1,
        output_tokens=50,
        thinking_tokens=20,
    )


def make_runner(script: Script) -> Callable[..., Path]:
    def runner(config_path: Path, **kwargs: object) -> Path:
        return run_task(config_path, backend=ScriptedBackend(script), **kwargs)  # type: ignore[arg-type]

    return runner


def write_experiment(root: Path, arms: list[tuple[str, str]], *, proposals: int) -> Path:
    task = root / "task"
    (task / "seed").mkdir(parents=True)
    (task / "judge").mkdir()
    (task / "seed" / "solver.py").write_text(f"SCORE = {SEED_SCORE}\n", encoding="utf-8")
    (task / "judge" / "score.py").write_text(JUDGE, encoding="utf-8")
    (task / "lethe.toml").write_text(TASK, encoding="utf-8")
    lines = [
        'name = "e2e"',
        'task = "task/lethe.toml"',
        "replicates = 1",
        "[overrides]",
        f'budget = {{ proposals = {proposals}, primary = "proposals", usd = 0 }}',
    ]
    for name, preset in arms:
        lines += ["[[arms]]", f'name = "{name}"', f'preset = "{preset}"']
    path = root / "e2e.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class Harness:
    experiment_file: Path
    runs_root: Path
    records: list[RunRecord]

    @property
    def experiment_dir(self) -> Path:
        return self.runs_root / "e2e"


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Harness:
    root = tmp_path_factory.mktemp("harness")
    experiment = write_experiment(
        root, [("solo", "solo"), ("team-forget", "team-forget")], proposals=12
    )
    runs_root = root / "runs"
    records = run_matrix(experiment, runs_root=runs_root, runner=make_runner(anchoring_script))
    return Harness(experiment, runs_root, records)


@pytest.fixture(scope="module")
def summaries(harness: Harness) -> dict[str, RunSummary]:
    return {s.arm: s for s in load_summaries(harness.experiment_dir)}


# ---------------------------------------------------------------------------------------------
# genuine logs through the analysis
# ---------------------------------------------------------------------------------------------


def test_the_matrix_runs_the_real_harness_to_completion(harness: Harness) -> None:
    assert [(r.arm, r.rep, r.status) for r in harness.records] == [
        ("solo", 1, "completed"),
        ("team-forget", 1, "completed"),
    ]


def test_real_runs_are_finished_with_a_budget_reason(summaries: dict[str, RunSummary]) -> None:
    for summary in summaries.values():
        assert summary.finished
        assert summary.finish_reason is not None
        assert "budget exhausted" in summary.finish_reason
        assert summary.direction == "max"
        assert summary.seed_score == SEED_SCORE
        assert summary.budget_usd is None  # the cap is in proposals, not dollars


def test_arms_come_back_in_the_order_they_ran(summaries: dict[str, RunSummary]) -> None:
    assert list(summaries) == ["solo", "team-forget"]


def test_spend_and_proposals_follow_the_turn_records(summaries: dict[str, RunSummary]) -> None:
    for summary in summaries.values():
        assert summary.proposals == len(summary.turns) >= 12
        assert summary.spend_usd == pytest.approx(0.1 * len(summary.turns))
        assert [t.spend for t in summary.turns] == sorted(t.spend for t in summary.turns)
        assert summary.turns[-1].spend == pytest.approx(summary.spend_usd)


def test_a_single_anchored_agent_stalls_after_its_productive_turns(
    summaries: dict[str, RunSummary],
) -> None:
    solo = summaries["solo"]
    assert solo.final_best == SEED_SCORE + SESSION_PRODUCTIVE_TURNS * GAIN_PER_TURN
    assert solo.team_forgets == 0
    assert {t.status for t in solo.turns[:SESSION_PRODUCTIVE_TURNS]} == {"candidate"}
    assert {t.status for t in solo.turns[SESSION_PRODUCTIVE_TURNS:]} == {"empty"}
    assert all(t.lines_changed > 0 for t in solo.turns[:SESSION_PRODUCTIVE_TURNS])
    assert all(t.lines_changed == 0 for t in solo.turns[SESSION_PRODUCTIVE_TURNS:])


def test_forgetting_lets_the_team_get_past_the_anchoring_plateau(
    summaries: dict[str, RunSummary],
) -> None:
    solo, team = summaries["solo"], summaries["team-forget"]
    assert team.team_forgets >= 1  # how many depends on how the barrier meets a tiny budget
    assert team.final_best is not None and solo.final_best is not None
    assert team.final_best > solo.final_best


def test_rounds_of_a_real_forgetting_run_chain_from_best_to_best(
    summaries: dict[str, RunSummary],
) -> None:
    team = summaries["team-forget"]
    assert len(team.rounds) == team.team_forgets + 1
    assert team.rounds[0].seed == SEED_SCORE
    for earlier, later in zip(team.rounds, team.rounds[1:], strict=False):
        assert later.seed == earlier.best  # a forget restarts every agent from the best so far
    bests = [r.best for r in team.rounds]
    assert bests == sorted(bests)  # type: ignore[type-var]
    assert bests[-1] == team.final_best


def test_the_best_vs_spend_curve_of_a_real_run(summaries: dict[str, RunSummary]) -> None:
    for summary in summaries.values():
        spends = [spend for spend, _ in summary.curve]
        bests = [best for _, best in summary.curve]
        assert summary.curve[0] == (0.0, SEED_SCORE)
        assert spends == sorted(spends)
        assert bests == sorted(bests)
        assert spends[-1] == pytest.approx(summary.spend_usd)
        assert bests[-1] == summary.final_best


def test_a_second_invocation_skips_every_finished_run(harness: Harness) -> None:
    never = make_runner(lambda call: pytest.fail("a finished run must not be rerun"))  # type: ignore[arg-type,return-value]
    records = run_matrix(harness.experiment_file, runs_root=harness.runs_root, runner=never)
    assert {r.status for r in records} == {"skipped"}


def test_the_analysis_cli_reads_the_real_logs(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    code = analyze_main([str(harness.experiment_dir), "--compare", "team-forget:solo"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Table 1" in out and "Table 2" in out
    assert "team-forget" in out and "solo" in out
    assert "needs at least 2" in out  # one replicate per arm cannot support a test


# ---------------------------------------------------------------------------------------------
# resuming after the harness was cut short
# ---------------------------------------------------------------------------------------------


def stop_at_call(call_number: int, exception: BaseException) -> Script:
    """The anchoring script, but the `call_number`-th call anywhere raises `exception` once."""
    seen = {"calls": 0}

    def script(call: ScriptCall) -> ScriptReply:
        seen["calls"] += 1
        if seen["calls"] == call_number:
            raise exception
        return anchoring_script(call)

    return script


def one_solo_arm(tmp_path: Path) -> tuple[Path, Path, Path]:
    experiment = write_experiment(tmp_path, [("solo", "solo")], proposals=6)
    runs_root = tmp_path / "runs"
    return experiment, runs_root, runs_root / "e2e" / "solo" / "rep1"


def test_a_cancelled_run_is_parked_and_rerun_not_skipped(tmp_path: Path) -> None:
    """Ctrl-C reaches the orchestrator as a cancellation. Whatever it writes, the replicate must
    not count as done: the next invocation parks it and runs it again."""
    experiment, runs_root, run_dir = one_solo_arm(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        run_matrix(
            experiment,
            runs_root=runs_root,
            runner=make_runner(stop_at_call(3, asyncio.CancelledError())),
        )
    assert (run_dir / EVENTS_FILE).is_file()
    assert not is_run_complete(run_dir / EVENTS_FILE)  # the harness may have written "interrupted"

    records = run_matrix(experiment, runs_root=runs_root, runner=make_runner(anchoring_script))

    assert [r.status for r in records] == ["completed"]
    assert is_run_complete(run_dir / EVENTS_FILE)
    parked = [p for p in run_dir.parent.iterdir() if ".incomplete-" in p.name]
    assert len(parked) == 1
    assert not is_run_complete(parked[0] / EVENTS_FILE)
    reason = finish_reason(run_dir / EVENTS_FILE)
    assert reason is not None and "budget exhausted" in reason


def test_a_crashed_run_is_recorded_as_failed_and_retried(tmp_path: Path) -> None:
    experiment, runs_root, run_dir = one_solo_arm(tmp_path)

    records = run_matrix(
        experiment,
        runs_root=runs_root,
        runner=make_runner(stop_at_call(3, RuntimeError("the agent script blew up"))),
    )

    assert [r.status for r in records] == ["failed"]
    assert "blew up" in records[0].message
    assert not is_run_complete(run_dir / EVENTS_FILE)

    retry = run_matrix(experiment, runs_root=runs_root, runner=make_runner(anchoring_script))

    assert [r.status for r in retry] == ["completed"]
    assert len([p for p in run_dir.parent.iterdir() if ".incomplete-" in p.name]) == 1
