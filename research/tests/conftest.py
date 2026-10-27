"""Shared fixtures: a builder for synthetic event logs and a RunSummary factory.

Tests cannot import sibling modules (pytest runs with `--import-mode=importlib`), so helpers are
exposed as fixtures that return a class or a factory function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RunFinished,
    RunStarted,
    TurnFinished,
)
from lethe.store import EventStore

SEED_ID = "c0000"


class RunBuilder:
    """Writes a realistic, schema-valid `events.jsonl` with `lethe.store.EventStore`.

    Event order mirrors the harness: RunStarted, the seed candidate c0000 (agent "seed") with its
    judge result, EpochStarted(start); then per turn CandidateCommitted, TurnFinished,
    JudgeFinished (candidate turns) or a lone TurnFinished (empty and error turns). A forget is
    ForgetDecided followed by EpochStarted, and the log ends with RunFinished.
    Timestamps advance by one second per event, starting at `t0`.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        direction: str = "max",
        budget_usd: float | None = 3.0,
        agents: Sequence[str] = ("a1", "a2", "a3", "a4"),
        t0: float = 1_000.0,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.direction = direction
        self.agents = tuple(agents)
        self.scores: dict[str, float] = {}  # valid candidates only
        self._now = t0
        self._candidates = 0
        self._epochs = {agent: 1 for agent in self.agents}
        self._turns = {agent: 0 for agent in self.agents}
        self._unjudged: dict[str, float] = {}  # candidates committed with judge=False
        if config is None:
            config = {"budget": {"usd": budget_usd or 0, "proposals": 0, "primary": "usd"}}
        self.config = dict(config)
        self.store = EventStore(run_dir / "events.jsonl", clock=self._clock)

    # -- plumbing ---------------------------------------------------------------------------

    def _clock(self) -> float:
        self._now += 1.0
        return self._now

    def _next_id(self) -> str:
        cand_id = f"c{self._candidates:04d}"
        self._candidates += 1
        return cand_id

    def best(self) -> tuple[str | None, float | None]:
        """Best valid candidate so far under the run's direction."""
        if not self.scores:
            return None, None
        pick = max if self.direction == "max" else min
        cand_id = pick(self.scores, key=self.scores.__getitem__)
        return cand_id, self.scores[cand_id]

    def _commit(self, agent: str, cand_id: str, parent: str | None, lines: int) -> None:
        self.store.append(
            CandidateCommitted(
                cand_id=cand_id,
                sha=f"{self._candidates:040x}",
                parent_id=parent,
                agent=agent,
                epoch=self._epochs.get(agent, 0),
                session_id=None if agent == "seed" else f"{agent}-e{self._epochs[agent]}",
                judged_key=f"key-{cand_id}",
                lines_added=lines,
                lines_deleted=0,
                files_changed=1,
                summary=f"candidate {cand_id}",
            )
        )

    def _judge(self, cand_id: str, score: float | None, *, valid: bool = True) -> None:
        if valid and score is not None:
            self.scores[cand_id] = score
        self.store.append(
            JudgeFinished(
                cand_id=cand_id,
                valid=valid and score is not None,
                score=score if valid else None,
                feedback="",
                metrics={},
                cases=None,
                duration_s=0.5,
                cache_hit=False,
                error=None if valid else "invalid candidate",
            )
        )

    def _turn(
        self,
        agent: str,
        status: str,
        cand_id: str | None,
        *,
        cost: float,
        lines: int,
        out: int,
        think: int,
        plateau: bool,
    ) -> None:
        self._turns[agent] += 1
        self.store.append(
            TurnFinished(
                agent=agent,
                epoch=self._epochs[agent],
                session_id=f"{agent}-e{self._epochs[agent]}",
                turn_in_session=self._turns[agent],
                status=status,  # type: ignore[arg-type]
                cand_id=cand_id,
                cost_usd=cost,
                input_tokens=1_000,
                output_tokens=out,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                thinking_tokens=think,
                files_read=1,
                edit_calls=0 if cand_id is None else 1,
                lines_changed=lines,
                plateau_claim=plateau,
                duration_s=5.0,
                error="timeout" if status == "error" else None,
            )
        )

    # -- the run, in order ------------------------------------------------------------------

    def start(self, seed_score: float | None = 100.0) -> RunBuilder:
        """RunStarted, the seed candidate and its judge result (None = invalid seed)."""
        self.store.append(
            RunStarted(
                run_id=f"{self.run_dir.parent.name}-{self.run_dir.name}",
                task_name="toy",
                agents=self.agents,
                direction=self.direction,  # type: ignore[arg-type]
                config=self.config,
                lethe_version="0.1.0",
            )
        )
        seed_id = self._next_id()
        self._commit("seed", seed_id, None, lines=0)
        self._judge(seed_id, seed_score, valid=seed_score is not None)
        self.store.append(
            EpochStarted(
                scope="start",
                seeds={agent: SEED_ID for agent in self.agents},
                epochs=dict(self._epochs),
                reason="run start",
            )
        )
        return self

    def candidate(
        self,
        agent: str,
        score: float,
        *,
        cost: float = 0.1,
        lines: int = 20,
        out: int = 400,
        think: int = 200,
        plateau: bool = False,
        judge: bool = True,
    ) -> str:
        """A turn that commits a valid candidate with `score`. Returns its id.

        With `judge=False` the judge result is withheld until `judge_pending(cand_id)`: the real
        harness judges asynchronously, so other agents' turns finish in between.
        """
        cand_id = self._next_id()
        parent = self.best()[0] or SEED_ID
        self._commit(agent, cand_id, parent, lines)
        self._turn(
            agent,
            "candidate",
            cand_id,
            cost=cost,
            lines=lines,
            out=out,
            think=think,
            plateau=plateau,
        )
        if judge:
            self._judge(cand_id, score)
        else:
            self._unjudged[cand_id] = score
        return cand_id

    def judge_pending(self, cand_id: str) -> None:
        """Emit the withheld judge result of a candidate committed with `judge=False`."""
        self._judge(cand_id, self._unjudged.pop(cand_id))

    def invalid(
        self, agent: str, *, cost: float = 0.1, lines: int = 20, out: int = 400, think: int = 200
    ) -> str:
        """A turn that commits a candidate the judge rejects (score None)."""
        cand_id = self._next_id()
        self._commit(agent, cand_id, self.best()[0] or SEED_ID, lines)
        self._turn(
            agent, "candidate", cand_id, cost=cost, lines=lines, out=out, think=think, plateau=False
        )
        self._judge(cand_id, None, valid=False)
        return cand_id

    def empty(
        self,
        agent: str,
        *,
        cost: float = 0.1,
        out: int = 150,
        think: int = 80,
        plateau: bool = False,
    ) -> None:
        """A turn that ends without any edit (status "empty")."""
        self._turn(agent, "empty", None, cost=cost, lines=0, out=out, think=think, plateau=plateau)

    def error(self, agent: str, *, cost: float = 0.0) -> None:
        """A turn that failed (timeout, rate limit...): not a proposal."""
        self._turn(agent, "error", None, cost=cost, lines=0, out=0, think=0, plateau=False)

    def forget(
        self, *, seeds: Mapping[str, str] | None = None, reason: str = "budget_fractions"
    ) -> None:
        """A team forget: ForgetDecided + EpochStarted(team). Default seed: the best candidate."""
        if seeds is None:
            best_id = self.best()[0] or SEED_ID
            seeds = {agent: best_id for agent in self.agents}
        self.store.append(ForgetDecided(scope="team", agents=self.agents, reason=reason))
        for agent in self.agents:
            self._epochs[agent] += 1
            self._turns[agent] = 0
        self.store.append(
            EpochStarted(scope="team", seeds=dict(seeds), epochs=dict(self._epochs), reason=reason)
        )

    def reset_agent(self, agent: str, *, reason: str = "stalled") -> None:
        """A single-agent reset (the stalled trigger): scope "agent"."""
        best_id = self.best()[0] or SEED_ID
        self.store.append(ForgetDecided(scope="agent", agents=(agent,), reason=reason))
        self._epochs[agent] += 1
        self._turns[agent] = 0
        self.store.append(
            EpochStarted(
                scope="agent",
                seeds={agent: best_id},
                epochs={agent: self._epochs[agent]},
                reason=reason,
            )
        )

    def finish(self, reason: str = "budget", status: str = "completed") -> Path:
        """RunFinished with the best candidate; returns the run directory."""
        best_id, best_score = self.best()
        self.store.append(
            RunFinished(reason=reason, best_cand_id=best_id, best_score=best_score, status=status)
        )
        return self.run_dir


@pytest.fixture
def run_builder() -> type[RunBuilder]:
    """The `RunBuilder` class: `run_builder(run_dir, direction="min").start(seed_score=50.0)`."""
    return RunBuilder


@pytest.fixture
def build_analysis_experiment(run_builder: type[RunBuilder]) -> Callable[..., Path]:
    """Factory writing a synthetic experiment of three arms with three replicates each.

    Final best per replicate (the seed is always 100): solo 110/120/100, parallel 120/140/130,
    team-forget 150/160/140. The solo arm has one agent, the others two; team-forget forgets once
    in the middle of every run. Each run has four turns of cost 0.5, 0.25, 0.5 and 0.25: a
    candidate halfway to the final best, an empty turn with a plateau claim, the final candidate,
    and a plain empty turn. `build(dir, direction="min")` flips the direction only.
    """
    finals_by_arm = {
        "solo": [110.0, 120.0, 100.0],
        "parallel": [120.0, 140.0, 130.0],
        "team-forget": [150.0, 160.0, 140.0],
    }

    def build(exp: Path, *, direction: str = "max") -> Path:
        t0 = 1_000.0
        for arm, finals in finals_by_arm.items():
            agents = ("a1",) if arm == "solo" else ("a1", "a2")
            for rep, final in enumerate(finals, 1):
                t0 += 100.0
                b = run_builder(exp / arm / f"rep{rep}", direction=direction, agents=agents, t0=t0)
                b.start(100.0)
                b.candidate("a1", (100.0 + final) / 2, cost=0.5, lines=80, think=300)
                b.empty("a1", cost=0.25, plateau=True)
                if arm == "team-forget":
                    b.forget()
                b.candidate("a1", final, cost=0.5, lines=40, think=150)
                b.empty("a1", cost=0.25)
                b.finish()
        return exp

    return build


@pytest.fixture
def make_summary() -> Callable[..., Any]:
    """Factory for `research.summary.RunSummary` objects with sensible defaults."""
    from research.summary import RoundScores, RunSummary

    def factory(
        arm: str = "solo",
        rep: int = 1,
        *,
        direction: str = "max",
        finished: bool = True,
        finish_reason: str | None = "usd budget exhausted",
        seed_score: float | None = 100.0,
        final_best: float | None = 110.0,
        spend_usd: float = 3.0,
        proposals: int = 10,
        team_forgets: int = 0,
        agent_resets: int = 0,
        rounds: Sequence[tuple[float | None, float | None]] | None = None,
        curve: Sequence[tuple[float, float]] | None = None,
        turns: Sequence[Any] = (),
        budget_usd: float | None = 3.0,
        started_at: float = 0.0,
        run_dir: Path | None = None,
    ) -> Any:
        if rounds is None:
            rounds = [(seed_score, final_best)]
        if curve is None:
            curve = [] if final_best is None else [(0.0, final_best)]
        return RunSummary(
            arm=arm,
            rep=rep,
            run_dir=run_dir or Path(f"/runs/{arm}/rep{rep}"),
            direction=direction,  # type: ignore[arg-type]
            finished=finished,
            finish_reason=finish_reason,
            started_at=started_at,
            seed_score=seed_score,
            final_best=final_best,
            spend_usd=spend_usd,
            proposals=proposals,
            team_forgets=team_forgets,
            agent_resets=agent_resets,
            rounds=tuple(RoundScores(seed, best) for seed, best in rounds),
            curve=tuple(curve),
            turns=tuple(turns),
            budget_usd=budget_usd,
        )

    return factory
