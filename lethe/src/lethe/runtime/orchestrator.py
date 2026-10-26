"""The control loop: N agents in parallel, a judge, and forgetting.

Plain code by design (never an LLM): an LLM orchestrator would accumulate exactly the history
that forgetting is meant to clear. Every state change goes through `emit`, which appends to the
event log and folds into `RunState`, so a live run and a replay see the same state.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

from lethe import __version__
from lethe.agents.base import AgentBackend
from lethe.config import RunConfig
from lethe.domain.events import (
    CandidateCommitted,
    Envelope,
    EpochStarted,
    Event,
    ForgetDecided,
    JudgeFinished,
    RateLimited,
    RunFinished,
    RunStarted,
    TurnFinished,
)
from lethe.domain.state import SEED_ID, RunState
from lethe.domain.types import TurnResult
from lethe.policy.reducer import TeamPolicy, TurnPlan, plan_turn
from lethe.policy.seeding import seed_policy_from_name
from lethe.policy.triggers import ForgetDecision, trigger_from_dict
from lethe.runtime.barrier import ForgetBarrier
from lethe.runtime.budget import Pacer, exhausted, spend_fraction
from lethe.telemetry.signals import is_plateau_claim

EventHook = Callable[[Envelope, RunState], None]
TurnOutcome = Literal["ok", "stop"]


class Orchestrator:
    MAX_CONSECUTIVE_ERRORS = 5
    SUMMARY_CHARS = 2000

    def __init__(
        self,
        config: RunConfig,
        *,
        run_id: str,
        run_dir: Path,
        archive: Any,
        materializer: Any,
        judge: Any,
        backend: AgentBackend,
        store: Any,
        monotonic: Callable[[], float] = time.monotonic,
        new_session_id: Callable[[], str] | None = None,
        pacer: Pacer | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.config = config
        self.state = RunState()
        self.run_id, self.run_dir = run_id, Path(run_dir)
        self.archive, self.materializer, self.judge = archive, materializer, judge
        self.backend, self.store = backend, store
        self.pacer = pacer or Pacer(pause_at=config.budget.pause_at_utilization)
        self._monotonic = monotonic
        self._new_session_id = new_session_id or (lambda: str(uuid.uuid4()))
        self._on_event = on_event
        self.agents = config.team.agent_ids
        self.team = TeamPolicy(self.agents, config.team.peers, config.team.session)  # type: ignore[arg-type]
        triggers = [trigger_from_dict(spec) for spec in config.forget.when]
        self._team_triggers = [t for t in triggers if t.scope == "team"]
        self._agent_triggers = [t for t in triggers if t.scope == "agent"]
        self._seeding = seed_policy_from_name(config.forget.seed)
        self._barrier = ForgetBarrier(self._commit_team_forget, active=self.agents)
        self._active = set(self.agents)
        self._workspaces: dict[str, Path] = {}
        self._reset_pending: dict[str, ForgetDecision] = {}
        self._errors = dict.fromkeys(self.agents, 0)
        self._stopped_by_errors: set[str] = set()
        self._in_flight = 0
        self._stop_reason: str | None = None
        self._t0: float | None = None
        self._seed_paths: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ public
    def emit(self, event: Event) -> Envelope:
        envelope = self.store.append(event)
        self.state.apply(envelope)
        if self._on_event is not None:
            self._on_event(envelope, self.state)
        return envelope

    async def run(self) -> RunState:
        cfg = self.config
        self.emit(
            RunStarted(
                self.run_id,
                cfg.task.name,
                self.agents,
                cfg.judge.direction,
                cfg.resolved,
                __version__,
            )
        )
        try:
            await self._commit_seed()
            self._t0 = self._monotonic()
            await self._start_epochs("start", dict.fromkeys(self.agents, SEED_ID), "run start")
            await asyncio.gather(*(self._agent_loop(agent) for agent in self.agents))
        except asyncio.CancelledError:
            self._finish("interrupted")
            raise
        if self._stop_reason is None and self._stopped_by_errors == set(self.agents):
            self._stop_reason = "all agents stopped after repeated errors"
        self._finish(self._stop_reason or "finished")
        return self.state

    # ----------------------------------------------------------------- phases
    async def _commit_seed(self) -> None:
        seed_dir = self.config.task.seed_dir
        snap = await asyncio.to_thread(
            self.archive.snapshot, seed_dir, parent_sha=None, message="task seed", index_name="seed"
        )
        self.emit(
            CandidateCommitted(
                SEED_ID, snap.sha, None, "seed", 0, None, snap.judged_key, 0, 0, 0, "task seed"
            )
        )
        self._seed_paths = frozenset(await asyncio.to_thread(self.archive.files, snap.sha))
        await self._judge(SEED_ID)

    async def _start_epochs(self, scope: str, seeds: dict[str, str], reason: str) -> None:
        keep = None if self.config.forget.carry_scratch else self._clean_keep
        epochs = {agent: self.state.epoch(agent) + 1 for agent in seeds}
        for agent, cand_id in seeds.items():
            sha = self.state.candidates[cand_id].sha
            self._workspaces[agent] = await asyncio.to_thread(
                self.materializer.prepare_epoch, agent, epochs[agent], sha, keep=keep
            )
        self.emit(EpochStarted(scope, dict(seeds), epochs, reason))  # type: ignore[arg-type]

    def _clean_keep(self, path: str) -> bool:
        return self.config.task.judged(path) or path in self._seed_paths

    async def _agent_loop(self, agent: str) -> None:
        while True:
            await self._barrier.arrive(agent)
            if agent in self._reset_pending:
                decision = self._reset_pending.pop(agent)
                best = self.state.best()
                await self._start_epochs(
                    "agent", {agent: best.cand_id if best else SEED_ID}, decision.reason
                )
            await self.pacer.wait()
            reason = self._stop_reason or self._exhausted(in_flight=self._in_flight)
            if reason is not None:
                self._stop_reason = reason
                await self._leave(agent)
                return
            # In flight = started but not yet counted as a proposal; judging happens after the
            # turn is counted, so it must not hold the slot (or the turn is counted twice).
            self._in_flight += 1
            try:
                outcome, to_judge = await self._turn(agent)
            finally:
                self._in_flight -= 1
            if to_judge is not None:
                await self._judge(to_judge)
            if outcome == "stop":
                await self._leave(agent)
                return
            self._check_triggers(agent)

    async def _leave(self, agent: str) -> None:
        self._active.discard(agent)
        await self._barrier.depart(agent)

    async def _turn(self, agent: str) -> tuple[TurnOutcome, str | None]:
        """Run one agent turn; returns the outcome and the new candidate to judge, if any."""
        cfg = self.config
        plan = plan_turn(
            self.state,
            agent,
            team=self.team,
            task_description=cfg.task.description,
            archive=self.archive,
            judged=cfg.task.judged,
            new_session_id=self._new_session_id,
        )
        workspace = self._workspaces[agent]
        await asyncio.to_thread(self.materializer.render, workspace, plan.bundle)
        transcript = (
            self.run_dir
            / "transcripts"
            / agent
            / f"e{plan.epoch}"
            / f"{plan.session.session_id}-t{plan.turn_in_session}.jsonl"
        )
        result = await self.backend.run_turn(
            agent=agent,
            workspace=workspace,
            prompt=plan.prompt,
            session=plan.session,
            transcript_path=transcript,
        )
        if result.error_kind == "rate_limit":
            resume_at = self.pacer.note_rate_limited(result.rate_limit)
            resets = result.rate_limit.resets_at if result.rate_limit else None
            self.emit(RateLimited(agent, resets, f"rate limited; agents resume at {resume_at:.0f}"))
            return "ok", None
        if not result.ok:
            self._errors[agent] += 1
            self._emit_turn(plan, result, status="error", cand_id=None, lines=0)
            if self._errors[agent] >= self.MAX_CONSECUTIVE_ERRORS:
                self._stopped_by_errors.add(agent)
                return "stop", None
            return "ok", None
        self._errors[agent] = 0
        self.pacer.observe(result.rate_limit)
        current = plan.current
        snap = await asyncio.to_thread(
            self.archive.snapshot,
            workspace,
            parent_sha=current.sha,
            message=f"{agent} epoch {plan.epoch} turn {plan.turn_in_session}",
            index_name=agent,
        )
        cand_id: str | None = None
        if snap.changed_judged:
            cand_id = self.state.next_cand_id()
            self.emit(
                CandidateCommitted(
                    cand_id,
                    snap.sha,
                    current.cand_id,
                    agent,
                    plan.epoch,
                    result.session_id,
                    snap.judged_key,
                    snap.lines_added,
                    snap.lines_deleted,
                    snap.files_changed,
                    result.final_text[: self.SUMMARY_CHARS],
                )
            )
        lines = snap.lines_added + snap.lines_deleted if cand_id else 0
        self._emit_turn(
            plan, result, status="candidate" if cand_id else "empty", cand_id=cand_id, lines=lines
        )
        return "ok", cand_id

    def _emit_turn(
        self, plan: TurnPlan, result: TurnResult, *, status: str, cand_id: str | None, lines: int
    ) -> None:
        usage, meter = result.usage, result.rate_limit
        self.emit(
            TurnFinished(
                agent=plan.agent,
                epoch=plan.epoch,
                session_id=result.session_id or plan.session.session_id,
                turn_in_session=plan.turn_in_session,
                status=status,  # type: ignore[arg-type]
                cand_id=cand_id,
                cost_usd=result.cost_usd,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                thinking_tokens=usage.thinking_tokens,
                files_read=result.files_read,
                edit_calls=result.edit_calls,
                lines_changed=lines,
                plateau_claim=is_plateau_claim(result.final_text),
                duration_s=result.duration_s,
                error=result.error if status == "error" else None,
                five_hour_utilization=meter.five_hour_utilization if meter else None,
                seven_day_utilization=meter.seven_day_utilization if meter else None,
            )
        )

    async def _judge(self, cand_id: str) -> None:
        cand = self.state.candidates[cand_id]
        seed_sha = self.state.candidates[SEED_ID].sha
        result = await self.judge.evaluate(
            cand_id=cand_id,
            judged_key=cand.judged_key,
            materialize=partial(self.archive.checkout_overlay, seed_sha, cand.sha),
        )
        self.emit(
            JudgeFinished(
                cand_id,
                result.valid,
                result.score,
                result.feedback,
                dict(result.metrics),
                dict(result.cases) if result.cases is not None else None,
                result.duration_s,
                result.cache_hit,
                result.error,
            )
        )

    # --------------------------------------------------------------- forgetting
    def _check_triggers(self, agent: str) -> None:
        if self._exhausted(in_flight=0) is not None:
            return
        fraction = self._spend_fraction()
        if self._barrier.pending is None:
            for trigger in self._team_triggers:
                decision = trigger.check(self.state, spend_fraction=fraction)  # type: ignore[union-attr]
                if decision is not None:
                    self.emit(ForgetDecided("team", decision.agents, decision.reason))
                    self._barrier.request(decision)
                    return
        if self._barrier.pending is None and agent not in self._reset_pending:
            for trigger in self._agent_triggers:
                decision = trigger.check_agent(self.state, agent, spend_fraction=fraction)  # type: ignore[union-attr]
                if decision is not None:
                    self.emit(ForgetDecided("agent", decision.agents, decision.reason))
                    self._reset_pending[agent] = decision
                    return

    async def _commit_team_forget(self, decision: ForgetDecision) -> None:
        agents = [a for a in self.agents if a in self._active]
        seeds = self._seeding.seeds(self.state, agents)
        await self._start_epochs("team", seeds, decision.reason)
        for agent in agents:
            self._reset_pending.pop(agent, None)

    # ------------------------------------------------------------------ budget
    def _elapsed(self) -> float:
        return 0.0 if self._t0 is None else self._monotonic() - self._t0

    def _spend_fraction(self) -> float:
        return spend_fraction(
            self.config.budget,
            spent_usd=self.state.spent_usd,
            proposals=self.state.proposals,
            elapsed_s=self._elapsed(),
        )

    def _exhausted(self, *, in_flight: int) -> str | None:
        return exhausted(
            self.config.budget,
            spent_usd=self.state.spent_usd,
            proposals=self.state.proposals + in_flight,
            elapsed_s=self._elapsed(),
        )

    def _finish(self, reason: str) -> None:
        if self.state.finished is not None:
            return
        best = self.state.best()
        self.emit(RunFinished(reason, best.cand_id if best else None, best.score if best else None))
