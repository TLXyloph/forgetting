"""The control loop: N agents in parallel, a judge, and forgetting.

Plain code by design (never an LLM): an LLM orchestrator would accumulate exactly the history
that forgetting is meant to clear. Every state change goes through `emit`, which appends to the
event log and folds into `RunState`, so a live run and a replay see the same state.

Every way out of `run` logs `RunFinished` with a status: "completed" (a budget ran out),
"interrupted" (cancelled, e.g. Ctrl-C), "crashed" (an unexpected exception, re-raised once it is
logged) or "stopped" (every agent gave up after repeated errors, or safe mode refused API-key
billing, which stops all agents at once). No agent is still running when it is written.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
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
    RunStatus,
    TurnFinished,
)
from lethe.domain.state import SEED_ID, RunState
from lethe.domain.types import TurnResult, Usage
from lethe.policy.reducer import TeamPolicy, TurnPlan, plan_turn
from lethe.policy.seeding import seed_policy_from_name
from lethe.policy.triggers import ForgetDecision, trigger_from_dict
from lethe.runtime.barrier import ForgetBarrier
from lethe.runtime.budget import Pacer, exhausted, spend_fraction
from lethe.telemetry.signals import is_plateau_claim

EventHook = Callable[[Envelope, RunState], None]
TurnOutcome = Literal["ok", "stop", "fatal"]
BudgetGate = Literal["go", "wait", "stop"]

_LOG = logging.getLogger(__name__)
_BILLING = "safe mode refused API-key billing; set agent.allow_api_billing = true to allow it"


class _RunStopped(Exception):
    """Raised in an agent loop to stop every agent at once (a fatal turn)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Orchestrator:
    MAX_CONSECUTIVE_ERRORS = 5
    SUMMARY_CHARS = 2000
    BUDGET_POLL_S = 0.2  # how often an agent re-checks a budget only in-flight turns hold

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
        self.pacer = pacer or Pacer(
            pause_at=config.budget.pause_at_utilization,
            weekly_pause_at=config.budget.pause_at_weekly_utilization,
        )
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
        self._fatal_reason: str | None = None
        self._t0: float | None = None

    # ------------------------------------------------------------------ public
    def emit(self, event: Event) -> Envelope:
        envelope = self.store.append(event)
        self.state.apply(envelope)
        if self._on_event is not None:
            try:
                self._on_event(envelope, self.state)
            except Exception as exc:  # e.g. BrokenPipeError from `lethe run | head`
                _LOG.warning("progress output failed (%r); the run continues without it", exc)
                self._on_event = None
        return envelope

    async def run(self) -> RunState:
        cfg = self.config
        self.emit(
            RunStarted(
                self.run_id,
                cfg.task.name,
                self.agents,
                cfg.judge.direction,
                self._logged_config(),
                __version__,
            )
        )
        try:
            await self._commit_seed()
            self._t0 = self._monotonic()
            await self._start_epochs("start", dict.fromkeys(self.agents, SEED_ID), "run start")
            await self._run_agents()
        except asyncio.CancelledError:
            self._finish("interrupted", "interrupted")
            raise
        except _RunStopped as stop:
            self._finish(stop.reason, "stopped")
            return self.state
        except Exception as exc:
            self._finish(f"crashed: {type(exc).__name__}: {exc}", "crashed")
            raise
        if self._stop_reason is None and self._stopped_by_errors == set(self.agents):
            self._finish("all agents stopped after repeated errors", "stopped")
        else:
            self._finish(self._stop_reason or "finished")
        return self.state

    # ----------------------------------------------------------------- phases
    async def _run_agents(self) -> None:
        """Run every agent loop. The first failure (or a fatal stop) cancels the others, and all
        of them are gone before this raises, so nothing is logged after `RunFinished`."""
        tasks = [asyncio.ensure_future(self._agent_loop(agent)) for agent in self.agents]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        await self._judge(SEED_ID)

    async def _start_epochs(self, scope: str, seeds: dict[str, str], reason: str) -> None:
        # A clean start (carry_scratch = false) rebuilds each workspace from the original seed
        # overlaid with its starting candidate's judged files: scratch files and edits to the
        # seed's other files stay behind.
        base = None if self.config.forget.carry_scratch else self.state.candidates[SEED_ID].sha
        epochs = {agent: self.state.epoch(agent) + 1 for agent in seeds}
        for agent, cand_id in seeds.items():
            sha = self.state.candidates[cand_id].sha
            self._workspaces[agent] = await asyncio.to_thread(
                self.materializer.prepare_epoch, agent, epochs[agent], sha, base_sha=base
            )
        self.emit(EpochStarted(scope, dict(seeds), epochs, reason))  # type: ignore[arg-type]

    async def _agent_loop(self, agent: str) -> None:
        while True:
            await self._barrier.arrive(agent)
            await self.pacer.wait()
            if self._barrier.pending is not None:
                continue  # a team forget became pending while we waited: join it first
            gate = self._budget_gate()
            if gate == "stop":
                await self._leave(agent)
                return
            if gate == "wait":
                await asyncio.sleep(self.BUDGET_POLL_S)
                continue
            # In flight = started but not yet counted as a proposal; judging happens after the
            # turn is counted, so it must not hold the slot (or the turn is counted twice). A
            # pending reset is applied inside the slot, right before the turn it is for.
            self._in_flight += 1
            try:
                if not await self._apply_reset(agent):
                    continue
                outcome, to_judge = await self._turn(agent)
            finally:
                self._in_flight -= 1
            if to_judge is not None:
                await self._judge(to_judge)
            if outcome == "fatal":
                raise _RunStopped(self._fatal_reason or _BILLING)
            if outcome == "stop":
                await self._leave(agent)
                return
            self._check_triggers(agent)

    def _budget_gate(self) -> BudgetGate:
        """May a turn start now? The run stops only once counted turns have used the budget up;
        while in-flight turns merely reserve the rest, "wait": they may still fail and give
        their share back."""
        if self._stop_reason is None:
            self._stop_reason = self._exhausted(in_flight=0)
        if self._stop_reason is not None:
            return "stop"
        if self._exhausted(in_flight=self._in_flight) is not None:
            return "wait"
        return "go"

    def _stopping(self) -> bool:
        return self._stop_reason is not None or self._exhausted(in_flight=0) is not None

    async def _apply_reset(self, agent: str) -> bool:
        """Start the agent's pending reset epoch, if any. False when the run is stopping: the
        reset is dropped, since an epoch without a turn would only be an empty round."""
        decision = self._reset_pending.pop(agent, None)
        if decision is None:
            return True
        if self._stopping():
            return False
        best = self.state.best()
        seeds = {agent: best.cand_id if best else SEED_ID}
        await self._start_epochs("agent", seeds, decision.reason)
        return True

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
        try:
            await asyncio.to_thread(self.materializer.render, workspace, plan.bundle)
        except Exception as exc:  # the agent never ran: an error turn that cost nothing
            failed = TurnResult(plan.session.session_id, False, "", 0.0, Usage(), (), 0.0)
            message = f"could not render the workspace: {type(exc).__name__}: {exc}"
            return self._error_turn(plan, replace(failed, error=message), "harness"), None
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
        if result.error_kind == "billing":
            self._emit_turn(
                plan, result, status="error", cand_id=None, lines=0, error_kind="billing"
            )
            self._fatal_reason = result.error or _BILLING
            return "fatal", None
        if not result.ok:
            return self._error_turn(plan, result, result.error_kind), None
        self.pacer.observe(result.rate_limit)
        current = plan.current
        try:
            snap = await asyncio.to_thread(
                self.archive.snapshot,
                workspace,
                parent_sha=current.sha,
                message=f"{agent} epoch {plan.epoch} turn {plan.turn_in_session}",
                index_name=agent,
            )
        except Exception as exc:  # the turn ran (and cost): an error turn that keeps its cost
            message = f"could not snapshot the workspace: {type(exc).__name__}: {exc}"
            return self._error_turn(plan, replace(result, ok=False, error=message), "harness"), None
        self._errors[agent] = 0
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

    def _error_turn(self, plan: TurnPlan, result: TurnResult, kind: str | None) -> TurnOutcome:
        """Log a failed turn; "stop" once the agent failed MAX_CONSECUTIVE_ERRORS times in a row."""
        self._errors[plan.agent] += 1
        self._emit_turn(plan, result, status="error", cand_id=None, lines=0, error_kind=kind)
        if self._errors[plan.agent] >= self.MAX_CONSECUTIVE_ERRORS:
            self._stopped_by_errors.add(plan.agent)
            return "stop"
        return "ok"

    def _emit_turn(
        self,
        plan: TurnPlan,
        result: TurnResult,
        *,
        status: str,
        cand_id: str | None,
        lines: int,
        error_kind: str | None = None,
    ) -> None:
        usage, meter = result.usage, result.rate_limit
        failed = status == "error"
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
                error=result.error if failed else None,
                five_hour_utilization=meter.five_hour_utilization if meter else None,
                seven_day_utilization=meter.seven_day_utilization if meter else None,
                error_kind=error_kind if failed else None,
                tool_calls=len(result.tool_calls),
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
        for agent in agents:
            self._reset_pending.pop(agent, None)  # the team forget supersedes them
        if self._stopping():
            return  # the budget is spent: a new round would have no turns (all are released)
        seeds = self._seeding.seeds(self.state, agents)
        await self._start_epochs("team", seeds, decision.reason)

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

    # ---------------------------------------------------------------- logging
    def _logged_config(self) -> dict[str, Any]:
        """The resolved config for RunStarted: judge.env values hidden (they may be tokens), and
        the task text itself even when it came from description_file (so a replay has it)."""
        config = copy.deepcopy(self.config.resolved)
        judge = config.get("judge")
        if isinstance(judge, dict) and isinstance(judge.get("env"), dict):
            judge["env"] = dict.fromkeys(judge["env"], "***")
        task = config.get("task")
        if isinstance(task, dict):
            task["description"] = self.config.task.description
        return config

    def _finish(self, reason: str, status: RunStatus = "completed") -> None:
        if self.state.finished is not None:
            return
        best = self.state.best()
        self.emit(
            RunFinished(
                reason, best.cand_id if best else None, best.score if best else None, status
            )
        )
