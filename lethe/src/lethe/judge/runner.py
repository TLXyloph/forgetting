"""Runs a task's judge command on a candidate tree, with timeouts, retries, caching, and limits."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import re
import shutil
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from lethe.domain.types import Direction, JudgeResult
from lethe.judge.contract import JudgeInfraError, _output_tail, parse_judge_output
from lethe.workspace.paths import safe_join

_CAND_ID = re.compile(r"[A-Za-z0-9_.-]+")
_MAX_CAPTURE_BYTES = 1024 * 1024  # per stream (stdout, stderr) and for result.json
_READ_CHUNK = 64 * 1024
_REAP_GRACE_S = 5.0  # how long to wait for a killed process to be collected
_INFRA_ERROR = "judge infrastructure failure (exit 75)"
_SEEDS = random.SystemRandom()


@dataclass(frozen=True)
class JudgeSpec:
    """How to score a candidate. `run` and `build` are shell commands run with cwd = judge tree."""

    run: str
    direction: Direction
    timeout_s: float = 600.0
    build: str | None = None
    concurrency: int = 1
    cache: bool = True
    max_feedback_chars: int = 4000
    split: str = "dev"
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CommandResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class _Outcome:
    """What one evaluation produced; `cacheable` is False for failures that are not the
    candidate's fault (or may not repeat): timeouts, infrastructure errors, materialization.
    """

    result: JudgeResult
    cacheable: bool = True


class JudgeRunner:
    """Evaluates candidates. Safe to call `evaluate` concurrently; at most `spec.concurrency` run.

    For each evaluation the runner creates `work_root/<cand_id>-<n>/{tree,out}`, calls
    `materialize(tree)` in a worker thread to write the full judge tree, then runs `spec.build`
    (if set) and `spec.run` via the shell with cwd=tree and env = os.environ + spec.env +
      LETHE_TASK_DIR (task_dir), LETHE_CANDIDATE (tree), LETHE_OUT (out dir),
      LETHE_SPLIT (spec.split), LETHE_SEED (fresh random 32-bit int), LETHE_CAND_ID.
    Each command runs in its own process group; on timeout the whole group is killed and the
    candidate is invalid ("judge timed out after Ns"). A failed build is invalid with the build's
    output as feedback. `JudgeInfraError` (exit 75) is retried once, then reported as invalid
    with `error` set. The tree directory is deleted afterwards; `out/` is kept.

    With `spec.cache`, results are memoized by `judged_key`: a repeat key returns the stored
    result with `cache_hit=True` without running anything.

    Details beyond the above: concurrent evaluations of one key share a single run (the others
    return it as a cache hit); the judge's stdin is /dev/null; stdout/stderr are read
    concurrently and only their last 1 MiB is kept; timeouts, infrastructure failures, and
    materialization failures are reported but never memoized, so a resubmission is retried.
    """

    def __init__(self, spec: JudgeSpec, *, task_dir: Path, work_root: Path) -> None:
        if spec.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {spec.concurrency!r}")
        if not spec.timeout_s > 0:  # written this way so NaN is rejected too
            raise ValueError(f"timeout_s must be > 0, got {spec.timeout_s!r}")
        if not isinstance(spec.run, str) or not spec.run.strip():
            raise ValueError("run must be a non-empty command")
        if spec.max_feedback_chars < 0:
            raise ValueError(f"max_feedback_chars must be >= 0, got {spec.max_feedback_chars!r}")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in spec.env.items()):
            raise ValueError("env must map strings to strings")
        self._spec = spec
        # Absolute (not resolved: symlinks stay) so the judge can use them from any cwd.
        self._task_dir = Path(os.path.abspath(task_dir))
        self._work_root = Path(os.path.abspath(work_root))
        self._cache: dict[str, JudgeResult] = {}
        self._inflight: dict[str, asyncio.Future[JudgeResult | None]] = {}
        self._slots: asyncio.Semaphore | None = None
        self._slots_loop: asyncio.AbstractEventLoop | None = None

    async def evaluate(
        self,
        *,
        cand_id: str,
        judged_key: str,
        materialize: Callable[[Path], None],
    ) -> JudgeResult:
        """Judge one candidate. `cand_id` names its work dirs, `judged_key` identifies its content
        for the cache, and `materialize(tree)` writes the complete judge tree into `tree`.

        A bad candidate or a broken judge is an invalid `JudgeResult`, never an exception; only a
        malformed `cand_id` raises (`ValueError`). Cancelling the call kills the judge's process
        group and re-raises.
        """
        if not isinstance(cand_id, str) or not _CAND_ID.fullmatch(cand_id):
            raise ValueError(f"cand_id must match [A-Za-z0-9_.-]+, got {cand_id!r}")
        if not self._spec.cache:
            return (await self._evaluate_once(cand_id, materialize)).result

        while True:
            cached = self._cache.get(judged_key)
            if cached is not None:
                return _as_cache_hit(cached)
            running = self._inflight.get(judged_key)
            if running is None:
                break
            # Another caller is already evaluating this key: wait for it, but do not let our
            # own cancellation cancel the shared evaluation.
            shared = await asyncio.shield(running)
            if shared is not None:
                return _as_cache_hit(shared)
            # The evaluation we waited for was abandoned (cancelled or crashed): take over.
        return await self._evaluate_and_share(cand_id, judged_key, materialize)

    # --- one evaluation (de-duplicated, then retried once on infrastructure failure) ---

    async def _evaluate_and_share(
        self, cand_id: str, judged_key: str, materialize: Callable[[Path], None]
    ) -> JudgeResult:
        shared: asyncio.Future[JudgeResult | None] = asyncio.get_running_loop().create_future()
        self._inflight[judged_key] = shared
        published: JudgeResult | None = None
        try:
            outcome = await self._evaluate_once(cand_id, materialize)
            published = outcome.result
            if outcome.cacheable:
                self._cache[judged_key] = _detached(outcome.result)
            return outcome.result
        finally:
            # Waiters get `None` if we were cancelled or crashed, and then retry themselves.
            del self._inflight[judged_key]
            shared.set_result(published)

    async def _evaluate_once(self, cand_id: str, materialize: Callable[[Path], None]) -> _Outcome:
        async with self._concurrency_slots():
            started = time.monotonic()
            try:
                outcome = await self._attempt(cand_id, materialize)
            except JudgeInfraError:
                try:
                    outcome = await self._attempt(cand_id, materialize)  # fresh attempt dir
                except JudgeInfraError as exc:
                    failure = JudgeResult(
                        valid=False, score=None, feedback=str(exc), error=_INFRA_ERROR
                    )
                    outcome = _Outcome(failure, cacheable=False)
            elapsed = time.monotonic() - started
        return _Outcome(replace(outcome.result, duration_s=elapsed), outcome.cacheable)

    def _concurrency_slots(self) -> asyncio.Semaphore:
        # Semaphores bind to the loop that first makes them wait, so a runner reused across
        # several `asyncio.run` calls gets a fresh one per loop.
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(self._spec.concurrency)
            self._slots_loop = loop
        return self._slots

    # --- one attempt: materialize, build, run, parse ---

    async def _attempt(self, cand_id: str, materialize: Callable[[Path], None]) -> _Outcome:
        attempt_dir = self._new_attempt_dir(cand_id)
        tree, out = attempt_dir / "tree", attempt_dir / "out"
        tree.mkdir()
        out.mkdir()
        try:
            try:
                await asyncio.to_thread(materialize, tree)
            except Exception as exc:
                message = f"could not materialize candidate: {exc}"
                failure = JudgeResult(valid=False, score=None, feedback=message, error=message)
                return _Outcome(failure, cacheable=False)

            env = self._environment(cand_id, tree, out)
            if self._spec.build and self._spec.build.strip():
                build = await self._run_command(self._spec.build, cwd=tree, env=env)
                if build.timed_out:
                    return self._timeout_outcome()
                if build.exit_code != 0:
                    failure = JudgeResult(
                        valid=False,
                        score=None,
                        feedback=_output_tail(
                            build.stdout, build.stderr, self._spec.max_feedback_chars
                        ),
                        error=f"build failed (exit {build.exit_code})",
                    )
                    return _Outcome(failure)

            run = await self._run_command(self._spec.run, cwd=tree, env=env)
            if run.timed_out:
                return self._timeout_outcome()
            try:
                result_json = _read_result_json(out)
            except OSError as exc:
                unreadable = JudgeResult(
                    valid=False, score=None, feedback=f"could not read result.json: {exc}"
                )
                return _Outcome(unreadable)
            return _Outcome(
                parse_judge_output(
                    stdout=run.stdout,
                    stderr=run.stderr,
                    exit_code=run.exit_code,
                    result_json=result_json,
                    max_feedback_chars=self._spec.max_feedback_chars,
                )
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, tree, ignore_errors=True)

    def _new_attempt_dir(self, cand_id: str) -> Path:
        """Create `work_root/<cand_id>-<n>` for the first unused n (mkdir is atomic)."""
        self._work_root.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            attempt += 1
            directory = safe_join(self._work_root, f"{cand_id}-{attempt}")
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            return directory

    def _environment(self, cand_id: str, tree: Path, out: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self._spec.env)
        env.update(
            LETHE_TASK_DIR=str(self._task_dir),
            LETHE_CANDIDATE=str(tree),
            LETHE_OUT=str(out),
            LETHE_SPLIT=self._spec.split,
            LETHE_SEED=str(_SEEDS.randrange(2**32)),
            LETHE_CAND_ID=cand_id,
        )
        return env

    def _timeout_outcome(self) -> _Outcome:
        message = f"judge timed out after {self._spec.timeout_s:g}s"
        failure = JudgeResult(valid=False, score=None, feedback=message, error=message)
        return _Outcome(failure, cacheable=False)

    # --- running one shell command in its own process group ---

    async def _run_command(self, command: str, *, cwd: Path, env: dict[str, str]) -> _CommandResult:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        # The pipes are drained by tasks that outlive a timeout, so a flooding judge can never
        # leave the process unable to exit because its pipe buffers are full.
        readers = [
            asyncio.ensure_future(_read_tail(process.stdout)),
            asyncio.ensure_future(_read_tail(process.stderr)),
        ]
        try:
            try:
                exit_code = await asyncio.wait_for(process.wait(), self._spec.timeout_s)
            except TimeoutError:
                _kill_group(process)
                await _reap(process)
                return _CommandResult(timed_out=True)
            stdout, stderr = await asyncio.gather(*readers)
            return _CommandResult(exit_code, _decode(stdout), _decode(stderr))
        except asyncio.CancelledError:
            _kill_group(process)
            await _reap(process)
            raise
        finally:
            for reader in readers:
                reader.cancel()


def _detached(result: JudgeResult) -> JudgeResult:
    """A copy of `result` that shares no mutable mappings with it."""
    cases = None if result.cases is None else dict(result.cases)
    return replace(result, metrics=dict(result.metrics), cases=cases)


def _as_cache_hit(result: JudgeResult) -> JudgeResult:
    """The stored result as a cache hit: it ran nothing, so it took no time."""
    return replace(_detached(result), cache_hit=True, duration_s=0.0)


def _read_result_json(out: Path) -> str | None:
    """The text of `out/result.json` (first 1 MiB), or None when the judge wrote none."""
    path = out / "result.json"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return _decode(handle.read(_MAX_CAPTURE_BYTES))


async def _read_tail(stream: asyncio.StreamReader) -> bytes:
    """Read `stream` to EOF, keeping only its last `_MAX_CAPTURE_BYTES` bytes."""
    kept = bytearray()
    while chunk := await stream.read(_READ_CHUNK):
        kept += chunk
        if len(kept) > 2 * _MAX_CAPTURE_BYTES:
            del kept[: len(kept) - _MAX_CAPTURE_BYTES]
    return bytes(kept[-_MAX_CAPTURE_BYTES:])


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the process's whole group (the leader may be gone while children live on)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Wait for a killed process to be collected, but never hang on a stray pipe holder."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), _REAP_GRACE_S)
