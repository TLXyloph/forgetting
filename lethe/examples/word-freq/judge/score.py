"""Hidden judge for the word-freq example (a performance task). Python 3.11+, stdlib only.

It never runs inside an agent workspace; lethe reaches it through $LETHE_TASK_DIR and runs it with
cwd = the judge tree (the seed with the candidate's editable files overlaid).

  1. Correctness gate: about sixty hidden, generated cases of many kinds are run in one subprocess.
     Any mismatch makes the candidate invalid; the feedback names the failing case *kinds* only,
     never their contents.
  2. Timing: `top_words(corpus, 50)` in five fresh subprocesses, each on its own corpus of about
     300k words drawn afresh for every evaluation from LETHE_SEED, so nothing can be precomputed
     or carried from one run to the next. The clock runs here, never in the candidate's process:
     the child imports the candidate and says it is ready, the judge starts the clock and sends
     the corpus on stdin, and the clock stops when the child's result arrives. Every result is
     checked against the reference.
         score = median wall-clock milliseconds of those five calls     (minimize)

Candidate code runs as `python -s -B -c <runner>`: the runner is inline (the candidate never
learns where this file is), its data comes on stdin (never as a path), and its environment holds
no LETHE_* variables and no credentials. The data (reference, corpora, gate cases) lives in the
sibling module wordfreq_data.py. Prints exactly one JSON line; exit 75 means the judge itself
broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import json
import os
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zlib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import IO, Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

TIMING_K = 50
TIMING_RUNS = 5
GATE_TIMEOUT_S = 60.0
RUN_TIMEOUT_S = 30.0  # per timed run: import, the call, and the replies
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_REPLY_BYTES = 4 * 1024 * 1024  # a timed reply is a few kB: a longer line is a broken reply
# All the candidate inherits from the judge's environment (plus LC_*): enough to run Python.
INHERITED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TEMP", "TMP")
INHERITED_ENV += ("TZ", "SYSTEMROOT")

# The runner the candidate's process executes (`python -s -B -c`). It imports the candidate from
# the judge tree, and replies with one JSON object per line on its real stdout; the candidate's
# own prints go to stderr.
PRELUDE = """
import json, os, sys, traceback
root = os.getcwd()
sys.path.insert(0, root)
reply = sys.stdout
sys.stdout = sys.stderr

def where(exc):
    frames = [f"{os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}"
              for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith(root)]
    return f" at {frames[-1]}" if frames else ""

def send(line):
    reply.write(line + "\\n")
    reply.flush()

try:
    from wordfreq import top_words
except Exception as exc:
    send(json.dumps({"fatal": f"{type(exc).__name__}{where(exc)}"}))
    sys.exit(1)
"""
GATE_RUNNER = (
    PRELUDE
    + """
for i, (text, k) in enumerate(json.loads(sys.stdin.read())):
    try:
        send(json.dumps({"i": i, "result": top_words(text, k)}))
    except Exception as exc:
        send(json.dumps({"i": i, "error": f"raised {type(exc).__name__}{where(exc)}"}))
"""
)
TIME_RUNNER = (
    PRELUDE
    + """
import gc
k = int(sys.argv[1])
gc.collect()
send(json.dumps({"ready": True}))
text = sys.stdin.buffer.read().decode("utf-8")
try:
    send(json.dumps({"result": top_words(text, k)}))
except Exception as exc:
    send(json.dumps({"error": f"raised {type(exc).__name__}{where(exc)}"}))
"""
)


def _sibling(name: str) -> ModuleType:
    """Load a module that lives next to this file (without touching sys.path)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_data = _sibling("wordfreq_data")
WORD_RE, build_corpus, gate_cases, reference = (
    _data.WORD_RE,
    _data.build_corpus,
    _data.gate_cases,
    _data.reference,
)


# ---------------------------------------------------------------------------------------------
# running the candidate
# ---------------------------------------------------------------------------------------------


def _limits(cpu_seconds: int) -> None:  # runs in the child between fork and exec
    """Bound what a runaway candidate can do: output size, core dumps, and CPU time.

    The CPU limit is only a backstop (e.g. if lethe kills this judge before the judge kills its
    child); the wall-clock timeout is what normally ends a slow candidate.
    """
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def candidate_env() -> dict[str, str]:
    """Just enough to run Python: no LETHE_* variables (no seed, no paths), no credentials."""
    env = {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}
    env["PYTHONHASHSEED"] = "0"  # identical candidates must score identically
    return env


def spawn(runner: str, args: list[str], cwd: Path, **pipes: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-s", "-B", "-c", runner, *args],
        cwd=cwd,
        env=candidate_env(),
        start_new_session=True,
        preexec_fn=functools.partial(_limits, max(60, int(GATE_TIMEOUT_S * 8))),
        **pipes,
    )


def tail_of(stream: IO[bytes], limit: int = 600) -> str:
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - limit))
    return stream.read().decode("utf-8", errors="replace").strip()


def describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


def parse_lines(data: bytes) -> list[dict[str, Any]]:
    out = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a line cut off by the kill
        if isinstance(rec, dict):
            out.append(rec)
    return out


def normalize(result: object) -> list[tuple[str, int]] | None:
    """The result as (str, int) pairs, or None when it is not a list of such pairs."""
    if not isinstance(result, list):
        return None
    out: list[tuple[str, int]] = []
    for item in result:
        if not isinstance(item, list) or len(item) != 2:
            return None
        word, count = item
        if not isinstance(word, str) or type(count) is not int:
            return None
        out.append((word, count))
    return out


class Replies:
    """The child's reply lines, read from its stdout pipe against a deadline."""

    def __init__(self, pipe: IO[bytes]) -> None:
        self._fd = pipe.fileno()
        self._buffer = b""
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._fd, selectors.EVENT_READ)

    def next(self, deadline: float) -> tuple[str, dict[str, Any]]:
        """("ok", record), or ("timeout" | "closed" | "garbled", {})."""
        while b"\n" not in self._buffer:
            if len(self._buffer) > MAX_REPLY_BYTES:
                return "garbled", {}
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                return "timeout", {}
            chunk = os.read(self._fd, 1 << 16)
            if not chunk:
                return "closed", {}
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            record = json.loads(line)
        except ValueError:
            return "garbled", {}
        return ("ok", record) if isinstance(record, dict) else ("garbled", {})

    def close(self) -> None:
        self._selector.close()


def _send(stdin: IO[bytes], data: bytes) -> None:
    with contextlib.suppress(OSError):  # the child may die or close stdin early
        stdin.write(data)
        stdin.close()


# ---------------------------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------------------------


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def run_gate(cwd: Path) -> dict[str, Any] | None:
    """None when every hidden case passes; otherwise an invalid result."""
    cases = gate_cases()
    payload = json.dumps([[text, k] for _kind, text, k in cases]).encode("utf-8")
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as out:
        stdin.write(payload)
        stdin.seek(0)
        with tempfile.TemporaryFile() as err:
            proc = spawn(GATE_RUNNER, [], cwd, stdin=stdin, stdout=out, stderr=err)
            timed_out = False
            try:
                proc.wait(timeout=GATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _kill_group(proc.pid)
            code = proc.wait()
            stderr_tail = tail_of(err)
        out.seek(0)
        replies = parse_lines(out.read(MAX_FILE_BYTES))
    fatal = next((r for r in replies if "fatal" in r), None)
    if fatal is not None:
        return invalid(f"The candidate could not be loaded: {fatal['fatal']}")
    records = {r["i"]: r for r in replies if isinstance(r.get("i"), int)}
    if timed_out or len(records) < len(cases):
        stuck = next((cases[i][0] for i in range(len(cases)) if i not in records), "?")
        how = (
            f"timed out after {GATE_TIMEOUT_S:g} s"
            if timed_out
            else describe_exit(code) + (f" ({stderr_tail})" if stderr_tail else "")
        )
        return invalid(
            f"The correctness cases did not finish: the candidate process {how} on a case of kind "
            f"'{stuck}' ({len(records)} of {len(cases)} cases done)."
        )
    failures: Counter[str] = Counter()
    notes: set[str] = set()
    for i, (kind, text, k) in enumerate(cases):
        rec = records[i]
        if "error" in rec:
            failures[kind] += 1
            notes.add(f"{kind}: {rec['error']}")
            continue
        got = normalize(rec.get("result"))
        if got is None:
            failures[kind] += 1
            notes.add(f"{kind}: the return value is not a list of (str, int) pairs")
        elif got != reference(text, k):
            failures[kind] += 1
    if not failures:
        return None
    kinds = ", ".join(f"{kind} x{n}" for kind, n in sorted(failures.items()))
    detail = (" " + "; ".join(sorted(notes)[:3]) + ".") if notes else ""
    return invalid(
        f"Incorrect output on {sum(failures.values())} of {len(cases)} hidden correctness cases "
        f"(kinds: {kinds}).{detail} Timing is only measured once all cases pass.",
        failed_cases=sum(failures.values()),
    )


def time_one(cwd: Path, corpus: bytes, run: int) -> tuple[float | None, dict[str, Any] | str]:
    """One timed call in a fresh child: (seconds, its result record), or (None, why it failed)."""
    with tempfile.TemporaryFile() as err:
        proc = spawn(
            TIME_RUNNER,
            [str(TIMING_K)],
            cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
        )
        assert proc.stdin is not None and proc.stdout is not None
        replies, sending = Replies(proc.stdout), False
        try:
            deadline = time.monotonic() + RUN_TIMEOUT_S
            state, ready = replies.next(deadline)
            if state == "ok" and ready.get("ready") is True:
                started = time.perf_counter()
                sending = True
                threading.Thread(target=_send, args=(proc.stdin, corpus), daemon=True).start()
                state, record = replies.next(deadline)
                seconds = time.perf_counter() - started
                if state == "ok":
                    return seconds, record
            elif state == "ok" and "fatal" in ready:
                return None, f"Timing run {run} failed: {ready['fatal']}"
        finally:
            _kill_group(proc.pid)
            code = proc.wait()
            replies.close()
            proc.stdout.close()
            if not sending:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
        if state == "timeout":
            return None, f"Timing run {run} exceeded {RUN_TIMEOUT_S:g} s on the benchmark corpus."
        detail = tail_of(err)
        return None, (
            f"Timing run {run}: the candidate process {describe_exit(code)} without a result."
            + (f" stderr tail: {detail}" if detail else "")
        )


def timing_corpora() -> list[str]:
    """The corpora of this evaluation: drawn afresh from LETHE_SEED, one per timed run."""
    try:
        base = int(os.environ.get("LETHE_SEED", "0"))
    except ValueError:
        base = 0
    seeds = [zlib.crc32(f"word-freq/timing/{base}/{run}".encode()) for run in range(TIMING_RUNS)]
    return [build_corpus(seed) for seed in seeds]


def run_timing(cwd: Path, corpora: list[str]) -> dict[str, Any]:
    millis: list[float] = []
    for run, corpus in enumerate(corpora, start=1):
        seconds, outcome = time_one(cwd, corpus.encode("utf-8"), run)
        if seconds is None or isinstance(outcome, str):
            return invalid(str(outcome))
        if "error" in outcome:
            return invalid(f"Timing run {run} failed: {outcome['error']}")
        if normalize(outcome.get("result")) != reference(corpus, TIMING_K):
            return invalid(
                f"Timing run {run}: the output on the benchmark corpus differs from the expected "
                "result."
            )
        millis.append(seconds * 1000.0)
    median = statistics.median(millis)
    return {
        "valid": True,
        "score": round(median, 3),
        "feedback": (
            f"All hidden correctness cases passed. Median {median:.1f} ms over {TIMING_RUNS} runs "
            f"of top_words(corpus, {TIMING_K}) (min {min(millis):.1f}, max {max(millis):.1f}); "
            "lower is better."
        ),
        "metrics": {
            "median_ms": round(median, 3),
            "min_ms": round(min(millis), 3),
            "max_ms": round(max(millis), 3),
        },
    }


def judge(cwd: Path) -> dict[str, Any]:
    failed = run_gate(cwd)
    if failed is not None:
        return failed
    corpora = timing_corpora()
    result = run_timing(cwd, corpora)
    if result["valid"]:
        result["metrics"]["corpus_words"] = len(WORD_RE.findall(corpora[0].lower()))
        result["metrics"]["correctness_cases"] = len(gate_cases())
    return result


def main() -> None:
    result = judge(Path.cwd())
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
