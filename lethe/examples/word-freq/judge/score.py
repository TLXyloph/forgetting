"""Hidden judge for the word-freq example (a performance task). Python 3.11+, stdlib only.

It never runs inside an agent workspace; lethe reaches it through $LETHE_TASK_DIR and runs it with
cwd = the judge tree (the seed with the candidate's editable files overlaid).

  1. Correctness gate: about sixty hidden, generated cases of many kinds are run in one subprocess.
     Any mismatch makes the candidate invalid; the feedback names the failing case *kinds* only,
     never their contents.
  2. Timing: `top_words(corpus, 50)` on a fixed generated corpus of about 300k words, in five
     fresh subprocesses (so nothing can be cached between runs); only the call is timed, and each
     run's output is checked against the reference as well.
         score = median wall-clock milliseconds of those five calls     (minimize)

The data (reference, corpus, gate cases) lives in the sibling module wordfreq_data.py and is
generated from fixed seeds, so every machine and Python version sees identical text. Prints exactly
one JSON line; exit 75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

# Bound before any candidate code is imported, so the candidate cannot replace the clock used to
# time it (a module-level `time.perf_counter = ...` in the candidate would not affect this name).
_perf_counter = time.perf_counter
TIMING_K = 50
TIMING_RUNS = 5
GATE_TIMEOUT_S = 60.0
RUN_TIMEOUT_S = 30.0
MAX_FILE_BYTES = 64 * 1024 * 1024


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


def _candidate_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LETHE_")}
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_child(
    args: list[str], stderr_path: Path, cwd: Path, timeout: float
) -> tuple[int, bool, float]:
    """Run this file in child mode on the candidate: (exit code, timed_out, wall seconds)."""
    argv = [sys.executable, "-P", str(Path(__file__).resolve()), *args]
    with open(stderr_path, "wb") as stderr:
        started = time.perf_counter()
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=_candidate_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=functools.partial(_limits, max(60, int(timeout * 8))),
        )
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_group(proc.pid)
        proc.wait()
    return proc.returncode, timed_out, time.perf_counter() - started


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a line cut off by the kill
        if isinstance(rec, dict):
            out.append(rec)
    return out


def tail(path: Path, limit: int = 600) -> str:
    try:
        return path.read_bytes()[-limit:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


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


# ---------------------------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------------------------


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def run_gate(scratch: Path, cwd: Path) -> dict[str, Any] | None:
    """None when every hidden case passes; otherwise an invalid result."""
    cases = gate_cases()
    cases_path, out_path, err_path = (
        scratch / "cases.json",
        scratch / "gate.jsonl",
        scratch / "gate.err",
    )
    cases_path.write_text(json.dumps(cases))
    code, timed_out, _ = run_child(
        ["--gate", str(cases_path), str(out_path)], err_path, cwd, GATE_TIMEOUT_S
    )
    records = {r["i"]: r for r in read_jsonl(out_path) if isinstance(r.get("i"), int)}
    fatal = next((r for r in read_jsonl(out_path) if "fatal" in r), None)
    if fatal is not None:
        return invalid(f"The candidate could not be loaded: {fatal['fatal']}")
    if timed_out or len(records) < len(cases):
        stuck = next((cases[i][0] for i in range(len(cases)) if i not in records), "?")
        how = (
            f"timed out after {GATE_TIMEOUT_S:g} s"
            if timed_out
            else f"{describe_exit(code)}" + (f" ({tail(err_path)})" if tail(err_path) else "")
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


def run_timing(scratch: Path, cwd: Path, corpus: str) -> dict[str, Any]:
    corpus_path = scratch / "corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    expected = reference(corpus, TIMING_K)
    millis: list[float] = []
    for run in range(1, TIMING_RUNS + 1):
        out_path, err_path = scratch / f"time{run}.json", scratch / f"time{run}.err"
        code, timed_out, wall = run_child(
            ["--time", str(corpus_path), str(TIMING_K), str(out_path)], err_path, cwd, RUN_TIMEOUT_S
        )
        recs = read_jsonl(out_path)
        rec = recs[0] if recs else {}
        if timed_out:
            return invalid(
                f"Timing run {run} exceeded {RUN_TIMEOUT_S:g} s on the benchmark corpus."
            )
        if "fatal" in rec or "error" in rec:
            return invalid(f"Timing run {run} failed: {rec.get('fatal') or rec['error']}")
        if "seconds" not in rec:
            detail = tail(err_path)
            return invalid(
                f"Timing run {run}: the candidate process {describe_exit(code)}."
                + (f" stderr tail: {detail}" if detail else "")
            )
        seconds = rec["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not (0 < seconds <= wall)
        ):
            return invalid(f"Timing run {run}: the reported call time is not credible.")
        if normalize(rec.get("result")) != expected:
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
    with tempfile.TemporaryDirectory(prefix="wordfreq-judge-") as tmp:
        scratch = Path(tmp)
        failed = run_gate(scratch, cwd)
        if failed is not None:
            return failed
        corpus = build_corpus()
        result = run_timing(scratch, cwd, corpus)
        if result["valid"]:
            result["metrics"]["corpus_words"] = len(WORD_RE.findall(corpus.lower()))
            result["metrics"]["correctness_cases"] = len(gate_cases())
        return result


# ---------------------------------------------------------------------------------------------
# children: run inside the candidate subprocess
# ---------------------------------------------------------------------------------------------


def _frames(exc: BaseException, root: str) -> str:
    """Where in the candidate's own files an exception came from (judge frames are hidden)."""
    where = [
        f"{os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}"
        for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.startswith(root)
    ]
    return f" at {where[-1]}" if where else ""


def _load_candidate(out: Any, root: str) -> Any:
    try:
        import wordfreq  # the candidate

        return wordfreq.top_words
    except Exception as exc:
        out.write(json.dumps({"fatal": f"{type(exc).__name__}{_frames(exc, root)}"}) + "\n")
        out.flush()
        return None


def _call(fn: Any, text: str, k: int, root: str) -> tuple[dict[str, Any], float]:
    started = _perf_counter()
    try:
        result = fn(text, k)
        seconds = _perf_counter() - started
        json.dumps(result)
    except Exception as exc:
        return {"error": f"raised {type(exc).__name__}{_frames(exc, root)}"}, 0.0
    return {"result": result}, seconds


def child_gate(cases_path: str, out_path: str) -> int:
    root = os.getcwd()
    sys.path.insert(0, root)
    sys.stdout = sys.stderr  # stray prints must not reach the results channel
    with open(out_path, "w", encoding="utf-8") as out:
        fn = _load_candidate(out, root)
        if fn is None:
            return 1
        for i, (_kind, text, k) in enumerate(json.loads(Path(cases_path).read_text())):
            rec, _seconds = _call(fn, text, k, root)
            out.write(json.dumps({"i": i, **rec}) + "\n")
            out.flush()
    return 0


def child_time(corpus_path: str, k: str, out_path: str) -> int:
    root = os.getcwd()
    sys.path.insert(0, root)
    sys.stdout = sys.stderr
    with open(out_path, "w", encoding="utf-8") as out:
        fn = _load_candidate(out, root)
        if fn is None:
            return 1
        text = Path(corpus_path).read_text(encoding="utf-8")
        gc.collect()
        rec, seconds = _call(fn, text, int(k), root)
        out.write(json.dumps({**rec, "seconds": seconds}) + "\n")
    return 0


def main() -> None:
    result = judge(Path.cwd())
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gate":
        sys.exit(child_gate(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 5 and sys.argv[1] == "--time":
        sys.exit(child_time(sys.argv[2], sys.argv[3], sys.argv[4]))
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
