"""Hidden judge for the bin-packing example. Python 3.11+, standard library only.

It never runs inside an agent workspace; lethe reaches it through $LETHE_TASK_DIR and runs it with
cwd = the judge tree (the seed with the candidate's editable files overlaid).

  1. Build 24 fixed instances (3 item-size distributions x 8 sizes, 100-1000 items each).
  2. Run the candidate's `solve` on all of them in ONE subprocess with a total time limit
     (30 s; BIN_PACKING_TIMEOUT_S overrides it, which the tests use). The subprocess is
     `python -s -B -c <runner>`: the runner is inline (the candidate never learns where this
     file is), the instances come on stdin and the results go back on stdout (no paths), and
     its environment holds no LETHE_* variables and no credentials.
  3. Validate every returned packing here, in the parent, and score it:
         score = 100 * mean(lower_bound / bins_used),  lower_bound = ceil(sum(sizes) / capacity)

The instances use only `random.Random(seed).random()` (whose stream Python guarantees) so they are
identical on every Python version; they do not depend on LETHE_SEED or LETHE_SPLIT.
Prints exactly one JSON line. Exit 75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_S = 30.0
ITEM_COUNTS = (100, 150, 200, 300, 400, 500, 750, 1000)
TRIPLET_COUNTS = (102, 150, 198, 300, 402, 501, 750, 999)  # multiples of 3
DISTRIBUTIONS = ("uniform", "wide", "triplet")
MAX_FILE_BYTES = 64 * 1024 * 1024
# All the candidate inherits from the judge's environment (plus LC_*): enough to run Python.
INHERITED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TEMP", "TMP")
INHERITED_ENV += ("TZ", "SYSTEMROOT")

# What the candidate's process runs (`python -s -B -c`): import `solver` from the judge tree, then
# solve the instances read from stdin, one JSON result per line on the real stdout (the
# candidate's own prints go to stderr).
RUNNER = """
import json, os, sys, time, traceback
root = os.getcwd()
sys.path.insert(0, root)
results = sys.stdout
sys.stdout = sys.stderr

def where(exc):
    frames = [f"{os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}"
              for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith(root)]
    return f" ({frames[-1]})" if frames else ""

def send(record):
    results.write(json.dumps(record) + "\\n")
    results.flush()

try:
    from solver import solve
except Exception as exc:
    send({"name": "<import>", "fatal": f"{type(exc).__name__}: {exc}{where(exc)}"})
    sys.exit(1)
for inst in json.loads(sys.stdin.read()):
    started = time.perf_counter()
    try:
        bins = solve(list(inst["sizes"]), inst["capacity"])
        rec = {"name": inst["name"], "bins": bins, "seconds": time.perf_counter() - started}
        line = json.dumps(rec)
    except Exception as exc:
        line = json.dumps({"name": inst["name"],
                           "error": f"raised {type(exc).__name__}: {exc}{where(exc)}"})
    results.write(line + "\\n")
    results.flush()
"""


# ---------------------------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------------------------


class Rng:
    """Tiny deterministic PRNG facade over `random.Random(seed).random()`."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed).random

    def randint(self, lo: int, hi: int) -> int:
        return lo + min(hi - lo, int(self._random() * (hi - lo + 1)))

    def shuffle(self, items: list[int]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]


@dataclass(frozen=True)
class Instance:
    name: str
    dist: str
    sizes: list[int]
    capacity: int

    @property
    def lower_bound(self) -> int:
        return math.ceil(sum(self.sizes) / self.capacity)


def _uniform(rng: Rng, n: int) -> tuple[list[int], int]:
    return [rng.randint(20, 100) for _ in range(n)], 150


def _wide(rng: Rng, n: int) -> tuple[list[int], int]:
    return [rng.randint(10, 700) for _ in range(n)], 1000


def _triplet(rng: Rng, n: int) -> tuple[list[int], int]:
    """n/3 bins, each filled exactly by three items in (C/4, C/2): the optimum equals the bound."""
    sizes: list[int] = []
    for _ in range(n // 3):
        x = rng.randint(251, 498)
        rest = 1000 - x
        y = rng.randint(max(251, rest - 499), min(499, rest - 251))
        sizes += [x, y, rest - y]
    rng.shuffle(sizes)
    return sizes, 1000


_MAKERS = {"uniform": _uniform, "wide": _wide, "triplet": _triplet}


def build_instances() -> list[Instance]:
    out: list[Instance] = []
    for dist in DISTRIBUTIONS:
        for n in TRIPLET_COUNTS if dist == "triplet" else ITEM_COUNTS:
            rng = Rng(zlib.crc32(f"bin-packing/{dist}/{n}".encode()))
            sizes, capacity = _MAKERS[dist](rng, n)
            out.append(Instance(f"{dist}-n{n}", dist, sizes, capacity))
    return out


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
    """Just enough to run Python: no LETHE_* variables, no credentials."""
    env = {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}
    env["PYTHONHASHSEED"] = "0"  # identical candidates must score identically
    return env


def _tail(stream: IO[bytes], limit: int = 700) -> str:
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - limit))
    return stream.read().decode("utf-8", errors="replace").strip()


def _describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


def run_candidate(
    instances: bytes, cwd: Path, timeout: float
) -> tuple[int, bool, dict[str, dict[str, Any]], str]:
    """Run the runner on the candidate with `instances` on stdin.

    Returns (exit code, timed_out, results by instance name, stderr tail). Its stdin, stdout and
    stderr are anonymous temporary files: no path to them exists, and no pipe can be held open.
    """
    argv = [sys.executable, "-s", "-B", "-c", RUNNER]
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(instances)
        stdin.seek(0)
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=candidate_env(),
            stdin=stdin,
            stdout=stdout,
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
        stdout.seek(0)
        results = read_results(stdout.read(MAX_FILE_BYTES).decode("utf-8", errors="replace"))
        return proc.returncode, timed_out, results, _tail(stderr)


def read_results(text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a line cut off by the kill
        if isinstance(rec, dict) and isinstance(rec.get("name"), str):
            out[rec["name"]] = rec
    return out


# ---------------------------------------------------------------------------------------------
# validation and scoring
# ---------------------------------------------------------------------------------------------


def check_packing(bins: Any, sizes: list[int], capacity: int) -> tuple[int | None, str]:
    """Return (bins used, "") for a valid packing, else (None, reason). Empty bins are ignored."""
    if not isinstance(bins, list):
        return None, f"solve() must return a list of bins (got {type(bins).__name__})"
    n = len(sizes)
    seen = [False] * n
    used = 0
    for b, items in enumerate(bins):
        if not isinstance(items, list):
            return None, f"bin {b} is not a list (got {type(items).__name__})"
        load = 0
        for item in items:
            if type(item) is not int:
                return None, f"bin {b} holds a non-int index ({type(item).__name__}); use ints"
            if not 0 <= item < n:
                return None, f"item index {item} is out of range 0..{n - 1}"
            if seen[item]:
                return None, f"item index {item} is packed twice"
            seen[item] = True
            load += sizes[item]
        if load > capacity:
            return None, f"bin {b} holds total size {load}, above the capacity {capacity}"
        used += 1 if items else 0
    absent = [i for i, flag in enumerate(seen) if not flag]
    if absent:
        return None, f"{len(absent)} item(s) missing from the packing (first: index {absent[0]})"
    return used, ""


def _seconds(rec: dict[str, Any]) -> float:
    value = rec.get("seconds")
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return float(value) if ok else 0.0


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def evaluate(instances: list[Instance], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, float] = {}
    bins_used: dict[str, int] = {}
    problems: list[str] = []
    for inst in instances:
        rec = results.get(inst.name)
        if rec is None:
            continue
        if "error" in rec:
            problems.append(f"{inst.name}: {rec['error']}")
            continue
        used, reason = check_packing(rec.get("bins"), inst.sizes, inst.capacity)
        if used is None:
            problems.append(f"{inst.name}: {reason}")
            continue
        bins_used[inst.name] = used
        cases[inst.name] = 100.0 * inst.lower_bound / used
    if problems:
        shown = "; ".join(problems[:3]) + (
            f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        )
        return invalid(f"Invalid on {len(problems)} of {len(instances)} instances. {shown}")
    score = sum(cases.values()) / len(cases)
    lines = [
        f"Score {score:.3f} = mean over {len(cases)} instances of 100 * lower_bound / bins_used."
    ]
    per_dist = []
    for dist in DISTRIBUTIONS:
        vals = [cases[i.name] for i in instances if i.dist == dist]
        per_dist.append(f"{dist} {sum(vals) / len(vals):.2f}")
    lines.append("Average by distribution: " + ", ".join(per_dist) + ".")
    worst = sorted(instances, key=lambda i: (cases[i.name], i.name))[:3]
    lines.append(
        "Worst instances: "
        + "; ".join(
            f"{i.name} {cases[i.name]:.2f} "
            f"({bins_used[i.name]} bins vs lower bound {i.lower_bound})"
            for i in worst
        )
        + "."
    )
    solve_s = sum(_seconds(results[i.name]) for i in instances)
    lines.append(f"Time spent in solve(): {solve_s:.2f} s.")
    return {
        "valid": True,
        "score": round(score, 6),
        "feedback": " ".join(lines),
        "metrics": {
            "instances": len(cases),
            "total_bins": sum(bins_used.values()),
            "total_lower_bound": sum(i.lower_bound for i in instances),
            "solve_seconds": round(solve_s, 4),
        },
        "cases": {k: round(v, 6) for k, v in cases.items()},
    }


def judge(cwd: Path, timeout: float) -> dict[str, Any]:
    instances = build_instances()
    payload = json.dumps(
        [{"name": i.name, "sizes": i.sizes, "capacity": i.capacity} for i in instances]
    ).encode("utf-8")
    started = time.perf_counter()
    code, timed_out, results, stderr_tail = run_candidate(payload, cwd, timeout)
    elapsed = time.perf_counter() - started
    done = [i for i in instances if i.name in results]
    stuck = next((i.name for i in instances if i.name not in results), None)
    if timed_out:
        return invalid(
            f"Timed out after {timeout:g} s in total: {len(done)} of {len(instances)} "
            f"instances finished; it was still working on {stuck}.",
            finished=len(done),
        )
    fatal = next((r for r in results.values() if "fatal" in r), None)
    if fatal is not None:
        return invalid(f"The candidate could not be loaded: {fatal['fatal']}")
    if stuck is not None and not any("error" in r for r in results.values()):
        return invalid(
            f"The candidate process {_describe_exit(code)} while solving {stuck} "
            f"({len(done)} of {len(instances)} instances finished)."
            + (f" stderr tail: {stderr_tail}" if stderr_tail else ""),
            finished=len(done),
        )
    result = evaluate(instances, results)
    if result["valid"]:
        result["metrics"]["wall_seconds"] = round(elapsed, 3)
    return result


def main() -> None:
    timeout = float(os.environ.get("BIN_PACKING_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    result = judge(Path.cwd(), timeout)
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
