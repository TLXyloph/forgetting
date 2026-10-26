"""Hidden judge for the bin-packing example. Python 3.11+, standard library only.

It never runs inside an agent workspace; lethe reaches it through $LETHE_TASK_DIR and runs it with
cwd = the judge tree (the seed with the candidate's editable files overlaid).

  1. Build 24 fixed instances (3 item-size distributions x 8 sizes, 100-1000 items each).
  2. Run the candidate's `solve` on all of them in ONE subprocess with a total time limit
     (30 s; BIN_PACKING_TIMEOUT_S overrides it, which the tests use).
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
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_S = 30.0
ITEM_COUNTS = (100, 150, 200, 300, 400, 500, 750, 1000)
TRIPLET_COUNTS = (102, 150, 198, 300, 402, 501, 750, 999)  # multiples of 3
DISTRIBUTIONS = ("uniform", "wide", "triplet")
MAX_FILE_BYTES = 64 * 1024 * 1024


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


def _candidate_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LETHE_")}
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    return env


def _tail(path: Path, limit: int = 700) -> str:
    try:
        data = path.read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


def run_candidate(
    instances_path: Path, results_path: Path, stderr_path: Path, cwd: Path, timeout: float
) -> tuple[int, bool]:
    """Run the driver child on the candidate; returns (exit code, timed_out).

    `-P` keeps the cwd and the script directory off sys.path (the driver adds the candidate's
    tree itself); unlike `-I` it leaves PYTHONHASHSEED effective.
    """
    argv = [sys.executable, "-P", str(Path(__file__).resolve()), "--child"]
    argv += [str(instances_path), str(results_path)]
    with open(stderr_path, "wb") as stderr:
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
    return proc.returncode, timed_out


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
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
    with tempfile.TemporaryDirectory(prefix="binpack-judge-") as tmp:
        scratch = Path(tmp)
        instances_path = scratch / "instances.json"
        results_path = scratch / "results.jsonl"
        stderr_path = scratch / "stderr.txt"
        instances_path.write_text(
            json.dumps(
                [{"name": i.name, "sizes": i.sizes, "capacity": i.capacity} for i in instances]
            )
        )
        started = time.perf_counter()
        code, timed_out = run_candidate(instances_path, results_path, stderr_path, cwd, timeout)
        elapsed = time.perf_counter() - started
        results = read_results(results_path)
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
            detail = _tail(stderr_path)
            return invalid(
                f"The candidate process {_describe_exit(code)} while solving {stuck} "
                f"({len(done)} of {len(instances)} instances finished)."
                + (f" stderr tail: {detail}" if detail else ""),
                finished=len(done),
            )
        result = evaluate(instances, results)
        if result["valid"]:
            result["metrics"]["wall_seconds"] = round(elapsed, 3)
        return result


# ---------------------------------------------------------------------------------------------
# child: runs inside the candidate subprocess
# ---------------------------------------------------------------------------------------------


def _frames(exc: BaseException, root: str) -> str:
    """Where in the candidate's own files the exception came from (judge frames are hidden)."""
    where = [
        f"{os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}"
        for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.startswith(root)
    ]
    return f" ({where[-1]})" if where else ""


def child_main(instances_path: str, results_path: str) -> int:
    root = os.getcwd()
    sys.path.insert(0, root)
    sys.stdout = sys.stderr  # stray prints must not reach the results channel
    with open(results_path, "w", encoding="utf-8") as results:

        def write(rec: dict[str, Any]) -> None:
            results.write(json.dumps(rec) + "\n")
            results.flush()

        try:
            import solver  # the candidate

            solve = solver.solve
        except Exception as exc:
            write({"name": "<import>", "fatal": f"{type(exc).__name__}: {exc}{_frames(exc, root)}"})
            return 1
        for inst in json.loads(Path(instances_path).read_text()):
            started = time.perf_counter()
            try:
                bins = solve(list(inst["sizes"]), inst["capacity"])
                seconds = time.perf_counter() - started
                rec = {"name": inst["name"], "bins": bins, "seconds": seconds}
                line = json.dumps(rec)
            except Exception as exc:
                rec = {
                    "name": inst["name"],
                    "error": f"raised {type(exc).__name__}: {exc}{_frames(exc, root)}",
                }
                line = json.dumps(rec)
            results.write(line + "\n")
            results.flush()
    return 0


def main() -> None:
    timeout = float(os.environ.get("BIN_PACKING_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    result = judge(Path.cwd(), timeout)
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        sys.exit(child_main(sys.argv[2], sys.argv[3]))
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
