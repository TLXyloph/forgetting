"""Hidden judge for toy-mis: maximum independent set on 12 fixed geometric graphs.

Python 3.11+, standard library only. It never runs inside an agent workspace; lethe reaches it
through $LETHE_TASK_DIR and runs it with cwd = the judge tree (the seed with the candidate's
editable files overlaid).

For each graph the candidate (`python3 solver.py`, the graph on stdin) gets 2 seconds of wall-clock
time, interpreter start-up included, and a minimal environment (no LETHE_* variables, no
credentials). Per graph:
  * timeout            -> the graph scores 0 (the candidate stays valid; the feedback says so);
  * crash, bad ids, repeated ids or two adjacent vertices -> the candidate is invalid;
  * otherwise          -> 100 * |S| / |S_ref|, where S_ref is minimum-degree greedy, computed here.
    score = mean over the 12 graphs                                              (maximize)

Graphs are random geometric graphs on integer coordinates (exact arithmetic) drawn from a PRNG built
on `random.Random(seed).random()`, whose stream Python guarantees: identical on every machine and
Python version. They do not depend on LETHE_SEED or LETHE_SPLIT. Prints exactly one JSON line; exit
75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
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

TIME_LIMIT_S = 2.0
SIZES = (200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800)
DEGREES = (8, 12, 16, 10, 14, 20, 12, 18, 10, 16, 14, 12)  # target average degree
COORD = 1_000_000
MAX_OUTPUT_BYTES = 1_000_000
MAX_FILE_BYTES = 64 * 1024 * 1024
# All the candidate inherits from the judge's environment (plus LC_*): enough to run Python.
INHERITED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TEMP", "TMP")
INHERITED_ENV += ("TZ", "SYSTEMROOT")


@dataclass(frozen=True)
class Graph:
    name: str
    n: int
    edges: list[tuple[int, int]]

    @property
    def text(self) -> str:
        lines = [f"{self.n} {len(self.edges)}", *(f"{u} {v}" for u, v in self.edges)]
        return "\n".join(lines) + "\n"

    def adjacency(self) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(self.n)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj


def build_graphs() -> list[Graph]:
    graphs = []
    for k, (n, degree) in enumerate(zip(SIZES, DEGREES, strict=True), start=1):
        draw = random.Random(zlib.crc32(f"toy-mis/{k}/{n}".encode())).random
        points = [(int(draw() * COORD), int(draw() * COORD)) for _ in range(n)]
        radius = int(COORD * math.sqrt(degree / (math.pi * n)))
        limit = radius * radius
        edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2 <= limit
        ]
        graphs.append(Graph(f"g{k:02d}-n{n}", n, edges))
    return graphs


def min_degree_greedy(adj: list[set[int]]) -> list[int]:
    """Reference: repeatedly take a vertex of minimum remaining degree (ties: smallest id)."""
    alive = set(range(len(adj)))
    degree = [len(a) for a in adj]
    chosen: list[int] = []
    while alive:
        v = min(alive, key=lambda x: (degree[x], x))
        chosen.append(v)
        removed = {v} | (adj[v] & alive)
        alive -= removed
        for r in removed:
            for u in adj[r]:
                if u in alive:
                    degree[u] -= 1
    return chosen


# ---------------------------------------------------------------------------------------------
# running the candidate
# ---------------------------------------------------------------------------------------------


def _limits() -> None:  # runs in the child between fork and exec
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 65))  # a backstop, not the time limit


def run_solver(
    cwd: Path, stdin_path: Path, stdout_path: Path, stderr_path: Path
) -> tuple[int, bool, float]:
    """Run solver.py on one graph; returns (exit code, timed_out, wall seconds)."""
    env = {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    with (
        open(stdin_path, "rb") as stdin,
        open(stdout_path, "wb") as out,
        open(stderr_path, "wb") as err,
    ):
        started = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "solver.py"],
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=out,
            stderr=err,
            start_new_session=True,
            preexec_fn=_limits,
        )
        timed_out = False
        try:
            proc.wait(timeout=TIME_LIMIT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return proc.returncode, timed_out, time.perf_counter() - started


def _tail(path: Path, limit: int = 600) -> str:
    try:
        return path.read_bytes()[-limit:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def parse_vertices(path: Path, n: int) -> tuple[list[int] | None, str]:
    """The printed ids, or (None, reason) if the output is not a list of distinct ids in range."""
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_OUTPUT_BYTES + 1)
    except OSError:
        return None, "no output was captured"
    if len(raw) > MAX_OUTPUT_BYTES:
        return None, "the output is larger than 1 MB"
    chosen: list[int] = []
    seen: set[int] = set()
    for token in raw.decode("utf-8", errors="replace").split():
        if not (token.isascii() and token.isdigit() and len(token) <= 9):
            return None, f"token {token[:20]!r} is not a vertex id"
        v = int(token)
        if v >= n:
            return None, f"vertex id {v} is out of range 0..{n - 1}"
        if v in seen:
            return None, f"vertex id {v} is printed more than once"
        seen.add(v)
        chosen.append(v)
    return chosen, ""


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def judge(cwd: Path) -> dict[str, Any]:
    graphs = build_graphs()
    cases: dict[str, float] = {}
    lines: list[str] = []
    timeouts: list[str] = []
    slowest = 0.0
    total_size = total_ref = 0
    with tempfile.TemporaryDirectory(prefix="toymis-judge-") as tmp:
        scratch = Path(tmp)
        for g in graphs:
            adj = g.adjacency()
            ref_size = len(min_degree_greedy(adj))
            stdin_path = scratch / f"{g.name}.in"
            stdin_path.write_text(g.text)
            out_path, err_path = scratch / f"{g.name}.out", scratch / f"{g.name}.err"
            code, timed_out, seconds = run_solver(cwd, stdin_path, out_path, err_path)
            slowest = max(slowest, min(seconds, TIME_LIMIT_S))
            if timed_out:
                cases[g.name] = 0.0
                timeouts.append(g.name)
                lines.append(f"{g.name}: 0.0 (timed out after {TIME_LIMIT_S:g} s)")
                total_ref += ref_size
                continue
            if code != 0:
                what = f"exited with code {code}" if code > 0 else f"was killed by signal {-code}"
                detail = _tail(err_path)
                return invalid(
                    f"On {g.name} (n={g.n}) the solver {what}."
                    + (f" stderr tail: {detail}" if detail else "")
                )
            chosen, problem = parse_vertices(out_path, g.n)
            if chosen is None:
                return invalid(f"On {g.name} (n={g.n}): {problem}.")
            members = set(chosen)
            for v in chosen:
                clash = adj[v] & members
                if clash:
                    return invalid(
                        f"On {g.name} (n={g.n}): vertices {v} and {min(clash)} are adjacent, "
                        "so the set is not independent."
                    )
            cases[g.name] = 100.0 * len(chosen) / ref_size
            total_size += len(chosen)
            total_ref += ref_size
            lines.append(f"{g.name}: {cases[g.name]:.1f} (|S|={len(chosen)}, reference {ref_size})")
    score = sum(cases.values()) / len(cases)
    head = f"Score {score:.3f} = mean over {len(cases)} graphs of 100 * |S| / |S_ref|."
    if timeouts:
        head += f" {len(timeouts)} graph(s) timed out and scored 0: {', '.join(timeouts)}."
    return {
        "valid": True,
        "score": round(score, 6),
        "feedback": head + " " + "; ".join(lines) + f". Slowest run {slowest:.2f} s.",
        "metrics": {
            "graphs": len(cases),
            "timeouts": len(timeouts),
            "total_size": total_size,
            "total_reference": total_ref,
            "slowest_seconds": round(slowest, 3),
        },
        "cases": {k: round(v, 6) for k, v in cases.items()},
    }


def main() -> None:
    sys.stdout.write(json.dumps(judge(Path.cwd()), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
