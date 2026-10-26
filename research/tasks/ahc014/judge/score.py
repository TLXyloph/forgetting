"""Hidden judge for ahc014 (AtCoder Heuristic Contest 014, "RectJoin"). Python 3.11+, stdlib only.

Runs with cwd = the judge tree, where `[judge].build` has already compiled the candidate's
main.cpp into ./main. For the official inputs 0000..0029 (from the tools that
research/scripts/fetch_ahc014.sh downloads and builds into research/tasks/ahc014/vendor/):

  1. ./main runs with the input on stdin and a 5 s wall-clock limit (AHC014_TIME_LIMIT_S
     overrides it); stdout goes to a file;
  2. the official `vis` binary scores that output: `vis <input> <output>` prints `Score = <n>`.

A run that times out, crashes, or whose output `vis` rejects scores 0 on that input (as in the
contest's system test). score = mean over the 30 inputs, `cases` = score per input (maximize). When
all 30 inputs fail the candidate is invalid. If the tools are missing the result is invalid with
instructions. AHC014_TOOLS_DIR overrides the vendor directory (used by the tests, which plug in a
fake `vis`). Inputs are scored one after another so that time-limited programs see the same load;
a throwaway run on empty input comes first, so a new binary's one-off start-up cost (macOS scans
freshly built executables) is not charged to input 0000.
Prints exactly one JSON line; exit 75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

INPUT_IDS = [f"{i:04d}" for i in range(30)]
TIME_LIMIT_S = 5.0
VIS_TIMEOUT_S = 20.0
MAX_FILE_BYTES = 64 * 1024 * 1024
SCORE_RE = re.compile(r"Score\s*=\s*(\d+)")


def find_tools(vendor: Path) -> tuple[Path, Path] | None:
    """(vis binary, input directory) of the built official tools under `vendor`, if present."""
    for vis in sorted(vendor.glob("**/target/release/vis")):
        inputs = vis.parents[2] / "in"
        if vis.is_file() and os.access(vis, os.X_OK) and inputs.is_dir():
            return vis, inputs
    return None


def _limits() -> None:  # runs in the child between fork and exec
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 65))  # a backstop, not the time limit


def run_limited(
    argv: list[str],
    *,
    cwd: Path,
    stdin_path: Path | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> tuple[int, bool, float]:
    """Run a process in its own session; returns (exit code, timed_out, wall seconds)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LETHE_")}
    with contextlib.ExitStack() as files:
        stdin = files.enter_context(open(stdin_path, "rb")) if stdin_path else subprocess.DEVNULL
        out = files.enter_context(open(stdout_path, "wb"))
        err = files.enter_context(open(stderr_path, "wb"))
        started = time.perf_counter()
        proc = subprocess.Popen(
            argv,
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
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return proc.returncode, timed_out, time.perf_counter() - started


def tail(path: Path, limit: int = 300) -> str:
    try:
        return path.read_bytes()[-limit:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def read_header(path: Path) -> tuple[int, int]:
    """N and M from the first line of an input file."""
    n, m = path.read_text(encoding="utf-8").split(None, 2)[:2]
    return int(n), int(m)


def score_case(
    name: str, main_bin: Path, inputs: Path, vis: Path, scratch: Path, cwd: Path, limit: float
) -> tuple[int, str, str, float]:
    """(score, status, detail, seconds) for one input. status is OK, TLE, RE or WA."""
    in_path = inputs / f"{name}.txt"
    out_path, err_path = scratch / f"{name}.out", scratch / f"{name}.err"
    code, timed_out, seconds = run_limited(
        [str(main_bin)],
        cwd=cwd,
        stdin_path=in_path,
        stdout_path=out_path,
        stderr_path=err_path,
        timeout=limit,
    )
    if timed_out:
        return 0, "TLE", f"no answer within {limit:g} s", min(seconds, limit)
    if code != 0:
        how = f"exit code {code}" if code > 0 else f"signal {-code}"
        return 0, "RE", f"the program ended with {how}; {tail(err_path)}".rstrip("; "), seconds
    vis_out, vis_err = scratch / f"{name}.vis", scratch / f"{name}.viserr"
    vcode, vtimed_out, _ = run_limited(
        [str(vis), str(in_path), str(out_path)],
        cwd=scratch,
        stdin_path=None,
        stdout_path=vis_out,
        stderr_path=vis_err,
        timeout=VIS_TIMEOUT_S,
    )
    text = vis_out.read_text(encoding="utf-8", errors="replace") if vis_out.exists() else ""
    found = SCORE_RE.search(text)
    message = " ".join(SCORE_RE.sub("", text, count=1).split())[:200]
    if vtimed_out or vcode != 0 or found is None:
        detail = message or tail(vis_err) or f"vis exited with code {vcode}"
        return 0, "WA", detail, seconds
    score = int(found.group(1))
    if score == 0 and message:
        return 0, "WA", message, seconds
    return score, "OK", "", seconds


def judge(cwd: Path) -> dict[str, Any]:
    task_dir = Path(os.environ.get("LETHE_TASK_DIR") or Path(__file__).resolve().parents[1])
    vendor = Path(os.environ.get("AHC014_TOOLS_DIR") or task_dir / "vendor")
    tools = find_tools(vendor)
    if tools is None:
        return invalid(
            f"The official AHC014 tools were not found under {vendor}. Run "
            "research/scripts/fetch_ahc014.sh to download and build them (it needs Rust: "
            "install it with rustup), then evaluate again."
        )
    vis, inputs = tools
    main_bin = cwd / "main"
    if not main_bin.is_file():
        return invalid("./main does not exist: the build step did not produce it.")
    missing = [n for n in INPUT_IDS if not (inputs / f"{n}.txt").is_file()]
    if missing:
        return invalid(f"The tools' input directory {inputs} lacks {', '.join(missing[:3])} ...")
    limit = float(os.environ.get("AHC014_TIME_LIMIT_S", TIME_LIMIT_S))

    cases: dict[str, float] = {}
    status: dict[str, str] = {}
    failures: list[str] = []
    slowest = 0.0
    with tempfile.TemporaryDirectory(prefix="ahc014-judge-") as tmp:
        # A freshly built binary pays a one-off start-up cost on some systems (macOS scans new
        # executables: 0.1-0.7 s). Pay it here, on empty input, not inside input 0000's time limit.
        warm_out, warm_err = Path(tmp) / "warmup.out", Path(tmp) / "warmup.err"
        run_limited(
            [str(main_bin)],
            cwd=cwd,
            stdin_path=None,
            stdout_path=warm_out,
            stderr_path=warm_err,
            timeout=1.0,
        )
        for name in INPUT_IDS:
            score, state, detail, seconds = score_case(
                name, main_bin, inputs, vis, Path(tmp), cwd, limit
            )
            cases[name], status[name] = float(score), state
            slowest = max(slowest, seconds)
            if state != "OK":
                failures.append(f"{name} {state}: {detail}")
    if len(failures) == len(INPUT_IDS):
        return invalid(
            f"The program failed on all {len(INPUT_IDS)} inputs. First failure: {failures[0]}"
        )
    mean = sum(cases.values()) / len(cases)
    low = sorted(INPUT_IDS, key=lambda n: (cases[n], n))[:3]
    low_text = ", ".join(
        "{} (N={}, M={}) {:,.0f}".format(n, *read_header(inputs / f"{n}.txt"), cases[n])
        for n in low
    )
    parts = [
        f"Mean score {mean:,.0f} over {len(INPUT_IDS)} inputs (official vis; higher is better).",
        f"Lowest: {low_text}.",
        f"Slowest run {slowest:.2f} s of {limit:g} s.",
    ]
    if failures:
        shown = "; ".join(failures[:3]) + (
            f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        )
        parts.append(f"{len(failures)} input(s) scored 0: {shown}.")
    return {
        "valid": True,
        "score": round(mean, 3),
        "feedback": " ".join(parts),
        "metrics": {
            "inputs": len(INPUT_IDS),
            "failures": len(failures),
            "timeouts": sum(1 for s in status.values() if s == "TLE"),
            "slowest_seconds": round(slowest, 3),
        },
        "cases": cases,
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
