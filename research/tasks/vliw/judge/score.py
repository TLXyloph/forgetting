"""Hidden judge for the vliw task (Anthropic's original performance take-home). Python 3.11+.

Runs with cwd = the judge tree: the cloned take-home (the seed, see research/scripts/fetch_vliw.sh)
with the candidate's `perf_takehome.py` overlaid. Nothing else of the candidate is ever judged, so
edits to `problem.py` or `tests/` (the README warns that models "fix" N_CORES) cannot change the
score.

The check replicates `tests/submission_tests.py` (forest height 10, 16 rounds, batch 256): the
final `inp_values` array must equal the reference's on 8 random inputs, and the score is the cycle
count of a 9th run on the frozen simulator. It is split into two processes so that candidate code
never shares a process, or a directory, with the simulator that measures it:

  stage A (untrusted)  runs in a THROWAWAY COPY of the judge tree, as `python -s -B -c <runner>`
                       (inline: the candidate never learns where this file is) with a minimal
                       environment: imports the candidate's KernelBuilder, builds the kernel,
                       dumps `kb.instrs` as JSON. Whatever it writes into the tree is discarded.
                       Limit: VLIW_BUILD_TIMEOUT_S (default 300 s).
  stage B (trusted)    reads `tests/frozen_problem.py` from the pristine tree, checks its sha256
                       against FROZEN_SHA256 (the take-home commit fetch_vliw.sh pins), executes
                       exactly those bytes, rebuilds the program from the JSON (plain data) and
                       runs it: no candidate code is imported here. Its report comes back on
                       stdout. Limit: VLIW_RUN_TIMEOUT_S (default 300 s).

The 8 correctness inputs are seeded from LETHE_SEED (a fresh random value per evaluation, like the
original's unseeded generator); the cycle run uses a fixed seed so the score is a pure function of
the candidate. Any exception or mismatch makes the candidate invalid, with a traceback tail.
Prints exactly one JSON line; exit 75 means the judge itself broke, never the candidate.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from typing import IO, Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

BASELINE = 147734
HEIGHT, ROUNDS, BATCH = 10, 16, 256
N_NODES = 2 ** (HEIGHT + 1) - 1
CORRECTNESS_RUNS = 8
CYCLE_SEED = 123
BUILD_TIMEOUT_S = 300.0
RUN_TIMEOUT_S = 300.0
MAX_FILE_BYTES = 512 * 1024 * 1024
TAIL_CHARS = 1500
# sha256 of tests/frozen_problem.py at commit 5452f74bd977807ac2e74f3d29432b9df6f25197 of
# anthropics/original_performance_takehome, the commit research/scripts/fetch_vliw.sh checks out.
FROZEN_SHA256 = "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9"
# All the stages inherit from the judge's environment (plus LC_*): no LETHE_*, no credentials.
INHERITED_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TEMP", "TMP")
INHERITED_ENV += ("TZ", "SYSTEMROOT")

# Stage A, run as `python -s -B -c STAGE_A <program.json> <report.json> <height> <n_nodes> <batch>
# <rounds> <tail chars>` with cwd = a throwaway copy of the judge tree.
STAGE_A = """
import json, os, sys, time, traceback
program_path, report_path = sys.argv[1], sys.argv[2]
height, n_nodes, batch, rounds, tail = (int(x) for x in sys.argv[3:8])
root = os.getcwd()
sys.path.insert(0, root)
sys.stdout = sys.stderr  # whatever the candidate prints must not matter

def where(exc):
    frames = [f"  {os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}: {f.line or ''}"
              .rstrip() for f in traceback.extract_tb(exc.__traceback__)
              if f.filename.startswith(root)]
    return "\\n".join(frames[-6:])

def reject(obj):
    raise TypeError(f"{type(obj).__name__} cannot be part of an instruction")

try:
    started = time.perf_counter()
    from perf_takehome import KernelBuilder
    kb = KernelBuilder()
    kb.build_kernel(height, n_nodes, batch, rounds)
    seconds = time.perf_counter() - started
    instrs = kb.instrs
    if not isinstance(instrs, list):
        raise TypeError(f"kb.instrs must be a list of instruction bundles, got {type(instrs)}")
    with open(program_path, "w", encoding="utf-8") as f:
        json.dump(instrs, f, default=reject, allow_nan=False)
    report = {"bundles": len(instrs), "build_seconds": seconds}
except BaseException as exc:
    report = {"error": f"{type(exc).__name__}: {exc}\\n{where(exc)}"[-tail:]}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f)
sys.exit(0 if "error" not in report else 1)
"""


# ---------------------------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------------------------


def _limits(cpu_seconds: int) -> None:  # runs in the child between fork and exec
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))


def run_stage(argv: list[str], cwd: Path, timeout: float) -> tuple[int, bool, bytes, str]:
    """Run one stage in its own session with a minimal environment.

    Returns (exit code, timed_out, stdout, stderr tail). Both streams are anonymous temporary
    files, so nothing else can find (or rewrite) what a stage reports on stdout.
    """
    env = {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}
    env["PYTHONHASHSEED"] = "0"  # identical candidates must score identically
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=functools.partial(_limits, max(120, int(timeout * 4))),
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
        stdout.seek(0)
        return proc.returncode, timed_out, stdout.read(1024 * 1024), tail(stderr)


def read_report(data: bytes) -> dict[str, Any] | None:
    try:
        report = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    return report if isinstance(report, dict) else None


def tail(stream: IO[bytes], limit: int = 600) -> str:
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - limit))
    return stream.read().decode("utf-8", errors="replace").strip()


def invalid(feedback: str, **metrics: Any) -> dict[str, Any]:
    return {"valid": False, "feedback": feedback, "metrics": metrics}


def describe_exit(code: int) -> str:
    return f"exited with code {code}" if code >= 0 else f"was killed by signal {-code}"


def judge(cwd: Path) -> dict[str, Any]:
    frozen = cwd / "tests" / "frozen_problem.py"
    if not frozen.is_file() or not (cwd / "perf_takehome.py").is_file():
        return invalid(
            "The judge tree has no tests/frozen_problem.py or perf_takehome.py: the take-home "
            "was not fetched completely. Run research/scripts/fetch_vliw.sh."
        )
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        return invalid(
            f"tests/frozen_problem.py is not the pinned take-home (sha256 {digest[:12]}, expected "
            f"{FROZEN_SHA256[:12]}): fetch it again with research/scripts/fetch_vliw.sh."
        )
    try:
        base_seed = int(os.environ.get("LETHE_SEED", "0"))
    except ValueError:
        base_seed = 0
    build_timeout = float(os.environ.get("VLIW_BUILD_TIMEOUT_S", BUILD_TIMEOUT_S))
    run_timeout = float(os.environ.get("VLIW_RUN_TIMEOUT_S", RUN_TIMEOUT_S))
    with tempfile.TemporaryDirectory(prefix="vliw-judge-") as tmp:
        scratch = Path(tmp)
        program, built, copy = scratch / "program.json", scratch / "build.json", scratch / "tree"
        shutil.copytree(cwd, copy, symlinks=True)  # stage A may write anything: only here
        sizes = [str(n) for n in (HEIGHT, N_NODES, BATCH, ROUNDS, TAIL_CHARS)]
        argv = [sys.executable, "-s", "-B", "-c", STAGE_A, str(program), str(built), *sizes]
        code, timed_out, _, err = run_stage(argv, copy, build_timeout)
        if timed_out:
            return invalid(f"build_kernel did not finish within {build_timeout:g} s.")
        try:
            report = read_report(built.read_bytes())
        except OSError:
            report = None
        if report is None or "error" in report:
            died = f"the build process {describe_exit(code)}. {err}"
            detail = (report or {}).get("error") or died
            return invalid(f"Building the kernel failed: {detail}")
        seeds = [(base_seed * 7919 + 104729 * i) & 0xFFFFFFFF for i in range(CORRECTNESS_RUNS)]
        argv = [sys.executable, "-s", "-S", str(Path(__file__).resolve()), "--run", str(cwd)]
        argv += [str(program), json.dumps(seeds)]
        code, timed_out, out, err = run_stage(argv, scratch, run_timeout)
        if timed_out:
            return invalid(
                f"Simulating the kernel did not finish within {run_timeout:g} s "
                "(an endless loop in the generated program?)."
            )
        sim = read_report(out)
        if sim is None:
            return invalid(f"The simulation process {describe_exit(code)}. {err}")
        if "error" in sim:
            return invalid(sim["error"], frozen_problem_sha=digest[:12])
    cycles = int(sim["cycles"])
    if cycles <= 0:  # unreachable for a correct kernel; keeps a candidate from crashing the judge
        return invalid("The program finished in 0 cycles: no bundle used any machine engine.")
    speedup = BASELINE / cycles
    return {
        "valid": True,
        "score": cycles,
        "feedback": (
            f"Correct on {CORRECTNESS_RUNS} random inputs. Cycles: {cycles:,} "
            f"({speedup:.2f}x faster than the {BASELINE:,}-cycle baseline; lower is better). "
            f"The program has {report['bundles']:,} instruction bundles."
        ),
        "metrics": {
            "cycles": cycles,
            "speedup": round(speedup, 4),
            "bundles": int(report["bundles"]),
            "build_seconds": round(float(report["build_seconds"]), 3),
            "sim_seconds": round(float(sim["sim_seconds"]), 3),
            "frozen_problem_sha": sim["frozen_problem_sha"],
        },
    }


# ---------------------------------------------------------------------------------------------
# stage B: trusted. Runs the program on the frozen simulator; imports no candidate code.
# ---------------------------------------------------------------------------------------------


def _where(exc: BaseException, root: str) -> str:
    """Frames in the simulator, one per line (the judge's own are hidden)."""
    frames = [
        f"  {os.path.relpath(f.filename, root)}:{f.lineno} in {f.name}: {f.line or ''}".rstrip()
        for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.startswith(root)
    ]
    return "\n".join(frames[-6:])


def _plain(x: Any) -> Any:
    return tuple(_plain(y) for y in x) if isinstance(x, list) else x


def load_frozen(root: str) -> tuple[types.ModuleType | None, str]:
    """Execute the pristine tree's tests/frozen_problem.py, but only if those exact bytes are the
    pinned version; returns (module or None, their sha256)."""
    path = Path(root) / "tests" / "frozen_problem.py"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FROZEN_SHA256:
        return None, digest
    module = types.ModuleType("frozen_problem")
    module.__file__ = str(path)
    sys.modules["frozen_problem"] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module, digest


def stage_run(root: str, program_path: str, seeds_json: str) -> int:
    reply = sys.stdout
    sys.stdout = sys.stderr  # nothing the simulator prints may reach the report
    fp, digest = load_frozen(root)
    if fp is None:
        report = {
            "error": "tests/frozen_problem.py changed while the kernel was being built (sha256 "
            f"{digest[:12]}); the candidate is rejected."
        }
        reply.write(json.dumps(report) + "\n")
        return 0

    bundles = json.loads(Path(program_path).read_text(encoding="utf-8"))
    program = [{e: [_plain(s) for s in slots] for e, slots in b.items()} for b in bundles]
    runs = [
        (f"random input {i + 1} of {CORRECTNESS_RUNS}", s)
        for i, s in enumerate(json.loads(seeds_json))
    ]
    runs.append(("the cycle-count input", CYCLE_SEED))
    report: dict[str, Any] = {}
    sim_seconds = 0.0
    cycles = 0
    for label, seed in runs:
        random.seed(seed)
        forest = fp.Tree.generate(HEIGHT)
        inp = fp.Input.generate(forest, BATCH, ROUNDS)
        mem = fp.build_mem_image(forest, inp)
        machine = fp.Machine(mem, program, fp.DebugInfo(scratch_map={}), n_cores=fp.N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        started = time.perf_counter()
        try:
            machine.run()
        except BaseException as exc:
            pc = machine.cores[0].pc - 1
            bundle = json.dumps(bundles[pc]) if 0 <= pc < len(bundles) else "?"
            report = {
                "error": (
                    f"The simulator raised {type(exc).__name__}: {exc} on {label}, while executing "
                    f"bundle {pc}: {bundle[:300]}\n{_where(exc, root)}"
                )[-TAIL_CHARS:]
            }
            break
        sim_seconds += time.perf_counter() - started
        *_, ref_mem = fp.reference_kernel2(mem)  # the last yield is the final memory
        p = ref_mem[6]
        if machine.mem[p : p + len(inp.values)] != ref_mem[p : p + len(inp.values)]:
            report = {
                "error": f"Incorrect output values on {label} (forest height {HEIGHT}, "
                f"{ROUNDS} rounds, batch {BATCH})."
            }
            break
        cycles = machine.cycle
    if not report:
        report = {"cycles": cycles, "sim_seconds": sim_seconds / len(runs)}
    report["frozen_problem_sha"] = digest[:12]
    reply.write(json.dumps(report) + "\n")
    reply.flush()
    return 0


def main() -> None:
    sys.stdout.write(json.dumps(judge(Path.cwd()), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--run":
        sys.exit(stage_run(sys.argv[2], sys.argv[3], sys.argv[4]))
    try:
        main()
    except Exception:  # the judge itself broke: lethe retries and does not charge the candidate
        traceback.print_exc()
        sys.exit(75)
