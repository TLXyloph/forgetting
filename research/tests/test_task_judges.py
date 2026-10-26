"""Run the research tasks' judges the way lethe would (seed copied to a tree, cwd = tree, LETHE_*
variables set, `[judge].build` then `[judge].run` through a shell), plus broken candidates.

toy-mis always runs. vliw runs only when its vendor clone exists (research/scripts/fetch_vliw.sh);
ahc014 runs against the real tools only when they are built (research/scripts/fetch_ahc014.sh), but
its judge logic is also exercised with a fake `vis`, so it is covered without Rust.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lethe.workspace.paths import PathFilter

RESEARCH = Path(__file__).resolve().parents[1]
TASKS, SCRIPTS = RESEARCH / "tasks", RESEARCH / "scripts"
VLIW_VENDOR = TASKS / "vliw" / "vendor" / "original_performance_takehome"
AHC_VENDOR = TASKS / "ahc014" / "vendor"
NAMES = ["toy-mis", "vliw", "ahc014"]
# fmt: off
ALLOWED_KEYS = {
    "task": {"name", "description", "description_file", "seed", "editable", "exclude"},
    "judge": {"run", "direction", "timeout", "build", "concurrency", "cache", "max_feedback_chars",
              "split", "env"},
    "budget": {"usd", "proposals", "wall", "primary", "pause_at_utilization"},
}
# fmt: on

needs_vliw = pytest.mark.skipif(not VLIW_VENDOR.is_dir(), reason="fetch_vliw.sh not run")
needs_real_ahc = pytest.mark.skipif(
    not any(os.access(p, os.X_OK) for p in AHC_VENDOR.glob("**/target/release/vis")),
    reason="run research/scripts/fetch_ahc014.sh (needs Rust)",
)
needs_cxx = pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")


@dataclass
class JudgeRun:
    result: dict[str, Any]
    proc: subprocess.CompletedProcess[str]


def load_config(name: str) -> dict[str, Any]:
    return tomllib.loads((TASKS / name / "lethe.toml").read_text(encoding="utf-8"))


_MODULES: dict[str, ModuleType] = {}


def judge_module(name: str) -> ModuleType:
    """Import a task's judge/score.py (once) to reach its generators and reference code."""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            f"judge_{name.replace('-', '_')}", TASKS / name / "judge" / "score.py"
        )
        assert spec is not None and spec.loader is not None
        _MODULES[name] = module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return _MODULES[name]


def run_judge(
    name: str,
    tmp_path: Path,
    overlay: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> JudgeRun:
    """Judge the seed (plus `overlay` files) the way lethe's JudgeRunner does."""
    task_dir, cfg = TASKS / name, load_config(name)
    tree, out = tmp_path / "tree", tmp_path / "out"
    shutil.copytree(
        task_dir / cfg["task"]["seed"], tree, ignore=shutil.ignore_patterns("__pycache__", ".git")
    )
    out.mkdir()
    for rel, text in (overlay or {}).items():
        (tree / rel).write_text(text, encoding="utf-8")
    env = {**os.environ, **{k: str(v) for k, v in cfg["judge"].get("env", {}).items()}}
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env.update(LETHE_TASK_DIR=str(task_dir), LETHE_CANDIDATE=str(tree), LETHE_OUT=str(out))
    env.update(LETHE_SPLIT="dev", LETHE_SEED="1", LETHE_CAND_ID="c0000", **(extra_env or {}))
    kwargs: dict[str, Any] = {"shell": True, "cwd": tree, "env": env, "capture_output": True}
    if build := cfg["judge"].get("build"):
        assert subprocess.run(build, text=True, timeout=timeout, **kwargs).returncode == 0
    proc = subprocess.run(cfg["judge"]["run"], text=True, timeout=timeout, **kwargs)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"judge must print exactly one JSON line:\n{proc.stdout}\n{proc.stderr}"
    return JudgeRun(result=json.loads(lines[0]), proc=proc)


def check_contract(run: JudgeRun) -> None:
    """Cross-check with lethe's real contract parser (skipped while it is not implemented)."""
    from lethe.judge.contract import parse_judge_output

    res = run.result
    try:
        parsed = parse_judge_output(
            stdout=run.proc.stdout, stderr=run.proc.stderr, exit_code=run.proc.returncode,
            result_json=None,
        )  # fmt: skip
    except NotImplementedError:
        return
    assert parsed.valid is res["valid"] and parsed.feedback == res["feedback"]
    assert parsed.score == (pytest.approx(res["score"]) if res["valid"] else None)
    assert dict(parsed.metrics) == res["metrics"]
    assert (dict(parsed.cases) if parsed.cases else None) == res.get("cases")


def assert_valid(run: JudgeRun) -> dict[str, Any]:
    res = run.result
    assert run.proc.returncode == 0, run.proc.stderr
    assert res["valid"] is True, res
    assert isinstance(res["score"], (int, float)) and math.isfinite(res["score"])
    assert isinstance(res["feedback"], str) and res["feedback"]
    assert all(math.isfinite(v) for v in (res.get("cases") or {}).values())
    assert all(isinstance(v, (int, float, str, bool)) for v in (res.get("metrics") or {}).values())
    check_contract(run)
    return res


def assert_invalid(run: JudgeRun, *needles: str) -> dict[str, Any]:
    res = run.result
    assert run.proc.returncode == 0, run.proc.stderr
    assert res["valid"] is False and res.get("score") is None and res["feedback"], res
    for needle in needles:
        assert needle.lower() in res["feedback"].lower(), res["feedback"]
    check_contract(run)
    return res


# --- configs, layout and the vendoring rules ---


@pytest.mark.parametrize("name", NAMES)
def test_config_follows_schema_and_states_no_strategy(name: str) -> None:
    cfg = load_config(name)
    assert set(cfg) == {"task", "judge", "budget"}, "tasks set only [task], [judge], [budget]"
    for section, keys in cfg.items():
        assert set(keys) <= ALLOWED_KEYS[section], f"unknown keys in [{section}]"
    task, judge, budget = cfg["task"], cfg["judge"], cfg["budget"]
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", task["name"]) and task["name"] == name
    assert task["description"].strip() and task["editable"]
    assert judge["direction"] in ("max", "min") and "$LETHE_TASK_DIR/judge/" in judge["run"]
    assert (TASKS / name / "judge" / "score.py").is_file() and budget["usd"] > 0
    seed = TASKS / name / task["seed"]
    if seed.is_dir():
        files = [p.relative_to(seed).as_posix() for p in seed.rglob("*") if p.is_file()]
        for pattern in task["editable"]:
            assert any(PathFilter([pattern]).matches(f) for f in files), pattern
        assert not list(seed.rglob("score.py")), "judge code must not live in the seed"
    else:
        assert name == "vliw", "only vliw's seed is fetched at setup time"
    text = task["description"].lower()
    for hint in ("local search", "annealing", "hill", "genetic", "beam search", "try to ", "tip:"):
        assert hint not in text, f"description must not suggest a strategy ({hint!r})"
    assert "maximize" in text or "minimize" in text


def test_vliw_and_ahc014_configs_match_the_spec() -> None:
    vliw, ahc = load_config("vliw"), load_config("ahc014")
    assert vliw["task"]["seed"] == "vendor/original_performance_takehome"
    assert vliw["task"]["editable"] == ["perf_takehome.py"] and vliw["judge"]["direction"] == "min"
    assert "147,734" in vliw["task"]["description"]
    assert ahc["task"]["editable"] == ["main.cpp"] and ahc["judge"]["direction"] == "max"
    assert ahc["judge"]["build"] == "c++ -O2 -std=c++17 main.cpp -o main"
    assert "https://atcoder.jp/contests/ahc014/tasks/ahc014_a" in ahc["task"]["description"]


def test_vendor_directories_are_git_ignored_and_nothing_is_copied_out_of_them() -> None:
    ignored = (RESEARCH.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "research/tasks/*/vendor/" in ignored
    banned = {"perf_takehome.py", "frozen_problem.py", "submission_tests.py"}
    for root in (RESEARCH, RESEARCH.parent / "lethe"):
        for path in root.rglob("*"):
            if path.name in banned and "vendor" not in path.relative_to(root).parts:
                pytest.fail(f"{path}: the take-home must never be copied outside vendor/")


# --- toy-mis ---

MIS_REFERENCE_SIZES = [52, 49, 47, 75, 68, 56, 92, 76, 126, 99, 114, 145]
MIS_GRAPHS_DIGEST = "1385417e5bab3fd4"
MIS_TEMPLATE = """
import sys
def main():
    data = sys.stdin.read().split()
    n, m = int(data[0]), int(data[1])
    adj = [set() for _ in range(n)]
    for i in range(m):
        u, v = int(data[2 + 2 * i]), int(data[3 + 2 * i])
        adj[u].add(v)
        adj[v].add(u)
    {body}
main()
"""
MIN_DEGREE = inspect.getsource(judge_module("toy-mis").min_degree_greedy) + MIS_TEMPLATE.format(
    body="print(' '.join(map(str, min_degree_greedy(adj))))"
)


def mis(body: str) -> dict[str, str]:
    return {"solver.py": MIS_TEMPLATE.format(body=body)}


def test_toy_mis_seed_scores_below_the_reference(tmp_path: Path) -> None:
    res = assert_valid(run_judge("toy-mis", tmp_path))
    assert len(res["cases"]) == 12 and 50 < res["score"] < 100
    assert res["score"] == pytest.approx(sum(res["cases"].values()) / 12)
    assert res["metrics"]["timeouts"] == 0 and res["metrics"]["total_reference"] == 999
    assert "g01-n200" in res["feedback"]


def test_toy_mis_graphs_and_reference_are_pinned_and_correct() -> None:
    judge = judge_module("toy-mis")
    graphs = judge.build_graphs()
    assert [g.n for g in graphs] == list(judge.SIZES) and (graphs[0].n, graphs[-1].n) == (200, 800)
    for g in graphs:
        assert all(0 <= u < v < g.n for u, v in g.edges) and len(set(g.edges)) == len(g.edges)
    blob = json.dumps(
        [[g.name, g.n, g.edges] for g in graphs], sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(blob.encode()).hexdigest()[:16] == MIS_GRAPHS_DIGEST
    assert [len(judge.min_degree_greedy(g.adjacency())) for g in graphs] == MIS_REFERENCE_SIZES
    for g in graphs[:3]:
        adj = g.adjacency()
        alive, chosen = set(range(g.n)), []
        while alive:  # degrees recomputed from scratch each round: slow but obviously correct
            v = min(alive, key=lambda x: (len(adj[x] & alive), x))
            chosen.append(v)
            alive -= {v} | adj[v]
        assert chosen == judge.min_degree_greedy(adj)


def test_toy_mis_reference_algorithm_scores_100(
    tmp_path: Path,
) -> None:
    code = MIN_DEGREE.replace(
        "def main():",
        "def main():\n    import os\n"
        "    assert not any(k.startswith('LETHE_') for k in os.environ)",
    )
    res = assert_valid(run_judge("toy-mis", tmp_path, {"solver.py": code}))
    assert res["score"] == pytest.approx(100.0)
    assert all(v == pytest.approx(100.0) for v in res["cases"].values())


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("print(' '.join(str(i) for i in range(n)))", "adjacent"),
        ("print(n)", "out of range"),
        ("print('0 0')", "more than once"),
        ("print('x1')", "not a vertex id"),
        ("print('9' * 6000)", "not a vertex id"),
        ("raise RuntimeError('boom')", "RuntimeError: boom"),
        ("sys.exit(3)", "exited with code 3"),
    ],
)
def test_toy_mis_rejects_bad_answers(body: str, needle: str, tmp_path: Path) -> None:
    res = assert_invalid(run_judge("toy-mis", tmp_path, mis(body)), needle)
    assert "g01-n200" in res["feedback"], "names the graph that failed"


def test_toy_mis_timeout_zeroes_one_graph_and_empty_output_is_valid(
    tmp_path: Path,
) -> None:
    seed = (TASKS / "toy-mis" / "seed" / "solver.py").read_text(encoding="utf-8")
    sleepy = seed.replace(
        "    adjacent: list[list[int]]",
        "    if n == 200:\n        import time\n\n        time.sleep(30)\n"
        "    adjacent: list[list[int]]",
    )
    assert sleepy != seed
    res = assert_valid(run_judge("toy-mis", tmp_path / "a", {"solver.py": sleepy}))
    assert res["cases"]["g01-n200"] == 0.0 and res["metrics"]["timeouts"] == 1
    assert all(v > 0 for k, v in res["cases"].items() if k != "g01-n200")
    assert "timed out" in res["feedback"] and "g01-n200" in res["feedback"]
    assert assert_valid(run_judge("toy-mis", tmp_path / "b", mis("print()")))["score"] == 0.0


# --- vliw (needs the clone made by research/scripts/fetch_vliw.sh) ---


def vliw_candidate(extra: str) -> dict[str, str]:
    """The take-home's own perf_takehome.py plus `extra`, written only into a temporary tree."""
    base = (VLIW_VENDOR / "perf_takehome.py").read_text(encoding="utf-8")
    return {"perf_takehome.py": base + "\n" + extra}


def patch_build(body: str) -> str:
    """Code that wraps KernelBuilder.build_kernel and then runs `body` on the finished builder."""
    indented = "".join(f"    {line}\n" for line in body.splitlines())
    return (
        "_orig_build = KernelBuilder.build_kernel\n\n"
        "def _patched(self, *args, **kwargs):\n    _orig_build(self, *args, **kwargs)\n"
        f"{indented}\nKernelBuilder.build_kernel = _patched\n"
    )


STRIP_PAUSE = 'self.instrs = [i for i in self.instrs if i != {"flow": [("pause",)]}]'
DROP_STORES = 'self.instrs = [i for i in self.instrs if "store" not in i]'
TOO_MANY_SLOTS = 'self.instrs.append({"alu": [("+", 0, 0, 0)] * 13})'
NOT_JSON = 'self.instrs.append({"alu": [("+", {1, 2}, 0, 0)]})'
# fmt: off
VLIW_BAD = {  # id: (extra code, words the feedback must contain)
    "drops-stores": (patch_build(DROP_STORES), ["Incorrect"]),
    "empty-program": (patch_build("self.instrs = []"), ["Incorrect"]),
    "build-raises": (patch_build("raise RuntimeError('boom in build')"), ["boom in build"]),
    "system-exit": (patch_build("raise SystemExit(7)"), ["SystemExit"]),
    "too-many-slots": (patch_build(TOO_MANY_SLOTS), ["AssertionError", "bundle"]),
    "not-json": (patch_build(NOT_JSON), ["set"]),
    "syntax-error": ("def broken(:\n", ["SyntaxError"]),
}
# fmt: on


@needs_vliw
def test_vliw_seed_matches_the_baseline_and_a_faster_kernel_scores_lower(tmp_path: Path) -> None:
    res = assert_valid(run_judge("vliw", tmp_path / "seed"))
    assert res["score"] == 147734 == res["metrics"]["cycles"] and "147,734" in res["feedback"]
    assert res["metrics"]["speedup"] == pytest.approx(1.0)
    assert len(res["metrics"]["frozen_problem_sha"]) == 12
    faster = vliw_candidate(patch_build(STRIP_PAUSE))
    better = assert_valid(run_judge("vliw", tmp_path / "faster", faster))
    assert better["score"] < 147734 and better["metrics"]["cycles"] == better["score"]


@needs_vliw
def test_vliw_candidate_cannot_tamper_with_the_simulator_that_measures_it(tmp_path: Path) -> None:
    tamper = (  # would report 1 cycle if candidate code shared a process with the simulator
        "import problem as _problem\n_run = _problem.Machine.run\n\n"
        "def _fast(self):\n    _run(self)\n    self.cycle = 1\n\n_problem.Machine.run = _fast\n"
    )
    res = assert_valid(run_judge("vliw", tmp_path, vliw_candidate(tamper)))
    assert res["score"] == 147734, "the measuring simulator never imports candidate code"


@needs_vliw
@pytest.mark.parametrize("case", list(VLIW_BAD))
def test_vliw_rejects_wrong_or_broken_kernels(case: str, tmp_path: Path) -> None:
    extra, needles = VLIW_BAD[case]
    assert_invalid(run_judge("vliw", tmp_path, vliw_candidate(extra)), *needles)


@needs_vliw
def test_vliw_endless_program_and_slow_builder_hit_the_timeouts(tmp_path: Path) -> None:
    env = {"VLIW_BUILD_TIMEOUT_S": "3", "VLIW_RUN_TIMEOUT_S": "3"}
    loop = vliw_candidate(patch_build('self.instrs.append({"flow": [("jump", len(self.instrs))]})'))
    assert_invalid(run_judge("vliw", tmp_path / "a", loop, env), "did not finish")
    hang = vliw_candidate(patch_build("while True:\n    pass"))
    assert_invalid(run_judge("vliw", tmp_path / "b", hang, env), "build_kernel did not finish")


def test_vliw_incomplete_take_home_is_reported_with_the_fetch_hint(tmp_path: Path) -> None:
    result = judge_module("vliw").judge(tmp_path)
    assert result["valid"] is False and "fetch_vliw.sh" in result["feedback"]


# --- ahc014 ---

FAKE_VIS = """#!/bin/sh
m=$(awk 'NR==1{print $2}' "$1"); k=$(awk 'NR==1{print $1}' "$2")
case "$k" in ''|-*) printf 'Score = 0\nIllegal output\n';; *) echo "Score = $((1000*m+k))";; esac
"""


def make_fake_tools(root: Path) -> dict[str, str]:
    """A vendor dir shaped like the unpacked, built official tools with a fake vis; returns env."""
    crate = root / "tools"
    (crate / "in").mkdir(parents=True)
    for i in range(30):
        n = 31 + 2 * (i % 15)
        points = "".join(f"{n // 4 + j % 5} {n // 4 + j // 5}\n" for j in range(n + i))
        (crate / "in" / f"{i:04d}.txt").write_text(f"{n} {n + i}\n{points}")
    vis = crate / "target" / "release" / "vis"
    vis.parent.mkdir(parents=True)
    vis.write_text(FAKE_VIS)
    vis.chmod(0o755)
    return {"AHC014_TOOLS_DIR": str(root)}


def cpp(body: str) -> dict[str, str]:
    code = (
        "#include <cstdio>\n\nint main() {\n  int n = 0, m = 0;\n"
        '  if (std::scanf("%d %d", &n, &m) != 2) return 1;\n'
        f'{body}\n  std::printf("0\\n");\n  return 0;\n}}\n'
    )
    return {"main.cpp": code}


@needs_cxx
def test_ahc014_seed_is_scored_by_the_vis_binary_on_all_30_inputs(tmp_path: Path) -> None:
    env = make_fake_tools(tmp_path / "vendor")
    res = assert_valid(run_judge("ahc014", tmp_path, extra_env=env))
    assert list(res["cases"]) == [f"{i:04d}" for i in range(30)]
    for i, score in enumerate(res["cases"].values()):
        assert score == 1000 * (31 + 2 * (i % 15) + i), "vis got the input first, the output second"
    assert res["score"] == pytest.approx(sum(res["cases"].values()) / 30)
    assert res["metrics"]["failures"] == 0 and "N=" in res["feedback"]


@needs_cxx
def test_ahc014_failed_inputs_score_zero_without_invalidating_the_candidate(tmp_path: Path) -> None:
    env = make_fake_tools(tmp_path / "vendor") | {"AHC014_TIME_LIMIT_S": "1"}
    code = cpp(
        "  if (n == 41) return 3;\n  volatile int sink = 0;\n"
        "  if (n == 45) for (;;) sink = sink + 1;\n"
        '  if (n == 49) { std::printf("-1\\n"); return 0; }'
    )
    res = assert_valid(run_judge("ahc014", tmp_path, code, env))
    bad = {f"{i:04d}" for i in range(30) if 31 + 2 * (i % 15) in (41, 45, 49)}
    assert len(bad) == 6 and all(res["cases"][k] == 0 for k in bad)
    assert all(v > 0 for k, v in res["cases"].items() if k not in bad)
    assert res["metrics"]["failures"] == 6 and res["metrics"]["timeouts"] == 2
    for word in ("RE", "TLE", "WA", "Illegal output"):
        assert word in res["feedback"], res["feedback"]


@needs_cxx
def test_ahc014_all_inputs_failing_is_invalid_and_missing_tools_explained(
    tmp_path: Path,
) -> None:
    env = make_fake_tools(tmp_path / "vendor")
    assert_invalid(run_judge("ahc014", tmp_path / "a", cpp("  return 2;"), env), "all 30", "code 2")
    empty = {"AHC014_TOOLS_DIR": str(tmp_path / "empty")}
    assert_invalid(
        run_judge("ahc014", tmp_path / "b", extra_env=empty), "fetch_ahc014.sh", "rustup"
    )


@needs_cxx
@needs_real_ahc
def test_ahc014_real_tools_score_the_seed_and_reject_an_illegal_move(tmp_path: Path) -> None:
    res = assert_valid(run_judge("ahc014", tmp_path / "seed"))
    assert len(res["cases"]) == 30 and all(v > 0 for v in res["cases"].values())
    illegal = cpp('  std::printf("1\\n0 0 0 0 0 0 0 0\\n");\n  return 0;')
    assert_invalid(run_judge("ahc014", tmp_path / "illegal", illegal), "all 30 inputs")


# --- the fetch scripts (offline branches only; nothing here touches the network) ---


def run_script(script: str, env_var: str, vendor: Path, path: str | None = None) -> Any:
    env = {**os.environ, env_var: str(vendor), **({"PATH": path} if path else {})}
    return subprocess.run(
        ["bash", str(SCRIPTS / script)], env=env, capture_output=True, text=True, timeout=60
    )


def check_script(script: str) -> None:
    path = SCRIPTS / script
    assert os.access(path, os.X_OK) and path.read_text().startswith("#!/usr/bin/env bash")
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_fetch_vliw_is_idempotent_and_refuses_an_incomplete_clone(tmp_path: Path) -> None:
    check_script("fetch_vliw.sh")
    (tmp_path / "original_performance_takehome").mkdir()
    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path)
    assert done.returncode == 1 and "perf_takehome.py" in done.stderr
    (tmp_path / "original_performance_takehome" / "perf_takehome.py").write_text("")
    done = run_script("fetch_vliw.sh", "VLIW_VENDOR_DIR", tmp_path)
    assert done.returncode == 0 and "already present" in done.stdout


def test_fetch_ahc014_explains_how_to_get_rust_and_is_idempotent(tmp_path: Path) -> None:
    check_script("fetch_ahc014.sh")
    if shutil.which("cargo", path="/usr/bin:/bin"):
        pytest.skip("cargo lives in /usr/bin on this machine")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "Cargo.toml").write_text("[package]\n")
    no_cargo = run_script("fetch_ahc014.sh", "AHC014_VENDOR_DIR", tmp_path, "/usr/bin:/bin")
    assert no_cargo.returncode == 1 and "Install Rust (rustup)" in no_cargo.stderr
    vis = tmp_path / "tools" / "target" / "release" / "vis"
    vis.parent.mkdir(parents=True)
    vis.write_text("#!/bin/sh\n")
    vis.chmod(0o755)
    built = run_script("fetch_ahc014.sh", "AHC014_VENDOR_DIR", tmp_path, "/usr/bin:/bin")
    assert built.returncode == 0 and "already built" in built.stdout
