"""Run the research tasks' judges the way lethe would (seed copied to a tree, cwd = tree, LETHE_*
variables set, `[judge].build` then `[judge].run` through a shell), plus broken candidates.

toy-mis always runs. vliw runs only when its vendor clone exists (research/scripts/fetch_vliw.sh);
ahc014 runs against the real tools only when they are built (research/scripts/fetch_ahc014.sh), but
its judge logic is also exercised with a fake `vis`, so it is covered without Rust. The helpers
live in task_judges_support.py.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from task_judges_support import (  # noqa: E402
    ALLOWED_KEYS,
    MIN_DEGREE,
    MIS_TEMPLATE,
    NAMES,
    RESEARCH,
    TASKS,
    assert_invalid,
    assert_valid,
    cpp,
    judge_module,
    load_config,
    make_fake_tools,
    needs_cxx,
    needs_real_ahc,
    needs_vliw,
    patch_build,
    run_judge,
    vliw_candidate,
)

from lethe.workspace.paths import PathFilter  # noqa: E402

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
