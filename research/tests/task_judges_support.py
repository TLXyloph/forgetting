"""Shared helpers for the research task-judge tests: run a judge exactly the way lethe would.

A judge tree is the task's seed (plus overlay files) copied to a temp dir; the judge runs through
a shell with cwd = that tree and the LETHE_* variables set, and prints one JSON line.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

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
# The reference procedure as a candidate program (scores exactly 100).
MIN_DEGREE = inspect.getsource(judge_module("toy-mis").min_degree_greedy) + MIS_TEMPLATE.format(
    body="print(' '.join(map(str, min_degree_greedy(adj))))"
)
