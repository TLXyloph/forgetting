"""Shared helpers for the example-judge tests: run a judge exactly the way lethe would.

lethe materializes a judge tree (the seed with the candidate's editable files overlaid), runs
`[judge].run` through a shell with cwd = that tree and the LETHE_* variables set, and reads the last
non-empty stdout line. These helpers do the same thing without importing lethe's runner.
"""

from __future__ import annotations

import importlib.util
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

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NAMES = ["bin-packing", "word-freq", "regex-dates"]
# fmt: off
ALLOWED_KEYS = {
    "task": {"name", "description", "description_file", "seed", "editable", "exclude"},
    "judge": {"run", "direction", "timeout", "build", "concurrency", "cache", "max_feedback_chars",
              "split", "env"},
    "budget": {"usd", "proposals", "wall", "primary", "pause_at_utilization"},
}
# fmt: on


@dataclass
class JudgeRun:
    result: dict[str, Any]
    proc: subprocess.CompletedProcess[str]


def load_config(name: str) -> dict[str, Any]:
    return tomllib.loads((EXAMPLES / name / "lethe.toml").read_text(encoding="utf-8"))


_MODULES: dict[str, ModuleType] = {}


def judge_module(name: str) -> ModuleType:
    """Import an example's judge/score.py (once) to reach its generators and reference code."""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            f"judge_{name.replace('-', '_')}", EXAMPLES / name / "judge" / "score.py"
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
    timeout: float = 180.0,
) -> JudgeRun:
    """Judge the seed (plus `overlay` files) the way lethe's JudgeRunner does."""
    task_dir, cfg = EXAMPLES / name, load_config(name)
    tree, out = tmp_path / "tree", tmp_path / "out"
    shutil.copytree(
        task_dir / cfg["task"]["seed"], tree, ignore=shutil.ignore_patterns("__pycache__")
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


_SEED_RUNS: dict[str, JudgeRun] = {}


def seed_run(name: str, tmp_path_factory: pytest.TempPathFactory) -> JudgeRun:
    """Judge an example's seed once per session; several tests inspect the same judgment."""
    if name not in _SEED_RUNS:
        _SEED_RUNS[name] = run_judge(name, tmp_path_factory.mktemp(f"seed-{name}"))
    return _SEED_RUNS[name]


def contract_parse(run: JudgeRun) -> Any:
    """Parse the judge's output with lethe's real contract parser (None if not implemented yet)."""
    from lethe.judge.contract import parse_judge_output

    try:
        return parse_judge_output(
            stdout=run.proc.stdout, stderr=run.proc.stderr, exit_code=run.proc.returncode,
            result_json=None,
        )  # fmt: skip
    except NotImplementedError:
        return None


def assert_valid(run: JudgeRun) -> dict[str, Any]:
    res = run.result
    assert run.proc.returncode == 0, run.proc.stderr
    assert res["valid"] is True, res
    assert isinstance(res["score"], (int, float)) and math.isfinite(res["score"])
    assert isinstance(res["feedback"], str) and res["feedback"]
    assert all(math.isfinite(v) for v in (res.get("cases") or {}).values())
    assert all(isinstance(v, (int, float, str, bool)) for v in (res.get("metrics") or {}).values())
    if (parsed := contract_parse(run)) is not None:
        assert parsed.valid is True and parsed.score == pytest.approx(res["score"])
        assert parsed.feedback == res["feedback"] and dict(parsed.metrics) == res["metrics"]
        assert (dict(parsed.cases) if parsed.cases else None) == res.get("cases")
    return res


def assert_invalid(run: JudgeRun, *needles: str) -> dict[str, Any]:
    res = run.result
    assert run.proc.returncode == 0, run.proc.stderr
    assert res["valid"] is False and res.get("score") is None and res["feedback"], res
    for needle in needles:
        assert needle.lower() in res["feedback"].lower(), res["feedback"]
    if (parsed := contract_parse(run)) is not None:
        assert parsed.valid is False and parsed.score is None and parsed.feedback == res["feedback"]
    return res
