"""What candidate code can see and influence inside the research judges.

Candidates get a minimal environment (no LETHE_* variables, no credentials) and never the hidden
judge's path. The VLIW judge builds the kernel in a throwaway copy of the judge tree and loads the
simulator only from the pristine tree, after checking its sha256 against the pinned take-home: a
verified exploit had the candidate rewrite `tests/frozen_problem.py` and score 1 cycle.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_judges_support import (  # noqa: E402
    MIN_DEGREE,
    TASKS,
    VLIW_VENDOR,
    assert_valid,
    judge_module,
    make_fake_tools,
    needs_cxx,
    needs_vliw,
    patch_build,
    run_judge,
    vliw_candidate,
)

SECRETS = {"ANTHROPIC_API_KEY": "sk-test-secret", "AWS_SECRET_ACCESS_KEY": "aws-test-secret"}
# Raises if the judge leaks its path, its data files, LETHE_* variables or secrets to the code.
NOSY = """
import os as _os, sys as _sys
_seen = " ".join(_sys.argv) + " " + " ".join(_os.environ)
for _leak in ("score.py", "LETHE_", "ANTHROPIC_API_KEY", "AWS_SECRET"):
    assert _leak not in _seen, _leak
"""
# The reviewer's exploit: at import, append a Machine.run patch to the simulator stage B loads.
REWRITE_SIMULATOR = """
import os as _os
_p = _os.path.join(_os.getcwd(), "tests", "frozen_problem.py")
with open(_p, "a") as _f:
    _f.write("\\n_orig_run = Machine.run\\n"
             "def _fast_run(self, *a, **k):\\n"
             "    r = _orig_run(self, *a, **k)\\n"
             "    self.cycle = 1\\n"
             "    return r\\n"
             "Machine.run = _fast_run\\n")
"""


@needs_vliw
def test_vliw_pins_the_simulator_of_the_fetched_take_home() -> None:
    frozen = (VLIW_VENDOR / "tests" / "frozen_problem.py").read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == judge_module("vliw").FROZEN_SHA256


@needs_vliw
def test_vliw_candidate_cannot_rewrite_the_simulator_that_measures_it(tmp_path: Path) -> None:
    run = run_judge("vliw", tmp_path, vliw_candidate(REWRITE_SIMULATOR))
    res = assert_valid(run)
    assert res["score"] == 147734 and res["metrics"]["cycles"] == 147734
    assert "_fast_run" not in (tmp_path / "tree" / "tests" / "frozen_problem.py").read_text()


@needs_vliw
def test_vliw_refuses_a_simulator_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    shutil.copytree(VLIW_VENDOR, tree, ignore=shutil.ignore_patterns(".git"))
    with (tree / "tests" / "frozen_problem.py").open("a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    result = judge_module("vliw").judge(tree)
    assert result["valid"] is False
    assert (
        "not the pinned take-home" in result["feedback"] and "fetch_vliw.sh" in result["feedback"]
    )


@needs_vliw
def test_vliw_build_sees_no_judge_paths_variables_or_secrets(tmp_path: Path) -> None:
    nosy = patch_build("pass") + NOSY
    assert_valid(run_judge("vliw", tmp_path, vliw_candidate(nosy), extra_env=SECRETS))


def test_toy_mis_solver_sees_no_judge_variables_or_secrets(tmp_path: Path) -> None:
    code = NOSY + MIN_DEGREE
    res = assert_valid(run_judge("toy-mis", tmp_path, {"solver.py": code}, extra_env=SECRETS))
    assert res["score"] == 100.0


@needs_cxx
def test_ahc014_program_sees_no_judge_variables_or_secrets(tmp_path: Path) -> None:
    env = make_fake_tools(tmp_path / "vendor") | SECRETS
    code = (
        "#include <cstdio>\n#include <cstdlib>\n\nint main() {\n"
        '  const char *leaks[] = {"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "LETHE_SEED",\n'
        '                         "LETHE_TASK_DIR", "AHC014_TOOLS_DIR"};\n'
        "  for (const char *name : leaks) if (std::getenv(name)) return 4;\n"
        '  std::printf("0\\n");\n  return 0;\n}\n'
    )
    res = assert_valid(run_judge("ahc014", tmp_path, {"main.cpp": code}, extra_env=env))
    assert res["metrics"]["failures"] == 0


def test_the_vliw_build_runner_is_inline() -> None:
    text = (TASKS / "vliw" / "judge" / "score.py").read_text(encoding="utf-8")
    assert '"-s", "-B", "-c", STAGE_A' in text and '"--build"' not in text
    assert 'env["PYTHONHASHSEED"] = "0"' in text  # deterministic scores (no -I, which drops it)
