"""The judge boundary: what a judge, and the candidate code it runs, can see and influence.

Regression for a verified exploit: a candidate wrote `../out/result.json` (the judge tree's
sibling out dir) and that file overrode the judge's honest verdict on stdout. Now stdout mode
never reads a file, file mode reads only a private out dir away from the tree, and the judge's
environment is an allowlist (no API keys or cloud credentials reach candidate code).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from lethe.judge.runner import JudgeRunner, JudgeSpec

HONEST_JUDGE = """
    import json, os, subprocess, sys
    subprocess.run([sys.executable, "candidate.py"], check=False)
    saw_out = os.environ.get("LETHE_OUT") is not None
    print(json.dumps({"score": 1.0, "feedback": "honest", "metrics": {"saw_out": saw_out}}))
"""

FORGER = """
    import json, os
    forged = json.dumps({"score": 1000000.0, "feedback": "hacked"})
    here = os.getcwd()
    targets = [here, os.path.join(here, ".."), os.path.join(here, "..", "out")]
    if os.environ.get("LETHE_OUT"):
        targets.append(os.environ["LETHE_OUT"])
    for target in targets:
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "result.json"), "w") as handle:
            handle.write(forged)
"""

FILE_JUDGE = """
    import json, os, subprocess, sys
    subprocess.run([sys.executable, "candidate.py"], check=False)
    out = os.environ["LETHE_OUT"]
    metrics = {"out": out, "tree": os.getcwd()}
    with open(os.path.join(out, "result.json"), "w") as handle:
        json.dump({"score": 2.0, "feedback": "from the judge", "metrics": metrics}, handle)
"""

REPORTING_JUDGE = """
    import json, os
    metrics = {"tree": os.getcwd(), "candidate": os.environ["LETHE_CANDIDATE"]}
    print(json.dumps({"score": 1, "metrics": metrics}))
"""

ENV_DUMP = """
    import json, os
    print(json.dumps({"score": 1, "feedback": " ".join(sorted(os.environ))}))
"""

SECRETS = ("ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "LETHE_LEAK", "PYTHONPATH")
INHERITED = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TEMP", "TMP", "TZ"}
LETHE_VARS = {"LETHE_TASK_DIR", "LETHE_CANDIDATE", "LETHE_SPLIT", "LETHE_SEED", "LETHE_CAND_ID"}
# What `sh -c` itself exports to its children, and what macOS adds to every process.
SHELL_ADDED = {"PWD", "OLDPWD", "SHLVL", "_", "__CF_USER_TEXT_ENCODING"}


def test_a_candidate_cannot_set_its_own_score_in_stdout_mode(make_runner, judge_tree, evaluate):
    result = evaluate(make_runner(), judge_tree(HONEST_JUDGE, **{"candidate.py": FORGER}))

    assert (result.valid, result.score, result.feedback) == (True, 1.0, "honest")
    assert result.metrics == {"saw_out": False}  # stdout mode sets no LETHE_OUT at all


def test_file_mode_reads_only_a_private_out_dir_away_from_the_tree(
    make_runner, judge_tree, evaluate
):
    materialize = judge_tree(FILE_JUDGE, **{"candidate.py": FORGER})

    result = evaluate(make_runner(output="file"), materialize)

    assert (result.valid, result.score, result.feedback) == (True, 2.0, "from the judge")
    out, tree = Path(result.metrics["out"]).resolve(), Path(result.metrics["tree"]).resolve()
    assert out.name.startswith("lethe-out-")
    assert tree.parent not in (out, out.parent, *out.parents)  # neither inside nor beside
    assert out not in tree.parents and out.parent != tree.parent
    assert not out.exists() and not tree.parent.exists()  # both removed after the evaluation


def test_file_mode_without_a_result_file_is_invalid(make_runner, judge_tree, evaluate):
    result = evaluate(make_runner(output="file"), judge_tree("print('{\"score\": 5}')"))
    assert (result.valid, result.score) == (False, None)
    assert "result.json" in result.feedback


def test_file_mode_still_reports_a_crash_with_its_output(make_runner, judge_tree, evaluate):
    source = "import sys; sys.stderr.write('judge blew up'); sys.exit(2)"
    result = evaluate(make_runner(output="file"), judge_tree(source))
    assert result.error == "judge exited with code 2" and "judge blew up" in result.feedback


def test_each_attempt_gets_a_fresh_tree_that_is_removed_afterwards(
    tmp_path, make_runner, judge_tree, evaluate
):
    work = tmp_path / "work"
    work.mkdir()
    (work / "keep.txt").write_text("not the runner's")
    runner = make_runner(cache=False)

    results = [evaluate(runner, judge_tree(REPORTING_JUDGE)) for _ in range(2)]

    trees = [Path(r.metrics["tree"]).resolve() for r in results]
    assert trees[0] != trees[1]
    for tree, result in zip(trees, results, strict=True):
        assert tree.name == "tree" and tree.parent.name.startswith("lethe-judge-")
        assert tree.parent.parent == work.resolve()
        assert Path(result.metrics["candidate"]).resolve() == tree
    assert [p.name for p in work.iterdir()] == ["keep.txt"]


def test_without_a_work_root_trees_live_in_the_system_temp_dir(tmp_path, judge_tree, py):
    runner = JudgeRunner(JudgeSpec(run=py("judge.py"), direction="max"), task_dir=tmp_path)

    result = asyncio.run(
        runner.evaluate(cand_id="c1", judged_key="k", materialize=judge_tree(REPORTING_JUDGE))
    )

    tree = Path(result.metrics["tree"]).resolve()
    assert tree.parent.name.startswith("lethe-judge-")
    assert tree.parent.parent == Path(tempfile.gettempdir()).resolve()
    assert not tree.parent.exists()


def test_the_judge_environment_is_an_allowlist(monkeypatch, make_runner, judge_tree, evaluate):
    for name in SECRETS:
        monkeypatch.setenv(name, "secret-value-123")
    for name, value in (("LANG", "C.UTF-8"), ("LC_ALL", "C"), ("LC_CTYPE", "C"), ("TZ", "UTC")):
        monkeypatch.setenv(name, value)

    result = evaluate(make_runner(env={"JUDGE_FLAG": "1"}), judge_tree(ENV_DUMP))

    seen = set(result.feedback.split())
    assert not seen & set(SECRETS)
    assert {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "JUDGE_FLAG"} | LETHE_VARS <= seen
    unexpected = seen - INHERITED - LETHE_VARS - SHELL_ADDED - {"JUDGE_FLAG", "SYSTEMROOT"}
    assert {name for name in unexpected if not name.startswith("LC_")} == set()
