"""JudgeRunner behaviour: environment, build/run, contract wiring, retry, cache, concurrency."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from lethe.judge.runner import JudgeRunner, JudgeSpec

ENV_JUDGE = """
    import json, os
    keys = ["LETHE_TASK_DIR", "LETHE_CANDIDATE", "LETHE_OUT", "LETHE_SPLIT", "LETHE_SEED",
            "LETHE_CAND_ID", "EXTRA_FLAG"]
    seen = {key: os.environ.get(key) for key in keys}
    seen["cwd"] = os.getcwd()
    with open(os.path.join(os.environ["LETHE_OUT"], "env.json"), "w") as handle:
        json.dump(seen, handle)
    print(json.dumps({"score": 0.25, "feedback": "looks fine", "metrics": {"n": 3},
                      "cases": {"a": 1}}))
"""

INFRA_JUDGE = """
    import os, sys
    path = os.environ["COUNTER"]
    with open(path, "a") as handle:
        handle.write("run\\n")
    with open(path) as handle:
        runs = len(handle.read().splitlines())
    if os.environ.get("ALWAYS_FAIL") == "1" or runs == 1:
        sys.stderr.write("scratch disk full\\n")
        sys.exit(75)
    print('{"score": 2}')
"""

# --- construction -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"concurrency": 0},
        {"concurrency": -2},
        {"timeout_s": 0},
        {"timeout_s": -1.0},
        {"timeout_s": float("nan")},
        {"run": ""},
        {"run": "   \n"},
        {"max_feedback_chars": -1},
        {"env": {"FLAG": 1}},
    ],
)
def test_invalid_spec_is_rejected_at_construction(tmp_path, overrides):
    params = {"run": "true", "direction": "max", **overrides}
    with pytest.raises(ValueError):
        JudgeRunner(JudgeSpec(**params), task_dir=tmp_path, work_root=tmp_path / "work")


def test_construction_touches_nothing_on_disk(tmp_path):
    JudgeRunner(JudgeSpec(run="true", direction="min"), task_dir=tmp_path, work_root=tmp_path / "w")
    assert not (tmp_path / "w").exists()


# --- the success path ------------------------------------------------------------------------


def test_success_reports_the_result_and_exposes_the_judge_environment(
    tmp_path, make_runner, judge_tree, evaluate
):
    runner = make_runner(split="test", env={"EXTRA_FLAG": "on", "LETHE_SPLIT": "spoofed"})
    materialize = judge_tree(ENV_JUDGE)

    result = evaluate(runner, materialize, cand_id="c0007", key="key-1")

    assert (result.valid, result.score, result.feedback) == (True, 0.25, "looks fine")
    assert result.metrics == {"n": 3}
    assert result.cases == {"a": 1.0}
    assert result.error is None
    assert result.cache_hit is False
    assert result.duration_s > 0
    (tree,) = materialize.trees
    assert tree == tmp_path / "work" / "c0007-1" / "tree"
    out = tree.parent / "out"
    seen = json.loads((out / "env.json").read_text())
    assert seen["LETHE_TASK_DIR"] == str(tmp_path / "task")
    assert seen["LETHE_CANDIDATE"] == str(tree)
    assert seen["LETHE_OUT"] == str(out)
    assert seen["LETHE_SPLIT"] == "test"  # LETHE_* wins over spec.env
    assert seen["LETHE_CAND_ID"] == "c0007"
    assert seen["EXTRA_FLAG"] == "on"
    assert 0 <= int(seen["LETHE_SEED"]) < 2**32
    assert Path(seen["cwd"]).resolve() == tree.resolve()


def test_each_evaluation_gets_a_fresh_random_seed(make_runner, judge_tree, evaluate):
    source = """
        import json, os
        print(json.dumps({"score": 1, "metrics": {"seed": int(os.environ["LETHE_SEED"])}}))
    """
    runner = make_runner(cache=False)
    seeds = {evaluate(runner, judge_tree(source)).metrics["seed"] for _ in range(3)}
    assert len(seeds) == 3
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_materialize_runs_in_a_worker_thread(make_runner, evaluate):
    threads: list[int] = []

    def materialize(tree: Path) -> None:
        threads.append(threading.get_ident())
        (tree / "judge.py").write_text("print(3)")

    result = evaluate(make_runner(), materialize)
    assert result.score == 3.0
    assert threads and threads[0] != threading.get_ident()


def test_duration_covers_the_judge_run(make_runner, judge_tree, evaluate):
    source = "import time; time.sleep(0.3); print(1)"
    result = evaluate(make_runner(), judge_tree(source))
    assert 0.3 <= result.duration_s < 10


def test_judge_stdin_is_not_inherited(make_runner, judge_tree, evaluate):
    source = """
        import os
        null = os.stat(os.devnull)
        here = os.fstat(0)
        print(1 if (here.st_rdev, here.st_mode) == (null.st_rdev, null.st_mode) else 0)
    """
    # pytest already points fd 0 at /dev/null; make it a pipe so that inheriting would show.
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        result = evaluate(make_runner(), judge_tree(source))
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)
    assert result.score == 1.0


# --- result.json and exit codes ---------------------------------------------------------------


def test_result_json_in_the_out_dir_beats_stdout(make_runner, judge_tree, evaluate):
    source = """
        import json, os
        path = os.path.join(os.environ["LETHE_OUT"], "result.json")
        json.dump({"score": 7, "feedback": "from file"}, open(path, "w"))
        print('{"score": 1}')
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert (result.valid, result.score, result.feedback) == (True, 7.0, "from file")


def test_malformed_result_json_is_invalid(make_runner, judge_tree, evaluate):
    source = """
        import os
        open(os.path.join(os.environ["LETHE_OUT"], "result.json"), "w").write("{oops")
        print('{"score": 1}')
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert result.valid is False
    assert result.score is None
    assert result.feedback.startswith("result.json is not valid JSON: ")


def test_judge_that_prints_a_bare_number(make_runner, judge_tree, evaluate):
    result = evaluate(make_runner(), judge_tree("print('building...'); print(0.875)"))
    assert (result.valid, result.score) == (True, 0.875)


def test_nonzero_exit_is_invalid_with_error_and_output_tail(make_runner, judge_tree, evaluate):
    source = """
        import sys
        print('{"score": 9, "feedback": "almost"}')
        sys.stderr.write("Traceback (most recent call last):\\nValueError: boom\\n")
        sys.exit(3)
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert result.valid is False
    assert result.score is None
    assert result.error == "judge exited with code 3"
    assert result.feedback.startswith("almost\n--- judge output (tail) ---\n")
    assert result.feedback.endswith("ValueError: boom")


def test_feedback_is_limited_by_the_spec(make_runner, judge_tree, evaluate):
    source = "import json; print(json.dumps({'score': 1, 'feedback': 'f' * 1000}))"
    result = evaluate(make_runner(max_feedback_chars=40), judge_tree(source))
    assert result.feedback == "f" * 40 + "\n…[truncated]"


# --- build step -------------------------------------------------------------------------------


def test_build_runs_before_the_judge_in_the_same_tree(make_runner, judge_tree, py, evaluate):
    build = """
        import os
        with open("built.txt", "w") as handle:
            handle.write("artifact for " + os.environ["LETHE_CAND_ID"])
    """
    source = """
        import json
        print(json.dumps({"score": 1, "feedback": open("built.txt").read()}))
    """
    runner = make_runner(build=py("build.py"))
    result = evaluate(runner, judge_tree(source, **{"build.py": build}), cand_id="c0042")
    assert (result.valid, result.feedback) == (True, "artifact for c0042")


def test_failed_build_is_invalid_and_the_judge_never_runs(
    tmp_path, make_runner, judge_tree, py, evaluate
):
    build = """
        import sys
        sys.stderr.write("error[E0308]: mismatched types\\n")
        sys.exit(3)
    """
    source = "import os; open(os.environ['RAN'], 'w').write('ran'); print(1)"
    runner = make_runner(build=py("build.py"), env={"RAN": str(tmp_path / "ran")})

    result = evaluate(runner, judge_tree(source, **{"build.py": build}))

    assert result.valid is False
    assert result.score is None
    assert result.error == "build failed (exit 3)"
    assert "error[E0308]: mismatched types" in result.feedback
    assert not (tmp_path / "ran").exists()


def test_build_exit_75_is_a_build_failure_and_is_not_retried(
    tmp_path, make_runner, judge_tree, py, evaluate, count_runs
):
    build = """
        import os, sys
        with open(os.environ["COUNTER"], "a") as handle:
            handle.write("run\\n")
        sys.exit(75)
    """
    counter = tmp_path / "count.txt"
    runner = make_runner(build=py("build.py"), env={"COUNTER": str(counter)})
    result = evaluate(runner, judge_tree("print(1)", **{"build.py": build}))
    assert result.valid is False
    assert result.error == "build failed (exit 75)"
    assert count_runs(counter) == 1  # only the judge's own exit 75 means "infrastructure"


def test_build_output_is_tail_truncated(make_runner, judge_tree, py, evaluate):
    build = """
        import sys
        sys.stderr.write("noise " * 5000 + "BUILD END")
        sys.exit(1)
    """
    runner = make_runner(build=py("build.py"), max_feedback_chars=200)
    result = evaluate(runner, judge_tree("print(1)", **{"build.py": build}))
    assert result.error == "build failed (exit 1)"
    assert result.feedback.endswith("BUILD END")
    assert len(result.feedback) <= 200 + len("\n…[truncated]")


def test_blank_build_command_is_skipped(make_runner, judge_tree, evaluate):
    result = evaluate(make_runner(build="   "), judge_tree("print(2)"))
    assert result.score == 2.0


# --- infrastructure failures (exit 75) ------------------------------------------------------


def test_exit_75_is_retried_once_in_a_fresh_attempt_dir(
    tmp_path, make_runner, judge_tree, evaluate, count_runs
):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)})
    materialize = judge_tree(INFRA_JUDGE)

    result = evaluate(runner, materialize)

    assert (result.valid, result.score, result.error) == (True, 2.0, None)
    assert count_runs(counter) == 2
    work = tmp_path / "work"
    assert materialize.trees == [work / "c0001-1" / "tree", work / "c0001-2" / "tree"]
    for attempt in ("c0001-1", "c0001-2"):
        assert (work / attempt / "out").is_dir()
        assert not (work / attempt / "tree").exists()


def test_exit_75_twice_is_invalid_with_an_infrastructure_error(
    tmp_path, make_runner, judge_tree, evaluate, count_runs
):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter), "ALWAYS_FAIL": "1"})

    result = evaluate(runner, judge_tree(INFRA_JUDGE))

    assert count_runs(counter) == 2  # exactly one retry
    assert result.valid is False
    assert result.score is None
    assert result.error == "judge infrastructure failure (exit 75)"
    assert "scratch disk full" in result.feedback
    assert result.duration_s > 0

    evaluate(runner, judge_tree(INFRA_JUDGE))  # infrastructure failures are never cached
    assert count_runs(counter) == 4


# --- materialize failures and directories -------------------------------------------------------


def test_materialize_exception_becomes_an_invalid_result(
    tmp_path, make_runner, judge_tree, evaluate
):
    def broken(tree: Path) -> None:
        (tree / "half-written.txt").write_text("x")
        raise RuntimeError("disk exploded")

    runner = make_runner()
    result = evaluate(runner, broken)

    assert result.valid is False
    assert result.score is None
    assert result.error == "could not materialize candidate: disk exploded"
    assert "disk exploded" in result.feedback
    assert not (tmp_path / "work" / "c0001-1" / "tree").exists()
    assert (tmp_path / "work" / "c0001-1" / "out").is_dir()

    retry = evaluate(runner, judge_tree("print(4)"))  # failures are not cached
    assert (retry.valid, retry.score, retry.cache_hit) == (True, 4.0, False)


def test_tree_is_removed_and_out_is_kept(tmp_path, make_runner, judge_tree, evaluate):
    source = """
        import os
        open(os.path.join(os.environ["LETHE_OUT"], "artifact.txt"), "w").write("kept")
        open("scratch.bin", "w").write("temporary")
        print(1)
    """
    evaluate(make_runner(), judge_tree(source), cand_id="c0003")
    attempt = tmp_path / "work" / "c0003-1"
    assert not (attempt / "tree").exists()
    assert (attempt / "out" / "artifact.txt").read_text() == "kept"


def test_attempt_numbers_continue_after_existing_directories(
    tmp_path, make_runner, judge_tree, evaluate
):
    (tmp_path / "work" / "c0001-1").mkdir(parents=True)
    (tmp_path / "work" / "c0001-2").mkdir()
    runner = make_runner(cache=False)
    materialize = judge_tree("print(1)")

    evaluate(runner, materialize)
    evaluate(runner, materialize)

    work = tmp_path / "work"
    assert materialize.trees == [work / "c0001-3" / "tree", work / "c0001-4" / "tree"]


@pytest.mark.parametrize(
    "cand_id",
    ["", "a/b", "../x", "x/..", "a b", "c0001\n", "é", "c1\x00", "a;b", "$(id)", "a*", None, 7],
)
def test_bad_candidate_ids_are_rejected_before_anything_happens(
    tmp_path, make_runner, judge_tree, evaluate, cand_id
):
    materialize = judge_tree("print(1)")
    with pytest.raises(ValueError):
        evaluate(make_runner(), materialize, cand_id=cand_id)
    assert materialize.trees == []
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("cand_id", ["c0001", "seed.v2-3_x", "..", "A", "9"])
def test_good_candidate_ids_are_accepted(tmp_path, make_runner, judge_tree, evaluate, cand_id):
    result = evaluate(make_runner(), judge_tree("print(1)"), cand_id=cand_id)
    assert result.valid is True
    assert (tmp_path / "work" / f"{cand_id}-1" / "out").is_dir()
