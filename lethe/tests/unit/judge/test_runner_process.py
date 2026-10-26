"""JudgeRunner process handling: timeouts, process-group kills, cancellation, output limits."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

SPAWNING_JUDGE = """
    import os, subprocess, sys, time
    child = subprocess.Popen(["sleep", "60"])
    with open(os.path.join(os.environ["LETHE_OUT"], "child.pid"), "w") as handle:
        handle.write(str(child.pid))
    time.sleep(60)
"""


def _child_pid(tree: Path) -> int:
    return int((tree.parent / "out" / "child.pid").read_text())


# --- timeouts -----------------------------------------------------------------------------------


def test_timeout_kills_the_whole_process_group(make_runner, judge_tree, evaluate, wait_until_dead):
    runner = make_runner(timeout_s=1.0)
    materialize = judge_tree(SPAWNING_JUDGE)

    started = time.monotonic()
    result = evaluate(runner, materialize)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0 + 2.0
    assert result.valid is False
    assert result.score is None
    assert result.feedback == "judge timed out after 1s"
    assert result.error == "judge timed out after 1s"
    assert 1.0 <= result.duration_s < 3.0
    (tree,) = materialize.trees
    assert wait_until_dead(_child_pid(tree)), "the judge's child process survived the timeout"
    assert not tree.exists()


def test_timeout_message_formats_the_limit_with_g(make_runner, judge_tree, evaluate):
    result = evaluate(make_runner(timeout_s=0.5), judge_tree("import time; time.sleep(30)"))
    assert result.error == "judge timed out after 0.5s"


def test_each_command_gets_its_own_timeout_budget(make_runner, judge_tree, py, evaluate):
    # build and judge take ~0.45s each: each is under the 0.8s budget, their sum is not
    build = "import time; time.sleep(0.45)"
    source = "import time; time.sleep(0.45); print(1)"
    runner = make_runner(build=py("build.py"), timeout_s=0.8)
    result = evaluate(runner, judge_tree(source, **{"build.py": build}))
    assert result.valid is True
    assert result.score == 1.0


def test_build_timeout_kills_the_build_and_never_runs_the_judge(
    tmp_path, make_runner, judge_tree, py, evaluate, wait_until_dead
):
    build = """
        import os, subprocess, time
        child = subprocess.Popen(["sleep", "60"])
        with open(os.path.join(os.environ["LETHE_OUT"], "child.pid"), "w") as handle:
            handle.write(str(child.pid))
        time.sleep(60)
    """
    source = "import os; open(os.environ['RAN'], 'w').write('ran'); print(1)"
    runner = make_runner(build=py("build.py"), timeout_s=1.0, env={"RAN": str(tmp_path / "ran")})
    materialize = judge_tree(source, **{"build.py": build})

    result = evaluate(runner, materialize)

    assert result.valid is False
    assert result.error == "judge timed out after 1s"
    assert not (tmp_path / "ran").exists()
    assert wait_until_dead(_child_pid(materialize.trees[0]))


def test_timed_out_results_are_not_cached(tmp_path, make_runner, judge_tree, evaluate):
    source = """
        import os, time
        path = os.environ["COUNTER"]
        with open(path, "a") as handle:
            handle.write("run\\n")
        if len(open(path).read().splitlines()) == 1:
            time.sleep(60)  # the first run hangs, the second is fine
        print(3)
    """
    runner = make_runner(timeout_s=0.7, env={"COUNTER": str(tmp_path / "count.txt")})

    first = evaluate(runner, judge_tree(source))
    second = evaluate(runner, judge_tree(source))

    assert first.error == "judge timed out after 0.7s"
    assert (second.valid, second.score, second.cache_hit) == (True, 3.0, False)


def test_timeout_still_returns_promptly_when_the_judge_floods_its_output(
    make_runner, judge_tree, evaluate
):
    source = """
        import sys
        chunk = "x" * 65536
        while True:
            sys.stdout.write(chunk)
            sys.stderr.write(chunk)
    """
    started = time.monotonic()
    result = evaluate(make_runner(timeout_s=0.7), judge_tree(source))
    assert time.monotonic() - started < 0.7 + 3.0
    assert result.error == "judge timed out after 0.7s"


# --- cancellation ------------------------------------------------------------------------------


def _wait_for_file(path: Path, timeout: float = 10.0) -> None:
    async def wait() -> None:
        deadline = time.monotonic() + timeout
        while not path.exists():
            assert time.monotonic() < deadline, f"{path} never appeared"
            await asyncio.sleep(0.02)

    asyncio.run(wait())


def test_cancelling_an_evaluation_kills_the_process_group_and_frees_the_slot(
    tmp_path, make_runner, judge_tree, wait_until_dead
):
    runner = make_runner(concurrency=1)
    materialize = judge_tree(SPAWNING_JUDGE)

    async def main():
        task = asyncio.create_task(
            runner.evaluate(cand_id="c0001", judged_key="slow", materialize=materialize)
        )
        pid_file = tmp_path / "work" / "c0001-1" / "out" / "child.pid"
        while not pid_file.exists():
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # the single concurrency slot must be free again: a fresh evaluation completes
        return await asyncio.wait_for(
            runner.evaluate(
                cand_id="c0002",
                judged_key="fast",
                materialize=lambda tree: (tree / "judge.py").write_text("print(6)"),
            ),
            timeout=20,
        )

    after = asyncio.run(main())

    assert after.score == 6.0
    (tree,) = materialize.trees
    assert wait_until_dead(_child_pid(tree)), "the judge's child process survived cancellation"
    assert not tree.exists()


def test_when_the_evaluating_caller_is_cancelled_a_waiting_caller_takes_over(
    tmp_path, make_runner, judge_tree, count_runs
):
    source = """
        import os, time
        with open(os.environ["COUNTER"], "a") as handle:
            handle.write("run\\n")
        time.sleep(0.6)
        print('{"score": 8}')
    """
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)}, concurrency=2)

    async def main():
        leader = asyncio.create_task(
            runner.evaluate(cand_id="a", judged_key="same", materialize=judge_tree(source))
        )
        while count_runs(counter) < 1:
            await asyncio.sleep(0.02)
        waiter = asyncio.create_task(
            runner.evaluate(cand_id="b", judged_key="same", materialize=judge_tree(source))
        )
        await asyncio.sleep(0.1)  # let the waiter attach to the leader's evaluation
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        return await asyncio.wait_for(waiter, timeout=20)

    result = asyncio.run(main())

    assert (result.valid, result.score) == (True, 8.0)
    assert result.cache_hit is False  # the waiter ran the judge itself
    assert count_runs(counter) == 2


def test_cancelling_a_waiting_caller_does_not_disturb_the_evaluation(
    tmp_path, make_runner, judge_tree, count_runs
):
    source = """
        import os, time
        with open(os.environ["COUNTER"], "a") as handle:
            handle.write("run\\n")
        time.sleep(0.6)
        print('{"score": 4}')
    """
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)}, concurrency=2)

    async def main():
        leader = asyncio.create_task(
            runner.evaluate(cand_id="a", judged_key="same", materialize=judge_tree(source))
        )
        while count_runs(counter) < 1:
            await asyncio.sleep(0.02)
        waiter = asyncio.create_task(
            runner.evaluate(cand_id="b", judged_key="same", materialize=judge_tree(source))
        )
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        return await asyncio.wait_for(leader, timeout=20)

    result = asyncio.run(main())

    assert (result.valid, result.score, result.cache_hit) == (True, 4.0, False)
    assert count_runs(counter) == 1


# --- output limits --------------------------------------------------------------------------------


def test_a_chatty_judge_is_bounded_but_its_last_line_still_counts(
    make_runner, judge_tree, evaluate
):
    source = """
        import sys
        line = "y" * 1023 + "\\n"
        for _ in range(3 * 1024):  # ~3 MiB, more than the 1 MiB capture limit
            sys.stdout.write(line)
        print('{"score": 12, "feedback": "after the flood"}')
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert (result.valid, result.score, result.feedback) == (True, 12.0, "after the flood")


def test_a_crashing_judge_reports_the_end_of_a_huge_stderr(make_runner, judge_tree, evaluate):
    source = """
        import sys
        sys.stderr.write("e" * (3 * 1024 * 1024))
        sys.stderr.write("\\nFinal failure: out of cheese\\n")
        sys.exit(1)
    """
    result = evaluate(make_runner(max_feedback_chars=300), judge_tree(source))
    assert result.error == "judge exited with code 1"
    assert result.feedback.endswith("Final failure: out of cheese")
    assert len(result.feedback) <= 300 + len("\n…[truncated]")


def test_undecodable_output_does_not_break_parsing(make_runner, judge_tree, evaluate):
    source = """
        import sys
        sys.stdout.buffer.write(b"\\xff\\xfe garbage \\x80\\n")
        sys.stdout.buffer.write(b'{"score": 2, "feedback": "caf\\xc3\\xa9"}\\n')
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert (result.valid, result.score, result.feedback) == (True, 2.0, "café")


def test_oversized_result_json_is_invalid_rather_than_read_whole(make_runner, judge_tree, evaluate):
    source = """
        import os
        path = os.path.join(os.environ["LETHE_OUT"], "result.json")
        with open(path, "w") as handle:
            handle.write('{"score": 1, "feedback": "' + "z" * (3 * 1024 * 1024) + '"}')
        print(1)
    """
    result = evaluate(make_runner(), judge_tree(source))
    assert result.valid is False
    assert result.feedback.startswith("result.json is not valid JSON")
