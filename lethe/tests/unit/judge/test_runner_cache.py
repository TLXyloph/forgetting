"""JudgeRunner caching and concurrency: memoized results, in-flight de-duplication, slot limits."""

from __future__ import annotations

import asyncio
from pathlib import Path

from lethe.judge.runner import JudgeRunner

COUNTING_JUDGE = """
    import os
    with open(os.environ["COUNTER"], "a") as handle:
        handle.write("run\\n")
    print('{"score": 5, "feedback": "counted"}')
"""

SLOW_COUNTING_JUDGE = """
    import os, time
    with open(os.environ["COUNTER"], "a") as handle:
        handle.write("run\\n")
    time.sleep(0.5)
    print('{"score": 5, "feedback": "slow"}')
"""

INTERVAL_JUDGE = """
    import os, time
    start = time.time()
    time.sleep(0.2)
    end = time.time()
    name = os.environ["LETHE_CAND_ID"] + ".txt"
    with open(os.path.join(os.environ["TRACE_DIR"], name), "w") as handle:
        handle.write(f"{start} {end}")
    print('{"score": 1}')
"""


# --- cache ---------------------------------------------------------------------------------------


def test_cache_hit_returns_the_stored_result_without_running_anything(
    tmp_path, make_runner, judge_tree, evaluate, count_runs
):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)})
    first_tree, second_tree = judge_tree(COUNTING_JUDGE), judge_tree(COUNTING_JUDGE)

    first = evaluate(runner, first_tree, cand_id="c0001", key="same")
    second = evaluate(runner, second_tree, cand_id="c0002", key="same")

    assert count_runs(counter) == 1
    assert second_tree.trees == []  # not even materialized
    assert first.cache_hit is False and first.duration_s > 0
    assert second.cache_hit is True and second.duration_s == 0.0
    assert (second.valid, second.score, second.feedback) == (True, 5.0, "counted")
    assert second.metrics == first.metrics and second.cases == first.cases


def test_different_keys_run_separately(tmp_path, make_runner, judge_tree, evaluate, count_runs):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)})
    evaluate(runner, judge_tree(COUNTING_JUDGE), cand_id="c0001", key="k1")
    result = evaluate(runner, judge_tree(COUNTING_JUDGE), cand_id="c0002", key="k2")
    assert result.cache_hit is False
    assert count_runs(counter) == 2


def test_invalid_verdicts_are_cached_too(tmp_path, make_runner, judge_tree, evaluate, count_runs):
    source = """
        import os
        with open(os.environ["COUNTER"], "a") as handle:
            handle.write("run\\n")
        print('{"valid": false, "feedback": "wrong answer"}')
    """
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)})
    first = evaluate(runner, judge_tree(source))
    again = evaluate(runner, judge_tree(source), cand_id="c0002")
    assert count_runs(counter) == 1
    assert again.cache_hit is True
    assert (again.valid, again.score, again.feedback) == (False, None, "wrong answer")
    assert first.cache_hit is False


def test_cached_results_are_isolated_from_callers_that_mutate_them(
    tmp_path, make_runner, judge_tree, evaluate
):
    source = """
        print('{"score": 1, "metrics": {"n": 1}, "cases": {"a": 1}}')
    """
    runner = make_runner()
    first = evaluate(runner, judge_tree(source))
    first.metrics["n"] = 999  # a careless caller
    first.cases["a"] = 999.0

    second = evaluate(runner, judge_tree(source), cand_id="c0002")
    assert (second.metrics, second.cases) == ({"n": 1}, {"a": 1.0})
    second.metrics["n"] = -1
    second.cases["a"] = -1.0

    third = evaluate(runner, judge_tree(source), cand_id="c0003")
    assert (third.metrics, third.cases) == ({"n": 1}, {"a": 1.0})


def test_cache_can_be_disabled(tmp_path, make_runner, judge_tree, evaluate, count_runs):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)}, cache=False)
    first = evaluate(runner, judge_tree(COUNTING_JUDGE), key="same")
    second = evaluate(runner, judge_tree(COUNTING_JUDGE), key="same")
    assert (first.cache_hit, second.cache_hit) == (False, False)
    assert count_runs(counter) == 2


def test_concurrent_evaluations_of_one_key_run_the_judge_once(
    tmp_path, make_runner, judge_tree, count_runs
):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)}, concurrency=4)
    materializers = [judge_tree(SLOW_COUNTING_JUDGE) for _ in range(3)]

    async def main():
        return await asyncio.gather(
            *(
                runner.evaluate(cand_id=f"c{i}", judged_key="same", materialize=m)
                for i, m in enumerate(materializers)
            )
        )

    results = asyncio.run(main())

    assert count_runs(counter) == 1
    assert sum(len(m.trees) for m in materializers) == 1
    assert sorted(r.cache_hit for r in results) == [False, True, True]
    assert {(r.valid, r.score, r.feedback) for r in results} == {(True, 5.0, "slow")}
    assert all(r.duration_s == 0.0 for r in results if r.cache_hit)
    assert all(r.duration_s >= 0.5 for r in results if not r.cache_hit)


def test_concurrent_evaluations_run_every_time_when_caching_is_off(
    tmp_path, make_runner, judge_tree, count_runs
):
    counter = tmp_path / "count.txt"
    runner = make_runner(env={"COUNTER": str(counter)}, concurrency=2, cache=False)

    async def main():
        return await asyncio.gather(
            runner.evaluate(cand_id="a", judged_key="same", materialize=judge_tree(COUNTING_JUDGE)),
            runner.evaluate(cand_id="b", judged_key="same", materialize=judge_tree(COUNTING_JUDGE)),
        )

    results = asyncio.run(main())
    assert count_runs(counter) == 2
    assert [r.cache_hit for r in results] == [False, False]


# --- concurrency limit -----------------------------------------------------------------------


def _intervals(trace_dir: Path) -> list[tuple[float, float]]:
    spans = []
    for path in sorted(trace_dir.glob("*.txt")):
        start, end = path.read_text().split()
        spans.append((float(start), float(end)))
    return sorted(spans)


def _run_judges(runner: JudgeRunner, judge_tree, count: int) -> None:
    async def main():
        await asyncio.gather(
            *(
                runner.evaluate(
                    cand_id=f"c{i}", judged_key=f"k{i}", materialize=judge_tree(INTERVAL_JUDGE)
                )
                for i in range(count)
            )
        )

    asyncio.run(main())


def test_concurrency_one_serializes_evaluations(tmp_path, make_runner, judge_tree):
    trace = tmp_path / "trace"
    trace.mkdir()
    runner = make_runner(env={"TRACE_DIR": str(trace)}, concurrency=1)

    _run_judges(runner, judge_tree, 3)

    spans = _intervals(trace)
    assert len(spans) == 3
    for (_, earlier_end), (later_start, _) in zip(spans, spans[1:], strict=False):
        assert later_start >= earlier_end  # no overlap


def test_concurrency_two_allows_exactly_two_at_once(tmp_path, make_runner, judge_tree):
    trace = tmp_path / "trace"
    trace.mkdir()
    runner = make_runner(env={"TRACE_DIR": str(trace)}, concurrency=2)

    _run_judges(runner, judge_tree, 4)

    spans = _intervals(trace)
    assert len(spans) == 4
    # peak overlap: how many intervals contain each interval's start
    peak = max(sum(1 for s, e in spans if s <= start < e) for start, _ in spans)
    assert peak == 2


def test_one_runner_can_be_used_across_event_loops(tmp_path, make_runner, judge_tree):
    runner = make_runner(concurrency=1)

    for round_ in ("first", "second"):

        async def main(round_=round_):
            await asyncio.gather(
                *(
                    runner.evaluate(
                        cand_id=f"{round_}{i}",
                        judged_key=f"{round_}-{i}",
                        materialize=judge_tree("print(1)"),
                    )
                    for i in range(2)
                )
            )

        asyncio.run(main())  # both calls contend for the single slot

    assert len(list((tmp_path / "work").glob("*-1"))) == 4
