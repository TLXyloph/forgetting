from __future__ import annotations

import asyncio

from lethe.policy.triggers import ForgetDecision
from lethe.runtime.barrier import ForgetBarrier

DECISION = ForgetDecision("team", ("a1", "a2", "a3"), "test")


def make(commits):
    async def commit(decision):
        commits.append(decision)

    return ForgetBarrier(commit, active=("a1", "a2", "a3"))


def test_arrive_without_pending_returns_immediately():
    commits = []

    async def main():
        barrier = make(commits)
        await asyncio.wait_for(barrier.arrive("a1"), 1)

    asyncio.run(main())
    assert commits == []


def test_last_arrival_commits_once_and_releases_everyone():
    commits = []

    async def main():
        barrier = make(commits)
        assert barrier.request(DECISION) is True
        assert barrier.request(DECISION) is False  # already pending
        await asyncio.wait_for(
            asyncio.gather(barrier.arrive("a1"), barrier.arrive("a2"), barrier.arrive("a3")), 1
        )
        assert barrier.pending is None

    asyncio.run(main())
    assert commits == [DECISION]


def test_departing_agent_no_longer_blocks_the_barrier():
    commits = []

    async def main():
        barrier = make(commits)
        barrier.request(DECISION)
        waiting = asyncio.gather(barrier.arrive("a1"), barrier.arrive("a2"))
        await asyncio.sleep(0.01)
        assert commits == []
        await barrier.depart("a3")
        await asyncio.wait_for(waiting, 1)

    asyncio.run(main())
    assert commits == [DECISION]


def test_pending_forget_is_dropped_when_everyone_departs():
    commits = []

    async def main():
        barrier = make(commits)
        barrier.request(DECISION)
        for agent in ("a1", "a2", "a3"):
            await barrier.depart(agent)
        assert barrier.pending is None

    asyncio.run(main())
    assert commits == []


def test_barrier_can_be_reused_for_a_second_forget():
    commits = []

    async def main():
        barrier = make(commits)
        for _ in range(2):
            barrier.request(DECISION)
            await asyncio.wait_for(
                asyncio.gather(*(barrier.arrive(a) for a in ("a1", "a2", "a3"))), 1
            )

    asyncio.run(main())
    assert commits == [DECISION, DECISION]
