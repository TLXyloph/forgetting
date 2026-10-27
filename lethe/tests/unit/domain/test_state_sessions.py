from __future__ import annotations

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    JudgeFinished,
    RunStarted,
    TurnFinished,
)


def start(feed):
    feed(RunStarted("r", "t", ("a1",), "max", {}, "0"))
    feed(CandidateCommitted("c0000", "s0", None, "seed", 0, None, "k0", 0, 0, 0, "seed"))
    feed(JudgeFinished("c0000", True, 1.0, "", {}, None, 0.0, False, None))
    feed(EpochStarted("start", {"a1": "c0000"}, {"a1": 1}, "start"))


def turn(feed, n, *, status="empty", error_kind=None, session="s1"):
    feed(
        TurnFinished(
            "a1",
            1,
            session,
            n,
            status,
            None,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            False,
            1.0,
            error="boom" if error_kind else None,
            error_kind=error_kind,
        )
    )


def test_one_transient_error_keeps_the_session(feeder):
    state, feed = feeder
    start(feed)
    turn(feed, 1)
    turn(feed, 2, status="error", error_kind="api")
    assert state.current_session("a1") == "s1"


def test_two_consecutive_transient_errors_start_a_fresh_session(feeder):
    state, feed = feeder
    start(feed)
    turn(feed, 1)
    turn(feed, 2, status="error", error_kind="api")
    turn(feed, 2, status="error", error_kind="timeout")
    assert state.current_session("a1") is None


def test_a_recovered_session_is_resumable_again(feeder):
    state, feed = feeder
    start(feed)
    turn(feed, 1)
    turn(feed, 2, status="error", error_kind="api")
    turn(feed, 2)
    turn(feed, 3, status="error", error_kind="api")
    assert state.current_session("a1") == "s1"
