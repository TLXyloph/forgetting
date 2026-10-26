from __future__ import annotations

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RunFinished,
    RunStarted,
    TurnFinished,
)


def started(feed, agents=("a1", "a2"), direction="max"):
    feed(RunStarted("r1", "toy", tuple(agents), direction, {}, "0.1.0"))
    feed(cand("c0000", agent="seed", epoch=0, parent=None))
    feed(judged("c0000", 10.0))
    feed(EpochStarted("start", {a: "c0000" for a in agents}, {a: 1 for a in agents}, "run start"))


def cand(cid, *, agent="a1", epoch=1, parent="c0000", session="s1", added=3, deleted=1):
    return CandidateCommitted(
        cid,
        f"sha-{cid}",
        parent,
        agent,
        epoch,
        session,
        f"key-{cid}",
        added,
        deleted,
        1,
        f"summary {cid}",
    )


def judged(cid, score, valid=True):
    return JudgeFinished(cid, valid, score if valid else None, "fb", {}, None, 0.1, False, None)


def turn(
    agent="a1",
    epoch=1,
    session="s1",
    n=1,
    status="candidate",
    cid=None,
    cost=0.5,
    lines=4,
    claim=False,
):
    return TurnFinished(
        agent, epoch, session, n, status, cid, cost, 10, 100, 0, 0, 0, 1, 1, lines, claim, 1.0
    )


def test_seed_is_best_and_epoch_seed(feeder):
    state, feed = feeder
    started(feed)
    assert state.best().cand_id == "c0000"
    assert state.epoch("a1") == 1
    assert state.epoch_seed("a1").cand_id == "c0000"
    assert state.current_candidate("a1").cand_id == "c0000"
    assert state.current_session("a1") is None


def test_candidates_update_current_and_best(feeder):
    state, feed = feeder
    started(feed)
    feed(cand("c0001"))
    feed(turn(cid="c0001"))
    feed(judged("c0001", 12.0))
    assert state.current_candidate("a1").cand_id == "c0001"
    assert state.current_candidate("a2").cand_id == "c0000"
    assert state.best().cand_id == "c0001"
    assert state.candidates["c0001"].score == 12.0
    assert state.candidates["c0001"].lines_changed == 4
    assert state.current_session("a1") == "s1"
    assert state.session_turns("a1") == 1


def test_min_direction_and_invalid_candidates(feeder):
    state, feed = feeder
    started(feed, direction="min")
    feed(cand("c0001"))
    feed(judged("c0001", 0.0, valid=False))
    feed(cand("c0002", parent="c0001"))
    feed(judged("c0002", 7.0))
    assert state.best().cand_id == "c0002"
    assert state.candidates["c0001"].valid is False


def test_ties_keep_the_earlier_best(feeder):
    state, feed = feeder
    started(feed)
    feed(cand("c0001"))
    feed(judged("c0001", 10.0))
    assert state.best().cand_id == "c0000"


def test_spend_proposals_and_errors(feeder):
    state, feed = feeder
    started(feed)
    feed(turn(status="empty", cost=0.25, lines=0))
    feed(turn(n=2, status="error", cost=0.1, lines=0))
    assert state.proposals == 1
    assert abs(state.spent_usd - 0.35) < 1e-9
    assert state.current_session("a1") is None  # last turn errored -> next turn starts fresh


def test_team_forget_starts_new_epoch_and_hides_old_session(feeder):
    state, feed = feeder
    started(feed)
    feed(cand("c0001"))
    feed(turn(cid="c0001"))
    feed(judged("c0001", 15.0))
    feed(ForgetDecided("team", ("a1", "a2"), "budget_fractions@1"))
    feed(
        EpochStarted(
            "team", {"a1": "c0001", "a2": "c0001"}, {"a1": 2, "a2": 2}, "budget_fractions@1"
        )
    )
    assert state.epoch("a1") == 2
    assert state.current_candidate("a2").cand_id == "c0001"
    assert state.own_candidates("a1") == []
    assert state.current_session("a1") is None
    assert state.forget_reasons("team") == ["budget_fractions@1"]
    assert state.team_round_start_seq() > state.epoch_start_seq("a1") - 1


def test_peer_candidates_since(feeder):
    state, feed = feeder
    started(feed)
    feed(cand("c0001", agent="a2", session="s2"))
    since = state.epoch_start_seq("a1")
    assert [c.cand_id for c in state.candidates_of("a2", since_seq=since)] == ["c0001"]


def test_best_history_records_spend(feeder):
    state, feed = feeder
    started(feed)
    feed(turn(cost=1.0, cid="c0001"))
    feed(cand("c0001"))
    feed(judged("c0001", 11.0))
    entry = state.best_history[-1]
    assert entry.cand_id == "c0001" and entry.spent_usd == 1.0 and entry.proposals == 1


def test_session_turn_list_only_counts_current_session(feeder):
    state, feed = feeder
    started(feed)
    feed(turn(session="s1", n=1, lines=50))
    feed(turn(session="s1", n=2, lines=40))
    assert [t.lines_changed for t in state.session_turn_list("a1")] == [50, 40]
    assert state.next_cand_id() == "c0001"


def test_run_finished_recorded(feeder):
    state, feed = feeder
    started(feed)
    feed(RunFinished("budget", "c0000", 10.0))
    assert state.finished is not None and state.finished.reason == "budget"
