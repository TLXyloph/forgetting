from __future__ import annotations

from itertools import count

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    JudgeFinished,
    RunStarted,
    TurnFinished,
)
from lethe.policy.reducer import TeamPolicy, peers_of, plan_turn
from lethe.workspace.paths import PathFilter

JUDGED = PathFilter(["solver.py"])


class FakeArchive:
    def __init__(self):
        self.diffs = []

    def files(self, sha):
        return {"solver.py": f"blob-{sha}", "notes.md": f"notes-{sha}"}

    def read_file(self, sha, path):
        return f"{path}@{sha}".encode()

    def diff(self, base_sha, sha, *, judged_only=True):
        self.diffs.append((base_sha, sha, judged_only))
        return f"diff {base_sha}..{sha}"


def ids():
    counter = count(1)
    return lambda: f"session-{next(counter)}"


def setup(feed, agents=("a1", "a2", "a3")):
    feed(RunStarted("r", "t", agents, "max", {}, "0"))
    feed(CandidateCommitted("c0000", "sha0", None, "seed", 0, None, "k0", 0, 0, 0, "seed"))
    feed(JudgeFinished("c0000", True, 10.0, "seed feedback", {"m": 1}, None, 0, False, None))
    feed(EpochStarted("start", {a: "c0000" for a in agents}, {a: 1 for a in agents}, "start"))


def candidate(feed, cid, agent, session, score, parent="c0000", n=1):
    feed(
        CandidateCommitted(cid, f"sha-{cid}", parent, agent, 1, session, cid, 5, 2, 1, f"did {cid}")
    )
    feed(
        TurnFinished(agent, 1, session, n, "candidate", cid, 0.1, 0, 0, 0, 0, 0, 1, 1, 7, False, 1)
    )
    feed(JudgeFinished(cid, score is not None, score, f"fb {cid}", {}, None, 0, False, None))


def plan(state, agent, archive=None, peers="all", session="continue", new_id=None):
    team = TeamPolicy(agents=state.agents, peers=peers, session=session)
    return plan_turn(
        state,
        agent,
        team=team,
        task_description="Do the task.",
        archive=archive or FakeArchive(),
        judged=JUDGED,
        new_session_id=new_id or ids(),
    )


def test_first_turn_is_a_fresh_session_with_full_prompt(feeder):
    state, feed = feeder
    setup(feed)
    p = plan(state, "a1")
    assert p.session.resume is False and p.session.session_id == "session-1"
    assert p.turn_in_session == 1 and p.epoch == 1
    assert p.prompt.startswith("# Task\n\nDo the task.")
    assert "Branch 2: best `c0000` at 10" in p.prompt and "Branch 3:" in p.prompt
    assert p.bundle.current.cand_id == "c0000"
    assert p.bundle.feedback == "seed feedback" and dict(p.bundle.metrics) == {"m": 1}
    assert p.bundle.lineage == ()
    assert [b.branch for b in p.bundle.peers] == [2, 3]
    assert "# Task" in p.bundle.task_md


def test_later_turn_resumes_session_and_shows_lineage(feeder):
    state, feed = feeder
    setup(feed)
    candidate(feed, "c0001", "a1", "S1", 12.0)
    archive = FakeArchive()
    p = plan(state, "a1", archive=archive)
    assert p.session.resume is True and p.session.session_id == "S1"
    assert p.turn_in_session == 2
    assert p.prompt.startswith("# Feedbacks and Updates")
    assert "Your current candidate is `c0001`, score 12" in p.prompt
    card = p.bundle.lineage[0]
    assert card.cand_id == "c0001" and card.diff == "diff sha0..sha-c0001"
    assert card.lines_changed == 7 and card.valid is True


def test_peer_branch_shows_best_valid_candidate_with_judged_files_only(feeder):
    state, feed = feeder
    setup(feed)
    candidate(feed, "c0001", "a2", "S2", 11.0)
    candidate(feed, "c0002", "a2", "S2", 15.0, parent="c0001", n=2)
    candidate(feed, "c0003", "a2", "S2", None, parent="c0002", n=3)
    p = plan(state, "a1")
    branch2 = p.bundle.peers[0]
    assert branch2.best.cand_id == "c0002"
    assert dict(branch2.best_files) == {"solver.py": b"solver.py@sha-c0002"}
    assert [c.cand_id for c in branch2.pool] == ["c0003", "c0002", "c0001"]
    assert "Branch 2: best `c0002` at 15" in p.prompt


def test_no_peers_and_ring_topology(feeder):
    state, feed = feeder
    setup(feed)
    p = plan(state, "a1", peers="none")
    assert p.bundle.peers == () and "Branch" not in p.prompt
    assert peers_of("a1", TeamPolicy(("a1", "a2", "a3"), "ring", "continue")) == ["a2"]
    assert peers_of("a3", TeamPolicy(("a1", "a2", "a3"), "ring", "continue")) == ["a1"]
    assert peers_of("a2", TeamPolicy(("a1", "a2", "a3"), "all", "continue")) == ["a1", "a3"]


def test_fresh_session_mode_never_resumes(feeder):
    state, feed = feeder
    setup(feed)
    candidate(feed, "c0001", "a1", "S1", 12.0)
    p = plan(state, "a1", session="fresh")
    assert p.session.resume is False and p.prompt.startswith("# Task")
    assert p.bundle.lineage[0].cand_id == "c0001"  # the search state is still visible


def test_after_team_forget_old_candidates_are_hidden(feeder):
    state, feed = feeder
    setup(feed)
    candidate(feed, "c0001", "a1", "S1", 12.0)
    candidate(feed, "c0002", "a2", "S2", 13.0)
    feed(
        EpochStarted("team", {a: "c0002" for a in state.agents}, {a: 2 for a in state.agents}, "f")
    )
    p = plan(state, "a1")
    assert p.session.resume is False and p.epoch == 2
    assert p.bundle.current.cand_id == "c0002"
    assert p.bundle.lineage == ()
    assert all(b.pool == () for b in p.bundle.peers)
    assert "Round 2." in p.prompt
