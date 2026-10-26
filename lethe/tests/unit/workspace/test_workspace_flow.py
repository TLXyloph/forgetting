"""Archive and Materializer together, in the order the orchestrator uses them (real git)."""

from __future__ import annotations

from lethe.domain.types import CandidateCard, ContextBundle, PeerBranch
from lethe.workspace.materializer import Materializer


def a_bundle() -> ContextBundle:
    current = CandidateCard("c0000", None, 1.0, True, 0, "task seed", "")
    peer_best = CandidateCard("c0007", "c0000", 2.0, True, 3, "peer idea", "PEER DIFF\n")
    return ContextBundle(
        task_md="# Task\n\nImprove `solver.py`.\n",
        current=current,
        feedback="Looks fine.",
        metrics={"score": 1.0},
        lineage=(current,),
        peers=(PeerBranch(2, peer_best, {"solver.py": b"peer solver\n"}, (peer_best,)),),
    )


def test_a_full_round_trip_never_archives_what_the_harness_rendered(
    archive, make_tree, take, tree_contents, tmp_path, seed_files
):
    seed = take(make_tree(tmp_path / "seed", seed_files), index_name="seed")
    materializer = Materializer(archive, tmp_path / "ws")

    # epoch 1: a fresh workspace, rendered context, one agent edit, one snapshot
    ws1 = materializer.prepare_epoch("agent-1", 1, seed.sha)
    materializer.render(ws1, a_bundle())
    (ws1 / "solver.py").write_text("def solve(x):\n    return x * 2\n")
    (ws1 / "scratch_notes.txt").write_text("agent scratch")
    snap = archive.snapshot(ws1, parent_sha=seed.sha, message="turn 1", index_name="agent-1")

    assert (ws1 / "TASK.md").is_file() and (ws1 / ".context/peers/branch2/best/solver.py").is_file()
    assert set(archive.files(snap.sha)) == {*seed_files, "scratch_notes.txt"}  # no TASK.md/.context
    assert snap.changed_judged is True
    assert (snap.lines_added, snap.lines_deleted, snap.files_changed) == (1, 1, 1)
    shown_to_agents = archive.diff(seed.sha, snap.sha)
    assert "a/solver.py" in shown_to_agents and "scratch_notes.txt" not in shown_to_agents

    # the judge sees the seed's support files plus the candidate's solver only
    archive.checkout_overlay(seed.sha, snap.sha, tmp_path / "judge")
    judged_tree = tree_contents(tmp_path / "judge")
    assert judged_tree["solver.py"] == b"def solve(x):\n    return x * 2\n"
    assert set(judged_tree) == set(seed_files)  # scratch_notes.txt is not judged: dropped

    # forgetting: epoch 2 starts from the best candidate with a clean slate
    ws2 = materializer.prepare_epoch("agent-1", 2, snap.sha, keep=lambda p: p in seed_files)
    assert tree_contents(ws2) == judged_tree  # scratch files dropped, best solver carried over
    assert not (ws2 / ".context").exists() and not (ws2 / "TASK.md").exists()
    assert (ws1 / "scratch_notes.txt").exists()  # the old workspace is left alone
