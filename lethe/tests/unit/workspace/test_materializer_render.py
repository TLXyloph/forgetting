"""Materializer.render: the TASK.md / .context/ files an agent sees before each turn."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lethe.domain.types import CandidateCard, ContextBundle, PeerBranch
from lethe.workspace.materializer import Materializer


class NoArchive:
    """`render` must never touch the archive; any use of it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"render used the archive: {name}")


def card(
    cand_id: str,
    *,
    parent: str | None = "c0000",
    score: float | None = None,
    valid: bool | None = None,
    lines: int = 0,
    summary: str = "",
    diff: str = "",
) -> CandidateCard:
    return CandidateCard(
        cand_id=cand_id,
        parent_id=parent,
        score=score,
        valid=valid,
        lines_changed=lines,
        summary=summary,
        diff=diff,
    )


def bundle(**overrides) -> ContextBundle:
    fields = {
        "task_md": "# Task\n\nMake `solver.py` faster.\n",
        "current": card("c0003", score=0.75, valid=True),
        "feedback": "",
        "metrics": {},
        "lineage": (),
        "peers": (),
    }
    fields.update(overrides)
    return ContextBundle(**fields)


@pytest.fixture
def ws(tmp_path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def render(ws, tmp_path):
    materializer = Materializer(NoArchive(), tmp_path / "root")

    def _render(b: ContextBundle) -> Path:
        materializer.render(ws, b)
        return ws

    return _render


def files_under(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    )


# --- layout ------------------------------------------------------------------------------------


def test_render_writes_the_documented_layout(render):
    mine = (
        card("c0003", parent="c0002", score=0.75, valid=True, lines=12, diff="DIFF3\n"),
        card("c0002", parent="c0001", valid=False, lines=4, diff="DIFF2\n"),
    )
    peer1 = PeerBranch(
        branch=1,
        best=card("c0009", score=0.9, valid=True),
        best_files={"solver.py": b"print(1)\n", "src/helper.py": b"X = 1\n"},
        pool=(card("c0009", diff="P9\n"), card("c0008", diff="P8\n")),
    )
    peer3 = PeerBranch(branch=3, best=None, best_files={}, pool=())

    ws = render(bundle(lineage=mine, peers=(peer1, peer3)))

    assert files_under(ws) == [
        ".context/candidates/c0002.diff",
        ".context/candidates/c0003.diff",
        ".context/feedback.md",
        ".context/lineage.md",
        ".context/peers/branch1/best/solver.py",
        ".context/peers/branch1/best/src/helper.py",
        ".context/peers/branch1/c0008.diff",
        ".context/peers/branch1/c0009.diff",
        ".context/peers/branch1/pool.md",
        ".context/peers/branch3/pool.md",
        "TASK.md",
    ]
    assert (ws / ".context/candidates/c0003.diff").read_text() == "DIFF3\n"
    assert (ws / ".context/candidates/c0002.diff").read_text() == "DIFF2\n"
    assert (ws / ".context/peers/branch1/c0009.diff").read_text() == "P9\n"
    assert (ws / ".context/peers/branch1/best/solver.py").read_bytes() == b"print(1)\n"
    assert (ws / ".context/peers/branch1/best/src/helper.py").read_bytes() == b"X = 1\n"


def test_empty_diffs_still_get_a_file_so_the_pointers_in_the_tables_resolve(render):
    ws = render(bundle(lineage=(card("c0003", diff=""),)))

    assert (ws / ".context/candidates/c0003.diff").read_text() == ""


def test_task_md_is_the_bundles_text_verbatim(render):
    for text in ("# Task\n\nbody\n", "no trailing newline", "", "café — ✓\n"):
        ws = render(bundle(task_md=text))
        assert (ws / "TASK.md").read_bytes() == text.encode("utf-8")


def test_render_creates_the_workspace_directory_if_it_is_missing(tmp_path):
    target = tmp_path / "not" / "yet" / "there"

    Materializer(NoArchive(), tmp_path / "root").render(target, bundle())

    assert (target / "TASK.md").is_file()


# --- feedback.md -------------------------------------------------------------------------------


def test_feedback_md_for_a_valid_candidate_lists_score_feedback_and_metrics(render):
    b = bundle(
        current=card("c0003", score=0.75, valid=True),
        feedback="Good progress.\nTry caching the lookup.",
        metrics={"loss": 0.25, "runs": 3, "note": "ok"},
    )

    text = (render(b) / ".context/feedback.md").read_text()

    assert text == (
        "# Current candidate `c0003`\n\n"
        "Score: 0.75 (valid)\n\n"
        "## Judge feedback\n\n"
        "Good progress.\nTry caching the lookup.\n\n"
        "## Metrics\n\n"
        "- loss: 0.25\n- runs: 3\n- note: ok\n"
    )


def test_feedback_md_for_an_invalid_candidate(render):
    b = bundle(current=card("c0004", valid=False), feedback="SyntaxError on line 3")

    text = (render(b) / ".context/feedback.md").read_text()

    assert text == (
        "# Current candidate `c0004`\n\n"
        "Score: n/a (invalid)\n\n"
        "## Judge feedback\n\n"
        "SyntaxError on line 3\n"
    )


def test_feedback_md_for_a_candidate_not_judged_yet_has_no_metrics_section(render):
    b = bundle(current=card("c0005", valid=None), feedback="   \n", metrics={})

    text = (render(b) / ".context/feedback.md").read_text()

    assert text == (
        "# Current candidate `c0005`\n\n"
        "Score: n/a (not judged yet)\n\n"
        "## Judge feedback\n\n"
        "(none)\n"
    )


@pytest.mark.parametrize(
    ("score", "shown"),
    [(3.0, "3"), (0.123456789, "0.123456789"), (1234567.0, "1234567"), (951931.25, "951931.25")],
)
def test_scores_use_the_prompts_compact_formatting_everywhere(render, score, shown):
    """The same formatter as the prompts: large scores never collapse to `1.23457e+06`."""
    mine = card("c1", score=score, valid=True)
    peer = PeerBranch(branch=2, best=mine, best_files={}, pool=(mine,))
    ws = render(bundle(current=mine, lineage=(mine,), peers=(peer,)))
    assert f"Score: {shown} (valid)" in (ws / ".context/feedback.md").read_text()
    assert f"| c1 | c0000 | {shown} | 0 |" in (ws / ".context/lineage.md").read_text()
    pool = (ws / ".context/peers/branch2/pool.md").read_text()
    assert f"Best: `c1` at {shown}" in pool and f"| c1 | c0000 | {shown} | 0 |" in pool


def test_feedback_md_keeps_each_metric_on_one_line(render):
    b = bundle(metrics={"multi": "first\nsecond", "flag": True})

    text = (render(b) / ".context/feedback.md").read_text()

    assert "- multi: first second\n" in text
    assert "- flag: True\n" in text


def test_text_files_are_utf8_and_survive_unencodable_characters(render):
    b = bundle(
        feedback="naïve — ✓",
        lineage=(card("c1", summary="bad \ud800 surrogate", diff="d"),),
    )

    ws = render(b)

    assert "naïve — ✓" in (ws / ".context/feedback.md").read_bytes().decode("utf-8")
    assert "bad" in (ws / ".context/lineage.md").read_text(encoding="utf-8")


# --- lineage.md --------------------------------------------------------------------------------


def test_lineage_md_lists_candidates_newest_first_with_status_words(render):
    mine = (
        card("c0003", parent="c0002", score=0.75, valid=True, lines=12, summary="tuned lr"),
        card("c0002", parent="c0001", valid=False, lines=4, summary="broke the parser"),
        card("c0001", parent="c0000", valid=None, lines=0, summary=""),
    )

    text = (render(bundle(lineage=mine)) / ".context/lineage.md").read_text()

    assert text == (
        "# Your candidates this round (newest first)\n\n"
        "| id | parent | score | lines changed | summary |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| c0003 | c0002 | 0.75 | 12 | tuned lr |\n"
        "| c0002 | c0001 | invalid | 4 | broke the parser |\n"
        "| c0001 | c0000 | pending | 0 |  |\n\n"
        "Diffs: `.context/candidates/<id>.diff`\n"
    )


def test_lineage_md_shows_a_dash_for_a_missing_parent(render):
    text = (
        render(bundle(lineage=(card("c0000", parent=None, score=1.0, valid=True),)))
        / ".context/lineage.md"
    ).read_text()

    assert "| c0000 | - | 1 | 0 |  |" in text


def test_lineage_md_flattens_escapes_and_truncates_summaries(render):
    summary = "first line\nsecond | has a pipe\n" + "x" * 400
    mine = (card("c0001", score=0.5, valid=True, summary=summary),)

    text = (render(bundle(lineage=mine)) / ".context/lineage.md").read_text()

    row = next(line for line in text.splitlines() if line.startswith("| c0001 |"))
    cell = row.removeprefix("| c0001 | c0000 | 0.5 | 0 | ").removesuffix(" |")
    assert cell.startswith("first line second \\| has a pipe xxx")
    assert cell.endswith("...")
    assert len(cell.replace("\\|", "|")) <= 200


def test_lineage_md_ids_cannot_break_the_table(render):
    mine = (card("c|1\nx", parent="p|0", score=0.5, valid=True),)

    text = (render(bundle(lineage=mine)) / ".context/lineage.md").read_text()

    rows = [line for line in text.splitlines() if line.startswith("| c")]
    assert rows == ["| c\\|1 x | p\\|0 | 0.5 | 0 |  |"]


def test_lineage_md_keeps_short_summaries_whole(render):
    summary = "y" * 200
    mine = (card("c0001", summary=summary),)

    text = (render(bundle(lineage=mine)) / ".context/lineage.md").read_text()

    assert f"| {summary} |" in text


# --- peers -------------------------------------------------------------------------------------


def test_pool_md_names_the_peers_best_candidate_and_lists_its_pool(render):
    best = card("c0007", parent="c0005", score=0.9, valid=True, lines=8, summary="vectorized")
    pool = (best, card("c0006", parent="c0005", valid=False, lines=2, summary="typo"))
    peer = PeerBranch(branch=2, best=best, best_files={"solver.py": b"x\n"}, pool=pool)

    ws = render(bundle(peers=(peer,)))

    assert (ws / ".context/peers/branch2/pool.md").read_text() == (
        "# Branch 2\n\n"
        "Best: `c0007` at 0.9\n\n"
        "| id | parent | score | lines changed | summary |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| c0007 | c0005 | 0.9 | 8 | vectorized |\n"
        "| c0006 | c0005 | invalid | 2 | typo |\n\n"
        "Diffs: `.context/peers/branch2/<id>.diff`\n"
        "Best candidate files: `.context/peers/branch2/best/`\n"
    )


def test_pool_md_without_a_best_candidate_says_so(render):
    peer = PeerBranch(branch=1, best=None, best_files={}, pool=(card("c0002", valid=None),))

    ws = render(bundle(peers=(peer,)))

    text = (ws / ".context/peers/branch1/pool.md").read_text()
    assert text.startswith("# Branch 1\n\nBest: none yet\n\n| id | parent |")
    assert "| c0002 | c0000 | pending | 0 |  |" in text
    assert "best/" not in text
    assert not (ws / ".context/peers/branch1/best").exists()


def test_peer_best_files_are_written_byte_exactly(render):
    files = {"solver.py": b"a\r\nb\r\n", "weights.bin": bytes(range(256)), "empty": b""}
    peer = PeerBranch(branch=1, best=card("c1", valid=True, score=1.0), best_files=files, pool=())

    ws = render(bundle(peers=(peer,)))

    for name, data in files.items():
        assert (ws / ".context/peers/branch1/best" / name).read_bytes() == data


def test_unsafe_peer_file_paths_are_skipped_silently(render, tmp_path):
    files = {
        "../escape.txt": b"x",
        "/abs.txt": b"y",
        "a/../../b.txt": b"z",
        "..": b"w",
        "": b"v",
        "ok/nested.py": b"fine",
    }
    peer = PeerBranch(branch=1, best=card("c1", valid=True, score=1.0), best_files=files, pool=())
    before = set(tmp_path.rglob("*"))

    ws = render(bundle(peers=(peer,)))

    assert (ws / ".context/peers/branch1/best/ok/nested.py").read_bytes() == b"fine"
    created = {p for p in tmp_path.rglob("*") if p not in before and p.is_file()}
    assert created == {
        ws / "TASK.md",
        ws / ".context/feedback.md",
        ws / ".context/lineage.md",
        ws / ".context/peers/branch1/pool.md",
        ws / ".context/peers/branch1/best/ok/nested.py",
    }


# --- ids used in file names --------------------------------------------------------------------


def test_cards_with_unsafe_ids_get_no_diff_file(render, tmp_path):
    mine = (
        card("../evil", diff="e"),
        card("a/b", diff="f"),
        card("with space", diff="g"),
        card("", diff="h"),
        card("ok-1.v2_x", diff="fine"),
    )
    peer = PeerBranch(branch=1, best=None, best_files={}, pool=(card("../pool-evil", diff="p"),))

    ws = render(bundle(lineage=mine, peers=(peer,)))

    assert files_under(ws / ".context/candidates") == ["ok-1.v2_x.diff"]
    assert files_under(ws / ".context/peers/branch1") == ["pool.md"]
    assert not (tmp_path / "evil.diff").exists()
    assert not (ws / "evil.diff").exists()


# --- re-rendering ------------------------------------------------------------------------------


def test_a_second_render_removes_everything_the_first_one_wrote(render):
    first = bundle(
        task_md="old task",
        lineage=(card("c0001", diff="d1"), card("c0002", diff="d2")),
        peers=(
            PeerBranch(1, card("c9", valid=True, score=1.0), {"old.py": b"1"}, (card("c9"),)),
            PeerBranch(2, None, {}, ()),
        ),
    )
    ws = render(first)
    (ws / "solver.py").write_text("agent's own work")
    (ws / "notes").mkdir()
    (ws / "notes/mine.txt").write_text("agent's notes")

    second = bundle(task_md="new task", lineage=(card("c0003", diff="d3"),), peers=())
    render(second)

    assert files_under(ws) == [
        ".context/candidates/c0003.diff",
        ".context/feedback.md",
        ".context/lineage.md",
        "TASK.md",
        "notes/mine.txt",
        "solver.py",
    ]
    assert (ws / "TASK.md").read_text() == "new task"
    assert (ws / "solver.py").read_text() == "agent's own work"


def test_rendering_the_same_bundle_twice_gives_identical_files(render):
    b = bundle(
        feedback="f",
        metrics={"a": 1},
        lineage=(card("c1", score=1.0, valid=True, summary="s", diff="d"),),
        peers=(PeerBranch(1, card("c2", valid=True, score=2.0), {"x.py": b"x"}, (card("c2"),)),),
    )
    ws = render(b)
    snapshot = {p: (ws / p).read_bytes() for p in files_under(ws)}

    render(b)

    assert {p: (ws / p).read_bytes() for p in files_under(ws)} == snapshot


# --- hostile workspaces ------------------------------------------------------------------------


def test_a_context_symlink_is_replaced_not_followed(render, ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me")
    os.symlink(outside, ws / ".context", target_is_directory=True)

    render(bundle())

    assert not (ws / ".context").is_symlink()
    assert (ws / ".context/feedback.md").is_file()
    assert sorted(p.name for p in outside.iterdir()) == ["precious.txt"]
    assert (outside / "precious.txt").read_text() == "keep me"


def test_a_context_file_is_replaced_by_the_directory(render, ws):
    (ws / ".context").write_text("an agent made this a file")

    render(bundle())

    assert (ws / ".context").is_dir()
    assert (ws / ".context/lineage.md").is_file()


def test_a_context_directory_with_agent_content_is_wiped(render, ws):
    (ws / ".context/sub").mkdir(parents=True)
    (ws / ".context/sub/agent_note.txt").write_text("scribble")
    (ws / ".context/feedback.md").write_text("tampered")

    render(bundle(feedback="real feedback"))

    assert not (ws / ".context/sub").exists()
    assert "real feedback" in (ws / ".context/feedback.md").read_text()


def test_a_task_md_symlink_is_replaced_not_followed(render, ws, tmp_path):
    target = tmp_path / "precious.txt"
    target.write_text("keep me")
    os.symlink(target, ws / "TASK.md")

    render(bundle(task_md="the real task"))

    assert not (ws / "TASK.md").is_symlink()
    assert (ws / "TASK.md").read_text() == "the real task"
    assert target.read_text() == "keep me"


def test_a_task_md_directory_is_replaced_by_the_file(render, ws):
    (ws / "TASK.md").mkdir()
    (ws / "TASK.md/inner.txt").write_text("x")

    render(bundle(task_md="the real task"))

    assert (ws / "TASK.md").read_text() == "the real task"
