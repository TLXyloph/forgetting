from __future__ import annotations

from lethe.domain import types
from lethe.policy.prompts import (
    PeerStatus,
    continue_prompt,
    first_turn_prompt,
    format_score,
    status_block,
    task_md,
)


def test_format_score_is_compact_and_handles_none():
    assert format_score(1103000.0) == "1103000"
    assert format_score(0.961) == "0.961"
    assert format_score(None) == "n/a"
    assert format_score is types.format_score  # one formatter for prompts and .context/


def test_status_block_matches_paper_shape_for_a_team():
    text = status_block(
        round_no=2,
        current_id="c0007",
        current_score=951931.0,
        current_valid=True,
        direction="max",
        peers=[PeerStatus(2, "c0009", 960000.0), PeerStatus(3, None, None)],
    )
    assert text.splitlines() == [
        "Round 2.",
        "Your current candidate is `c0007`, score 951931 (higher is better).",
        "Branch 2: best `c0009` at 960000; its pool is under `.context/peers/branch2/`.",
        "Branch 3: no valid candidate yet; its pool is under `.context/peers/branch3/`.",
    ]


def test_status_block_for_rejected_candidate_and_min_direction():
    text = status_block(
        round_no=1,
        current_id="c0003",
        current_score=None,
        current_valid=False,
        direction="min",
        peers=[],
    )
    assert "`c0003`, which the judge rejected (see `.context/feedback.md`)" in text
    assert "Branch" not in text


def test_first_turn_prompt_has_task_status_and_paper_instructions():
    prompt = first_turn_prompt(task_description="Maximize X.", status="Round 1.")
    assert prompt.startswith("# Task\n\nMaximize X.\n\n# Feedbacks and Updates\n\nRound 1.")
    assert "Do not evaluate the change yourself" in prompt
    assert "Do not commit; the harness commits your working tree" in prompt
    assert prompt.rstrip().endswith("Another process evaluates it and invokes you again.")


def test_continue_prompt_is_short_reminder():
    prompt = continue_prompt(status="Round 3.")
    assert prompt.startswith("# Feedbacks and Updates\n\nRound 3.\n\n# Continue")
    assert "TASK.md holds the full instructions" in prompt


def test_task_md_contains_description_and_instructions():
    text = task_md(task_description="Maximize X.")
    assert "Maximize X." in text and "# Instructions" in text
