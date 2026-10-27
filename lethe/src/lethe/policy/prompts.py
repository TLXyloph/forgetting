"""Agent-facing text, adapted from the paper's Appendix B prompts.

The prompts deliberately describe only the task and the loop: they give the agent no search
strategy. The one structural change from the paper is that candidates are shown as files under
`.context/` (diffs and full copies) instead of git commits, because agents work in a workspace
with no repository and, by default, without a shell.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lethe.domain.types import Direction, format_score

__all__ = [
    "CONTINUE",
    "INSTRUCTIONS",
    "PeerStatus",
    "continue_prompt",
    "first_turn_prompt",
    "format_score",
    "status_block",
    "task_md",
]

INSTRUCTIONS = """\
`.context/` holds what was prepared for this turn: the candidate's feedback, its lineage, other \
candidates, or the whole run. Read it before changing anything. Every candidate is a snapshot \
the harness took of a working tree: `.context/candidates/<id>.diff` shows how one of your \
candidates changed its parent, and `.context/peers/branch<i>/` (when present) holds another \
agent's pool, its diffs, and a full copy of its best candidate under `best/`.
Improve the candidate, then end the session and say what you changed. Do not evaluate the \
change yourself: another process scores your candidate after you stop and invokes you again \
with the result, so a simulator or judge built here only spends this session. Do not commit; \
the harness commits your working tree, scores it, and continues from it in a later turn. Work \
inside this directory only: reads and writes outside it are refused, and there is no need for \
background tasks. Anything you leave here (tests, scripts) is committed with your change and \
is there next turn.

Once your candidate is in place, end the session. Another process evaluates it and invokes \
you again."""

CONTINUE = """\
The judge's feedback and the run's state are in `.context/` as before; TASK.md holds the full \
instructions. Improve the candidate, then end the session. Another process scores it and \
invokes you again."""


@dataclass(frozen=True)
class PeerStatus:
    branch: int
    best_id: str | None
    best_score: float | None


def status_block(
    *,
    round_no: int,
    current_id: str,
    current_score: float | None,
    current_valid: bool | None,
    direction: Direction,
    peers: Sequence[PeerStatus],
) -> str:
    better = "higher" if direction == "max" else "lower"
    lines = [f"Round {round_no}."]
    if current_valid and current_score is not None:
        lines.append(
            f"Your current candidate is `{current_id}`, score {format_score(current_score)} "
            f"({better} is better)."
        )
    else:
        lines.append(
            f"Your current candidate is `{current_id}`, which the judge rejected "
            f"(see `.context/feedback.md`); {better} scores are better."
        )
    for peer in peers:
        pool = f"its pool is under `.context/peers/branch{peer.branch}/`."
        if peer.best_id is None:
            lines.append(f"Branch {peer.branch}: no valid candidate yet; {pool}")
        else:
            lines.append(
                f"Branch {peer.branch}: best `{peer.best_id}` at "
                f"{format_score(peer.best_score)}; {pool}"
            )
    return "\n".join(lines)


def first_turn_prompt(*, task_description: str, status: str) -> str:
    return (
        f"# Task\n\n{task_description.strip()}\n\n"
        f"# Feedbacks and Updates\n\n{status}\n\n"
        f"# Instructions\n\n{INSTRUCTIONS}\n"
    )


def continue_prompt(*, status: str) -> str:
    return f"# Feedbacks and Updates\n\n{status}\n\n# Continue\n\n{CONTINUE}\n"


def task_md(*, task_description: str) -> str:
    return f"# Task\n\n{task_description.strip()}\n\n# Instructions\n\n{INSTRUCTIONS}\n"
