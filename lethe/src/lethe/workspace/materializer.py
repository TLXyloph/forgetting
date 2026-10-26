"""Builds and refreshes agent workspaces: the only view of the run an agent ever gets."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from lethe.domain.types import CandidateCard, ContextBundle, PeerBranch
from lethe.workspace.archive import Archive
from lethe.workspace.paths import UnsafePathError, safe_join

CONTEXT_DIR = ".context"
TASK_FILE = "TASK.md"

_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_SUMMARY_LIMIT = 200
_TABLE_HEAD = "| id | parent | score | lines changed | summary |\n| --- | --- | --- | --- | --- |\n"


class Materializer:
    """Creates per-agent, per-epoch workspaces under `ws_root/<agent>/e<epoch>/` and renders
    `TASK.md` and `.context/` into them before each turn.

    Rendered layout (rewritten from scratch every `render` call):
        TASK.md                                   bundle.task_md
        .context/feedback.md                      current candidate id, score/validity,
                                                  feedback, metrics
        .context/lineage.md                       own candidates this epoch, newest first
        .context/candidates/<cand_id>.diff        diff of each lineage card vs. its parent
        .context/peers/branch<i>/pool.md          peer's candidates this epoch
                                                  (id, score, lines, summary)
        .context/peers/branch<i>/<cand_id>.diff   diff of each pool card
        .context/peers/branch<i>/best/<path>      full judged files of the peer's best candidate
    Every written path is validated with `safe_join` (peer file names come from agent output).
    """

    def __init__(self, archive: Archive, ws_root: Path) -> None:
        self.archive = archive
        self.ws_root = Path(ws_root)  # created lazily, by the first prepare_epoch

    def prepare_epoch(
        self,
        agent: str,
        epoch: int,
        seed_sha: str,
        *,
        keep: Callable[[str], bool] | None = None,
    ) -> Path:
        """Create `ws_root/<agent>/e<epoch>/` holding the seed's files (filtered by `keep`).

        Raises FileExistsError if that workspace already exists.
        """
        if not _is_safe_name(agent):
            raise ValueError(f"invalid agent name {agent!r}: use [A-Za-z0-9_.-]+, not '.' or '..'")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError(f"epoch must be an integer >= 1, got {epoch!r}")
        path = self.ws_root / agent / f"e{epoch}"
        if path.is_symlink() or path.exists():
            raise FileExistsError(f"workspace already exists: {path}")
        self.archive.checkout(seed_sha, path, keep=keep)
        return path

    def render(self, workspace: Path, bundle: ContextBundle) -> None:
        """Rewrite `TASK.md` and `.context/` in `workspace` from `bundle`.

        Whatever the agent left at those two paths (a symlink, a file, a directory with its own
        notes) is removed first, never followed; every other file in the workspace is untouched.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        context = workspace / CONTEXT_DIR
        _remove(context)
        _remove(workspace / TASK_FILE)
        context.mkdir()

        _write(workspace, TASK_FILE, bundle.task_md)
        _write(context, "feedback.md", _feedback_md(bundle))
        _write(context, "lineage.md", _lineage_md(bundle.lineage))
        for card in bundle.lineage:
            _write_diff(context, "candidates", card)
        for peer in bundle.peers:
            self._render_peer(context, peer)

    def _render_peer(self, context: Path, peer: PeerBranch) -> None:
        branch_dir = f"peers/branch{int(peer.branch)}"
        _write(context, f"{branch_dir}/pool.md", _pool_md(peer, branch_dir))
        for card in peer.pool:
            _write_diff(context, branch_dir, card)
        best_dir = context / branch_dir / "best"
        for path, data in peer.best_files.items():
            try:
                target = safe_join(best_dir, path)
            except UnsafePathError:
                continue  # file names come from agent output: silently drop the unsafe ones
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


# --- filesystem helpers ------------------------------------------------------------------------


def _is_safe_name(value: object) -> bool:
    return isinstance(value, str) and value not in (".", "..") and bool(_NAME_RE.fullmatch(value))


def _remove(path: Path) -> None:
    """Delete whatever sits at `path` (symlink, file, directory), never following a link."""
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _write(root: Path, rel: str, text_or_bytes: str | bytes) -> None:
    target = safe_join(root, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode("utf-8", errors="replace")
    target.write_bytes(text_or_bytes)


def _write_diff(context: Path, directory: str, card: CandidateCard) -> None:
    if _is_safe_name(card.cand_id):  # card ids become file names
        _write(context, f"{directory}/{card.cand_id}.diff", card.diff)


# --- markdown ----------------------------------------------------------------------------------


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


def _cell(value: object) -> str:
    return _one_line(value).replace("|", "\\|")


def _summary_cell(summary: str) -> str:
    flat = _one_line(summary)
    if len(flat) > _SUMMARY_LIMIT:
        flat = flat[: _SUMMARY_LIMIT - 3].rstrip() + "..."
    return flat.replace("|", "\\|")


def _score_cell(card: CandidateCard) -> str:
    if card.valid is None:
        return "pending"
    if not card.valid or card.score is None:
        return "invalid"
    return f"{card.score:g}"


def _table(cards: Iterable[CandidateCard]) -> str:
    rows = (
        f"| {_cell(c.cand_id)} | {_cell(c.parent_id) if c.parent_id else '-'} | {_score_cell(c)} "
        f"| {c.lines_changed} | {_summary_cell(c.summary)} |\n"
        for c in cards
    )
    return _TABLE_HEAD + "".join(rows)


def _feedback_md(bundle: ContextBundle) -> str:
    card = bundle.current
    if card.valid is None:
        score = "Score: n/a (not judged yet)"
    elif card.valid and card.score is not None:
        score = f"Score: {card.score:g} (valid)"
    else:
        score = "Score: n/a (invalid)"
    feedback = bundle.feedback.strip("\n").rstrip() if bundle.feedback.strip() else "(none)"
    sections = [
        f"# Current candidate `{_one_line(card.cand_id)}`",
        score,
        f"## Judge feedback\n\n{feedback}",
    ]
    if bundle.metrics:
        lines = "\n".join(f"- {_one_line(k)}: {_one_line(v)}" for k, v in bundle.metrics.items())
        sections.append(f"## Metrics\n\n{lines}")
    return "\n\n".join(sections) + "\n"


def _lineage_md(lineage: Iterable[CandidateCard]) -> str:
    return (
        "# Your candidates this round (newest first)\n\n"
        + _table(lineage)
        + f"\nDiffs: `{CONTEXT_DIR}/candidates/<id>.diff`\n"
    )


def _pool_md(peer: PeerBranch, branch_dir: str) -> str:
    if peer.best is None:
        best = "Best: none yet"
    else:
        score = "n/a" if peer.best.score is None else f"{peer.best.score:g}"
        best = f"Best: `{_one_line(peer.best.cand_id)}` at {score}"
    footer = f"Diffs: `{CONTEXT_DIR}/{branch_dir}/<id>.diff`\n"
    if peer.best_files:
        footer += f"Best candidate files: `{CONTEXT_DIR}/{branch_dir}/best/`\n"
    return f"# Branch {int(peer.branch)}\n\n{best}\n\n{_table(peer.pool)}\n{footer}"
