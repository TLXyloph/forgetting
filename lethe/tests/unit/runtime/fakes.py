"""In-memory fakes of the orchestrator's collaborators (archive, materializer, judge, backend)."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from lethe.config import RunConfig, load_config
from lethe.domain.events import Envelope, Event
from lethe.domain.types import JudgeResult, RateLimitInfo, ToolCall, TurnResult, Usage
from lethe.runtime.orchestrator import Orchestrator
from lethe.workspace.archive import Snapshot

FILE = "solver.txt"


def make_config(tmp_path: Path, seed_value: str = "10", **overrides) -> RunConfig:
    seed = tmp_path / "task" / "seed"
    seed.mkdir(parents=True)
    (seed / FILE).write_text(seed_value)
    (seed / "README.txt").write_text("not judged")
    cfg = tmp_path / "task" / "lethe.toml"
    cfg.write_text(
        '[task]\nname = "toy"\ndescription = "Make the number big."\nseed = "seed"\n'
        f'editable = ["{FILE}"]\n\n[judge]\nrun = "unused"\ndirection = "max"\n'
    )
    return load_config(cfg, overrides=overrides)


class FakeArchive:
    def __init__(self):
        self.trees: dict[str, dict[str, str]] = {}
        self.snapshots: list[tuple[str, str | None]] = []

    def _key(self, tree):
        return hashlib.sha256(tree.get(FILE, "").encode()).hexdigest()

    def snapshot(self, workdir, *, parent_sha, message, index_name):
        tree = {p.name: p.read_text() for p in Path(workdir).iterdir() if p.is_file()}
        sha = f"sha{len(self.trees)}"
        self.trees[sha] = tree
        self.snapshots.append((sha, parent_sha))
        parent = self.trees.get(parent_sha) if parent_sha else None
        changed = parent is None or self._key(parent) != self._key(tree)
        lines = 0 if parent is None or not changed else 2
        return Snapshot(sha, self._key(tree), lines // 2, lines // 2, int(changed), changed)

    def checkout_overlay(self, base_sha, cand_sha, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / FILE).write_text(self.trees[cand_sha][FILE])

    def files(self, sha):
        return {name: f"blob-{sha}-{name}" for name in self.trees[sha]}

    def read_file(self, sha, path):
        return self.trees[sha][path].encode()

    def diff(self, base_sha, sha, *, judged_only=True):
        return f"diff {base_sha}..{sha}"


class FakeMaterializer:
    def __init__(self, archive: FakeArchive, root: Path):
        self.archive, self.root = archive, root
        # (agent, epoch, seed_sha, keep, base_sha) per prepared workspace
        self.prepared: list[tuple[str, int, str, Callable | None, str | None]] = []
        self.rendered: list = []

    def prepare_epoch(self, agent, epoch, seed_sha, *, keep=None, base_sha=None):
        ws = self.root / agent / f"e{epoch}"
        ws.mkdir(parents=True)
        files = dict(self.archive.trees[seed_sha])
        if base_sha is not None:  # overlay: the base's other files plus the seed's judged file
            files = {n: t for n, t in self.archive.trees[base_sha].items() if n != FILE}
            files[FILE] = self.archive.trees[seed_sha][FILE]
        for name, text in files.items():
            if keep is None or keep(name):
                (ws / name).write_text(text)
        self.prepared.append((agent, epoch, seed_sha, keep, base_sha))
        return ws

    def render(self, workspace, bundle):
        self.rendered.append((workspace, bundle))


class FakeJudge:
    """Scores the number written in solver.txt; non-numbers are invalid."""

    def __init__(self):
        self.calls: list[str] = []

    async def evaluate(self, *, cand_id, judged_key, materialize):
        import tempfile

        self.calls.append(cand_id)
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            materialize(tree)
            text = (tree / FILE).read_text()
        try:
            return JudgeResult(
                True, float(text), f"value {text}", {"len": len(text)}, {"x": float(text)}
            )
        except ValueError:
            return JudgeResult(False, None, "not a number")


@dataclass
class Reply:
    value: str | None = None  # new solver.txt content, None = no edit
    text: str = "changed it"
    cost: float = 0.1
    error_kind: str | None = None
    rate_limit: RateLimitInfo | None = None
    error: str = "boom"
    delay: float = 0.0  # real seconds the turn takes
    tool_calls: tuple[ToolCall, ...] = (ToolCall("Read", FILE), ToolCall("Edit", FILE))


@dataclass
class FakeBackend:
    """Agents follow `script(agent, call_index_for_agent, prompt, session) -> Reply`."""

    script: Callable[..., Reply]
    calls: list[dict] = field(default_factory=list)

    async def run_turn(self, *, agent, workspace, prompt, session, transcript_path):
        index = sum(1 for c in self.calls if c["agent"] == agent)
        reply = self.script(agent, index, prompt, session)
        if reply.delay:
            await asyncio.sleep(reply.delay)
        self.calls.append(
            {
                "agent": agent,
                "workspace": workspace,
                "prompt": prompt,
                "session": session,
                "transcript": transcript_path,
            }
        )
        if reply.value is not None:
            (Path(workspace) / FILE).write_text(reply.value)
        return TurnResult(
            session_id=session.session_id,
            ok=reply.error_kind is None,
            final_text=reply.text,
            cost_usd=reply.cost,
            usage=Usage(output_tokens=100),
            tool_calls=reply.tool_calls,
            duration_s=0.01,
            error_kind=reply.error_kind,  # type: ignore[arg-type]
            error=reply.error if reply.error_kind else None,
            rate_limit=reply.rate_limit,
        )


class FakeStore:
    def __init__(self):
        self.events: list[Event] = []

    def append(self, event: Event) -> Envelope:
        envelope = Envelope(seq=len(self.events), ts=1000.0 + len(self.events), event=event)
        self.events.append(event)
        return envelope

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def of(self, cls) -> list:
        return [e for e in self.events if isinstance(e, cls)]


class FakeTime:
    """A clock for the Pacer: `sleep` advances it, after a real pause of `tick` seconds."""

    def __init__(self, tick: float = 0.0):
        self.now = 1000.0
        self.tick = tick

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds
        await asyncio.sleep(self.tick)


def build(tmp_path, script, *, pacer=None, events=None, backend=None, **overrides):
    """An Orchestrator on the fakes; `backend` replaces FakeBackend(script) when given."""
    config = make_config(tmp_path, **overrides)
    archive = FakeArchive()
    materializer = FakeMaterializer(archive, tmp_path / "ws")
    judge, store = FakeJudge(), FakeStore()
    backend = backend if backend is not None else FakeBackend(script)
    counter = count(1)
    orch = Orchestrator(
        config,
        run_id="run-1",
        run_dir=tmp_path / "run",
        archive=archive,
        materializer=materializer,
        judge=judge,
        backend=backend,
        store=store,
        new_session_id=lambda: f"sess-{next(counter)}",
        pacer=pacer,
        on_event=(lambda env, state: events.append(env.event.kind)) if events is not None else None,
    )
    return orch, store, backend, materializer, judge
