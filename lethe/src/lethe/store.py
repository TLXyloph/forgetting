"""Append-only event store backed by a JSONL file."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from lethe.domain.events import Envelope, Event, decode, encode


class EventStore:
    """Appends events to `path`, assigning monotonically increasing sequence numbers.

    Each append is flushed and fsynced so a crash never loses an acknowledged event.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        last = -1
        if path.exists():
            for envelope in read_events(path):
                last = envelope.seq
        self._next_seq = last + 1

    def append(self, event: Event) -> Envelope:
        envelope = Envelope(seq=self._next_seq, ts=self._clock(), event=event)
        line = encode(envelope) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._next_seq += 1
        return envelope


def read_events(path: Path) -> Iterator[Envelope]:
    """Yield every envelope in `path`, skipping a torn final line from an interrupted write."""
    with path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            yield decode(line)
        except ValueError:
            if index == len(lines) - 1:
                return
            raise
