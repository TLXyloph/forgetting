"""Shared test helpers for the controller-owned tests (imported as `tests_support`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from lethe.domain.events import Envelope, Event
from lethe.domain.state import RunState


@dataclass
class Feeder:
    """Applies events to a RunState with increasing sequence numbers."""

    state: RunState = field(default_factory=RunState)
    seq: int = 0

    def __call__(self, event: Event) -> Envelope:
        envelope = Envelope(seq=self.seq, ts=1000.0 + self.seq, event=event)
        self.seq += 1
        self.state.apply(envelope)
        return envelope


FeedFn = Callable[[Event], Envelope]
