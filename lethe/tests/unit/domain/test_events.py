"""Event schema evolution: new fields have defaults, so logs written before them stay readable."""

from __future__ import annotations

import json

import pytest

from lethe.domain.events import Envelope, RunFinished, TurnFinished, decode, encode


def test_run_finished_from_an_old_log_counts_as_completed():
    old = {"seq": 9, "ts": 1.0, "kind": "run_finished", "reason": "finished"}
    old |= {"best_cand_id": "c0000", "best_score": 5.0}
    assert decode(json.dumps(old)).event.status == "completed"


def test_turn_finished_from_an_old_log_has_no_error_kind():
    turn = TurnFinished("a1", 1, "s1", 1, "error", None, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, False, 1.0)
    record = json.loads(encode(Envelope(3, 2.0, turn)))
    del record["error_kind"]
    assert decode(json.dumps(record)).event.error_kind is None


@pytest.mark.parametrize(
    "event",
    [
        RunFinished("crashed: RuntimeError: boom", None, None, status="crashed"),
        RunFinished("all agents stopped after repeated errors", "c0001", 2.0, status="stopped"),
        TurnFinished(
            "a1",
            2,
            "s9",
            3,
            "error",
            None,
            0.5,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            False,
            9.0,
            error="timed out",
            error_kind="timeout",
        ),
    ],
)
def test_new_fields_survive_the_log(event):
    assert decode(encode(Envelope(1, 2.0, event))).event == event
