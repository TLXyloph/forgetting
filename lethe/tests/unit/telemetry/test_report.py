from __future__ import annotations

import csv

from lethe.domain.events import (
    CandidateCommitted,
    EpochStarted,
    ForgetDecided,
    JudgeFinished,
    RateLimited,
    RunFinished,
    RunStarted,
    TurnFinished,
)
from lethe.telemetry.live import format_event
from lethe.telemetry.report import build_report, format_report, write_csvs

AGENTS = ("a1", "a2")


def history(feed):
    feed(RunStarted("run-9", "toy", AGENTS, "max", {}, "0.1.0"))
    feed(CandidateCommitted("c0000", "s0", None, "seed", 0, None, "k0", 0, 0, 0, "seed"))
    feed(JudgeFinished("c0000", True, 100.0, "", {}, None, 0.0, False, None))
    feed(
        EpochStarted("start", dict.fromkeys(AGENTS, "c0000"), dict.fromkeys(AGENTS, 1), "run start")
    )
    turns = (("a1", 80, 110.0), ("a2", 40, 105.0), ("a1", 10, None))
    for n, (agent, lines, score) in enumerate(turns, start=1):
        cid = f"c{n:04d}"
        status = "candidate" if score is not None else "empty"
        if score is not None:
            feed(
                CandidateCommitted(cid, cid, "c0000", agent, 1, f"s{agent}", cid, lines, 0, 1, "x")
            )
        feed(
            TurnFinished(
                agent,
                1,
                f"s{agent}",
                1,
                status,
                cid if score else None,
                0.5,
                1,
                200,
                0,
                0,
                50,
                2,
                1,
                lines if score else 0,
                score is None,
                3.0,
            )
        )
        if score is not None:
            feed(JudgeFinished(cid, True, score, "", {}, None, 0.1, False, None))
    feed(ForgetDecided("team", AGENTS, "budget_fractions@1 (spend at 50% of budget)"))
    feed(
        EpochStarted(
            "team",
            dict.fromkeys(AGENTS, "c0001"),
            dict.fromkeys(AGENTS, 2),
            "budget_fractions@1 (spend at 50% of budget)",
        )
    )
    feed(CandidateCommitted("c0003", "c3", "c0001", "a2", 2, "s2b", "c3", 60, 5, 1, "y"))
    feed(
        TurnFinished(
            "a2",
            2,
            "s2b",
            1,
            "candidate",
            "c0003",
            0.5,
            1,
            200,
            0,
            0,
            50,
            2,
            1,
            65,
            False,
            3.0,
            None,
            0.42,
            0.1,
        )
    )
    feed(JudgeFinished("c0003", True, 121.0, "", {}, None, 0.1, False, None))
    feed(RateLimited("a1", 2000.0, "rate limited"))
    feed(RunFinished("proposal budget exhausted (4 of 4)", "c0003", 121.0))


def test_report_summarizes_run_rounds_and_agents(feeder):
    state, feed = feeder
    history(feed)
    report = build_report(state)
    assert report.best_id == "c0003" and report.best_score == 121.0
    assert report.seed_score == 100.0 and round(report.improvement_pct, 1) == 21.0
    assert report.proposals == 4 and report.spent_usd == 2.0
    assert [r.round for r in report.rounds] == [1, 2]
    assert report.rounds[0].best_score == 110.0 and report.rounds[1].seed_id == "c0001"
    assert round(report.rounds[1].gain_pct, 1) == 10.0
    a1 = next(row for row in report.agents if row.agent == "a1")
    assert (a1.turns, a1.candidates, a1.empty, a1.plateau_claims) == (2, 1, 1, 1)
    assert report.rate_limited == 1 and report.five_hour_utilization == 0.42
    text = format_report(report)
    assert "c0003" in text and "Round" in text and "+21.0%" in text


def test_write_csvs(tmp_path, feeder):
    state, feed = feeder
    history(feed)
    paths = write_csvs(state, tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["best_curve.csv", "rounds.csv", "turns.csv"]
    rows = list(csv.DictReader((tmp_path / "turns.csv").open()))
    assert len(rows) == 4 and rows[0]["agent"] == "a1" and rows[0]["lines_changed"] == "80"
    curve = list(csv.DictReader((tmp_path / "best_curve.csv").open()))
    assert [r["cand_id"] for r in curve] == ["c0000", "c0001", "c0003"]


def test_live_lines(feeder):
    state, feed = feeder
    history_envelopes = []

    def capture(event):
        envelope = feed(event)
        history_envelopes.append((envelope, format_event(envelope, state)))

    history(capture)
    lines = [line for _, line in history_envelopes if line]
    assert any(line.startswith("[a1 r1 t1] c0001") and "110" in line for line in lines)
    assert any("forget" in line.lower() for line in lines)
    assert any("new best" in line for line in lines)
    assert any("rate" in line.lower() for line in lines)
