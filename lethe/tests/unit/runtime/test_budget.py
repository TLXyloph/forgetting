from __future__ import annotations

import asyncio

from lethe.domain.types import RateLimitInfo
from lethe.runtime.budget import BudgetSpec, Pacer, exhausted, spend_fraction


def test_spend_fraction_uses_the_primary_unit():
    assert spend_fraction(BudgetSpec(usd=10.0), spent_usd=2.5, proposals=0, elapsed_s=0) == 0.25
    spec = BudgetSpec(usd=0, proposals=40, primary="proposals")
    assert spend_fraction(spec, spent_usd=99, proposals=10, elapsed_s=0) == 0.25
    spec = BudgetSpec(usd=0, wall_s=100.0, primary="wall")
    assert spend_fraction(spec, spent_usd=0, proposals=0, elapsed_s=50) == 0.5


def test_exhausted_reports_the_first_cap_reached():
    spec = BudgetSpec(usd=10.0, proposals=100, wall_s=3600)
    assert exhausted(spec, spent_usd=9.99, proposals=99, elapsed_s=10) is None
    assert "usd" in exhausted(spec, spent_usd=10.0, proposals=1, elapsed_s=10)
    assert "proposal" in exhausted(spec, spent_usd=0, proposals=100, elapsed_s=10)
    assert "wall" in exhausted(spec, spent_usd=0, proposals=0, elapsed_s=3600)


class FakeTime:
    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_pauses_when_window_utilization_crosses_threshold():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    assert pacer.observe(RateLimitInfo("allowed", 0.5, 0.1, 2000.0)) is None
    until = pacer.observe(RateLimitInfo("allowed_warning", 0.96, 0.1, 2000.0))
    assert until == 2000.0 + Pacer.GRACE_S
    asyncio.run(pacer.wait())
    assert t.now >= until


def test_pacer_backs_off_exponentially_without_reset_time():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep, base_backoff_s=60, max_backoff_s=200)
    assert pacer.note_rate_limited(None) == 1060.0
    assert pacer.note_rate_limited(RateLimitInfo("rejected", None, None, None)) == 1120.0
    assert pacer.note_rate_limited(None) == 1200.0  # capped at 200 s
    t.now = 1300.0  # the pause elapsed and a later turn went through
    pacer.observe(RateLimitInfo("allowed", 0.1, 0.1, None))
    assert pacer.note_rate_limited(None) == 1360.0  # backoff reset after a healthy turn


def test_pacer_uses_reset_time_when_rate_limited():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    assert (
        pacer.note_rate_limited(RateLimitInfo("rejected", 1.0, 0.2, 5000.0))
        == 5000.0 + Pacer.GRACE_S
    )


def test_pacer_wait_is_immediate_when_not_paused():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    asyncio.run(pacer.wait())
    assert t.sleeps == []


def test_a_full_five_hour_window_pauses_until_that_window_resets():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    # the event's own resetsAt belongs to the 7-day window it names: not the one to wait for
    info = RateLimitInfo(
        "allowed_warning", 0.97, 0.5, resets_at=900000.0, five_hour_resets_at=4000.0
    )
    assert pacer.observe(info) == 4000.0 + Pacer.GRACE_S


def test_without_a_five_hour_reset_the_events_reset_is_the_fallback():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    assert (
        pacer.observe(RateLimitInfo("allowed_warning", 0.97, 0.5, 3000.0)) == 3000.0 + Pacer.GRACE_S
    )


def test_the_weekly_threshold_is_off_by_default():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, clock=t.clock, sleep=t.sleep)
    info = RateLimitInfo("allowed", 0.1, 0.99, resets_at=2000.0, seven_day_resets_at=9000.0)
    assert pacer.observe(info) is None


def test_a_weekly_window_past_its_threshold_pauses_until_that_window_resets():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, weekly_pause_at=0.8, clock=t.clock, sleep=t.sleep)
    below = RateLimitInfo("allowed", 0.1, 0.79, resets_at=2000.0, seven_day_resets_at=9000.0)
    assert pacer.observe(below) is None
    # the event's own resetsAt names the 5-hour window here: not the one to wait for
    crossed = RateLimitInfo("allowed", 0.1, 0.8, resets_at=2000.0, seven_day_resets_at=9000.0)
    assert pacer.observe(crossed) == 9000.0 + Pacer.GRACE_S
    asyncio.run(pacer.wait())
    assert t.now >= 9000.0 + Pacer.GRACE_S


def test_both_windows_full_waits_for_the_later_reset():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, weekly_pause_at=0.8, clock=t.clock, sleep=t.sleep)
    info = RateLimitInfo(
        "allowed_warning", 0.97, 0.9, 2000.0, five_hour_resets_at=4000.0, seven_day_resets_at=9000.0
    )
    assert pacer.observe(info) == 9000.0 + Pacer.GRACE_S


def test_without_a_weekly_reset_the_events_reset_is_the_fallback():
    t = FakeTime()
    pacer = Pacer(pause_at=0.95, weekly_pause_at=0.8, clock=t.clock, sleep=t.sleep)
    assert pacer.observe(RateLimitInfo("allowed", 0.1, 0.85, 3000.0)) == 3000.0 + Pacer.GRACE_S
