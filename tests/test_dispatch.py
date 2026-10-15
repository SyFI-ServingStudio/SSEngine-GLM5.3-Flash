"""The controller's dispatch rule (ppimpl/runtime/dispatch.py): timing of the queued plan and the in-flight cap."""

import heapq

import pytest

from ppimpl.runtime.dispatch import NO, NOW, WAIT, DispatchRule, Stage0Predictor

MS = 1_000_000


def test_predictor_ewma_per_size_bucket():
    p = Stage0Predictor(alpha=0.5)
    assert p.predict_ns(8192) is None
    p.observe(8192, 57 * MS)
    assert p.predict_ns(8192) == pytest.approx(57 * MS)
    assert p.predict_ns(4096) == pytest.approx(28.5 * MS)      # nearest bucket's rate
    p.observe(8192, 59 * MS)
    assert p.predict_ns(8192) == pytest.approx(58 * MS)        # EWMA
    p.observe(1024, 2 * MS)                                    # small microbatches: their own rate
    assert p.predict_ns(1000) == pytest.approx(1000 * 2 * MS / 1024)
    assert p.predict_ns(8000) == pytest.approx(8000 * 58 * MS / 8192)


def test_idle_stage0_forms_at_once_under_the_cap():
    r = DispatchRule(4, True, 3.0)
    assert r.decide(0, 0) == (NOW, None)
    assert r.decide(3, 0) == (NOW, None)
    assert r.decide(4, 0) == (NO, None)        # stage 0 idle: a 5th would run, not queue


def test_queued_plan_forms_lead_before_predicted_end():
    r = DispatchRule(4, True, 3.0)
    r.dispatched(0, 0, 8192)
    assert r.decide(1, 1 * MS) == (NOW, None)  # no sample yet: form at once
    r.dispatched(1, 1 * MS, 8192)
    assert r.decide(2, 1 * MS) == (NO, None)   # one already queued
    r.stage0_done(0, 0, 50 * MS)               # plan 1 starts at 50 ms; predicted end 100 ms
    assert r.predicted_end_ns() == 100 * MS
    assert r.decide(4, 60 * MS) == (WAIT, 97 * MS)
    assert r.decide(4, 97 * MS) == (NOW, None)  # the 5th, queued behind plan 1
    assert r.decide(5, 97 * MS) == (NO, None)   # never a 6th
    r.dispatched(2, 97 * MS, 8192)
    assert r.decide(4, 98 * MS) == (NO, None)   # stage 0 has a running and a queued plan


def test_prediction_starts_at_dispatch_when_stage0_was_idle():
    r = DispatchRule(4, True, 3.0)
    r.dispatched(0, 0, 8192)
    r.stage0_done(0, 0, 50 * MS)
    r.dispatched(1, 200 * MS, 8192)            # stage 0 idle in between
    assert r.predicted_end_ns() == 250 * MS


def test_without_queue_one_stage0_must_be_free():
    r = DispatchRule(4, False, 3.0)
    assert r.cap == 4
    r.dispatched(0, 0, 8192)
    assert r.decide(1, 0) == (NO, None)


def simulate(rule: DispatchRule | None, n_mb=60, stage_ms=10.0, report_ms=0.3, form_ms=0.8):
    """A 4-stage pipeline of equal stages, integer ns. The host learns of a stage end `report_ms` later; a plan reaches
    stage 0 `form_ms` after it is formed. rule=None: the old rule (<= 4 in flight, one queued, form on a report).
    Returns the stage-0 gaps (ms), the peak in flight and the peak number of plans waiting at stage 0."""
    st, rep, form = int(stage_ms * MS), int(report_ms * MS), int(form_ms * MS)
    events, seq = [], [0]
    free = [0] * 4
    s0_start, s0_end, arrived = {}, {}, {}
    state = dict(in_flight=0, peak=0, next_mb=0, queued=0, timer=None)
    on0 = []                                  # dispatched, stage 0 not reported done (host view)

    def push(at, kind, data=None):
        seq[0] += 1
        heapq.heappush(events, (at, seq[0], kind, data))

    def try_form(now):
        while state["next_mb"] < n_mb:
            if rule is None:
                if state["in_flight"] >= 4 or len(on0) > 1:
                    return
            else:
                v, at = rule.decide(state["in_flight"], now)
                if v == WAIT:
                    if state["timer"] != at:
                        state["timer"] = at
                        push(at, "timer")
                    return
                if v == NO:
                    return
                rule.dispatched(state["next_mb"], now, 8192)
            mb = state["next_mb"]
            state["next_mb"] += 1
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            on0.append(mb)
            push(now + form, "plan", mb)

    push(0, "timer")
    steps = 0
    while events:
        steps += 1
        assert steps < 100_000
        t, _, kind, data = heapq.heappop(events)
        if kind == "timer":
            state["timer"] = None
            try_form(t)
        elif kind == "plan":                  # at stage 0: runs when stage 0 is free
            arrived[data] = t
            start = max(t, free[0])
            # plans that reached stage 0 and wait there (not started) at this instant, besides this one
            state["queued"] = max(state["queued"], sum(1 for m in arrived if m != data and s0_start[m] > t))
            free[0] = start + st
            s0_start[data], s0_end[data] = start, start + st
            push(free[0] + rep, "done0", data)
            end = free[0]
            for s in range(1, 4):
                free[s] = end = max(end, free[s]) + st
            push(end + rep, "done3", data)
        elif kind == "done0":
            on0.remove(data)
            if rule is not None:
                rule.stage0_done(data, s0_start[data], s0_end[data])
            try_form(t)
        elif kind == "done3":
            state["in_flight"] -= 1
            try_form(t)
    gaps = [(s0_start[k + 1] - s0_end[k]) / MS for k in range(n_mb - 1)]
    return gaps, state["peak"], state["queued"]


def test_rule_closes_the_stage0_host_gap():
    old_gaps, old_peak, _ = simulate(None)
    assert old_peak == 4
    assert max(old_gaps[10:]) > 1.0                    # report -> form -> plan, every microbatch
    gaps, peak, queued = simulate(DispatchRule(4, True, 3.0))
    assert peak == 5 and queued <= 1
    assert max(gaps[3:]) == pytest.approx(0.0, abs=1e-6)   # back to back once a sample exists


def test_lead_shorter_than_host_time_leaves_the_difference():
    gaps, _, _ = simulate(DispatchRule(4, True, 0.5), form_ms=0.8)
    assert max(gaps[5:]) == pytest.approx(0.3, abs=1e-3)
