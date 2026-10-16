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


def simulate(rule: DispatchRule | None, n_mb=60, stage_ms=10.0, report_ms=0.3, form_ms=0.8, out: dict | None = None):
    """A 4-stage pipeline (equal stages, or `stage_ms` per stage), integer ns. The host learns of a stage end
    `report_ms` later; a plan reaches stage 0 `form_ms` after it is formed. rule=None: the old rule (<= 4 in flight,
    one queued, form on a report). Returns the stage-0 gaps (ms), the peak in flight and the peak number of plans
    waiting at stage 0; `out` gets per microbatch its form time and stage starts and ends (ns)."""
    sts = [int(x * MS) for x in (stage_ms if isinstance(stage_ms, list) else [stage_ms] * 4)]
    st, rep, form = sts[0], int(report_ms * MS), int(form_ms * MS)
    formed, starts, ends = {}, {}, {}
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
            formed[mb] = now
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
            starts[data], ends[data] = [start], [start + st]
            push(free[0] + rep, "done0", data)
            end = free[0]
            for s in range(1, 4):
                begin = max(end, free[s])
                free[s] = end = begin + sts[s]
                starts[data].append(begin)
                ends[data].append(end)
                if s < 3:
                    push(end + rep, "done", (s, data))
            push(end + rep, "done3", data)
        elif kind == "done0":
            on0.remove(data)
            if rule is not None:
                rule.stage0_done(data, s0_start[data], s0_end[data])
                rule.stage_done(0, data, s0_start[data], s0_end[data])
            try_form(t)
        elif kind == "done":
            if rule is not None:
                rule.stage_done(data[0], data[1], starts[data[1]][data[0]], ends[data[1]][data[0]])
        elif kind == "done3":
            state["in_flight"] -= 1
            if rule is not None:
                rule.stage_done(3, data, starts[data][3], ends[data][3])
            try_form(t)
    if out is not None:
        out.update(formed=formed, starts=starts, ends=ends)
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


def test_pacing_moves_the_queue_before_a_slow_stage_to_forming_time():
    """Stage 2 slower than stage 0: without pacing the extra in-flight plan queues in front of stage 2 and its rounds
    wait there; with pacing the plan forms later (its rounds can still join), stage 2 stays as busy, and the time from
    forming to the stage-3 end drops to the stages' compute plus host time."""
    stage_ms = [10.0, 10.0, 12.0, 10.0]

    def run(margin):
        out = {}
        simulate(DispatchRule(4, True, 3.0, margin), n_mb=80, stage_ms=stage_ms, out=out)
        ids = range(20, 75)
        queue2 = max((out["starts"][k][2] - out["ends"][k][1]) / MS for k in ids)
        idle2 = max((out["starts"][k + 1][2] - out["ends"][k][2]) / MS for k in ids)
        span = sum((out["ends"][k][3] - out["formed"][k]) / MS for k in ids) / len(ids)
        return queue2, idle2, span

    q, idle, span = run(None)
    assert q > 10.0 and idle == pytest.approx(0.0, abs=1e-6)
    q, idle, paced_span = run(1.0)
    assert q <= 1.5 and idle == pytest.approx(0.0, abs=1e-6)      # stage 2 still back to back
    assert paced_span < span - 10.0
    assert paced_span == pytest.approx(sum(stage_ms) + 0.8 + 1.0, abs=1.5)


def test_pacing_does_not_bind_when_stage0_is_slowest():
    a, b = {}, {}
    simulate(DispatchRule(4, True, 3.0), n_mb=60, stage_ms=[12.0, 10.0, 10.0, 10.0], out=a)
    gaps, _, _ = simulate(DispatchRule(4, True, 3.0, 1.0), n_mb=60, stage_ms=[12.0, 10.0, 10.0, 10.0], out=b)
    assert max(gaps[5:]) == pytest.approx(0.0, abs=1e-6)
    assert a["starts"][59] == b["starts"][59]
