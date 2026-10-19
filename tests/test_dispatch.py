"""The controller's dispatch rule (ssengine/runtime/dispatch.py): forming ahead of a projected start, with the in-flight
limit on the GPU (stage 0 waits for microbatch k-4's exit)."""

import heapq

import pytest

from ssengine.runtime.dispatch import NO, NOW, WAIT, DispatchRule, StagePredictor

MS = 1_000_000


def test_predictor_ewma_per_size_bucket():
    p = StagePredictor(alpha=0.5)
    assert p.predict_ns(8192) is None
    p.observe(8192, 57 * MS)
    assert p.predict_ns(8192) == pytest.approx(57 * MS)
    assert p.predict_ns(4096) == pytest.approx(28.5 * MS)      # nearest bucket's rate
    p.observe(8192, 59 * MS)
    assert p.predict_ns(8192) == pytest.approx(58 * MS)        # EWMA
    p.observe(1024, 2 * MS)                                    # small microbatches: their own rate
    assert p.predict_ns(1000) == pytest.approx(1000 * 2 * MS / 1024)
    assert p.predict_ns(8000) == pytest.approx(8000 * 58 * MS / 8192)


def test_cold_start_forms_with_stage0_free_and_room():
    r = DispatchRule(4, 3.0)
    assert r.decide(0) == (NOW, None)
    r.dispatched(0, 0, 8192)
    assert r.decide(1 * MS) == (NO, None)       # no sample: wait for stage 0's report
    done(r, 0, [(0, 50), (50, 100), (100, 150), (150, 200)], stages=(0,))
    assert r.decide(51 * MS) == (NOW, None)       # stage 1 has no sample: stage 0's rate stands in


def done(r: DispatchRule, mb: int, spans, stages=(0, 1, 2, 3)):
    for s in stages:
        a, b = spans[s]
        r.stage_done(s, mb, a * MS, b * MS)


def test_forms_lead_and_host_before_the_projected_start():
    r = DispatchRule(4, 3.0)
    r.dispatched(0, 0, 8192)
    done(r, 0, [(1, 51), (51, 101), (101, 151), (151, 201)], stages=(0,))   # host 1 ms, stage 0 50 ms
    r.dispatched(1, 52 * MS, 8192)
    # plan 1: start max(52 + 1, 51) = 53, end 103 -> plan 2 starts at 103, forms at 103 - 1 - 3 = 99
    assert r.decide(60 * MS) == (WAIT, 99 * MS)
    assert r.decide(99 * MS) == (NOW, None)
    assert r.exiting == 0


def test_the_gate_moves_the_start_to_the_exit_of_k_minus_4():
    """Stage 3 slower: the 5th plan starts when the 1st leaves stage 3, and forms that long ahead."""
    r = DispatchRule(4, 0.0)
    r.dispatched(0, 0, 1000)
    # mb 0: stage 0 10 ms, stage 3 100 ms (all reported): exit at 130
    done(r, 0, [(0, 10), (10, 20), (20, 30), (30, 130)])
    r.host.rate = {r.host.bucket(1000): 0.0}
    for k in range(1, 5):
        r.dispatched(k, 10 * MS * k, 1000)   # each forms as stage 0 frees; only stage 0 reports
        r.stage_done(0, k, 10 * MS * k, 10 * MS * (k + 1))
    p = r.project()
    # mb 1..4 on stage 3: 130 -> 230 -> 330 -> 430 -> 530; mb 5 starts at mb 1's exit (230)
    assert p["ends"][1][3] == pytest.approx(230 * MS)
    assert p["ends"][4][3] == pytest.approx(530 * MS)
    assert p["start"] == pytest.approx(230 * MS)
    assert p["exiting"] == 1                 # mb 1 leaves before mb 5 starts
    assert r.decide(100 * MS) == (WAIT, 230 * MS)


def ideal(n_mb: int, sts: list[int], depth: int = 4, t0: int = 0) -> list[int]:
    """The targets' stage-0 starts: max(stage 0 free, exit of k - depth), no host time, the first at `t0`."""
    free, starts, exits = [t0] * 4, [], []
    for k in range(n_mb):
        s0 = max(free[0], exits[k - depth] if k >= depth else 0)
        starts.append(s0)
        end = s0
        for s in range(4):
            begin = end if s == 0 else max(end, free[s])
            free[s] = end = begin + sts[s]
        exits.append(end)
    return starts


def simulate(rule: DispatchRule | None, n_mb=60, stage_ms=10.0, report_ms=0.3, form_ms=0.8, actual=None,
             out: dict | None = None, report_delay=None, depth: int = 4):
    """A 4-stage pipeline, integer ns, with the GPU gate: stage 0 starts mb k at max(plan arrived, stage 0 free, exit
    of k-4). The host learns of a stage end `report_ms` later; a plan reaches stage 0 `form_ms` after it is formed.
    `actual(k, s)`: a stage time other than `stage_ms` (ms). `report_delay(k, s)`: a report latency other than
    `report_ms` (ms); each stage's reports still arrive in order (one reporter per worker), across stages in any. rule=None: form on reports (stage 0 free, < 4 in flight
    by reports). Returns the stage-0 starts (ns), the peak in flight on the GPU, the peak number waiting at stage 0."""
    sts = [int(x * MS) for x in (stage_ms if isinstance(stage_ms, list) else [stage_ms] * 4)]
    rep, form = int(report_ms * MS), int(form_ms * MS)
    events, seq = [], [0]
    free = [0] * 4
    starts, ends, formed, arrived = {}, {}, {}, {}
    state = dict(in_flight=0, next_mb=0, timer=None)
    on0 = []
    last_report = [0] * 4

    def t_of(k, s):
        return int(actual(k, s) * MS) if actual else sts[s]

    def push(at, kind, data=None):
        seq[0] += 1
        heapq.heappush(events, (at, seq[0], kind, data))

    def try_form(now):
        while state["next_mb"] < n_mb:
            if rule is None:
                if state["in_flight"] >= 4 or on0:
                    return
            else:
                v, at = rule.decide(now)
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
            on0.append(mb)
            push(now + form, "plan", mb)

    push(0, "timer")
    steps = 0
    while events:
        steps += 1
        assert steps < 200_000
        t, _, kind, data = heapq.heappop(events)
        if kind == "timer":
            state["timer"] = None
            try_form(t)
        elif kind == "plan":                  # plans reach stage 0 in order; the GPU runs them in order
            k = data
            arrived[k] = t
            gate = ends[k - depth][3] if k >= depth else 0
            start = max(t, free[0], gate)
            starts[k], ends[k] = [start], [start + t_of(k, 0)]
            free[0] = ends[k][0]
            end = free[0]
            for s in range(1, 4):
                begin = max(end, free[s])
                free[s] = end = begin + t_of(k, s)
                starts[k].append(begin)
                ends[k].append(end)
            for s in range(4):
                at = ends[k][s] + (int(report_delay(k, s) * MS) if report_delay else rep)
                at = max(at, last_report[s])
                last_report[s] = at
                push(at, "done", (s, k))
        elif kind == "done":
            s, k = data
            if s == 0:
                on0.remove(k)
            if s == 3:
                state["in_flight"] -= 1
            if rule is not None:
                rule.stage_done(s, k, starts[k][s], ends[k][s])
            try_form(t)
    ev = sorted([(starts[k][0], 1) for k in starts] + [(ends[k][3], -1) for k in starts], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    waiting = max(sum(1 for m in arrived if arrived[m] <= starts[k][0] < starts[m][0]) for k in starts)
    if out is not None:
        out.update(formed=formed, starts=starts, ends=ends, arrived=arrived)
    return [starts[k][0] for k in range(n_mb)], peak, waiting


def steps(starts: list[int], first: int) -> list[int]:
    return [b - a for a, b in zip(starts[first:], starts[first + 1:])]


def test_rule_starts_stage0_as_the_targets_do():
    s0, peak, waiting = simulate(DispatchRule(4, 3.0))
    want = ideal(60, [10 * MS] * 4)
    assert peak <= 4 and waiting <= 2
    assert steps(s0, 8) == pytest.approx(steps(want, 8), abs=1)   # back to back once samples exist
    old, old_peak, _ = simulate(None)
    assert old_peak <= 4
    assert old[-1] - want[-1] > 50 * MS                     # report -> form -> plan, every microbatch


def test_slow_later_stage_follows_the_exit_gate_not_reports():
    """Stage 2 slower: stage 0 starts k at the exit of k-4 (no report latency, no plan delivery)."""
    sts = [10.0, 10.0, 12.0, 10.0]
    s0, peak, _ = simulate(DispatchRule(4, 3.0), n_mb=80, stage_ms=sts)
    want = ideal(80, [int(x * MS) for x in sts])
    assert peak <= 4
    assert steps(s0, 10) == pytest.approx(steps(want, 10), abs=1)


def test_report_latency_longer_than_a_microbatch():
    """Small microbatches (2 ms) with 3 ms reports: still back to back."""
    s0, peak, _ = simulate(DispatchRule(4, 1.0), n_mb=80, stage_ms=2.0, report_ms=3.0, form_ms=0.8)
    want = ideal(80, [2 * MS] * 4)
    assert peak <= 4
    assert steps(s0, 20) == pytest.approx(steps(want, 20), abs=1)


def test_noisy_stage_times_cost_at_most_the_late_error():
    """Times vary +-10%: plans formed early wait on the GPU (no cost); stage 0 idles only where a stage ended more than
    `lead` earlier than projected."""
    import random
    rnd = random.Random(7)
    table = {(k, s): 10.0 * rnd.uniform(0.9, 1.1) for k in range(80) for s in range(4)}
    out = {}
    s0, peak, _ = simulate(DispatchRule(4, 2.0), n_mb=80, actual=lambda k, s: table[(k, s)], out=out)
    assert peak <= 4
    ends = out["ends"]
    idle = sum(max(s0[k] - max(ends[k - 1][0], ends[k - 4][3] if k >= 4 else 0), 0) for k in range(10, 80))
    assert idle < 0.002 * (s0[79] - s0[10])


def test_host_time_learned_only_when_stage0_was_ready():
    r = DispatchRule(4, 3.0)
    r.dispatched(0, 0, 8192)
    done(r, 0, [(0.7, 50), (50, 100), (100, 150), (150, 200)], stages=(0,))
    assert r.host.predict_ns(8192) == pytest.approx(0.7 * MS)
    r.dispatched(1, 10 * MS, 8192)                 # formed while stage 0 was busy: its start is not host time
    done(r, 1, [(50, 100), (100, 150), (150, 200), (200, 250)], stages=(0,))
    assert r.host.predict_ns(8192) == pytest.approx(0.7 * MS)


def test_stage0_report_after_the_last_stages_report():
    """7878: mb 6279's stage-0 report reached the controller after its stage-3 report (stage 0's reporter held up),
    then mb 6280's stage-0 report came: the rule asserted on a queue head. Reports are keyed by mb_id."""
    r = DispatchRule(4, 3.0)
    for k in range(3):
        r.dispatched(k, k * MS, 4096)
    done(r, 0, [(0, 10), (10, 20), (20, 30), (30, 40)])
    done(r, 1, [(10, 20), (20, 30), (30, 40), (40, 50)], stages=(1, 2, 3))     # stage 0's report of 1 is late
    assert 1 not in r.flight and list(r.on_stage0) == [2]
    done(r, 2, [(20, 30), (30, 40), (40, 50), (50, 60)], stages=(0,))
    done(r, 1, [(10, 20), (20, 30), (30, 40), (40, 50)], stages=(0,))          # the late one: ignored
    assert not r.on_stage0 and list(r.flight) == [2]
    assert r.decide(61 * MS)[0] in (NOW, WAIT)


def test_reports_in_any_order_across_stages():
    """Jittered stage times, stage-0 reports sometimes held 50-500 ms (behind a slow forward on the host), the other
    stages' 0.1-3 ms: the pipeline runs to the end, at most 4 in flight, the rule's bookkeeping empty after."""
    import random
    rnd = random.Random(11)
    table = {(k, s): 10.0 * rnd.uniform(0.7, 1.3) for k in range(300) for s in range(4)}
    delay = {(k, s): (rnd.uniform(50, 500) if s == 0 and rnd.random() < 0.05 else rnd.uniform(0.1, 3.0))
             for k in range(300) for s in range(4)}
    rule = DispatchRule(4, 3.0)
    s0, peak, _ = simulate(rule, n_mb=300, actual=lambda k, s: table[(k, s)], report_delay=lambda k, s: delay[(k, s)])
    assert len(s0) == 300 and peak <= 4
    assert not rule.flight and not rule.on_stage0


def test_host_time_from_worker_stamps_on_every_microbatch():
    """With the worker's stamps, a plan formed while stage 0 was busy is a sample too: delivery + prepare + launch,
    without the time the plan waited for the GPU (job 8242: the old rule saw almost no samples once saturated)."""
    r = DispatchRule(4, 3.0)
    r.dispatched(0, 0, 8192)
    r.stage_done(0, 0, int(20 * MS), int(100 * MS), (int(1 * MS), int(1 * MS), int(16 * MS), int(16 * MS), 0))
    assert r.host.predict_ns(8192) == pytest.approx(20 * MS)      # 1 delivery + 15 prepare + 4 launch
    r.dispatched(1, int(50 * MS), 8192)            # formed while mb 0 computes; prepared by 70, GPU free at 100
    r.stage_done(0, 1, int(102 * MS), int(180 * MS), (int(51 * MS), int(51 * MS), int(70 * MS), int(70 * MS), 0))
    # sample 1 + 19 + 2 (start - previous end) = 22 ms, not the 52 ms from forming to start
    assert r.host.predict_ns(8192) == pytest.approx(0.75 * 20 * MS + 0.25 * 22 * MS)


def test_host_time_leaves_out_the_in_flight_gate_wait():
    """A stage-0 start set by the gate (microbatch k - max_in_flight leaving the last stage) is a GPU wait: the
    projection already starts the plan there, so the host time does not count it (ev_st10_full_n300: the gate set 84%
    of starts and inflated the lead 14.4 -> ~10 ms needed)."""
    r = DispatchRule(4, 3.0)
    for k in range(4):
        r.dispatched(k, k * int(MS), 4096)
    for k in range(4):                                # stage 0 reports, no stamps
        r.stage_done(0, k, int((10 + 30 * k) * MS), int((40 + 30 * k) * MS))
    r.stage_done(3, 0, int(150 * MS), int(200 * MS))  # mb 0 leaves the pipeline at 200 ms
    r.dispatched(4, int(150 * MS), 4096)              # prepared by 155; stage 0 free at 130; the gate opens at 200
    before = r.host.predict_ns(4096)
    r.stage_done(0, 4, int(200 * MS) + 10_000, int(230 * MS), (int(151 * MS), int(151 * MS), int(155 * MS), int(155 * MS), 0))
    # 1 delivery + 4 prepare + 0.01 launch after the gate, not + 45 ms waiting for it
    want = int(1 * MS) + int(4 * MS) + 10_000
    assert r.host.predict_ns(4096) == pytest.approx(want if before is None else 0.75 * before + 0.25 * want)


def test_plans_form_by_the_host_time_bound_at_the_recent_largest_size():
    """Forming allows the host time's mean plus two deviations, at the largest of the last few plans' sizes."""
    from ssengine.runtime.dispatch import DispatchRule, HostPredictor
    h = HostPredictor()
    for ns in (10 * MS, 20 * MS, 10 * MS, 20 * MS):
        h.observe(32768, ns)
    assert h.bound_ns(32768) > h.predict_ns(32768) + 5 * MS
    r = DispatchRule()
    r.host.observe(32768, 20 * MS)
    r.host.observe(512, 1 * MS)
    for i, t in enumerate((32768, 512, 512)):
        r.dispatched(i, 0, t)
    assert r.form_host_ns() == pytest.approx(20 * MS)
