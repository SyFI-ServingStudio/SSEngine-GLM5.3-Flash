"""research/bench/ttft_attrib.py: a round's TTFT split along its critical path sums back to its TTFT."""
from research.bench.ttft_attrib import EXTRA, PARTS, Busy, split_round


def test_split_round_parts_sum_to_ttft():
    # mb 0 (another round) and mbs 1, 2 (this round's two chunks) through 4 stages, 10 ms each, 1 ms hops
    stages = [{0: (0, 10), 1: (12, 22), 2: (22, 32)}]
    for k in (1, 2, 3):
        prev = stages[-1]
        stages.append({m: (prev[m][1] + 1 + (3 if (k, m) == (2, 2) else 0), prev[m][1] + 11 + (3 if (k, m) == (2, 2) else 0))
                       for m in prev})
    busy = [Busy([(a, b, m) for m, (a, b) in st.items()]) for st in stages]
    token = stages[3][2][1] + 2
    o = split_round(5, 8, [1, 2], stages, busy, token)
    assert abs(sum(o[k] for k in PARTS) - (token - 5)) < 1e-9
    assert o["queue"] == 3 and o["s0_wait_dispatched"] == 4
    assert o["s0_own"] == 20 and o["s0_other"] == 0 and o["s0_idle"] == 0
    assert o["pre_s0_busy"] == 5 and o["pre_s0_idle"] == 2 and o["pre_s0_microbatches"] == 0
    assert o["s1_own"] == 11 and o["s1_idle"] == 0 and o["s1_other"] == 0   # 1 ms behind its own first chunk
    # stage 2: mb 2 starts 4 ms after its stage-1 end (43 -> 47); its first chunk runs there until 44: own, then idle
    assert o["s2_own"] == 11 and o["s2_other"] == 0 and o["s2_idle"] == 3
    assert o["tail"] == 2
    assert abs(o["pre_s0_busy"] + o["pre_s0_idle"] - (o["queue"] + o["s0_wait_dispatched"])) < 1e-9
    assert set(EXTRA) <= set(o)
