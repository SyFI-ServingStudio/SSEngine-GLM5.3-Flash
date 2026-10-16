"""Scheduling check chunk for chunk and force check (6_scheduler.md §7), on the CPU replay."""

import pytest
from bench.replay import check_force, check_sched

from ppimpl.config import SCHEDULER_STEPS


@pytest.mark.parametrize("step", list(SCHEDULER_STEPS))
def test_scheduling_check_chunk_for_chunk(step):
    diffs, ttft = check_sched(step)
    assert diffs == []
    assert ttft["_max_ttft_dev"] < 0.01


@pytest.mark.parametrize("step", list(SCHEDULER_STEPS))
def test_scheduling_check_unchanged_with_dispatch_ahead(step):
    """With 4 in flight on 4 stages, forming one microbatch ahead never changes a decision of this check."""
    assert check_sched(step, dispatch_ahead=1)[0] == []


@pytest.mark.parametrize("step", list(SCHEDULER_STEPS))
def test_force_check(step):
    diffs, got = check_force(step)
    assert diffs == []
    if step == "good":
        assert 60 <= got["long_ttft_s"] <= 65


@pytest.mark.parametrize("kind,variant", [("prefix", None), ("tier", "hbm_hit"), ("tier", "hbm"),
                                          ("tier", "dram_unbal"), ("tier", "ssd_unbal")])
def test_bringups(kind, variant):
    """5_prefix_cache.md §6 / 7_offloading.md §6 on the fit: the hit is exactly P, from the tier the run allows; read
    time and both rounds' TTFT within 10%."""
    from bench.replay import run_bringup
    for row in run_bringup(kind, variant):
        assert row["cached"] == [0 if variant == "hbm" else row["P"]]
        assert row["tier"] == [{"dram_unbal": "dram", "ssd_unbal": "ssd"}.get(variant, "None")]
        assert row["read_ms"] == pytest.approx(row["read_target"], rel=0.01)
        assert row["round0_ms"] == pytest.approx(row["round0_target"], rel=0.10)
        assert row["round1_ms"] == pytest.approx(row["round1_target"], rel=0.10)
