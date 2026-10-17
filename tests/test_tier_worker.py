"""The workers' tier engines on CPU (ppimpl/runtime/tier_worker.py): the controller's index (slot mode) decides the ops,
`tier_move` turns them into what the workers get, four engines (one per quarter, all four fake stages' views in this
process) run them into their own DRAM arenas. A context written at the end of round 0, evicted from HBM and read
back into other blocks is byte-equal; growth rewrites only the new blocks; the fake engine reports ops after their
bandwidth time."""

import ctypes
import os
import time
from types import SimpleNamespace as NS

import pytest
import torch

from ppimpl.config import BLOCK_TOKENS as B, TierConfig
from ppimpl.engine.kv import Allocation
from ppimpl.engine.scheduler import Request
from ppimpl.engine.tiers import Tiers
from ppimpl.runtime.controller import tier_move
from ppimpl.runtime.tier_copy import Layout
from ppimpl.runtime.tier_worker import TierEngine
from ppimpl.runtime.transport import TierBatch, decode_tier, encode
from test_tier_copy import block_bytes, stages


def engines(nslots: int | None = 40, ssd_slots: int | None = None, staging_slots: int = 0, ssd_dir=None, **ring):
    """Four engines over the fake stages, with DRAM arenas of `nslots` slots (SSD files of `ssd_slots`), and the
    index's config. `ring`: TierConfig's hops / ring_depth / ring_chunk_slots."""
    views, ranges = stages()
    L = Layout(ranges, 0, 4)

    def gb(n):
        return None if n is None else (n + 0.5) * L.slot_bytes / 1e9
    tc = TierConfig(dram_gb=gb(nslots), ssd_gb=gb(ssd_slots), staging_gb=gb(staging_slots), slot_bytes=L.slot_bytes,
                    state_slots=L.state_slots, ssd_dir=None if ssd_dir is None else str(ssd_dir), **ring)
    cfg = NS(tiers=tc, num_stages=4, model="glm53", run_dir=str(ssd_dir))
    engs = []
    for q in range(4):
        e = TierEngine(cfg, q, torch.device("cpu"), None)
        assert e.attach(ranges) == (L.slot_bytes, L.state_slots)
        assert nslots is None or e.dram.nbytes >= nslots * L.slot_bytes
        engs.append(e)
    return views, ranges, engs, tc


def run(engs, ops):
    """Every engine runs the batch (as received over the wire); returns the bytes each moved per op."""
    msg = decode_tier(encode(TierBatch([tier_move(o) for o in ops])))
    moved = {}
    for e in engs:
        for m in msg.ops:
            e.submit(m)
            if m.tier == "ssd":                      # the SSD thread's work, here
                e._ssd_op(*e.ssd_q[e.direction(m)].get_nowait())
            op_id, wait, _, nbytes = e.q[e.direction(m)].get_nowait()
            assert op_id == m.op_id and wait is None
            moved.setdefault(op_id, []).append(nbytes)
    return moved


def scramble(views):
    for v in views:
        v.view(torch.uint8).random_(0, 255) if v.dtype == torch.uint8 else v.normal_()


RINGS = [dict(hops="direct"), dict(), dict(ring_depth=2, ring_chunk_slots=1), dict(ring_depth=1, ring_chunk_slots=2),
         dict(ring_depth=4, ring_chunk_slots=1)]


@pytest.mark.parametrize("ring", RINGS, ids=lambda r: "-".join(f"{k}={v}" for k, v in r.items()) or "default")
def test_write_evict_read_back_is_byte_equal_and_growth_moves_only_new_blocks(ring):
    views, ranges, engs, tc = engines(**ring)
    assert (engs[0].ring is None) == (ring.get("hops") == "direct")
    t = Tiers(tc)
    assert t.tiers[0].num_slots == 40
    tokens = 2 * B + 77
    r0 = Request("s_r0", tokens, session="s", token_ids=None)
    a0 = Allocation([3, 7, 1], [2, 4, 6, 8, 9, 10])
    want = block_bytes(views, ranges, a0.kv_blocks, a0.state_blocks)
    t.store(r0, a0)
    ops = t.ready_ops()
    assert [o.op for o in ops] == ["write"]
    moved = run(engs, ops)
    st = sum(r.nbytes for r in ranges if r.kind == "state")
    kv = sum(r.nbytes for r in ranges if r.kind == "kv")
    assert sum(moved[ops[0].op_id]) == 3 * kv + st                 # the four quarters: every byte once
    for o in ops:
        t.op_done(o.op_id)
    scramble(views)                                                # HBM lost the context
    r1 = Request("s_r1", tokens + 50, session="s", prefix_tokens=tokens)
    a1 = Allocation([5, 0 + 11, 2, 9], [1, 3, 5, 7, 11, 0])
    assert t.lookup(r1) == (0, tokens)
    rd = t.read(r1, 0, tokens, 0, a1, 0)
    ops = t.ready_ops()
    assert ops == [rd]
    run(engs, ops)
    t.op_done(rd.op_id)
    assert block_bytes(views, ranges, a1.kv_blocks[:3], a1.state_blocks) == want
    # round 1 ends at 3 B + 10: blocks 0, 1 kept in DRAM, blocks 2, 3 and the state written
    r1.num_tokens = 3 * B + 10
    t.store(r1, Allocation(a1.kv_blocks, a1.state_blocks))
    (w,) = t.ready_ops()
    assert w.block_start == 2 and tier_move(w).b_lo == 2 and tier_move(w).b_hi == 4
    moved = run(engs, [w])
    assert sum(moved[w.op_id]) == 2 * kv + st
    t.op_done(w.op_id)
    want = block_bytes(views, ranges, a1.kv_blocks, a1.state_blocks)
    scramble(views)
    r2 = Request("s_r2", 3 * B + 99, session="s", prefix_tokens=3 * B + 10)
    a2 = Allocation([8, 6, 4, 10], [6, 5, 4, 3, 2, 1])
    rd = t.read(r2, 0, 3 * B + 10, 0, a2, 0)
    run(engs, t.ready_ops())
    t.op_done(rd.op_id)
    assert block_bytes(views, ranges, a2.kv_blocks, a2.state_blocks) == want


def drive(t, engs):
    """Run every op in dependency order; returns them."""
    done = []
    while ops := t.ready_ops():
        run(engs, ops)
        for o in ops:
            t.op_done(o.op_id)
        done += ops
    assert not t.waiting
    return done


def snapshot(views, ranges, kv, st):
    return block_bytes(views, ranges, kv, st)


@pytest.mark.parametrize("ring", [dict(), dict(ring_depth=2, ring_chunk_slots=1)], ids=["default", "chunk1"])
def test_ssd_demote_copy_up_and_staging_are_byte_equal(tmp_path, ring):
    """DRAM holds one 2-block context: Y's write demotes X to SSD (DRAM slots -> file); X's read copies it up
    through new DRAM slots (demoting Y) into other HBM blocks; Y then reads from SSD. SSD only: writes and reads
    through the staging arena. Every context byte-equal; the files are deleted at close."""
    T = 2 * B - 5
    X, Y = Allocation([1, 2], [3, 4, 5, 6, 7, 8]), Allocation([9, 10], [0, 9, 10, 11, 1, 2])
    for dram in (None, "one"):
        d = tmp_path / str(dram)
        ss = Layout(stages()[1], 0, 4).state_slots
        views, ranges, engs, tc = engines(None if dram is None else 2 + ss, 60, 30, d, **ring)   # DRAM: one context
        t = Tiers(tc)
        want = {}
        for name, a in (("x", X), ("y", Y)):
            want[name] = snapshot(views, ranges, a.kv_blocks, a.state_blocks)
            t.store(Request(f"{name}0", T, session=name), a)
            kinds = [o.op for o in drive(t, engs)]
            assert kinds == (["write"] if dram is None or name == "x" else ["demote", "write"]), kinds
        scramble(views)
        for name, a in (("x", Allocation([5, 6], [0, 1, 2, 3, 4, 5])), ("y", Allocation([7, 8], [6, 7, 8, 9, 10, 11]))):
            r = Request(f"{name}1", T + 9, session=name, prefix_tokens=T)
            assert t.lookup(r) == (len(t.tiers) - 1, T)                # in SSD
            rd = t.read(r, len(t.tiers) - 1, T, 0, a, 0)
            ops = drive(t, engs)
            assert rd in ops and rd.host_tier == ("staging" if dram is None else "dram")
            assert [o.op for o in ops] == (["read"] if dram is None or name == "y" else ["demote", "read"])
            assert snapshot(views, ranges, a.kv_blocks, a.state_blocks) == want[name]
        paths = [e.ssd.path for e in engs]
        for e in engs:
            e.close()
        assert not any(os.path.exists(x) for x in paths)


def test_fake_engine_reports_after_bandwidth_time():
    """The fake model's engine: no bytes, each op done after slots x slot_bytes / bandwidth, one after another per
    direction; TierDone goes out over the report socket."""
    import zmq

    from ppimpl.runtime.transport import TierDone, TierMove, decode_to_controller

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    tc = TierConfig(dram_gb=1.0, ssd_gb=None, slot_bytes=10_000_000, dram_read_gb_s=1.0)
    cfg = NS(tiers=tc, num_stages=4, model="fake", report_addr=lambda: f"tcp://127.0.0.1:{port}")
    e = TierEngine(cfg, 2, torch.device("cpu"), ctx)
    assert e.attach(None) == (10_000_000, 3)
    for name in ("read", "write"):
        import threading
        th = threading.Thread(target=e._done_loop, args=(name,), daemon=True)
        th.start()
        e.threads.append(th)
    t0 = time.monotonic()
    # read of 2 blocks + 3 state slots = 50 MB at 1 GB/s: 50 ms; two of them back to back: 100 ms
    for i in range(2):
        e.submit(TierMove(i, "read", "dram", None, 0, 2, 0, [1, 2], [1] * 6, [0, 1], [2, 3, 4], [], []))
    got = [decode_to_controller(pull.recv()) for _ in range(2)]
    dt = time.monotonic() - t0
    assert all(isinstance(g, TierDone) and g.stage == 2 and g.nbytes == 50_000_000 for g in got)
    assert [g.op_id for g in got] == [0, 1] and 0.09 < dt < 0.5
    for q in e.q.values():
        q.put(None)
    for th in e.threads:
        th.join(5)
    pull.close(linger=0)
    ctx.term()


def test_arena_is_plain_memory_on_cpu():
    from ppimpl.runtime.tier_worker import HostArena
    a = HostArena(3 << 20, None, False)
    assert a.nbytes == 4 << 20
    ctypes.memset(a.ptr + a.nbytes - 1, 7, 1)
