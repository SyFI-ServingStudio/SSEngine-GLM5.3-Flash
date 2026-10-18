"""The pool's members on CPU (ppimpl/kvpool/member.py): the leader (slot mode) decides the ops and `tier_move` turns
them into what the members get, four members (one per quarter, all four fake stages' views in this process) run them
into their own DRAM arenas. A context written at the end of round 0, evicted from HBM and read back into other blocks
is byte-equal; growth rewrites only the new blocks; the fake member reports ops after their bandwidth time."""

import os
import time
from types import SimpleNamespace as NS

import msgspec
import pytest
import torch

from ppimpl.config import BLOCK_TOKENS as B, TierConfig
from ppimpl.engine.scheduler import Request
from ppimpl.kvpool import BalancedKVPool, KVViews, shards
from ppimpl.kvpool.member import PoolMember, tier_move
from ppimpl.kvpool.placement import BalancedPlacement
from ppimpl.runtime.transport import TierBatch, decode_tier, encode
from test_kvpool_placement import block_bytes, stages


def members(nslots: int | None = 40, ssd_slots: int | None = None, staging_slots: int = 0, ssd_dir=None, **ring):
    """Four members over the fake stages, with DRAM arenas of `nslots` slots (SSD files of `ssd_slots`), and the
    leader's config. `ring`: TierConfig's hops / ring_depth / ring_chunk_slots."""
    views, ranges = stages()
    L = BalancedPlacement(ranges, 4, 0)

    def gb(n):
        return None if n is None else (n + 0.5) * L.slot_bytes / 1e9
    tc = TierConfig(dram_gb=gb(nslots), ssd_gb=gb(ssd_slots), staging_gb=gb(staging_slots), slot_bytes=L.slot_bytes,
                    state_slots=L.state_slots, ssd_dir=None if ssd_dir is None else str(ssd_dir), **ring)
    cfg = NS(tiers=tc, num_stages=4, model="glm53", run_dir=str(ssd_dir))
    if ssd_slots is not None:                   # the shard set (kept across runs: made once)
        shards.create(str(ssd_dir), 1, 4, ssd_slots and gb(ssd_slots))
        tc = msgspec.structs.replace(tc, ssd_dir=os.path.join(ssd_dir, "set0"))
        cfg = NS(tiers=tc, num_stages=4, model="glm53", run_dir=str(ssd_dir))
    mems = []
    for q in range(4):
        e = PoolMember(cfg, q, torch.device("cpu"), None)
        assert e.attach(ranges) == (L.slot_bytes, L.state_slots)
        assert nslots is None or e.dram.nbytes >= nslots * L.slot_bytes
        mems.append(e)
    return views, ranges, mems, tc


def run(mems, ops):
    """Every member runs the batch (as received over the wire); returns the bytes each moved per op."""
    msg = decode_tier(encode(TierBatch([tier_move(o) for o in ops])))
    moved = {}
    for e in mems:
        for m in msg.ops:
            e.submit(m)
            if m.tier == "ssd":                      # the SSD thread's work, here
                e._ssd_op(*e.ssd_q[e.direction(m)].get_nowait())
                if m.op == "read":                   # and the copy thread's
                    e._ssd_read_copy(*e.ssd_copy_q.get_nowait())
            op_id, wait, _, nbytes, _, _ = e.q[e.direction(m)].get(timeout=30)   # SSD writes: reported async
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
    views, ranges, mems, tc = members(**ring)
    assert (mems[0].mover.ring is None) == (ring.get("hops") == "direct")
    pool = BalancedKVPool(tc)
    assert pool.manager.tiers[0].num_slots == 40
    tokens = 2 * B + 77
    r0 = Request("s_r0", tokens, session="s", token_ids=None)
    a0 = KVViews([3, 7, 1], [2, 4, 6, 8, 9, 10], tokens)
    want = block_bytes(views, ranges, a0.kv_blocks, a0.state_blocks)
    h = pool.write_back("s", tokens, None, a0)                    # HBM lets it go: DRAM is its victim tier
    ops = pool.ready_ops()
    assert [o.op for o in ops] == ["write"] and h.ops == {ops[0].op_id}
    moved = run(mems, ops)
    st = sum(r.nbytes for r in ranges if r.kind == "state")
    kv = sum(r.nbytes for r in ranges if r.kind == "kv")
    assert sum(moved[ops[0].op_id]) == 3 * kv + st                 # the four quarters: every byte once
    assert pool.complete(ops[0].op_id, 0) == (ops[0], h) and h.done()
    scramble(views)                                                # HBM lost the context
    r1 = Request("s_r1", tokens + 50, session="s", prefix_tokens=tokens)
    a1 = KVViews([5, 0 + 11, 2, 9], [1, 3, 5, 7, 11, 0], tokens)
    hit = pool.lookup("s", r1)
    assert (hit.tier, hit.tokens) == ("dram", tokens)
    h = pool.load("s", hit, a1, 0, r1.req_id)
    ops = pool.ready_ops()
    assert [o.op_id for o in ops] == list(h.ops) and ops[0].op == "read"
    run(mems, ops)
    assert pool.complete(ops[0].op_id, 7)[1] is h and h.t_land == 7
    assert block_bytes(views, ranges, a1.kv_blocks[:3], a1.state_blocks) == want
    # round 1 ends at 3 B + 10: blocks 0, 1 kept in DRAM, blocks 2, 3 and the state written
    r1.num_tokens = 3 * B + 10
    pool.write_back("s", r1.num_tokens, None, KVViews(a1.kv_blocks, a1.state_blocks, r1.num_tokens))
    (w,) = pool.ready_ops()
    assert w.block_start == 2 and tier_move(w).b_lo == 2 and tier_move(w).b_hi == 4
    moved = run(mems, [w])
    assert sum(moved[w.op_id]) == 2 * kv + st
    pool.complete(w.op_id, 0)
    want = block_bytes(views, ranges, a1.kv_blocks, a1.state_blocks)
    scramble(views)
    r2 = Request("s_r2", 3 * B + 99, session="s", prefix_tokens=3 * B + 10)
    a2 = KVViews([8, 6, 4, 10], [6, 5, 4, 3, 2, 1], 3 * B + 10)
    h = pool.load("s", pool.lookup("s", r2), a2, 0)
    (rd,) = pool.ready_ops()
    run(mems, [rd])
    pool.complete(rd.op_id, 0)
    assert h.done() and block_bytes(views, ranges, a2.kv_blocks, a2.state_blocks) == want


def drive(pool, mems):
    """Run every op in dependency order; returns them."""
    done = []
    while ops := pool.ready_ops():
        run(mems, ops)
        for o in ops:
            pool.complete(o.op_id, 0)
        done += ops
    assert not pool.manager.waiting
    return done


@pytest.mark.parametrize("ring", [dict(), dict(ring_depth=2, ring_chunk_slots=1)], ids=["default", "chunk1"])
def test_ssd_demote_and_staging_are_byte_equal(tmp_path, ring):
    """DRAM holds one 2-block context: Y's write-back demotes X to SSD (DRAM slots -> file, io_uring); X's read comes
    back through the staging arena into other HBM blocks, Y's from DRAM. SSD only: writes and reads through the
    staging arena. Every context byte-equal; the files are deleted at close."""
    T = 2 * B - 5
    X, Y = KVViews([1, 2], [3, 4, 5, 6, 7, 8], T), KVViews([9, 10], [0, 9, 10, 11, 1, 2], T)
    for dram in (None, "one"):
        d = tmp_path / str(dram)
        ss = BalancedPlacement(stages()[1], 4, 0).state_slots
        views, ranges, mems, tc = members(None if dram is None else 2 + ss, 60, 30, d, dram_write_ahead=None,
                                          **ring)   # DRAM: one context, demoted when evicted
        pool = BalancedKVPool(tc)
        want = {}
        for name, a in (("x", X), ("y", Y)):
            want[name] = block_bytes(views, ranges, a.kv_blocks, a.state_blocks)
            pool.write_back(name, T, None, a)
            kinds = [o.op for o in drive(pool, mems)]
            assert kinds == (["write"] if dram is None or name == "x" else ["demote", "write"]), kinds
        scramble(views)
        for name, a in (("x", KVViews([5, 6], [0, 1, 2, 3, 4, 5], T)), ("y", KVViews([7, 8], [6, 7, 8, 9, 10, 11], T))):
            r = Request(f"{name}1", T + 9, session=name, prefix_tokens=T)
            hit = pool.lookup(name, r)
            assert (hit.tier, hit.tokens) == ("ssd" if dram is None or name == "x" else "dram", T)
            h = pool.load(name, hit, a, 0)
            mine = set(h.ops)
            ops = drive(pool, mems)
            (rd,) = [o for o in ops if o.op_id in mine]
            assert h.done() and rd.host_tier == ("staging" if hit.tier == "ssd" else None)
            assert [o.op for o in ops] == ["read"]
            assert block_bytes(views, ranges, a.kv_blocks, a.state_blocks) == want[name]
        paths = [e.ssd.path for e in mems]
        for e in mems:
            e.close()
        assert all(os.path.exists(x) for x in paths)                 # kept for the next run


def test_kept_ssd_file_is_never_served_stale(tmp_path):
    """A second run over the first run's shard set (same files, x's bytes still in them) starts with an empty index:
    x is found nowhere (computed again), and what it reads back is only what it wrote itself."""
    T = 2 * B - 5
    X, Y = KVViews([1, 2], [3, 4, 5, 6, 7, 8], T), KVViews([9, 10], [0, 9, 10, 11, 1, 2], T)
    views, ranges, mems, tc = members(None, 60, 30, tmp_path)
    pool = BalancedKVPool(tc)
    pool.write_back("x", T, None, X)
    drive(pool, mems)
    for e in mems:
        e.close()
    views, ranges, mems, tc = members(None, 60, 30, tmp_path)         # the next run: same files
    pool = BalancedKVPool(tc)
    assert pool.lookup("x", Request("x1", T + 9, session="x", prefix_tokens=T)) is None
    want = block_bytes(views, ranges, Y.kv_blocks, Y.state_blocks)
    pool.write_back("y", T, None, Y)                                 # takes the slots x had
    drive(pool, mems)
    scramble(views)
    a = KVViews([5, 6], [0, 1, 2, 3, 4, 5], T)
    r = Request("y1", T + 9, session="y", prefix_tokens=T)
    pool.load("y", pool.lookup("y", r), a, 0)
    drive(pool, mems)
    assert block_bytes(views, ranges, a.kv_blocks, a.state_blocks) == want
    for e in mems:
        e.close()


def test_member_threads_route_dram_ops_by_direction(tmp_path):
    """The real members' threads end to end (CPU): the receive thread hands DRAM ops to their direction's submit
    thread, the done threads report every op over the report socket, a context written and read back through them is
    byte-equal, and Shutdown ends every thread."""
    import zmq

    from ppimpl.runtime.transport import Shutdown, TierDone, decode_to_controller

    views, ranges, mems, tc = members()
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    addr = lambda s: f"ipc://{tmp_path}/tier{s}"
    for e in mems:
        e.ctx, e.cfg = ctx, NS(**vars(e.cfg), tier_addr=addr, report_addr=lambda: f"tcp://127.0.0.1:{port}")
        e.start()
    pushes = []
    for q in range(4):
        p = ctx.socket(zmq.PUSH)
        p.connect(addr(q))
        pushes.append(p)

    def send(ops):
        for p in pushes:
            p.send(encode(TierBatch([tier_move(o) for o in ops])))
        got = [decode_to_controller(pull.recv()) for _ in range(4 * len(ops))]
        assert all(isinstance(g, TierDone) for g in got)
        assert sorted((g.op_id, g.stage) for g in got) == sorted((o.op_id, q) for o in ops for q in range(4))

    pool = BalancedKVPool(tc)
    T = 2 * B + 77
    a = KVViews([3, 7, 1], [2, 4, 6, 8, 9, 10], T)
    want = block_bytes(views, ranges, a.kv_blocks, a.state_blocks)
    pool.write_back("s", T, None, a)
    ops = pool.ready_ops()
    send(ops)
    for o in ops:
        pool.complete(o.op_id, 0)
    scramble(views)
    b = KVViews([5, 11, 2, 9], [1, 3, 5, 7, 11, 0], T)
    h = pool.load("s", pool.lookup("s", Request("s_r1", T + 50, session="s", prefix_tokens=T)), b, 0)
    ops = pool.ready_ops()
    send(ops)
    for o in ops:
        pool.complete(o.op_id, 0)
    assert h.done() and block_bytes(views, ranges, b.kv_blocks[:3], b.state_blocks) == want
    for p in pushes:
        p.send(encode(Shutdown()))
        p.close(linger=1000)
    for e in mems:
        for th in e.threads:
            th.join(10)
            assert not th.is_alive(), th.name
    pull.close(linger=0)
    ctx.term()


def test_fake_member_reports_after_bandwidth_time():
    """The fake model's member: no bytes, each op done after slots x slot_bytes / bandwidth, one after another per
    direction; TierDone goes out over the report socket."""
    import zmq

    from ppimpl.runtime.transport import TierDone, TierMove, decode_to_controller

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    tc = TierConfig(dram_gb=1.0, ssd_gb=None, slot_bytes=10_000_000, dram_read_gb_s=1.0)
    cfg = NS(tiers=tc, num_stages=4, model="fake", report_addr=lambda: f"tcp://127.0.0.1:{port}")
    e = PoolMember(cfg, 2, torch.device("cpu"), ctx)
    assert e.attach(None) == (10_000_000, 3)
    for name in ("read", "write"):
        import threading
        th = threading.Thread(target=e._done_loop, args=(name,), daemon=True)
        th.start()
        e.threads.append(th)
    t0 = time.monotonic()
    # read of 2 blocks + 3 state slots = 50 MB at 1 GB/s: 50 ms; two of them back to back: 100 ms
    for i in range(2):
        e.submit(TierMove(i, "read", "dram", None, 0, 2, 0, [[1, 2]], [[1] * 6], [0, 1], [2, 3, 4], [], []))
    got = [decode_to_controller(pull.recv()) for _ in range(2)]
    dt = time.monotonic() - t0
    assert all(isinstance(g, TierDone) and g.stage == 2 and g.nbytes == 50_000_000 for g in got)
    assert [g.op_id for g in got] == [0, 1] and 0.09 < dt < 0.5
    assert all(g.start_ns <= g.submit_ns <= g.end_ns and g.gpu_ns is None for g in got)
    for q in e.q.values():
        q.put(None)
    for th in e.threads:
        th.join(5)
    pull.close(linger=0)
    ctx.term()


def test_ssd_writes_are_reported_when_their_last_file_write_is_done():
    """Writes and demotes do not hold the SSD write thread: each is reported once all its file writes completed, in
    completion order, and a failed one fails the run."""
    import queue
    import threading
    from concurrent.futures import Future

    from ppimpl.runtime.transport import WorkerError

    e = PoolMember.__new__(PoolMember)
    e.stage, e.q, e._ssd_writes, e._ssd_cv = 0, {"write": queue.Queue()}, 0, threading.Condition()
    a, b = [Future(), Future()], [Future()]
    e._report_when_done(NS(op_id=1, op="write"), a, 0, 10, "write")
    e._report_when_done(NS(op_id=2, op="write"), b, 0, 10, "write")
    assert e._ssd_writes == 2 and e.q["write"].empty()
    b[0].set_result(None)                                   # op 2 lands first
    a[0].set_result(None)
    assert e.q["write"].get_nowait()[0] == 2 and e.q["write"].empty()
    a[1].set_result(None)
    assert e.q["write"].get_nowait()[0] == 1 and e._ssd_writes == 0
    c = [Future(), Future()]
    e._report_when_done(NS(op_id=3, op="write"), c, 0, 10, "write")
    c[0].set_exception(OSError("disk"))
    c[1].set_result(None)
    assert isinstance(e.q["write"].get_nowait(), WorkerError) and e.q["write"].empty() and e._ssd_writes == 0
