"""One GPU's mover (ppimpl/kvpool/mover.py) on CPU (memmove): pieces of two segments (KV slots, state slots) written to
host slots and read back are byte-equal through direct and two hops at several ring shapes; the segments' slots are
cut into ring chunks together, and `ready` is called before each chunk's host slots are read."""

import numpy as np
import pytest
import torch

from ppimpl.kvpool.mover import Mover, Segment

SB = 4096


def segments(dev, host, rng, local: bool = False):
    """Two segments over `host`'s slots: 5 KV slots of 3 pieces (2,816 B used of each), 2 state slots of pieces
    split at the slot boundary; device pieces anywhere in `dev`, 16-byte aligned. `local`: each KV slot's middle
    piece and the second state piece are in this GPU's own HBM (straight between host and HBM, not via the ring)."""
    base, hbase = dev.ctypes.data, host.ctypes.data + (-host.ctypes.data) % SB
    kv_len = np.array([1024, 768, 1024] * 5, dtype=np.int64)
    kv_off = (np.repeat(np.arange(5), 3) * SB + np.tile([0, 1024, 1792], 5)).astype(np.int64)
    st_len = np.array([3000, 1096, 2000], dtype=np.int64)
    st_off = np.array([0, 3000, SB], dtype=np.int64)
    pos = rng.permutation(len(dev) // 4096)[:18] * 4096 + base
    kv_loc, st_loc = (np.tile([False, True, False], 5), np.array([False, True, False])) if local else (None, None)
    return [Segment([hbase + s * SB for s in (6, 2, 0, 4, 8)], 2816, pos[:15], kv_off, kv_len, kv_loc),
            Segment([hbase + s * SB for s in (1, 9)], SB, pos[15:], st_off, st_len, st_loc)]


@pytest.mark.parametrize("local", [False, True], ids=["all_peer", "some_local"])
@pytest.mark.parametrize("ring", [dict(hops="direct"), dict(), dict(ring_depth=1, ring_chunk_slots=1),
                                  dict(ring_depth=2, ring_chunk_slots=3)], ids=lambda r: str(r) or "default")
def test_write_then_read_is_byte_equal(ring, local):
    rng = np.random.default_rng(0)
    dev = rng.integers(0, 255, 64 * 4096, dtype=np.uint8)
    host = np.zeros(11 * SB, dtype=np.uint8)
    segs = segments(dev, host, rng, local)
    want = dev.copy()
    m = Mover(torch.device("cpu"), SB, **ring)
    assert (m.ring is None) == (ring.get("hops") == "direct")
    ev, nbytes = m.transfer_async("write", segs)
    assert ev is None and nbytes == sum(int(s.lens.sum()) for s in segs)
    hb = host.ctypes.data
    assert all(np.array_equal(host[h - hb:h - hb + n], dev[p - dev.ctypes.data:p - dev.ctypes.data + n])
               for s in segs for p, o, n in zip(s.dev.tolist(), s.off.tolist(), s.lens.tolist())
               for h in [s.host_ptrs[o // SB] + o % SB])          # each piece at its host place
    dev[:] = 0
    calls = []
    m.transfer_async("read", segs, ready=lambda i, k: calls.append((i, k)))
    at = [(p - dev.ctypes.data, n) for s in segs for p, n in zip(s.dev.tolist(), s.lens.tolist())]
    assert all(np.array_equal(dev[a:a + n], want[a:a + n]) for a, n in at)
    if ring.get("hops") == "direct":
        assert calls == [(0, 5), (1, 2)]
    else:
        c = ring.get("ring_chunk_slots", 4)      # chunks over both segments' 7 slots in order
        want = []
        for a in range(0, 7, c):
            b = min(a + c, 7)
            want += ([(0, min(b, 5))] if a < 5 else []) + ([(1, b - 5)] if b > 5 else [])
        assert calls == want


@pytest.mark.parametrize("gated", [False, True])
def test_a_transfer_holds_its_direction_unless_it_yields(gated):
    """A transfer without a gate holds its direction from its first ring chunk to its last: a second one submitted
    meanwhile from another thread starts only after it (reads complete in order, not chunk by chunk interleaved). A
    gated transfer (a write yielding to reads) takes the direction per chunk, so another can go between its chunks."""
    import threading

    rng = np.random.default_rng(1)
    dev = rng.integers(0, 255, 64 * 4096, dtype=np.uint8)
    host = np.zeros(11 * SB, dtype=np.uint8)
    segs = segments(dev, host, rng)
    m = Mover(torch.device("cpu"), SB, ring_depth=2, ring_chunk_slots=1)
    order, between, go = [], threading.Event(), threading.Event()

    def first_ready(i, k):
        order.append(("a", i, k))
        if len(order) == 2:                  # mid-transfer: let the other one try, then go on
            between.set()
            go.wait(0.5)

    gate = (lambda: None) if gated else None
    t = threading.Thread(target=m.transfer_async, args=("read", segs, first_ready), kwargs=dict(gate=gate))
    t.start()
    assert between.wait(5)
    u = threading.Thread(target=m.transfer_async, args=("read", segs[1:], lambda i, k: order.append(("b", i, k))))
    u.start()
    u.join(0.2)
    go.set()
    t.join(5)
    u.join(5)
    firsts = [o[0] for o in order]
    if gated:
        assert firsts[:3] == ["a", "a", "b"]              # b went between a's chunks
    else:
        assert firsts == ["a"] * 7 + ["b"] * 2            # a's 7 chunks, then b's


def test_runs_merge_contiguous_pieces_within_a_slot_only():
    """`_runs`: pieces contiguous on both sides merge, but never across slots (adjacent host slots may sit in two
    registered allocations: one copy must not span them)."""
    from ppimpl.kvpool.mover import _runs
    src = np.array([0, 100, 300, 400, 500], dtype=np.int64)
    dst = np.array([1000, 1100, 1300, 1400, 1500], dtype=np.int64)
    lens = np.array([100, 100, 100, 100, 100], dtype=np.int64)
    slot = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    s, d, n = _runs(src, dst, lens, slot)
    assert s.tolist() == [0, 300, 500] and d.tolist() == [1000, 1300, 1500] and n.tolist() == [200, 200, 100]


def _random_segments(rng, base: int, hbase: int):
    """2-3 segments of 1-9 slots, 1-5 pieces per slot (some contiguous on both sides, so runs merge), random own-GPU
    flags; device pieces anywhere."""
    segs, nxt = [], 0
    for _ in range(rng.integers(2, 4)):
        nslots = int(rng.integers(1, 10))
        hosts = [hbase + int(s) * SB for s in rng.permutation(64)[:nslots]]
        dev, off, lens = [], [], []
        for k in range(nslots):
            at = 0
            for _ in range(int(rng.integers(1, 6))):
                n = int(rng.integers(1, 6)) * 64
                if at + n > SB:
                    break
                dev.append(base + nxt if rng.random() < 0.5 else base + int(rng.integers(0, 1 << 20)) * 16)
                nxt += n
                off.append(k * SB + at)
                lens.append(n)
                at += n + (0 if rng.random() < 0.5 else 64)
        local = rng.random(len(lens)) < 0.3 if rng.random() < 0.7 else None
        segs.append(Segment(hosts, SB, np.array(dev, np.int64), np.array(off, np.int64), np.array(lens, np.int64),
                            local))
    return segs


@pytest.mark.parametrize("ring", [dict(hops="direct"), dict(), dict(ring_depth=1, ring_chunk_slots=1),
                                  dict(ring_depth=2, ring_chunk_slots=3), dict(ring_depth=3, ring_chunk_slots=2)],
                         ids=lambda r: str(r) or "default")
def test_rust_mover_issues_the_same_batches(ring):
    """tier_rs's mover (the Rust engine) issues the same batches as this one for the same transfer: stream, and
    every copy's source, destination and length in order (ring chunks, runs per slot, own-GPU pieces), both
    directions, gated or not, over several transfers sharing the ring's chunk positions."""
    tier_rs = pytest.importorskip("tier_rs")
    rng = np.random.default_rng(3)
    m = Mover(torch.device("cpu"), SB, **ring)
    ring_ptr = None if m.ring is None else m.ring.ctypes.data
    recs: list = []
    m._batch = lambda stream, src, dst, lens, *a, **k: recs.append(
        (stream, np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(lens).tolist()))
    rust_pos = {"read": 0, "write": 0}
    for trial in range(40):
        segs = _random_segments(rng, 1 << 40, 1 << 30)
        d = "read" if trial % 2 else "write"
        gated = bool(trial % 3 == 0)
        recs.clear()
        m.transfer_async(d, segs, gate=(lambda: None) if gated else None)
        # the Rust mover starts at ring chunk 0: replay the Python ring position by leading transfers of one slot
        seg_args = [(list(s.host_ptrs), s.dev.tolist(), s.off.tolist(), s.lens.tolist(),
                     None if s.local is None else [bool(x) for x in s.local]) for s in segs]
        got = tier_rs.plan_transfer(d, seg_args, SB, ring.get("hops", "two"), ring.get("ring_depth", 2),
                                    ring.get("ring_chunk_slots", 4), ring_ptr, gated, rust_pos[d])
        assert [tuple(g) for g in got] == recs, (trial, d, gated)
        if m.ring is not None:
            rust_pos[d] = m._ring_next[d]
