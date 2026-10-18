"""One GPU's mover (ppimpl/kvpool/mover.py) on CPU (memmove): pieces of two segments (KV slots, state slots) written to
host slots and read back are byte-equal through direct and two hops at several ring shapes; the segments' slots are
cut into ring chunks together, and `ready` is called before each chunk's host slots are read."""

import numpy as np
import pytest
import torch

from ppimpl.kvpool.mover import Mover, Segment

SB = 4096


def segments(dev, host, rng):
    """Two segments over `host`'s slots: 5 KV slots of 3 pieces (2,816 B used of each), 2 state slots of pieces
    split at the slot boundary; device pieces anywhere in `dev`, 16-byte aligned."""
    base, hbase = dev.ctypes.data, host.ctypes.data + (-host.ctypes.data) % SB
    kv_len = np.array([1024, 768, 1024] * 5, dtype=np.int64)
    kv_off = (np.repeat(np.arange(5), 3) * SB + np.tile([0, 1024, 1792], 5)).astype(np.int64)
    st_len = np.array([3000, 1096, 2000], dtype=np.int64)
    st_off = np.array([0, 3000, SB], dtype=np.int64)
    pos = rng.permutation(len(dev) // 4096)[:18] * 4096 + base
    return [Segment([hbase + s * SB for s in (6, 2, 0, 4, 8)], 2816, pos[:15], kv_off, kv_len),
            Segment([hbase + s * SB for s in (1, 9)], SB, pos[15:], st_off, st_len)]


@pytest.mark.parametrize("ring", [dict(hops="direct"), dict(), dict(ring_depth=1, ring_chunk_slots=1),
                                  dict(ring_depth=2, ring_chunk_slots=3)], ids=lambda r: str(r) or "default")
def test_write_then_read_is_byte_equal(ring):
    rng = np.random.default_rng(0)
    dev = rng.integers(0, 255, 64 * 4096, dtype=np.uint8)
    host = np.zeros(11 * SB, dtype=np.uint8)
    segs = segments(dev, host, rng)
    want = dev.copy()
    m = Mover(torch.device("cpu"), SB, **ring)
    assert (m.ring is None) == (ring.get("hops") == "direct")
    ev, nbytes = m.transfer_async("write", segs)
    assert ev is None and nbytes == sum(int(s.lens.sum()) for s in segs)
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
