"""Layer 4 of the KV pool (lanes/m6 TIERS.md §2.3, §3, §13.2): one GPU's copies between host slots and the stages' KV.

Copy engines only, no SMs, no wait on the compute stream: `cuMemcpyBatchAsync` (vLLM `ops.swap_blocks_batch`) on
the mover's own streams; on CPU (tests) `ctypes.memmove`, done at once.

Two hops (`TierConfig.hops`, default; §2.3: a copy engine writing host data straight into the other GPUs' HBM runs at
31.7 GB/s against 47.6 into its own): the host side of a slot is contiguous, so a read copies each slot whole into a
ring of chunks in this GPU's own HBM (one copy per slot, its own PCIe link), then one batch scatters the chunk's
pieces from the ring to the stages' KV over NVLink. Writes gather into the ring, then copy each slot to host. Each
direction has its ring (`ring_depth` chunks of `ring_chunk_slots` slots) and two streams (host side, NVLink side);
chunk k+1's host copy overlaps chunk k's scatter, and a ring chunk is reused once the copy that read it is done
(events). The ring is allocated after the model, from the HBM the KV pool (the contract's 7,144 blocks) leaves;
`info` says how much that is. `hops: direct` puts the pieces straight between host and the stages' KV (one batch per
transfer). Nothing above this layer sees the ring.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import NamedTuple

import numpy as np


def runq_ns() -> int:
    """This thread's time runnable but waiting for a CPU so far (/proc/thread-self/schedstat; 0 if unavailable)."""
    try:
        with open("/proc/thread-self/schedstat") as f:
            return int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0


class Segment(NamedTuple):
    """Host slots and the pieces that live in them. Piece i's host side is `off[i]` bytes into the slots counted
    `slot_bytes` apart (slot k at k x slot_bytes, k < len(host_ptrs)); `host_len` bytes of each slot are used."""
    host_ptrs: list[int]
    host_len: int
    dev: np.ndarray
    off: np.ndarray
    lens: np.ndarray


class Mover:
    """One GPU's copies (`TierConfig.hops`, `ring_depth`, `ring_chunk_slots`), host slots of `slot_bytes`."""

    def __init__(self, device, slot_bytes: int, hops: str = "two", ring_depth: int = 2, ring_chunk_slots: int = 4):
        self.device, self.cuda, self.sb = device, device.type == "cuda", slot_bytes
        self.ring = None
        self.info = "direct"
        self._lock = threading.Lock()              # ring chunks and their order, across the submitting threads
        if hops == "two":                          # per direction: `depth` chunks of `chunk` slots, after the KV pool
            self.depth, self.chunk = ring_depth, ring_chunk_slots
            half = self.depth * self.chunk * slot_bytes
            if self.cuda:
                import torch
                free = torch.cuda.mem_get_info(device)[0]
                self.ring = torch.empty(2 * half, dtype=torch.uint8, device=device)
                ring_ptr = self.ring.data_ptr()
                self.info = (f"two hops: ring 2 x {self.depth} x {self.chunk} slots = {2 * half / 1e9:.3f} GB "
                             f"of {free / 1e9:.2f} GB free HBM after the model and KV pool")
            else:
                self.ring = np.zeros(2 * half + 4096, dtype=np.uint8)
                ring_ptr = self.ring.ctypes.data
                self.info = f"two hops: ring 2 x {self.depth} x {self.chunk} slots"
            self._ring_base = {"read": ring_ptr, "write": ring_ptr + half}
            self._ring_next = {"read": 0, "write": 0}
            self._ring_free = {"read": [None] * self.depth, "write": [None] * self.depth}
        if self.cuda:
            import torch
            from vllm import _custom_ops
            self.swap = _custom_ops.swap_blocks_batch
            self.streams = {k: torch.cuda.Stream(device) for k in ("read", "write", "read_nvl", "write_nvl")}

    def mark(self, direction: str):
        """A timing event on the stream a transfer's first copy goes on (CUDA; None on CPU)."""
        if not self.cuda:
            return None
        import torch
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self.streams["write_nvl" if direction == "write" and self.ring is not None else direction])
        return ev

    def transfer_async(self, direction: str, segments: list[Segment], ready=None, tr: list | None = None):
        """"read" host -> HBM or "write" HBM -> host of every segment's pieces. Two hops: each segment is cut into ring
        chunks of its own; direct: one batch of all pieces. `ready(i, k)`: called before the host side of segment i's
        slots below k is read (an SSD read waits for its file reads there). `tr`: per-batch trace (`_batch`). Returns
        (the last copy's event, None on CPU; bytes of HBM pieces moved)."""
        sb = self.sb
        if self.ring is None:
            if ready is not None:
                for i, s in enumerate(segments):
                    ready(i, len(s.host_ptrs))
            host = np.concatenate([np.asarray(s.host_ptrs, dtype=np.int64)[s.off // sb] + s.off % sb
                                   for s in segments])
            dev = np.concatenate([s.dev for s in segments])
            lens = np.concatenate([s.lens for s in segments])
            src, dst = (host, dev) if direction == "read" else (dev, host)
            return self._batch(direction, src, dst, lens, tr=tr), int(lens.sum())
        C = self.chunk
        ev, nbytes = None, 0
        for i, s in enumerate(segments):
            slot = s.off // sb                      # non-decreasing: pieces never cross a slot (placement splits them)
            for a in range(0, len(s.host_ptrs), C):
                b = min(a + C, len(s.host_ptrs))
                if ready is not None:
                    ready(i, b)
                lo, hi = np.searchsorted(slot, [a, b])
                dev, roff, lens = s.dev[lo:hi], s.off[lo:hi] - a * sb, s.lens[lo:hi]
                n = b - a
                nbytes += int(lens.sum())
                with self._lock:
                    pos = self._ring_next[direction]
                    self._ring_next[direction] = (pos + 1) % self.depth
                    rbase = self._ring_base[direction] + pos * C * sb
                    rings = [rbase + k * sb for k in range(n)]
                    hl = np.full(n, s.host_len, dtype=np.int64)
                    if direction == "read":         # host -> ring (PCIe), then ring -> the stages (NVLink)
                        h = self._batch("read", s.host_ptrs[a:b], rings, hl, (self._ring_free[direction][pos],), tr)
                        ev = self._batch("read_nvl", roff + rbase, dev, lens, (h,), tr)
                    else:                           # the stages -> ring, then ring -> host
                        g = self._batch("write_nvl", dev, roff + rbase, lens, (self._ring_free[direction][pos],), tr)
                        ev = self._batch("write", rings, s.host_ptrs[a:b], hl, (g,), tr)
                    self._ring_free[direction][pos] = ev
        return ev, nbytes

    def _batch(self, stream: str, src, dst, lens, wait=(), tr: list | None = None):
        """One `cuMemcpyBatchAsync` on `stream` after the events `wait`; its end event (None on CPU, done). `tr`: append
        (stream, pieces, bytes, host ns before the call, host ns after it, thread CPU ns in it, ns runnable without a
        CPU in it, GPU start event, end event): the rest of the call is a wait for a lock (GIL or driver)."""
        if not self.cuda:
            for s, t, k in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(lens).tolist()):
                ctypes.memmove(t, s, k)
            return None
        import torch
        st = self.streams[stream]
        for ev in wait:
            if ev is not None:
                st.wait_event(ev)
        args = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)) for x in (src, dst, lens)]
        if tr is not None:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record(st)
            h0, c0, q0 = time.perf_counter_ns(), time.thread_time_ns(), runq_ns()
        with torch.cuda.stream(st):
            self.swap(*args)
        if tr is not None:
            h1, c1, q1 = time.perf_counter_ns(), time.thread_time_ns(), runq_ns()
        ev = torch.cuda.Event(blocking=True, enable_timing=True)   # timing: TierDone.gpu_ns
        ev.record(st)
        if tr is not None:
            tr.append((stream, len(args[2]), int(args[2].sum()), h0, h1, c1 - c0, q1 - q0, ev0, ev))
        return ev
