"""Layer 4 of the KV pool (lanes/m6 TIERS.md §2.3, §3, §13.2): one GPU's copies between host slots and the stages' KV.

Copy engines only, no SMs, no wait on the compute stream: `cuMemcpyBatchAsync` (vLLM `ops.swap_blocks_batch`) on
the mover's own streams; on CPU (tests) `ctypes.memmove`, done at once.

Two hops (`TierConfig.hops`, default; §2.3: a copy engine writing host data straight into the other GPUs' HBM runs at
31.7 GB/s against 47.6 into its own): the host side of a slot is contiguous, so a read copies each slot whole into a
ring of chunks in this GPU's own HBM (one copy per slot, its own PCIe link), then one batch scatters the chunk's
pieces from the ring to the stages' KV over NVLink. Writes gather into the ring, then copy each slot to host. Each
direction has its ring (`ring_depth` chunks of `ring_chunk_slots` slots), two streams (host side, NVLink side) and a
lock, which a transfer holds from its first chunk to its last unless it yields per chunk (a write, `gate`): reads
complete in the order they were submitted, each at the link's rate; chunk k+1's host copy overlaps chunk k's scatter. A transfer's slots (all its segments in order) are cut into
chunks together, so per chunk there are two batch calls, their pointer arrays sliced from arrays built once per
transfer. Each ring chunk has two events, recorded again at each use (no event per batch): its first hop done (the
second hop waits for it) and its second hop done (the next first hop into the chunk waits for it). A transfer takes
one pooled event for its end (and one for its start mark, `mark`). The ring is allocated after the model, from the
HBM the KV pool (the contract's 7,144 blocks) leaves; `info` says how much that is. `hops: direct` puts the pieces
straight between host and the stages' KV (one batch per transfer). Nothing above this layer sees the ring.
"""

from __future__ import annotations

import contextlib
import ctypes
import threading
import time
from typing import NamedTuple

import numpy as np

EVENT_POOL = 1024  # pooled CUDA events per mover: an op takes 2 (start mark, end), + 2 per batch with the trace on
ANCHOR_S = 1.0     # a thread's GPU-to-host clock anchor is taken again after this long (`Mover.host_ns`)

# per direction: (first hop's stream, second hop's stream); the ring sits between them
HOPS = {"read": ("read", "read_nvl"), "write": ("write_nvl", "write")}


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
        # a direction's ring and streams: a transfer that does not yield (`gate`) holds it from its first chunk to its
        # last, so transfers in one direction (a DRAM read and an SSD read's copy) run one after another on the link,
        # not chunk by chunk interleaved (which would make both end late)
        self._lock = {"read": threading.RLock(), "write": threading.RLock()}
        self._anchor = threading.local()
        self._ev_lock = threading.Lock()           # the event pool (taken inside a direction's lock, never around it)
        if self.cuda:
            import torch
            from vllm import _custom_ops
            self.swap = _custom_ops.swap_blocks_batch
            self.streams = {k: torch.cuda.Stream(device) for k in ("read", "write", "read_nvl", "write_nvl")}
            # torch creates a CUDA event at its first record: every event is recorded once here, so a transfer only
            # records (no cudaEventCreate on its path, which waits for the driver behind the launch thread)
            self._events = [torch.cuda.Event(blocking=True, enable_timing=True) for _ in range(EVENT_POOL)]
            self._made(self._events)
        self.events_made = EVENT_POOL if self.cuda else 0   # the pool's events, and those made when it ran empty
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
                # per direction and ring chunk: (first hop done, second hop done)
                self._chunk_ev = {d: [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(self.depth)]
                                  for d in HOPS}
                self._made([e for evs in self._chunk_ev.values() for pair in evs for e in pair])
            else:
                self.ring = np.zeros(2 * half + 4096, dtype=np.uint8)
                ring_ptr = self.ring.ctypes.data
                self.info = f"two hops: ring 2 x {self.depth} x {self.chunk} slots"
            self._ring_base = {"read": ring_ptr, "write": ring_ptr + half}
            self._ring_next = {"read": 0, "write": 0}

    def _made(self, events):
        st = self.streams["read"]
        for ev in events:
            ev.record(st)
        st.synchronize()

    def event(self, owned: list | None):
        """A pooled event (blocking sync, timing), appended to `owned` (the op's events, `release`d once it is done)."""
        with self._ev_lock:
            ev = self._events.pop() if self._events else None
            if ev is None:
                self.events_made += 1
        if ev is None:
            import torch
            ev = torch.cuda.Event(blocking=True, enable_timing=True)
        if owned is not None:
            owned.append(ev)
        return ev

    def release(self, owned: list):
        """Back to the pool: the events of a transfer whose last copy completed (and whose trace was read)."""
        if owned:
            with self._ev_lock:
                self._events.extend(owned)

    def mark(self, direction: str, owned: list | None = None):
        """A timing event on the stream a transfer's first copy goes on (CUDA; None on CPU)."""
        if not self.cuda:
            return None
        ev = self.event(owned)
        ev.record(self.streams[HOPS[direction][0] if self.ring is not None else direction])
        return ev

    def host_ns(self, ev) -> int | None:
        """The host time (time_ns) at which the completed event `ev` ran on the GPU, through this thread's anchor: an
        event of its own recorded on an idle stream of its own and waited for (so it ran, to the wake-up latency, just
        before the host read its clock), taken again every ANCHOR_S (clock drift). None on CPU."""
        if not self.cuda or ev is None:
            return None
        a, now = self._anchor, time.monotonic()
        if getattr(a, "at", None) is None or now - a.at > ANCHOR_S:
            import torch
            if getattr(a, "ev", None) is None:
                a.stream, a.ev = torch.cuda.Stream(self.device), torch.cuda.Event(enable_timing=True)
            a.ev.record(a.stream)
            a.ev.synchronize()
            a.ns, a.at = time.time_ns(), now
        return a.ns + int(a.ev.elapsed_time(ev) * 1e6)

    def transfer_async(self, direction: str, segments: list[Segment], ready=None, tr: list | None = None,
                       owned: list | None = None, gate=None, marks: list | None = None):
        """"read" host -> HBM or "write" HBM -> host of every segment's pieces. Two hops: the segments' slots, in
        order, are cut into ring chunks; direct: one batch of all pieces. `ready(i, k)`: called before the host side
        of segment i's slots below k is read (an SSD read waits for its file reads there). `tr`: per-batch trace
        (`_batch`). `owned`: gets every pooled event the transfer took (CUDA; `release` them once its last copy
        completed). `gate()`: called before each ring chunk (two hops), after the previous chunk's first hop completed
        (the host does not run ahead of the copies), e.g. a write yielding to reads; without it the transfer holds its
        direction from its first chunk to its last. `marks`: gets the transfer's start mark (`mark`, recorded as its
        first chunk is submitted, in the direction). Returns (the transfer's end event, None on CPU or with nothing to
        move; bytes of HBM pieces)."""
        with self._lock[direction] if gate is None else contextlib.nullcontext():
            if marks is not None:
                with self._lock[direction]:
                    marks.append(self.mark(direction, owned))
            return self._transfer(direction, segments, ready, tr, owned, gate)

    def _transfer(self, direction: str, segments: list[Segment], ready, tr: list | None, owned: list | None, gate):
        sb = self.sb
        nslots = [len(s.host_ptrs) for s in segments]
        first = np.cumsum([0] + nslots)            # global slot index of each segment's first slot
        hosts = np.fromiter((p for s in segments for p in s.host_ptrs), dtype=np.int64, count=int(first[-1]))
        hlen = np.repeat(np.array([s.host_len for s in segments], dtype=np.int64), nslots)
        gslot = np.concatenate([s.off // sb + f for s, f in zip(segments, first)]).astype(np.int64)
        inslot = np.concatenate([s.off % sb for s in segments]).astype(np.int64)
        dev = np.concatenate([s.dev for s in segments]).astype(np.int64)
        lens = np.concatenate([s.lens for s in segments]).astype(np.int64)
        nbytes = int(lens.sum())
        n = int(first[-1])
        if self.ring is None:
            if ready is not None:
                for i, k in enumerate(nslots):
                    ready(i, k)
            if not len(lens):
                return None, 0
            host = hosts[gslot] + inslot
            src, dst = (host, dev) if direction == "read" else (dev, host)
            with self._lock[direction]:
                self._batch(direction, src, dst, lens, tr=tr, owned=owned)
                return self._end(direction, owned), nbytes
        C = self.chunk
        rel = (np.arange(n, dtype=np.int64) % C) * sb       # a slot's place in its ring chunk
        prel = (gslot % C) * sb + inslot                    # a piece's place in its ring chunk
        cuts = np.searchsorted(gslot, np.arange(0, n + C, C))   # pieces of chunk j: cuts[j]:cuts[j + 1]
        s1, s2 = HOPS[direction]
        end = None
        prev = None                                # the previous chunk's first hop done (gated transfers)
        for j, a in enumerate(range(0, n, C)):
            b = min(a + C, n)
            if gate is not None:
                if prev is not None:
                    prev.synchronize()
                gate()
            if ready is not None:                  # each segment's slots below this chunk's end
                for i in range(len(segments)):
                    if first[i] < b and first[i + 1] > a:
                        ready(i, int(min(b, first[i + 1]) - first[i]))
            lo, hi = cuts[j], cuts[j + 1]
            with self._lock[direction]:
                pos = self._ring_next[direction]
                self._ring_next[direction] = (pos + 1) % self.depth
                rbase = self._ring_base[direction] + pos * C * sb
                ev1, ev2 = self._chunk_ev[direction][pos] if self.cuda else (None, None)
                rslots, rpieces = rbase + rel[a:b], rbase + prel[lo:hi]
                if gate is not None and self.cuda:
                    prev = self.event(owned)
                if direction == "read":            # host -> ring (PCIe), then ring -> the stages (NVLink)
                    self._batch(s1, hosts[a:b], rslots, hlen[a:b], ev2, ev1, tr, owned)
                    self._batch(s2, rpieces, dev[lo:hi], lens[lo:hi], ev1, ev2, tr, owned)
                else:                              # the stages -> ring, then ring -> host
                    self._batch(s1, dev[lo:hi], rpieces, lens[lo:hi], ev2, ev1, tr, owned)
                    if prev is not None:
                        prev.record(self.streams[s1])
                    self._batch(s2, rslots, hosts[a:b], hlen[a:b], ev1, ev2, tr, owned)
                if b == n:
                    end = self._end(s2, owned)
        return end, nbytes

    def _end(self, stream: str, owned: list | None):
        """The transfer's end: a pooled event after its last copy on `stream` (None on CPU)."""
        if not self.cuda:
            return None
        ev = self.event(owned)
        ev.record(self.streams[stream])
        return ev

    def _batch(self, stream: str, src, dst, lens, wait=None, done=None, tr: list | None = None,
               owned: list | None = None):
        """One `cuMemcpyBatchAsync` on `stream`, after the event `wait`; records `done` after it (CUDA). `tr`: append
        (stream, pieces, bytes, host ns before the call, host ns after it, thread CPU ns in it, GPU start event, end
        event; the two events pooled, in `owned`): the rest of the call is a wait for a lock (GIL or driver)."""
        if not self.cuda:
            for s, t, k in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(lens).tolist()):
                ctypes.memmove(t, s, k)
            return
        import torch
        st = self.streams[stream]
        if wait is not None:
            st.wait_event(wait)
        args = [torch.from_numpy(x) for x in (src, dst, lens)]
        if tr is not None:
            ev0 = self.event(owned)
            ev0.record(st)
            h0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        with torch.cuda.stream(st):
            self.swap(*args)
        if tr is not None:
            h1, c1 = time.perf_counter_ns(), time.thread_time_ns()
            ev1 = self.event(owned)
            ev1.record(st)
            tr.append((stream, len(lens), int(lens.sum()), h0, h1, c1 - c0, ev0, ev1))
        if done is not None:
            done.record(st)
