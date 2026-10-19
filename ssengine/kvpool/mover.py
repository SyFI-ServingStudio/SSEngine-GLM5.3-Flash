"""Layer 4 of the KV pool (dev:research/TIERS.md §2.3, §3, §13.2): one GPU's copies between host slots and the stages' KV.

Copy engines only, no SMs, no wait on the compute stream: `cuMemcpyBatchAsync` on the mover's own streams; on CPU
(tests) `ctypes.memmove`, done at once. The copies, stream waits and event records go straight to the driver
(`Driver`, ctypes on libcuda: no torch dispatch, stream context or tensor wrapping per call; the GIL is released
in each batch call), so a transfer's host work is the driver's own (job 8202: 4.3 ms per DRAM op through torch and vLLM's
`swap_blocks_batch`, of it 1.7 ms CPU in the driver).

Two hops (`TierConfig.hops`, default; §2.3: a copy engine writing host data straight into the other GPUs' HBM runs at
31.7 GB/s against 47.6 into its own): a read copies the other stages' part of each slot (contiguous runs: a slot
packs its pieces by stage) into a ring of chunks in this GPU's own HBM (its own PCIe link), then one batch scatters
them from the ring to the other stages' KV over NVLink. Writes gather into the ring, then copy the runs to host. This
GPU's own pieces (`Segment.local`) go straight between host and its KV in the same PCIe batch: a same-GPU
device-to-device copy waits behind the stage's kernels (job 8248's nsys: 4-7 s per 37 s of the NVLink batches were
gaps before own-GPU pieces, 0.1-0.3 s before peer pieces). Each
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

CUDA_ERROR_NOT_READY = 600
EVENT_POOL = 1024  # pooled CUDA events per mover: an op takes 2 (start mark, end), + 2 per batch with the trace on
ANCHOR_S = 1.0     # a thread's GPU-to-host clock anchor is taken again after this long (`Mover.host_ns`)

# per direction: (first hop's stream, second hop's stream); the ring sits between them
HOPS = {"read": ("read", "read_nvl"), "write": ("write_nvl", "write")}


class Driver:
    """The driver calls of a transfer, through ctypes: `cuMemcpyBatchAsync` at its CUDA 12.8 ABI (as vLLM's
    `swap_blocks_batch`: one attribute, source in stream order), `cuStreamWaitEvent`, `cuEventRecord`. Streams and
    events are torch's (`cuda_stream`, `cuda_event` handles). The batch (tens of us in the driver) releases the GIL;
    the record and the wait (1-4 us) keep it (`ctypes.PyDLL`): a release hands the GIL to a waiting thread, and the
    tier threads' ~5k releases per second kept stage 0's main thread waiting for it in short slices, 25-40% of its
    host prepare (jobs 8349/8350, nsys: take_gil waits p50 20-30 us)."""

    def __init__(self):
        from cuda.bindings import driver
        err, fn, _status = driver.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080,
                                                   driver.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_DEFAULT)
        if err != driver.CUresult.CUDA_SUCCESS or not int(fn):
            raise RuntimeError(f"cuMemcpyBatchAsync (CUDA 12.8 ABI) not available: {err}")
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        self._batch = ctypes.CFUNCTYPE(ctypes.c_int, vp, vp, vp, sz, vp, vp, sz, vp, vp)(int(fn))
        lib = ctypes.PyDLL("libcuda.so.1")      # the GIL held through the call
        self._wait, self._record = lib.cuStreamWaitEvent, lib.cuEventRecord
        self._wait.argtypes, self._wait.restype = [vp, vp, ctypes.c_uint], ctypes.c_int
        self._record.argtypes, self._record.restype = [vp, vp], ctypes.c_int
        self._elapsed = lib.cuEventElapsedTime
        self._elapsed.argtypes, self._elapsed.restype = [ctypes.POINTER(ctypes.c_float), vp, vp], ctypes.c_int
        self.host_ns = {"record": [], "wait": []}   # per call, ns on the host (instrumentation: `summary`)
        self._attr = (ctypes.c_uint32 * 6)(1, 0, 0, 0, 0, 0)    # CUmemcpyAttributes: srcAccessOrder STREAM (1)
        self._attr_idx = (ctypes.c_size_t * 1)(0)

    @staticmethod
    def _ok(rc: int, what: str):
        if rc:
            raise RuntimeError(f"{what}: CUresult {rc}")

    def batch(self, stream: int, src: np.ndarray, dst: np.ndarray, lens: np.ndarray):
        """One `cuMemcpyBatchAsync` of src[i] -> dst[i], lens[i] bytes (int64 arrays) on `stream`."""
        fail = ctypes.c_size_t(0)
        src, dst, lens = (np.ascontiguousarray(x, dtype=np.int64) for x in (src, dst, lens))
        rc = self._batch(dst.ctypes.data, src.ctypes.data, lens.ctypes.data, len(lens), ctypes.addressof(self._attr),
                         ctypes.addressof(self._attr_idx), 1, ctypes.byref(fail), stream)
        self._ok(rc, f"cuMemcpyBatchAsync of {len(lens)} copies failed at {fail.value}")

    def elapsed_ms(self, a, b) -> float:
        """ms from event `a` to event `b`, both completed (as torch's `elapsed_time`, without its two event queries,
        each a GIL release): `cuEventElapsedTime` with the GIL held. The calling thread needs the device's context
        current (`torch.cuda.set_device`)."""
        ms = ctypes.c_float()
        rc = self._elapsed(ctypes.byref(ms), a.cuda_event, b.cuda_event)
        if rc == CUDA_ERROR_NOT_READY:          # not expected (callers read completed events): wait, then read
            a.synchronize()
            b.synchronize()
            rc = self._elapsed(ctypes.byref(ms), a.cuda_event, b.cuda_event)
        self._ok(rc, "cuEventElapsedTime")
        return ms.value

    def wait(self, stream: int, ev):
        t = time.perf_counter_ns()
        rc = self._wait(stream, ev.cuda_event, 0)
        self.host_ns["wait"].append(time.perf_counter_ns() - t)
        self._ok(rc, "cuStreamWaitEvent")

    def record(self, ev, stream: int):
        t = time.perf_counter_ns()
        rc = self._record(ev.cuda_event, stream)
        self.host_ns["record"].append(time.perf_counter_ns() - t)
        self._ok(rc, "cuEventRecord")

    def summary(self) -> str:
        """Host time of the GIL-holding calls (a blocked one stalls every Python thread), us p50 / p99 / max."""
        out = []
        for k, v in self.host_ns.items():
            if v:
                a = np.asarray(v) / 1e3
                big = a[a > 1000]
                out.append(f"{k} n {len(a)} p50 {np.median(a):.1f} p99 {np.percentile(a, 99):.1f} max {a.max():.0f} us, "
                           f">1 ms {len(big)} sum {big.sum() / 1e3:.0f} ms")
        return "; ".join(out)


def _runs(src: np.ndarray, dst: np.ndarray, lens: np.ndarray, slot: np.ndarray):
    """Pieces in order merged into contiguous runs (each piece starting where the previous ended on both sides),
    within a slot: adjacent host slots may sit in different registered host allocations, and a copy must not span
    two (cuMemcpyBatchAsync: CUDA_ERROR_INVALID_VALUE)."""
    if len(lens) < 2:
        return src, dst, lens
    cut = np.flatnonzero(np.r_[True, (src[1:] != src[:-1] + lens[:-1]) | (dst[1:] != dst[:-1] + lens[:-1])
                               | (slot[1:] != slot[:-1])])
    return src[cut], dst[cut], np.add.reduceat(lens, cut)


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
    local: np.ndarray | None = None     # pieces in this GPU's own HBM (two hops: copied straight, never via the ring)


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
        self.gate_wait_ns: list[int] = []          # a gated write's host wait for its previous chunk (instrumentation)
        self._ev_lock = threading.Lock()           # the event pool (taken inside a direction's lock, never around it)
        if self.cuda:
            import torch
            self.drv = Driver()
            self.streams = {k: torch.cuda.Stream(device) for k in ("read", "write", "read_nvl", "write_nvl")}
            self._st = {k: st.cuda_stream for k, st in self.streams.items()}   # the driver's stream handles
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
                # (blocking: a gated write's host waits for its previous chunk's first hop on that chunk's event)
                self._chunk_ev = {d: [(torch.cuda.Event(blocking=True), torch.cuda.Event(blocking=True))
                                      for _ in range(self.depth)] for d in HOPS}
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
            ev.record(self.streams["read"])     # torch creates it at its first record (a handle for the driver)
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
        self.drv.record(ev, self._st[HOPS[direction][0] if self.ring is not None else direction])
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
        return a.ns + int(self.drv.elapsed_ms(a.ev, ev) * 1e6)

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
        gslot = np.concatenate([s.off // sb + f for s, f in zip(segments, first)]).astype(np.int64)
        inslot = np.concatenate([s.off % sb for s in segments]).astype(np.int64)
        dev = np.concatenate([s.dev for s in segments]).astype(np.int64)
        lens = np.concatenate([s.lens for s in segments]).astype(np.int64)
        local = np.concatenate([np.zeros(len(s.lens), bool) if s.local is None else np.asarray(s.local, bool)
                                for s in segments])
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
        prel = (gslot % C) * sb + inslot                    # a piece's place in its ring chunk
        hpiece = hosts[gslot] + inslot                      # a piece's host address
        cuts = np.searchsorted(gslot, np.arange(0, n + C, C))   # pieces of chunk j: cuts[j]:cuts[j + 1]
        s1, s2 = HOPS[direction]
        end = None
        prev = None                                # the previous chunk's first hop done (gated transfers)
        for j, a in enumerate(range(0, n, C)):
            b = min(a + C, n)
            if gate is not None:
                if prev is not None:
                    t = time.perf_counter_ns()
                    prev.synchronize()
                    self.gate_wait_ns.append(time.perf_counter_ns() - t)
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
                loc = local[lo:hi]
                peer = ~loc
                rp, dp, lp = rbase + prel[lo:hi][peer], dev[lo:hi][peer], lens[lo:hi][peer]   # via the ring
                hr, rr, lr = _runs(hpiece[lo:hi][peer], rp, lp, gslot[lo:hi][peer])   # host side, runs per slot
                hl, dl, ll = hpiece[lo:hi][loc], dev[lo:hi][loc], lens[lo:hi][loc]          # straight
                if direction == "read":            # host -> ring and own pieces (PCIe), then ring -> the peers (NVLink)
                    self._batch(s1, np.r_[hr, hl], np.r_[rr, dl], np.r_[lr, ll], ev2, ev1, tr, owned)
                    self._batch(s2, rp, dp, lp, ev1, ev2, tr, owned)
                else:                              # the peers -> ring, then ring and own pieces -> host
                    self._batch(s1, dp, rp, lp, ev2, ev1, tr, owned)
                    if gate is not None:
                        prev = ev1                 # its first hop done (recorded again depth chunks on, or by
                                                   # another write in between: a wait for that chunk too)
                    self._batch(s2, np.r_[rr, dl], np.r_[hr, hl], np.r_[lr, ll], ev1, ev2, tr, owned)
                if b == n:
                    end = self._end(s2, owned)
        return end, nbytes

    def _end(self, stream: str, owned: list | None):
        """The transfer's end: a pooled event after its last copy on `stream` (None on CPU)."""
        if not self.cuda:
            return None
        ev = self.event(owned)
        self.drv.record(ev, self._st[stream])
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
        st, drv = self._st[stream], self.drv
        if wait is not None:
            drv.wait(st, wait)
        if not len(lens):
            if done is not None:
                drv.record(done, st)
            return
        if tr is not None:
            ev0 = self.event(owned)
            drv.record(ev0, st)
            h0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        drv.batch(st, src, dst, lens)
        if tr is not None:
            h1, c1 = time.perf_counter_ns(), time.thread_time_ns()
            ev1 = self.event(owned)
            drv.record(ev1, st)
            tr.append((stream, len(lens), int(lens.sum()), h0, h1, c1 - c0, ev0, ev1))
        if done is not None:
            drv.record(done, st)
