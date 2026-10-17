"""Worker side of the DRAM/SSD tiers (lanes/m6 TIERS.md §3-§6).

Every worker moves its 1/n share (n = PP degree, `num_stages`) of every tier op (balanced storage,
`ppimpl/runtime/tier_copy.py`). The controller sends a `TierBatch` of ops whose dependencies are done to all n
workers' tier sockets; a worker's tier thread builds the (device, host, length) pieces of its share over all stages'
KV (the other stages' KV opened by CUDA IPC at start-up), submits them as `cuMemcpyBatchAsync` calls (vLLM
`ops.swap_blocks_batch`) on its read streams (host -> HBM) or write streams (HBM -> host), and a done thread per
direction reports `TierDone` when the copies completed. Copy engines only, no SMs, no wait on the compute stream: a
write's blocks are final and pinned until it is done, a read's blocks are held by a request that computes only after
all n `TierDone`s.

Two hops (`TierConfig.hops`, default; TIERS.md §2.3: a copy engine writing host data straight into the other GPUs'
HBM runs at 31.7 GB/s against 47.6 into its own): the host side of a slot is contiguous, so a read copies each slot
whole into a ring of chunks in GPU q's own HBM (one copy per slot, its own PCIe link), then one `cuMemcpyBatchAsync`
scatters the chunk's pieces from the ring to the stages' KV over NVLink. Writes gather into the ring, then copy each
slot to host. Each direction has its ring (`ring_depth` chunks of `ring_chunk_slots` slots; a context's state is cut
into chunks like its KV) and two streams (host side, NVLink side); chunk k+1's host copy overlaps chunk k's scatter,
and a ring chunk is reused once the copy that read it is done (events). The ring is allocated after the model, from
the HBM the KV pool (the contract's 7,144 blocks) leaves; `attach` prints how much that is. `hops: direct` puts the
pieces straight between host and the stages' KV (one batch per op).

DRAM: one pinned arena per worker on its GPU's NUMA node: libnuma, MADV_HUGEPAGE, first touch on a side thread
while the model loads, then `cudaHostRegister` in chunks of whole slots (a piece never crosses a slot, so never a
registration). Not torch's pinned allocator: it rounds 150 GB up to 256 GB.

SSD: one preallocated file per worker in `TierConfig.ssd_dir` (default $PPIMPL_SSD_DIR, else the run dir), slot i
at offset i x slot_bytes, O_DIRECT `preadv`/`pwritev` of whole slots from a pool of 8 threads (GIL released), into
and out of host slots: DRAM's (a demote; an SSD read's copy-up) or a staging arena's (SSD without DRAM, or a context
larger than DRAM). An SSD read lands slot by slot and each KV block's H2D is enqueued as soon as its slot is in, the
state after its slots: read time ~ SSD time + one block. SSD ops run on two threads (reads; writes and demotes), one
op at a time each. The file is deleted when the worker stops.

The fake model has no KV: its engine moves nothing and reports each op after the time the configured bandwidth
gives its bytes, one op after another per direction (CPU tests, fake runs).
"""

from __future__ import annotations

import ctypes
import errno
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ppimpl.runtime.tier_copy import Layout, ViewRange
from ppimpl.runtime.transport import Shutdown, TierBatch, TierDone, TierMove, WorkerError, decode_tier, encode

MADV_HUGEPAGE = 14
REGISTER_CHUNK = 1 << 30      # cudaHostRegister per ~1 GiB of whole slots, from 4 threads
TOUCH_THREADS = 4
IO_THREADS = 8                # SSD preadv/pwritev pool per worker (fio: 8 psync threads, 53 GB/s on one file)
FAKE_BLOCK_BYTES = 8576 * 5996  # one block over all stages (GLM-5.3 PP4: 4 x 1,499 B/token); a GPU's slot is 1/n of
                                # it rounded up to 4 KiB (PP4: 12,857,344 B, the real balanced slot, TIERS.md §3)


def tier_num_slots(capacity_gb: float | None, slot_bytes: int) -> int:
    """Slots of a tier of `capacity_gb` per GPU (as `ppimpl.engine.tiers.TierStore.num_slots`)."""
    return int(capacity_gb * 1e9 // slot_bytes) if capacity_gb else 0


def _ok(ret):
    err, *rest = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


# --- host memory ----------------------------------------------------------------------------------------------------

class HostArena:
    """`nbytes` of host memory on NUMA node `node` (CUDA: pinned once `register` ran; CPU: a plain numpy buffer)."""

    def __init__(self, nbytes: int, node: int | None, cuda: bool):
        self.nbytes = -(-nbytes // (2 << 20)) * (2 << 20)
        self.cuda, self.node = cuda, node
        self._chunks: list[tuple[int, int]] = []
        self.touch_s = self.register_s = 0.0
        if not cuda:                       # 4 KiB-aligned (O_DIRECT)
            self._buf = np.zeros(self.nbytes + 4096, dtype=np.uint8)
            self.ptr = self._buf.ctypes.data + (-self._buf.ctypes.data) % 4096
            return
        self._numa = ctypes.CDLL("libnuma.so.1")
        self._numa.numa_alloc_onnode.restype = ctypes.c_void_p
        self._numa.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
        self._numa.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.ptr = self._numa.numa_alloc_onnode(self.nbytes, node)
        if not self.ptr:
            raise MemoryError(f"numa_alloc_onnode({self.nbytes}, {node})")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise(self.ptr, self.nbytes, MADV_HUGEPAGE)   # advisory: no THP is not an error

    def touch(self):
        """First touch from a few threads (pages land on the arena's node; ctypes.memset releases the GIL)."""
        if not self.cuda:
            return
        t0 = time.perf_counter()
        step = -(-self.nbytes // TOUCH_THREADS)
        ts = [threading.Thread(target=ctypes.memset, args=(self.ptr + a, 0, min(step, self.nbytes - a)))
              for a in range(0, self.nbytes, step)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.touch_s = time.perf_counter() - t0

    def register(self, slot_bytes: int):
        """Pin in chunks of whole slots (CUDA). Chunks register from 4 threads."""
        if not self.cuda:
            return
        from cuda.bindings import runtime as rt
        t0 = time.perf_counter()
        chunk = max(1, REGISTER_CHUNK // slot_bytes) * slot_bytes
        spans = [(a, min(chunk, self.nbytes - a)) for a in range(0, self.nbytes, chunk)]
        flags = rt.cudaHostRegisterPortable | rt.cudaHostRegisterMapped
        errs = []

        def run(part, dev):
            try:
                _ok(rt.cudaSetDevice(dev))
                for a, n in part:
                    _ok(rt.cudaHostRegister(self.ptr + a, n, flags))
            except Exception as e:    # noqa: BLE001 - re-raised below
                errs.append(e)

        import torch
        dev = torch.cuda.current_device()
        ts = [threading.Thread(target=run, args=(spans[i::4], dev)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise errs[0]
        self._chunks = spans
        self.register_s = time.perf_counter() - t0

    def close(self):
        if not self.cuda or not self.ptr:
            return
        from cuda.bindings import runtime as rt
        for a, _ in self._chunks:
            rt.cudaHostUnregister(self.ptr + a)
        self._chunks = []
        self._numa.numa_free(self.ptr, self.nbytes)
        self.ptr = 0


class SsdFile:
    """One worker's SSD tier: a preallocated file of whole slots, read and written with O_DIRECT (buffered where the
    file system has no O_DIRECT, as tmpfs in CPU tests)."""

    def __init__(self, path: str, nbytes: int, slot_bytes: int):
        self.path, self.slot_bytes = path, slot_bytes
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        try:
            self.fd = os.open(path, flags | os.O_DIRECT, 0o600)
            self.direct = True
        except OSError as e:
            if e.errno != errno.EINVAL:
                raise
            self.fd = os.open(path, flags, 0o600)
            self.direct = False
        os.posix_fallocate(self.fd, 0, nbytes)

    def read(self, slot: int, ptr: int):
        buf = (ctypes.c_char * self.slot_bytes).from_address(ptr)
        n = os.preadv(self.fd, [buf], slot * self.slot_bytes)
        if n != self.slot_bytes:
            raise OSError(f"{self.path}: slot {slot}: read {n} of {self.slot_bytes} B")

    def write(self, slot: int, ptr: int):
        buf = (ctypes.c_char * self.slot_bytes).from_address(ptr)
        n = os.pwritev(self.fd, [buf], slot * self.slot_bytes)
        if n != self.slot_bytes:
            raise OSError(f"{self.path}: slot {slot}: wrote {n} of {self.slot_bytes} B")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            os.unlink(self.path)


# --- the other stages' KV -------------------------------------------------------------------------------------------

def share_ranges(stage: int, n: int, ranges: list[ViewRange]) -> tuple[list[ViewRange], list[int]]:
    """All stages' ranges with device addresses valid in this process: each worker exports the allocations behind its
    ranges (cuMemGetAddressRange base, cudaIpcGetMemHandle), and opens the others' (lazy peer access). Collective
    over the workers' gloo group (the pipe's). Returns (ranges, opened pointers to close)."""
    import torch.distributed as dist
    from cuda.bindings import driver, runtime as rt

    allocs: dict[int, bytes] = {}
    mine = []
    for r in ranges:
        base, _size = _ok(driver.cuMemGetAddressRange(driver.CUdeviceptr(r.base)))
        base = int(base)
        if base not in allocs:
            try:
                h = _ok(rt.cudaIpcGetMemHandle(base))
            except RuntimeError as e:
                raise RuntimeError(f"stage {stage}: {r.name} is not IPC-shareable (KV must be allocated without "
                                   f"expandable segments: StageModel(ipc_kv=True))") from e
            allocs[base] = ctypes.string_at(h.getPtr(), 64)
        mine.append((r.name, r.kind, r.group, base, r.base - base, r.stride, r.nbytes))
    allh = [None] * n
    dist.all_gather_object(allh, {"allocs": allocs, "ranges": mine})
    out, opened = list(ranges), []
    for s in range(n):
        if s == stage:
            continue
        local = {}
        for base, raw in allh[s]["allocs"].items():
            h = rt.cudaIpcMemHandle_t()
            ctypes.memmove(h.getPtr(), raw, 64)
            local[base] = int(_ok(rt.cudaIpcOpenMemHandle(h, rt.cudaIpcMemLazyEnablePeerAccess)))
            opened.append(local[base])
        for name, kind, group, base, off, stride, nbytes in allh[s]["ranges"]:
            out.append(ViewRange(s, name, kind, local[base] + off, stride, nbytes, group))
    return out, opened


# --- the engine -----------------------------------------------------------------------------------------------------

class TierEngine:
    """One worker's tier side: its tier socket, host arenas, streams, and the threads that run and report ops.

    Built in two steps so the arena's first touch overlaps the model load: `TierEngine(...)` allocates and starts
    touching; `attach(ranges)` (after the model, collective over the workers) opens the peers' KV, builds the layout,
    registers the arena and returns (slot_bytes, state_slots) for `Ready`; `start()` begins serving ops."""

    def __init__(self, cfg, stage: int, device, zmq_ctx):
        tc = cfg.tiers
        self.cfg, self.tc, self.stage, self.n = cfg, tc, stage, cfg.num_stages
        self.device, self.cuda = device, device.type == "cuda"
        self.ctx = zmq_ctx
        self.layout: Layout | None = None
        self.slot_bytes = tc.slot_bytes or 0
        self.opened: list[int] = []
        self.dram: HostArena | None = None
        self.staging: HostArena | None = None
        self.ssd: SsdFile | None = None
        self.pool: ThreadPoolExecutor | None = None
        self._touch: threading.Thread | None = None
        self.real = False
        self.threads: list[threading.Thread] = []
        self.stop = threading.Event()
        self.q = {"read": queue.Queue(), "write": queue.Queue()}       # done threads: (op id, wait, start, bytes)
        self.ssd_q = {"read": queue.Queue(), "write": queue.Queue()}   # SSD threads: (op, start)
        self._due: dict[str, float] = {}           # fake engine: when each direction's last op is done
        self.ring = None
        self.ring_info = ""
        self._lock = threading.Lock()              # ring chunks and their order, across the submitting threads
        if tc.hops not in ("two", "direct"):
            raise ValueError(f"tiers.hops: {tc.hops!r} (two | direct)")
        if tc.ring_depth < 1 or tc.ring_chunk_slots < 1:
            raise ValueError(f"tiers.ring_depth {tc.ring_depth}, ring_chunk_slots {tc.ring_chunk_slots}: >= 1")
        real = cfg.model != "fake"
        if tc.dram_gb and self.cuda and real:
            from ppimpl.runtime.cpus import _gpu_node
            self.dram = HostArena(int(tc.dram_gb * 1e9), _gpu_node(device.index), True)
            self._touch = threading.Thread(target=self.dram.touch, name="tier-touch", daemon=True)
            self._touch.start()

    def attach(self, ranges: list[ViewRange] | None) -> tuple[int, int]:
        """`ranges`: this stage's KV ranges (`tier_copy.stage_ranges`), None for a model without KV (fake)."""
        if ranges is None:
            self.slot_bytes = self.slot_bytes or -(-FAKE_BLOCK_BYTES // self.n // 4096) * 4096
            return self.slot_bytes, self.tc.state_slots
        import torch
        self.real = True
        allr, self.opened = share_ranges(self.stage, self.n, ranges) if self.cuda else (ranges, [])
        self.layout = Layout(allr, self.stage, self.n, slot_bytes=self.tc.slot_bytes)
        sb = self.slot_bytes = self.layout.slot_bytes
        node = None
        if self.cuda:
            from ppimpl.runtime.cpus import _gpu_node
            node = _gpu_node(self.device.index)
        if self.dram is None and self.tc.dram_gb:
            self.dram = HostArena(tier_num_slots(self.tc.dram_gb, sb) * sb, node, self.cuda)
            self.dram.touch()
        if self._touch is not None:
            self._touch.join()
        if self.dram is not None:
            self.dram.register(sb)
            assert tier_num_slots(self.tc.dram_gb, sb) * sb <= self.dram.nbytes
        if self.tc.ssd_gb:
            self.staging = HostArena(tier_num_slots(self.tc.staging_gb, sb) * sb, node, self.cuda)
            self.staging.touch()
            self.staging.register(sb)
            d = self.tc.ssd_dir or os.environ.get("PPIMPL_SSD_DIR") or os.path.join(self.cfg.run_dir, "ssd")
            self.ssd = SsdFile(os.path.join(d, f"tier_ssd_gpu{self.stage}_{os.getpid()}.bin"),
                               tier_num_slots(self.tc.ssd_gb, sb) * sb, sb)
            self.pool = ThreadPoolExecutor(IO_THREADS, thread_name_prefix="tier-io")
        if self.tc.hops == "two":                  # per direction: `depth` chunks of `chunk` slots, after the KV pool
            self.depth = self.tc.ring_depth
            self.chunk = self.tc.ring_chunk_slots
            half = self.depth * self.chunk * sb
            if self.cuda:
                free = torch.cuda.mem_get_info(self.device)[0]
                self.ring = torch.empty(2 * half, dtype=torch.uint8, device=self.device)
                self.ring_ptr = self.ring.data_ptr()
                self.ring_info = (f"two hops: ring 2 x {self.depth} x {self.chunk} slots = {2 * half / 1e9:.3f} GB "
                                  f"of {free / 1e9:.2f} GB free HBM after the model and KV pool")
            else:
                self.ring = np.zeros(2 * half + 4096, dtype=np.uint8)
                self.ring_ptr = self.ring.ctypes.data
                self.ring_info = f"two hops: ring 2 x {self.depth} x {self.chunk} slots"
            self._ring_base = {"read": self.ring_ptr, "write": self.ring_ptr + half}
            self._ring_next = {"read": 0, "write": 0}
            self._ring_free = {"read": [None] * self.depth, "write": [None] * self.depth}
        if self.cuda:
            from vllm import _custom_ops
            self.swap = _custom_ops.swap_blocks_batch
            self.streams = {k: torch.cuda.Stream(self.device) for k in ("read", "write", "read_nvl", "write_nvl")}
        print(f"stage {self.stage} tiers: slot {self.slot_bytes} B, state {self.layout.state_slots} slots, "
              f"{len(self.layout.kv_len)} kv + {len(self.layout.st_len)} state pieces per block/context, "
              + (f"dram {self.dram.nbytes / 1e9:.1f} GB (touch {self.dram.touch_s:.1f} s, register "
                 f"{self.dram.register_s:.1f} s), " if self.dram else "")
              + (f"ssd {self.ssd.path} (O_DIRECT {self.ssd.direct}), staging {self.staging.nbytes / 1e9:.1f} GB, "
                 if self.ssd else "")
              + f"{len(self.opened)} peer allocations, "
              + (self.ring_info or "direct"), flush=True)
        return self.slot_bytes, self.layout.state_slots

    def start(self):
        for name in ("read", "write"):
            t = threading.Thread(target=self._done_loop, args=(name,), name=f"tier-{name}", daemon=True)
            t.start()
            self.threads.append(t)
            if self.ssd is not None:
                t = threading.Thread(target=self._ssd_loop, args=(name,), name=f"tier-ssd-{name}", daemon=True)
                t.start()
                self.threads.append(t)
        t = threading.Thread(target=self._recv_loop, name="tier-recv", daemon=True)
        t.start()
        self.threads.append(t)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=10)
        if self.cuda:
            import torch
            torch.cuda.synchronize(self.device)
            from cuda.bindings import runtime as rt
            for p in self.opened:
                rt.cudaIpcCloseMemHandle(p)
        if self.pool is not None:
            self.pool.shutdown()
        if self.ssd is not None:
            self.ssd.close()
        for a in (self.dram, self.staging):
            if a is not None:
                a.close()

    # --- threads ----------------------------------------------------------------------------------------------------

    def _recv_loop(self):
        import zmq
        sock = self.ctx.socket(zmq.PULL)
        sock.bind(self.cfg.tier_addr(self.stage))
        err = None
        try:
            if self.cuda:
                import torch
                torch.cuda.set_device(self.device)
            while not self.stop.is_set():
                if not sock.poll(200):
                    continue
                msg = decode_tier(sock.recv())
                if isinstance(msg, Shutdown):
                    break
                for op in msg.ops:
                    self.submit(op)
        except BaseException:   # noqa: BLE001 - reported to the controller, which fails the run
            err = WorkerError(self.stage, "tier thread:\n" + traceback.format_exc())
        finally:
            sock.close(linger=0)
            for d in ("read", "write"):
                if self.ssd is not None:     # the SSD thread passes the end on to the done thread
                    self.ssd_q[d].put(err)
                else:
                    self.q[d].put(err)
                err = None

    def _ssd_loop(self, direction: str):
        err = None
        try:
            if self.cuda:
                import torch
                torch.cuda.set_device(self.device)
            while (item := self.ssd_q[direction].get()) is not None:
                if isinstance(item, WorkerError):
                    err = item
                    break
                self._ssd_op(*item)
        except BaseException:   # noqa: BLE001 - reported to the controller, which fails the run
            err = WorkerError(self.stage, "tier SSD thread:\n" + traceback.format_exc())
        finally:
            self.q[direction].put(err)

    def _done_loop(self, direction: str):
        import zmq
        push = self.ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 5000)
        push.connect(self.cfg.report_addr())
        try:
            while (item := self.q[direction].get()) is not None:
                if isinstance(item, WorkerError):
                    push.send(encode(item))
                    break
                op_id, wait, start_ns, nbytes = item
                if isinstance(wait, float):     # fake engine: due time (monotonic)
                    d = wait - time.monotonic()
                    if d > 0:
                        time.sleep(d)
                elif wait is not None:
                    wait.synchronize()
                push.send(encode(TierDone(self.stage, op_id, start_ns, time.time_ns(), nbytes)))
        finally:
            push.close()

    # --- one op -----------------------------------------------------------------------------------------------------

    @staticmethod
    def direction(op: TierMove) -> str:
        return "read" if op.op == "read" else "write"

    def submit(self, op: TierMove):
        start_ns = time.time_ns()
        d = self.direction(op)
        if not self.real:
            ssd = "ssd" in (op.tier, op.host_tier)
            blocks = len(op.slots) if ssd else op.b_hi - op.b_lo     # an SSD op moves whole slots
            nbytes = (blocks + len(op.state_slots)) * self.slot_bytes
            bw = (self.tc.ssd_read_gb_s if ssd else self.tc.dram_read_gb_s) * 1e9
            self._due[d] = max(self._due.get(d, 0.0), time.monotonic()) + nbytes / bw
            self.q[d].put((op.op_id, self._due[d], start_ns, nbytes))
            return
        if op.tier == "ssd":
            self.ssd_q[d].put((op, start_ns))
            return
        if op.tier != "dram" or op.host_tier is not None or op.op not in ("read", "write"):
            raise ValueError(f"tier op {op.op} {op.tier} via {op.host_tier}")
        ev, nbytes = self._move(d, self.dram.ptr, op.kv_blocks, op.state_blocks, op.slots, op.state_slots, op.b_lo,
                                op.b_hi, op.block_start)
        self.q[d].put((op.op_id, ev, start_ns, nbytes))

    def _batch(self, stream: str, src, dst, lens, wait=()):
        """One `cuMemcpyBatchAsync` on `stream` after the events `wait`; its end event (None on CPU, done)."""
        if not self.cuda:
            for s, t, k in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(lens).tolist()):
                ctypes.memmove(t, s, k)
            return None
        import torch
        st = self.streams[stream]
        for ev in wait:
            if ev is not None:
                st.wait_event(ev)
        with torch.cuda.stream(st):
            self.swap(*(torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)) for x in (src, dst, lens)))
        ev = torch.cuda.Event(blocking=True)
        ev.record(st)
        return ev

    def _move(self, d: str, base: int, kv, st, slots, sslots, b_lo: int, b_hi: int, block_start: int, ready=None):
        """Blocks [b_lo, b_hi) and (if `sslots`) the state between HBM and host slots at `base` (block i in slot
        slots[i - block_start]), "read" host -> HBM or "write" HBM -> host. `ready(i)`: called before the host side of
        blocks below i (None: the state) is read (an SSD read waits for its file reads there). Returns (the last
        copy's event, bytes of HBM pieces moved)."""
        L = self.layout
        if self.ring is None:                       # direct: one batch of the pieces, host side at `base`
            if ready is not None:
                ready(b_hi)
                ready(None)
            dev, host, lens = L.pieces(kv, st, slots, sslots, b_lo, b_hi, block_start)
            host = host + base
            src, dst = (host, dev) if d == "read" else (dev, host)
            return self._batch(d, src, dst, lens), int(lens.sum())
        sb, C = self.slot_bytes, self.chunk
        units = [("kv", i, min(i + C, b_hi)) for i in range(b_lo, b_hi, C)]
        if len(sslots):                             # the state: its pieces over len(sslots) slots, cut like the KV
            sdev, sroff, slen = L.pieces(kv, st, [], list(range(len(sslots))), 0, 0, 0)
            sidx = sroff // sb                      # pieces never cross a slot (Layout splits them)
            units += [("state", j, min(j + C, len(sslots))) for j in range(0, len(sslots), C)]
        ev, nbytes = None, 0
        for kind, a, b in units:
            n = b - a
            if kind == "kv":
                if ready is not None:
                    ready(b)
                dev, roff, lens = L.pieces(kv, st, list(range(n)), [], a, b, a)
                hosts, hlen = [base + h * sb for h in slots[a - block_start:b - block_start]], L.kv_slot_bytes
            else:
                if ready is not None:
                    ready(None)
                m = (sidx >= a) & (sidx < b)
                dev, roff, lens = sdev[m], sroff[m] - a * sb, slen[m]
                hosts, hlen = [base + h * sb for h in sslots[a:b]], sb
            nbytes += int(lens.sum())
            with self._lock:
                pos = self._ring_next[d]
                self._ring_next[d] = (pos + 1) % self.depth
                rbase = self._ring_base[d] + pos * C * sb
                rings = [rbase + k * sb for k in range(n)]
                hl = np.full(n, hlen, dtype=np.int64)
                if d == "read":                     # host -> ring (PCIe), then ring -> the stages (NVLink)
                    h = self._batch("read", hosts, rings, hl, (self._ring_free[d][pos],))
                    ev = self._batch("read_nvl", roff + rbase, dev, lens, (h,))
                else:                               # the stages -> ring, then ring -> host
                    g = self._batch("write_nvl", dev, roff + rbase, lens, (self._ring_free[d][pos],))
                    ev = self._batch("write", rings, hosts, hl, (g,))
                self._ring_free[d][pos] = ev
        return ev, nbytes

    def _ssd_op(self, op: TierMove, start_ns: int):
        """An SSD op, on this direction's SSD thread: its file I/O on the pool, its copies on the CUDA stream."""
        sb, L = self.slot_bytes, self.layout
        host = self.dram if op.host_tier == "dram" else self.staging
        files = op.slots + op.state_slots
        hosts = op.host_slots + op.host_state
        assert len(files) == len(hosts), (op.op_id, len(files), len(hosts))
        if op.op == "read":                 # file -> host slots (all blocks), to HBM [b_lo, b_hi) as slots land
            futs = [self.pool.submit(self.ssd.read, f, host.ptr + h * sb) for f, h in zip(files, hosts)]
            n = len(op.slots)

            def ready(i):
                for f in (futs[:i] if i is not None else futs[n:]):
                    f.result()
            ev, _ = self._move("read", host.ptr, op.kv_blocks, op.state_blocks, op.host_slots, op.host_state,
                               op.b_lo, op.b_hi, 0, ready)
            for f in futs:
                f.result()
        else:
            if op.op == "write":            # HBM -> staging, then the file
                ev, _ = self._move("write", host.ptr, op.kv_blocks, op.state_blocks, op.host_slots, op.host_state,
                                   op.b_lo, op.b_hi, op.block_start)
                if ev is not None:
                    ev.synchronize()
            elif op.op != "demote":
                raise ValueError(f"SSD op {op.op}")
            for f in [self.pool.submit(self.ssd.write, f, host.ptr + h * sb) for f, h in zip(files, hosts)]:
                f.result()
            ev = None
        self.q[self.direction(op)].put((op.op_id, ev, start_ns, len(files) * sb))

