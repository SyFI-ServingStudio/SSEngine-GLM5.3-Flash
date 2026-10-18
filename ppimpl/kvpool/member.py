"""A stage process's side of the KV pool (lanes/m6 TIERS.md §3-§6, §13.2): `PoolMember`, the data plane.

Every member moves its 1/n share (n = PP degree, `num_stages`) of every op (balanced storage, `placement.py`). The
leader sends a `TierBatch` of ops whose dependencies are done to all n members' tier sockets (`tier_move`); a
member's receive thread hands each DRAM op to its direction's submit thread (reads never wait behind writes' host
work), which maps the op's slots to host addresses (`backends.py`), asks the placement for the pieces of its share
over all stages' KV (the other stages' KV opened by CUDA IPC at start-up, `share_ranges`), hands them to its
`mover.Mover` (read streams host -> HBM, write streams HBM -> host), and a done thread per direction reports
`TierDone` when the copies completed. No wait on the compute stream: a write's blocks are final and pinned until it
is done, a read's blocks are held by a request that computes only after all n `TierDone`s.

SSD ops run on two threads, their file I/O on `SsdTier`'s io_uring (reads first in the kernel), into and out of host
slots: DRAM's (a demote) or a staging arena's (every SSD read; a write-back to SSD without
DRAM, or of a context larger than DRAM). An SSD read lands slot by slot and each KV block's H2D is enqueued as soon as its slot is
in, the state after its slots: read time ~ SSD time + one block. The file is shard q (this member's quarter of every
slot) of the set the leader took (`shards.py`), kept for the next run; its contents are never trusted. Reads run one
op at a time; writes and demotes do not wait for their file writes: many are in flight at once (the leader orders
ops that share slots, `TierOp.deps`), and each is reported when its last slot is on the SSD.

The fake model has no KV: its member moves nothing and reports each op after the time the configured bandwidth
gives its bytes, one op after another per direction (CPU tests, fake runs).
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
import traceback

from ppimpl.kvpool.backends import HostTier, SsdTier, _ok, cgroup_peak_gb, node_cpus, num_slots
from ppimpl.kvpool.shards import shard_bytes, shard_path
from ppimpl.kvpool.layout import ViewRange
from ppimpl.kvpool.manager import TierOp
from ppimpl.kvpool.mover import EVENT_POOL, Mover, Segment
from ppimpl.kvpool.placement import BalancedPlacement
from ppimpl.runtime.transport import Shutdown, TierDone, TierMove, WorkerError, decode_tier, encode

PROBE_S, PROBE_LATE_S = 0.005, 0.020   # the GIL / CPU probe (`PoolMember._probe_loop`, with the trace)
FAKE_BLOCK_BYTES = 8576 * 5996  # one block over all stages (GLM-5.3 PP4: 4 x 1,499 B/token); a GPU's slot is 1/n of
                                # it rounded up to 4 KiB (PP4: 12,857,344 B, the real balanced slot, TIERS.md §3)


def tier_move(o: TierOp) -> TierMove:
    """What every member needs of a tier op: HBM blocks [b_lo, b_hi) of the context move (none for a demote)."""
    b_lo, b_hi = o.blocks
    return TierMove(o.op_id, o.op, o.tier, o.host_tier, b_lo, b_hi, o.block_start, [b[:b_hi] for b in o.kv_blocks],
                    o.state_blocks, o.slots, o.state_slots, o.host_slots, o.host_state)


def share_ranges(stage: int, n: int, ranges: list[ViewRange]) -> tuple[list[ViewRange], list[int]]:
    """All stages' ranges with device addresses valid in this process: each member exports the allocations behind its
    ranges (cuMemGetAddressRange base, cudaIpcGetMemHandle), and opens the others' (lazy peer access). Collective
    over the members' gloo group (the pipe's). Returns (ranges, opened pointers to close)."""
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


class PoolMember:
    """One stage process's pool side: its tier socket, host tiers, mover, and the threads that run and report ops.

    Built in two steps so the arena's first touch overlaps the model load: `PoolMember(...)` allocates and starts
    touching; `attach(ranges)` (after the model, collective over the members) opens the peers' KV, builds the
    placement, registers the arena, allocates the mover's ring and returns (slot_bytes, state_slots) for `Ready`;
    `start()` begins serving ops."""

    def __init__(self, cfg, stage: int, device, zmq_ctx, touch_cpus: set[int] | None = None):
        tc = cfg.tiers
        self.cfg, self.tc, self.stage, self.n = cfg, tc, stage, cfg.num_stages
        self._touch_cpus = touch_cpus
        self.touch_wait_s = 0.0
        self.device, self.cuda = device, device.type == "cuda"
        # PPIMPL_TIER_TRACE=0 turns off the per-batch trace of DRAM ops (tier_trace_s<stage>.jsonl in the run dir: host
        # time before / in each batch call, the calling thread's CPU time in it, each batch's GPU start / end)
        self.trace = self.cuda and os.environ.get("PPIMPL_TIER_TRACE", "1") != "0"
        self._traces: dict[int, tuple] = {}
        self._owned: dict[int, list] = {}          # op id -> the mover's pooled events it took (released when done)
        self._trace_log = None
        self.ctx = zmq_ctx
        self.placement: BalancedPlacement | None = None
        self.mover: Mover | None = None
        self.slot_bytes = tc.slot_bytes or 0
        self.opened: list[int] = []
        self.dram: HostTier | None = None
        self.staging: HostTier | None = None
        self.ssd: SsdTier | None = None
        self._touch: threading.Thread | None = None
        self.real = False
        self.threads: list[threading.Thread] = []
        self.stop = threading.Event()
        self.q = {"read": queue.Queue(), "write": queue.Queue()}       # done threads: (op id, wait, start, bytes)
        self.ssd_q = {"read": queue.Queue(), "write": queue.Queue()}   # SSD threads: (op, start)
        self.sub_q = {"read": queue.Queue(), "write": queue.Queue()}   # DRAM ops' submit threads: op
        self._ssd_writes = 0                       # SSD writes / demotes whose file writes are in flight
        self._ssd_cv = threading.Condition()
        self._due: dict[str, float] = {}           # fake member: when each direction's last op is done
        if tc.hops not in ("two", "direct"):
            raise ValueError(f"tiers.hops: {tc.hops!r} (two | direct)")
        if tc.ring_depth < 1 or tc.ring_chunk_slots < 1:
            raise ValueError(f"tiers.ring_depth {tc.ring_depth}, ring_chunk_slots {tc.ring_chunk_slots}: >= 1")
        real = cfg.model != "fake"
        if tc.dram_gb and self.cuda and real:
            from ppimpl.runtime.cpus import _gpu_node
            node = _gpu_node(device.index)
            self.dram = HostTier(int(tc.dram_gb * 1e9), node, True)
            self._touch = threading.Thread(target=self.dram.touch, args=(node_cpus(touch_cpus, node),),
                                           name="tier-touch", daemon=True)
            self._touch.start()

    def attach(self, ranges: list[ViewRange] | None) -> tuple[int, int]:
        """`ranges`: this stage's KV ranges (`layout.from_stage_kv`), None for a model without KV (fake)."""
        if ranges is None:
            self.slot_bytes = self.slot_bytes or -(-FAKE_BLOCK_BYTES // self.n // 4096) * 4096
            return self.slot_bytes, self.tc.state_slots
        self.real = True
        allr, self.opened = share_ranges(self.stage, self.n, ranges) if self.cuda else (ranges, [])
        self.placement = P = BalancedPlacement(allr, self.n, self.stage, slot_bytes=self.tc.slot_bytes)
        sb = self.slot_bytes = P.slot_bytes
        node = None
        if self.cuda:
            from ppimpl.runtime.cpus import _gpu_node
            node = _gpu_node(self.device.index)
        if self.dram is None and self.tc.dram_gb:
            self.dram = HostTier(num_slots(self.tc.dram_gb, sb) * sb, node, self.cuda)
            self.dram.touch(node_cpus(self._touch_cpus, node))
        t = time.perf_counter()
        if self._touch is not None:
            self._touch.join()
        self.touch_wait_s = time.perf_counter() - t      # attach (after the model) waiting for the first touch
        if self.dram is not None and self.dram.touch_error is not None:
            raise self.dram.touch_error
        if self.dram is not None:
            self.dram.register(sb)
            assert num_slots(self.tc.dram_gb, sb) * sb <= self.dram.nbytes
        if self.tc.ssd_gb:
            self.staging = HostTier(num_slots(self.tc.staging_gb, sb) * sb, node, self.cuda)
            self.staging.touch(node_cpus(self._touch_cpus, node))
            if self.staging.touch_error is not None:
                raise self.staging.touch_error
            self.staging.register(sb)
            if not self.tc.ssd_dir:
                raise ValueError("SSD tier: tiers.ssd_dir is the shard set the leader took (shards.ShardSet)")
            assert num_slots(self.tc.ssd_gb, sb) * sb <= shard_bytes(self.tc.ssd_gb)
            self.ssd = SsdTier(shard_path(self.tc.ssd_dir, self.stage), shard_bytes(self.tc.ssd_gb), sb,
                               [(a.base, a.nbytes) for a in (self.dram, self.staging) if a is not None])
        self.mover = Mover(self.device, sb, self.tc.hops, self.tc.ring_depth, self.tc.ring_chunk_slots)
        print(f"stage {self.stage} tiers: slot {self.slot_bytes} B, state {P.state_slots} slots, "
              f"{len(P.kv_len)} kv + {len(P.st_len)} state pieces per block/context, "
              + (f"dram {self.dram.nbytes / 1e9:.1f} GB on node {self.dram.node} (touch {self.dram.touch_s:.1f} s, "
                 f"of it after the model {self.touch_wait_s:.1f} s, register {self.dram.register_s:.1f} s, {self.dram.placement()}), "
                 if self.dram else "")
              + (f"{self.ssd.info}, staging {self.staging.nbytes / 1e9:.1f} GB "
                 f"({self.staging.placement()}), " if self.ssd else "")
              + f"{len(self.opened)} peer allocations, "
              + self.mover.info
              + (f", cgroup memory.peak {pk:.0f} GB" if (pk := cgroup_peak_gb()) is not None else ""), flush=True)
        return self.slot_bytes, P.state_slots

    def start(self):
        for name in ("read", "write"):
            t = threading.Thread(target=self._done_loop, args=(name,), name=f"tier-{name}", daemon=True)
            t.start()
            self.threads.append(t)
            t = threading.Thread(target=self._submit_loop, args=(name,), name=f"tier-submit-{name}", daemon=True)
            t.start()
            self.threads.append(t)
            if self.ssd is not None:
                t = threading.Thread(target=self._ssd_loop, args=(name,), name=f"tier-ssd-{name}", daemon=True)
                t.start()
                self.threads.append(t)
        t = threading.Thread(target=self._recv_loop, name="tier-recv", daemon=True)
        t.start()
        self.threads.append(t)
        if self.trace:
            t = threading.Thread(target=self._probe_loop, name="tier-probe", daemon=True)
            t.start()
            self.threads.append(t)

    def _probe_loop(self):
        """Instrumentation (with the trace): a thread that wakes every PROBE_S and logs each wake-up more than
        PROBE_LATE_S late to tier_probe_s<stage>.jsonl, with how long it was runnable without a CPU then. Late but not
        waiting for a CPU: another thread held the GIL (a tier copy call that blocks at the same time waits for it)."""
        from ppimpl.kvpool.mover import runq_ns
        with open(os.path.join(self.cfg.run_dir, f"tier_probe_s{self.stage}.jsonl"), "a") as f:
            while not self.stop.is_set():
                t0, q0 = time.perf_counter(), runq_ns()
                time.sleep(PROBE_S)
                late, q = time.perf_counter() - t0 - PROBE_S, runq_ns() - q0
                if late > PROBE_LATE_S:
                    f.write(json.dumps({"at_ns": time.time_ns(), "late_ms": round(late * 1e3, 2),
                                        "runq_ms": round(q / 1e6, 2)}) + "\n")
                    f.flush()

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=10)
        if self.mover is not None and self.cuda:
            print(f"stage {self.stage} tier events: {self.mover.events_made} made, {EVENT_POOL} of them at start",
                  flush=True)
        if self.cuda:
            import torch
            torch.cuda.synchronize(self.device)
            from cuda.bindings import runtime as rt
            for p in self.opened:
                rt.cudaIpcCloseMemHandle(p)
        if self._trace_log is not None:
            self._trace_log.close()
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
                for op in sorted(msg.ops, key=lambda o: o.op != "read"):    # reads first
                    if self.real and op.tier == "dram":
                        self.sub_q[self.direction(op)].put(op)   # its direction's submit thread
                    else:
                        self.submit(op)                          # SSD: to its thread; fake: timed here
        except BaseException:   # noqa: BLE001 - reported to the controller, which fails the run
            err = WorkerError(self.stage, "tier thread:\n" + traceback.format_exc())
        finally:
            sock.close(linger=0)
            for d in ("read", "write"):      # the submit thread passes the end on (SSD thread, done thread)
                self.sub_q[d].put(err)
                err = None

    def _submit_loop(self, direction: str):
        """A direction's DRAM ops, one after another: reads do not wait behind writes' host work."""
        err = None
        try:
            if self.cuda:
                import torch
                torch.cuda.set_device(self.device)
            while (item := self.sub_q[direction].get()) is not None:
                if isinstance(item, WorkerError):
                    err = item
                    break
                self.submit(item)
        except BaseException:   # noqa: BLE001 - reported to the controller, which fails the run
            err = WorkerError(self.stage, f"tier {direction} submit thread:\n" + traceback.format_exc())
        finally:
            (self.ssd_q if self.ssd is not None else self.q)[direction].put(err)

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
            with self._ssd_cv:                  # the writes in flight report before the end does
                self._ssd_cv.wait_for(lambda: self._ssd_writes == 0, timeout=60)
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
                op_id, wait, start_ns, nbytes, submit_ns, mark = item
                gpu_ns = None
                if isinstance(wait, float):     # fake member: due time (monotonic)
                    d = wait - time.monotonic()
                    if d > 0:
                        time.sleep(d)
                elif wait is not None:
                    wait.synchronize()
                    if mark is not None:
                        gpu_ns = int(mark.elapsed_time(wait) * 1e6)
                push.send(encode(TierDone(self.stage, op_id, start_ns, time.time_ns(), nbytes, submit_ns, gpu_ns)))
                if (tr := self._traces.pop(op_id, None)) is not None:
                    try:                        # instrumentation: a failure turns the trace off, not the run
                        self._write_trace(op_id, start_ns, mark, *tr)
                    except Exception:           # noqa: BLE001
                        print(f"stage {self.stage} tier trace off:\n{traceback.format_exc()}", flush=True)
                        self.trace = False
                if (owned := self._owned.pop(op_id, None)) is not None:
                    self.mover.release(owned)   # its copies completed (wait), its trace is read
        finally:
            push.close()

    def _write_trace(self, op_id: int, start_ns: int, mark, kind: str, t0: int, tr: list):
        """One line per traced op (done thread, after its last copy completed): us from the op's submit start (host)
        and from its mark (GPU)."""
        if self._trace_log is None:
            self._trace_log = open(os.path.join(self.cfg.run_dir, f"tier_trace_s{self.stage}.jsonl"), "a")
        us = lambda ns: round(ns / 1e3, 1)
        self._trace_log.write(json.dumps({"op": op_id, "kind": kind, "start_ns": start_ns, "batches": [
            {"stream": s, "pieces": n, "bytes": b, "host_at_us": us(h0 - t0), "call_us": us(h1 - h0), "cpu_us": us(c),
             "gpu_start_us": round(mark.elapsed_time(e0) * 1e3, 1),
             "gpu_end_us": round(mark.elapsed_time(e1) * 1e3, 1)}
            for s, n, b, h0, h1, c, e0, e1 in tr]}) + "\n")

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
            self.q[d].put((op.op_id, self._due[d], start_ns, nbytes, time.time_ns(), None))
            return
        if op.tier == "ssd":
            self.ssd_q[d].put((op, start_ns))
            return
        if op.tier != "dram" or op.host_tier is not None or op.op not in ("read", "write"):
            raise ValueError(f"tier op {op.op} {op.tier} via {op.host_tier}")
        # GPU time of the op (TierDone.gpu_ns), from the stream its first copy is on
        owned: list = []
        mark = self.mover.mark(d, owned)
        tr = [] if self.trace else None
        t0 = time.perf_counter_ns()
        ev, nbytes = self._move(d, self.dram, op, op.slots, op.state_slots, op.block_start, tr=tr, owned=owned)
        if tr is not None:
            self._traces[op.op_id] = (op.op, t0, tr)
        if owned:
            self._owned[op.op_id] = owned
        self.q[d].put((op.op_id, ev, start_ns, nbytes, time.time_ns(), mark))

    def _move(self, d: str, host: HostTier, op: TierMove, slots: list[int], sslots: list[int], block_start: int,
              ready=None, tr: list | None = None, owned: list | None = None):
        """The op's blocks [b_lo, b_hi) and (if `sslots`) the state between HBM and `host`'s slots (block i in slot
        slots[i - block_start]), "read" host -> HBM or "write" HBM -> host. `ready(i)`: called before the host side of
        blocks below i (None: the state) is read (an SSD read waits for its file reads there). Returns (the last
        copy's event, bytes of HBM pieces moved). `owned`: gets the mover's pooled events the copies took."""
        P, b_lo, b_hi = self.placement, op.b_lo, op.b_hi
        dev, off, lens = P.pieces(op.kv_blocks, op.state_blocks, list(range(b_hi - b_lo)), [], b_lo, b_hi, b_lo)
        segs = [Segment([host.ptr(s) for s in slots[b_lo - block_start:b_hi - block_start]], P.kv_slot_bytes, dev,
                        off, lens)]
        if len(sslots):
            segs.append(Segment([host.ptr(s) for s in sslots], self.slot_bytes,
                                *P.pieces(op.kv_blocks, op.state_blocks, [], list(range(len(sslots))), 0, 0, 0)))
        wait = None if ready is None else lambda i, k: ready(b_lo + k if i == 0 else None)
        return self.mover.transfer_async(d, segs, wait, tr, owned)

    def _ssd_op(self, op: TierMove, start_ns: int):
        """An SSD op, on this direction's SSD thread: its file I/O on the SSD pool, its copies on the mover."""
        sb = self.slot_bytes
        host = self.dram if op.host_tier == "dram" else self.staging
        files = op.slots + op.state_slots
        hosts = [host.ptr(h) for h in op.host_slots + op.host_state]
        assert len(files) == len(hosts), (op.op_id, len(files), len(hosts))
        if op.op == "read":                 # file -> host slots (all blocks), to HBM [b_lo, b_hi) as slots land
            futs = self.ssd.read(files, hosts)
            n = len(op.slots)

            def ready(i):
                for f in (futs[:i] if i is not None else futs[n:]):
                    f.result()
            owned: list = []
            ev, _ = self._move("read", host, op, op.host_slots, op.host_state, 0, ready, owned=owned)
            for f in futs:
                f.result()
            if owned:
                self._owned[op.op_id] = owned
        else:
            if op.op == "write":            # HBM -> staging, then the file
                owned = []
                ev, _ = self._move("write", host, op, op.host_slots, op.host_state, op.block_start, owned=owned)
                if ev is not None:
                    ev.synchronize()
                self.mover.release(owned)
            elif op.op != "demote":
                raise ValueError(f"SSD op {op.op}")
            self._report_when_written(op, self.ssd.write(files, hosts), start_ns, len(files) * sb)
            return
        self.q[self.direction(op)].put((op.op_id, ev, start_ns, len(files) * sb, time.time_ns(), None))

    def _report_when_written(self, op: TierMove, futs: list, start_ns: int, nbytes: int):
        """Report an SSD write or demote once its last file write completed (on the io_uring's thread), without
        holding the SSD write thread: the next op's writes queue behind it in the ring. A failed write fails the run
        (`WorkerError`)."""
        done = (op.op_id, None, start_ns, nbytes)
        if not futs:
            self.q["write"].put((*done, time.time_ns(), None))
            return
        left, failed, lock = [len(futs)], [False], threading.Lock()
        with self._ssd_cv:
            self._ssd_writes += 1

        def one(f):
            exc = f.exception()
            with lock:
                left[0] -= 1
                last = left[0] == 0
                first_failure = exc is not None and not failed[0]
                failed[0] |= exc is not None
            if first_failure:
                self.q["write"].put(WorkerError(self.stage, f"tier SSD write of op {op.op_id}: {exc!r}"))
            if last:
                if not failed[0]:
                    self.q["write"].put((*done, time.time_ns(), None))
                with self._ssd_cv:
                    self._ssd_writes -= 1
                    self._ssd_cv.notify_all()

        for f in futs:
            f.add_done_callback(one)
