"""One stage worker: plan -> (recv) -> forward -> (send) -> report (ppimpl/runtime/README.md, "Worker loop").

The main thread only enqueues GPU work. A reporter thread waits for each microbatch's end event and sends `StageDone`
with GPU times on the host clock, so the controller and all stages share one time base.
"""

import cProfile
import gc
import io
import json
import os
import pstats
import queue
import signal
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import torch
import zmq

from ppimpl.config import HIDDEN_WIDTH, RunConfig
from ppimpl.runtime.cpus import cpu_list, pin_process, threads, worker_cpus
from ppimpl.runtime.host_stamps import WorkerStamps
from ppimpl.runtime.transport import (open_pipe, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_worker,
                                      encode, report_socket, worker_sockets)

ANCHOR_EVERY_S = 2.0   # re-anchor CUDA events to the host clock (keeps elapsed_time short and float32-exact)


class Clock:
    """Marks on the compute stream, converted to host-clock ns. CPU: marks are host times already."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        if self.cuda:
            self._stream = torch.cuda.Stream(device)   # idle: an event recorded here completes at once
            self.anchor = None
            self._anchored_at = 0.0
            self.reanchor()

    def reanchor(self):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self._stream)
        while not ev.query():
            pass
        self.anchor = (ev, time.time_ns())
        self._anchored_at = time.monotonic()

    def maybe_reanchor(self):
        if self.cuda and time.monotonic() - self._anchored_at > ANCHOR_EVERY_S:
            self.reanchor()

    def mark(self):
        if not self.cuda:
            return time.time_ns()
        ev = torch.cuda.Event(enable_timing=True, blocking=True)
        ev.record()
        return ev

    @staticmethod
    def to_ns(mark, anchor) -> int:
        if anchor is None:
            return mark
        ev, host_ns = anchor
        return host_ns + int(ev.elapsed_time(mark) * 1e6)


class RunAhead:
    """Bounds how far the host launches ahead of the GPU, so the main thread never blocks inside a launch.

    CUDA's launch queue holds ~1,024 kernels (measured/issues.md: the host got at most ~1.7 eager forwards ahead). A
    launch into a full queue blocks the main thread inside model.run, and a tier thread's next CUDA call (event
    create / record, cuMemcpyBatchAsync) waits behind it: 100-370 ms late starts of tier ops on stages 2/3 (job 8095,
    minutes 2-5) with eager microbatches (> 8,192 tokens, no CUDA graph; ~400-600 kernels each) queued back to back,
    against a graph replay's ~16 graphs + 15 eager breaks. Rules, before microbatch k is launched:
    - k-2 has ended (depth 2): k-1 stays queued while k is launched, so the GPU idles only if launching k takes longer
      than computing k-1 (a replay's host side ~5 ms vs >= 12 ms of compute; eager ~20 ms vs >= 70 ms).
    - an eager k waits until no other eager microbatch is queued: two eager forwards fill the queue. Back to back,
      the GPU then waits for the wake-up and k's first launch only (eager host time < its GPU time after that).
    The waits are on blocking-sync events (`Clock.mark`): the thread sleeps with the GIL released and holds no driver
    lock. `waits` / `wait_s`: how often and how long the main thread waited (printed at shutdown)."""

    DEPTH = 2

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.q: deque = deque()            # (end event, eager) of the microbatches launched and not known to be done
        self.waits, self.wait_s = 0, 0.0

    def before(self, eager: bool):
        if not self.cuda:
            return
        t = None
        while self.q and (len(self.q) >= self.DEPTH or (eager and any(e for _, e in self.q))):
            ev, _ = self.q.popleft()
            if not ev.query():
                t = time.perf_counter() if t is None else t
                ev.synchronize()
        if t is not None:
            self.waits += 1
            self.wait_s += time.perf_counter() - t

    def after(self, end, eager: bool):
        if self.cuda:
            self.q.append((end, eager))


class ProfileWindow:
    """cudaProfilerStart/Stop around microbatches [a, b), for `nsys --capture-range=cudaProfilerApi`."""

    def __init__(self, mbs: list[int] | None):
        self.a, self.b = mbs if mbs else (None, None)

    def before(self, mb_id: int):
        if mb_id == self.a:
            torch.cuda.profiler.start()

    def after(self, mb_id: int):
        if self.b is not None and mb_id == self.b - 1:
            torch.cuda.profiler.stop()


class HostProfile:
    """cProfile of the host side of model.forward (metadata, launches) for microbatches [a, b) on the stages listed,
    from PPIMPL_HOST_PROFILE="a:b[@stages]" (default stage 0): <run dir>/hostprof/stage<s>_mb<m>.txt, the top
    functions by cumulative time. A diagnosis aid: the profiler slows the host down."""

    def __init__(self, stage: int, run_dir: str):
        spec = os.environ.get("PPIMPL_HOST_PROFILE", "")
        rng, _, stages = spec.partition("@")
        self.on = bool(rng) and stage in [int(x) for x in (stages or "0").split(",")]
        if self.on:
            self.a, self.b = (int(x) for x in rng.split(":"))
            self.dir = Path(run_dir) / "hostprof"
            self.dir.mkdir(exist_ok=True)
        self.stage, self.prof = stage, None

    def before(self, mb_id: int):
        if self.on and self.a <= mb_id < self.b:
            self.prof = cProfile.Profile()
            self.prof.enable()

    def after(self, mb_id: int):
        if self.prof is not None:
            self.prof.disable()
            out = io.StringIO()
            pstats.Stats(self.prof, stream=out).sort_stats("cumulative").print_stats(60)
            (self.dir / f"stage{self.stage}_mb{mb_id}.txt").write_text(out.getvalue())
            self.prof = None


class FirstUse:
    """After `Ready`: vLLM's JIT monitor (Triton, CuTeDSL and TileLang compiles, Triton autotuning; verbose: the
    specialization) plus, per Triton compile or disk-cache load, the microbatch and the host time it took: "stage S
    mb M: first use after warm-up: triton <kernel> T ms", also appended to `jit_stage<S>.jsonl` in the run dir.
    Warm-up is meant to have run every variant (StageModel.warmup); each such line is one it missed, and fails the
    run (bench/run.py). The hooks run only on a compile path, never on a cached
    launch."""

    def __init__(self, stage: int, run_dir: str | None = None):
        self.stage, self.mb, self.t0 = stage, None, {}
        self.log = None if run_dir is None else Path(run_dir) / f"jit_stage{stage}.jsonl"

    def install(self):
        try:
            from triton import knobs
        except ImportError:
            return
        from vllm.utils import jit_monitor

        knobs.runtime.jit_cache_hook = self._start
        knobs.runtime.jit_post_compile_hook = self._done
        jit_monitor.activate(mode="warn", verbose=True)    # chains to self._done

    def _start(self, **kw):
        self.t0[str(kw.get("key"))] = time.perf_counter()
        return False                      # compile as usual

    def _done(self, **kw):
        t = self.t0.pop(str(kw.get("key")), None)
        ms = -1.0 if t is None else 1e3 * (time.perf_counter() - t)
        name = getattr(kw.get("fn"), "name", "?")
        print(f"stage {self.stage} mb {self.mb}: first use after warm-up: triton {name} {ms:.1f} ms", flush=True)
        if self.log is not None:          # read by bench/run.py: any line fails the run (run_failure)
            with open(self.log, "a") as f:
                f.write(json.dumps({"stage": self.stage, "mb": self.mb, "kernel": name, "ms": round(ms, 1)}) + "\n")


class Reporter(threading.Thread):
    """Owns the worker's report socket. Items: a message to send as is, or a finished-microbatch record."""

    def __init__(self, ctx: zmq.Context, addr: str, device: torch.device):
        super().__init__(daemon=True)
        self.ctx, self.addr, self.device = ctx, addr, device
        self.q: queue.Queue = queue.Queue()

    def run(self):
        sock = report_socket(self.ctx, self.addr)
        copy_stream = None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            copy_stream = torch.cuda.Stream(self.device)   # the token copy must not queue behind later plans
        while (item := self.q.get()) is not None:
            if not isinstance(item, tuple):
                sock.send(encode(item))
                continue
            stage, mb_id, start, end, anchor, wait, tokens, tok_ids, host = item
            if wait is not None:
                wait.synchronize()
            toks = None
            if tok_ids is not None:
                if copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(wait)
                        tokens = tokens.cpu()
                toks = list(zip(tok_ids, tokens.tolist()))
            sock.send(encode(StageDone(stage, mb_id, Clock.to_ns(start, anchor), Clock.to_ns(end, anchor), toks,
                                       host)))
        sock.close()


def build_model(cfg: RunConfig, stage: int, device: torch.device):
    if cfg.model == "fake":
        from ppimpl.model.fake import FakeStageModel
        return FakeStageModel(cfg, stage, device)
    from ppimpl.model.stage import StageModel
    return StageModel(cfg, stage, device, ipc_kv=cfg.tiers is not None)   # tiers: peers copy our KV (CUDA IPC)

# PPIMPL_TIER_SWITCH_S=S: the GIL switch interval with tiers on (default 0: Python's 5 ms). For the A/B under session
# load (job 7966, idle: no effect; in session runs a tier op's submit waits for the GIL behind the launch thread).
TIER_SWITCH_S = float(os.environ.get("PPIMPL_TIER_SWITCH_S", "0"))


@torch.inference_mode()   # all of it: model construction, warm-up, receive, forward, send (StageModel's warm-up
def run_worker(cfg: RunConfig, stage: int):   # makes inference tensors, which autograd-tracked code may not use)
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the controller stops us (Shutdown), or we see it is gone
    if cfg.tiers is not None and TIER_SWITCH_S > 0:   # the tier threads get the GIL within S of the launch thread
        sys.setswitchinterval(TIER_SWITCH_S)
    parent = os.getppid()
    n = cfg.num_stages
    last = stage == n - 1
    cuda = cfg.device == "cuda"
    device = torch.device("cuda", stage) if cuda else torch.device("cpu")
    cpus = None
    job_cpus = os.sched_getaffinity(0)   # before the pinning below: the tier arena's first touch uses the job's CPUs
    if cuda:
        torch.cuda.set_device(device)
        cpus = worker_cpus(stage, n)   # whole cores on the GPU's NUMA node, before our threads start (ppimpl/runtime/cpus.py)
        if cpus:
            pin_process(cpus)
    torch.set_num_threads(1)           # the host side only launches kernels; CPU tensor work here is tiny
    ctx = zmq.Context()
    plans = worker_sockets(ctx, cfg.plan_addr(stage))
    reporter = Reporter(ctx, cfg.report_addr(), device)
    reporter.start()
    pipe, tiers, ok, mbs = None, None, False, 0
    try:
        if cfg.tiers is not None:   # DRAM arena first-touched while the model loads (ppimpl/kvpool/member.py)
            from ppimpl.kvpool.member import PoolMember
            tiers = PoolMember(cfg, stage, device, ctx, touch_cpus=job_cpus)
        width = cfg.fake.hidden_width if cfg.model == "fake" else HIDDEN_WIDTH
        pipe = open_pipe(cfg, stage, device, width)   # before the model: its KV pool is sized from what is left
        t_model = time.monotonic()
        model = build_model(cfg, stage, device)
        slots = (0, 0)
        if tiers is not None:       # collective over the workers (CUDA IPC of every stage's KV)
            t_attach = time.monotonic()
            slots = tiers.attach(model.tier_ranges() if hasattr(model, "tier_ranges") else None)
            tiers.start()
            print(f"stage {stage} start-up: model (load, warm-up, graphs) {t_attach - t_model:.1f} s, tier attach "
                  f"{time.monotonic() - t_attach:.1f} s (first-touch wait {tiers.touch_wait_s:.1f} s)", flush=True)
        # which FlashInfer tactics the stage runs with (StageModel.tactics: source file|tuned_in_process, file, sha256)
        info = {"stage": stage, "tactics": getattr(model, "tactics", None),
                "cuda_graphs": getattr(model, "graph_info", None)}
        print(f"stage {stage}: {json.dumps(info)}", flush=True)
        (Path(cfg.run_dir) / f"stage{stage}.json").write_text(json.dumps(info) + "\n")
        clock = Clock(device)
        window = ProfileWindow(cfg.profile_mbs if cuda else None)
        ahead = RunAhead(cuda)
        hostprof = HostProfile(stage, cfg.run_dir)
        stamps = WorkerStamps(stage, cfg.run_dir, model)
        print(f"stage {stage} cpus: {cpu_list(cpus) if cpus else 'not pinned'}, threads {threads()}", flush=True)
        first_use = FirstUse(stage, cfg.run_dir)
        if cuda:
            first_use.install()
        # warm-up left ~1.15M objects on the heap: each automatic gen-2 collection walked them all, 0.2-0.4 s on the
        # host inside a forward (7895: 421 ms; bench 7897: 297/194 ms, 0.0 ms frozen). Frozen, collections only
        # walk what serving allocates.
        gc.collect()
        gc.freeze()
        reporter.q.put(Ready(stage, model.num_blocks, *slots))
        while True:
            while not plans.poll(1000):
                if os.getppid() != parent:
                    raise RuntimeError("controller is gone")
            msg = decode_to_worker(plans.recv())
            if isinstance(msg, Shutdown):
                ok = True
                if cuda:   # allocator retries (cache flush + cudaMalloc) stall the device; logged per run
                    m = torch.cuda.memory_stats(device)
                    print(f"stage {stage} allocator: " + json.dumps({k: m.get(k) for k in (
                        "num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")}
                        | {"peak_reserved_gb": round(m.get("reserved_bytes.all.peak", 0) / 1e9, 2),
                           "threads": threads(), "run_ahead_waits": ahead.waits,
                           "run_ahead_wait_s": round(ahead.wait_s, 1)}), flush=True)
                break
            plan: Plan = msg
            plan_ns = time.time_ns()
            mbs += 1
            window.before(plan.mb_id)
            if cuda:
                torch.cuda.nvtx.range_push(f"stage {stage} mb {plan.mb_id}")   # bench/nsys_trace.py attributes kernels
            clock.maybe_reanchor()
            first_use.mb = plan.mb_id
            hostprof.before(plan.mb_id)
            stamps.before(plan)
            fwd_ns = time.time_ns()
            # the host side (metadata, KDA plan, their copies) before the stream waits and the start mark: it runs
            # while the GPU still runs earlier microbatches, instead of inside this one's compute window (7895:
            # stage 0's compute exceeded stage 2's by the KDA plan's host time, 15-110 ms at 1.1 s/min)
            batch = model.prepare(plan)
            hidden_in = pipe.recv(plan.num_tokens, model.hidden_width) if stage > 0 else None
            if stage == 0:
                pipe.gate(mbs)                        # at most max_in_flight in flight (ppimpl/runtime/dispatch.py)
            eager = getattr(batch, "bucket", 0) is None   # StageModel: no CUDA graph for this size
            ahead.before(eager)                       # bounded run-ahead (RunAhead): never block inside a launch
            ready_ns = time.time_ns()                 # prepared, stream waits queued: the GPU may start from here
            start = clock.mark()
            out = model.run(batch, hidden_in)
            del batch
            stamps.after()
            hostprof.after(plan.mb_id)
            end = clock.mark()
            ahead.after(end, eager)
            if last:
                pipe.exited(mbs)
            host = (plan_ns, fwd_ns, ready_ns, time.time_ns())
            if hidden_in is not None:
                pipe.release()                        # CePipe: the sender may reuse the slot once this forward ran
            del hidden_in
            anchor = clock.anchor if cuda else None
            wait = end if cuda else None
            if not last:
                pipe.send(out, wait)
                reporter.q.put((stage, plan.mb_id, start, end, anchor, wait, None, None, host))
            else:
                tok_ids = [c.req_id for c in plan.chunks if c.is_last]
                reporter.q.put((stage, plan.mb_id, start, end, anchor, wait, out, tok_ids, host))
            if cuda:
                torch.cuda.nvtx.range_pop()
            window.after(plan.mb_id)
    except BaseException:
        reporter.q.put(WorkerError(stage, traceback.format_exc()))
        raise
    finally:
        if tiers is not None:
            tiers.close()
        reporter.q.put(None)
        reporter.join(timeout=10)
        if pipe is not None and ok:   # a failed stage exits without waiting on its peers
            pipe.close(mbs)
        plans.close(linger=0)
        ctx.term()
