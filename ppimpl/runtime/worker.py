"""One stage worker: plan -> (recv) -> forward -> (send) -> report (ppimpl/runtime/README.md, "Worker loop").

The main thread only enqueues GPU work. A reporter thread waits for each microbatch's end event and sends `StageDone`
with GPU times on the host clock, so the controller and all stages share one time base.
"""

import json
import os
import queue
import signal
import threading
import time
import traceback
from pathlib import Path

import torch
import zmq

from ppimpl.config import RunConfig
from ppimpl.runtime.transport import (Pipe, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_worker,
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
            stage, mb_id, start, end, anchor, wait, tokens, tok_ids = item
            if wait is not None:
                wait.synchronize()
            toks = None
            if tok_ids is not None:
                if copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(wait)
                        tokens = tokens.cpu()
                toks = list(zip(tok_ids, tokens.tolist()))
            sock.send(encode(StageDone(stage, mb_id, Clock.to_ns(start, anchor), Clock.to_ns(end, anchor), toks)))
        sock.close()


def build_model(cfg: RunConfig, stage: int, device: torch.device):
    if cfg.model == "fake":
        from ppimpl.model.fake import FakeStageModel
        return FakeStageModel(cfg, stage, device)
    from ppimpl.model.stage import StageModel
    return StageModel(cfg, stage, device)


@torch.inference_mode()   # all of it: model construction, warm-up, receive, forward, send (StageModel's warm-up
def run_worker(cfg: RunConfig, stage: int):   # makes inference tensors, which autograd-tracked code may not use)
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the controller stops us (Shutdown), or we see it is gone
    parent = os.getppid()
    n = cfg.num_stages
    last = stage == n - 1
    cuda = cfg.device == "cuda"
    device = torch.device("cuda", stage) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    ctx = zmq.Context()
    plans = worker_sockets(ctx, cfg.plan_addr(stage))
    reporter = Reporter(ctx, cfg.report_addr(), device)
    reporter.start()
    pipe, ok = None, False
    # Diagnosis only: the host waits for each microbatch's GPU work before taking the next plan, so nothing built on
    # the host for microbatch k+1 can overwrite what microbatch k's queued copies still read.
    sync_each_mb = cuda and os.environ.get("PPIMPL_SYNC_EACH_MB") == "1"
    try:
        pipe = Pipe(stage, n, device, cfg.dist_init_method())
        model = build_model(cfg, stage, device)
        # which FlashInfer tactics the stage runs with (StageModel.tactics: source file|tuned_in_process, file, sha256)
        info = {"stage": stage, "tactics": getattr(model, "tactics", None)}
        print(f"stage {stage}: {json.dumps(info)}", flush=True)
        (Path(cfg.run_dir) / f"stage{stage}.json").write_text(json.dumps(info) + "\n")
        clock = Clock(device)
        window = ProfileWindow(cfg.profile_mbs if cuda else None)
        reporter.q.put(Ready(stage, model.num_blocks))
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
                        | {"peak_reserved_gb": round(m.get("reserved_bytes.all.peak", 0) / 1e9, 2)}), flush=True)
                break
            plan: Plan = msg
            window.before(plan.mb_id)
            if cuda:
                torch.cuda.nvtx.range_push(f"stage {stage} mb {plan.mb_id}")   # bench/nsys_trace.py attributes kernels
            clock.maybe_reanchor()
            hidden_in = pipe.recv(plan.num_tokens, model.hidden_width) if stage > 0 else None
            start = clock.mark()
            out = model.forward(plan, hidden_in)
            end = clock.mark()
            del hidden_in
            if sync_each_mb:
                torch.cuda.current_stream(device).synchronize()
            anchor = clock.anchor if cuda else None
            wait = end if cuda else None
            if not last:
                pipe.send(out, wait)
                reporter.q.put((stage, plan.mb_id, start, end, anchor, wait, None, None))
            else:
                tok_ids = [c.req_id for c in plan.chunks if c.is_last]
                reporter.q.put((stage, plan.mb_id, start, end, anchor, wait, out, tok_ids))
            if cuda:
                torch.cuda.nvtx.range_pop()
            window.after(plan.mb_id)
    except BaseException:
        reporter.q.put(WorkerError(stage, traceback.format_exc()))
        raise
    finally:
        reporter.q.put(None)
        reporter.join(timeout=10)
        if pipe is not None and ok:   # a failed stage exits without waiting on its peers
            pipe.close()
        plans.close(linger=0)
        ctx.term()
