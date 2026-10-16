"""One stage on one GPU: build with warm-up, then capture the CUDA graphs of the buckets given (or run one eager
forward of that many tokens). Each call is its own process, so a CUDA error names its bucket.

  python -m bench.graph_probe --stage 0 --buckets 7168            # warm-up + capture of 7168 only
  CUDA_LAUNCH_BLOCKING=1 python -m bench.graph_probe --stage 0 --eager 7168
  python -m bench.graph_probe --stage 0 --full-pool --together --buckets 8192,7168   # as the server captures

Prints "[graph_probe] ... ok" or raises.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace as NS


def main():
    import torch

    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--buckets", default="", help="comma-separated token counts to capture")
    ap.add_argument("--eager", type=int, default=0, help="instead: one eager forward of this many tokens")
    ap.add_argument("--together", action="store_true", help="capture all --buckets in one call (as the server does)")
    ap.add_argument("--full-pool", action="store_true", help="the server's KV pool (all free memory), not 64 blocks")
    ap.add_argument("--loop", action="store_true",
                    help="capture --buckets with a copy of StageModel._capture_graphs that prints after each bucket")
    ap.add_argument("--pool-per-bucket", action="store_true", help="with --loop: a fresh graph pool per bucket")
    ap.add_argument("--keep-eager-args", action="store_true",
                    help="with --loop: eager breaks hold strong references to their arguments (no weak refs)")
    ap.add_argument("--trace-ops", action="store_true",
                    help="with --trace-eager: also sync after every torch op and Triton launch inside the eager breaks "
                         "and print the last ones completed before a fault")
    ap.add_argument("--trace-eager", action="store_true",
                    help="with --loop: during capture, sync after each eager break and print its op and its tensor "
                         "arguments (int: min/max, float: non-finite count), to name the one that faults")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    t0 = time.perf_counter()
    pool = {} if a.full_pool else {"num_blocks": 64, "max_model_len": 8192 + 8576}
    model = StageModel(bench_config(8192, cuda_graphs=False), a.stage, dev, **pool)
    torch.cuda.synchronize(dev)
    print(f"[graph_probe] stage {a.stage}: built and warmed up in {time.perf_counter() - t0:.0f} s", flush=True)
    if a.eager:
        T = a.eager
        plan = NS(mb_id=0, chunks=[NS(req_id="e", start=0, num_tokens=T, kv_blocks=[1], is_last=True,
                                      state_blocks=list(range(2, 8)), token_ids=[1] * T)])
        hidden = None if model.first else torch.zeros(T, model.hidden_width, dtype=torch.bfloat16, device=dev)
        with torch.inference_mode():
            model.forward(plan, hidden)
        torch.cuda.synchronize(dev)
        print(f"[graph_probe] stage {a.stage}: eager {T} ok", flush=True)
    buckets = [int(x) for x in a.buckets.split(",") if x]
    if a.loop:   # on a side stream, as StageModel.capture_graphs (the default stream cannot capture)
        caller, side = torch.cuda.current_stream(dev), torch.cuda.Stream(dev)
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            capture_loop(model, sorted(buckets, reverse=True), a)
        caller.wait_stream(side)
        torch.cuda.synchronize(dev)
        return
    for b in [tuple(buckets)] if a.together else [(b,) for b in buckets]:
        model.capture_graphs(buckets=b)
        torch.cuda.synchronize(dev)
        print(f"[graph_probe] stage {a.stage}: capture {b} ok: {model.graph_info}", flush=True)


class SyncEachOp:
    """Inside: every torch op (TorchDispatchMode) and Triton launch (launch_exit_hook) is followed by a device sync;
    `done` lists those that completed, in order."""

    def __enter__(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from triton import knobs

        done = self.done = []

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                torch.cuda.synchronize()
                done.append(str(func))
                return out

        def exit_hook(meta=None, *args, **kwargs):     # meta: triton's LazyDict of the launch
            torch.cuda.synchronize()
            done.append("triton " + str(getattr(meta, "data", {}).get("name", "?")))

        self.mode = Mode()
        self.old_hook = knobs.runtime.launch_exit_hook
        knobs.runtime.launch_exit_hook = exit_hook
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        from triton import knobs

        self.mode.__exit__(*exc)
        knobs.runtime.launch_exit_hook = self.old_hook
        return False


def capture_loop(model, buckets, a):
    """StageModel._capture_graphs, step by step: eager pass, sync, capture, sync, print, per bucket."""
    import torch
    from vllm.compilation import breakable_cudagraph as bc

    if a.keep_eager_args:
        bc._weak_ref_capture_arg = lambda arg: arg
    if a.trace_eager:
        add_eager = bc.BreakableCUDAGraphCapture.add_eager

        def traced(self, fn):
            cells = {k: c.cell_contents for k, c in zip(fn.__code__.co_freevars, fn.__closure__ or ())}
            op = getattr(cells.get("fn"), "__qualname__", "?")
            args = list(cells.get("weak_args", ())) + list(cells.get("weak_kwargs", {}).values())

            def desc(t):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    return type(t).__name__
                if t.dtype in (torch.int32, torch.int64, torch.int16):
                    return f"{t.dtype} {tuple(t.shape)} [{int(t.min())}, {int(t.max())}]"
                if t.is_floating_point() and t.element_size() >= 2:
                    return f"{t.dtype} {tuple(t.shape)} nonfinite {int((~torch.isfinite(t)).sum())}"
                return f"{t.dtype} {tuple(t.shape)}"

            self._end_segment()             # add_eager's own end is then a no-op; no sync while capturing
            torch.cuda.synchronize()
            print(f"[graph_probe]   eager {self.num_eager_breaks} {op} in: " + "; ".join(desc(t) for t in args),
                  flush=True)

            def run():
                if not a.trace_ops:
                    out = fn()
                    torch.cuda.synchronize()
                    return out
                with SyncEachOp() as ops:
                    try:
                        out = fn()
                        torch.cuda.synchronize()
                    except Exception:
                        print("[graph_probe]   last ops before the fault: " + " | ".join(ops.done[-8:]), flush=True)
                        raise
                return out
            out = add_eager(self, run)
            self.segments[-1] = fn          # replay the original
            print(f"[graph_probe]   eager {self.num_eager_breaks - 1} {op} ok", flush=True)
            return out
        bc.BreakableCUDAGraphCapture.add_eager = traced
    pool = torch.cuda.graph_pool_handle()
    hidden = None if model.first else torch.zeros(buckets[0], model.hidden_width, dtype=torch.bfloat16,
                                                  device=model.device)
    from ppimpl.model.stage import GraphEntry

    with torch.inference_mode():
        for Tb in buckets:
            if a.pool_per_bucket:
                pool = torch.cuda.graph_pool_handle()
            plan = NS(mb_id=-10, chunks=[NS(req_id="g", start=0, num_tokens=Tb, kv_blocks=[1], is_last=True,
                                            state_blocks=list(range(2, 8)), token_ids=[1] * Tb)])
            b = model.meta.build(plan, with_tokens=model.first, bucket=Tb)
            e = GraphEntry(Tb, torch.empty_like(b.graph_rows), None if hidden is None else hidden[:Tb])
            e.rows.copy_(b.graph_rows)
            e.slots = {name: e.rows[1] for name in b.slots}
            with model._graph_context(b, e):
                model._graph_body(e)
            torch.cuda.synchronize(model.device)
            print(f"[graph_probe] {Tb}: eager ok", flush=True)
            e.capture = bc.BreakableCUDAGraphCapture(pool=pool)
            with model._graph_context(b, e), e.capture:
                e.out = model._graph_body(e)
            torch.cuda.synchronize(model.device)
            print(f"[graph_probe] {Tb}: capture ok", flush=True)
            model.graphs[Tb] = e
    print(f"[graph_probe] loop {buckets} ok", flush=True)


if __name__ == "__main__":
    main()
