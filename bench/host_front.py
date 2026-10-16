"""Stage forward host cost vs GPU time for one request continuing a cached prefix (a round >= 1 microbatch).

  python -m bench.host_front [--stage 0] [--prefix 2048] [--append 256] [--reps 30] [--out x.json] [--profile]

Two timings of the same forward (CUDA graphs on, as the server runs):
- `direct`: the forward enqueued on an idle GPU, as stage 0 runs it: compute = start event (host t = 0) -> end
  event. The GPU waits for the host wherever the host is slower than the kernels it launches.
- `ahead`: the same forward enqueued behind a GPU sleep, so the host is ahead of the GPU, as stages 1-3 run it
  (they enqueue before the upstream hidden state arrives): compute = the GPU time alone.
direct - ahead is what the host costs the stage on the GPU timeline. With --idle-ms X also `direct_sleep` (the
thread sleeps X ms before each forward, as a worker blocks on its plan socket between microbatches) and
`direct_spin` (it busy-waits X ms instead): what an idle, cold host thread adds. Also the host time of `forward` (perf
counter), and with --profile one cProfile of a direct forward (top functions by own and by cumulative time).
With --stamps, low-overhead host timings of direct forwards (sys.monitoring on the functions in STAMPS only, plus a
stamp at the start of every graph segment and eager break): when each segment starts after the forward's start,
and each function's inclusive time per forward.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS

# module:qualname of the host functions --stamps times (inclusive, per forward)
STAMPS = [
    "ppimpl.model.stage:StageModel.prepare", "ppimpl.model.stage:StageModel._run_graph",
    "ppimpl.model.metadata:StageMetadata.build", "ppimpl.model.metadata:Pack.copy",
    "ppimpl.model.metadata:StageMetadata._kda_plan", "ppimpl.model.pp_patch:kda_prefill_forward",
    "ppimpl.model.pp_patch:kpool_compress_insert", "ppimpl.model.pp_patch:kpool_seed_tail_cache",
    "ppimpl.model.pp_patch:_routed_top_k_per_row_prefill", "ppimpl.model.topk_canon:canonicalize",
    "ppimpl.model.pp_patch:build_and_route",
    "vllm.models.glm5next.nvidia.sparse_indexer:sparse_attn_indexer_kpool",
    "vllm.model_executor.layers.attention.mla_attention:MLAAttention.forward_impl",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse:FlashInferMLASparseImpl.forward_mqa",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse:FlashInferMLASparseImpl._run_mqa_kernel",
    "flashinfer.mla._core:trtllm_batch_decode_with_kv_cache_mla",
    "flashinfer.kda_kernels.persistent.forward:fwd", "flashinfer.kda_kernels.persistent.forward:_launch_forward",
    "vllm.utils.deep_gemm:fp8_fp4_mqa_logits", "triton.runtime.jit:JITFunction.run",
]


class Stamps:
    """Inclusive host time of chosen functions (sys.monitoring PY_START/PY_RETURN on their code objects only)."""

    def __init__(self, names, extra=()):
        import importlib
        import inspect
        import sys

        self.mon = sys.monitoring
        self.tool = self.mon.PROFILER_ID
        self.codes, self.total, self.calls, self.stack = {}, {}, {}, {}
        for name in names:
            mod, _, qual = name.partition(":")
            try:
                obj = importlib.import_module(mod)
                for part in qual.split("."):
                    obj = getattr(obj, part)
                code = inspect.unwrap(obj).__code__
            except Exception as e:  # noqa: BLE001 (a missing name is reported, not fatal)
                print(f"[host_front] stamps: {name}: {type(e).__name__} {e}", flush=True)
                continue
            self.codes[code] = qual
        for name, fn in extra:
            self.codes[fn.__code__] = name
        self.mon.use_tool_id(self.tool, "host_front")
        ev = self.mon.events
        self.mon.register_callback(self.tool, ev.PY_START, self._start)
        self.mon.register_callback(self.tool, ev.PY_RETURN, self._end)
        self.mon.register_callback(self.tool, ev.PY_UNWIND, self._end)

    def on(self):
        ev = self.mon.events
        for code in self.codes:
            self.mon.set_local_events(self.tool, code, ev.PY_START | ev.PY_RETURN)

    def off(self):
        for code in self.codes:
            self.mon.set_local_events(self.tool, code, 0)

    def _start(self, code, offset):
        if code in self.codes:
            self.stack.setdefault(code, []).append(time.perf_counter_ns())

    def _end(self, code, offset, *rest):
        st = self.stack.get(code)
        if st:
            t = time.perf_counter_ns() - st.pop()
            if not st:                       # outermost call only (recursion)
                q = self.codes[code]
                self.total[q] = self.total.get(q, 0) + t
                self.calls[q] = self.calls.get(q, 0) + 1


def main():
    import torch

    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--append", type=int, default=256)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--idle-ms", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    model = StageModel(bench_config(8192, cuda_graphs=True), a.stage, dev, num_blocks=64, max_model_len=8192 + 8576)
    B = model.kv.block_tokens
    n_kv = -(-(a.prefix + a.append) // B)
    assert n_kv + 7 <= model.num_blocks, (n_kv, model.num_blocks, B)
    plan = NS(mb_id=0, chunks=[NS(req_id="r", start=a.prefix, num_tokens=a.append, kv_blocks=list(range(1, 1 + n_kv)),
                                  is_last=True, state_blocks=list(range(1 + n_kv, 7 + n_kv)),
                                  token_ids=[random.Random(a.append).randrange(1, 150_000) for _ in range(a.append)])])
    # (varied ids: one repeated id routes every token to the same experts, and the MoE then costs far less than served)
    hidden = None if model.first else torch.zeros(a.append, model.hidden_width, dtype=torch.bfloat16, device=dev)
    res = {"stage": a.stage, "prefix": a.prefix, "append": a.append, "block_tokens": B}

    def one(ahead: bool, idle: str = ""):
        torch.cuda.synchronize(dev)
        if idle == "sleep":
            time.sleep(a.idle_ms / 1e3)
        elif idle == "spin":
            until = time.perf_counter() + a.idle_ms / 1e3
            while time.perf_counter() < until:
                pass
        if ahead:
            torch.cuda._sleep(200_000_000)          # ~100 ms of GPU spin: the host enqueues the forward meanwhile
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        with torch.inference_mode():
            b = model.prepare(plan)
            prep.append(1e3 * (time.perf_counter() - t0))
            model.run(b, hidden)
        e1.record()
        host = 1e3 * (time.perf_counter() - t0)
        e1.synchronize()
        return host, e0.elapsed_time(e1)

    prep = []

    for _ in range(5):
        one(False)
    modes = ("direct", "ahead") + (("direct_sleep", "direct_spin") if a.idle_ms else ())
    for mode in modes:
        h, g = zip(*(one(mode == "ahead", mode.partition("_")[2]) for _ in range(a.reps)))
        res[mode] = {"host_ms": round(statistics.median(h), 3), "gpu_ms": round(statistics.median(g), 3),
                     "prepare_ms": round(statistics.median(prep[-a.reps:]), 3)}
        print(f"[host_front] stage {a.stage} P {a.prefix} A {a.append} {mode}: forward host "
              f"{res[mode]['host_ms']:.3f} ms (prepare {res[mode]['prepare_ms']:.3f}), compute "
              f"{res[mode]['gpu_ms']:.3f} ms", flush=True)
    res["host_cost_on_gpu_ms"] = round(res["direct"]["gpu_ms"] - res["ahead"]["gpu_ms"], 3)
    print(f"[host_front] direct - ahead = {res['host_cost_on_gpu_ms']:.3f} ms", flush=True)
    for mode in modes[2:]:
        print(f"[host_front] {mode} - ahead = {res[mode]['gpu_ms'] - res['ahead']['gpu_ms']:.3f} ms", flush=True)
    if a.profile:
        torch.cuda.synchronize(dev)
        prof = cProfile.Profile()
        prof.enable()
        with torch.inference_mode():
            model.forward(plan, hidden)
        prof.disable()
        torch.cuda.synchronize(dev)
        st = pstats.Stats(prof).stats          # {(file, line, fn): (cc, nc, tottime, cumtime, callers)}
        for key, col in (("own", 2), ("cumulative", 3)):
            rows = sorted(st.items(), key=lambda kv: -kv[1][col])[:40]
            lines = [f"{v[col] * 1e6:9.0f} us {v[1]:5d}x  {k[2]} ({Path(k[0]).name}:{k[1]})" for k, v in rows]
            print(f"[host_front] profile by {key} time (us, calls):\n" + "\n".join(lines), flush=True)
    if a.stamps:
        res["stamps"] = stamps(model, plan, hidden, dev, a.reps)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def stamps(model, plan, hidden, dev, reps: int) -> dict:
    """--stamps: segment start times and function times over `reps` direct forwards."""
    import torch

    T = sum(c.num_tokens for c in plan.chunks)
    e = model.graphs[min(b for b in model.graphs if b >= T)] if model.graphs else None
    seg_t: list[list[float]] = []
    t0 = [0]
    if e is not None:
        orig = list(e.capture.segments)

        def wrap(i, f):
            def seg():
                seg_t[-1].append((i, time.perf_counter_ns() - t0[0]))
                return f()
            return seg
        e.capture.segments[:] = [wrap(i, f) for i, f in enumerate(orig)]
    builders = {type(ag.get_metadata_builder()) for _, ag in model.meta.groups}
    st = Stamps(STAMPS, [(f"{c.__name__}.build", c.build) for c in builders])
    import gc
    gct: dict = {}
    gc_t0 = [0]

    def gc_cb(phase, info):
        if phase == "start":
            gc_t0[0] = time.perf_counter_ns()
        else:
            g = f"gc gen{info['generation']}"
            gct[g] = gct.get(g, 0) + time.perf_counter_ns() - gc_t0[0]
            gct[g + " n"] = gct.get(g + " n", 0) + 1
    host = []
    for r in range(reps + 3):
        torch.cuda.synchronize(dev)
        seg_t.append([])
        if r >= 3:
            st.on()
            if r == 3:
                gc.callbacks.append(gc_cb)
        t0[0] = time.perf_counter_ns()
        with torch.inference_mode():
            model.forward(plan, hidden)
        host.append((time.perf_counter_ns() - t0[0]) / 1e6)
        st.off()
        torch.cuda.synchronize(dev)
    gc.callbacks.remove(gc_cb)
    if e is not None:
        e.capture.segments[:] = orig
    kinds = ["graph" if getattr(f, "__name__", "") == "replay" else "eager" for f in (orig if e else [])]
    starts = {}
    for run in seg_t[3:]:
        for i, t in run:
            starts.setdefault(i, []).append(t / 1e6)
    out = {"host_ms": round(statistics.median(host[3:]), 3),
           "segments": [{"i": i, "kind": kinds[i] if i < len(kinds) else "?",
                         "start_ms": round(statistics.median(v), 3)} for i, v in sorted(starts.items())],
           "functions": {q: {"ms_per_forward": round(st.total[q] / 1e6 / reps, 3),
                             "calls_per_forward": st.calls[q] / reps} for q in sorted(st.total, key=lambda q: -st.total[q])},
           "gc": {g: (round(v / 1e6 / reps, 4) if not g.endswith(" n") else v / reps) for g, v in sorted(gct.items())}}
    print(f"[host_front] stamps: forward host {out['host_ms']:.3f} ms (monitored); segment starts (ms):", flush=True)
    print("  " + "  ".join(f"{d['i']}{d['kind'][0]}@{d['start_ms']:.3f}" for d in out["segments"]), flush=True)
    for q, v in out["functions"].items():
        print(f"  {v['ms_per_forward']:7.3f} ms  {v['calls_per_forward']:5.1f}x  {q}", flush=True)
    print(f"  gc per forward (ms, collections): {out['gc']}", flush=True)
    return out


if __name__ == "__main__":
    main()
