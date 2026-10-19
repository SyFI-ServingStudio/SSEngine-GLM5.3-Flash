"""Low-overhead host timings of a stage forward, for diagnosis (research.bench.host_front --stamps; the worker with
PPIMPL_HOST_STAMPS="a:b[@stages]", default stage 0, writes <run dir>/host_stamps.jsonl for microbatches [a, b)).

Per forward: the inclusive host time of chosen functions (`sys.monitoring` PY_START/PY_RETURN on their code objects
only: the rest of the process runs unmonitored), when each CUDA graph segment and eager break of the replayed graph
started after the forward was called, and the time spent in cyclic GC. Unlike cProfile this keeps the forward's
host time within a few percent of an unmonitored one, so the launch-bound gaps it explains are the real ones.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

# module:qualname of the host functions timed (inclusive)
FUNCTIONS = [
    "ppimpl.model.stage:StageModel.prepare", "ppimpl.model.stage:StageModel._run_graph",
    "ppimpl.model.metadata:StageMetadata.build", "ppimpl.model.metadata:Pack.copy",
    "ppimpl.model.metadata:StageMetadata._kda_plan", "ppimpl.model.pp_patch:kda_prefill_forward",
    "ppimpl.model.pp_patch:kpool_compress_insert", "ppimpl.model.pp_patch:kpool_seed_tail_cache",
    "ppimpl.model.pp_patch:_routed_top_k_per_row_prefill", "ppimpl.model.topk_canon:canonicalize",
    "ppimpl.model.pp_patch:build_and_route",
    "vllm.forward_context:set_forward_context",
    "vllm.models.glm5next.nvidia.sparse_indexer:sparse_attn_indexer_kpool",
    "vllm.model_executor.layers.attention.mla_attention:MLAAttention.forward_impl",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse:FlashInferMLASparseImpl.forward_mqa",
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse:FlashInferMLASparseImpl._run_mqa_kernel",
    "vllm.v1.attention.backends.mla.indexer:build_prefill_chunk_metadata",
    "flashinfer.mla._core:trtllm_batch_decode_with_kv_cache_mla",
    "flashinfer.kda_kernels.persistent.forward:fwd", "flashinfer.kda_kernels.persistent.forward:_launch_forward",
    "vllm.utils.deep_gemm:fp8_fp4_mqa_logits", "triton.runtime.jit:JITFunction.run",
    # compiles in a forward (in_run_jit sees only vLLM's registered JIT kernels)
    "triton.compiler.compiler:compile", "torch._dynamo.convert_frame:_compile", "torch._inductor.compile_fx:compile_fx",
]


class HostStamps:
    def __init__(self, model=None, names=FUNCTIONS):
        self.mon = sys.monitoring
        self.tool = self.mon.PROFILER_ID
        self.codes: dict = {}
        self.missing: list[str] = []
        for name in names:
            if callable(name):                  # a function itself (tests)
                self.codes[inspect.unwrap(name).__code__] = name.__name__
                continue
            mod, _, qual = name.partition(":")
            try:
                obj = importlib.import_module(mod)
                for part in qual.split("."):
                    obj = getattr(obj, part)
                self.codes[inspect.unwrap(obj).__code__] = qual
            except Exception as e:  # noqa: BLE001 (a name this vLLM/FlashInfer lacks is reported, not fatal)
                self.missing.append(f"{name}: {type(e).__name__}")
        meta = getattr(model, "meta", None)
        for _, ag in getattr(meta, "groups", ()):
            cls = type(ag.get_metadata_builder())
            self.codes[inspect.unwrap(cls.build).__code__] = f"{cls.__name__}.build"
        if self.mon.get_tool(self.tool) == "ppimpl-host-stamps":   # a previous instance: replace it
            self.mon.set_events(self.tool, 0)
            self.mon.free_tool_id(self.tool)
        self.mon.use_tool_id(self.tool, "ppimpl-host-stamps")
        ev = self.mon.events
        self.mon.register_callback(self.tool, ev.PY_START, self._start)
        self.mon.register_callback(self.tool, ev.PY_RETURN, self._end)
        self.stack: dict = {}
        self.total: dict = {}
        self.calls: dict = {}
        self.segs: list = []
        self.gc_ns = 0
        self.gc_long: list = []
        self._gc_t = 0
        self.t0 = 0
        self.active = False
        for e in getattr(model, "graphs", {}).values():    # stamp each segment's start (only while active)
            e.capture.segments[:] = [self._wrap(i, f) for i, f in enumerate(e.capture.segments)]

    def _wrap(self, i, f):
        kind = "g" if getattr(f, "__name__", "") == "replay" else "e"

        def seg():
            if self.active:
                self.segs.append((i, kind, time.perf_counter_ns() - self.t0))
            return f()
        return seg

    def _start(self, code, offset):
        if code in self.codes:
            self.stack.setdefault(code, []).append(time.perf_counter_ns())

    def _end(self, code, offset, retval):
        st = self.stack.get(code)
        if st:
            t = time.perf_counter_ns() - st.pop()
            if not st:
                q = self.codes[code]
                self.total[q] = self.total.get(q, 0) + t
                self.calls[q] = self.calls.get(q, 0) + 1

    def _gc(self, phase, info):
        if phase == "start":
            self._gc_t = time.perf_counter_ns()
        else:
            t = time.perf_counter_ns() - self._gc_t
            self.gc_ns += t
            if t > 5_000_000:                                    # a pause worth naming: generation, collected
                self.gc_long.append([info.get("generation"), round(t / 1e6, 2), info.get("collected")])

    def begin(self):
        self.total, self.calls, self.segs, self.stack, self.gc_ns, self.gc_long = {}, {}, [], {}, 0, []
        ev = self.mon.events
        for code in self.codes:
            self.mon.set_local_events(self.tool, code, ev.PY_START | ev.PY_RETURN)
        gc.callbacks.append(self._gc)
        self.active = True
        self.t0 = time.perf_counter_ns()

    def end(self) -> dict:
        host = time.perf_counter_ns() - self.t0
        self.active = False
        gc.callbacks.remove(self._gc)
        for code in self.codes:
            self.mon.set_local_events(self.tool, code, 0)
        return {"host_ms": round(host / 1e6, 4), "gc_ms": round(self.gc_ns / 1e6, 4), "gc_long": self.gc_long,
                "seg": [[i, k, round(t / 1e6, 4)] for i, k, t in self.segs],
                "fn": {q: [round(v / 1e6, 4), self.calls[q]] for q, v in sorted(self.total.items(), key=lambda x: -x[1])}}


COMPILES = ("compile", "_compile", "compile_fx")


class WorkerStamps:
    """The worker side: PPIMPL_HOST_STAMPS="a:b[@stages]" (stages default 0) stamps microbatches [a, b);
    "slow:<ms>[@stages]" stamps every forward and keeps those whose host time passed <ms>, or with a compile or a GC
    pause over 5 ms, plus the CUDA caching allocator's device allocations during it (a whole serving run: why a
    forward's host time sometimes reaches 0.5 s, 7878 mb 6280)."""

    def __init__(self, stage: int, run_dir: str, model):
        spec = os.environ.get("PPIMPL_HOST_STAMPS", "")
        rng, _, stages = spec.partition("@")
        self.on = bool(rng) and stage in [int(x) for x in (stages or "0").split(",")]
        self.stage, self.cur = stage, None
        self.slow_ms = None
        self.device = getattr(model, "device", None)
        if self.on and rng.startswith("slow:"):
            self.slow_ms = float(rng.split(":")[1])
            self.a, self.b = 0, 1 << 62
        elif self.on:
            self.a, self.b = (int(x) for x in rng.split(":"))
        if self.on:
            self.st = HostStamps(model)
            self.f = open(Path(run_dir) / f"host_stamps_stage{stage}.jsonl", "w")
            if self.st.missing:
                print(f"stage {stage} host stamps: not found: {self.st.missing}", flush=True)

    def _allocs(self) -> int:
        if self.device is None or getattr(self.device, "type", "") != "cuda":
            return 0
        import torch
        return torch.cuda.memory_stats(self.device).get("num_device_alloc", 0)

    def before(self, plan):
        if self.on and self.a <= plan.mb_id < self.b:
            self.cur = (plan.mb_id, [(c.start, c.num_tokens) for c in plan.chunks],
                        self._allocs() if self.slow_ms is not None else 0)
            self.st.begin()

    def after(self):
        if self.cur is not None:
            rec = self.st.end()
            mb, chunks, allocs = self.cur
            self.cur = None
            if self.slow_ms is not None:
                rec["device_allocs"] = self._allocs() - allocs
                if not (rec["host_ms"] > self.slow_ms or rec["gc_long"] or rec["device_allocs"]
                        or any(q in rec["fn"] for q in COMPILES)):
                    return
            self.f.write(json.dumps({"stage": self.stage, "index": mb, "chunks": chunks} | rec) + "\n")
            self.f.flush()
