"""Low-overhead host timings of a stage forward, for diagnosis (bench.host_front --stamps; the worker with
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
            self.gc_ns += time.perf_counter_ns() - self._gc_t

    def begin(self):
        self.total, self.calls, self.segs, self.stack, self.gc_ns = {}, {}, [], {}, 0
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
        return {"host_ms": round(host / 1e6, 4), "gc_ms": round(self.gc_ns / 1e6, 4),
                "seg": [[i, k, round(t / 1e6, 4)] for i, k, t in self.segs],
                "fn": {q: [round(v / 1e6, 4), self.calls[q]] for q, v in sorted(self.total.items(), key=lambda x: -x[1])}}


class WorkerStamps:
    """The worker side: PPIMPL_HOST_STAMPS="a:b[@stages]" (stages default 0)."""

    def __init__(self, stage: int, run_dir: str, model):
        spec = os.environ.get("PPIMPL_HOST_STAMPS", "")
        rng, _, stages = spec.partition("@")
        self.on = bool(rng) and stage in [int(x) for x in (stages or "0").split(",")]
        self.stage, self.cur = stage, None
        if self.on:
            self.a, self.b = (int(x) for x in rng.split(":"))
            self.st = HostStamps(model)
            self.f = open(Path(run_dir) / f"host_stamps_stage{stage}.jsonl", "w")
            if self.st.missing:
                print(f"stage {stage} host stamps: not found: {self.st.missing}", flush=True)

    def before(self, plan):
        if self.on and self.a <= plan.mb_id < self.b:
            self.cur = (plan.mb_id, [(c.start, c.num_tokens) for c in plan.chunks])
            self.st.begin()

    def after(self):
        if self.cur is not None:
            rec = self.st.end()
            mb, chunks = self.cur
            self.f.write(json.dumps({"stage": self.stage, "index": mb, "chunks": chunks} | rec) + "\n")
            self.f.flush()
            self.cur = None
