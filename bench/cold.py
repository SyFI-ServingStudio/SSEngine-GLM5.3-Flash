"""First-forward ("cold") cost after the warm-up: what a request shape hits for the first time.

    python bench/cold.py [--lengths 1024,2048,4096,8192] [--budget 131072]

Builds stage 0 with the output head like a real stage (budget = the run config's microbatch budget, so the warm-up
is the runtime's), then per length: the first forward and the second, each timed per unit with a device sync
around it (wall ms), plus what happened inside the first one only: Triton compiles and autotunes (JIT cache misses,
by kernel name), FlashInfer autotuner lookups that missed (op, shapes). A unit whose first call is much slower than
its second names the cold path.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as lb  # noqa: E402

EVENTS: list = []
UNIT_MS: dict = collections.defaultdict(float)
ON = {"v": False}


def instrument(model) -> None:
    import torch
    import triton.runtime.autotuner as ta
    import triton.runtime.jit as tj
    from flashinfer.autotuner import AutoTuner

    def unit(obj, attr, name):
        orig = getattr(obj, attr)

        def call(*a, **kw):
            if not ON["v"]:
                return orig(*a, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            UNIT_MS[name] += (time.perf_counter() - t) * 1e3
            return out

        setattr(obj, attr, call)

    seen = set()
    for layer in model.layers:
        unit(layer, "forward", f"L{layer.layer_idx}")
        kind = (layer.layer_kind, type(getattr(layer, "mlp", None)).__name__)
        if kind in seen:
            continue
        seen.add(kind)       # the first layer of each kind: every submodule, timed the same way
        for name, mod in layer.named_modules():
            if name and name.count(".") <= 2:
                unit(mod, "forward", f"L{layer.layer_idx}.{name}")
        if layer.layer_kind == "kda":
            unit(layer.self_attn, "_forward", f"L{layer.layer_idx}.self_attn._forward")
    import causal_conv1d_cuda
    import flashinfer.kda_kernels.persistent.forward as kda_fwd

    unit(kda_fwd, "fwd", "kda.flashinfer_fwd")
    unit(causal_conv1d_cuda, "causal_conv1d_fwd", "kda.conv")
    if model.embed is not None:
        unit(model.embed, "forward", "embed")
    unit(model, "output_head", "output_head")
    unit(model, "prepare", "prepare")

    # Triton: a compile is a cache miss in JITFunction (the `_do_compile` path); an autotune is Autotuner._bench
    if hasattr(tj.JITFunction, "_do_compile"):
        orig_c = tj.JITFunction._do_compile

        def _do_compile(self, *a, **kw):
            t = time.perf_counter()
            out = orig_c(self, *a, **kw)
            if ON["v"]:
                EVENTS.append(("triton_compile", getattr(self, "__name__", str(self))[:80],
                               round((time.perf_counter() - t) * 1e3, 1)))
            return out

        tj.JITFunction._do_compile = _do_compile
    orig_b = ta.Autotuner._bench

    def _bench(self, *a, **kw):
        if ON["v"]:
            EVENTS.append(("triton_autotune", getattr(self.fn, "__name__", str(self.fn))[:80], None))
        return orig_b(self, *a, **kw)

    ta.Autotuner._bench = _bench
    orig_s = AutoTuner.search_cache

    def search_cache(self, custom_op, runners, input_shapes, tuning_config, inputs=None, **kw):
        out = orig_s(self, custom_op, runners, input_shapes, tuning_config, inputs=inputs, **kw)
        if ON["v"] and not out[0]:
            EVENTS.append(("flashinfer_miss", custom_op, str(input_shapes)[:160]))
        return out

    AutoTuner.search_cache = search_cache


def dynamo_counters() -> dict:
    """torch.compile activity (frames compiled, recompiles, graph breaks) as flat counters."""
    from torch._dynamo.utils import counters

    return {f"{g}.{k}": v for g, d in counters.items() for k, v in d.items() if isinstance(v, int)}


def main() -> None:
    import torch
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="1024,1000,1100,2048,3000,4096,6000,8192")
    ap.add_argument("--budget", type=int, default=131072)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    dev = torch.device("cuda", 0)
    model = StageModel(bench_config(a.budget), 0, dev, warmup=False, last=True, num_blocks=lb.NUM_BLOCKS,
                       max_model_len=a.budget + 8576)
    t0 = time.time()
    model.warmup()
    print(f"[cold] warm-up {time.time() - t0:.0f} s, tactics {model.tactics}", flush=True)
    instrument(model)
    tokens = lb.enwik9_tokens(CHECKPOINT, max(lengths))
    res = {}
    with torch.inference_mode():
        for L in lengths:
            plan = lb.one_request_plan(tokens[:L])
            row = {}
            for k in ("first", "second"):
                UNIT_MS.clear()
                EVENTS.clear()
                dyn0 = dynamo_counters()
                ON["v"] = True
                torch.cuda.synchronize()
                t = time.perf_counter()
                model.forward(plan, None)
                torch.cuda.synchronize()
                ON["v"] = False
                dyn = {key: v - dyn0.get(key, 0) for key, v in dynamo_counters().items() if v != dyn0.get(key, 0)}
                if dyn:
                    EVENTS.append(("dynamo", dyn))
                row[k] = {"ms": round((time.perf_counter() - t) * 1e3, 2),
                          "units": {u: round(v, 2) for u, v in UNIT_MS.items()}, "events": list(EVENTS)}
            slow = {u: (row["first"]["units"][u], row["second"]["units"].get(u)) for u in row["first"]["units"]
                    if row["first"]["units"][u] > 1.5 * row["second"]["units"].get(u, 0) + 1}
            row["slow_units"] = slow
            res[L] = row
            print(f"[cold] L={L}: first {row['first']['ms']} ms, second {row['second']['ms']} ms; slow units {slow}; "
                  f"events {row['first']['events'][:40]}", flush=True)
    out = lb.MEASURED / "host" / f"cold_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"[cold] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
