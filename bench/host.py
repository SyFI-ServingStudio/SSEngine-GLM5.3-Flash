"""Host (CPU) time of one stage-0 forward, per layer: where it goes and how many launches it issues.

    python bench/host.py --lengths 1024,8192,131072 [--runs 5] [--spy-seconds 20]

One GPU, stage 0 with the output head (the M2 bench model, `bench/layers.py`), router routing, the runtime's
warm-up (FlashInfer autotune). Per length:

1. plain: per-unit host wall time with `perf_counter_ns` around every layer / embed / hc_expand / output_head and
   `prepare`, no profiler. Two variants: `idle` (forward issued with the GPU idle: the queue never fills at short L,
   so this is pure host time) and `busy` (a GPU sleep longer than the host time is queued first, as the M2 bench
   did: every launch is queued, nothing runs).
2. profiled (`idle`, unit annotations only): per unit the wall span, the number of kernel launches, time inside CUDA
   runtime/driver calls, time inside torch/C++ ops outside those calls, and the rest (Python). Per op name the
   self time, per layer type. Calls > 1 ms are listed with their launch ordinal and how many launched kernels had
   not finished on the GPU at that moment (queue back-pressure vs a sync).
3. `--spy-seconds S`: forwards for S seconds inside `spy_loop_marker` at each length <= 8k while
   `py-spy record --native` samples the process; `bench/host.py --spy <raw file>` folds the samples under the
   marker.

Output: `measured/host/<job>.json` and a printed summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as lb  # noqa: E402  (also puts the lane on sys.path)

OUT = lb.MEASURED / "host"
LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel")


class Clock:
    """perf_counter_ns around unit callables; `acc[unit]` sums ns over the current forward."""
    acc: dict = collections.defaultdict(int)
    on = False

    @staticmethod
    def wrap(obj, attr: str, unit: str):
        orig = getattr(obj, attr)

        def call(*a, **kw):
            if not Clock.on:
                return orig(*a, **kw)
            t = time.perf_counter_ns()
            try:
                return orig(*a, **kw)
            finally:
                Clock.acc[unit] += time.perf_counter_ns() - t

        setattr(obj, attr, call)


def install(model) -> None:
    import torch
    import vllm.models.glm5next.common.model as glm_model

    for layer in model.layers:
        Clock.wrap(layer, "forward", f"L{layer.layer_idx}")
        lb.Scopes.wrap(layer, "forward", f"L{layer.layer_idx}", unit=True)
    if model.embed is not None:
        Clock.wrap(model.embed, "forward", "embed")
        lb.Scopes.wrap(model.embed, "forward", "embed", unit=True)
    Clock.wrap(glm_model, "hc_expand", "hc_expand")
    lb.Scopes.wrap(glm_model, "hc_expand", "hc_expand", unit=True)
    Clock.wrap(model, "output_head", "output_head")
    lb.Scopes.wrap(model, "output_head", "output_head", unit=True)
    Clock.wrap(model, "prepare", "prepare")
    lb.Scopes.wrap(model, "prepare", "prepare", unit=True)
    del torch


def plain(model, plan, runs: int, busy_ms: float, cpm: float) -> dict:
    import torch

    per = collections.defaultdict(list)
    for _ in range(runs):
        torch.cuda.synchronize()
        if busy_ms:
            torch.cuda._sleep(int(busy_ms * cpm))
        Clock.acc.clear()
        Clock.on = True
        t = time.perf_counter_ns()
        model.forward(plan, None)
        total = time.perf_counter_ns() - t
        Clock.on = False
        torch.cuda.synchronize()
        for k, v in Clock.acc.items():
            per[k].append(v / 1e3)
        per["forward"].append(total / 1e3)
    return {k: round(min(v), 1) for k, v in per.items()} | {"_median_forward_us": round(sorted(per["forward"])[runs // 2], 1)}


def profiled(model, plan, trace_dir: Path, tag: str) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    lb.Scopes.on = False
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for r in range(2):
            with record_function(f"run{r}"):
                model.forward(plan, None)
            torch.cuda.synchronize()
    path = trace_dir / f"{tag}.json"
    prof.export_chrome_trace(str(path))
    res = analyze(json.loads(path.read_text())["traceEvents"])
    path.unlink()
    return res


def analyze(ev: list[dict]) -> dict:
    """Second run of the trace: per unit span / launches / runtime / op / python, op self times, blocking calls."""
    runs = sorted((e for e in ev if e.get("ph") == "X" and lb.RUN.match(e.get("name", ""))
                   and e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    run = runs[-1]
    tid = run["tid"]
    t0, t1 = run["ts"], run["ts"] + run["dur"]
    host = sorted((e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and t0 <= e["ts"] <= t1
                   and e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver")),
                  key=lambda e: (e["ts"], -e["dur"]))
    gpu = {e["args"]["correlation"]: e for e in ev if e.get("ph") == "X" and e.get("cat") in lb.GPU_CATS
           and "correlation" in e.get("args", {})}
    units = [e for e in host if e["cat"] == "user_annotation" and (lb.UNIT.match(e["name"]) or e["name"] == "prepare")]

    def unit_of(ts):
        for u in units:
            if u["ts"] <= ts <= u["ts"] + u["dur"]:
                return u["name"]
        return "other"

    # self time: subtract direct children on the same thread (events are properly nested)
    stack, self_us, top_op = [], {}, set()
    for i, e in enumerate(host):
        end = e["ts"] + e["dur"]
        while stack and host[stack[-1]]["ts"] + host[stack[-1]]["dur"] < e["ts"] + 1e-3:
            stack.pop()
        parent = stack[-1] if stack else None
        self_us[i] = e["dur"]
        if parent is not None:
            self_us[parent] -= e["dur"]
        if e["cat"] in ("cpu_op", "cuda_runtime", "cuda_driver") and not any(
                host[j]["cat"] in ("cpu_op", "cuda_runtime", "cuda_driver") for j in stack):
            top_op.add(i)
        if end > e["ts"]:
            stack.append(i)

    per_unit = collections.defaultdict(lambda: collections.Counter())
    ops = collections.defaultdict(collections.Counter)
    launched_end = []           # GPU end time of every launched kernel, in launch order
    blocking = []
    for i, e in enumerate(host):
        if e["cat"] == "user_annotation":
            continue
        u = unit_of(e["ts"])
        c = per_unit[u]
        if e["cat"] in ("cuda_runtime", "cuda_driver"):
            c["runtime_us"] += e["dur"]
            c["runtime_calls"] += 1
            is_launch = e["name"] in LAUNCH
            if is_launch:
                c["launches"] += 1
            corr = e.get("args", {}).get("correlation")
            if e["dur"] > 1000:
                pending = sum(1 for g in launched_end if g > e["ts"])
                blocking.append({"call": e["name"], "ms": round(e["dur"] / 1e3, 2), "unit": u,
                                 "launch_ordinal": len(launched_end), "pending_kernels_at_call": pending})
            if is_launch and corr in gpu:
                g = gpu[corr]
                launched_end.append(g["ts"] + g["dur"])
        ops[lb.unit_type(u) if lb.UNIT.match(u) else u][e["name"]] += self_us[i]
        if i in top_op:
            c["op_us"] += e["dur"]
    out_units = {}
    for u in units:
        c = per_unit[u["name"]]
        span = u["dur"]
        out_units[u["name"]] = {"span_us": round(span, 1), "launches": c["launches"],
                                "runtime_calls": c["runtime_calls"], "runtime_us": round(c["runtime_us"], 1),
                                "op_outside_runtime_us": round(c["op_us"] - c["runtime_us"], 1),
                                "python_us": round(span - c["op_us"], 1)}
    top_ops = {t: [(n, round(v, 1)) for n, v in cnt.most_common(25)] for t, cnt in ops.items()}
    return {"run_us": round(run["dur"], 1), "units": out_units, "top_self_us": top_ops,
            "kernels": sum(1 for e in ev if e.get("cat") == "kernel" and t0 <= e["ts"]),
            "blocking": blocking[:30]}


def by_type(units: dict, n_runs: int = 1) -> dict:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for u, row in units.items():
        t = lb.unit_type(u) if lb.UNIT.match(u) else u
        for k, v in (row.items() if isinstance(row, dict) else [("host_us", row)]):
            agg[t][k].append(v)
    return {t: {k: round(sum(v) / len(v), 1) for k, v in d.items()} | {"n": len(next(iter(d.values())))}
            for t, d in agg.items()}


def spy_loop_marker(model, plan, seconds: float, raw: Path) -> dict:
    """Forwards for `seconds` while `py-spy record --native` samples this process (attached by pid)."""
    import subprocess

    import torch

    spy = subprocess.Popen(["py-spy", "record", "--native", "--rate", "200", "--format", "raw", "-o", str(raw),
                            "-p", str(os.getpid()), "--duration", str(int(seconds))],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2)            # attach
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds + 1:
        model.forward(plan, None)
        torch.cuda.synchronize()
        n += 1
    out = spy.communicate(timeout=120)[0]
    return {"forwards": n, "ms_per_forward": round((time.perf_counter() - t) / n * 1e3, 2), "raw": str(raw),
            "py_spy": out[-500:]}


def bench(args) -> None:
    import torch
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    lengths = [int(x) for x in args.lengths.split(",")]
    dev = torch.device("cuda", 0)
    t0 = time.time()
    model = StageModel(bench_config(max(lengths)), 0, dev, warmup=False, last=True, num_blocks=lb.NUM_BLOCKS,
                       max_model_len=max(max(lengths), 8192) + 8576)
    install(model)
    print(f"[host] built in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        model.warmup()
    print(f"[host] warmup (autotune) {time.time() - t0:.0f} s", flush=True)
    cpm = lb.sleep_cycles_per_ms()
    tokens = lb.enwik9_tokens(CHECKPOINT, max(lengths))
    trace_dir = Path(tempfile.mkdtemp(prefix="pph-", dir=os.environ.get("TMPDIR")))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}.json"
    result = {}
    for L in lengths:
        plan = lb.one_request_plan(tokens[:L])
        with torch.inference_mode():
            model.forward(plan, None)          # first forward at this length (allocator, builders)
            torch.cuda.synchronize()
            idle = plain(model, plan, args.runs, 0, cpm)
            busy_ms = 2 * idle["forward"] / 1e3 + 50
            busy = plain(model, plan, args.runs, busy_ms, cpm)
            prof = profiled(model, plan, trace_dir, f"L{L}")
            spy = None
            if args.spy_seconds and L <= 8192:
                spy = spy_loop_marker(model, plan, args.spy_seconds,
                                      OUT / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}_spy_L{L}.txt")
        units = {k: v for k, v in idle.items() if not k.startswith("_") and k != "forward"}
        r = {"prompt_tokens": L, "idle_us": idle, "busy_us": busy, "busy_sleep_ms": round(busy_ms, 1),
             "idle_by_type": by_type(units), "profiled": prof, "profiled_by_type": by_type(prof["units"]),
             "spy": spy}
        result[L] = r
        out_path.write_text(json.dumps(result, indent=1))
        print(f"[host] L={L}: forward idle {idle['forward'] / 1e3:.1f} ms (median {idle['_median_forward_us'] / 1e3:.1f}),"
              f" busy {busy['forward'] / 1e3:.1f} ms; profiled run {prof['run_us'] / 1e3:.1f} ms, "
              f"{prof['kernels']} kernels", flush=True)
        for t, row in r["profiled_by_type"].items():
            print(f"[host]   {t:16s} idle {r['idle_by_type'].get(t, {}).get('host_us', '-')} us | prof {row}",
                  flush=True)
        for b in prof["blocking"][:8]:
            print(f"[host]   blocking {b}", flush=True)
    print(f"[host] wrote {out_path}", flush=True)


def fold_spy(raw: Path, marker: str = "spy_loop_marker", depth_files=("vllm", "ppimpl", "torch", "flashinfer")):
    """py-spy raw (collapsed) samples under the marker -> self samples per frame and inclusive per frame."""
    self_c, incl = collections.Counter(), collections.Counter()
    total = 0
    for line in raw.read_text().splitlines():
        stack, _, n = line.rpartition(" ")
        frames = stack.split(";")
        idx = next((i for i, f in enumerate(frames) if marker in f), None)
        if idx is None:
            continue
        n = int(n)
        total += n
        frames = frames[idx + 1:]
        if frames:
            self_c[frames[-1]] += n
        for f in set(frames):
            incl[f] += n
    print(f"[spy] {total} samples under {marker}")
    print("[spy] top self:")
    for f, n in self_c.most_common(40):
        print(f"  {100 * n / total:5.1f}%  {f[:200]}")
    print("[spy] top inclusive:")
    for f, n in incl.most_common(60):
        print(f"  {100 * n / total:5.1f}%  {f[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="1024,8192,131072")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--spy-seconds", type=float, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--spy", type=Path, default=None, help="fold a py-spy raw file and exit")
    a = ap.parse_args()
    if a.spy:
        fold_spy(a.spy)
        return
    bench(a)


if __name__ == "__main__":
    main()
