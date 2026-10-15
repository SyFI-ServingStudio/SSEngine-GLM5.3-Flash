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


def stage_outputs(model, plan):
    """Stage 0's last hidden state and FP32 logits of the returned rows."""
    from vllm.forward_context import set_forward_context

    b = model.prepare(plan)
    with set_forward_context(b.attn, model.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
        x, r, p, c = model.run_layers(b, None)
        return x.clone(), model.head_logits(b, x, r, p, c).clone()


def kernel_sequence(model, plan) -> list[tuple]:
    """Names of the GPU activities (kernels, copies, fills) of one forward's `run` (the metadata is built outside the
    profiled region), in start order; compared as a multiset (kernels on vLLM's aux stream race with the main
    stream in start order, 7106 at 2k): a different tactic or code path shows as a different name or count."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    b = model.prepare(plan)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model.run(b, None)
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    ev.sort(key=lambda e: e.time_range.start)
    return [(e.name, ) for e in ev]


def _same(a, b) -> dict:
    return {"hidden_equal": bool(torch_equal(a[0], b[0])), "logits_equal": bool(torch_equal(a[1], b[1])),
            "hidden_max_abs": float((a[0].float() - b[0].float()).abs().max())}


def torch_equal(x, y) -> bool:
    import torch

    return torch.equal(x, y)


def check_trims(model, tokens, lengths) -> dict:
    """Outputs without the host trims (twice: the run-to-run baseline), then with them (installed in place). Where
    the baseline is bitwise equal the trimmed run must be too; where it is not (non-deterministic kernels), the
    kernel sequence (names in launch order) must be identical."""
    import torch

    from ppimpl.model import pp_patch

    plans = {L: lb.one_request_plan(tokens[:L]) for L in lengths if L <= 8192}
    with torch.inference_mode():
        before = {L: stage_outputs(model, p) for L, p in plans.items()}
        again = {L: stage_outputs(model, p) for L, p in plans.items()}
        seq_before = {L: kernel_sequence(model, p) for L, p in plans.items()}
        trims = model.install_host_trims()
        after = {L: stage_outputs(model, p) for L, p in plans.items()}
        seq_after = {L: kernel_sequence(model, p) for L, p in plans.items()}
        torch.cuda.synchronize()
    res = {"installed": trims, "memo": dict(pp_patch.AUTOTUNE_MEMO_STATS)}
    ok = True
    for L in plans:
        base, trim = _same(before[L], again[L]), _same(before[L], after[L])
        seq_equal = sorted(seq_before[L]) == sorted(seq_after[L])
        deterministic = base["hidden_equal"] and base["logits_equal"]
        passed = seq_equal and (not deterministic or (trim["hidden_equal"] and trim["logits_equal"]))
        res[L] = {"repeat": base, "trimmed": trim, "kernel_sequence_equal": seq_equal,
                  "kernels": len(seq_after[L]), "pass": passed}
        if not seq_equal:
            diff = [(i, x, y) for i, (x, y) in enumerate(zip(seq_before[L], seq_after[L])) if x != y][:5]
            res[L]["first_diffs"] = [(i, x[0][:80], y[0][:80]) for i, x, y in diff]
        ok &= passed
    res["pass"] = ok
    return res


def compare_metadata(model, tokens, lengths, old_path: Path) -> dict:
    """Stage-0 outputs with this tree's StageMetadata vs the StageMetadata module at `old_path` (a previous commit's
    metadata.py), on the same model and KV: one request per length <= 8k, plus a two-request plan whose short
    request is reordered first (decode class). All requests start at 0, so re-running a plan is idempotent."""
    import importlib.util
    from types import SimpleNamespace as NS

    import torch

    spec = importlib.util.spec_from_file_location("ppimpl_old_metadata", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(old)
    old_meta = old.StageMetadata(model.vc, model.kv, model.device)
    new_meta = model.meta
    plans = {str(L): lb.one_request_plan(tokens[:L]) for L in lengths if L <= 8192}
    plans["two_requests"] = NS(mb_id=0, chunks=[
        NS(req_id="a", start=0, num_tokens=1000, kv_blocks=[1], state_blocks=list(range(2, 8)), is_last=True,
           token_ids=tokens[:1000]),
        NS(req_id="b", start=0, num_tokens=128, kv_blocks=[8], state_blocks=list(range(9, 15)), is_last=True,
           token_ids=tokens[2000:2128])])
    res = {}
    with torch.inference_mode():
        for name, plan in plans.items():
            model.meta = old_meta
            a = stage_outputs(model, plan)
            a2 = stage_outputs(model, plan)
            seq_a = kernel_sequence(model, plan)
            model.meta = new_meta
            b = stage_outputs(model, plan)
            seq_b = kernel_sequence(model, plan)
            torch.cuda.synchronize()
            base, new = _same(a, a2), _same(a, b)
            deterministic = base["hidden_equal"] and base["logits_equal"]
            same_kernels = sorted(seq_a) == sorted(seq_b)   # as a multiset: aux-stream kernels race in start order
            res[name] = {"repeat": base, "packed": new, "kernel_sequence_equal": same_kernels,
                         "pass": same_kernels and (not deterministic or (new["hidden_equal"] and new["logits_equal"]))}
    model.meta = new_meta
    res["pass"] = all(v["pass"] for v in res.values())
    return res


def bench(args) -> None:
    import torch
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    lengths = [int(x) for x in args.lengths.split(",")]
    dev = torch.device("cuda", 0)
    t0 = time.time()
    model = StageModel(bench_config(max(lengths)), 0, dev, warmup=False, last=True, num_blocks=lb.NUM_BLOCKS,
                       max_model_len=max(max(lengths), 8192) + 8576, host_trims=not args.check_trims)
    install(model)
    print(f"[host] built in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        model.warmup()
    print(f"[host] warmup (autotune) {time.time() - t0:.0f} s", flush=True)
    if args.cuda_graphs:
        model.capture_graphs()
    cpm = lb.sleep_cycles_per_ms()
    tokens = lb.enwik9_tokens(CHECKPOINT, max(lengths))
    result = {}
    if args.check_trims:
        result["trims_check"] = check_trims(model, tokens, lengths)
        print(f"[host] trims check: {result['trims_check']}", flush=True)
    if args.compare_metadata:
        result["metadata_check"] = compare_metadata(model, tokens, lengths, args.compare_metadata)
        print(f"[host] metadata check: {result['metadata_check']}", flush=True)
    if args.check_trims or args.compare_metadata:
        # checks only: the profiler they use leaves CUPTI attached, which adds launch overhead to any later timing
        # (7106: +1.7 ms per 1k forward); time in a separate process
        out = OUT / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}_checks.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str))
        print(f"[host] wrote {out}", flush=True)
        return
    trace_dir = Path(tempfile.mkdtemp(prefix="pph-", dir=os.environ.get("TMPDIR")))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}.json"
    from ppimpl.model import pp_patch

    # phase 1, plain timing of every length before any profiler runs: a profiler session leaves CUPTI attached,
    # which adds ~4 us to every later launch (7106)
    plain_res = {}
    for L in lengths:
        plan = lb.one_request_plan(tokens[:L])
        with torch.inference_mode():
            model.forward(plan, None)          # first forward at this length (allocator, builders)
            torch.cuda.synchronize()
            idle = plain(model, plan, args.runs, 0, cpm)
            busy_ms = 2 * idle["forward"] / 1e3 + 50
            busy = plain(model, plan, args.runs, busy_ms, cpm)
        plain_res[L] = (idle, busy, busy_ms)
        print(f"[host] L={L}: forward idle {idle['forward'] / 1e3:.1f} ms (median "
              f"{idle['_median_forward_us'] / 1e3:.1f}), busy {busy['forward'] / 1e3:.1f} ms", flush=True)
    # phase 2: py-spy (no CUPTI), then the profiled runs
    for L in lengths:
        plan = lb.one_request_plan(tokens[:L])
        idle, busy, busy_ms = plain_res[L]
        with torch.inference_mode():
            spy = None
            if args.spy_seconds and L <= 8192:
                spy = spy_loop_marker(model, plan, args.spy_seconds,
                                      OUT / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}_spy_L{L}.txt")
        units = {k: v for k, v in idle.items() if not k.startswith("_") and k != "forward"}
        result[L] = {"prompt_tokens": L, "cuda_graphs": model.graph_info, "autotune_memo": dict(pp_patch.AUTOTUNE_MEMO_STATS), "idle_us": idle,
                     "busy_us": busy, "busy_sleep_ms": round(busy_ms, 1), "idle_by_type": by_type(units),
                     "spy": spy}
    for L in lengths:
        plan = lb.one_request_plan(tokens[:L])
        with torch.inference_mode():
            prof = profiled(model, plan, trace_dir, f"L{L}")
        r = result[L]
        r.update(profiled=prof, profiled_by_type=by_type(prof["units"]))
        out_path.write_text(json.dumps(result, indent=1))
        print(f"[host] L={L}: profiled run {prof['run_us'] / 1e3:.1f} ms, {prof['kernels']} kernels", flush=True)
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
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="capture the stage's CUDA graphs after warm-up: lengths up to the largest bucket replay")
    ap.add_argument("--check-trims", action="store_true",
                    help="build without the host trims, compare outputs before/after installing them, then time")
    ap.add_argument("--compare-metadata", type=Path, default=None,
                    help="a previous commit's metadata.py: compare stage-0 outputs with it vs this tree's")
    ap.add_argument("--spy", type=Path, default=None, help="fold a py-spy raw file and exit")
    a = ap.parse_args()
    if a.spy:
        fold_spy(a.spy)
        return
    bench(a)


if __name__ == "__main__":
    main()
