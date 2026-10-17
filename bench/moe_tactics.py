"""Every FlashInfer tactic of the trtllm NVFP4 fused MoE, timed under sustained load at given routings (1 GPU).

The serving tactic file (ppimpl/data/flashinfer_tactics_b200.json, bench/tune_tactics.py) was tuned on the
autotuner's own inputs: FlashInfer regenerates every dynamic-dim input of the profile, so the router logits were
`torch.rand` (U[0,1)) and the routing came from those plus the layer's correction bias, not from served tokens. This
times all valid tactics of the GLM-5.3 MoE (288 experts, top 8, hidden 4096, intermediate 2048) at each token count
and routing histogram, so the tactic the file holds can be checked against the best one at served routing.

Inputs are the ServingStudioSim profiler runner's (`nvfp4_fused_moe`, PROFILER): random NVFP4 weights, quantized
random hidden states and router logits whose top-k gives exactly `per_expert_batches` tokens per expert. The tactic
is forced by answering FlashInfer's `AutoTuner.choose_one` for this op; the call is otherwise vLLM's monolithic call
(`trtllm_fp4_block_scale_moe`, DeepSeek-V3 routing, PDL on). Timing: per (T, routing), `--rounds` passes over all
tactics in a rotated order, each pass `--calls` back-to-back calls between two CUDA events after `--warm` calls, so
the GPU stays loaded as in serving and slow clock drift spreads over all tactics alike. The SM clock is sampled
(NVML) per pass. Each tactic's FC1/FC2 kernel names come from one profiled call.

Routings (`--routing`, repeatable `name=source`):
  - `json:<path>[:<key>]`  a JSON list of 288 counts, or an object of named lists (key picks one, else all);
  - `experts:<run dir>:<stage>[:<n>]`  the first n (default 4) microbatches of exactly T routed rows in
    `<run dir>/experts_stage<S>.jsonl` (PPIMPL_LOG_EXPERTS), every MoE layer of each as its own histogram, timed and
    reported as the mean over histograms; for T without such microbatches, T/4096-scaled sums of 4096-row ones;
  - `synth:<layer>`  what the autotuner tuned on: sigmoid(U[0,1)) + layer <layer>'s correction bias, top 8;
  - `uniform`  sigmoid(U[0,1)) top 8, zero bias (the profiler's own tuning input).

  python -m bench.moe_tactics --tokens 4096,6144,8192 --routing experts:measured/runs_m5r/<run>:2 \\
      --routing synth:3 --routing json:<sim rows>.json --out measured/moe_tactics/<job>.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from bench.common import PROFILER
from ppimpl.model.moe_tuning import scale_hist

OP = "flashinfer::trtllm_fp4_block_scale_moe"
E, TOP_K, HIDDEN, INTER = 288, 8, 4096, 2048
ARGS = {"num_experts": E, "num_local_experts": E, "top_k": TOP_K, "hidden_size": HIDDEN, "intermediate_size": INTER,
        "input_dtype": "bf16", "weight_format": "nvfp4_e2m1", "group_size": 16, "routing_method": "deepseek_v3",
        "n_group": 1, "topk_group": 1, "routed_scaling_numerator": 5, "routed_scaling_denominator": 2}


def synth_hist(tokens: int, bias, seed: int = 0) -> list[int]:
    import torch
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(tokens, E, generator=g).sigmoid()
    if bias is not None:
        s = s + bias
    return torch.bincount(s.topk(TOP_K, dim=1).indices.flatten(), minlength=E).tolist()


def layer_bias(layer: int):
    from safetensors import safe_open
    from ppimpl.config import CHECKPOINT
    idx = json.loads((Path(CHECKPOINT) / "model.safetensors.index.json").read_text())["weight_map"]
    key = next(k for k in idx if k.endswith(f"layers.{layer}.mlp.gate.e_score_correction_bias"))
    with safe_open(str(Path(CHECKPOINT) / idx[key]), "pt") as f:
        return f.get_tensor(key).float()


def routings(specs: list[str], tokens: int) -> dict[str, list[list[int]]]:
    """name -> histograms (each sums to tokens * top_k)."""
    out = {}
    for spec in specs:
        name, _, src = spec.partition("=") if "=" in spec else (spec.split(":")[0], "", spec)
        kind, _, rest = src.partition(":")
        if kind == "json":
            path, _, key = rest.partition(":")
            d = json.loads(Path(path).read_text())
            items = {key: d[key]} if key else (d if isinstance(d, dict) else {name: d})
            for k, h in items.items():
                h = h.get(str(tokens), h.get("hist")) if isinstance(h, dict) else h
                if h is not None:
                    out[f"{name}.{k}" if len(items) > 1 else name] = [scale_hist(h, tokens)]
        elif kind == "experts":
            run, stage, *n = rest.split(":")
            n = int(n[0]) if n else 4
            lines = [json.loads(l) for l in open(Path(run) / f"experts_stage{stage}.jsonl")]
            rows = lambda l: sum(l["counts"][0]) // TOP_K          # noqa: E731
            exact = [l for l in lines if rows(l) == tokens][:n]
            if exact:
                out[name] = [c for l in exact for c in l["counts"]]
            else:                                                   # sum k 4096-row microbatches, then rescale
                base = [l for l in lines if rows(l) == 4096]
                k = -(-tokens // 4096)
                groups = [base[i:i + k] for i in range(0, min(len(base), n * k), k) if len(base[i:i + k]) == k]
                out[name] = [scale_hist([sum(x) for x in zip(*(g[j]["counts"][li] for j in range(k)))], tokens)
                             for g in groups for li in range(len(g[0]["counts"]))]
        elif kind == "synth":
            out[name] = [synth_hist(tokens, layer_bias(int(rest)))]
        elif kind == "uniform":
            out[name] = [synth_hist(tokens, None)]
        else:
            raise SystemExit(f"unknown routing source {src}")
    return out


class Forced:
    """Answers AutoTuner.choose_one for OP with `tactic`; records the op's valid tactics."""
    tactic = None
    valid = None

    @classmethod
    def install(cls):
        from flashinfer.autotuner import AutoTuner
        orig = AutoTuner.choose_one

        def choose_one(self, custom_op, runners, tuning_config, inputs, **kw):
            if custom_op != OP:
                return orig(self, custom_op, runners, tuning_config, inputs, **kw)
            if cls.valid is None:
                cls.valid = [t for t in runners[0].get_valid_tactics(inputs, None)]
            return runners[0], list(cls.tactic if cls.tactic is not None else cls.valid[0])

        AutoTuner.choose_one = choose_one


def build_call(tokens: int):
    """(set_routing(hist), call()) on the profiler runner's inputs."""
    import torch
    import flashinfer
    if str(PROFILER) not in sys.path:
        sys.path.append(str(PROFILER))
    from profiling.runners.moe import nvfp4_fused_moe as R
    from profiling.runners.moe.exact_topk import exact_topk_ids

    _, quantize, prepare = R._load_runtime("vllm")
    hidden, hidden_scale, _ = R._quantized_hidden(torch, quantize, tokens, HIDDEN)
    w1, s1, w2, s2 = R._prepared_weights(torch, prepare, stack="vllm", experts=E, hidden_size=HIDDEN,
                                         intermediate_size=INTER)
    ones = torch.ones(E, dtype=torch.float32, device="cuda")
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    state = {}

    def set_routing(hist):
        ids = exact_topk_ids(num_tokens=tokens, top_k=TOP_K, per_expert_batches=tuple(hist))
        logits, bias = R._forced_routing(torch, ids, {**ARGS, "num_tokens": tokens})
        state.update(logits=logits, bias=bias)

    def call():
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=state["logits"], routing_bias=state["bias"], hidden_states=hidden,
            hidden_states_scale=hidden_scale, gemm1_weights=w1, gemm1_weights_scale=s1, gemm1_bias=None,
            gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, gemm2_weights=w2, gemm2_weights_scale=s2,
            gemm2_bias=None, output1_scale_scalar=ones, output1_scale_gate_scalar=ones, output2_scale_scalar=ones,
            num_experts=E, top_k=TOP_K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0,
            local_num_experts=E, routed_scaling_factor=2.5, routing_method_type=R.ROUTING_METHODS["deepseek_v3"],
            do_finalize=True, activation_type=3, output=out, enable_pdl=True,
            tune_max_num_tokens=R.vllm_tune_max_num_tokens(tokens))

    return set_routing, call


def kernel_names(call) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        call()
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    pick = lambda pre: next((n for n in names if n.startswith(pre)), None)    # noqa: E731
    return {"fc1": pick("bmm_E2m1"), "fc2": pick("bmm_Bfloat16")}


def sm_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def time_tactics(call, tactics, rounds: int, calls: int, warm: int) -> tuple[dict, list]:
    import torch
    times = {t: [] for t in tactics}
    clocks = []
    for r in range(rounds):
        order = tactics[r % len(tactics):] + tactics[:r % len(tactics)]
        for t in order:
            Forced.tactic = t
            for _ in range(warm):
                call()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(calls):
                call()
            b.record()
            b.synchronize()
            times[t].append(a.elapsed_time(b) / calls)
        clocks.append(sm_clock())
    return {t: statistics.median(v) for t, v in times.items()}, clocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,6144,8192")
    ap.add_argument("--routing", action="append", required=True)
    ap.add_argument("--file", action="append", default=[], help="tactic files to report (default: the serving one)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--hists", type=int, default=6, help="histograms timed per routing (spread over the list)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    Forced.install()
    files = {}
    for f in a.file or ["ppimpl/data/flashinfer_tactics_b200.json"]:
        files[f] = {int(k.split("((")[1].split(",")[0]): tuple(v[1])
                    for k, v in json.loads(Path(f).read_text()).items() if OP in k}
    report = {"job": os.environ.get("SLURM_JOB_ID"), "cases": []}
    for tokens in map(int, a.tokens.split(",")):
        set_routing, call = build_call(tokens)
        hists = routings(a.routing, tokens)
        Forced.valid = None
        set_routing(next(iter(hists.values()))[0])
        call()
        tactics = [tuple(t) for t in Forced.valid]
        names = {}
        for t in tactics:
            Forced.tactic = t
            names[t] = kernel_names(call)
        for rname, hs in hists.items():
            step = max(1, len(hs) // a.hists)
            per = []
            t0 = time.perf_counter()
            for h in hs[::step][:a.hists]:
                set_routing(h)
                med, clocks = time_tactics(call, tactics, a.rounds, a.calls, a.warm)
                per.append((med, clocks, max(h)))
            mean = {t: statistics.fmean(p[0][t] for p in per) for t in tactics}
            best = min(mean, key=mean.get)
            served = {}
            for f, tab in files.items():
                bucket = min((b for b in tab if b >= tokens), default=max(tab))
                st = tab[bucket]
                served[f] = {"bucket": bucket, "tactic": st, "ms": mean.get(st),
                             "over_best": round(mean[st] / mean[best], 4) if st in mean else None}
            case = {"tokens": tokens, "routing": rname, "hists": len(per), "max_load": [p[2] for p in per],
                    "served": served, "best_tactic": best, "best_ms": mean[best],
                    "sm_mhz": [c for p in per for c in p[1]], "wall_s": round(time.perf_counter() - t0, 1),
                    "tactics": sorted(([list(t), round(mean[t], 4), names[t]["fc1"], names[t]["fc2"]]
                                       for t in tactics), key=lambda x: x[1])}
            report["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "tactics"}), flush=True)
            for row in case["tactics"][:5]:
                print("   ", row[0], row[1], (row[2] or "")[:80], flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
        del set_routing, call
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
