"""Model correctness on GPU: the stage split against the unsplit layers, and the 4-stage chain against vLLM.

    python -m research.bench.chain split  [--prompt-len 1024]   # 1 GPU: layers 0-22 unsplit vs split at 12 (stage edge)
    python -m research.bench.chain chain  [--prompt-len 1024]   # 4 GPUs: StageModel 0..3, serving transport, greedy token
    python -m research.bench.chain vllm   [--prompt-len 1024]   # 4 GPUs: vLLM fork TP4 + EP on the same prompt (reference)

`split` (test b): one model holding layers 0-22 runs them (i) back to back, as one stage would, and (ii) as stage 0
(0-11) + edge (`hc_post`, flatten, view) + stage 1 (12-22, standalone `hc_pre`); both end with layer 22's `hc_post`.
Reports the relative error of (ii) against (i).
`chain` (tests c, d): four processes, one `StageModel` each, joined by the serving transport (`open_pipe`, the CE
ring, as in the runtime and research/bench/kvcheck.py). Stage 3 returns the greedy token and its top-5 logits; every stage
traces its forward and checks it against the kernel manifest.
`vllm` runs the reference on the same token ids (max_tokens 1, greedy, top-5 logprobs).
Prompt: the first L tokens of enwik9. Results: `research/measured/chain/<mode>_<L>_<job>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from benchmarks.common import MEASURED, NUM_BLOCKS, enwik9_tokens, one_request_plan

OUT = MEASURED / "chain"
JOB = os.environ.get("SLURM_JOB_ID", "local")


def _write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{JOB}.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"[chain] wrote {path}: {json.dumps(data)[:1500]}", flush=True)


def _prompt(L: int) -> list[int]:
    from ssengine.model.stage import CHECKPOINT

    return enwik9_tokens(CHECKPOINT, L)


def split(L: int) -> None:
    import torch
    from ssengine.model import pp_patch
    from ssengine.model.stage import StageModel, bench_config
    from vllm.forward_context import set_forward_context

    dev = torch.device("cuda", 0)
    cfg = bench_config(max(L, 8192))
    t0 = time.time()
    m = StageModel(cfg, 0, dev, layers=range(0, 23), first=True, last=False, num_blocks=NUM_BLOCKS, warmup=False,
                   max_model_len=max(L, 8192) + 8576)
    print(f"[chain] split model built in {time.time() - t0:.0f} s", flush=True)
    plan = one_request_plan(_prompt(L))
    cut = 12
    with torch.inference_mode():
        b = m.prepare(plan)
        with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x = m.embed(b.token_ids)
            r = p = c = None
            for layer in m.layers:
                x, r, p, c = layer(b.positions, x, r, p, c)
            unsplit = pp_patch.edge_send(m.layers[-1], x, r, p, c).float()
        b = m.prepare(plan)
        with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x = m.embed(b.token_ids)
            r = p = c = None
            for layer in m.layers[:cut]:
                x, r, p, c = layer(b.positions, x, r, p, c)
            hidden = pp_patch.edge_send(m.layers[cut - 1], x, r, p, c)
            x, r, p, c = pp_patch.edge_recv(hidden), None, None, None
            for layer in m.layers[cut:]:
                x, r, p, c = layer(b.positions, x, r, p, c)
            split_out = pp_patch.edge_send(m.layers[-1], x, r, p, c).float()
        torch.cuda.synchronize()
    diff = (split_out - unsplit)
    rel = (diff.norm() / unsplit.norm()).item()
    per_tok = (diff.norm(dim=1) / unsplit.norm(dim=1).clamp_min(1e-6))
    cos = torch.nn.functional.cosine_similarity(split_out, unsplit, dim=1)
    _write(f"split_{L}", {"prompt_tokens": L, "layers": "0-22", "cut": cut, "rel_fro_error": rel,
                          "max_token_rel_error": per_tok.max().item(), "min_token_cosine": cos.min().item(),
                          "finite": bool(torch.isfinite(split_out).all().item()),
                          "pass": rel < 2e-2 and cos.min().item() > 0.999})


def _rel(a, b) -> dict:
    import torch

    a, b = a.float().flatten(1), b.float().flatten(1)
    d = a - b
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return {"rel_fro": (d.norm() / b.norm().clamp_min(1e-30)).item(), "min_cos": cos.min().item(),
            "max_abs": d.abs().max().item()}


def edge(L: int) -> None:
    """Diagnostics for the stage edge: layer 12's attention boundary computed (A) by mega_mhc (design routing),
    (B) by the fork's fused TileLang post+pre, (C) by hc_post then a standalone hc_pre (the stage edge); and the
    unsplit 0-22 run repeated (noise), and with the fork's fused op at every boundary."""
    import torch
    from ssengine.model import pp_patch
    from ssengine.model.stage import StageModel, bench_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp

    dev = torch.device("cuda", 0)
    cfg = bench_config(max(L, 8192))
    m = StageModel(cfg, 0, dev, layers=range(0, 23), first=True, last=False, num_blocks=NUM_BLOCKS, warmup=False,
                   max_model_len=max(L, 8192) + 8576)
    plan = one_request_plan(_prompt(L))
    fork_op = MHCFusedPostPreOp()
    res = {"prompt_tokens": L}

    def run(layers, cut=None):
        b = m.prepare(plan)
        with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x = m.embed(b.token_ids)
            r = p = c = None
            states = []
            for i, layer in enumerate(layers):
                if cut is not None and i == cut:
                    x, r, p, c = pp_patch.edge_recv(pp_patch.edge_send(layers[i - 1], x, r, p, c)), None, None, None
                states.append((x, r, p, c))
                x, r, p, c = layer(b.positions, x, r, p, c)
            return pp_patch.edge_send(layers[-1], x, r, p, c), states

    with torch.inference_mode():
        u1, states = run(m.layers)
        u2, _ = run(m.layers)
        res["unsplit_repeat"] = _rel(u2, u1)
        x, r, p, c = states[12]
        L12 = m.layers[12]
        args = (L12.hc_attn_fn, L12.hc_attn_scale, L12.hc_attn_base)
        kw = dict(norm_weight=L12.input_layernorm.weight.data, norm_eps=L12.input_layernorm.variance_epsilon)
        A = L12.hc_fused_post_pre(x, r, p, c, *args, **kw)
        saved = L12.mhc_fused_post_pre_op
        L12.mhc_fused_post_pre_op = fork_op
        B = L12.hc_fused_post_pre(x, r, p, c, *args, **kw)
        L12.mhc_fused_post_pre_op = saved
        post_out = m.layers[11].hc_post(x, r, p, c)
        Cp, Cc, Cy = L12.hc_pre(post_out, *args, **kw)
        res["edge_residual_mega_vs_fork"] = _rel(A[0], B[0])
        res["edge_y_mega_vs_fork"] = _rel(A[3], B[3])
        res["edge_residual_split_vs_fork"] = _rel(post_out, B[0])
        res["edge_y_split_vs_fork"] = _rel(Cy, B[3])
        res["edge_post_split_vs_fork"] = _rel(Cp, B[1])
        res["edge_comb_split_vs_fork"] = _rel(Cc, B[2])
        res["edge_post_mega_vs_fork"] = _rel(A[1], B[1])
        res["edge_comb_mega_vs_fork"] = _rel(A[2], B[2])
        s1, _ = run(m.layers, cut=12)
        res["split_vs_unsplit_mega"] = _rel(s1, u1)
        for layer in m.layers:
            layer.mhc_fused_post_pre_op = fork_op
        f1, fstates = run(m.layers)
        fs, _ = run(m.layers, cut=12)
        res["unsplit_fork_vs_mega"] = _rel(f1, u1)
        # inputs of layer i (layer 0: the embedding) under the two boundary ops: where the runs drift apart
        res["layer_input_fork_vs_mega"] = [round(_rel(a[0], b[0])["rel_fro"], 6) for a, b in zip(fstates, states)]
        res["layer_residual_fork_vs_mega"] = [None if a[1] is None else round(_rel(a[1], b[1])["rel_fro"], 6)
                                              for a, b in zip(fstates, states)]
        res["split_vs_unsplit_fork"] = _rel(fs, f1)
        torch.cuda.synchronize()
    res["pass"] = (res["split_vs_unsplit_fork"]["rel_fro"] < 1e-3 and res["edge_y_split_vs_fork"]["rel_fro"] < 1e-3
                   and res["edge_y_mega_vs_fork"]["rel_fro"] < 1e-2)
    _write(f"edge_{L}", res)


def _chain_worker(rank: int, L: int, port: int, profile: bool) -> None:
    import torch
    from ssengine import kernels as km
    from ssengine.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ssengine.model.stage import StageModel, bench_config
    from ssengine.runtime.transport import open_pipe

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    pipe = open_pipe(RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=L)), rank, dev, HIDDEN_WIDTH)
    cfg = bench_config(max(L, 8192))
    t0 = time.time()
    m = StageModel(cfg, rank, dev, num_blocks=NUM_BLOCKS, max_model_len=max(L, 8192) + 8576)
    print(f"[chain] rank {rank} layers {m.layer_range} built in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    plan = one_request_plan(_prompt(L))
    result = {"stage": rank, "layers": [m.layer_range.start, m.layer_range.stop]}

    def step():
        hidden = pipe.recv(L, HIDDEN_WIDTH) if rank > 0 else None
        b = m.prepare(plan)
        if rank < 3:
            out = m.run(b, hidden).contiguous()
            done = torch.cuda.Event()
            done.record()
            if hidden is not None:
                pipe.release()
            pipe.send(out, done)
            return out
        from vllm.forward_context import set_forward_context

        with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x, r, p, c = m.run_layers(b, hidden)
            logits = m.head_logits(b, x, r, p, c)
        pipe.release()
        return logits

    with torch.inference_mode():
        out = step()          # first pass (JIT/autotune), then the research/measured/traced one
        torch.cuda.synchronize()
        if profile:
            from torch.profiler import ProfilerActivity
            from torch.profiler import profile as prof_ctx

            with prof_ctx(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = step()
                torch.cuda.synchronize()
            try:
                names = sorted({e.name() for e in prof.profiler.kineto_results.events()
                                if e.device_type() == torch.autograd.DeviceType.CUDA})
                chk = km.check_trace(names, rank, optional=frozenset({19, 20, 21} if L <= 2048 else set()))
                result["trace_check"] = {"ok": chk.ok, "summary": chk.summary(), "kernels": names}
            except Exception as e:  # noqa: BLE001 - the token result must still be written
                result["trace_check"] = {"ok": False, "summary": f"trace check failed: {type(e).__name__}: {e}"}
        else:
            out = step()
            torch.cuda.synchronize()
    if rank == 3:
        logits = out[0]
        top = logits.topk(5)
        result.update({"token": int(top.indices[0]), "top5_ids": top.indices.tolist(),
                       "top5_logprobs": torch.log_softmax(logits, -1)[top.indices].tolist(),
                       "finite": bool(torch.isfinite(logits).all().item())})
    _write(f"chain_{L}_rank{rank}", result)
    pipe.close()


def chain(L: int, profile: bool) -> None:
    import torch.multiprocessing as mp

    port = 29500 + int(JOB) % 10000 if JOB.isdigit() else 29511
    mp.spawn(_chain_worker, args=(L, port, profile), nprocs=4, join=True)


def vllm_reference(L: int) -> None:
    from ssengine.model.stage import CHECKPOINT
    from vllm import LLM, SamplingParams

    ids = _prompt(L)
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4, enable_expert_parallel=True, kv_cache_dtype="fp8_e4m3",
              language_model_only=True, max_model_len=max(L, 8192) + 64, enable_prefix_caching=False,
              enforce_eager=True, kda_prefill_backend="triton")   # accuracy reference: not FlashKDA (issues.md)
    out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0, logprobs=5))
    o = out[0].outputs[0]
    lp = o.logprobs[0]
    top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:5]
    _write(f"vllm_{L}", {"prompt_tokens": L, "token": int(o.token_ids[0]), "top5_ids": [int(k) for k, _ in top],
                         "top5_logprobs": [v.logprob for _, v in top]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["split", "chain", "vllm", "edge"])
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    a = ap.parse_args()
    if a.mode == "split":
        split(a.prompt_len)
    elif a.mode == "edge":
        edge(a.prompt_len)
    elif a.mode == "chain":
        chain(a.prompt_len, a.profile)
    else:
        vllm_reference(a.prompt_len)


if __name__ == "__main__":
    main()
