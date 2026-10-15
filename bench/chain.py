"""Model correctness on GPU: the stage split against the unsplit layers, and the 4-stage chain against vLLM.

    python bench/chain.py split  [--prompt-len 1024]   # 1 GPU: layers 0-22 unsplit vs split at 12 (stage edge)
    python bench/chain.py chain  [--prompt-len 1024]   # 4 GPUs: StageModel 0..3, NCCL send/recv, greedy token
    python bench/chain.py vllm   [--prompt-len 1024]   # 4 GPUs: vLLM fork TP4 + EP on the same prompt (reference)

`split` (test b): one model holding layers 0-22 runs them (i) back to back, as one stage would, and (ii) as stage 0
(0-11) + edge (`hc_post`, flatten, view) + stage 1 (12-22, standalone `hc_pre`); both end with layer 22's `hc_post`.
Reports the relative error of (ii) against (i).
`chain` (tests c, d): four processes, one `StageModel` each, joined in one 4-rank NCCL world (vLLM's groups are
single-rank inside it, as in the runtime); hidden rows go stage to stage by `dist.send`/`recv`. Stage 3 returns the
greedy token and its top-5 logits; every stage traces its forward and checks it against the kernel manifest.
`vllm` runs the reference on the same token ids (max_tokens 1, greedy, top-5 logprobs).
Prompt: the first L tokens of enwik9. Results: `measured/chain/<mode>_<L>_<job>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE))
sys.path.insert(0, str(LANE / "bench"))

OUT = LANE / "measured" / "chain"
JOB = os.environ.get("SLURM_JOB_ID", "local")


def _write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{JOB}.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"[chain] wrote {path}: {json.dumps(data)[:1500]}", flush=True)


def _prompt(L: int) -> list[int]:
    from layers import enwik9_tokens
    from ppimpl.model.stage import CHECKPOINT

    return enwik9_tokens(CHECKPOINT, L)


def split(L: int) -> None:
    import torch
    from layers import NUM_BLOCKS, one_request_plan
    from ppimpl.model import pp_patch
    from ppimpl.model.stage import CHECKPOINT, StageModel
    from vllm.forward_context import set_forward_context

    dev = torch.device("cuda", 0)
    cfg = SimpleNamespace(mb_tokens=max(L, 8192), checkpoint=CHECKPOINT, pool_tokens=None)
    t0 = time.time()
    m = StageModel(cfg, 0, dev, layers=range(0, 23), first=True, last=False, num_blocks=NUM_BLOCKS,
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


def _chain_worker(rank: int, L: int, port: int, profile: bool) -> None:
    import torch
    import torch.distributed as dist
    from layers import NUM_BLOCKS, one_request_plan
    from ppimpl import kernels as km
    from ppimpl.model.stage import CHECKPOINT, HIDDEN_WIDTH, StageModel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4,
                            device_id=dev)
    cfg = SimpleNamespace(mb_tokens=max(L, 8192), checkpoint=CHECKPOINT, pool_tokens=None,
                          layer_split=km.SPLIT)
    t0 = time.time()
    m = StageModel(cfg, rank, dev, num_blocks=NUM_BLOCKS, max_model_len=max(L, 8192) + 8576)
    print(f"[chain] rank {rank} layers {m.layer_range} built in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    plan = one_request_plan(_prompt(L))
    result = {"stage": rank, "layers": [m.layer_range.start, m.layer_range.stop]}

    def step():
        hidden = None
        if rank > 0:
            hidden = torch.empty(L, HIDDEN_WIDTH, dtype=torch.bfloat16, device=dev)
            dist.recv(hidden, rank - 1)
        b = m.prepare(plan)
        if rank < 3:
            out = m.run(b, hidden)
            dist.send(out.contiguous(), rank + 1)
            return out
        from vllm.forward_context import set_forward_context

        with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x, r, p, c = m.run_layers(b, hidden)
            return m.head_logits(b, x, r, p, c)

    with torch.inference_mode():
        out = step()          # first pass (JIT/autotune), then the measured/traced one
        torch.cuda.synchronize()
        if profile:
            from torch.profiler import ProfilerActivity
            from torch.profiler import profile as prof_ctx

            with prof_ctx(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = step()
                torch.cuda.synchronize()
            names = sorted({e.name for e in prof.profiler.kineto_results.events()
                            if e.device_type() == torch.autograd.DeviceType.CUDA})
            chk = km.check_trace(names, rank, optional=frozenset({19, 20, 21} if L <= 2048 else set()))
            result["trace_check"] = {"ok": chk.ok, "summary": chk.summary(), "kernels": names}
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
    dist.barrier()
    dist.destroy_process_group()


def chain(L: int, profile: bool) -> None:
    import torch.multiprocessing as mp

    port = 29500 + int(JOB) % 10000 if JOB.isdigit() else 29511
    mp.spawn(_chain_worker, args=(L, port, profile), nprocs=4, join=True)


def vllm_reference(L: int) -> None:
    from ppimpl.model.stage import CHECKPOINT
    from vllm import LLM, SamplingParams

    ids = _prompt(L)
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4, enable_expert_parallel=True, kv_cache_dtype="fp8_e4m3",
              language_model_only=True, max_model_len=max(L, 8192) + 64, enable_prefix_caching=False,
              enforce_eager=True)
    out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0, logprobs=5))
    o = out[0].outputs[0]
    lp = o.logprobs[0]
    top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:5]
    _write(f"vllm_{L}", {"prompt_tokens": L, "token": int(o.token_ids[0]), "top5_ids": [int(k) for k, _ in top],
                         "top5_logprobs": [v.logprob for _, v in top]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["split", "chain", "vllm"])
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    a = ap.parse_args()
    if a.mode == "split":
        split(a.prompt_len)
    elif a.mode == "chain":
        chain(a.prompt_len, a.profile)
    else:
        vllm_reference(a.prompt_len)


if __name__ == "__main__":
    main()
