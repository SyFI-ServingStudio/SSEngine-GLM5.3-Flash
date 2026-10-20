"""Every block-FP8 MoE backend of the vLLM fork on one GLM-5.3-Flash FP8 MoE layer (1 GPU): time and accuracy.

    python -m research.bench.fp8_moe_backends [--backends flashinfer_trtllm,...] [--tokens 256,1024,...] [--layer 4]

Each backend (`KernelConfig.moe_backend`, oracle/fp8.py) runs in its own process: the decoder layer `--layer` of the FP8
checkpoint is built as a stage builds it (`stage.build_vllm_config`, `pp_patch.stage_scope`, the layer's real weights,
`process_weights_after_loading`, the serving MoE buckets), with only `moe_backend` changed. The timed call is the
layer's whole MoE as served (`Glm5NextMoE.forward`: router GEMM, routing, activation quantization, routed experts,
combine, and the shared expert), on BF16 hidden rows.

Routing: served histograms (`moe_tuning.profile_for`, the folded experts_stage*.jsonl of `--routing`, a session run
of the engine) through `moe_tuning.forced_logits` at each token count. FlashInfer backends are autotuned once per shape
on that routing (as `benchmarks/tune_tactics.py` does), then every backend is timed under a CUDA graph (eager when it
cannot be captured: the column says which), median of `--iters` replays after warm-up.

Accuracy: at `--check-tokens` rows of N(0, 1) hidden, routed by the layer's real router, each backend's output against
an FP32 reference built from the checkpoint's FP8 weights dequantized by their 128 x 128 block scales (same routing:
sigmoid + correction bias, top 8, renormalized, x routed_scaling_factor; SwiGLU clamp at swiglu_limit; plus the
shared expert): relative Frobenius error and minimum row cosine. The reference does not quantize activations; the
kernels quantize them to FP8 per 128 channels, so ~3e-2 relative error is the expected floor.

TFLOPS counts the routed and shared experts' GEMMs (2 x tokens x 9 x (4096 x 4096 + 2048 x 4096)).
Writes research/measured/fp8_moe/<job>.json and a markdown table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = ("flashinfer_trtllm", "flashinfer_cutlass", "deep_gemm", "cutlass", "triton")
TOKENS = (256, 1024, 2048, 4096, 8192, 16384)
# --backends auto with SSENGINE_CHECKPOINT=<NVFP4 snapshot> --check-tokens 0: the NVFP4 MoE as served, same scope
OUT = Path(__file__).resolve().parents[1] / "measured" / "fp8_moe"
ROUTING = Path(os.environ.get("SSENGINE_ROUTING_RUN",
                              "/raid/kanzhu/ServingStudio/pp-impl/old-wt/m5-routing/measured/runs_m5r/"
                              "experts_good_n400_7801"))
H, I, TOPK = 4096, 2048, 8


def flops(tokens: int) -> float:
    return 2.0 * tokens * (TOPK + 1) * (H * 2 * I + I * H)


def build(backend: str, layer: int, max_tokens: int):
    import torch
    from ssengine.config import CHECKPOINT
    from ssengine.model import pp_patch
    from ssengine.model.stage import build_vllm_config, init_vllm_distributed
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import init_workspace_manager

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    vc = build_vllm_config(CHECKPOINT, max_tokens, moe_backend=backend)
    ctx = set_current_vllm_config(vc)
    ctx.__enter__()
    import vllm.ir
    vc.kernel_config.ir_op_priority.set_default()
    vllm.ir.set_default_torch_wrap(vc.compilation_config.ir_enable_torch_wrap)
    init_vllm_distributed(dev)
    init_workspace_manager(dev)
    layers = range(layer, layer + 1)
    with set_default_torch_dtype(vc.model_config.dtype), torch.device(dev):
        with pp_patch.stage_scope(layers, False, False):
            model = initialize_model(vllm_config=vc)
    model.load_weights(pp_patch.stage_weights(Path(CHECKPOINT), layers, False, False))
    process_weights_after_loading(model, vc.model_config, dev)
    pp_patch.install_moe_buckets()
    return vc, model.language_model.model.layers[layer].mlp, dev


def reference(layer: int, x, logits, cfg) -> "torch.Tensor":
    """FP32 MoE of `layer` from the checkpoint's FP8 weights, dequantized by block."""
    import torch
    from safetensors import safe_open
    from ssengine.config import CHECKPOINT

    ck = Path(CHECKPOINT)
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp."
    dev = x.device

    def get(name):
        with safe_open(ck / wm[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def deq(base):
        w, s = get(base + ".weight"), get(base + ".weight_scale_inv")
        s = s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        return (w.float() * s).to(dev)

    lim = cfg.swiglu_limit

    def mlp(base, rows):
        g, u, d = deq(base + ".gate_proj"), deq(base + ".up_proj"), deq(base + ".down_proj")
        a = rows @ g.T
        b = rows @ u.T
        h = torch.nn.functional.silu(a.clamp(max=lim)) * b.clamp(-lim, lim)
        return h @ d.T

    xf = x.float()
    bias = get(pre + "gate.e_score_correction_bias").float().to(dev)
    scores = torch.sigmoid(logits.float())
    ids = torch.topk(scores + bias, TOPK, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True) * cfg.routed_scaling_factor
    out = torch.zeros_like(xf)
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        out.index_add_(0, rows, mlp(pre + f"experts.{e}", xf[rows]) * w[rows, slot, None])
    return out + mlp(pre + "shared_experts", xf)


def one(backend: str, tokens: list[int], layer: int, iters: int, check_tokens: int, out: Path) -> dict:
    import torch
    from flashinfer.autotuner import autotune
    from ssengine.config import CHECKPOINT, checkpoint_quantization
    from ssengine.model import moe_tuning
    from vllm.forward_context import set_forward_context

    res = {"backend": backend, "layer": layer}
    t0 = time.time()
    vc, moe, dev = build(backend, layer, max(tokens))
    res["build_s"] = round(time.time() - t0, 1)
    res["experts_impl"] = type(getattr(moe.experts, "quant_method", moe.experts)).__name__
    qm = getattr(moe.experts, "quant_method", None)
    for attr in ("fp8_backend", "moe_backend", "backend"):
        if qm is not None and hasattr(qm, attr):
            res["selected"] = str(getattr(qm, attr))
            break
    cfg = vc.model_config.hf_text_config
    gate = moe.gate
    fixed = {}

    def gate_forward(x):
        return gate._return(fixed["logits"][: x.shape[0]])

    gate.forward = gate_forward
    profile = moe_tuning.routing_profile([ROUTING])
    T_max = max(tokens + [check_tokens])
    fixed["logits"] = torch.empty(T_max, cfg.n_routed_experts, dtype=gate.out_dtype, device=dev)
    xbuf = torch.randn(T_max, H, dtype=torch.bfloat16, device=dev)

    def call(T):
        with set_forward_context(None, vc, num_tokens=T):
            return moe(xbuf[:T])

    with torch.inference_mode():
        # accuracy, real router on N(0, 1) rows (FP8 checkpoint only: the reference dequantizes FP8 blocks)
        T = check_tokens or 1
        fixed["logits"][:T] = (xbuf[:T].float() @ gate.weight.float().T).to(fixed["logits"].dtype)
        got = call(T).float()
        ref = reference(layer, xbuf[:T], fixed["logits"][:T], cfg) if check_tokens else got
        d = got - ref
        res["check"] = None if not check_tokens else {"tokens": T, "rel_fro": (d.norm() / ref.norm()).item(),
                        "min_cos": torch.nn.functional.cosine_similarity(got, ref, dim=1).min().item(),
                        "finite": bool(torch.isfinite(got).all().item())}
        out.write_text(json.dumps(res, indent=1))
        print(f"[moe] {backend} check {res['check']}", flush=True)
        res["checkpoint_quant"] = checkpoint_quantization(CHECKPOINT)
        res["times"] = {}
        for T in tokens:
            fixed["logits"][:T] = moe_tuning.forced_logits(moe_tuning.profile_for(profile, T), T,
                                                           fixed["logits"].dtype, dev)
            row = {}
            try:
                with moe_tuning.real_routing(profile), autotune(True):
                    call(T)
                torch.cuda.synchronize()
                for _ in range(3):
                    call(T)
                torch.cuda.synchronize()
                mode = "graph"
                try:
                    g = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        call(T)
                        with torch.cuda.graph(g, stream=s):
                            call(T)
                    torch.cuda.current_stream().wait_stream(s)
                    run = g.replay
                except Exception as e:  # noqa: BLE001 - not capturable: time it eagerly
                    mode, row["graph_error"] = "eager", f"{type(e).__name__}: {str(e)[:200]}"
                    torch.cuda.synchronize()
                    run = lambda: call(T)  # noqa: E731
                for _ in range(5):
                    run()
                ms = []
                for _ in range(iters):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    run()
                    b.record()
                    b.synchronize()
                    ms.append(a.elapsed_time(b))
                med = statistics.median(ms)
                row.update(mode=mode, ms=round(med, 4), tflops=round(flops(T) / med / 1e9, 1))
            except Exception as e:  # noqa: BLE001 - a backend that cannot run this shape
                row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            res["times"][T] = row
            out.write_text(json.dumps(res, indent=1))      # partial results survive a timeout
            print(f"[moe] {backend} T={T} {row}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--check-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=330, help="per backend, s")
    ap.add_argument("--one", help="internal: run one backend, write JSON here")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    if a.one:
        b = a.backends
        try:
            res = one(b, tokens, a.layer, a.iters, a.check_tokens, Path(a.one))
        except Exception as e:  # noqa: BLE001
            import traceback
            res = json.loads(Path(a.one).read_text()) if Path(a.one).exists() else {"backend": b}
            res |= {"error": f"{type(e).__name__}: {str(e)[:500]}", "trace": traceback.format_exc()[-3000:]}
        Path(a.one).write_text(json.dumps(res, indent=1))
        return
    job = os.environ.get("SLURM_JOB_ID", "local")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for b in a.backends.split(","):
        part = OUT / f"{job}_{b}.json"
        t0 = time.time()
        try:
            rc = subprocess.run([sys.executable, "-m", "research.bench.fp8_moe_backends", "--one", str(part),
                                 "--backends", b, "--tokens", a.tokens, "--layer", str(a.layer), "--iters",
                                 str(a.iters), "--check-tokens", str(a.check_tokens)], timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        r = json.loads(part.read_text()) if part.exists() else {"backend": b, "error": f"rc {rc}"}
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"[moe] {b} done rc={rc} in {r['wall_s']} s: {json.dumps(r)[:600]}", flush=True)
    (OUT / f"{job}.json").write_text(json.dumps(results, indent=1))
    print("\n| backend | impl | rel err | min cos | " + " | ".join(f"{t} tok ms (TF)" for t in tokens) + " |")
    print("|---" * (4 + len(tokens)) + "|")
    for r in results:
        c = r.get("check") or {}
        cells = []
        for t in tokens:
            x = (r.get("times") or {}).get(str(t)) or (r.get("times") or {}).get(t) or {}
            cells.append(f"{x['ms']:.3f} ({x['tflops']:.0f}){'' if x.get('mode') == 'graph' else ' e'}" if "ms" in x
                         else "fail")
        err = f"{c['rel_fro']:.4f}" if c else r.get("error", "")[:60]
        cos = f"{c['min_cos']:.4f}" if c else ""
        print(f"| {r['backend']} | {r.get('experts_impl', '')} | {err} | {cos} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
