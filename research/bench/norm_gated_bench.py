"""KDA's output gated RMSNorm (fla `layer_norm_gated_fwd_kernel`, sigmoid gate, per 128-wide head) at the served
shape, as vLLM launches it (16 rows per program, 8 warps) vs other rows-per-program / warp counts (1 GPU).

  python -m research.bench.norm_gated_bench [--tokens 4096] [--heads 64] [--reps 200]

Each variant must be bitwise equal to vLLM's launch (the per-row reduction is the same); times by CUDA events over
`reps` back-to-back launches, interleaved by variant, with the SM clock.
"""

import argparse
import json
import statistics


def main():
    import torch
    from vllm.third_party.flash_linear_attention.ops.kda import layer_norm_gated_fwd_kernel, rms_norm_gated

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    T, H, D = a.tokens, a.heads, 128
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, T, H, D, generator=gen, device=dev).to(torch.bfloat16)
    g = torch.randn(T, H * D, generator=gen, device=dev).to(torch.bfloat16).reshape(-1, H, D)
    w = (1 + 0.1 * torch.randn(D, generator=gen, device=dev)).to(torch.bfloat16)
    eps = 1e-5
    ref = rms_norm_gated(x, g, w, None, "sigmoid", eps=eps)

    def launch(bt, warps):
        x2 = x.reshape(-1, D)
        y = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=dev)
        grid = (triton_cdiv(x2.shape[0], bt),)
        layer_norm_gated_fwd_kernel[grid](x=x2, g=g, y=y, w=w, b=None, residual=None, residual_out=None, mean=None,
                                          rstd=rstd, eps=eps, T=x2.shape[0], H=H, g_stride_n=g.stride(0), D=D, BD=D,
                                          BT=bt, ACTIVATION="sigmoid", IS_RMS_NORM=True, num_warps=warps)
        return y

    variants = {"vllm": lambda: rms_norm_gated(x, g, w, None, "sigmoid", eps=eps)}
    for bt in (16, 32, 64, 128):
        for warps in (2, 4, 8):
            variants[f"bt{bt}_w{warps}"] = (lambda bt=bt, warps=warps: launch(bt, warps))
    ok = {}
    for k, f in variants.items():
        ok[k] = bool(torch.equal(f().reshape(ref.shape), ref))
    times = {k: [] for k in variants}
    for _ in range(5):
        for k, f in variants.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(20_000_000)
            e0.record()
            for _ in range(a.reps):
                f()
            e1.record()
            e1.synchronize()
            times[k].append(1e3 * e0.elapsed_time(e1) / a.reps)
    res = {k: {"us": round(statistics.median(v), 2), "bitwise": ok[k]} for k, v in times.items()}
    base = res["vllm"]["us"]
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["us"]):
        print(f"[norm_gated] {k:12} {v['us']:7.2f} us  x{v['us'] / base:.3f}  bitwise {v['bitwise']}", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1) + "\n")


def triton_cdiv(a, b):
    return -(-a // b)


if __name__ == "__main__":
    main()
