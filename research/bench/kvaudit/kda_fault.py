"""Which KDA kernel faults on one long call: each kernel alone, random inputs, CUDA_LAUNCH_BLOCKING=1.

    python -m research.bench.kvaudit.kda_fault --tokens 262080 262208 524288      # 1 GPU

kda_probe 7525 died with an illegal memory access at 524,288 tokens x 64 heads x 128 (4.3e9 elements, past int32).
Every (kernel, T) case runs in its own subprocess, because a fault poisons the CUDA context. The case reports ok, or
the exit code and the last error line.

- `flashinfer`: FlashInfer persistent `fwd` (ours, the design row)
- `flashkda`: FlashKDA `_flashkda_C.fwd` (vLLM on SM100)
- `triton_l2norm`: the fork's `l2norm_fwd`, the first kernel of `chunk_kda_with_fused_gate`
- `triton_chunk`: the fork's `chunk_kda_with_fused_gate`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

KERNELS = ("flashinfer", "flashkda", "triton_l2norm", "triton_chunk")


def one(kernel: str, T: int, H: int = 64, D: int = 128) -> None:
    import torch

    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.bfloat16)  # noqa: E731
    q, k, v, g = (r(1, T, H, D) for _ in range(4))
    beta = r(1, T, H)
    a_log = torch.randn(H, generator=gen, device=dev) * 0.5
    dt_bias = torch.randn(H * D, generator=gen, device=dev) * 0.5
    lb, scale = -5.0, D ** -0.5
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    zero = lambda: torch.zeros(1, H, D, D, dtype=torch.float32, device=dev)  # noqa: E731
    t0 = time.time()
    with torch.inference_mode():
        if kernel == "flashinfer":
            import flashinfer.kda_kernels.persistent.forward as fip

            o, st = torch.empty_like(q), zero()
            fip.fwd(q, k, v, g, beta, scale, o, a_log, dt_bias, lb, initial_state=st, final_state=st, cu_seqlens=cu,
                    plan=fip.make_plan([0, T], H, dev))
            fin = st
        elif kernel == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, 1), dtype=torch.uint8, device=dev)
            o, fin = torch.empty_like(q), zero()
            torch.ops._flashkda_C.fwd(q, k, v, g, beta, scale, o, ws, a_log, dt_bias.view(-1, D), lb, zero(), fin, cu,
                                      None, None)
        elif kernel == "triton_l2norm":
            from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

            o = l2norm_fwd(q.contiguous())
            fin = zero()
        else:
            from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate

            o, fin = chunk_kda_with_fused_gate(q=q, k=k, v=v, raw_g=g, beta=torch.sigmoid(beta.float()), A_log=a_log,
                                               g_bias=dt_bias, scale=scale, initial_state=zero(),
                                               output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                                               safe_gate=True, lower_bound=lb)
        torch.cuda.synchronize()
        # a kernel that silently skips the far half shows as zero/NaN output there
        tail = o.reshape(T, -1)[T // 2:]
        res = {"ok": True, "seconds": round(time.time() - t0, 2), "finite": bool(torch.isfinite(o).all()),
               "tail_zero_rows": int((tail.float().abs().amax(dim=-1) == 0).sum()),
               "state_finite": bool(torch.isfinite(fin).all())}
    print("RESULT " + json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="+", default=[262080, 262208, 524288])
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS))
    ap.add_argument("--one", nargs=2, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        one(a.one[0], int(a.one[1]))
        return
    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
    out = []
    for T in a.tokens:
        for kernel in a.kernels:
            p = subprocess.run([sys.executable, "-m", "research.bench.kvaudit.kda_fault", "--one", kernel, str(T)], env=env,
                               capture_output=True, text=True, timeout=600)
            res = next((json.loads(line[7:]) for line in p.stdout.splitlines() if line.startswith("RESULT ")), None)
            if res is None:
                err = [line for line in (p.stderr + p.stdout).splitlines() if "Error" in line or "error" in line]
                res = {"ok": False, "rc": p.returncode, "error": err[-1][:300] if err else (p.stderr[-300:])}
            res.update(kernel=kernel, T=T, elements=T * 64 * 128)
            out.append(res)
            print(f"[kda-fault] {json.dumps(res)}", flush=True)
    from research.bench.kvaudit import common as C

    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / f"kda_fault_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
