"""M1 correctness (`2_kernels.md` §5): smallest and largest row of each of the 35 configurations against FP32.

Inputs come from the ServingStudioSim profiler runners' builders (frozen `7bb224a6`, importable as `profiling`
from `third_party/sim_7bb224a6`), references from the runners and the vLLM fork's kernel tests. This file only
writes the comparisons. Where the fork calls a different callable than the profiler row (configs 4, 5, 7, 33;
see `measured/kernels_identity.md`) both are checked when both are importable.

    python -m bench.kernels_correctness --out measured/kernels_correctness.jsonl [--configs 1,2,...]

One JSON line per (config, row, callable): max abs error, error relative to the reference's max, RMS error ratio,
violations of `|a - r| <= atol + rtol * |r|`, the tolerance and where it comes from, pass.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
import traceback
from pathlib import Path

from bench.common import HANDOFF, MAIN, PROFILER

SPECS = HANDOFF / "data" / "kernel_specs.jsonl"
VLLM_TESTS = MAIN / "third_party" / "vllm"
# neither tree is an installed package: the profiler's `profiling` and the fork's `tests` import from their roots
for path in (str(PROFILER), str(VLLM_TESTS)):
    if path not in sys.path:
        sys.path.append(path)

import torch  # noqa: E402

SAMPLE_ROWS = 1024


# ---------------------------------------------------------------- configurations


def load_configs(path: Path = SPECS) -> list[dict]:
    """The 35 configurations of `2_kernels.md` §2 in spec-file order, with their smallest and largest rows."""
    groups: dict = collections.OrderedDict()
    for line in path.open():
        row = json.loads(line)
        positions = tuple(sorted({p.split(": ")[1] for p in row["positions"]}))
        groups.setdefault((row["kind"], row["backend"], positions), []).append(row)

    def size(spec: dict) -> int:
        out = 1
        for value in spec.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out *= max(1, int(value))
        return out

    configs = []
    for index, ((kind, backend, positions), rows) in enumerate(groups.items(), start=1):
        rows = sorted(rows, key=lambda r: size(r["spec"]))
        configs.append(
            {
                "config": index,
                "kind": kind,
                "backend": backend,
                "positions": list(positions),
                "rows": len(rows),
                "smallest": rows[0]["spec"],
                "largest": rows[-1]["spec"],
            }
        )
    return configs


# ---------------------------------------------------------------- metrics


def compare(actual, reference, *, atol: float, rtol: float, source: str) -> dict:
    a = actual.detach().float()
    r = reference.detach().float()
    finite = bool(torch.isfinite(a).all())
    diff = (a - r).abs()
    ref_max = float(r.abs().max()) if r.numel() else 0.0
    ref_rms = float(r.square().mean().sqrt()) if r.numel() else 0.0
    violations = int((diff > atol + rtol * r.abs()).sum())
    return {
        "numel": r.numel(),
        "max_abs_err": float(diff.max()) if r.numel() else 0.0,
        "ref_max_abs": ref_max,
        "err_vs_ref_max": float(diff.max()) / ref_max if ref_max else 0.0,
        "rms_err_ratio": float(diff.square().mean().sqrt()) / ref_rms if ref_rms else 0.0,
        "violations": violations,
        "atol": atol,
        "rtol": rtol,
        "tolerance_source": source,
        "finite": finite,
        "pass": finite and violations == 0,
    }


def exact(actual, reference, *, source: str) -> dict:
    a, r = actual.detach(), reference.detach()
    mismatches = int((a != r).sum()) if a.shape == r.shape else -1
    return {
        "numel": r.numel(),
        "mismatches": mismatches,
        "tolerance_source": source,
        "pass": mismatches == 0,
    }


def merge(*results: dict) -> dict:
    """Several outputs of one call: pass only if every output passes."""
    out = {"outputs": list(results), "pass": all(r["pass"] for r in results)}
    out["max_abs_err"] = max((r.get("max_abs_err", 0.0) for r in results), default=0.0)
    return out


def sample_rows(m: int, count: int = SAMPLE_ROWS, seed: int = 0):
    if m <= count:
        return torch.arange(m, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randperm(m - 2, generator=g)[: count - 2] + 1
    return torch.cat([torch.tensor([0, m - 1]), picks]).sort().values.cuda()


def fp32_highest():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ---------------------------------------------------------------- GEMMs

BF16_GEMM_TOL = dict(atol=2e-1, rtol=2e-2, source="vllm tests/kernels/test_flashinfer_bf16_gemm.py")


def check_single_gemm_torch(spec: dict, config: dict) -> list[dict]:
    import torch.nn.functional as F

    m, n, k = spec["m"], spec["n"], spec["k"]
    # profiling/runners/gemm/torch.py profile_single_gemm_linear
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    out = F.linear(x, w)
    rows = sample_rows(m)
    ref = x[rows].float() @ w.float().T
    results = [{"callable": "torch.nn.functional.linear (runner layout)", **compare(out[rows], ref, **_tol(BF16_GEMM_TOL))}]
    if config["config"] == 3:
        # The fork's f_b/g_b input is a column view of the KDA in_proj output (row stride 24896).
        base = torch.randn(m, 24896, dtype=torch.bfloat16, device="cuda")
        xv = base[:, 24896 - 2 * 128 : 24896 - 128]
        out_v = F.linear(xv, w)
        ref_v = xv[rows].float() @ w.float().T
        results.append(
            {"callable": "torch.nn.functional.linear (fork layout: row stride 24896)",
             **compare(out_v[rows], ref_v, **_tol(BF16_GEMM_TOL))}
        )
    return results


def _tol(t: dict) -> dict:
    return {"atol": t["atol"], "rtol": t["rtol"], "source": t["source"]}


def check_gemm_fp32_output(spec: dict, config: dict) -> list[dict]:
    from profiling.runners.gemm import gemm_fp32_output_torch_cublas as mod

    fp32_highest()
    shape = mod._validate_args(spec["m"], spec["n"], spec["k"], spec["input_dtype"])
    launch = mod._prepare(torch, shape)
    out = launch.run()
    rows = sample_rows(shape.m)
    ref = (launch.input_tensor[rows].double() @ launch.weight_transposed.double()).float()
    tol = dict(atol=0.002, rtol=0.01, source="runner gemm_fp32_output_torch_cublas._check_output")
    return [{"callable": "torch.mm" + (" fp32" if launch.fp32_input else " out_dtype=fp32"),
             **compare(out[rows], ref, **tol)}]


def check_batched_gemm(spec: dict, config: dict) -> list[dict]:
    from profiling.runners.gemm import batched_gemm as mod

    layout = mod._NO_ROPE_UNPADDED_LAYOUT
    builder = (
        mod._build_layout_q_absorb_operands
        if config["backend"] == "torch_mla_q_absorb_no_rope"
        else mod._build_layout_v_up_operands
    )
    ops = builder(torch, layout, num_batches=spec["num_batches"], m=spec["m"], torch_dtype=torch.bfloat16, device="cuda")
    torch.bmm(ops.lhs, ops.rhs, out=ops.out)
    rows = sample_rows(spec["m"])
    ref = torch.bmm(ops.lhs[:, rows].float(), ops.rhs.float())
    tol = dict(atol=2e-1, rtol=2e-2, source="runner batched_gemm check (rtol 2e-2, atol 2e-1)")
    results = [{"callable": "torch.bmm (runner layout)", **compare(ops.out[:, rows], ref, **tol)}]
    if config["backend"] == "torch_mla_q_absorb_no_rope":
        # The fork writes into a token-major [T, N, L] buffer (ldc = N * L).
        out_tm = torch.empty(spec["m"], spec["num_batches"], ops.rhs.shape[-1], dtype=torch.bfloat16, device="cuda")
        torch.bmm(ops.lhs, ops.rhs, out=out_tm.transpose(0, 1))
        results.append({"callable": "torch.bmm (fork layout: token-major out)",
                        **compare(out_tm.transpose(0, 1)[:, rows], ref, **tol)})
    return results


def _nvfp4_test_utils():
    from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes, dequantize_nvfp4_to_dtype

    return break_fp4_bytes, dequantize_nvfp4_to_dtype


def _ref_nvfp4_quant(x, global_scale):
    # tests/kernels/quantization/test_nvfp4_quant.py:ref_nvfp4_quant (imported there with pytest fixtures).
    from tests.kernels.quantization.test_nvfp4_quant import ref_nvfp4_quant

    return ref_nvfp4_quant(x, global_scale)


def check_nvfp4_quant(spec: dict, config: dict) -> list[dict]:
    from vllm import _custom_ops as ops
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales

    from profiling.runners.elementwise import nvfp4_quant as mod

    break_fp4_bytes, _ = _nvfp4_test_utils()
    m, h = spec["num_tokens"], spec["hidden_size"]
    src = torch.randn((m, h), dtype=torch.bfloat16, device="cuda")
    swizzled = spec["scale_format"] == "swizzled_e4m3"
    if swizzled:
        gs = mod._calibrated_global_scale(src)
        packed, scales = ops.scaled_fp4_quant(src, gs, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl")
        got_scales = recover_swizzled_scales(scales.view(torch.float8_e4m3fn), m, h)
        name = 'scaled_fp4_quant(swizzled, backend="flashinfer-cutedsl")'
    else:
        gs = torch.ones((), dtype=torch.float32, device="cuda")
        packed, scales = ops.scaled_fp4_quant(src, gs, is_sf_swizzled_layout=False)
        got_scales = scales.view(torch.float8_e4m3fn).reshape(m, h // 16).float()
        name = "scaled_fp4_quant(linear)"
    ref_vals, ref_scales = _ref_nvfp4_quant(src, gs.float().reshape(()))
    got_vals = break_fp4_bytes(packed.view(torch.uint8), torch.float32)
    source = "vllm tests/kernels/quantization/test_nvfp4_quant.py (exact)"
    return [{"callable": name, **merge(exact(got_vals, ref_vals, source=source),
                                       exact(got_scales, ref_scales.reshape(m, -1), source=source))}]


def check_fp4_gemm(spec: dict, config: dict) -> list[dict]:
    from profiling.runners.gemm import flashinfer_fp4 as mod

    _, dequant = _nvfp4_test_utils()
    m, n, k = spec["m"], spec["n"], spec["k"]
    ops_ = mod.prepare_operands(m, n, k, seed=0)
    out = mod.make_apply(ops_, n)()
    # prepare_operands' tensors, regenerated from the same seed, for the global scales.
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=g)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=g)
    x_gs = 448.0 * 6.0 / x.float().abs().amax()
    w_gs = 448.0 * 6.0 / w.float().abs().amax()
    del x, w
    rows = sample_rows(m)
    xd = dequant(ops_.x_fp4, ops_.x_blockscale, x_gs, torch.float32, "cuda", is_sf_128x4_layout=True)[rows]
    wd = dequant(ops_.weight, ops_.weight_scale, w_gs, torch.float32, "cuda", is_sf_128x4_layout=True)
    ref = xd @ wd.T
    tol = dict(atol=1e-1, rtol=1e-1, source="vllm tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py")
    return [{"callable": 'flashinfer_scaled_fp4_mm(backend="cute-dsl")', **compare(out[rows], ref, **tol)}]


# ---------------------------------------------------------------- norms


def check_q_kv_rms_norm(spec: dict, config: dict) -> list[dict]:
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    from profiling.runners.attention import q_kv_rms_norm_vllm_triton as mod

    t, qd, kvd, eps = spec["num_tokens"], spec["q_dim"], spec["kv_dim"], spec["rms_eps"]
    g = torch.Generator(device="cuda").manual_seed(43)
    qr, kv = torch.randn((t, qd + kvd), dtype=torch.bfloat16, device="cuda", generator=g).split([qd, kvd], dim=-1)
    qw = torch.randn(qd, dtype=torch.bfloat16, device="cuda", generator=g)
    kvw = torch.randn(kvd, dtype=torch.bfloat16, device="cuda", generator=g)
    a_q, a_kv = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)
    tol = dict(atol=1e-2, rtol=1e-2, source="vllm tests/kernels/core/test_fused_q_kv_rmsnorm.py")
    return [{"callable": "vllm.models.common.ops.fused_q_kv_rmsnorm",
             **merge(compare(a_q, mod._reference(torch, qr, qw, eps), **tol),
                     compare(a_kv, mod._reference(torch, kv, kvw, eps), **tol))}]


def check_rms_norm(spec: dict, config: dict) -> list[dict]:
    from vllm import _custom_ops as ops

    m, h = spec["m"], spec["hidden"]
    eps = 1e-5
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((m, h), dtype=torch.bfloat16, device="cuda", generator=g)
    w = torch.randn((h,), dtype=torch.bfloat16, device="cuda", generator=g)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, eps)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()
    tol = dict(atol=1e-2, rtol=1e-2, source="vllm tests/kernels/core/test_layernorm.py")
    return [{"callable": "torch.ops._C.rms_norm", **compare(out, ref, **tol)}]


# ---------------------------------------------------------------- mHC

MHC_BF16_TOL = dict(atol=1.6e-2, rtol=1e-2, source="vllm tests/kernels/test_mhc_kernels.py (bf16 residual / y)")
MHC_MIX_TOL = dict(atol=1e-3, rtol=1e-3, source="vllm tests/kernels/test_mhc_kernels.py (mixes, deep_gemm path)")
MHC_RUNNER_TOL = dict(atol=0.05, rtol=0.02, source="runner mhc/_common.assert_outputs_close")


def _mhc_compare(actual: tuple, expected: tuple, kinds: tuple[str, ...]) -> dict:
    lib = [compare(a, e, **_tol(MHC_MIX_TOL if kind == "mix" else MHC_BF16_TOL)) for a, e, kind in zip(actual, expected, kinds)]
    runner = [compare(a, e, **_tol(MHC_RUNNER_TOL)) for a, e in zip(actual, expected)]
    out = merge(*lib)
    out["pass_runner_tol"] = all(r["pass"] for r in runner)
    return out


def _mhc_shape(spec):
    from profiling.runners.mhc._common import Shape

    return Shape(spec["num_tokens"], spec["hidden_size"], spec["hc_mult"])


def check_mhc_pre(spec: dict, config: dict) -> list[dict]:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang

    from profiling.runners.mhc import _common as c
    from profiling.runners.mhc import mhc_pre_rms_norm_vllm_tilelang as mod

    fp32_highest()
    inputs = c.prepare_common(torch, _mhc_shape(spec))
    actual = mod._Launch(mhc_pre_tilelang, inputs).run()
    expected = c.reference_pre(torch, inputs)
    return [{"callable": "mhc_pre_tilelang", **_mhc_compare(actual, expected, ("mix", "mix", "bf16"))}]


def _post_pre_expected(inputs, x, post_mix, comb_mix):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    from profiling.runners.mhc import _common as c

    residual = mhc_post_torch(x, inputs.residual, post_mix, comb_mix)
    nxt = c.CommonInputs(residual, inputs.fn, inputs.hc_scale, inputs.hc_base, inputs.norm_weight)
    return (residual, *c.reference_pre(torch, nxt))


def check_mhc_fused_tilelang(spec: dict, config: dict) -> list[dict]:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang, mhc_post_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    from profiling.runners.mhc import _common as c
    from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_vllm_tilelang as mod

    fp32_highest()
    shape = _mhc_shape(spec)
    inputs = c.prepare_common(torch, shape)
    x = torch.randn((shape.num_tokens, shape.hidden_size), dtype=torch.bfloat16).cuda()
    post_mix, comb_mix, _ = c.reference_pre(torch, inputs)
    actual = mod._Launch(mhc_fused_post_pre_tilelang, x, post_mix, comb_mix, inputs).run()
    expected = _post_pre_expected(inputs, x, post_mix, comb_mix)
    results = [{"callable": "mhc_fused_post_pre_tilelang",
                **_mhc_compare(actual, expected, ("bf16", "mix", "mix", "bf16"))}]
    # The fork's final boundary (config 33's position): mhc_post_tilelang, then mean over streams.
    post = mhc_post_tilelang(x, inputs.residual, post_mix, comb_mix)
    ref = mhc_post_torch(x, inputs.residual, post_mix, comb_mix)
    results.append({"callable": "mhc_post_tilelang (fork final boundary)",
                    **compare(post, ref.float(), **_tol(MHC_BF16_TOL))})
    return results


def check_mhc_mega(spec: dict, config: dict) -> list[dict]:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    from vllm.utils.deep_gemm import mega_mhc

    from profiling.runners.mhc import _common as c
    from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted as mod
    from profiling.runners.mhc import mhc_fused_post_pre_rms_norm_vllm_tilelang as til

    fp32_highest()
    shape = mod.validate_args(spec["num_tokens"], spec["hidden_size"], spec["hc_mult"], spec["hidden_dtype"])
    inputs, x = mod._prepare(torch, shape)
    post_mix, comb_mix, _ = c.reference_pre(torch, inputs)
    post_mix, comb_mix = post_mix.contiguous(), comb_mix.contiguous()
    expected = _post_pre_expected(inputs, x, post_mix, comb_mix)
    kinds = ("bf16", "mix", "mix", "bf16")
    actual = mod._Launch(mega_mhc, torch, x, post_mix, comb_mix, inputs).run()
    results = [{"callable": "deep_gemm.mega_mhc (non-shifted; profiler row)", **_mhc_compare(actual, expected, kinds)}]
    actual = til._Launch(mhc_fused_post_pre_tilelang, x, post_mix, comb_mix, inputs).run()
    results.append({"callable": "mhc_fused_post_pre_tilelang (fork at this position)",
                    **_mhc_compare(actual, expected, kinds)})
    return results


# ---------------------------------------------------------------- DSA


def calc_diff(x, y) -> float:
    # vllm/utils/deep_gemm.py:calc_diff, used by tests/kernels/attention/test_deepgemm_attention.py (< 1e-3).
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denominator) if denominator > 0 else 0.0


def check_mqa_logits(spec: dict, config: dict) -> list[dict]:
    from profiling.runners.attention import dsa_mqa_logits_prefill as mod

    q_n, k_n, h, d = spec["num_queries"], spec["num_keys"], spec["num_heads"], spec["head_dim"]
    _, deep_gemm = mod._load_deepgemm_backend()
    ops_ = mod._build_operands(torch, num_queries=q_n, num_keys=k_n, num_heads=h, head_dim=d, device="cuda")
    # The runner's periodic values are replaced by random ones (same shapes, dtypes and spans).
    g = torch.Generator(device="cuda").manual_seed(7)
    ops_.q.copy_(torch.randn(ops_.q.shape, device="cuda", generator=g).to(torch.float8_e4m3fn))
    ops_.k.copy_(torch.randn(ops_.k.shape, device="cuda", generator=g).to(torch.float8_e4m3fn))
    ops_.k_scale.copy_(0.5 + torch.rand(ops_.k_scale.shape, device="cuda", generator=g))
    ops_.weights.copy_(torch.randn(ops_.weights.shape, device="cuda", generator=g) / h)
    out = mod._prepare_deepgemm_call(deep_gemm, ops_)()
    rows = sample_rows(q_n, 256)
    kf = ops_.k.float()
    worst, diffs, viol, n = 0.0, [], 0, 0
    got_all, ref_all = [], []
    for chunk in rows.split(16):
        dots = ops_.q[chunk].float() @ kf.T  # [c, H, K]
        ref = (dots.relu() * ops_.weights[chunk].unsqueeze(-1)).sum(1) * ops_.k_scale.unsqueeze(0)
        mask = ops_.valid_mask[chunk]
        got_all.append(out[chunk][:, :k_n][mask])
        ref_all.append(ref[mask])
    got, ref = torch.cat(got_all), torch.cat(ref_all)
    result = compare(got, ref, atol=1e-2, rtol=1e-2, source="calc_diff < 1e-3 (vllm test_deepgemm_attention.py)")
    result["calc_diff"] = calc_diff(got, ref)
    result["pass"] = result["finite"] and result["calc_diff"] < 1e-3
    return [{"callable": "deep_gemm.fp8_fp4_mqa_logits (clean_logits=False)", **result}]


def _topk_check(logits, row_ends, out, top_k: int) -> dict:
    """Every row: the selected values equal torch.topk's, as multisets (ties aside); short rows: 0..span-1."""
    bad, checked = 0, 0
    q = logits.shape[0]
    for chunk in torch.arange(q, device="cuda").split(64):
        spans = row_ends[chunk].long()
        got = out[chunk].long()
        width = int(spans.max())
        vals = logits[chunk, :width]
        cols = torch.arange(width, device="cuda")
        masked = vals.masked_fill(cols.unsqueeze(0) >= spans.unsqueeze(1), float("-inf"))
        long_rows = spans > top_k
        if bool(long_rows.any()):
            g = got[long_rows]
            in_range = ((g >= 0) & (g < spans[long_rows].unsqueeze(1))).all(1)
            picked = torch.gather(masked[long_rows], 1, g.clamp(0, width - 1)).sort(1).values
            expected = masked[long_rows].topk(top_k, dim=1).values.sort(1).values
            ok = in_range & (picked == expected).all(1)
            bad += int((~ok).sum())
        short = ~long_rows
        if bool(short.any()):
            natural = torch.arange(top_k, device="cuda").unsqueeze(0)
            exp = torch.where(natural < spans[short].unsqueeze(1), natural, torch.full_like(natural, -1))
            ok = (got[short].sort(1).values == exp.sort(1).values).all(1)
            bad += int((~ok).sum())
        checked += len(chunk)
    return {"rows_checked": checked, "bad_rows": bad, "tolerance_source": "2_kernels.md §5: set equals torch.topk",
            "pass": bad == 0}


def check_topk(spec: dict, config: dict) -> list[dict]:
    from dataclasses import replace

    from profiling.runners.attention import dsa_topk_prefill as mod

    q, keys, top_k, stride = spec["num_queries"], spec["num_keys"], spec["top_k"], spec["logits_row_stride"]
    ops_ = mod._build_operands(torch, num_queries=q, num_keys=keys, top_k=top_k, logits_row_stride=stride, device="cuda")
    ops_ = replace(ops_, valid_mask=None)  # not needed; frees num_queries x num_keys bytes
    torch.cuda.empty_cache()
    if config["backend"] == "deep_select":
        _, op, (_, out_align) = mod._load_deep_select_backend()
        ops_ = replace(ops_, out=mod._deep_select_output(torch, num_queries=q, top_k=top_k, output_align=out_align, device="cuda"))
        mod._launch_deep_select(op, ops_, top_k=top_k)
        name = "torch.ops.deep_select.topk"
    else:
        _, op = mod._load_vllm_cuda_backend()
        op(ops_.logits, ops_.row_starts, ops_.row_ends, ops_.out, q, ops_.logits.stride(0), ops_.logits.stride(1), top_k)
        name = "torch.ops._C.top_k_per_row_prefill"
    torch.cuda.synchronize()
    return [{"callable": name, **_topk_check(ops_.logits, ops_.row_ends, ops_.out, top_k)}]


def check_cache_append(spec: dict, config: dict) -> list[dict]:
    from vllm import _custom_ops as ops

    from profiling.runners.attention import mla_cache_append as mod

    t, rank, rope, block, in_dt, kv_dt, _ = mod._validate_args(
        spec["num_tokens"], spec["kv_lora_rank"], spec["rope_dim"], spec["block_size"],
        spec["input_dtype"], spec["kv_dtype"], spec["cache_format"], allow_fp8_cache=True, allow_zero_rope=True,
    )
    ops_ = mod._build_operands(torch, num_tokens=t, kv_lora_rank=rank, rope_dim=rope, block_size=block,
                               torch_dtype=in_dt.torch(), cache_torch_dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(ops_.kv_c, ops_.k_pe, ops_.cache, ops_.slot_mapping, "fp8_e4m3", scale)
    rows = ops_.cache[ops_.block_indices, ops_.block_offsets]
    expected = torch.cat([ops_.kv_c, ops_.k_pe], dim=-1).to(ops_.cache.dtype)
    return [{"callable": "vllm._custom_ops.concat_and_cache_mla",
             **exact(rows.view(torch.uint8), expected.view(torch.uint8), source="exact (2_kernels.md §5)")}]


def check_index_remap(spec: dict, config: dict) -> list[dict]:
    from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index

    from profiling.runners.attention import dsa_sparse_index_remap as mod

    args = dict(spec)
    args["return_valid_counts"] = bool(int(args["return_valid_counts"]))
    validated = mod._validate_args(**args)
    ops_ = mod._build_operands(torch, validated, device=torch.device("cuda"))
    try:
        mod._check_vllm_triton_correctness(torch, triton_convert_req_index_to_global_index, ops_, validated)
        ok, note = True, ""
    except Exception as exc:  # the runner's exact check raises on any difference
        ok, note = False, repr(exc)
    return [{"callable": "triton_convert_req_index_to_global_index",
             "tolerance_source": "exact (runner _check_vllm_triton_correctness)", "pass": ok, "note": note}]


def check_sparse_mla(spec: dict, config: dict) -> list[dict]:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from profiling.db.args import DType
    from profiling.runners.attention import dsa_sparse_mla_attention as mod

    validated = mod._validate_args(
        **{k: spec[k] for k in ("num_queries", "num_cache_tokens", "num_heads", "num_kv_heads", "selected_k",
                                "latent_dim", "rope_dim", "value_dim", "softmax_scale", "q_dtype", "cache_dtype",
                                "index_dtype", "output_dtype", "valid_counts", "index_distribution", "cache_layout")},
        expected_q_dtype=DType.FP8_E4M3, expected_cache_dtype=DType.FP8_E4M3,
        expected_cache_layout=mod._TRTLLM_FP8_NOPE_CACHE_LAYOUT, expected_rope_dim=0,
    )
    h, sel, scale = spec["num_heads"], spec["selected_k"], float(spec["softmax_scale"])
    ops_ = mod._build_trtllm_fp8_operands(torch, validated, num_heads=h, rope_dim=0, selected_k=sel, device=torch.device("cuda"))
    # The runner's linspace q/cache give identical rows; use random FP8 values in the same tensors.
    g = torch.Generator(device="cuda").manual_seed(11)
    ops_.query.copy_(torch.randn(ops_.query.shape, device="cuda", generator=g).to(torch.float8_e4m3fn))
    ops_.cache.copy_(torch.randn(ops_.cache.shape, device="cuda", generator=g).to(torch.float8_e4m3fn))
    out = mod._launch_trtllm_fp8(trtllm_batch_decode_with_kv_cache_mla, ops_, softmax_scale=scale, selected_k=sel, rope_dim=0)
    torch.cuda.synchronize()
    flat = ops_.cache.view(-1, ops_.cache.shape[-1])
    got_all, ref_all = [], []
    counts = ops_.seq_lens
    for r in sample_rows(spec["num_queries"], 256).tolist():
        cnt = int(counts[r])
        if cnt == 0:
            continue
        kv = flat[ops_.block_tables[r, 0, :cnt].long()].float()  # [cnt, 512]
        s = (ops_.query[r, 0].float() @ kv.T) * scale  # [H, cnt]
        ref_all.append(torch.softmax(s, -1) @ kv)
        got_all.append(out[r, 0].float())
    tol = dict(atol=1e-1, rtol=1e-1, source="flashinfer tests/attention/test_trtllm_gen_mla.py sparse, fp8")
    return [{"callable": "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla (sparse_mla_top_k)",
             **compare(torch.stack(got_all), torch.stack(ref_all), **tol)}]


# ---------------------------------------------------------------- routed experts


def check_moe(spec: dict, config: dict) -> list[dict]:
    return [_moe_case(spec, x_div=10.0), _moe_case(spec, x_div=1.0)]


def _moe_case(spec: dict, *, x_div: float) -> dict:
    import flashinfer
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )

    from profiling.runners.moe import nvfp4_fused_moe as mod
    from profiling.runners.moe.exact_topk import exact_topk_ids
    from profiling.runners.moe.fp8_block_fused_moe import forced_routing_logits

    _, dequant = _nvfp4_test_utils()
    args = mod._validate_args(stack="vllm", **{**spec, "per_expert_batches": json.loads(spec["per_expert_batches"])})
    t, h, inter, e, top_k = args["num_tokens"], args["hidden_size"], args["intermediate_size"], args["num_experts"], args["top_k"]
    routed_scale = args["routed_scaling_numerator"] / args["routed_scaling_denominator"]
    swiglu_limit = 10.0  # GLM-5.3-Flash config; the fork passes it, the profiler row passes None
    fp4max = 448.0 * 6.0
    g = torch.Generator(device="cuda").manual_seed(5)

    def quant_experts(rows: int, cols: int):
        packed = torch.empty((e, rows, cols // 2), dtype=torch.uint8, device="cuda")
        scales = torch.empty((e, rows, cols // 16), dtype=torch.float8_e4m3fn, device="cuda")
        gs = torch.empty(e, dtype=torch.float32, device="cuda")
        for i in range(e):
            w = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda", generator=g) / 15
            gs[i] = fp4max / w.float().abs().amax()
            p, s = ops.scaled_fp4_quant(w, gs[i], is_sf_swizzled_layout=False)
            packed[i] = p
            scales[i] = s.view(torch.float8_e4m3fn).reshape(rows, cols // 16)
        return packed, scales, gs

    w13, s13, gs13 = quant_experts(2 * inter, h)  # vLLM order [w1 (gate); w3 (up)]
    w2, s2, gs2 = quant_experts(h, inter)

    def deq(packed, scales, gs, i):
        return dequant(packed[i], scales[i].view(torch.uint8), gs[i], torch.float32, "cuda", is_sf_linear_layout=True)

    x = torch.randn((t, h), dtype=torch.bfloat16, device="cuda", generator=g) / x_div
    a1_gs = (fp4max / x.float().abs().amax()).reshape(())
    xq, xs = ops.scaled_fp4_quant(x, a1_gs, is_sf_swizzled_layout=False)
    xd = dequant(xq, xs.view(torch.uint8), a1_gs, torch.float32, "cuda", is_sf_linear_layout=True)

    ids = torch.tensor(exact_topk_ids(num_tokens=t, top_k=top_k, per_expert_batches=args["per_expert_batches"]),
                       dtype=torch.int64, device="cuda")
    logits = forced_routing_logits(torch, ids.tolist(), e, "cuda")
    bias = torch.zeros(e, dtype=torch.float32, device="cuda")
    scores = torch.sigmoid(logits)
    weights = torch.gather(scores, 1, ids)
    weights = weights / weights.sum(1, keepdim=True) * routed_scale

    rows = sample_rows(t, 512)
    sel_ids, sel_w = ids[rows], weights[rows]

    def reference(a2_gs=None):
        ref = torch.zeros((len(rows), h), dtype=torch.float32, device="cuda")
        amax = torch.zeros((), device="cuda")
        for i in range(e):
            hit = (sel_ids == i)
            tok = hit.any(1).nonzero().flatten()
            if tok.numel() == 0:
                continue
            h1 = xd[rows][tok] @ deq(w13, s13, gs13, i).T
            gate = h1[:, :inter].clamp(max=swiglu_limit)
            up = h1[:, inter:].clamp(-swiglu_limit, swiglu_limit)
            act = gate * torch.sigmoid(gate) * up
            amax = torch.maximum(amax, act.abs().max())
            y = act @ deq(w2, s2, gs2, i).T
            ref[tok] += (sel_w[tok] * hit[tok].float()).sum(1, keepdim=True) * y
        return ref, amax

    ref, act_amax = reference()
    a2_gs = fp4max / (1.5 * act_amax)  # a calibrated checkpoint input_scale, with margin for unsampled rows
    g1_alphas = 1.0 / (a1_gs * gs13)
    g2_alphas = 1.0 / (a2_gs * gs2)

    w13r, s13r = reorder_w1w3_to_w3w1(w13.clone(), s13.clone())
    pw13, ps13, pw2, ps2 = prepare_static_weights_for_trtllm_fp4_moe(
        w13r, w2, s13r.view(torch.uint8), s2.view(torch.uint8),
        hidden_size=h, intermediate_size=inter, num_experts=e, is_gated_activation=True,
    )
    out = torch.empty((t, h), dtype=torch.bfloat16, device="cuda")
    flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
        routing_logits=logits, routing_bias=bias,
        hidden_states=xq, hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(t, -1),
        gemm1_weights=pw13, gemm1_weights_scale=ps13.view(torch.float8_e4m3fn),
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
        gemm1_clamp_limit=swiglu_limit / g1_alphas,
        gemm2_weights=pw2, gemm2_weights_scale=ps2.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=g1_alphas * a2_gs, output1_scale_gate_scalar=g1_alphas, output2_scale_scalar=g2_alphas,
        num_experts=e, top_k=top_k, n_group=args["n_group"], topk_group=args["topk_group"],
        intermediate_size=inter, local_expert_offset=0, local_num_experts=e,
        routed_scaling_factor=routed_scale, routing_method_type=mod.ROUTING_METHODS["deepseek_v3"],
        do_finalize=True, activation_type=3, output=out, enable_pdl=True,
        tune_max_num_tokens=mod.vllm_tune_max_num_tokens(t),
    )
    torch.cuda.synchronize()
    tol = dict(atol=2e-1, rtol=2e-1, source="vllm tests/kernels/moe/test_trtllm_nvfp4_moe.py (inputs /10)")
    label = "inputs /10 (vLLM test scale)" if x_div != 1.0 else "inputs N(0,1): SwiGLU clamp 10 active"
    result = compare(out[rows], ref, **tol)
    result["clamp_active_fraction_bound"] = None
    return {"callable": f"flashinfer.fused_moe.trtllm_fp4_block_scale_moe (DeepSeekV3 routing, clamp 10; {label})",
            **result}


# ---------------------------------------------------------------- KDA


def check_conv(spec: dict, config: dict) -> list[dict]:
    from profiling.runners.attention import gdn_causal_conv_prefill_dao_channellast as dao
    from profiling.runners.attention import gdn_causal_conv_prefill_vllm_triton as tri
    from profiling.runners.attention.gdn_causal_conv_prefill_reference import gdn_causal_conv_prefill_reference

    results = []
    tol = dict(atol=1e-2, rtol=1e-2, source="runner gdn_causal_conv_prefill checks (1e-2, states exact)")
    # Profiler row's callable: Dao causal-conv1d channel-last, one call per sequence.
    try:
        fn = dao._load_callable()
        args = dao._validate_args(**spec)
        ops_ = dao._build_operands(torch, args, device=torch.device("cuda"))
        expected, _ = gdn_causal_conv_prefill_reference(
            ops_.semantic_x.clone(), ops_.weight, ops_.state_store.transpose(1, 2).clone(),
            torch.tensor(ops_.slot_indices, dtype=torch.int32, device="cuda"))
        got = dao._semantic_output(torch, dao._invoke(fn, ops_.weight, dao._sequence_views(ops_, args)))
        r = compare(got, expected.float(), **_tol(tol))
        try:
            dao._check_fresh(torch, fn, ops_, args)
            dao._check_initial_state(torch, fn, ops_, args)
        except Exception as exc:
            r["pass"], r["note"] = False, f"runner check: {exc!r}"
        results.append({"callable": "causal_conv1d.causal_conv1d_fn (Dao, profiler row)", **r})
        del ops_, expected, got
    except Exception as exc:
        results.append({"callable": "causal_conv1d.causal_conv1d_fn (Dao, profiler row)", "pass": False,
                        "skipped": True, "note": repr(exc)[:400]})
    torch.cuda.empty_cache()
    # The fork's callable: vLLM Triton causal_conv1d_fn over the packed varlen batch.
    fused, meta = tri._load_vllm_components()
    args = tri._validate_args(**spec)
    ops_ = tri._build_operands(torch, args, meta, device=torch.device("cuda"))
    expected, _ = gdn_causal_conv_prefill_reference(
        ops_.semantic_x.clone(), ops_.weight, ops_.state.clone(), ops_.slot_indices)
    got = tri._semantic_output(tri._invoke_fused(fused, ops_), args)
    r = compare(got, expected.float(), **_tol(tol))
    ops_ = tri._build_operands(torch, args, meta, device=torch.device("cuda"))
    try:
        tri._check_correctness(torch, fused, ops_, args)
    except Exception as exc:
        r["pass"], r["note"] = False, f"runner check: {exc!r}"
    results.append({"callable": "vllm causal_conv1d_fn (Triton, fork)", **r})
    return results


def check_kda(spec: dict, config: dict) -> list[dict]:
    """The row's callable, FlashInfer `recurrent_kda(backend="cute-dsl-persistent")` (pp-impl's choice), and the
    fork's FlashKDA call (`_flashkda_prefill`), on the row's shape."""
    results = check_kda_flashinfer(spec, config)
    torch.cuda.empty_cache()
    return results + _check_flashkda(spec, config)


def _check_flashkda(spec: dict, config: dict) -> list[dict]:
    import vllm._flashkda_C  # noqa: F401

    from profiling.runners.attention import kda_chunk_prefill_flashkda as mod
    from profiling.runners.attention.kda_chunk_prefill_reference import kda_chunk_prefill_reference, rmse_ratio

    fwd, ws = torch.ops._flashkda_C.fwd, torch.ops._flashkda_C.get_workspace_size
    shape = mod.validate_args(**spec)
    ops_ = mod.build_operands(torch, shape, ws, device=torch.device("cuda"))
    init = ops_.initial_state.clone()
    mod.invoke(fwd, ops_, shape.head_dim)
    torch.cuda.synchronize()
    exp_o, exp_s = kda_chunk_prefill_reference(
        ops_.q.squeeze(0), ops_.k.squeeze(0), ops_.v.squeeze(0), ops_.g.squeeze(0),
        torch.sigmoid(ops_.beta.squeeze(0).float()), ops_.a_log, ops_.dt_bias, init, ops_.boundaries,
        lower_bound=mod.LOWER_BOUND)
    o_err, s_err = rmse_ratio(exp_o, ops_.out.squeeze(0)), rmse_ratio(exp_s, ops_.final_state)
    finite = bool(torch.isfinite(ops_.out).all() and torch.isfinite(ops_.final_state).all())
    return [{"callable": "torch.ops._flashkda_C.fwd (FlashKDA, fork)", "o_rmse_ratio": o_err,
             "state_rmse_ratio": s_err, "rtol_rmse": mod.RMSE_RATIO_TOL,
             "tolerance_source": "runner kda_chunk_prefill RMSE_RATIO_TOL (the fork's KDA metric)",
             "finite": finite, "pass": finite and o_err < mod.RMSE_RATIO_TOL and s_err < mod.RMSE_RATIO_TOL}]


def check_kda_flashinfer(spec: dict, config: dict) -> list[dict]:
    """FlashInfer nightly 0.7.2.dev20261008, as in pp-impl/.venv."""
    from profiling.runners.attention import kda_chunk_prefill_flashinfer as mod
    from profiling.runners.attention.kda_chunk_prefill_reference import kda_chunk_prefill_reference, rmse_ratio

    import flashinfer.kda as fkda

    backend = mod.CUTE_PERSISTENT
    shape = mod.validate_args(backend, **spec)
    ops_ = mod.build_operands(torch, shape, device=torch.device("cuda"))
    init = ops_.initial_state.clone()
    output, _ = mod.invoke(fkda.recurrent_kda, backend, ops_)
    torch.cuda.synchronize()
    exp_o, exp_s = kda_chunk_prefill_reference(
        ops_.q.squeeze(0), ops_.k.squeeze(0), ops_.v.squeeze(0), ops_.g.squeeze(0),
        torch.sigmoid(ops_.beta_logit.float()).squeeze(0), ops_.a_log, ops_.dt_bias, init, ops_.boundaries,
        lower_bound=mod.LOWER_BOUND)
    o_err, s_err = rmse_ratio(exp_o, output.squeeze(0)), rmse_ratio(exp_s, ops_.initial_state)
    return [{"callable": 'flashinfer.kda.recurrent_kda(backend="cute-dsl-persistent") (profiler row)',
             "o_rmse_ratio": o_err, "state_rmse_ratio": s_err, "rtol_rmse": mod.RMSE_RATIO_TOL,
             "tolerance_source": "runner kda_chunk_prefill RMSE_RATIO_TOL",
             "pass": o_err < mod.RMSE_RATIO_TOL and s_err < mod.RMSE_RATIO_TOL}]


# ---------------------------------------------------------------- driver

CHECKS = {
    ("mhc_pre_rms_norm", "vllm_tilelang"): check_mhc_pre,
    ("single_gemm", "torch_linear_vllm"): check_single_gemm_torch,
    ("gdn_causal_conv_prefill", "dao_channellast"): check_conv,
    ("kda_chunk_prefill", "flashinfer_cute_persistent"): check_kda,
    ("mhc_fused_post_pre_rms_norm", "deepgemm_mega_nonshifted"): check_mhc_mega,
    ("nvfp4_quant", "vllm_cuda"): check_nvfp4_quant,
    ("single_gemm", "flashinfer_cutedsl"): check_fp4_gemm,
    ("q_kv_rms_norm", "vllm_triton"): check_q_kv_rms_norm,
    ("gemm_fp32_output", "torch_cublas"): check_gemm_fp32_output,
    ("dsa_mqa_logits_prefill", "deepgemm_fp8"): check_mqa_logits,
    ("dsa_topk_prefill", "deep_select"): check_topk,
    ("dsa_topk_prefill", "vllm_cuda"): check_topk,
    ("batched_gemm", "torch_mla_q_absorb_no_rope"): check_batched_gemm,
    ("batched_gemm", "torch_mla_v_up_unpadded"): check_batched_gemm,
    ("mla_cache_append", "vllm_cuda"): check_cache_append,
    ("dsa_sparse_index_remap", "vllm_triton"): check_index_remap,
    ("dsa_sparse_mla_attention", "flashinfer_trtllm_fp8"): check_sparse_mla,
    ("nvfp4_fused_moe", "flashinfer_trtllm_sm100"): check_moe,
    ("mhc_fused_post_pre_rms_norm", "vllm_tilelang"): check_mhc_fused_tilelang,
    ("rms_norm", "vllm_cuda"): check_rms_norm,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", default="", help="comma-separated config numbers (default: all)")
    parser.add_argument("--rows", default="smallest,largest")
    opts = parser.parse_args()
    wanted = {int(c) for c in opts.configs.split(",") if c}
    configs = [c for c in load_configs() if not wanted or c["config"] in wanted]
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    failures = 0
    with opts.out.open("a") as sink:
        for config in configs:
            check = CHECKS[(config["kind"], config["backend"])]
            for which in opts.rows.split(","):
                spec = config[which]
                start = time.time()
                try:
                    results = check(spec, config)
                except Exception as exc:
                    results = [{"callable": "?", "pass": False, "error": repr(exc)[:600],
                                "traceback": traceback.format_exc()[-2000:]}]
                torch.cuda.synchronize()
                for result in results:
                    line = {"config": config["config"], "kind": config["kind"], "backend": config["backend"],
                            "positions": config["positions"], "row": which, "spec": spec,
                            "seconds": round(time.time() - start, 2), **result}
                    failures += not result["pass"]
                    sink.write(json.dumps(line) + "\n")
                    sink.flush()
                    short = {k: line.get(k) for k in ("config", "kind", "row", "callable", "pass", "max_abs_err",
                                                      "err_vs_ref_max", "calc_diff", "o_rmse_ratio", "error")}
                    print(json.dumps(short), flush=True)
                torch.cuda.empty_cache()
    print(f"done: {failures} failing checks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
