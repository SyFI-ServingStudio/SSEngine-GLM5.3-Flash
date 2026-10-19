"""Kernel manifest: the 35 configurations of handoff `2_kernels.md` §2 (PLAN.md §1).

Per configuration: the DB kind/backend and fixed configuration of the profiler rows, the callable our stage runs
(import path) and how the profiler row calls it, the library version the rows were taken with and ours, the M1
call-path verdict (`research/measured/kernels_identity.md`), and the GPU kernel names it launches. Three uses:

- the M1 identity check against the profiler rows (`research/measured/kernels_identity.md` is the table form of this file);
- one entry point for `research/bench/` and the tests (`CONFIGS`, `by_position`);
- `check_trace`: a stage's traced GPU kernel names must be exactly the manifest's, with none of the fallbacks a
  silent backend switch would launch (FlashKDA, the fork's Triton conv, the TileLang fused mHC at a layer boundary,
  ...).

Indexer top-k picks DeepSelect or vLLM `topKPerRowPrefill` per shape, whichever profiler row is faster
(`2_kernels.md` §3): `topk_backend(num_queries, num_keys)`, from `data/topk_rows.json` (the 412 shapes that have
both rows in `kernel_specs.jsonl`).

Kernel-name patterns marked `names_verified=True` were read off traced stages (`_NAMES_TRACED`); the others come
from library sources or symbol tables. The manifest check (`check_trace`, GPU test
`tests/test_model_gpu.py::test_chain_token_matches_vllm_and_manifest`) applies them to a traced stage.
"""

from __future__ import annotations

import bisect
import functools
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

ROW_IMAGE_CU130 = "vibesim-profiler-vllm:cu130 (vLLM 0.23.0, torch 2.11.0, cuBLAS 13.1.0.3, FlashInfer 0.6.12, TileLang 0.1.9)"
ROW_IMAGE_3F667D7 = "cu130-3f667d7e (vLLM 0.28.1rc0, torch 2.13.0, cuBLAS 13.1.1.3, FlashInfer 0.6.18, TileLang 0.1.12)"
ROW_FORK = "vllm_upstream_fork_env (fork 892da0822f)"
OURS = ("fork 892da0822f, torch 2.13.0+cu130, cuBLAS 13.1.1.3, FlashInfer 0.7.2.dev20261008, cutlass-dsl 4.8.0, "
        "Triton 3.7.1, TileLang 0.1.12, DeepGEMM 2.8.0 (vendored), Dao causal-conv1d 1.7.0")


@dataclass(frozen=True)
class KernelConfig:
    config: int                      # row number in 2_kernels.md §2 (= research/measured/kernels_identity.md)
    kind: str
    backend: str
    fixed: str                       # fixed configuration of the rows
    positions: tuple[str, ...]       # layer_targets.jsonl positions, without the "pp.<layer type>." prefix
    layer_types: tuple[str, ...]
    callable: str                    # what our stage calls
    row_call: str                    # how the profiler row calls it
    row_version: str
    verdict: str                     # SAME / ARGS / DIFF (fork vs row) -> how ours follows the row
    covered: bool                    # identical callable, arguments and version: no re-measure
    route: str                       # where our path selects it (ssengine.model.pp_patch or the fork as is)
    names: tuple[str, ...]           # regexes of the GPU kernels it launches
    forbidden: tuple[str, ...] = ()  # regexes of fallbacks that must not appear in a stage that runs this config
    names_verified: bool = False
    by_name: bool = True             # False: the kernel name is shared with other kernels; a trace cannot prove it ran


KDA = ("first_kda_dense", "kda_dense", "kda_moe")
DENSE = ("first_kda_dense", "kda_dense")
MOE = ("kda_moe", "dsa_moe")
DSA = ("dsa_moe",)
LAYERS = ("first_kda_dense", "kda_dense", "kda_moe", "dsa_moe")

# cuBLAS / CUTLASS GEMM kernel names on B200 (cuBLAS 13.1): nvjet, cutlass, sm100 xmma.
_CUBLAS = (r"^nvjet_", r"cutlass", r"sm100_xmma", r"^gemv", r"gemmk1_kernel", r"splitKreduce_kernel")

# Kernels that serve two configs (the same kernel at two positions): a trace counts them for both.
_NVFP4_SWIZZLED = (r"cvt_fp16_to_fp4<", r"15cvt_fp16_to_fp4I")         # configs 8, 10 (linear scales: config 29)
_MM_FP4 = (r"BlockScaledPersistentDenseGemmKernel",)                 # configs 9, 11: flashinfer mm_fp4 (cute-dsl)

CONFIGS: tuple[KernelConfig, ...] = (
    KernelConfig(1, "mhc_pre_rms_norm", "vllm_tilelang", "hidden 4096, 4 streams, BF16",
                 ("attn_mhc_pre", "stage_edge.hc_pre"), ("first_kda_dense", "stage_edge_recv"),
                 "vllm.model_executor.layers.mhc.mhc_pre_tilelang (layer 0; stage-edge hc_pre on stages 1-3)",
                 "mhc_pre_tilelang: DeepGEMM tf32_hc_prenorm_gemm + TileLang big-fuse", ROW_IMAGE_CU130,
                 "SAME (eps scalar only)", False, "fork", (r"tf32_hc_prenorm_gemm", r"mhc_pre", r"big_fuse")),
    KernelConfig(2, "single_gemm", "torch_linear_vllm", "k 4096, n 24896", ("kda.in_proj",), KDA,
                 "torch.nn.functional.linear (UnquantizedLinearMethod)", "F.linear", ROW_IMAGE_3F667D7,
                 "SAME", True, "fork", _CUBLAS),
    KernelConfig(3, "single_gemm", "torch_linear_vllm", "k 128, n 8192", ("kda.f_b_proj", "kda.g_b_proj"), KDA,
                 "torch.nn.functional.linear on a column view of in_proj (lda 24896)", "F.linear, contiguous A",
                 ROW_IMAGE_3F667D7, "ARGS (lda 24896)", False, "fork", _CUBLAS),
    KernelConfig(4, "gdn_causal_conv_prefill", "dao_channellast", "24576 channels, width 4",
                 ("kda.short_conv_prefill",), KDA,
                 "ssengine.model.kda_conv.conv_qkv (Triton): the Dao 1.7.0 channel-last kernel's arithmetic for the "
                 "whole microbatch in one launch, BF16 weight, final conv states in a second launch; q, k, v and states "
                 "bitwise equal to causal_conv1d_fn per sequence (tests/test_kda_conv_gpu.py)",
                 "Dao causal_conv1d_fn, one launch per sequence", "causal-conv1d 1.7.0",
                 "DIFF in the fork (Triton, FP32 weight); ours is bitwise the row's output in fewer launches", False,
                 "pp_patch.kda_prefill", (r"kda_conv_qkv_kernel", r"kda_conv_state_kernel"),
                 (r"_causal_conv1d_fwd_kernel", r"causal_conv1d_channellast_fwd_kernel")),
    KernelConfig(5, "kda_chunk_prefill", "flashinfer_cute_persistent", "64 heads, head dim 128",
                 ("kda.chunk_prefill",), KDA,
                 "flashinfer.kda.recurrent_kda(backend='cute-dsl-persistent')",
                 "recurrent_kda(backend='cute-dsl-persistent')", "FlashInfer 0.7.2.dev20261008 + cutlass-dsl 4.8.0",
                 "DIFF in the fork (FlashKDA); ours follows the row", False, "pp_patch.kda_prefill",
                 (r"^kernel_cutlass__pkd_",), (r"(?i)flashkda", r"(?i)flash_kda")),   # FlashInfer persistent `_pkd`
    KernelConfig(6, "single_gemm", "torch_linear_vllm", "k 8192, n 4096", ("kda.o_proj",), KDA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_3F667D7, "SAME", True, "fork", _CUBLAS),
    KernelConfig(7, "mhc_fused_post_pre_rms_norm", "deepgemm_mega_nonshifted", "hidden 4096, 4 streams",
                 ("attn_mhc_post_pre", "ffn_mhc_post_pre"), LAYERS,
                 "deep_gemm.mega_mhc (non-shifted, one persistent launch)", "DeepGEMM mega_mhc", ROW_FORK,
                 "DIFF in the fork (TileLang 3-launch fused); ours follows the row", False, "pp_patch.mhc_boundary",
                 (r"mega_mhc",), (r"(?i)mhc_fused_post_pre",)),
    KernelConfig(8, "nvfp4_quant", "vllm_cuda", "swizzled, hidden 4096", ("dense_ffn.gate_up_input_quant",), DENSE,
                 "vllm._custom_ops.scaled_fp4_quant (swizzled)", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork",
                 _NVFP4_SWIZZLED),
    KernelConfig(9, "single_gemm", "flashinfer_cutedsl", "k 4096, n 24576", ("dense_ffn.gate_up",), DENSE,
                 "flashinfer.mm_fp4(backend='cute-dsl') via FlashInferCuteDslNvFp4LinearKernel", "same",
                 "FlashInfer 0.6.18 / cutlass-dsl 4.6.2", "SAME (version)", False, "fork",
                 _MM_FP4),
    KernelConfig(10, "nvfp4_quant", "vllm_cuda", "swizzled, hidden 12288", ("dense_ffn.down_input_quant",), DENSE,
                 "vllm._custom_ops.scaled_fp4_quant (swizzled)", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork",
                 _NVFP4_SWIZZLED),
    KernelConfig(11, "single_gemm", "flashinfer_cutedsl", "k 12288, n 4096", ("dense_ffn.down",), DENSE,
                 "flashinfer.mm_fp4(backend='cute-dsl')", "same", "FlashInfer 0.6.18 / cutlass-dsl 4.6.2",
                 "SAME (version)", False, "fork", _MM_FP4),
    KernelConfig(12, "single_gemm", "torch_linear_vllm", "k 4096, n 2048", ("dsa.fused_qkv_a",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(13, "q_kv_rms_norm", "vllm_triton", "q 1536, kv 512", ("dsa.q_kv_a_norm",), DSA,
                 "vllm fused_q_kv_rmsnorm (Triton)", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork",
                 (r"^kernel$",), by_name=False),   # VllmTritonJitKernel's Triton function is `kernel`, as are others
    KernelConfig(14, "single_gemm", "torch_linear_vllm", "k 1536, n 16384", ("dsa.q_b_proj",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_3F667D7, "SAME", True, "fork", _CUBLAS),
    KernelConfig(15, "single_gemm", "torch_linear_vllm", "k 1536, n 4096", ("dsa.indexer.wq_b",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(16, "single_gemm", "torch_linear_vllm", "k 4096, n 160", ("dsa.indexer.wk_weights",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(17, "gemm_fp32_output", "torch_cublas", "fp32 in/out, k 4096, n 32", ("dsa.indexer.head_weights",),
                 DSA, "torch.mm(x.float(), W) at precision highest", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork",
                 _CUBLAS),
    KernelConfig(18, "single_gemm", "torch_linear_vllm", "k 4096, n 128", ("dsa.indexer.kpool_gate_score",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(19, "dsa_mqa_logits_prefill", "deepgemm_fp8", "32 heads, head dim 128, FP8",
                 ("dsa.indexer.mqa_logits_prefill",), DSA,
                 "vllm.utils.deep_gemm.fp8_fp4_mqa_logits (DeepGEMM 2.8.0 sm100_mqa_logits)",
                 "fp8_mqa_logits, one sequence, token-granular keys", "DeepGEMM 2.5.0",
                 "ARGS (pool-granular ends, packed requests, DeepGEMM 2.8.0)", False, "fork", (r"_mqa_logits",)),
    KernelConfig(20, "dsa_topk_prefill", "deep_select", "top 512 pools, causal with tail",
                 ("dsa.indexer.topk_prefill",), DSA, "torch.ops.deep_select.topk (per request, row start 0)",
                 "torch.ops.deep_select.topk", ROW_FORK, "DIFF in the fork (never called); routed per shape",
                 False, "pp_patch.indexer_topk", (r"topk_select_\w+::topk_kernel",)),
    KernelConfig(21, "dsa_topk_prefill", "vllm_cuda", "top 512 pools, causal with tail",
                 ("dsa.indexer.topk_prefill",), DSA, "torch.ops._C.top_k_per_row_prefill",
                 "_C.top_k_per_row_prefill, k 512", ROW_IMAGE_3F667D7, "SAME", True, "pp_patch.indexer_topk",
                 (r"topKPerRowPrefill",)),
    KernelConfig(22, "batched_gemm", "torch_mla_q_absorb_no_rope", "64 x (k 256, n 512)", ("dsa.q_absorb",), DSA,
                 "torch.bmm into a token-major buffer", "torch.bmm into contiguous [N, T, 512]", ROW_IMAGE_3F667D7,
                 "ARGS (ldc 32768)", False, "fork", _CUBLAS),
    KernelConfig(23, "mla_cache_append", "vllm_cuda", "block 64, FP8, latent 512",
                 ("dsa.sparse_mla.mla_cache_append",), DSA, "vllm._custom_ops.concat_and_cache_mla", "same",
                 ROW_IMAGE_CU130, "SAME", True, "fork", (r"concat_and_cache_mla_kernel",)),
    KernelConfig(24, "dsa_sparse_index_remap", "vllm_triton", "2176 selected slots",
                 ("dsa.sparse_mla.index_remap",), DSA, "triton_convert_req_index_to_global_index",
                 "same, HAS_PREFILL_WORKSPACE=True on 275 of 325 rows", ROW_IMAGE_3F667D7,
                 "ARGS (HAS_PREFILL_WORKSPACE=False, one 4096 tile)", False, "fork",
                 (r"sparse_mla_index_remap_kernel",)),   # ConvertReqIndexToGlobalIndexKernel's Triton function
    KernelConfig(25, "dsa_sparse_mla_attention", "flashinfer_trtllm_fp8",
                 "64 heads, latent 512, 2176 selected, FP8 q and cache", ("dsa.sparse_mla.prefill",), DSA,
                 "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=2176)", "same",
                 "FlashInfer 0.6.18", "SAME", True, "fork", (r"fmhaSm100",), (r"sparse_attn_fwd",)),
    KernelConfig(26, "batched_gemm", "torch_mla_v_up_unpadded", "64 x (k 512, n 256)", ("dsa.v_up",), DSA,
                 "torch.bmm, unpadded", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork", _CUBLAS),
    KernelConfig(27, "single_gemm", "torch_linear_vllm", "k 16384, n 4096", ("dsa.o_proj",), DSA,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_3F667D7, "SAME", True, "fork", _CUBLAS),
    KernelConfig(28, "gemm_fp32_output", "torch_cublas", "bf16 in, fp32 out, k 4096, n 288",
                 ("moe.router.gate", "moe.router.gate_recompute"), MOE,
                 "torch.mm(x, W.T, out_dtype=fp32) (GateLinear; ll_bf16_gemm for m <= 16); once per layer",
                 "torch.mm(x, W.T, out_dtype=fp32), twice per layer", ROW_IMAGE_CU130,
                 "ARGS (one GEMM, not two; m <= 16 path)", False, "fork", _CUBLAS + (r"(?i)ll_bf16_gemm",)),
    KernelConfig(29, "nvfp4_quant", "vllm_cuda", "linear scales, hidden 4096", ("moe.routed_rank0.input_quant",), MOE,
                 "vllm._custom_ops.scaled_fp4_quant (linear)", "same", ROW_IMAGE_3F667D7, "SAME", True, "fork",
                 (r"cvt_fp16_to_fp4_sf_major", r"24cvt_fp16_to_fp4_sf_majorI")),
    KernelConfig(30, "nvfp4_fused_moe", "flashinfer_trtllm_sm100",
                 "288 experts, top 8, intermediate 2048, DeepSeek-V3 routing, scale 5/2",
                 ("moe.routed_rank0.fused_moe",), MOE, "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
                 "same, gemm1_clamp_limit=None", "FlashInfer 0.6.18",
                 "ARGS (clamp tensor 10/g1_alphas, deferred finalize)", False, "fork",
                 (r"moe::dev::", r"^bmm_", r"(?i)batchedGemm")),
    KernelConfig(31, "single_gemm", "torch_linear_vllm", "k 4096, n 4096", ("moe.shared_expert.gate_up",), MOE,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(32, "single_gemm", "torch_linear_vllm", "k 2048, n 4096", ("moe.shared_expert.down",), MOE,
                 "torch.nn.functional.linear", "F.linear", ROW_IMAGE_CU130, "SAME (cuBLAS version)", False, "fork",
                 _CUBLAS),
    KernelConfig(33, "mhc_fused_post_pre_rms_norm", "vllm_tilelang", "hidden 4096, 4 streams",
                 ("final_mhc_post", "stage_edge.hc_post"), ("output_head", "stage_edge_send"),
                 "vllm.model_executor.layers.mhc.mhc_post_tilelang + mean over streams (fork; kept if not slower "
                 "than the row, measured at M2)", "mhc_fused_post_pre_tilelang (post + prenorm GEMM + big-fuse)",
                 ROW_IMAGE_CU130, "DIFF (1 launch + mean instead of 3 launches)", False, "pp_patch.final_post",
                 (r"mhc_post_tilelang",)),
    KernelConfig(34, "rms_norm", "vllm_cuda", "hidden 4096", ("final_norm",), ("output_head",),
                 "vllm._custom_ops.rms_norm", "_C.rms_norm", ROW_IMAGE_CU130, "SAME (layernorm_kernels.cu changed)",
                 False, "fork", (r"rms_norm_kernel",)),
    KernelConfig(35, "single_gemm", "torch_linear_vllm", "k 4096, n 154880", ("lm_head",), ("output_head",),
                 "torch.nn.functional.linear on the is_last rows only", "F.linear", ROW_IMAGE_3F667D7, "SAME", True,
                 "pp_patch / stage.py (rows gathered before lm_head)", _CUBLAS),
)

# Name patterns seen on traced stages: 7940 (chain test, 1,024 tokens, stages 0-3) and 7885 (stage 2 at 45k + 10k
# tokens: indexer logits and vLLM top-k). Config 20 (DeepSelect) is from the library's symbol table; config 13's name
# is not its own (`by_name=False`); the cuBLAS-only configs count as one shared group.
_NAMES_TRACED = frozenset({1, 4, 5, 7, 8, 9, 10, 11, 19, 21, 23, 24, 25, 29, 30, 33, 34})
CONFIGS = tuple(replace(c, names_verified=c.config in _NAMES_TRACED) for c in CONFIGS)

# Glue the targets price as `elementwise / triton` (2_kernels.md §6): embedding, hc_expand, state gather/scatter,
# gated norm, act_and_mul, indexer k_norm / q_fwht_quant / weight_scale / kpool writes, expand_pools, DSA glue,
# MoE input/combine glue, hc_contract mean, q_fp8_quant. Any Triton kernel, PyTorch elementwise/reduce/copy/
# index_select kernel, scan, or memset.
GLUE = (r"^triton_", r"_kernel$", r"elementwise_kernel", r"reduce_kernel", r"(?i)memcpy|memset", r"CatArrayBatchedCopy",
        r"index_elementwise", r"(?i)fill", r"(?i)gather|scatter", r"(?i)embedding", r"(?i)act_and_mul|silu|swiglu",
        r"(?i)norm", r"(?i)expand_pools|kpool|fwht", r"(?i)copy",
        r"scaled_fp8_quant_kernel",        # q_fp8_quant: MLA query to FP8 before sparse MLA
        r"cub::detail::scan::DeviceScan",  # cumsum in the DSA index plumbing (`glue`, `prefill_glue`)
        r"indexSelect(Small|Large)Index")  # index_select: last rows before final hc_post / lm_head; prefill_gather

# Stage composition (3_layers.md): layer types per stage, split [12, 11, 11, 11].
SPLIT = (12, 11, 11, 11)
DSA_LAYERS = frozenset(range(3, 45, 4))


def layer_type(layer: int) -> str:
    if layer == 0:
        return "first_kda_dense"
    if layer in (1, 2):
        return "kda_dense"
    return "dsa_moe" if layer in DSA_LAYERS else "kda_moe"


def stage_layers(stage: int, split=SPLIT) -> range:
    start = sum(split[:stage])
    return range(start, start + split[stage])


def stage_layer_types(stage: int, split=SPLIT) -> set[str]:
    types = {layer_type(i) for i in stage_layers(stage, split)}
    if stage == 0:
        types.add("input_head")
    else:
        types.add("stage_edge_recv")   # standalone hc_pre on the received 4-stream residual (PLAN.md §1)
    if stage == len(split) - 1:
        types.add("output_head")
    else:
        types.add("stage_edge_send")   # the last layer's deferred hc_post before the send
    return types


def by_config(n: int) -> KernelConfig:
    return CONFIGS[n - 1]


def by_position(layer_type_: str, position: str, backend: str | None = None) -> list[KernelConfig]:
    """Configs at `pp.<layer type>.<position>` (layer_targets.jsonl naming)."""
    return [c for c in CONFIGS if layer_type_ in c.layer_types and position in c.positions
            and (backend is None or c.backend == backend)]


# ---- indexer top-k per shape ------------------------------------------------------------------------------------

DEEP_SELECT_MAX_TOP_K = 4096
DEEP_SELECT_MAX_ROW = 1 << 23
_TOPK_ROWS = Path(__file__).parent / "data" / "topk_rows.json"


@functools.lru_cache(maxsize=1)
def _topk_table() -> list[tuple[float, float, str, float, float]]:
    rows = json.loads(_TOPK_ROWS.read_text())["rows"]
    return [(math.log2(q), math.log2(k), "deep_select" if ds < vc else "vllm_cuda", ds, vc) for q, k, ds, vc in rows]


@functools.lru_cache(maxsize=4096)
def topk_backend(num_queries: int, num_keys: int, top_k: int = 512) -> str:
    """'deep_select' or 'vllm_cuda': the faster profiler row at the nearest profiled shape (log2 distance)."""
    if num_keys <= top_k:
        return "vllm_cuda"  # every key selected: both write the natural indices; vLLM's kernel skips the sort
    if top_k > DEEP_SELECT_MAX_TOP_K or num_keys >= DEEP_SELECT_MAX_ROW:
        return "vllm_cuda"
    lq, lk = math.log2(max(num_queries, 1)), math.log2(num_keys)
    best = min(_topk_table(), key=lambda r: (r[0] - lq) ** 2 + (r[1] - lk) ** 2)
    return best[2]


def topk_row_ms(num_queries: int, num_keys: int) -> tuple[float, float]:
    """(deep_select, vllm_cuda) row times at the nearest profiled shape."""
    lq, lk = math.log2(max(num_queries, 1)), math.log2(max(num_keys, 1))
    best = min(_topk_table(), key=lambda r: (r[0] - lq) ** 2 + (r[1] - lk) ** 2)
    return best[3], best[4]


# ---- trace check ------------------------------------------------------------------------------------------------

@dataclass
class TraceCheck:
    matched: dict[int, list[str]] = field(default_factory=dict)   # config -> kernel names
    glue: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)                # not in the manifest and not glue
    forbidden: list[tuple[int, str]] = field(default_factory=list)  # (config whose fallback it is, kernel name)
    unverifiable: dict[int, list[str]] = field(default_factory=dict)  # by_name=False configs -> candidate names
    missing: list[int] = field(default_factory=list)                # configs of this stage with no kernel

    @property
    def ok(self) -> bool:
        return not (self.unknown or self.forbidden or self.missing)

    def summary(self) -> str:
        lines = [f"ok={self.ok} matched={sorted(self.matched)} glue={len(self.glue)}"]
        lines += [f"  forbidden (config {c}): {n}" for c, n in self.forbidden]
        lines += [f"  unknown: {n}" for n in sorted(set(self.unknown))]
        lines += [f"  missing: config {c} {by_config(c).kind}/{by_config(c).backend}" for c in self.missing]
        lines += [f"  unverifiable by name: config {c} {by_config(c).kind}/{by_config(c).backend} "
                  f"(candidate kernels: {', '.join(sorted(set(n)))})" for c, n in sorted(self.unverifiable.items())]
        return "\n".join(lines)


def _any(patterns, name: str) -> bool:
    return any(re.search(p, name) for p in patterns)


# Configs whose kernel names are not distinctive (cuBLAS GEMMs are shared by many positions): a trace can show that
# some GEMM ran, not which position it served. They count as present when any cuBLAS kernel ran.
_SHARED_NAMES = frozenset(c.config for c in CONFIGS if set(_CUBLAS) & set(c.names))


def _distinct(c: KernelConfig) -> tuple[str, ...]:
    return tuple(p for p in c.names if p not in _CUBLAS)


def check_trace(kernel_names: list[str], stage: int, split=SPLIT, optional: frozenset[int] = frozenset()) -> TraceCheck:
    """Classify a stage's traced GPU kernel names against the manifest.

    A name goes to every non-GEMM config whose `names` match (one kernel can serve two positions, e.g. configs 8 and
    10), else to the GEMM configs, else glue, else unknown. A config whose kernel name is not its own (`by_name=False`,
    config 13's Triton `kernel`) is never matched: its candidate names go to `unverifiable`, listed in the summary,
    and it is missing when no candidate ran. `forbidden` lists fallbacks of configs this stage runs. `missing` lists
    this stage's configs with no kernel (`optional`: configs that may legitimately not run, e.g. the indexer on a
    microbatch of short prefills, or one of the two top-k backends).
    """
    types = stage_layer_types(stage, split)
    active = [c for c in CONFIGS if set(c.layer_types) & types]
    out = TraceCheck()
    for name in kernel_names:
        hits = [c for c in active if c.by_name and _any(_distinct(c), name)]
        weak = [c for c in active if not c.by_name and _any(_distinct(c), name)]
        if not hits and _any(_CUBLAS, name):
            for c in active:
                if c.config in _SHARED_NAMES:
                    out.matched.setdefault(c.config, []).append(name)
            continue
        bad = next((c for c in active if _any(c.forbidden, name)), None)
        if bad is not None:
            out.forbidden.append((bad.config, name))
        elif hits:
            for c in hits:
                out.matched.setdefault(c.config, []).append(name)
        elif weak:
            for c in weak:
                out.unverifiable.setdefault(c.config, []).append(name)
        elif _any(GLUE, name):
            out.glue.append(name)
        else:
            out.unknown.append(name)
    topk = {20, 21}
    for c in active:
        if c.config in out.matched or c.config in out.unverifiable or c.config in optional:
            continue
        if c.config in topk and topk & set(out.matched):
            continue  # per shape: one of the two
        out.missing.append(c.config)
    return out
