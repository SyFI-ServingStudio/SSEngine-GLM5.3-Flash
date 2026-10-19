"""Every runtime override of a library that ssengine makes (PLAN.md §1: the vLLM fork is used as a library, not vendored).

Nothing here edits `third_party/vllm` or any installed package. Each override is a per-instance attribute, a module
attribute swapped for the duration of a `with` block, or (marked "process-wide") a module attribute replaced once in a
stage process, which serves this one model and nothing else. Each one is deliberate: the reason is in the second column
and in the function's docstring, and the last column names what fails if it stops taking effect (a fork or FlashInfer
upgrade that renames the target, say). "Manifest check" is `ssengine.kernels.check_trace` on a traced forward: every GPU
kernel a stage launches must be the manifest's for its position, with no fallback (GPU test d,
`tests/test_model_gpu.py::test_chain_token_matches_vllm_and_manifest`; M2 `research/bench/layers.py` per run).

| Override | Replaces (library) | Why | Proved by |
|---|---|---|---|
| `stage_scope` | `vllm.distributed.utils.get_pp_indices`, glm5next `model.get_pp_group` (vLLM), while the model is built | build only this stage's decoder layers, embedding on stage 0, norm/lm_head on stage 3; vLLM's own groups stay single-rank | GPU tests b, c (`edge`, `chain` vs vLLM TP4) |
| `stage_weights` | the checkpoint iterator given to `load_weights` (vLLM) | read only this stage's tensors | `test_model_cpu.py::test_stage_weights_partition_the_text_model`; GPU tests b, c |
| `edge_send` / `edge_recv` | the fork's PP hand-off, which drops the mHC deferred state | send the 4-stream residual after the last layer's `hc_post`; the receiver's first layer runs a standalone `hc_pre` (issues.md, "mHC boundary at the stage edge") | GPU test b (split vs unsplit) |
| `route_mhc_boundaries` | each layer's `mhc_fused_post_pre_op` (vLLM TileLang) | DeepGEMM `mega_mhc`, non-shifted (config 7, `kernels_identity.md` Decisions) | manifest check (config 7) |
| `route_hc_expand` | glm5next `model.hc_expand` (vLLM), process-wide | one vectorized concatenation; the fork's broadcast copy ran at ~1 TB/s (M2: input_head x3.3) | M2 `input_head` time (issues.md, "M2 results") |
| `route_kda` | each KDA layer's `Glm5NextLinearAttention._forward` (vLLM) | the short conv by `kda_conv` (config 4: bitwise the Dao causal-conv1d call per sequence the design prices, in one launch per layer: ~78 us at 4,096 rows vs Dao 111-225 by sequence count, job 7927) + FlashInfer persistent `recurrent_kda` (config 5) on our state blocks, gathered and scattered by `kda_state` (5.1 + 4.4 us per layer vs torch's 12.7 + 9.4 at 2 sequences, job 7918) | manifest check (configs 4, 5); `research/bench/kvcheck.py`; `test_kda_conv_gpu.py` (q, k, v, conv states, state copies bitwise) |
| `install_topk_router` | `torch.ops._C.top_k_per_row_prefill` and `indexer.build_prefill_chunk_metadata` (vLLM), process-wide | per shape, DeepSelect or vLLM's kernel, whichever row is faster (configs 20/21); `topk_canon` makes the answer deterministic | manifest check (configs 20/21) on traces past 2,048 tokens (the 1k chain test runs no indexer; issues.md, "Manifest trace check"); `test_model_cpu.py::test_topk_choice_matches_the_faster_row_at_every_profiled_shape`; GPU test g |
| `install_decode_topk_canon` | glm5next `sparse_indexer.get_indexer_topk` (vLLM), process-wide | the indexer's decode top-k (1-token chunks) returns its set in a varying order: `topk_canon` fixes set and order, as on the prefill path | `test_model_gpu.py::test_decode_topk_canonical`; `research/bench/margin.py --trace-topk` (issues.md, job 7522) |
| `install_kpool_unaligned` | glm5next `sparse_indexer._kpool_compress_insert` and `kpool_compress.kpool_seed_tail_cache` (vLLM), process-wide | a chunk may start inside a 4-token indexer pool (prefix hits resume at the kept prompt's length); the fork's insert assumes pool-aligned starts. The straddling pool is written from the tail ring + batch rows with the same kernel | `test_kpool_cpu.py` (reference ops); `test_kpool_gpu.py` (fork ops, cache bytes bitwise vs unsplit) |
| `install_kda_split_heads` | `flashinfer.kda_kernels.persistent.forward._split_heads`, process-wide | host time: the KDA plan's head-split search (a Python LPT simulation over a 10 x 10 grid, 20-35 ms per multi-request microbatch, up to 110 ms; stage 0 waited for it, 1.1 s/min at good N=400, 7895) in C++ (`kda_schedule.py`); same schedule | `test_kda_schedule_cpu.py` (equal to the Python search on random microbatches, and the whole plan) |
| `install_kpool_expand` | glm5next `sparse_indexer.sparse_attn_indexer_kpool` (vLLM), process-wide: its source with three edits (`kpool_fast._EDITS`) | host and GPU time: the prefill pool-to-token expansion in one kernel written into the top-k buffer, in place of two casts, a 128-column expand and a copy (stage 2 at 4,096 rows: 0.21 ms per microbatch, job 7787); same values (index arithmetic) | `test_kpool_gpu.py::test_expand_into_matches_the_fork_bitwise`; `test_kpool_cpu.py::test_kpool_expand_installs` (a fork change to the edited lines fails at start-up) |
| `install_indexer_float` | glm5next `Indexer.forward` (vLLM), process-wide: its source with one edit (`glue_fast._INDEXER_EDITS`) | GPU time: the indexer's BF16 -> FP32 cast of the hidden rows by one Triton kernel, not torch's unrolled copy (49.2 -> 20.5 us at 4,096 rows, job 7914; 3 per stage-2 microbatch); exact | `test_glue_gpu.py::test_indexer_float_matches_torch_cast` |
| `install_indexer_split` | `DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks` (vLLM), process-wide | the indexer logits of one prefill go out in calls of at most 2 GiB, query rows in multiples of 512: DeepGEMM's `fp8_mqa_logits` costs the same per 512 query rows at long context, so the fork's greedy split (1,022 + 2 rows at 512k + 1k) paid a whole second call for 2 rows (issues.md, "Long context") | `test_indexer_split_cpu.py`; nsys at 512k |
| `install_sparse_mla_split` | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, process-wide | a launch over 32,768 rows returns without doing the work on B200; split as the target does (issues.md, "M2 results") | M2 `dsa_moe` at 64k/128k |
| `install_moe_buckets` | `flashinfer.fused_moe.shared.tuning` bucket functions, process-wide | the MoE's tactic buckets are the serving shapes (every CUDA-graph bucket), not FlashInfer's hybrid ones, which served 4,097-8,191 tokens with 8,192's tactic (`moe_tuning.py`) | `test_moe_tuning_cpu.py`; `benchmarks.tune_tactics verify` (0 misses) |
| `install_sm_h2d` | `vllm.utils.torch_utils.async_tensor_h2d` and every module binding of it (vLLM), process-wide | GPU time: the builders' host -> device copies (MLA chunked-context tables, up to MBs per microbatch with long cached prefixes) by a kernel reading mapped pinned memory (`sm_h2d`), not the copy engine the KV tier reads keep busy: copies queued behind them started up to tens of ms late (job 8367); same bytes | `test_sm_h2d_gpu.py`; GPU tests b, c, e |
| `install_autotune_memo` | FlashInfer `AutoTuner.choose_one`, process-wide | host time: memoize the tactic lookup outside tuning (same answer) | `test_host_trims_cpu.py::test_autotune_memo_matches_and_returns_callers_runner` |
| `install_mm_fp4_skip_check` | `flashinfer.mm_fp4`, process-wide | host time: `skip_check=True` after the warm-up ran the checks | GPU tests c, d (same kernels, same token) |
| `plain_parameters` | vLLM parameter subclasses on the loaded weights | host time: no `__torch_function__` round trip per op | `test_host_trims_cpu.py::test_plain_parameters_same_storage_and_attrs` |
| `install_host_trims` | also `vllm.envs.enable_envs_cache()`, as vLLM's engine does after start-up | host time | as above |

Elsewhere, same rules: `kv_tensors.design_split` (vLLM `get_pp_indices` around the KV group function: the design's
split, `test_model_cpu.py::test_kv_groups_place_layers_by_the_design_split`) and, as settings rather than patches,
`stage.build_vllm_config` (`sparse_mla_force_mqa`: short prompts stay on sparse MLA as priced) and the IR op
priorities `StageModel` sets as vLLM's worker does.
"""
from __future__ import annotations

import contextlib
import functools
import json
import re
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ssengine import kernels
from ssengine.model import topk_canon

HIDDEN = 4096
HC = 4

# ---------------------------------------------------------------- stage construction


class _StagePPGroup:
    """What glm5next's constructors ask of `get_pp_group()`: is this the first / last stage."""

    def __init__(self, first: bool, last: bool):
        self.is_first_rank, self.is_last_rank = first, last
        self.rank_in_group, self.world_size = 0, 1


@contextlib.contextmanager
def stage_scope(layers: range, first: bool, last: bool):
    """Build-time patches: decoder layers [layers.start, layers.stop), embedding iff `first`, norm/lm_head iff
    `last`. vLLM's process groups stay single-rank; only `make_layers` and glm5next's model.py read the PP rank."""
    import vllm.distributed.utils as dist_utils
    import vllm.models.glm5next.common.model as glm_model

    group = _StagePPGroup(first, last)
    saved = dist_utils.get_pp_indices, glm_model.get_pp_group
    dist_utils.get_pp_indices = lambda num_layers, rank, size: (layers.start, layers.stop)
    glm_model.get_pp_group = lambda: group
    try:
        yield
    finally:
        dist_utils.get_pp_indices, glm_model.get_pp_group = saved


_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def stage_weight_names(ckpt: Path, layers: range, first: bool, last: bool) -> dict[str, list[str]]:
    """Checkpoint file -> tensor names this stage loads."""
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    files: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        m = _LAYER.match(name)
        if m:
            keep = int(m.group(1)) in layers
        elif name.startswith("model.visual."):
            keep = False
        elif name == "model.language_model.embed_tokens.weight":
            keep = first
        elif name in ("model.language_model.norm.weight", "lm_head.weight"):
            keep = last
        else:
            raise KeyError(f"unclassified checkpoint tensor {name}")
        if keep:
            files.setdefault(file, []).append(name)
    return files


def stage_weights(ckpt: Path, layers: range, first: bool, last: bool) -> Iterator[tuple[str, torch.Tensor]]:
    """(checkpoint name, CPU tensor) for this stage only; the model's own name mapper takes it from there."""
    from safetensors import safe_open

    for file, names in sorted(stage_weight_names(ckpt, layers, first, last).items()):
        with safe_open(ckpt / file, framework="pt", device="cpu") as f:
            for name in names:
                yield name, f.get_tensor(name)


# ---------------------------------------------------------------- mHC


class MegaMhcPostPre(nn.Module):
    """`MHCFusedPostPreOp` replacement: DeepGEMM `mega_mhc`, non-shifted (one persistent launch), config 7.

    Same arguments and outputs as the fork's `mhc_fused_post_pre_tilelang` (residual, post, comb, layer input).
    """

    def forward(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                norm_eps=0.0):
        from vllm.utils.deep_gemm import mega_mhc

        assert norm_weight is not None and x.shape[0] <= 1 << 20
        new_residual = torch.empty_like(residual)
        new_post = torch.empty_like(post_layer_mix)
        new_comb = torch.empty_like(comb_res_mix)
        y = torch.empty(x.shape[0], x.shape[-1], dtype=torch.bfloat16, device=x.device)
        mega_mhc(x=x, residual=residual, shifted_prev_mix=None, post_mix=post_layer_mix,
                 comb_res_mix=comb_res_mix, fn=fn, mix_scales=hc_scale, mix_bases=hc_base,
                 hc_mult=residual.shape[1], hc_norm_eps=rms_eps, hc_pre_eps=hc_pre_eps,
                 hc_post_scale=hc_post_mult_value, sinkhorn_eps=hc_sinkhorn_eps,
                 num_sinkhorn_iters=sinkhorn_repeat, rmsnorm_weight=norm_weight, rmsnorm_eps=norm_eps,
                 rmsnorm_scale=1.0, new_residual=new_residual, new_prev_mix=None, new_post_mix=new_post,
                 new_comb_res_mix=new_comb, y_bf16=y)
        return new_residual, new_post, new_comb, y


def route_mhc_boundaries(layers) -> None:
    for layer in layers:
        layer.mhc_fused_post_pre_op = MegaMhcPostPre()


def edge_send(last_layer, x, residual, post, comb) -> torch.Tensor:
    """Sender side of the stage edge: the last layer's deferred hc_post, flattened to [T, 4 * 4096]."""
    return last_layer.hc_post(x, residual, post, comb).flatten(1)


def edge_recv(hidden: torch.Tensor) -> torch.Tensor:
    """Receiver side: the 4-stream residual [T, 4, 4096]; the first layer runs a standalone hc_pre on it."""
    return hidden.view(hidden.shape[0], HC, HIDDEN)


# ---------------------------------------------------------------- input head


def hc_expand_cat(x: torch.Tensor, n: int) -> torch.Tensor:
    """`mhc.hc_expand` ([T, H] -> [T, n, H] by replication) as one vectorized concatenation; the fork's broadcast
    `expand().contiguous()` copy ran at ~1 TB/s (M2: input_head x3.3 at 16k)."""
    T, H = x.shape
    out = torch.empty(T, n * H, dtype=x.dtype, device=x.device)
    step = max(1, (1 << 30) // (n * H))     # < 2^31 elements per launch: 32-bit indexing (M2: x3.3 at 128k)
    for a in range(0, T, step):
        torch.cat([x[a:a + step]] * n, dim=1, out=out[a:a + step])
    return out.view(T, n, H)


def route_hc_expand() -> None:
    import vllm.models.glm5next.common.model as glm_model

    glm_model.hc_expand = hc_expand_cat


# ---------------------------------------------------------------- KDA prefill


@dataclass
class KdaPrefillMeta:
    """One microbatch's KDA prefill: per sequence (first row, end row, state block, has initial state)."""
    num_tokens: int
    seqs: list[tuple[int, int, int, bool]]
    slots: torch.Tensor          # int64 [N] state blocks, on device
    plan: tuple                  # FlashInfer persistent KDA work plan for these sequence offsets
    conv_tabs: tuple             # kda_conv.tables on device (token blocks, sequences)
    inits: torch.Tensor          # int32 [N] has initial state, on device


def make_kda_plan_host(offsets: list[int], num_heads: int, num_sms: int) -> tuple:
    """FlashInfer persistent KDA `make_plan`'s schedule, on the host: ((cu32, soff, schain, spt0, sptn, ssrc, sdst)
    as int32 numpy arrays, G, nbuf, has_split, full_chunks). The plan `fwd` takes is these arrays on the device,
    then G, an FP32 [max(nbuf, 1), 128, 128] handoff buffer, int32 [max(nbuf, 1)] flags, has_split, full_chunks
    (`make_plan` builds the same tuple with device tensors; tests/test_kda_schedule_cpu.py)."""
    from flashinfer.kda_kernels.persistent import forward as fw

    lens = [end - start for start, end in zip(offsets, offsets[1:])]
    soff, schain, spt0, sptn, ssrc, sdst, G, nbuf = fw._piece_schedule(lens, num_heads, num_sms,
                                                                      torch.device("cpu"))
    arrays = tuple(np.asarray(x, dtype=np.int32) for x in
                   (offsets, soff.numpy(), schain.numpy(), spt0.numpy(), sptn.numpy(), ssrc.numpy(), sdst.numpy()))
    return arrays, G, nbuf, nbuf > 0, all(n % fw.C_TOK == 0 for n in lens)


def kda_prefill_forward(self, qkv_proj_states, g1, beta, core_attn_out) -> None:
    """`Glm5NextLinearAttention._forward` for prefill chunks (configs 4 and 5).

    Short conv: `kda_conv.conv_qkv`, the microbatch's packed q|k|v into three contiguous [T, H*D] buffers (FlashInfer
    takes contiguous q/k/v) in one launch, and the requests' final conv states in a second; the request's conv state
    is the initial state of a continuation chunk. Bitwise the Dao causal-conv1d channel-last call per sequence the
    design prices (BF16 merged weight; tests/test_kda_conv_gpu.py). Recurrence: FlashInfer persistent KDA (gate and
    beta sigmoid in kernel, lower bound -5) over an FP32 [N, H, 128, 128] state gathered from the requests' state
    blocks (zero for a fresh request) and scattered back, by `kda_state`'s wide-copy kernels.
    """
    from flashinfer.kda_kernels.persistent.forward import fwd
    from vllm.forward_context import get_forward_context

    from ssengine.model import kda_conv, kda_state

    meta = get_forward_context().attn_metadata[self.prefix]
    assert isinstance(meta, KdaPrefillMeta)
    conv_state, rec_state = self.kv_cache   # bound by vLLM: [blocks, W-1, C] bf16 (SD layout), [blocks, H, D, D] fp32
    T = meta.num_tokens
    qkv = qkv_proj_states[:T]
    H, D = self.local_num_heads, self.head_dim
    P = H * D
    q, k, v = kda_conv.conv_qkv(qkv, self._pp_conv_weight, conv_state, meta.conv_tabs[0], meta.conv_tabs[1], P)
    q, k, v = (t.view(1, T, H, D) for t in (q, k, v))
    g = g1[:, :T]
    b = beta[:, :T].contiguous()
    state = kda_state.gather(rec_state, meta.slots, meta.inits)
    fwd(q, k, v, g, b, D ** -0.5, core_attn_out[:, :T], self._pp_a_log, self.dt_bias, self.kda_lower_bound,
        initial_state=state, final_state=state, cu_seqlens=meta.plan[0], plan=meta.plan)
    kda_state.scatter(state, meta.slots, rec_state)


def route_kda(attn) -> None:
    """Install `kda_prefill_forward` on one `Glm5NextLinearAttention` (after its weights are loaded)."""
    def w(m):
        return m.weight.view(m.weight.size(0), m.weight.size(-1))

    attn._pp_conv_weight = torch.cat([w(attn.q_conv1d), w(attn.k_conv1d), w(attn.v_conv1d)]).to(
        torch.bfloat16).contiguous()
    assert attn.q_conv1d.bias is None
    attn._pp_a_log = attn.A_log.detach().reshape(-1).contiguous()
    # an eager break of the fork's breakable CUDA graph capture, as the `_forward` it replaces: per-microbatch
    # metadata (conv tables, KDA work plan) and launches run eagerly between graph segments
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    attn._forward = types.MethodType(eager_break_during_capture(kda_prefill_forward), attn)


# ---------------------------------------------------------------- indexer top-k


# ks.data_ptr() -> [(first row, end row, first key column, backend)] of one indexer prefill chunk. Filled when vLLM's
# indexer metadata builder makes the chunk (`build_prefill_chunk_metadata`, wrapped below with the host-side request
# boundaries it already has), read by the top-k router when the fork's indexer calls `top_k_per_row_prefill`.
TOPK_SEGMENTS: dict[int, list[tuple[int, int, int, str]]] = {}
TOPK_LOG: list[tuple[int, int, str]] | None = None   # (rows, keys, backend) per launch, when a caller wants them
_ORIG_TOPK = None
_DS_INPUT_ALIGN = 16


# DeepSelect with sorted_index returns the canonical top-k itself (topk_canon's rule: the keys above the k-th value,
# the lowest-index ties, ascending), so its rows skip canonicalize's extra read of the logits. Only set from the
# GPU proof in research/bench/topk_determinism.py (every shape and data kind there: canonical, repeatable, batch-independent).
DS_CANONICAL = False


def _deep_select(logits, row_ends, out, k, sorted_index: bool = False):
    torch.ops.deep_select.topk(logits, k, None, row_ends, False, sorted_index, None, out, None, -1, float("-inf"),
                               False, False)


def _routed_top_k_per_row_prefill(logits, ks, ke, dst, num_rows, stride0, stride1, k):
    canonical = _top_k_per_row_prefill(logits, ks, ke, dst, num_rows, stride0, stride1, k)
    if logits.shape[1] <= k:      # at most k keys: all of them, nothing to fix
        return
    if not canonical:             # fix the kernel's set and order (topk_canon.py)
        topk_canon.canonicalize(logits, ks, ke, dst[:num_rows], k)
        return
    lo = 0
    for a, b in canonical + [(num_rows, num_rows)]:     # the rows between DeepSelect's canonical row ranges
        if a > lo:
            topk_canon.canonicalize(logits[lo:a], ks[lo:a], ke[lo:a], dst[lo:a], k)
        lo = max(lo, b)


def _top_k_per_row_prefill(logits, ks, ke, dst, num_rows, stride0, stride1, k) -> list[tuple[int, int]]:
    """The top-k into dst; returns the row ranges whose answer is already canonical (DS_CANONICAL)."""
    segments = TOPK_SEGMENTS.get(ks.data_ptr())
    if segments is None or all(seg[3] == "vllm_cuda" for seg in segments):
        if TOPK_LOG is not None:
            TOPK_LOG.append((num_rows, logits.shape[1], "vllm_cuda"))
        _ORIG_TOPK(logits, ks, ke, dst, num_rows, stride0, stride1, k)
        return []
    canonical = []
    for lo, hi, col, backend in segments:
        if TOPK_LOG is not None:
            TOPK_LOG.append((hi - lo, -1, backend))
        if backend == "deep_select":
            # Row r's keys are [col, ke[r]); DeepSelect takes no row start, so the view starts at the request's
            # first key and the ends are relative to it. Indices come back relative to col, as vLLM's do to ks.
            _deep_select(logits[lo:hi, col:], ke[lo:hi] - col, dst[lo:hi], k, sorted_index=DS_CANONICAL)
            if DS_CANONICAL:
                canonical.append((lo, hi))
        else:
            _ORIG_TOPK(logits[lo:hi], ks[lo:hi], ke[lo:hi], dst[lo:hi], hi - lo, stride0, stride1, k)
    return canonical


def _topk_segments(start_idx, end_idx, query_start_loc_cpu, compressed_seq_lens_cpu, query_slice, top_k=512):
    """Per request of one indexer prefill chunk: its rows in the chunk, first key column and top-k backend."""
    qsl = [int(v) for v in query_start_loc_cpu[start_idx:end_idx + 1].tolist()]
    lens = [int(v) for v in compressed_seq_lens_cpu[start_idx:end_idx].tolist()]
    base = qsl[0]
    lo_q, hi_q = (0, qsl[-1] - base) if query_slice is None else (query_slice.start, query_slice.stop)
    segs, col = [], 0
    for r in range(end_idx - start_idx):
        lo, hi = max(qsl[r] - base, lo_q), min(qsl[r + 1] - base, hi_q)
        if hi > lo:
            backend = kernels.topk_backend(hi - lo, lens[r], top_k)
            if backend == "deep_select" and (col * 4) % _DS_INPUT_ALIGN:
                backend = "vllm_cuda"
            segs.append((lo - lo_q, hi - lo_q, col, backend))
        col += lens[r]
    return segs


def install_topk_router() -> None:
    """Route the fork's indexer prefill top-k (`torch.ops._C.top_k_per_row_prefill`) through the per-shape choice."""
    global _ORIG_TOPK, _DS_INPUT_ALIGN
    import vllm._custom_ops  # noqa: F401  registers torch.ops._C
    import vllm._deepselect_C  # noqa: F401
    import vllm.v1.attention.backends.mla.indexer as idx

    if _ORIG_TOPK is not None:
        return
    _DS_INPUT_ALIGN = int(torch.ops.deep_select.get_alignment_requirement()[0])
    _ORIG_TOPK = torch.ops._C.top_k_per_row_prefill
    torch.ops._C.top_k_per_row_prefill = _routed_top_k_per_row_prefill
    build = idx.build_prefill_chunk_metadata

    def build_and_route(start_idx, end_idx, query_start_loc, query_start_loc_cpu, uncompressed_seq_lens,
                        compressed_seq_lens, compressed_seq_lens_cpu, block_table, compress_ratio, query_slice=None,
                        **kw):
        md = build(start_idx, end_idx, query_start_loc, query_start_loc_cpu, uncompressed_seq_lens,
                   compressed_seq_lens, compressed_seq_lens_cpu, block_table, compress_ratio,
                   query_slice=query_slice, **kw)
        if md is not None:
            TOPK_SEGMENTS[md.cu_seqlen_ks.data_ptr()] = _topk_segments(
                start_idx, end_idx, query_start_loc_cpu, compressed_seq_lens_cpu, query_slice)
        return md

    idx.build_prefill_chunk_metadata = build_and_route


_ORIG_DECODE_TOPK = None
DECODE_TOPK_CANON = True      # research/bench/margin.py cases may turn it off (the before/after of the same process)
# Which top-k selects the decode rows: "prefill" (vLLM's top_k_per_row_prefill), "deep_select", or "kernel" (vLLM's
# decode dispatcher: cooperative/persistent). The cooperative kernel returned a wrong set, run to run, on a real
# 131,072-pool row (issues.md, job 7558); the prefill top-k gave a valid one every time.
DECODE_TOPK_PATH = "prefill"


def canonical_decode_topk(backend: str):
    """The indexer's decode top-k (`indexer_topk.get_indexer_topk` call site), then `topk_canon`: the same set and
    order for the same logits, as on the prefill path. vLLM's decode kernels return their set in an order that varies
    run to run (issues.md, jobs 7522, 7551), and the sparse attention sums in that order; the cooperative kernel also
    returned a wrong set on a real 131,072-pool row (7558), so the rows go to DECODE_TOPK_PATH. Rows end where the
    decode kernels end them; few long rows, so the split canonicalization."""
    from vllm.model_executor.layers import indexer_topk

    fn = _ORIG_DECODE_TOPK(backend)

    def run(logits, seq_lens, next_n, dst, k, max_seq_len):
        path = DECODE_TOPK_PATH if _ORIG_TOPK is not None else "kernel"
        if path == "deep_select" and not indexer_topk.is_deep_select_supported(logits, k):
            path = "prefill"
        if path == "deep_select":
            _deep_select(logits, fn._row_ends(seq_lens, next_n, logits.shape[0]), dst, k)
        elif path == "prefill":
            ke = fn._row_ends(seq_lens, next_n, logits.shape[0])
            _ORIG_TOPK(logits, torch.zeros_like(ke), ke, dst, logits.shape[0], logits.stride(0), logits.stride(1), k)
        else:
            fn(logits, seq_lens, next_n, dst, k, max_seq_len)
        if DECODE_TOPK_CANON and logits.shape[1] > k:
            ke = fn._row_ends(seq_lens, next_n, logits.shape[0])
            topk_canon.canonicalize_split(logits, torch.zeros_like(ke), ke, dst, k)

    return run


def install_decode_topk_canon() -> None:
    """Canonicalize the glm5next indexer's decode top-k (chunks of at most 256 tokens take that path)."""
    global _ORIG_DECODE_TOPK
    import vllm.models.glm5next.nvidia.sparse_indexer as si

    if _ORIG_DECODE_TOPK is not None:
        return
    _ORIG_DECODE_TOPK = si.get_indexer_topk
    si.get_indexer_topk = functools.cache(canonical_decode_topk)


# ---------------------------------------------------------------- kpool indexer: chunks that start inside a pool
# The fork's prefill K insert (`sparse_indexer._kpool_compress_insert`) pools each completing token with the 3 rows
# before it in the batch, so it assumes every chunk starts on a pool boundary (start % 4 == 0). A prefix hit resumes
# at the kept prompt's length and a chunk may end anywhere, so a chunk can start inside a pool: its first complete
# pool holds `start % 4` tokens of the earlier chunk, which sit in the request's tail ring (state_blocks[5], seeded at
# the end of every prefill chunk and kept with the context). The override masks that pool out of the batch-wide
# insert and, before the tail seed overwrites the ring, writes it with the same compress kernel from ring + batch
# rows (bitwise the pool an unsplit prefill writes: bf16 K and gate in both, same kernel).

@dataclass
class KpoolHeads:
    """One microbatch's prefill requests whose chunk starts inside a pool and completes it."""
    rows: torch.Tensor      # [U, kpool] int64: prefill-slice row of each pool token (ring tokens: any valid row)
    ring: torch.Tensor      # [U, kpool] bool: the token is from an earlier chunk (read from the tail ring)
    blocks: torch.Tensor    # [U] int64: the request's tail block
    last: torch.Tensor      # [U] int64: row of the pool's last token, whose slot is the pool's


KPOOL_HEADS: dict[int, KpoolHeads] = {}   # prefill-slice slot-mapping data_ptr -> heads (set per microbatch)
_KPOOL_PENDING: list[tuple] = []          # insert -> seed hand-off within one indexer call
_ORIG_KPOOL = None


def kpool_heads(starts, q_lens, row_starts, tail_blocks, first: int, row0: int, kpool: int):
    """Host side: the straddling pools of requests `first..` (the indexer's prefill requests). `row_starts[r]` is
    request r's first row in the batch, `row0` the prefill slice's first row. Returns numpy (rows, ring, blocks, last)
    or None."""
    import numpy as np

    rows, ring, blocks, last = [], [], [], []
    for r in range(first, len(starts)):
        m = int(starts[r]) % kpool
        need = kpool - m
        if m == 0 or int(q_lens[r]) < need:     # aligned, or the pool does not complete in this chunk (the ring has it)
            continue
        end = int(row_starts[r]) - row0 + need - 1
        j = np.arange(kpool)
        rows.append(np.maximum(end - (kpool - 1) + j, end - need + 1))
        ring.append(j < m)
        blocks.append(int(tail_blocks[r]))
        last.append(end)
    if not rows:
        return None
    return (np.stack(rows).astype(np.int64), np.stack(ring), np.asarray(blocks, dtype=np.int64),
            np.asarray(last, dtype=np.int64))


def kpool_heads_all(starts, q_lens, row_starts, tail_blocks, kpool: int):
    """`kpool_heads` of every request, rows counted from the batch's first row, and each head's request: built
    before the microbatch's one H2D copy, when the indexer's split into decodes and prefills is not known yet."""
    h = kpool_heads(starts, q_lens, row_starts, tail_blocks, 0, 0, kpool)
    req = [r for r in range(len(starts)) if int(starts[r]) % kpool and int(q_lens[r]) >= kpool - int(starts[r]) % kpool]
    return h, req


def kpool_heads_suffix(heads, req: list[int], first: int, row0: int):
    """The heads of requests `first..` with rows counted from `row0` (the indexer's prefill slice), from
    `kpool_heads_all`: a suffix of its arrays (numpy or device tensors, no copy but the row offsets), or None."""
    import bisect

    i = bisect.bisect_left(req, first)
    if heads is None or i == len(req):
        return None
    rows, ring, blocks, last = (a[i:] for a in heads)
    return (rows - row0 if row0 else rows), ring, blocks, (last - row0 if row0 else last)


_KPOOL_WINDOWS: dict = {}
_KPOOL_WINDOWS_OLD: list = []       # outgrown tables stay allocated: queued kernels may still read them


def kpool_windows(n: int, kpool: int, device) -> torch.Tensor:
    """[n, kpool] int64: row r's pool window r-kpool+1 .. r, clamped at 0. A view of one per-device table (grown on
    demand) instead of 7 small launches per DSA layer; the stage-0 host launches these before its first MoE."""
    key = (str(device), kpool)
    t = _KPOOL_WINDOWS.get(key)
    if t is None or t.shape[0] < n:
        m = max(n, 8192, 0 if t is None else 2 * t.shape[0])
        pos = torch.arange(m, device=device)
        if t is not None:
            _KPOOL_WINDOWS_OLD.append(t)
        t = (pos - (kpool - 1)).clamp_min(0)[:, None] + torch.arange(kpool, device=device)[None, :]
        _KPOOL_WINDOWS[key] = t
    return t[:n]


def kpool_compress_insert(k, gate_score, ape, kv_cache, slot_mapping, kpool, head_dim, round_scale, *, ops=None):
    """`sparse_indexer._kpool_compress_insert`, with straddling pools masked out and handed to the tail seed."""
    ops = ops or _ORIG_KPOOL["ops"]
    assert not _KPOOL_PENDING, "kpool: a straddling pool was not written (no tail seed after the insert)"
    n = slot_mapping.shape[0]
    heads = KPOOL_HEADS.get(slot_mapping.data_ptr()) if n else None
    if n >= kpool:
        write_mask = slot_mapping >= 0
        write_mask[:kpool - 1] = False          # rows whose pool starts before the slice
        if heads is not None:
            write_mask.index_fill_(0, heads.last, False)   # not `[idx] = False`: that copies the scalar H2D and syncs
        idx = kpool_windows(n, kpool, k.device)
        ops.kpool_compress_and_write_cache(kv_cache, k[idx], gate_score[idx], ape, slot_mapping.to(torch.int64),
                                           pool_size=kpool, head_dim=head_dim, write_mask=write_mask,
                                           round_scale=round_scale, write_cache=True, return_compressed=False)
    if heads is not None:
        _KPOOL_PENDING.append((heads, k, gate_score, ape, kv_cache, slot_mapping, kpool, head_dim, round_scale))


def kpool_seed_tail_cache(tail_kv_cache, key, gate_score, tslot, kpool, head_dim=128, *, ops=None, seed=None):
    """`kpool_ops.kpool_seed_tail_cache`, first writing the straddling pools from the ring the seed overwrites."""
    ops = ops or _ORIG_KPOOL["ops"]
    seed = seed or _ORIG_KPOOL["seed"]
    if _KPOOL_PENDING:
        heads, k, g, ape, kv_cache, sm, kp, hd, rs = _KPOOL_PENDING.pop()
        assert k.data_ptr() == key.data_ptr() and kp == kpool, "kpool: seed is not for the pending insert"
        ring = heads.ring[:, :, None]
        slot_k = torch.where(ring, tail_kv_cache[heads.blocks, 0], k[heads.rows])
        slot_s = torch.where(ring, tail_kv_cache[heads.blocks, 1], g[heads.rows])
        ops.kpool_compress_and_write_cache(kv_cache, slot_k, slot_s, ape, sm[heads.last].to(torch.int64),
                                           pool_size=kp, head_dim=hd, write_mask=None, round_scale=rs,
                                           write_cache=True, return_compressed=False)
    seed(tail_kv_cache, key, gate_score, tslot, kpool, head_dim)


def install_indexer_float() -> None:
    """`glue_fast.install_indexer_float`: glm5next `Indexer.forward` with the Triton BF16 -> FP32 cast (process-wide)."""
    from ssengine.model import glue_fast

    glue_fast.install_indexer_float()


def install_kda_split_heads() -> None:
    """`kda_schedule.install`: FlashInfer persistent KDA's head-split search in C++ (process-wide)."""
    from ssengine.model import kda_schedule

    kda_schedule.install()


def install_sm_h2d() -> None:
    """`sm_h2d.install_vllm`: vLLM's `async_tensor_h2d` copies by an SM kernel from mapped pinned memory
    (process-wide)."""
    from ssengine.model import sm_h2d

    sm_h2d.install_vllm()


def install_kpool_expand() -> None:
    """`kpool_fast.install`: the fork's indexer with the one-launch expansion into the top-k buffer (process-wide)."""
    from ssengine.model import kpool_fast

    kpool_fast.install()


def install_kpool_unaligned() -> None:
    """Route the fork's kpool prefill insert and tail seed through the overrides above (process-wide)."""
    global _ORIG_KPOOL
    import vllm.models.glm5next.nvidia.sparse_indexer as si
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kops

    if _ORIG_KPOOL is not None:
        return
    _ORIG_KPOOL = {"ops": types.SimpleNamespace(kpool_compress_and_write_cache=kops.kpool_compress_and_write_cache),
                   "seed": kops.kpool_seed_tail_cache}
    si._kpool_compress_insert = kpool_compress_insert
    kops.kpool_seed_tail_cache = kpool_seed_tail_cache


# ---------------------------------------------------------------- indexer prefill split

INDEXER_MAX_LOGITS_BYTES = 2 << 30   # one call's FP32 logits (the fork's default is 512 MiB)
INDEXER_ROW_ALIGN = 512              # query rows per DeepGEMM mqa_logits step at long context
_ORIG_INDEXER_SPLIT = None


def split_indexer_prefill_chunks(seq_lens: list[int], query_lens: list[int], workspace_size: int,
                                 max_logits_bytes: int = INDEXER_MAX_LOGITS_BYTES, request_offset: int = 0,
                                 align: int = INDEXER_ROW_ALIGN) -> list[tuple[slice, slice]]:
    """The fork's `_split_indexer_prefill_chunks` (requests packed while their keys fit the workspace and rows x keys
    fit the logits budget; one request too large for the budget is cut along its query rows), except that a request
    cut along its rows is cut into multiples of `align` rows when the budget holds at least that many.

    Why: DeepGEMM's `fp8_mqa_logits` takes the same time for 1 to 512 query rows at long context and grows per 512
    rows (L1 rows at 131,072 keys: 0.284 ms at 128 rows, 0.288 at 512, 0.573 at 1,024). The fork cut 1,024 rows at 512k
    context (131,328 pools) into 1,022 + 2 under its 512 MiB budget: 0.58 + 0.24 ms per layer instead of 0.57 (job
    7404). Here: one call of 1,024 rows; 8,192 rows go as 3,584 + 3,584 + 1,024."""
    chunks: list[tuple[slice, slice]] = []
    n, max_elems, end = len(seq_lens), max_logits_bytes // 4, 0
    while end < n:
        start, m, k = end, 0, 0
        while end < n:
            nm, nk = m + query_lens[end], k + seq_lens[end]
            if nk <= workspace_size and nm * nk <= max_elems:
                m, k, end = nm, nk, end + 1
            else:
                break
        if end == start:
            m, k, end = query_lens[end], seq_lens[end], end + 1
        max_q = max(1, max_elems // k) if k > 0 else max(1, m)
        if max_q < m and max_q >= align:
            max_q -= max_q % align
        req = slice(start + request_offset, end + request_offset)
        chunks += [(req, slice(q, min(q + max_q, m))) for q in range(0, m, max_q)]
    return chunks


def install_indexer_split() -> None:
    """Route the indexer metadata builder's prefill split through `split_indexer_prefill_chunks` (process-wide; the
    builder's own `max_logits_bytes`, from VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, is replaced by
    INDEXER_MAX_LOGITS_BYTES)."""
    global _ORIG_INDEXER_SPLIT
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder as B

    if _ORIG_INDEXER_SPLIT is not None:
        return
    _ORIG_INDEXER_SPLIT = B.__dict__["_split_indexer_prefill_chunks"]

    def split(seq_lens_cpu, query_lens_cpu, workspace_size, max_logits_bytes, request_offset=0):
        return split_indexer_prefill_chunks([int(v) for v in seq_lens_cpu.tolist()],
                                            [int(v) for v in query_lens_cpu.tolist()], workspace_size,
                                            INDEXER_MAX_LOGITS_BYTES, request_offset)

    B._split_indexer_prefill_chunks = staticmethod(split)


# ---------------------------------------------------------------- sparse MLA launch size

SPARSE_MLA_MAX_ROWS = 32768
_ORIG_TRTLLM_MLA = None


def install_sparse_mla_split(max_rows: int = SPARSE_MLA_MAX_ROWS) -> None:
    """TRTLLM sparse MLA (config 25) at more than 32,768 query rows: on B200 a 65,536-row launch returns without
    doing the work (M2: 0.72 ms at 64k vs 11.7 ms target), and the target splits at 32,768 rows (3_layers.md §6).
    The fork's backend imports `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` at call time, so the split
    wraps it there: launches of <= `max_rows` rows writing into one output, no extra copy."""
    global _ORIG_TRTLLM_MLA
    import flashinfer.decode as fd

    if _ORIG_TRTLLM_MLA is not None:
        return
    _ORIG_TRTLLM_MLA = orig = fd.trtllm_batch_decode_with_kv_cache_mla

    def split(query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, block_tables,
              seq_lens, max_seq_len, *args, **kw):
        B = query.shape[0]
        if B <= max_rows or args or kw.get("return_lse") or kw.get("out") is not None:
            return orig(query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim,
                        block_tables, seq_lens, max_seq_len, *args, **kw)
        out = torch.empty(B, query.shape[1], query.shape[2], kv_lora_rank, dtype=torch.bfloat16,
                          device=query.device)
        lens = kw.pop("sparse_mla_top_k_lens", None)
        for a in range(0, B, max_rows):
            b = min(B, a + max_rows)
            extra = {} if lens is None else {"sparse_mla_top_k_lens": lens[a:b]}
            orig(query[a:b], kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim,
                 block_tables[a:b], None if seq_lens is None else seq_lens[a:b], max_seq_len, out=out[a:b],
                 **kw, **extra)
        return out

    fd.trtllm_batch_decode_with_kv_cache_mla = split


def install_moe_buckets() -> None:
    """The trtllm NVFP4 MoE tunes and looks up its tactics at the serving buckets (`moe_tuning.install_buckets`)."""
    from ssengine.model import moe_tuning

    moe_tuning.install_buckets()


# ---------------------------------------------------------------- host trims (M3: stage-0 host time, job 7091)
#
# Per-call Python that returns the same answer on every call of a serving process, hoisted or memoized. None of them
# changes a launched kernel or its arguments (host time: research/measured/issues.md, "Host launch time", jobs 7091/7113; the
# chain/edge checks stay exact).

AUTOTUNE_MEMO: dict = {}
AUTOTUNE_MEMO_STATS = {"hit": 0, "miss": 0, "bypass": 0}
_AUTOTUNE_MEMO_MAX = 4096
_ORIG_CHOOSE_ONE = None
_ORIG_MM_FP4 = None


def _memo_sig(x):
    if isinstance(x, torch.Tensor):
        return (tuple(x.shape), x.dtype, x.device)
    if x is None or isinstance(x, (int, float, bool, str, torch.dtype)):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_memo_sig(v) for v in x)
    raise TypeError


def install_autotune_memo() -> None:
    """FlashInfer `AutoTuner.choose_one` outside tuning: a lookup of the winner cache by (op, runners, input shapes)
    that takes ~4% of stage-0 host time (lock, policy, bucket search, per-runner keys). Outside tuning its answer
    depends only on the op, the runners' hashes (their parameters), the inputs' shapes/dtypes and the winner cache,
    which changes only while tuning; so it is memoized on exactly those, cleared whenever tuning is active or any
    autotune override (skip ops, buckets) is in effect, and by `clear_autotune_memo` after the warm-up. The memo
    keeps the runner index, so the caller's own (fresh) runner object is returned."""
    global _ORIG_CHOOSE_ONE
    from flashinfer.autotuner.autotuner import AutoTuner

    if _ORIG_CHOOSE_ONE is not None:
        return
    _ORIG_CHOOSE_ONE = orig = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if (self.is_tuning_mode or self._effective_skip_ops or self._get_override_stack()
                or self._effective_measure_policy is not None):
            AUTOTUNE_MEMO.clear()
            AUTOTUNE_MEMO_STATS["bypass"] += 1
            return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
        try:
            key = (custom_op, tuple((type(r), hash(r)) for r in runners), _memo_sig(list(inputs)))
        except TypeError:
            AUTOTUNE_MEMO_STATS["bypass"] += 1
            return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
        hit = AUTOTUNE_MEMO.get(key)
        if hit is not None:
            AUTOTUNE_MEMO_STATS["hit"] += 1
            return runners[hit[0]], hit[1]
        AUTOTUNE_MEMO_STATS["miss"] += 1
        runner, tactic = orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
        idx = next(i for i, r in enumerate(runners) if r is runner)
        if len(AUTOTUNE_MEMO) >= _AUTOTUNE_MEMO_MAX:
            AUTOTUNE_MEMO.clear()
        AUTOTUNE_MEMO[key] = (idx, tactic)
        return runner, tactic

    AutoTuner.choose_one = choose_one


def clear_autotune_memo() -> None:
    AUTOTUNE_MEMO.clear()


def install_mm_fp4_skip_check() -> None:
    """`flashinfer.mm_fp4` re-validates backend, capability and problem size on every call (`inspect` bind and
    checks, ~1% of host time); the checks only raise. vLLM imports `flashinfer.mm_fp4` at call time and passes an
    explicit backend, so the wrapper passes `skip_check=True` (FlashInfer's own opt-out) after the warm-up has
    run the same shapes through the checks."""
    global _ORIG_MM_FP4
    import flashinfer

    if _ORIG_MM_FP4 is not None:
        return
    _ORIG_MM_FP4 = orig = flashinfer.mm_fp4

    def mm_fp4(*a, **kw):
        if kw.get("backend", "auto") != "auto":
            kw.setdefault("skip_check", True)
        return orig(*a, **kw)

    flashinfer.mm_fp4 = mm_fp4


def plain_parameters(modules) -> int:
    """vLLM's parameter subclasses override `__torch_function__` (a Python round trip on every op that takes a
    weight); after loading they are only tensors. Replace each by a plain `nn.Parameter` over the same storage,
    keeping its attributes. Returns the number replaced."""
    n = 0
    for root in modules:
        for mod in root.modules():
            for name, p in list(mod._parameters.items()):
                if p is None or type(p) is torch.nn.Parameter or not isinstance(p, torch.nn.Parameter):
                    continue
                new = torch.nn.Parameter(p.data, requires_grad=False)
                for k, v in vars(p).items():
                    setattr(new, k, v)
                mod._parameters[name] = new
                n += 1
    return n


def install_host_trims(modules) -> dict:
    """All of the above plus vLLM's env cache (the engine enables it after start-up: envs are read on every linear
    call otherwise). Call after the weights are loaded."""
    import vllm.envs as envs

    envs.enable_envs_cache()
    install_autotune_memo()
    install_mm_fp4_skip_check()
    return {"plain_parameters": plain_parameters(modules)}
