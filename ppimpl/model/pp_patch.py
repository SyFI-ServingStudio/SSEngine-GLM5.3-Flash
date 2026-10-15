"""Small overrides of the fork's glm5next modules for one pipeline stage (PLAN.md §1).

Nothing here edits `third_party/vllm`; every override is a per-instance attribute or a scoped module patch.

- `stage_scope`: while the model is built, vLLM's layer range and first/last-rank checks follow our stage, so
  `make_layers` creates only this stage's decoder layers (`PPMissingLayer` elsewhere), `embed_tokens` exists only on
  stage 0 and `norm`/`lm_head` only on stage 3. vLLM's own process groups stay single-rank (tp = pp = 1).
- `stage_weights`: iterate only this stage's tensors out of the checkpoint (names mapped to `Glm5NextForCausalLM`).
- Stage edge: the sender applies its last layer's deferred `hc_post` and sends the 4-stream residual
  `[T, 4 * 4096]` BF16; the receiver's first layer sees `post = None` and runs a standalone `hc_pre` (the fork's
  decoder layer already does this for any layer without incoming state).
- Kernel routing that follows the design rather than the fork (`measured/kernels_identity.md`, Decisions):
  - every attention/FFN mHC boundary -> DeepGEMM `mega_mhc`, non-shifted (config 7), `MegaMhcPostPre`;
  - KDA prefill -> Dao causal-conv1d channel-last, one launch per sequence, BF16 weight, conv state written (config
    4), then FlashInfer `recurrent_kda` cute-dsl-persistent with FP32 state gathered/scattered per request (config 5),
    `kda_prefill_forward`;
  - indexer top-k -> per request and shape, DeepSelect or `topKPerRowPrefill`, whichever profiler row is faster
    (configs 20/21), `install_topk_router`.
"""

from __future__ import annotations

import contextlib
import json
import re
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ppimpl import kernels

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
    return torch.cat([x] * n, dim=1).view(x.shape[0], n, x.shape[1])


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
    any_initial: bool


def make_kda_plan(offsets: list[int], num_heads: int, device: torch.device) -> tuple:
    from flashinfer.kda_kernels.persistent.forward import make_plan

    return make_plan(offsets, num_heads, device)


def kda_prefill_forward(self, qkv_proj_states, g1, beta, core_attn_out) -> None:
    """`Glm5NextLinearAttention._forward` for prefill chunks (configs 4 and 5).

    Short conv: Dao causal-conv1d channel-last, one launch per sequence over its rows of the packed q|k|v, BF16
    merged weight, the request's conv state passed as initial state (continuation chunks) and its final state
    written in place. Recurrence: FlashInfer persistent KDA (gate and beta sigmoid in kernel, lower bound -5) over
    an FP32 [N, H, 128, 128] state gathered from the requests' state blocks (zero for a fresh request) and
    scattered back.
    """
    import causal_conv1d_cuda
    from flashinfer.kda_kernels.persistent.forward import fwd
    from vllm.forward_context import get_forward_context

    meta = get_forward_context().attn_metadata[self.prefix]
    assert isinstance(meta, KdaPrefillMeta)
    conv_state, rec_state = self.kv_cache   # bound by vLLM: [blocks, W-1, C] bf16 (SD layout), [blocks, H, D, D] fp32
    T = meta.num_tokens
    qkv = qkv_proj_states[:T]
    H, D = self.local_num_heads, self.head_dim
    P = H * D
    # q, k, v convolved into three contiguous [T, H*D] buffers (FlashInfer takes contiguous q/k/v): one launch per
    # sequence and per q/k/v instead of one over the packed q|k|v plus three copies (M2: 3 x 0.3 ms at 16k)
    q, k, v = (torch.empty(T, P, dtype=qkv.dtype, device=qkv.device) for _ in range(3))
    w = self._pp_conv_weight
    for s, e, slot, has_init in meta.seqs:
        state = conv_state[slot]                        # [W-1, 3P], SD layout
        init_all = state.clone() if has_init else None
        for j, out in enumerate((q, k, v)):
            cols = slice(j * P, (j + 1) * P)
            init = None if init_all is None else init_all[:, cols].t().unsqueeze(0)
            causal_conv1d_cuda.causal_conv1d_fwd(qkv[s:e, cols].t().unsqueeze(0), w[cols], None, None, init,
                                                 out[s:e].t().unsqueeze(0), state[:, cols].t().unsqueeze(0), True)
    q, k, v = (t.view(1, T, H, D) for t in (q, k, v))
    g = g1[:, :T]
    b = beta[:, :T].contiguous()
    if meta.any_initial:
        state = rec_state.index_select(0, meta.slots)
        for i, (_, _, _, has_init) in enumerate(meta.seqs):
            if not has_init:
                state[i].zero_()
    else:
        state = torch.zeros(len(meta.seqs), H, D, D, dtype=torch.float32, device=qkv.device)
    fwd(q, k, v, g, b, D ** -0.5, core_attn_out[:, :T], self._pp_a_log, self.dt_bias, self.kda_lower_bound,
        initial_state=state, final_state=state, cu_seqlens=meta.plan[0], plan=meta.plan)
    rec_state.index_copy_(0, meta.slots, state)


def route_kda(attn) -> None:
    """Install `kda_prefill_forward` on one `Glm5NextLinearAttention` (after its weights are loaded)."""
    def w(m):
        return m.weight.view(m.weight.size(0), m.weight.size(-1))

    attn._pp_conv_weight = torch.cat([w(attn.q_conv1d), w(attn.k_conv1d), w(attn.v_conv1d)]).to(
        torch.bfloat16).contiguous()
    assert attn.q_conv1d.bias is None
    attn._pp_a_log = attn.A_log.detach().reshape(-1).contiguous()
    attn._forward = types.MethodType(kda_prefill_forward, attn)


# ---------------------------------------------------------------- indexer top-k


# ks.data_ptr() -> [(first row, end row, first key column, backend)] of one indexer prefill chunk. Filled when vLLM's
# indexer metadata builder makes the chunk (`build_prefill_chunk_metadata`, wrapped below with the host-side request
# boundaries it already has), read by the top-k router when the fork's indexer calls `top_k_per_row_prefill`.
TOPK_SEGMENTS: dict[int, list[tuple[int, int, int, str]]] = {}
TOPK_LOG: list[tuple[int, int, str]] | None = None   # (rows, keys, backend) per launch, when a caller wants them
_ORIG_TOPK = None
_DS_INPUT_ALIGN = 16


def _deep_select(logits, row_ends, out, k):
    torch.ops.deep_select.topk(logits, k, None, row_ends, False, False, None, out, None, -1, float("-inf"), False,
                               False)


def _routed_top_k_per_row_prefill(logits, ks, ke, dst, num_rows, stride0, stride1, k):
    segments = TOPK_SEGMENTS.get(ks.data_ptr())
    if segments is None or all(seg[3] == "vllm_cuda" for seg in segments):
        if TOPK_LOG is not None:
            TOPK_LOG.append((num_rows, logits.shape[1], "vllm_cuda"))
        return _ORIG_TOPK(logits, ks, ke, dst, num_rows, stride0, stride1, k)
    for lo, hi, col, backend in segments:
        if TOPK_LOG is not None:
            TOPK_LOG.append((hi - lo, -1, backend))
        if backend == "deep_select":
            # Row r's keys are [col, ke[r]); DeepSelect takes no row start, so the view starts at the request's
            # first key and the ends are relative to it. Indices come back relative to col, as vLLM's do to ks.
            _deep_select(logits[lo:hi, col:], ke[lo:hi] - col, dst[lo:hi], k)
        else:
            _ORIG_TOPK(logits[lo:hi], ks[lo:hi], ke[lo:hi], dst[lo:hi], hi - lo, stride0, stride1, k)


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


# ---------------------------------------------------------------- forced routing (M2)


def force_routing(gate, source) -> None:
    """The MoE gate still runs its GEMM, then hands the experts `source()` (FP32 [>= T, 288] logits) instead of
    its output; when `source()` is None the real router output goes through."""
    orig = gate.forward

    def forward(x):
        out = orig(x)
        logits = source()
        if logits is None:
            return out
        forced = logits[: x.shape[0]]
        return (forced, out[1]) if isinstance(out, tuple) else forced

    gate.forward = forward


def forced_routing_logits(per_expert_batches: list[int], num_tokens: int, top_k: int, device) -> torch.Tensor:
    """FP32 router logits whose top-k, under DeepSeek-V3 routing with the layer's correction bias, gives exactly
    `per_expert_batches` tokens per expert (profiling/runners/moe/exact_topk.py builds the assignment). Selected
    logits saturate the sigmoid at 1 and the others at 0, so a bias spread below 1 cannot change the choice."""
    from profiling.runners.moe.exact_topk import exact_topk_ids

    ids = exact_topk_ids(num_tokens=num_tokens, top_k=top_k, per_expert_batches=tuple(per_expert_batches))
    logits = torch.full((num_tokens, len(per_expert_batches)), -30.0, dtype=torch.float32, device=device)
    logits.scatter_(1, torch.tensor(ids, dtype=torch.int64, device=device), 30.0)
    return logits


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
