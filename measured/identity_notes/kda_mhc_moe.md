# Call-path identity: configs 4, 5, 1, 7, 33, 30 (KDA conv / KDA prefill / mHC / NVFP4 MoE)

Paths: `WT` = <workspace>/wt-pp-main, `FORK` = <workspace>/pp-impl/third_party/vllm (HEAD 892da0822f),
`FI070` = pp-impl/.venv/.../site-packages/flashinfer (0.7.0). Checkpoint config: hidden 4096, hc_mult 4, hc_eps 1e-6,
hc_sinkhorn_iters 20, rms_norm_eps 1e-5, KDA 64 heads x 128, short_conv_kernel_size 4, linear_lower_bound -5.0, 288 experts,
top-8, n_group 1, topk_group 1, routed_scaling_factor 2.5, scoring sigmoid, noaux_tc, swiglu_limit 10.0,
moe_router_dtype float32, routers + shared experts BF16 (quant `ignore`).

Note on "-dirty" commits: the frozen DB's `_profile_run.profiler_git_hash` is `92bf581bad…-dirty` for configs 4 and 7, i.e. the
runner was uncommitted. The runner first appears in a commit whose parent is 92bf581bad (4: `2de7cdcc`, 7: `9741a317`, both
2026-10-08, 25-35 min later); the runner file is byte-identical from that commit to HEAD. The dirty-tree content at measurement time
is reconstructed from those commits (exact dirty state unverified).

---

## Config 4 — gdn_causal_conv_prefill / dao_channellast
- **positions**: first_kda_dense / kda_dense / kda_moe `.kda.short_conv_prefill` (144 rows, channels 24576 at TP1, W=4).
- **RUNNER** (`WT/profiling/runners/attention/gdn_causal_conv_prefill_dao_channellast.py`, 2de7cdcc == HEAD; env
  `causal_conv1d_env` = host python + `~/profile_envs/causal_conv1d`, Dao-AILab causal-conv1d 1.7.0, `causal_conv1d_cuda` .so):
  `causal_conv1d.causal_conv1d_fn(x, weight, None, initial_states=None, return_final_states=True, final_states_out=<slot view>,
  activation="silu")` called **once per sequence** (`_invoke` L179-192). x = `(1, C, L)` channel-last view (stride(1)==1) of packed
  `(T, C)` BF16 (L146, L172); weight BF16 `(C, W)` contiguous; no bias; final state written into `(slots, W-1, C)` store via
  transposed view (L69-70). Timed: CUPTI on `causal_conv1d_channellast_fwd_kernel` only (L337), B launches per row.
- **FORK**: `FORK/vllm/models/glm5next/common/kda.py:570-582` →
  `vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn` (`causal_conv1d.py:481`), **one Triton launch**
  `_causal_conv1d_fwd_kernel[grid]` (`causal_conv1d.py:712`) over the whole varlen non-spec batch (q|k|v merged into one conv,
  kda.py:500-516), x = `qkv.transpose(0,1)` → channel-last detection `is_channel_last` (causal_conv1d.py:576), `conv_states`
  (bf16, `kda_state_dtype` = model dtype), `cache_indices`, `has_initial_state`, `query_start_loc`, `metadata`; bias None
  (`bias=False`, kda.py:247-266). Conv weights are **FP32** (`params_dtype=torch.float32`, kda.py:251/258/265; merged
  `_merged_conv_weight` kda.py:505-514), not BF16.
- **VERDICT**: DIFFERENT — Dao CUDA `causal_conv1d_channellast_fwd_kernel` ×B launches (BF16 weight, no history) vs vLLM Triton
  `_causal_conv1d_fwd_kernel` ×1 launch (FP32 weight, indexed state read/write, has_initial_state).
- **VERSIONS**: rows: causal-conv1d 1.7.0 (CUDA, sm_100 SASS). Ours: no causal-conv1d package; vLLM Triton kernel under triton 3.7.1.
- **COVERED-BY-DB**: no — different library/kernel.
- **NOTES**: the same kind has `vllm_triton` rows (144 each) from cu130-3f667d7e (82c85c3203, 82cc234a97); `causal_conv1d.py`
  changed 3f667d7eb..892da0822f only in a docstring (8c1557a79c) and the image's triton is 3.7.1 = ours, so those rows are the
  likely-covering backend (their weight dtype/args not checked here — unverified).

## Config 5 — kda_chunk_prefill / flashinfer_cute_persistent
- **positions**: first_kda_dense / kda_dense / kda_moe `.kda.chunk_prefill` (512 rows).
- **RUNNER** (`WT/profiling/runners/attention/kda_chunk_prefill_flashinfer.py` @703dee714a, identical at HEAD; env
  `flashinfer_kda_env` = flashinfer-python 0.7.2.dev20261008 @73b63f88, nvidia-cutlass-dsl 4.8.0):
  `flashinfer.kda.recurrent_kda(q=q.contiguous(), k=…, v=…, g=raw bf16 [1,T,H,K], beta=raw bf16 logits [1,T,H], A_log fp32 [H],
  dt_bias fp32 [H*K], scale=K**-0.5, initial_state fp32 [N,H,V,K] (in place), output_final_state=False,
  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=-5, cu_seqlens int64, output=prealloc bf16,
  beta_is_logit=True, backend="cute-dsl-persistent")` (L176-196) → `flashinfer/kda_prefill_persistent.py`
  `kda_kernels.persistent.forward.fwd`. Timed: CUPTI sum of all launches incl. 3 `.contiguous()` copies (L290).
- **FORK**: `kda.py:651-666` → `self._flashkda_prefill` → `torch.ops._flashkda_C.fwd(...)` (kda.py:380-397; q/k/v/g
  `.contiguous()`, raw beta, A_log, dt_bias, lower_bound, initial_state, final_state workspace, cu_seqlens) when
  `_resolve_kda_prefill_backend` (kda.py:127-150) returns "flashkda": `additional_config.kda_prefill_backend` defaults "auto"
  (kda.py:322-329); B200 major 10, head_dim 128, bf16, lower_bound -5 → **flashkda**. Triton `chunk_kda_with_fused_gate`
  (kda.py:670) only if "triton" is forced. The fork has no FlashInfer KDA call in glm5next.
  Launch configs: no `kda_prefill_backend` / `additional_config` in `WT/presets/alignment/*` (glm53_flash_fp8_b200_tp4_ep4
  campaign extra_args only set EP, fp8 KV, mm limits, KV bytes, max-num-seqs), nor in pp-impl outside the fork. The
  `kda_prefill_backend: flashinfer_cute_persistent` hits in `WT/artifact/pp_blog/**` are **simulator backend overrides**, not vLLM
  config. FlashKDA source: `FORK/cmake/external_projects/flashkda.cmake` (vllm-project/FlashKDA @ b59532f1), built
  `vllm/_flashkda_C.abi3.so` present.
- **VERDICT**: DIFFERENT — FlashInfer CuTe-DSL persistent KDA (`recurrent_kda(backend="cute-dsl-persistent")`) vs FlashKDA
  `_flashkda_C.fwd`.
- **VERSIONS**: rows: FlashInfer nightly 0.7.2.dev20261008 + cutlass-dsl 4.8.0. Ours: FlashInfer 0.7.0 has **no**
  `cute-dsl-persistent` and no `kda_kernels/persistent/` (backends auto/cute-dsl/cake; `kda.py:73`); its `cute-dsl` prefill is
  `kda_prefill_cute.py` → `kda_kernels.kda_chunked_bt16` (BT=16 chunked CuTe kernel, also present in the nightly as a separate
  backend) — not the persistent kernel under another name. cutlass-dsl 4.7.1 ours.
- **COVERED-BY-DB**: no — different callable (and the measured kernel does not exist in our FlashInfer).
- **NOTES**: kind also has `flashkda` rows (1536, "flash_kda 0.0.1+17a037d", profiler 1a7f985e-dirty); registration comment cites
  FlashKDA@17a037d, but our fork pins FlashKDA b59532f1 — version mismatch, coverage of that backend unverified.

## Config 1 — mhc_pre_rms_norm / vllm_tilelang
- **positions**: first_kda_dense.attn_mhc_pre (layer 0 only; 68 rows).
- **RUNNER** (`WT/profiling/runners/mhc/mhc_pre_rms_norm_vllm_tilelang.py` @f8bd625a4d; env vllm_env = image
  vibesim-profiler-vllm:cu130, fork 0b8edfb): `vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang(residual bf16 (T,4,4096),
  fn fp32 (24,16384) ×0.01 randn, hc_scale fp32 [0.5]*3, hc_base zeros fp32 (24), rms_eps=1e-6, hc_pre_eps=1e-6,
  hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20, norm_weight=ones bf16 (4096), norm_eps=1e-6)` (L33-46;
  constants `_deepseek_v4.py` @f8bd625a4d L16-22, L65-77). Timed: CUPTI all kernels (L76). In 0b8edfb the call is
  DeepGEMM `tf32_hc_prenorm_gemm` (if `is_deep_gemm_supported()`, n_splits = compute_num_split(64, 16384, cdiv(T,64))) else TileLang
  prenorm GEMM, then `mhc_pre_big_fuse_with_norm_tilelang` (`git show 0b8edfb56:…/mhc/tilelang.py` L165-256).
- **FORK**: `model.py:491-503` (layer 0: `hc_expand` copy, then `self.hc_pre`) → `MHCPreOp.forward_cuda` (`layers/mhc.py:110-138`,
  `enabled()` always True) → `torch.ops.vllm.mhc_pre_tilelang` = `kernels/mhc/tilelang.py:542` → `_hc_prenorm_gemm_outputs`
  (tilelang.py:22-71: DeepGEMM `tf32_hc_prenorm_gemm(x, fn, out, sqrsum, n_splits)` L57 when `is_deep_gemm_supported()`
  [`utils/deep_gemm.py:110-115`, VLLM_USE_DEEP_GEMM default 1], same split formula) + `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL(...,
  norm_weight, norm_eps)`. Args: residual (T,4,4096) bf16 from `hc_expand(...).contiguous()`, hc_attn_fn/scale/base fp32,
  rms_eps = 1e-5, hc_pre/sinkhorn eps 1e-6, post_mult 2.0, sinkhorn 20, norm_weight = input_layernorm bf16, norm_eps 1e-5.
- **VERDICT**: SAME-FN-DIFF-ARGS — same `mhc_pre_tilelang` (DeepGEMM TF32 prenorm GEMM + TileLang big-fuse w/ RMSNorm); only scalar
  eps (1e-6 vs 1e-5) and weight values differ; cannot change launches. The fork's preceding `hc_expand` copy is outside the call.
- **VERSIONS**: rows: fork 0b8edfb, tilelang 0.1.9, DeepGEMM pin 891d57b4 (deepgemm.cmake @0b8edfb). Ours: 892da0822f, tilelang
  0.1.12, DeepGEMM e1f418c2 (deepgemm.cmake:36). Sources changed: `kernels/mhc/tilelang.py` (+/-894), `tilelang_kernels.py`
  (+1131/-69; e.g. e52be1a62d TileLang warmup migration, 74d3b799e1 prenorm-GEMM fix, 9b49f92344, 5fcc6e7c73),
  `utils/deep_gemm.py` (+530), DeepGEMM bump (929aa2a7ed, 8263ea12bd, eb87980585).
- **COVERED-BY-DB**: no — kernel sources (TileLang kernels, DeepGEMM pin) and TileLang version changed.
- **NOTES**: whether DeepGEMM was importable in the cu130 image (else TileLang prenorm GEMM ran) — unverified.

## Config 7 — mhc_fused_post_pre_rms_norm / deepgemm_mega_nonshifted
- **positions**: dsa_moe / kda_dense / kda_moe `.attn_mhc_post_pre` and `.ffn_mhc_post_pre`, first_kda_dense.ffn_mhc_post_pre (88 rows).
- **RUNNER** (`WT/profiling/runners/mhc/mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted.py`, 9741a317 == HEAD; env
  `vllm_upstream_fork_env` = alignment fork 892da0822f venv): `vllm.utils.deep_gemm.mega_mhc(x bf16 (T,4096), residual, shifted_prev_mix=None,
  post_mix, comb_res_mix, fn, mix_scales, mix_bases (non-zero), hc_mult=4, eps 1e-6, post_scale 2.0, 20 sinkhorn iters,
  rmsnorm_weight (non-unit), rmsnorm_eps 1e-6, new_residual, new_prev_mix=None, new_post_mix, new_comb_res_mix, y_bf16)`
  (L99-129) → one persistent `sm100_mega_mhc_impl<…, kIsShifted=false, …>`. CUPTI all kernels (L188).
- **FORK**: every boundary goes `model.py:504-515` (attn, layers ≥1) and `model.py:530-540` (ffn) → `hc_fused_post_pre`
  (model.py:591-620, n_splits=1, tile_n=1) → `MHCFusedPostPreOp.forward_cuda` (`layers/mhc.py:738-774`) →
  `torch.ops.vllm.mhc_fused_post_pre_tilelang` (`kernels/mhc/tilelang.py:814`): T ≤ 32 (`tilelang_kernels.py:64,77`) →
  `_MHC_FUSED_TILELANG_KERNEL` + `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL`; T > 32 (prefill) → `_MHC_POST_TILELANG_KERNEL` (L944) +
  DeepGEMM `tf32_hc_prenorm_gemm` (via `_hc_prenorm_gemm_outputs`) + `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` = 3 launches.
  `mega_mhc` is dispatched only by `vllm/models/deepseek_v41/nvidia/ops/mhc.py:229` (`can_use_mega_mhc`); no glm5next path
  reaches it (grep of `mega_mhc` in vllm/).
- **VERDICT**: DIFFERENT — DeepGEMM `mega_mhc` (non-shifted, 1 persistent launch) vs TileLang post + DeepGEMM TF32 prenorm GEMM +
  TileLang big-fuse (3 launches for prefill).
- **VERSIONS**: rows ran our fork 892da0822f (DeepGEMM e1f418c2), same venv as ours — version matches, callable does not.
- **COVERED-BY-DB**: no — different callable.
- **NOTES**: runner docstring itself states "No vLLM model dispatches the non-shifted form yet" (L22).

## Config 33 — mhc_fused_post_pre_rms_norm / vllm_tilelang at final_mhc_post
- **positions**: final_mhc_post (69 rows).
- **RUNNER** (`WT/profiling/runners/mhc/mhc_fused_post_pre_rms_norm_vllm_tilelang.py` @f8bd625a4d; image cu130, fork 0b8edfb):
  `mhc_fused_post_pre_tilelang(x bf16 (T,4096), residual, post_mix, comb_mix, fn, hc_scale, hc_base zeros, 1e-6, 1e-6, 1e-6, 2.0,
  20, n_splits=1, tile_n=1, norm_weight ones, norm_eps 1e-6)` (L36-54) = post + next pre's prenorm GEMM + big-fuse/RMSNorm
  (0b8edfb: T ≤ 16 `mhc_fused_tilelang`, else `mhc_post_tilelang` + DeepGEMM/TileLang prenorm GEMM, then
  `mhc_pre_big_fuse_with_norm_tilelang`).
- **FORK**: last layer `model.py:551-554`: `self.hc_post` → `MHCPostOp.forward_cuda` (`layers/mhc.py:533-542`) →
  `torch.ops.vllm.mhc_post_tilelang` (`kernels/mhc/tilelang.py:791-811`, one `_MHC_POST_TILELANG_KERNEL` launch, L802), then
  `hc_contract` = `x.mean(dim=1)` (`layers/mhc.py:974-976`, one torch reduction); final RMSNorm is the separate `self.norm`
  (model.py, `Glm5NextModel.forward` after the loop). No next pre, no fn GEMM, no Sinkhorn.
- **VERDICT**: DIFFERENT — fused post+pre+RMSNorm (3 launches for prefill) vs `mhc_post_tilelang` + torch mean (2 launches); the
  row over-counts the prenorm GEMM and big-fuse.
- **VERSIONS**: rows 0b8edfb / tilelang 0.1.9; ours 892da0822f / tilelang 0.1.12; TileLang mHC sources changed (see config 1).
- **COVERED-BY-DB**: no — different callable (and version).
- **NOTES**: `mhc_terminal_head` (`WT/profiling/kernels/mhc_terminal_head.py`) is the DSv4 head: `mhc_post_tilelang` + MTP-buffer
  copy + `hc_head_fused_kernel_tilelang` + RMSNorm, gated to H200 (`_deepseek_v4.py` @f8bd625a4d L9-15 says GLM-5.3-Flash has no
  hc_head). It does not match the GLM ending either; only its first launch (`mhc_post_tilelang`) is shared.

## Config 30 — nvfp4_fused_moe / flashinfer_trtllm_sm100
- **positions**: dsa_moe / kda_moe `.moe.routed_rank0.fused_moe` (73 rows at TP1: E=288 local 288, H 4096, I 2048, top-8,
  deepseek_v3, n_group 1, topk_group 1, scale 5/2). 60 rows (T ≤ 8192) @e3c5fcf4ef, 13 rows (T 9216..65536) @0219dfc274.
- **RUNNER** (`WT/profiling/runners/moe/nvfp4_fused_moe.py`; env vllm_env = image cu130-3f667d7e, FlashInfer 0.6.18):
  `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` (monolithic; L432) with `routing_logits` FP32 forced logits from
  `exact_topk_ids` (L379-384, `_forced_routing` L235-237), `routing_bias` FP32 zeros, `n_group=1, topk_group=1,
  routed_scaling_factor=2.5, routing_method_type=2` (DeepSeekV3), hidden pre-quantized **outside** the timed closure by
  `vllm._custom_ops.scaled_fp4_quant(src, 1.0, is_sf_swizzled_layout=False)` (linear SF, L46-48), weights random uint8 + unit FP8
  scales through the image's `prepare_static_weights_for_trtllm_fp4_moe(…, is_gated_activation=True)` (L50-53, L107-143),
  `gemm1_bias/alpha/beta=None`, **`gemm1_clamp_limit=None`** (L448), scale scalars = ones, `activation_type=3` (SwiGLU),
  `do_finalize=True`, preallocated `output`, `enable_pdl=True`; `tune_max_num_tokens`: e3c5 rows → FlashInfer default 8192;
  0219 rows → `max(8192, next_pow2(T))` (L170-180, L413-416). Autotune: one in-process `autotune()` pass wrapped by
  `autotune_cached` (persisted tactic file per label, L471-472), then CUPTI `interval_union` (L476).
- **FORK**: `model.py:180-248` (`GateLinear` out_dtype FP32 via `deepseek_v2.py:135-145`, FP32 `e_score_correction_bias`,
  `FusedMoEFactory(use_grouped_topk=True, num_expert_group=1, topk_group=1, scoring_func=sigmoid, routed_scaling_factor=2.5,
  renormalize=True, swiglu_limit=10.0, router_logits_dtype=fp32)`) → `ModelOptNvFp4FusedMoE` (`quantization/modelopt.py:843`
  `select_nvfp4_moe_backend`) → oracle `oracle/nvfp4.py:182-309`: FLASHINFER_TRTLLM first (L195), kept by the clamp filter
  (L205-219), `TrtLlmNvFp4ExpertsMonolithic` preferred (L80-90; TP1 no all2all/EPLB). Routing type
  `get_routing_method_type` (`fused_moe/config.py:150-155`) → DeepSeekV3 (n_group 1 > 0). Input quant: separate
  `ops.scaled_fp4_quant(…, is_sf_swizzled_layout=False)` in `prepare_finalize/no_dp_ep.py:14-37` / `fused_moe/utils.py:214-219`
  (`is_scale_swizzled=False` for TRTLLM, `oracle/nvfp4.py:580-586`). Call: `experts/trtllm_nvfp4_moe.py:548-581`
  `trtllm_fp4_block_scale_moe(routing_logits fp32, routing_bias=e_score bias fp32, hidden fp4 + linear e4m3 SF, w13/w2 shuffled by
  `prepare_static_weights_for_trtllm_fp4_moe` (`flashinfer_fp4_moe.py:396`, after w1/w3 reorder and 64-alignment pad),
  gemm1_alpha/beta None, **gemm1_clamp_limit = per-expert fp32 tensor 10.0 / g1_alphas** (L89-100, L174-180),
  output1_scale_scalar=g1_scale_c, output1_scale_gate_scalar=g1_alphas, output2=g2_alphas, n_group 1, topk_group 1, scale 2.5,
  routing_method_type DeepSeekV3, do_finalize=True (defer needs tp>1, `config.py:1422-1438`), activation SILU→3,
  per_token_scale None, enable_pdl default (None → `device_support_pdl`, FlashInfer core.py:3445-3448 → True on SM100),
  tune_max_num_tokens = `fi_moe_largest_bucket` = max(max_num_batched_tokens·dp, 8192) (`fused_moe/utils.py:456-471`))`.
  Autotune: vLLM kernel warmup `flashinfer_autotune` (`warmup/kernel_warmup.py:249-257`, enabled at -O1..-O3), in-process.
- **VERDICT**: SAME-FN-DIFF-ARGS — same `trtllm_fp4_block_scale_moe` (DeepSeekV3 routing, same dtypes/layouts, separate linear-SF
  input quant outside the call in both). Differences: (1) **gemm1_clamp_limit None vs per-expert tensor** — FlashInfer passes it
  as a runtime pointer into the GEMM1 epilogue (`data/csrc/trtllm_fused_moe_runner.cu:702-708`,
  `trtllm_batched_gemm_runner.cu:360`), tactic selection is not keyed on it; the extra clamp is epilogue math — cubin-level effect on
  time unverified, expected small. (2) tune bound: for T ≤ max_num_batched_tokens both map T to the same bucket; the tactic is
  tuned separately in each process (runner: persisted file from its own forced-routing tuning; fork: vLLM warmup inputs), so the
  chosen tactic can differ. (3) bias/logit values and output preallocation — no launch change.
- **VERSIONS**: rows FlashInfer 0.6.18 (TRTLLM_GEN_BMM cubins `8ec29a98…/batched_gemm-fa419f4-31ee4e5`, uv-cache copy
  `artifacts.py:141-143`); ours 0.7.0 (`1d145b82…/batched_gemm-09795a1-31ee4e5`, `FI070/artifacts.py:141-143`) — batched-GEMM
  cubins changed. Weight-prep helper refactored 3f667d7→892da0822f (vectorized; same permute/interleave by reading).
- **COVERED-BY-DB**: no — FlashInfer 0.6.18 → 0.7.0 changes the TRTLLM-gen batched-GEMM cubin set; plus the clamp-limit arg.
- **NOTES**: `routing_replay_out` passed by the fork is None unless routing-replay capture is enabled (not checked further).
