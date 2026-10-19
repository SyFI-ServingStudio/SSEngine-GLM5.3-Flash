# Call-path identity: GEMM / quant / norm configurations

Read-only code research (no GPU runs). Rows are from frozen profile.db; fork = pp-impl/third_party/vllm @ 892da0822f;
paths under `vllm/` are relative to the fork checkout. Our env: torch 2.13.0+cu130, cuBLAS 13.1.1.3, triton 3.7.1,
flashinfer-python 0.7.0, nvidia-cutlass-dsl 4.7.1.

| config | kind / backend | position | verdict | covered by DB |
|---|---|---|---|---|
| 2 | single_gemm / torch_linear_vllm | KDA in_proj | SAME | yes |
| 3 | single_gemm / torch_linear_vllm | KDA f_b/g_b_proj | SAME-FN-DIFF-ARGS (A is strided view, lda 24896 vs 128) | no |
| 6 | single_gemm / torch_linear_vllm | KDA o_proj | SAME | yes |
| 8 | nvfp4_quant / vllm_cuda | dense FFN gate_up input quant | SAME | yes |
| 9 | single_gemm / flashinfer_cutedsl | dense FFN gate_up | SAME | no (FlashInfer 0.6.18->0.7.0, cutlass-dsl 4.6.2->4.7.1) |
| 10 | nvfp4_quant / vllm_cuda | dense FFN down input quant | SAME | yes |
| 11 | single_gemm / flashinfer_cutedsl | dense FFN down | SAME | no (same reason as 9) |
| 12 | single_gemm / torch_linear_vllm | DSA fused_qkv_a | SAME | no (rows torch 2.11/cuBLAS 13.1.0.3) |
| 13 | q_kv_rms_norm / vllm_triton | DSA q/kv A-norm | SAME | yes |
| 14 | single_gemm / torch_linear_vllm | DSA q_b_proj | SAME | yes |
| 15 | single_gemm / torch_linear_vllm | indexer wq_b | SAME | no (cuBLAS 13.1.0.3) |
| 16 | single_gemm / torch_linear_vllm | indexer wk_weights | SAME | no (cuBLAS 13.1.0.3) |
| 17 | gemm_fp32_output / torch_cublas | indexer head weights (fp32) | SAME | yes |
| 18 | single_gemm / torch_linear_vllm | indexer kpool gate | SAME | no (cuBLAS 13.1.0.3) |
| 22 | batched_gemm / torch_mla_q_absorb_no_rope | MLA q absorb | SAME-FN-DIFF-ARGS (token-major out, ldc 32768) | no |
| 26 | batched_gemm / torch_mla_v_up_unpadded | MLA V up | SAME | yes |
| 27 | single_gemm / torch_linear_vllm | DSA o_proj | SAME | yes |
| 28 | gemm_fp32_output / torch_cublas | router gate (+gate_recompute) | SAME for m>16; DIFFERENT for m<=16 (CuTe-DSL ll_bf16_gemm); gate_recompute absent in fork | no |
| 29 | nvfp4_quant / vllm_cuda | routed-expert input quant (linear scales) | SAME | yes |
| 31 | single_gemm / torch_linear_vllm | shared expert gate_up | SAME | no (cuBLAS 13.1.0.3) |
| 32 | single_gemm / torch_linear_vllm | shared expert down | SAME | no (cuBLAS 13.1.0.3) |
| 34 | rms_norm / vllm_cuda | final norm | SAME | no (layernorm_kernels.cu changed 0b8edfb->892da08) |
| 35 | single_gemm / torch_linear_vllm | lm_head | SAME | yes |

Token-range notes:
- 28: m <= 16 takes the CuTe-DSL `ll_bf16_gemm` path. The fork makes one router call per MoE layer, not two.
- 22 and 26: at 892da0822f, a prefill microbatch whose max context+query is <= 2048 runs dense MHA (FLASH_ATTN prefill).
  - That path issues no q_absorb and no v_up; only the decode tokens of a mixed batch run them.
  - The dense-MHA path adds a kv_b_proj decompression, which has no row.
- 34: rows start at m = 32.
- No other configuration switches kernel at small m.

# Identity: NVFP4 dense FFN + routed-expert input quant (configs 8, 9, 10, 11, 29)

Paths: RUNNER = `<workspace>/wt-pp-main` at `git show e3c5fcf4ef:<path>`; FORK = `<workspace>/pp-impl/third_party/vllm` at 892da0822f (prefix `vllm/` omitted below).
Rows' env (e3c5fcf4ef `profiling/exec/env.py:129-131`): `vllm_env` = `vibesim-profiler-vllm:cu130-3f667d7e` (fork 3f667d7, FlashInfer 0.6.18, cutlass-dsl 4.6.2, torch 2.13.0).
Checkpoint: `quant_method=modelopt`, `quant_algo=NVFP4`, group 16. Layers 0–2 are `mlp_layer_types=dense` and their `mlp.*` is not in `ignore`, so dense FFN is NVFP4. `mlp.gate` and `mlp.shared_experts*` are excluded (BF16). `swiglu_limit=10.0`, `n_group=1`, `topk_group=1`, `scoring_func=sigmoid`, `noaux_tc`, `norm_topk_prob=True`, `moe_router_dtype=float32`.
The fork commits are not ancestors of each other: 892da0822f is the line rebased onto upstream 04730e8.

## Common facts for the BF16 F.linear configs (2, 3, 6, 12, 14, 15, 16, 18, 27, 31, 32, 35)

- **RUNNER:** `profiling/runners/gemm/torch.py` `profile_single_gemm_linear`.
  - Line numbers at both f8bd625a4d and 82c85c3203: :78-111. HEAD: :75-105.
  - Allocates `activations = randn(m,k)` and `weight = randn(n,k)`, BF16, both contiguous (:104-105).
  - The timed closure is exactly `F.linear(activations, weight)` (:108), with no bias and no pre/post ops.
  - Timed with `Timer.cupti(kernel, interval_union=True)` (:111), which counts a split-K producer/reduce pair once, by GPU-busy union.
  - HEAD diff: only the `torch.cuda.is_available()` guard was removed. The call is unchanged.
- **Row environments** (backend_version is null because `_runtime_versions` looks up package "torch_linear_vllm"; `profiling/exec/local_worker.py:185-198`):
  - f8bd625a4d rows ran in `vllm_env` = `vibesim-profiler-vllm:cu130` (`git show f8bd625a4d:profiling/exec/env.py:153-155`): torch 2.11.0, cuBLAS 13.1.0.3.
  - 82c85c3203 rows ran in `vllm_env` = `cu130-3f667d7e` (`82c85c3203:profiling/exec/env.py:131`): torch 2.13.0, cuBLAS 13.1.1.3.
  - Ours: torch 2.13.0+cu130, cuBLAS 13.1.1.3.
  - The pre-rename DB (`git show 592bfd6d^:profiling/profile.db`) confirms that every f8bd625a4d row of these shapes was already stored under `torch_linear_vllm` (not a renamed `*_vllm_fork` backend).
- **FORK dispatch** (892da0822f), the same for all of these:
  - `LinearBase` subclass forward → `quant_method.apply`:
    - ReplicatedLinear: `vllm/model_executor/layers/linear.py:411-417`
    - Column/Merged: `linear.py:603-610`
    - Row: `linear.py:1762-1778`
  - → `UnquantizedLinearMethod.apply` (`linear.py:229-239`) → `self._gemm_impl` = `dispatch_unquantized_gemm("auto")` (`model_executor/layers/utils.py:593-605`).
  - `"auto"` is not a key of `_FLASHINFER_BF16_BACKENDS` (`utils.py:151-157`, only `flashinfer_cutedsl`), so the call resolves to `default_unquantized_gemm` → `torch.nn.functional.linear(x, weight, bias)` (`utils.py:83-88`).
  - `kernel_config.linear_backend` defaults to "auto" (`config/kernel.py:293`).
  - `VLLM_BATCH_INVARIANT` defaults to 0 (`envs.py:627`).
  - On CUDA, `process_weights_after_loading` leaves the weight as the `[n,k]` contiguous parameter (`linear.py:206-227`).
- **Why these layers are BF16 in the NVFP4 checkpoint:**
  - The checkpoint `ignore` list contains `layers.N.self_attn*`, `layers.N.mlp.shared_experts*`, `layers.N.mlp.gate` and `lm_head`.
  - `ModelOptQuantConfigBase.is_layer_excluded` → `UnquantizedLinearMethod` (`model_executor/layers/quantization/modelopt.py:178-238`; the `*` suffix is expanded at :254-272).
  - KDA projections are built with `quant_config` forced to None (`vllm/models/glm5next/common/kda.py:189-194`; the base class stores it at `layers/mamba/gdn/base.py:41`).
  - These are code-read conclusions. No load was run to confirm them.
- **Small-m kernel switches:** no BF16 F.linear site in this model takes a different kernel at small m.
  - The only candidate, `DeepSeekV2FusedQkvAProjLinear._use_min_latency_gemm`, needs weight 2112×7168 (`model_executor/models/deepseek_v2.py:970-981`). Ours is 2048×4096, so it is off.
- **Library behind the call:** ATen `at::linear` → `mm`/`addmm` → cuBLAS(Lt) from the `nvidia-cublas` wheel. No fork source is involved.

## Config 2 — single_gemm / torch_linear_vllm

- **Positions:** first_kda_dense / kda_dense / kda_moe `.kda.in_proj` (m 1..65536, n 24896, k 4096)
- **RUNNER:** see common; profiler 82c85c3203 (image cu130-3f667d7e).
- **FORK:**
  - `kda.py:217-232` `_Glm5NextMergedColumnParallelLinear` (q,k,v 3×8192 + b 64 + f_a 128 + g_a 128 = 24896), quant_config None.
  - Called at `kda.py:407` `self.in_proj_qkvbfg_a(hidden_states)` → `F.linear(x[m,4096] contiguous, W[24896,4096] contiguous)`.
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.13.0 / cuBLAS 13.1.1.3; ours torch 2.13.0 / cuBLAS 13.1.1.3. No fork source is involved.
- **COVERED-BY-DB:** yes
- **NOTES:** none.

## Config 3 — single_gemm / torch_linear_vllm

- **Positions:** {first_kda_dense, kda_dense, kda_moe}.kda.{f_b_proj, g_b_proj} (n 8192, k 128)
- **RUNNER:** see common; 82c85c3203 (cu130-3f667d7e). Contiguous A `[m,128]`, so lda = 128.
- **FORK:**
  - `kda.py:234-240` (f_b_proj) and `:284-290` (g_b_proj), both ColumnParallelLinear, quant_config None.
  - Called at `kda.py:424` `self.f_b_proj(f_a)` and `:427` `self.g_b_proj(g_a)`.
  - `f_a` and `g_a` are column-split **views** of the in_proj output (`kda.py:408-416`): shape `[m,128]`, strides (24896, 1).
  - `F.linear` → `mm` hands the strided A to cuBLAS without a copy, with lda = 24896.
- **VERDICT:** SAME-FN-DIFF-ARGS. Same F.linear and the same weight `[8192,128]`, but A's row stride is 24896 instead of 128.
  - lda is a cublasLt heuristic input: it changes A alignment and load pattern, and with only 128-wide rows the reads are strided.
  - It can change both the chosen kernel and its time. By how much is unverified (no GPU run).
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both.
- **COVERED-BY-DB:** no. The library matches, but the A operand layout differs (lda 24896 vs 128).
- **NOTES:** Both positions (f_b, g_b) share this layout.

## Config 6 — single_gemm / torch_linear_vllm

- **Positions:** {first_kda_dense, kda_dense, kda_moe}.kda.o_proj (n 4096, k 8192)
- **RUNNER:** see common; 82c85c3203 (cu130-3f667d7e).
- **FORK:**
  - `kda.py:292-298` RowParallelLinear, quant_config None.
  - Called at `kda.py:446` `self.o_proj(core_attn_out)`.
  - Input: `o_norm` output (FusedRMSNormGated on a fresh `torch.empty`) reshaped to `[m,8192]`, contiguous (`kda.py:429-445`).
  - TP1, so `input_is_parallel` passes the input through without a split.
- **VERDICT:** SAME
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both.
- **COVERED-BY-DB:** yes
- **NOTES:** none.

## Config 8 — nvfp4_quant / vllm_cuda (swizzled_e4m3, hidden 4096)
- **positions**: first_kda_dense.dense_ffn.gate_up_input_quant, kda_dense.dense_ffn.gate_up_input_quant
- **RUNNER** (e3c5fcf4ef): registration `profiling/kernels/nvfp4_quant.py:75-90` (`subprocess_env="vllm_env"`). Runner `profiling/runners/elementwise/nvfp4_quant.py:68-118`, swizzled branch at :97-112.
  - Call: `vllm._custom_ops.scaled_fp4_quant(x_bf16[m,4096], gs, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl")`. `gs` is a 0-dim fp32 tensor, `448*6/amax(x)` (:162-169).
  - Timing: `Timer.cupti(run_once, kernel_name="15cvt_fp16_to_fp4I")` (:194-196) counts only the quant kernel. The output allocation happens inside the closure but launches no kernel.
  - The runner is unchanged at HEAD (`git diff e3c5fcf4ef HEAD` is empty for this file).
- **FORK**: `models/glm5next/common/model.py:116-122` `gate_up_proj = MergedColumnParallelLinear(quant_config=modelopt)` → `layers/quantization/modelopt.py:2525` `select_linear_kernel` → `:2320` `init_nvfp4_linear_kernel(use_a16=False)` (`kernels/linear/__init__.py:1070`).
  - Defaults: `linear_backend="auto"` (`config/kernel.py:293`); no `VLLM_BATCH_INVARIANT`. The first entry of `_POSSIBLE_NVFP4_KERNELS[CUDA]` (`__init__.py:551-553`) is `FlashInferCuteDslNvFp4LinearKernel`, and its `is_supported` is `is_device_capability_family(100)` + flashinfer (`kernels/linear/nvfp4/flashinfer.py:116-123`).
  - `apply_weights` at `nvfp4/flashinfer.py:147-152` calls `scaled_fp4_quant(x, layer.input_global_scale_inv, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl")`.
  - That reaches `_custom_ops.py:1513-1584`: `use_8x4_sf_layout` is False (no "trtllm" in the backend, :1562) → `torch.ops._C.scaled_fp4_quant.out` (:1579) → `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu:209-282`, kernel `cvt_fp16_to_fp4<bf16,false,false>` (128x4 swizzle).
  - The input comes from the mHC `hc_fused_post_pre` op (`model.py:532-542`), which outputs bf16. There is no norm+quant fusion.
- **VERDICT**: SAME. Only the global-scale value differs: it is a checkpoint static scale in the fork and a calibrated amax in the runner. The kernel reads `SFScale[0]` only (`nvfp4_quant_kernels.cu:67`), so the kernel and grid are unchanged. The fork uses the same kernel for every m; there is no small-m path.
- **VERSIONS**: rows fork 3f667d7 vs ours 892da0822f.
  - `scaled_fp4_quant`, `create_fp4_output_tensors` and `create_fp4_scale_tensor` are identical apart from docstrings. 3f667d7 also had a `register_fake` for `.out`, which has no runtime effect.
  - `git diff 3f667d7eb 892da0822f` is empty for `csrc/libtorch_stable/quantization/fp4/` and for every header it includes (`cuda_vec_utils.cuh`, `cuda_utils.h`, `cutlass_extensions/common.hpp`, `dispatch_utils.h`, `launch_bounds_utils.h`, `torch_utils.h`).
  - Not verified: the nvcc version used for the image build vs our `_C_stable_libtorch.abi3.so` build.
- **COVERED-BY-DB**: yes. The callable is the same, and the kernel source is unchanged across 3f667d7..892da0822f.
- **NOTES**: The runner's own docstring cites the same `FlashInferCuteDslNvFp4LinearKernel` call.

## Config 9 — single_gemm / flashinfer_cutedsl (NVFP4 gate_up: n=24576, k=4096)
- **positions**: first_kda_dense.dense_ffn.gate_up, kda_dense.dense_ffn.gate_up
- **RUNNER** (e3c5fcf4ef): registration `profiling/kernels/single_gemm.py:246-275` (`flashinfer_fp4.profile_single_gemm`, `vllm_env`). Runner `profiling/runners/gemm/flashinfer_fp4.py`.
  - Operands (:66-109): the weight is `scaled_fp4_quant(w, is_sf_swizzled_layout=False)` → `swizzle_blockscale` → `pad_nvfp4_weight_for_cutlass` (:86-89). The activation is `scaled_fp4_quant(..., swizzled, backend="flashinfer-cutedsl")` (:92); all of this is outside the timed region.
  - Timed closure (:112-134): `pad_nvfp4_activation_for_cutlass` (a no-op at k=4096) → `vllm.utils.flashinfer.flashinfer_scaled_fp4_mm(x_fp4, w, x_sf, w_sf, alpha, bf16, backend="cute-dsl")` → `slice_nvfp4_output` (a no-op).
  - One warm call compiles the JIT kernel (:167). Then `Timer.cupti(apply)` sums the launches (:172). There is no autotune context, so the heuristic tactic is used.
  - HEAD differs only by dropping reference-only operands and adding an OOM except clause; the timed call is the same.
- **FORK**: `FlashInferCuteDslNvFp4LinearKernel.apply_weights` (`nvfp4/flashinfer.py:137-172`).
  - The pad uses `nvfp4_weight_padding_bytes(layer)`, which is 0 here. Then `flashinfer_scaled_fp4_mm(x_fp4, layer.weight, x_blockscale, layer.weight_scale, layer.alpha, x.dtype=bf16, backend="cute-dsl")` (:158-166).
  - Call chain: `utils/flashinfer.py:1023-1059` sets `use_8x4_sf_layout=False` (it is only True for trtllm) → custom op `vllm::flashinfer_mm_fp4` (:781-806) → `flashinfer.mm_fp4(A, B.t(), A_sf, B_sf.t(), alpha, bf16, block_size=16, use_nvfp4=True, backend="cute-dsl")`.
  - Weight prep (:129-135) is the same as the runner's.
  - Autotune: `kernel_warmup.py:292-308` adds `fp4_gemm` to `skip_ops` whenever this kernel is selected, so serving uses FlashInfer's untuned path. That path is `_select_sm100_mm_fp4_cute_dsl_tactic` (`flashinfer/gemm/gemm_base.py:7137+` runner `forward`, tactic -1).
  - An explicit backend="cute-dsl" never selects `cutedsl_low_latency`, which is only an `auto` candidate. There is no small-m kernel switch.
- **VERDICT**: SAME (same function, args and layout).
- **VERSIONS**:
  - vLLM wrappers 3f667d7→892da0822f: `flashinfer_scaled_fp4_mm` and `flashinfer_mm_fp4` are identical. The `nvfp4/flashinfer.py` diff only replaces `weights_padding_cols` with `nvfp4_weight_padding_bytes`, which is 0 at these shapes. CuteDsl is still first in `_POSSIBLE_NVFP4_KERNELS`.
  - FlashInfer 0.6.18 vs 0.7.0, comparing the uv-cache wheel `<workspace>/uv-cache/archive-v0/QBxZX3G_jkAYIKXEueumD/flashinfer` (assumed to match the image's 0.6.18; the image was not inspected) with `pp-impl/.venv`:
    - `gemm/gemm_mm_fp4_cute_dsl.py` and `dense_blockscaled_gemm_sm100_common.py` are identical.
    - `dense_blockscaled_gemm_sm100.py` only adds two unused `NamedBarrier` attributes.
    - `gemm/kernels/utils.py` extends `_M_BUCKETS` from ≤4096 to ≤32768 and drops the swap penalty for m>8192.
    - I ran the untuned selector on CPU (no GPU) under both versions for all 68 m at (24576, 4096) with sm_count 148. It returns the same tactic for every m. m≥1024 uses (256,256)/cluster (2,1)/no swap; m≤16 uses swap_ab with tile_n 8/16.
    - Not verified: the CuTe-DSL compiler differs (cutlass-dsl 4.6.2 → 4.7.1), and generated SASS was not compared.
- **COVERED-BY-DB**: no, strictly. The callable, kernel source and tactic are identical, but the FlashInfer and cutlass-dsl versions changed, and I could not prove that the DSL compiler change leaves the generated kernel unchanged. It is likely equivalent; a spot re-measure would settle it.
- **NOTES**: The rows time the GEMM only; the activation quant is config 8.

## Config 10 — nvfp4_quant / vllm_cuda (swizzled_e4m3, hidden 12288)
- **positions**: first_kda_dense.dense_ffn.down_input_quant, kda_dense.dense_ffn.down_input_quant
- **RUNNER**: same as config 8, with hidden_size=12288.
- **FORK**: `model.py:124-131` `down_proj = RowParallelLinear(quant_config=modelopt)`, TP1 so K=12288. The same `FlashInferCuteDslNvFp4LinearKernel.apply_weights` quant call as config 8 (`nvfp4/flashinfer.py:147-152`).
  - The producer is `SiluAndMulWithClamp(swiglu_limit=10.0)` (`model.py:139-148`, `layers/activation.py:208-249`, op `_C.silu_and_mul_with_clamp`).
  - The NVFP4 act+quant fusion pass (`compilation/passes/fusion/act_quant_fusion.py:125-175,291-292`) only matches `MatcherSiluAndMul` (plain `silu_and_mul`, `passes/fusion/matcher_utils.py:527`). It has no clamp variant, so the quant stays a standalone `scaled_fp4_quant`, even though `enable_act_fusion` is on for NVFP4 models (`config/vllm.py:165-174`).
- **VERDICT**: SAME, with the same global-scale note as config 8 and no small-m switch.
- **VERSIONS**: as config 8. The kernel source is unchanged.
- **COVERED-BY-DB**: yes.
- **NOTES**: If a later fork adds a clamp-aware SiLU+NVFP4 fusion, this position would merge into the activation kernel.

## Config 11 — single_gemm / flashinfer_cutedsl (NVFP4 down: n=4096, k=12288)
- **positions**: first_kda_dense.dense_ffn.down, kda_dense.dense_ffn.down
- **RUNNER**: same as config 9, with n=4096 and k=12288 (no pad or slice).
- **FORK**: `down_proj` (`model.py:124-131`) → the same `FlashInferCuteDslNvFp4LinearKernel` → `mm_fp4(backend="cute-dsl")` chain as config 9.
- **VERDICT**: SAME.
- **VERSIONS**: as config 9. The untuned tactic is identical in 0.6.18 and 0.7.0 for all 68 m at (4096, 12288): m≥1024 uses (256,256)/(2,1), m=128 uses (128,64), and m≤16 uses swap_ab. The cutlass-dsl 4.6.2 → 4.7.1 change is not verified.
- **COVERED-BY-DB**: no, strictly, for the same reason as config 9 (FlashInfer and CuTe-DSL compiler versions changed; source and tactic are identical).
- **NOTES**: none.

## Config 12 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.fused_qkv_a (n 2048 = q_lora 1536 + kv_lora 512 + rope 0, k 4096)
- **RUNNER:** see common; profiler **f8bd625a4d** (image cu130).
- **FORK:**
  - `glm5next/common/attention.py:447-452` `DeepSeekV2FusedQkvAProjLinear`.
  - Its `forward` (`deepseek_v2.py:984-995`) falls back to `MergedColumnParallelLinear.forward` because `_use_min_latency_gemm` is False: the weight is 2048×4096, not 2112×7168.
  - Called at `layers/mla.py:183` `self.fused_qkv_a_proj(hidden_states)` → `F.linear(x[m,4096] contiguous, W[2048,4096])`.
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3. cuBLAS changed, and that cannot be shown to leave the heuristic and kernel set untouched.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** none.

## Config 13 — q_kv_rms_norm / vllm_triton

- **Positions:** dsa_moe.dsa.q_kv_a_norm (q 1536, kv 512, eps 1e-5, bf16)
- **RUNNER:**
  - At f8bd625a4d this was kind `deepseek_v4_fused_q_kv_rmsnorm`, backend **`vllm_fork_triton`**, `subprocess_env="vllm_fork_env"` (`f8bd625a4d:profiling/kernels/deepseek_v4_fused_q_kv_rmsnorm.py:48-58`). It was renamed in DB commit 592bfd6d; the pre-rename DB shows these 68 rows under `vllm_fork_triton` with backend_version `0.1.dev20913+g3f667d7eb.precompiled`.
  - Callable: `f8bd625a4d:profiling/runners/attention/deepseek_v4_fused_q_kv_rmsnorm_vllm_triton.py` `profile_..._vllm_fork_triton`. It calls `vllm.models.common.ops.fused_qk_rmsnorm.fused_q_kv_rmsnorm(qr, kv, q_w, kv_w, eps)`.
  - `qr` and `kv` are split **views** of a `[m,2048]` buffer, so both have row stride 2048.
  - Timed with `Timer.cupti(launch, warmup=3, kernel_name=None)`.
- **FORK:**
  - `layers/mla.py:204-214`, `fused_q_kv_rmsnorm(q_c, kv_c, q_a_layernorm.weight, kv_a_layernorm.weight, eps)`, with `fuse_qkv_rmsnorm=True` (`glm5next/common/attention.py:583`).
  - `q_c` and `kv_c` are split views of `qkv_lora[m,2048]` (`mla.py:183-186, 202`), so the stride is 2048 and the layout is the same as the runner's.
  - → `FusedQKVRMSNormKernel.__call__` (`vllm/models/common/ops/fused_qk_rmsnorm.py:166-191`) → `kernel[grid](...)` (`model_executor/warmup/jit_warmup_triton_helper.py:257-300`).
  - Launch: grid (m, 2), BLOCK 2048, num_warps 8, launch_pdl = True on SM100.
- **VERDICT:** SAME
- **VERSIONS:**
  - Rows: fork 3f667d7 host venv. The checkout `old-wt/wt-glm5-3-flash-support/alignment/profiler/vllm` is at 3f667d7eb4, has `_version.py` 0.1.dev20913+g3f667d7eb, and its venv was installed 2026-09-24 with torch 2.13.0 / triton 3.7.1 / cuBLAS 13.1.1.3. That checkout is the inferred env; the profile host path is not recorded in the row.
  - Ours: triton 3.7.1.
  - Kernel source, `git diff 3f667d7eb 892da0822f -- vllm/models/common/ops/fused_qk_rmsnorm.py`: the `@triton.jit` body is byte-identical (only moved into the `VllmTritonJitKernel` class). Grid, constexprs, num_warps and PDL are unchanged. Strides stay runtime ints; the compile key only drives warmup.
- **COVERED-BY-DB:** yes. The Triton version is equal and the kernel body and launch are unchanged.
- **NOTES:** none.

## Config 14 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.q_b_proj (n 16384 = 64×256, k 1536)
- **RUNNER:** see common; 82c85c3203 (cu130-3f667d7e).
- **FORK:**
  - `attention.py:464-470` ColumnParallelLinear.
  - Called at `mla.py:221` `q_proj_layer(q_proj_input)`.
  - Input: the `qr_out` of `fused_q_kv_rmsnorm`, a fresh contiguous `torch.empty(qr.shape)` (`fused_qk_rmsnorm.py:208`).
- **VERDICT:** SAME
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both.
- **COVERED-BY-DB:** yes
- **NOTES:** none.

## Config 15 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.indexer.wq_b (n 4096 = 32×128, k 1536)
- **RUNNER:** see common; **f8bd625a4d** (image cu130).
- **FORK:**
  - `attention.py:250-256` ReplicatedLinear (prefix `...self_attn.indexer.wq_b`, excluded, so BF16).
  - Called at `attention.py:319` `self.wq_b(qr)`, where `qr = q_c` = the same contiguous normed q (`mla.py:214, 233`).
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** none.

## Config 16 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.indexer.wk_weights (n 160 = wk 128 + weights_proj 32, k 4096)
- **RUNNER:** see common; **f8bd625a4d** (image cu130).
- **FORK:**
  - `attention.py:259-266` MergedColumnParallelLinear `[128, 32]`, `quant_config=None`, `disable_tp=True`.
  - Called at `attention.py:324` `self.wk_weights_proj(hidden_states)` → `F.linear(x[m,4096] contiguous, W[160,4096])`.
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** The fork uses only the k part (`kw[:, :128]`). The head weights are recomputed in FP32 (config 17), so the 32 weights_proj output columns are computed and then discarded.

## Config 17 — gemm_fp32_output / torch_cublas

- **Positions:** dsa_moe.dsa.indexer.head_weights (input fp32, n 32, k 4096)
- **RUNNER:**
  - These rows were produced by backend **`torch_cublas_vllm_fork`** (`subprocess_env="vllm_fork_env"`, `f8bd625a4d:profiling/kernels/gemm_fp32_output.py:51-69`) and renamed to `torch_cublas` in 592bfd6d. The pre-rename DB shows these 68 rows under `torch_cublas_vllm_fork`, 2026-09-25T00:27.
  - At f8bd625a4d, `torch_cublas` itself rejected fp32 input (`_SUPPORTED_DTYPES={BF16}`).
  - Callable: `f8bd625a4d:profiling/runners/gemm/gemm_fp32_output_torch_cublas.py` `profile_gemm_fp32_output_torch_cublas_vllm_fork` → `_prepare` → `_Launch.run`.
    - Input: `torch.mm(input_fp32[m,4096] contiguous, weight[32,4096].t().contiguous()` → `[4096,32]` contiguous).
    - Requires matmul precision "highest".
    - The `x.float()` cast is outside the boundary.
  - Timer: `Timer.cupti(launch.run, warmup=5, kernel_name=None)`, which sums launches (an SGEMM plus splitKreduce pair is expected).
- **FORK:**
  - `attention.py:326-333`: `self._wp_fp32 = wk_weights_proj.weight[128:, :].t().contiguous().float()`, giving `[4096,32]` contiguous FP32, cached.
  - Then `torch.mm(hidden_states.float(), self._wp_fp32)`, where `hidden_states.float()` is a separate cast kernel giving a fresh contiguous `[m,4096]` FP32.
  - Precision: `VLLM_FLOAT32_MATMUL_PRECISION` defaults to "highest" (`envs.py:92`, applied in `v1/worker/gpu_worker.py:202-203`).
- **VERDICT:** SAME
- **VERSIONS:** rows: the fork 3f667d7 host venv (inferred as for config 13), torch 2.13.0 / cuBLAS 13.1.1.3. Ours: torch 2.13.0 / cuBLAS 13.1.1.3. No fork source is involved.
- **COVERED-BY-DB:** yes. The env identity is inferred from the backend_version-less row plus the pre-rename backend name; the checkout path itself is not recorded.
- **NOTES:** none.

## Config 18 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.indexer.kpool_gate_score (n 128, k 4096)
- **RUNNER:** see common; **f8bd625a4d** (image cu130).
- **FORK:**
  - `attention.py:380` `F.linear(hidden_states, self.index_kpool_compress_gate)` is a direct F.linear, not a LinearBase.
  - The gate is an `nn.Parameter [128,4096]` BF16 (`attention.py:244-246`). Input: `hidden_states[m,4096]`, contiguous.
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** none.

## Config 22 — batched_gemm / torch_mla_q_absorb_no_rope

- **Positions:** dsa_moe.dsa.q_absorb (68 rows, m 1..65536, num_batches 64, k 256, n 512, bf16)
- **RUNNER** (profiler 82c85c3203, vllm_env → image cu130-3f667d7e): `profiling/runners/gemm/batched_gemm.py@82c85c3203`
  `profile_mla_q_absorb_no_rope` :505 → `_profile_layout_bmm` :454 with `_build_layout_q_absorb_operands` :382.
  Timed closure is one `torch.bmm(lhs, rhs, out=out)` (:481):
  - lhs = `q_base[m,64,256][..., :256].transpose(0,1)`, so shape (64,m,256) with strides (256, 64·256, 1).
  - rhs = W_UK_T = `packed[64·512, 512].T.view(512,64,512).split([256,256])[0].permute(1,2,0)`: (64,256,512), a strided view of a contiguous kv_b_proj-shaped weight.
  - **out = `torch.empty((64, m, 512))`, contiguous** (:395), so ldc = 512 and the batch stride is m·512.
  - The closure has no pre/post ops. An untimed correctness check runs first (:484–487). Timer.cupti uses the default (no interval_union).
  - HEAD diff: only the B200-only GPU gate and the {8,16,32,64} head-count gate were removed. Operands and the call are unchanged.
- **FORK** (892da0822f): glm5next `Glm5NextMLAAttention` → `MultiHeadLatentAttentionWrapper` (`vllm/models/glm5next/common/attention.py:569`) → `MLAAttention.forward_impl`.
  - The absorbed (MQA) branch is at `vllm/model_executor/layers/attention/mla_attention.py:1022-1097`. Backend: FP8 KV on SM100 gives FLASHINFER_MLA_SPARSE (`vllm/platforms/cuda.py:100-105`). `FlashInferMLASparseImpl` (`v1/attention/backends/mla/flashinfer_mla_sparse.py:412-470`) sets no `q_pad_num_heads`, so the unpadded branch `mla_attention.py:1091-1097` runs:
    `mqa_ql_nope = q_nope.new_empty((B, N, L)); torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0,1))`.
  - lhs = `q.view(-1,64,256).split([256,0])[0].transpose(0,1)`, the same strides as the runner.
  - W_UK_T comes from `split_kv_b_proj` (:409-424) + `replace_parameter(..., W_UK.permute(1,2,0))` (:1325). That is the same strided view of the BF16 kv_b_proj weight; `self_attn*` is in the checkpoint's modelopt ignore list, so UnquantizedLinearMethod returns the weight as-is (quant_utils.py:506-527).
  - **out is token-major: shape (64,B,512) with strides (512, 64·512, 1)**, so ldc = 32768 and the batch stride is 512.
  - This changed in fork commit 1768273c13 ("NoPE MQA query without concat"). At 3f667d7 the out was a contiguous `new_empty((N,B,L))` (3f667d7:mla_attention.py:1002-1005), which is what the runner mirrors.
- **VERDICT:** SAME-FN-DIFF-ARGS. Both run torch.bmm with the same lhs/rhs layout, but the output layout differs: ldc changes 512→32768 and the batch stride changes m·512→512. ATen passes this strided out straight to cuBLAS without a copy (stride[2]==1, stride[1] ≥ size[2]). The different ldc and strideC are cublasLt heuristic inputs and change the store pattern, so the selected kernel and its time can differ. By how much is unverified (no GPU run).
- **VERSIONS:** torch 2.13.0 and cuBLAS 13.1.1.3 in both the rows' image cu130-3f667d7e and ours (2.13.0+cu130, cuBLAS 13.1.1.3). The call goes to torch/cuBLAS, and no fork kernel source is involved. The fork-side call site changed (1768273c13, 3f667d7→892da0822f), not the library.
- **COVERED-BY-DB:** no. The library versions match, but the rows were measured with a contiguous (N,m,L) output, while the fork writes into a token-major (m,N,L) buffer through a transposed out. That is a different cuBLAS problem descriptor.
- **Token ranges where the fork differs:**
  - The bmm runs only for MQA tokens.
  - A prefill microbatch whose max prefill seq_len (context + query) is ≤ 2048 (`index_topk`) takes dense MHA instead (`_use_dense_mha_prefill`, `sparse_mla_attention.py:130-138`; `mla_attention.py:982-1020`; FLASH_ATTN prefill backend supports (256,0,256), `prefill/flash_attn.py`, commit 238cb2b191). That path issues no q_absorb at all; only the decode tokens of a mixed batch are absorbed.
  - When seq_len > 2048, all q.size(0) tokens are absorbed.
  - Masked MHA is off with an FP8 KV cache (`sparse_mla_attention.py:121`).
  - There is no small-m kernel switch (no head padding).
- **NOTES:** At 3f667d7 (the fork the rows mirror), sparse prefill always took the absorbed path; dense-MHA prefill for the NoPE layout arrived in 238cb2b191, which is after 3f667d7.

## Config 26 — batched_gemm / torch_mla_v_up_unpadded

- **Positions:** dsa_moe.dsa.v_up (68 rows, m 1..65536, num_batches 64, k 512, n 256, bf16)
- **RUNNER** (82c85c3203, cu130-3f667d7e): `batched_gemm.py@82c85c3203` `profile_mla_v_up_unpadded` :521 → `_profile_layout_bmm` :454 with `_build_layout_v_up_operands` :399 (`padded_heads=None`, so 64 heads).
  `torch.bmm(lhs, rhs, out=out)` (:481):
  - lhs = `attn[m,64,512].transpose(0,1)`, which is (64,m,512).
  - rhs = W_UV = `packed.T.view(512,64,512).split(...)[1].transpose(0,1)`, which is (64,512,256).
  - out = `out_base[m, 64·256].view(m,64,256).transpose(0,1)`.
  - The closure has no pre/post ops, and an untimed correctness check runs first.
  - HEAD diff: only the gates (head count ≤ 64 now, GPU gate removed). Operands are unchanged.
- **FORK** (892da0822f): `MLAAttention._v_up_proj` `mla_attention.py:1374-1406`, called at :1157 after `FlashInferMLASparseImpl.forward_mqa`.
  `torch.bmm(x.view(-1,64,512).transpose(0,1), self.W_UV, out=out.view(-1,64,256).transpose(0,1))`:
  - x is the trtllm-gen MLA output `o.view(-1,N,L)`, contiguous (B,64,512) (`flashinfer_mla_sparse.py:736`).
  - W_UV = `W_UV.transpose(0,1)` of the strided kv_b_proj view (:1323).
  - out is the (B, 64·256) attention-output slice.
- **VERDICT:** SAME (same torch.bmm, same lhs/rhs/out shapes and strides; unpadded, because FlashInferMLASparse sets no `q_pad_num_heads`; num_attention_heads = 64 at TP1). The "V-up pads to 64" issue does not apply: unpadded and padded are identical at 64 heads.
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both. `_v_up_proj` is unchanged between 3f667d7 (`3f667d7:mla_attention.py:1300-1332`) and 892da0822f.
- **COVERED-BY-DB:** yes, for the tokens that take the absorbed path.
- **Token ranges where the fork differs:** the same gating as config 22.
  - v_up runs only for MQA tokens.
  - A prefill microbatch with max prefill seq_len ≤ 2048 runs dense MHA (FA prefill writes V directly, from the kv_b_proj decompression). That path issues no v_up; only its decode tokens do.
  - With seq_len > 2048, all tokens run v_up.
- **NOTES:** Out of my scope but relevant to config 28: fork commit 1768273c13 is titled "no duplicate router GEMM" and touches `vllm/models/glm5next/nvidia/model.py`. Check whether `gate_recompute` still exists at 892da0822f.

## Config 27 — single_gemm / torch_linear_vllm

- **Positions:** dsa_moe.dsa.o_proj (n 4096, k 16384)
- **RUNNER:** see common; 82c85c3203 (cu130-3f667d7e).
- **FORK:**
  - `attention.py:487-493` RowParallelLinear.
  - Called at `mla.py:255` `self.o_proj(attn_out)`.
  - `attn_out` is the MLA output buffer `torch.empty(output_shape=(m, 64·256))` (`mla_attention.py:861/881`), contiguous.
- **VERDICT:** SAME
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both.
- **COVERED-BY-DB:** yes
- **NOTES:** none.

## Config 28 — gemm_fp32_output / torch_cublas

- **Positions:**
  - dsa_moe.moe.router.gate
  - dsa_moe.moe.router.gate_recompute
  - kda_moe.moe.router.gate
  - kda_moe.moe.router.gate_recompute
  - All: bf16 input, n 288, k 4096.
- **RUNNER:**
  - At f8bd625a4d these rows were stored under `torch_cublas` (pre-rename DB, 2026-09-25T00:35), so the env was `vllm_env` image cu130.
  - Callable: `f8bd625a4d:profiling/runners/gemm/gemm_fp32_output_torch_cublas.py` `profile_gemm_fp32_output_torch_cublas` → `_Launch.run`: `torch.mm(x_bf16[m,4096], W_bf16[288,4096].T, out_dtype=torch.float32)`.
  - Timer: `Timer.cupti(..., warmup=5, kernel_name=None)`, with no interval union.
- **FORK:**
  - `glm5next/common/model.py:181-185` builds `GateLinear(4096, 288, out_dtype=fp32)`; `moe_router_dtype="float32"` comes from `deepseek_v2.py:135-145`. The weight is BF16 and `quant_config` is None.
  - `Glm5NextMoE.forward` no longer calls the gate (`model.py:262-267` passes `router_logits=hidden_states` as a placeholder).
  - The single call is `MoERunner` `router_logits, _ = self.gate(hidden_states)` (`fused_moe/runner/moe_runner.py:903-910`); `_fse_fuse_gate` is False because there is no shared_expert_gate.
  - `GateLinear.forward` (`fused_moe/router/gate_linear.py:192-244`) dispatches by m:
    - **m ≤ 16** (`LL_BF16_MAX_TOKENS`, :43): Tier 1 `ll_bf16_gemm(x, W)`, the CuTe-DSL kernel (`kernels/linear/cute_dsl/ll_bf16.py:336`; `is_available()` is true because nvidia-cutlass-dsl 4.7.1 is installed).
    - **m > 16**: Tier 4 `torch.mm(x, self.weight.T, out_dtype=torch.float32)` (:229-232).
    - Tiers 2 and 3 need an FP32 weight, so they are off.
- **VERDICT:**
  - m > 16: SAME.
  - m ≤ 16: DIFFERENT (fork: vLLM CuTe-DSL `ll_bf16_gemm`; runner: cuBLAS `torch.mm`).
  - **Position `gate_recompute` does not exist in 892da0822f.** Fork commit 1768273c13 ("no duplicate router GEMM", 2026-09-10; an ancestor of 892da0822f, not of 3f667d7) removed the extra `self.gate(hidden_states)` call in `Glm5NextMoE.forward`. That call is at `3f667d7eb:vllm/models/glm5next/nvidia/model.py:264`, beside `moe_runner.py:902`. The fork therefore runs the router GEMM **once** per MoE layer, not twice.
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no.
  - The cuBLAS/torch version differs.
  - For m ≤ 16 the callable differs, and no row exists for `ll_bf16_gemm`.
  - The gate_recompute rows price a call our fork does not make.
- **NOTES:** The handoff's "router runs twice per MoE layer" (`2_kernels.md` §2) reflects the 3f667d7 capture. At 892da0822f, drop one router GEMM per MoE layer from the targets.

## Config 29 — nvfp4_quant / vllm_cuda (linear_e4m3, hidden 4096)
- **positions**: dsa_moe.moe.routed_rank0.input_quant, kda_moe.moe.routed_rank0.input_quant
- **RUNNER** (e3c5fcf4ef): `profiling/runners/elementwise/nvfp4_quant.py:86-96`. Call: `ops.scaled_fp4_quant(x_bf16[m,4096], torch.ones(()), is_sf_swizzled_layout=False)` with backend left at its default `"none"`. Timing: `Timer.cupti(kernel_name="cvt_fp16_to_fp4_sf_major")`. Unchanged at HEAD.
- **FORK**:
  - MoE backend selection: `model.py:225-248` `FusedMoEFactory(quant_config=modelopt)` → `ModelOptNvFp4FusedMoE.__init__` (`modelopt.py:843-847`) → `select_nvfp4_moe_backend(weight=kNvfp4Static, act=kNvfp4Dynamic)` (`layers/fused_moe/oracle/nvfp4.py:182-310`). The defaults are `moe_backend="auto"` (`config/kernel.py:241`). `AVAILABLE_BACKENDS` (:194) starts with FLASHINFER_TRTLLM, which is in the clamp-capable set, so the `swiglu_limit` filter keeps it. `backend_to_kernel_cls` (:80-91) tries `TrtLlmNvFp4ExpertsMonolithic` first.
  - Monolithic support checks: TP1 with no all2all and no EPLB (`experts/trtllm_nvfp4_moe.py:463-469`). The routing method is DeepSeekV3: sigmoid + bias + renormalize + n_group=1>0 (`fused_moe/config.py:150-155`); `moe_renormalize` defaults True (`transformers_utils/configs/glm5_next.py:35,91`). fp32 router logits are accepted (:489-493).
  - Result: TRTLLM monolithic, `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` (:548).
  - Quant layout: `make_nvfp4_moe_quant_config` sets `is_scale_swizzled=False` for FLASHINFER_TRTLLM (`oracle/nvfp4.py:580-585`). `per_token_activation=False` (`make_nvfp4_moe_kernel` default, :599/615), so `expects_unquantized_inputs` is False and quant is not deferred into the MoE kernel.
  - Quant call: `FusedMoEKernel.apply_monolithic` (`modular_kernel.py:1543-1548`) → `MoEPrepareAndFinalizeNoDPEPMonolithic.prepare` (`prepare_finalize/no_dp_ep.py:122-129`) → `_quantize_input` (:14-36) → `moe_kernel_quantize_input(..., "nvfp4", is_scale_swizzled=False)` (`fused_moe/utils.py:357-381`) → `_nvfp4_quantize` → `ops.scaled_fp4_quant(A, a1_gscale, is_sf_swizzled_layout=False)` (:214-219).
  - That reaches `cvt_fp16_to_fp4_sf_major` (`nvfp4_quant_kernels.cu:283-309`), a standalone launch not fused into the MoE kernel. The modular fallback (`no_dp_ep.py:58-78`) would make the same call.
- **VERDICT**: SAME. Only the global-scale arg differs: the fork passes a `[288]` tensor of `1/amax(a13_scale)` (`quantization/utils/flashinfer_fp4_moe.py:360-364`, `quant_utils.py:72-82`), the runner a 0-dim `1.0`. The kernel reads only `[0]`. Backend "none" means there is no 8x4 small-m path, so the call is identical for every m.
- **VERSIONS**: same op and csrc as config 8. The quant source is unchanged 3f667d7→892da0822f. The 3f667d7→892da diffs in `oracle/nvfp4.py`, `trtllm_nvfp4_moe.py` and `fused_moe/utils.py` do not touch the scale-layout choice or the quant call.
- **COVERED-BY-DB**: yes.
- **NOTES**: Routed experts themselves are `nvfp4_fused_moe` / `flashinfer_trtllm_sm100` (handoff `2_kernels.md:63`), outside this set. Engine-level token padding (CUDA-graph sizes) is not considered here.

## Config 31 — single_gemm / torch_linear_vllm

- **Positions:** {dsa_moe, kda_moe}.moe.shared_expert.gate_up (n 4096 = 2×2048, k 4096)
- **RUNNER:** see common; **f8bd625a4d** (image cu130).
- **FORK:**
  - `model.py:214-222` → `Glm5NextMLP.gate_up_proj` MergedColumnParallelLinear (`model.py:117-124`), excluded (`mlp.shared_experts*`), so BF16.
  - Called at `model.py:145` via `MoERunner`'s `SharedExperts`, possibly on the overlap stream (`moe_runner.py:893-898`). Input: `hidden_states[m,4096]`, contiguous.
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** On the aux stream the GEMM can overlap the routed experts. That overlap affects measured wall time, not kernel identity.

## Config 32 — single_gemm / torch_linear_vllm

- **Positions:** {dsa_moe, kda_moe}.moe.shared_expert.down (n 4096, k 2048)
- **RUNNER:** see common; **f8bd625a4d** (image cu130; pre-rename DB date 2026-09-24T23:55).
- **FORK:**
  - `Glm5NextMLP.down_proj` RowParallelLinear (`model.py:125-132`).
  - Called at `model.py:147`. Input: the `SiluAndMulWithClamp(swiglu_limit=10)` output `[m,2048]`, a fresh contiguous tensor (`model.py:139-142`).
- **VERDICT:** SAME
- **VERSIONS:** rows torch 2.11.0 / cuBLAS 13.1.0.3; ours torch 2.13.0 / cuBLAS 13.1.1.3.
- **COVERED-BY-DB:** no. The cuBLAS/torch version differs.
- **NOTES:** none.

## Config 34 — rms_norm / vllm_cuda

- **Positions:** final_norm (hidden 4096, bf16; m 32..65536, 63 rows)
- **RUNNER:**
  - `f8bd625a4d:profiling/runners/norm/rms_norm_vllm_cuda.py` `profile_rms_norm_vllm_cuda`.
  - Call: `vllm._custom_ops.rms_norm(out, x[m,4096] contiguous, w[4096], eps 1e-5)` → `torch.ops._C.rms_norm`.
  - Timer: `Timer.cupti(kernel, kernel_name="rms_norm_kernel")`.
  - Env: image cu130 (vLLM fork 0b8edfb, backend_version `0.23.0+vibesim.0b8edfb...`).
  - HEAD diff: 15 deleted lines (guards); the call is unchanged.
- **FORK:**
  - `glm5next/common/model.py:748` `self.norm(hidden_states)`. The input is the `hc_contract` = `x.mean(dim=1)` output, fresh and contiguous (`model.py:552-553`, `layers/mhc.py:974-976`).
  - `RMSNorm.forward_cuda` → `forward_native` → `ir.ops.rms_norm` (`layers/layernorm.py:74-115`).
  - IR priority:
    - On CUDA the default is `["vllm_c","native"]` unless inductor compilation is on (`platforms/cuda.py:734-756`).
    - Glm5Next is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`config/vllm.py:78-90`), which sets `compilation_config.mode = NONE` (`config/vllm.py:830-858`, before `set_platform_defaults` at :1766).
    - So the selected impl is `vllm_c` (`kernels/vllm_c.py:23-44`) → `torch.ops._C.rms_norm(output, x, weight, eps)`.
  - `VLLM_USE_OINK_OPS` defaults to False (`envs.py:1227`).
- **VERDICT:** SAME (same op `torch.ops._C.rms_norm`, same args).
- **VERSIONS:** rows: fork 0b8edfb `_C`; ours: fork 892da0822f `_C_stable_libtorch`.
  - Source: `csrc/libtorch_stable/layernorm_kernels.cu` **changed** (`git diff 0b8edfb56 892da0822f`).
  - The kernel gained a `HasWeight` template parameter and a `weight_stride` argument (weight_row), and the weight multiply moved to fp32 (`x*s_variance*w` instead of `(scalar_t)(x*s_variance)*w`).
  - The host launch is unchanged: grid = m, block = min(hidden/vec, m<256 ? 1024 : 256).
- **COVERED-BY-DB:** no. The kernel source changed. The change is small for a memory-bound kernel, but its time effect is unverified.
- **NOTES:** Rows start at m=32, so there are no rows for 1–31 tokens.

## Config 35 — single_gemm / torch_linear_vllm

- **Positions:** lm_head (n 154880, k 4096)
- **RUNNER:** see common; 82c85c3203 (cu130-3f667d7e).
- **FORK:**
  - `model.py:948-953` `ParallelLMHead(154880, 4096, quant_config=modelopt)`. `lm_head` is excluded, so it gets `UnquantizedLinearMethod` (`modelopt.py:221-225`). The vocab is a multiple of 64, so there is no padding.
  - `compute_logits` (`model.py:1014-1018`) → `LogitsProcessor._apply_head` (`logits_processor.py:136-146`).
  - `head_dtype` is the model dtype because the config has no `head_dtype` (`config/model.py:2447-2466`), so the path is `lm_head.quant_method.apply` → `F.linear(h[m,4096], W[154880,4096])`.
  - `h` holds the gathered sampling rows (`hidden_states[logits_indices]`, `v1/worker/gpu_model_runner.py:4511`, contiguous).
- **VERDICT:** SAME
- **VERSIONS:** torch 2.13.0 / cuBLAS 13.1.1.3 in both.
- **COVERED-BY-DB:** yes
- **NOTES:** In a prefill microbatch, m is the number of sampled positions (about one per request), not the number of prefill tokens.
