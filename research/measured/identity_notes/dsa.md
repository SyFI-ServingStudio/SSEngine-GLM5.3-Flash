# Call-path identity: DSA (kpool indexer + sparse MLA), configs 19, 20, 21, 23, 24, 25

Fork = `pp-impl/third_party/vllm` @ 892da0822f (paths relative to it). Sim runners: `wt-pp-main` @ the profiler commit given.
Model facts (config.json): index_n_heads 32, index_head_dim 128, index_topk 2048, index_kpool 4, qk_rope_head_dim 0,
qk_nope_head_dim 256, kv_lora_rank 512, num_attention_heads 64, mla_use_nope. FP8 KV cache → sparse backend
FLASHINFER_MLA_SPARSE = `FlashInferMLASparseTRTLLMBackend` on SM100 (`vllm/platforms/cuda.py:98-104`,
`vllm/v1/attention/backends/registry.py:78-81`).

Common fork prefill flow (one DSA layer):
`Glm5NextMLAAttention` → `MultiHeadLatentAttentionWrapper.forward` (`vllm/model_executor/layers/mla.py:163-268`) →
`Indexer.forward` (`vllm/models/glm5next/common/attention.py:316-401`; 32 heads, so no head padding, :387) →
`SparseAttnIndexerKpool.forward_cuda` → `sparse_attn_indexer_kpool` (`vllm/models/glm5next/nvidia/sparse_indexer.py:94-618`) →
then `MLAAttention.forward_impl` (`vllm/model_executor/layers/attention/mla_attention.py:893-1180`) → `FlashInferMLASparseImpl.forward_mqa`
(`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:484-605`).

Short prefill gate: if `max_prefill_seq_len <= topk_tokens (2048)`, the indexer skips the logits and top-k and fills causal indices
(`sparse_indexer.py:253-272, 286`). So configs 19-21 run only when some prefill row's context is longer than 2048 tokens.

---

## Config 19: dsa_mqa_logits_prefill / deepgemm_fp8
- **Positions:** dsa_moe.dsa.indexer.mqa_logits_prefill
- **RUNNER:** `vllm.utils.deep_gemm.fp8_fp4_mqa_logits((q, None), (k, k_scale), weights, k_start, k_end, clean_logits=False)`.
  The runner picks the first of `fp8_mqa_logits` and `fp8_fp4_mqa_logits` that exists; the rows' fork has only the second.
  Arguments: q fp8_e4m3 [M,H,128] contiguous; k fp8 [N,128]; k_scale fp32 [N]; weights fp32 [M,H];
  k_start = zeros(M) int32; k_end = arange(N-M+1..N) int32 (single sequence, token-granular causal tail, requires M<=N).
  The timed closure is one call, timed with CUPTI filter "fp8_mqa_logits".
  Code: `f554be2fe8:profiling/runners/attention/dsa_mqa_logits_prefill.py:184-193, 475-494`; registered as backend
  `vllm_deepgemm_fp8` in `vllm_env` (`kernels/dsa_mqa_logits_prefill.py:56-74`). 90553f37a1 (1 row) changes only comments
  and the head check.
- **FORK:** the same wrapper, `fp8_fp4_mqa_logits((q_slice, None), (k_quant, k_scale.view(f32).squeeze(-1)), weights[chunk], chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, clean_logits=False)`
  at `sparse_indexer.py:315-324`. The wrapper is `vllm/utils/deep_gemm.py:593-636`, which calls `_dg.fp8_fp4_mqa_logits`.
  - Shapes: H=32, D=128. q comes from `fwht128_quant_fp8`, with the per-token scale folded into the weights (`attention.py:369-375`).
  - K is the gathered workspace of pool-granular FP8 keys plus fp32 scales, produced by `cp_gather_indexer_k_quant_cache`
    (:290-297; skipped for query sub-chunks after the first).
  - Spans are pool-granular: ks = the request's row start; ke = row_start + (pos+1)//4 (`v1/attention/backends/mla/indexer.py:452-470`).
    The key end advances by one per 4 queries, and several requests can share one call.
  - Chunking (`indexer.py:1210-1258, 1345`): requests are packed while `rows × pools × 4 B ≤ VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
    (512 MiB, `envs.py:1087`) and the K workspace fits. A single request over budget is split along queries into
    `max_q = 128Mi / N` rows. Each chunk is one logits call plus one top-k call.
  - `clean_logits=False`: yes.
- **VERDICT:** SAME-FN-DIFF-ARGS.
  - The function, dtypes and head count match.
  - The span layout differs: runner 1 key per query in one sequence; fork 1 pool per 4 queries, ks≠0 for packed requests.
    This changes the scheduled (query-tile × key-tile) work, so it changes time but not the kernel chosen.
- **VERSIONS:** rows in image cu130 (fork 0b8edfb), DeepGEMM 2.5.0 (cmake pin 891d57b4), JIT kernel
  `impls/sm100_fp8_mqa_logits.cuh` (checked inside the image). Ours: DeepGEMM 2.8.0 (pin e1f418c2, vllm-project/DeepGEMM),
  kernel `impls/sm100_mqa_logits.cuh`.
  - The kernel source was rewritten (931-line diff), and the scheduler and layout headers also differ.
  - The wrapper body in deep_gemm.py is unchanged.
- **COVERED-BY-DB:** no. The DeepGEMM kernel is different.
- **NOTES:** Memory note dsa-mqa-logits-prefill-b200-cupti-name: these H32 rows read about 5–15% slower than DeepGEMM 2.6.1
  (median 0.92). 2.6.1 to 2.8.0 also changed the headers, so that ratio is not ours. The DeepGEMM logits row stride
  (expected N+256) is unverified because its csrc is not on disk.

## Config 20: dsa_topk_prefill / deep_select
- **Positions:** dsa_moe.dsa.indexer.topk_prefill
- **RUNNER:** `torch.ops.deep_select.topk(logits, k, None, row_ends, False, False, None, out, None, -1, -inf, False, False)`.
  - logits: fp32 view [M, :N] of a [M, stride] buffer, seeded N(0,1).
  - row_ends: arange(N-M+1..N) int32; there is no `begin`.
  - out: int32 [M,k] with a 32 B-aligned stride. One logical call; CUPTI counts every launch (kernel_name=None).
  - Code: `011cdb91:profiling/runners/attention/dsa_topk_prefill.py:247-289, 316-388`; registration
    `011cdb91:profiling/kernels/dsa_topk_prefill.py:127-140` (`vllm_upstream_fork_env`).
  - The rows carry hash 151a70c7ec, which has no deep_select code. They were run from an uncommitted tree, committed
    next as 011cdb91 (child of 151a70c7ec, 2026-10-08). The runner is unchanged from 011cdb91 to HEAD.
- **FORK:** does not call DeepSelect on the kpool prefill path. Prefill top-k is hard-wired to
  `torch.ops._C.top_k_per_row_prefill` (`sparse_indexer.py:341-350`).
  - `kernel_config.sparse_indexer_topk_backend` reaches only the decode dispatcher `get_indexer_topk` (:578-585).
  - DeepSelect prefill exists only in the non-kpool generic indexer (`v1/attention/backends/mla/sparse_indexer.py:135-158`),
    which glm5next does not use.
  - The fork cannot choose a backend per shape for prefill.
- **VERDICT:** DIFFERENT. The fork calls `top_k_per_row_prefill` (config 21); the rows time `deep_select.topk`.
- **VERSIONS:** rows env = the alignment checkout at 892da0822f (per the brief), DeepSelect pin d96d33af
  (`cmake/external_projects/deepselect.cmake:22`), which matches ours.
- **COVERED-BY-DB:** no. The fork does not call this kernel.
- **NOTES:** The sim's `indexer_topk_backend` = deep_select or fastest (added in 011cdb91) models a fork change that does
  not exist at 892da0822f.

## Config 21: dsa_topk_prefill / vllm_cuda
- **Positions:** dsa_moe.dsa.indexer.topk_prefill
- **RUNNER:** `torch.ops._C.top_k_per_row_prefill(logits, row_starts, row_ends, out, M, logits.stride(0), logits.stride(1), top_k)`.
  - logits: fp32 [M, :N] view of a [M, logits_row_stride] buffer, seeded N(0,1).
  - row_starts = 0; row_ends = arange(N-M+1..N); out int32 [M,top_k] contiguous; top_k = 512.
  - CUPTI filter "topKPerRowPrefill" also captures the second radix launch when M>12288.
  - Code: `151a70c7ec:profiling/runners/attention/dsa_topk_prefill.py:161-195, 385-443`; `vllm_env`.
- **FORK:** the same op at `sparse_indexer.py:341-350`: (logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, pool_topk, num_rows,
  logits.stride(0), logits.stride(1), select_k = 2048//4 = 512).
  - pool_topk is a fresh `torch.empty((rows,512), int32)` (contiguous; the kernel assumes row stride = topK).
  - logits is the DeepGEMM output, row stride = DeepGEMM's padded stride (unverified, see 19).
  - Host: `csrc/libtorch_stable/sampler.cu:988-1019`. 1 launch, plus a second (radix) launch when rows > 12288.
- **VERDICT:** SAME-FN-DIFF-ARGS. Span layout as in 19: the fork's row_end is pool-granular, (pos+1)//4, and row_start≠0
  for packed requests. Scores are real logits, not N(0,1).
  - Per-row work scales with span length, so time can move.
  - The launch set is the same.
- **VERSIONS:** rows: image cu130-3f667d7e (fork 3f667d7, torch 2.13). Ours: 892da0822f.
  - `sampler.cu` changed (268 lines). The new `useTopK512Optimization` template flag defaults to false and is set only
    on ROCm gfx950 paths. With it false, `finalBinFits` reduces to the old test, and the radix and insertion sort are
    only refactored.
  - The prefill host function, the `topKPerRowPrefill` wrapper and the op schema are byte-identical.
  - The CUDA prefill logic is unchanged.
- **COVERED-BY-DB:** yes. The kernel is source-equivalent for the CUDA prefill instantiation, and both builds use torch 2.13/cu130.
- **NOTES:** Rows before 2026-10-08 used sorted linspace scores (1.7× slow); these rows are the seeded-normal re-run.

## Config 23: mla_cache_append / vllm_cuda
- **Positions:** dsa_moe.dsa.sparse_mla.mla_cache_append
- **RUNNER:** `vllm._custom_ops.concat_and_cache_mla(kv_c, k_pe, cache, slot_mapping, "fp8_e4m3", scale)`.
  - kv_c bf16 [T,512] contiguous; k_pe = [T,1,0].squeeze(1).
  - cache: float8_e4m3fn [blocks,64,512]; slot_mapping: int64 random permutation; scale: 1.0 fp32 0-d.
  - CUPTI "concat_and_cache_mla_kernel".
  - Code: `f8bd625a4d:profiling/runners/attention/mla_cache_append.py:425-494`; `vllm_env` (image cu130).
- **FORK:** `MLAAttention.update_kv_cache` (`mla_attention.py:767-803`) → `impl.do_kv_cache_update` → `ops.concat_and_cache_mla(kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten(), kv_cache_dtype, k_scale)`
  (`vllm/v1/attention/backend.py:1072-1092`; the FlashInfer sparse impl does not override it).
  - kv_c_normed: a fresh contiguous [T,512] bf16 from `fused_q_kv_rmsnorm` (`models/common/ops/fused_qk_rmsnorm.py:59-71`).
  - k_pe: width 0.
  - Cache: [kernel_blocks, 64, 512] FP8 (uint8). The manager block is larger (kpool and hybrid alignment,
    `platforms/interface.py:960-968`); the kernel block = max supported = 64 (`v1/worker/utils.py:330`,
    backend sizes [32,64]).
- **VERDICT:** SAME. The same op, dtypes, rope 0 and block 64. The cache tensor dtype tag (fp8 vs uint8) is irrelevant
  because the kernel reinterprets the data pointer.
- **VERSIONS:** rows: fork 0b8edfb (image cu130, torch 2.11). Ours: 892da0822f (torch 2.13).
  - `concat_and_cache_mla_kernel` is byte-identical; the launch macro and grid/block are the same.
  - The host adds only an `nvfp4_ds_mla` branch, which is not taken.
  - The `scaled_vec_conversion<uint8_t,__nv_bfloat16>` and `scaled_convert` bodies are identical (`quant_utils.cuh`).
- **COVERED-BY-DB:** yes. The kernel source is unchanged; only the toolchain differs.
- **NOTES:** Runs every prefill iteration, once per layer.

## Config 24: dsa_sparse_index_remap / vllm_triton
- **Positions:** dsa_moe.dsa.sparse_mla.index_remap
- **RUNNER:** `vllm.v1.attention.backends.mla.sparse_utils.triton_convert_req_index_to_global_index(req_id, block_table, token_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2176, HAS_PREFILL_WORKSPACE=<spec>, prefill_workspace_request_ids, prefill_workspace_starts, return_valid_counts=True)`.
  - CUPTI `interval_union=True` counts every launch, including wrapper allocations.
  - Code: `a1526f6f37:profiling/runners/attention/dsa_sparse_index_remap.py:893-1017` (d27fcdf0ac `:876-997` drops only
    the validation limits).
  - 275 of 325 rows have `workspace_partition = suffix:1@1` or `suffix:0@1`, so HAS_PREFILL_WORKSPACE=True; 50 rows are "none".
- **FORK:** in `FlashInferMLASparseImpl.forward_mqa`, `_convert_logical_to_physical_topk` (`flashinfer_mla_sparse.py:593`)
  → `SparseMLAIndexGroup._convert_once` (`v1/attention/backends/mla/index_group.py:114-164`) →
  `triton_convert_req_index_to_global_index(req_id_per_token, block_table, topk_buffer[:rows], BLOCK_SIZE=64, BLOCK_STRIDE_ROWS=<flat_kv_row_view>, NUM_TOPK_TOKENS=2176, return_valid_counts=True, out=<prealloc>, valid_counts_out=<prealloc>)`.
  - No prefill workspace: prefill rows map through the block table.
  - It runs on a side stream, once per layer (each DSA layer is its own index group).
- **VERDICT:** SAME-FN-DIFF-ARGS, and the code behind the function changed.
  - Rows' fork (3f667d7) `_remap_tiling`: 2176 is not a power of two, so 17 tiles × 128, 4 warps, atomic counts, plus
    `full_like(-1)` and `zeros` init launches.
  - Ours (a369a7becc "Reduce GLM sparse MLA preparation overhead"): one tile padded to 4096, 8 warps, no atomics, no init
    launches, preallocated outputs.
  - HAS_PREFILL_WORKSPACE differs for 275 rows.
  - Both the kernel configuration and the launch count differ.
- **VERSIONS:** rows: image cu130-3f667d7e. The image's `sparse_utils.py` sha256 equals git 3f667d7. triton 3.7.1 in both.
  `sparse_utils.py` 3f667d7..892da0822f: +154/-33.
- **COVERED-BY-DB:** no. The tiling and launch set changed.
- **NOTES:** The selected width 2176 = round_up(2048+4-1, 128) (`common/model.py:637-650`).

## Config 25: dsa_sparse_mla_attention / flashinfer_trtllm_fp8
- **Positions:** dsa_moe.dsa.sparse_mla.prefill
- **RUNNER:** `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(...)` with:
  - query fp8 [Q,1,64,512]; kv_cache fp8 [pages,1,64,512]; workspace 128 MiB.
  - qk_nope_head_dim=512, kv_lora_rank=512, qk_rope_head_dim=0.
  - block_tables int32 [Q,1,2176] of token slots, zero-padded; seq_lens = valid counts.
  - max_seq_len=2176, bmm1=0.0625, bmm2=1.0, sparse_mla_top_k=2176, sparse_mla_top_k_lens=seq_lens.clamp(1).
  - CUPTI "fmhaSm100", one launch.
  - Code: `82c85c3203:profiling/runners/attention/dsa_sparse_mla_attention.py:758-841, 938`; `vllm_env` (cu130-3f667d7e).
- **FORK:** `_run_mqa_kernel` (`flashinfer_mla_sparse.py:667-745`) calls the same function with:
  - query = fp8 q.unsqueeze(1) [rows,1,64,512]; kv_cache.unsqueeze(1); 394 MiB workspace.
  - qk_nope_head_dim=256; block_tables = remapped indices.unsqueeze(1), -1 padded; seq_lens = valid counts.
  - max_seq_len = sparse_mla_top_k = 2176 (buffer width, :698); sparse_mla_top_k_lens = seq_lens (:705-710, TP1 has no guard).
  - bmm1 = scale·q_scale·k_scale; bmm2 = k_scale; no LSE.
  - FlashInfer "auto" on SM100 sparse uses trtllm-gen only, because cute-dsl rejects sparse_mla_top_k>0 in both versions.
  - qk_nope_head_dim enters only the autotune cache key, not the kernel.
  - Launch count: one launch per layer over all of the iteration's rows (decode + prefill) when the sparse MQA path is
    used (`mla_attention.py:983-986, 1126`). I found no split of rows > 32,768 in the fork or in the FlashInfer launcher
    (gridDim.z = rows; the CUDA limit is 65535). The sim's 32,768-row per-launch limit is not fork behavior.
  - Q FP8 quant is a separate op: `self._decode_concat_quant_fp8_op` (`mla_attention.py:1102`), after the W_UK bmm (:1097).
  - Dense MHA for prefill with max seq ≤ 2048 exists in the code (`sparse_mla_attention.py:130-138, 415`;
    `mla_attention.py:1205-1227`). Whether an MLA prefill backend accepts qk=v=256 on SM100 is unverified. The sim assumes
    none, based on capture evidence.
- **VERDICT:** SAME-FN-DIFF-ARGS. The differences are qk_nope 512 vs 256 (autotune key only), 0 vs -1 padding past
  seq_lens (never read), scale values, workspace size and the index distribution. None of them changes the selected kernel.
- **VERSIONS:** rows: FlashInfer 0.6.18 (image cu130-3f667d7e). Ours: 0.7.0.
  - The trtllm-gen FMHA cubin artifact path (2d6a5a02…) and the CheckSumHash are identical, so the cubins are the same.
  - `fmhaKernels.cuh` selection diffs:
    - Half-tileQ now also requires maxSeqLenQ ≤ 16. q_len is 1 here, so there is no change.
    - CGA H512 needs a page size of 32, and ours is 64. The headDimV=512 CGA exclusion still holds.
  - The launcher adds DSv4-only arguments (passed as null).
- **COVERED-BY-DB:** yes. The cubin and the selection are the same for q_len=1, page 64 and headDimV=512.
- **NOTES:** The remap (24) and the Q quant are separate launches that the sim prices separately.

## Kpool expansion (selected pools → token slots)
- **Op:** `kpool_ops.expand_pools_and_append_tail(pool_ids:int64[rows,512], q_seq=pos+1, 4)`, one Triton launch
  (`vllm/models/glm5next/nvidia/ops/kpool_compress.py:843-916`), called at `sparse_indexer.py:352-372`.
- **Writes:** pool g becomes tokens 4g..4g+3 (columns 0..2047; -1 for unselected pools). Then the request's trailing
  incomplete pool, tokens [4·⌊(pos+1)/4⌋, pos], goes in columns 2048..2050. The output is [rows, 2051].
- **Into the buffer:** it is copied into `topk_indices_buffer[:, :2051]`. That buffer is 2176 wide
  (round_up(2051,128), `common/model.py:637-650`) and was pre-filled with -1 at `sparse_indexer.py:239`.
- **Extra ops around it:** an int32→int64 cast (:353), the expansion itself, and the slice copy into the buffer.
- **Valid count per row:** min(pos+1, 2048 + (pos+1) mod 4) ≤ 2051. The remap (24) compacts each row to these valid slots
  and reports the count, which becomes the attention's seq_lens / top_k_lens.
