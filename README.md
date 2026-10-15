# pp-impl: GLM-5.3-Flash NVFP4 · PP4 on B200, real implementation

Implements the sim-to-real handoff `glm53_flash_pp4_b200_v1` (ServingStudioSim branch `pp-main`,
`doc/handoff/glm53_flash_pp4_b200_v1/`, frozen commit `7bb224a6`). The handoff's README, contract and milestone
files 2–7 are the specification; its `data/` holds the targets.

## Provenance (fixed before any code)

1. **Engine provenance: hybrid.** Our own small engine (scheduler, KV/prefix cache, DRAM/SSD tiers, pipeline
   runtime, server). Kernels and model-layer code are reused from the vLLM fork (`servingstudio-alignment` line,
   commit `892da0822f`, `vllm/models/glm5next`): imported as a library where it works unchanged, vendored into
   `ppimpl/` only where we must change it (e.g. the stage split and the mHC hand-off between stages).
2. **Runtime dependencies:** PyTorch, the vLLM fork as a kernel/layer library (its custom ops, FlashInfer, DeepGEMM,
   DeepSelect, TileLang, causal-conv1d as it pins them), safetensors, FastAPI/Uvicorn (server), NCCL via
   torch.distributed. Nothing else without a note in this file.
3. **Model execution** is owned by `ppimpl` (one process per stage, our forward loop); no vLLM engine,
   scheduler, model runner or KV-cache manager runs.
4. **Attention backends:** DSA = DeepGEMM `fp8_mqa_logits` + top-k + FlashInfer TRTLLM sparse MLA; KDA = FlashInfer
   `recurrent_kda` (cute-dsl persistent); per `2_kernels.md`.
5. **Existing serving engines in execution:** none in the serving path. vLLM serves only as the separate
   correctness reference (`1_contract.md` §4, TP4 + EP).

Goal: small codebase, maximum reuse.

## Layout

| Path | Content |
|---|---|
| `ppimpl/` | the engine |
| `bench/` | harnesses that run the handoff's milestones and write `measured/` |
| `measured/` | measurements in the handoff's `measured/` format; `issues.md` is the gap log |
| `lanes/` | per-lane git worktrees (ignored) |
