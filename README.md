# pp-impl: GLM-5.3-Flash NVFP4 · PP4 on B200, real implementation

Implements the sim-to-real handoff `glm53_flash_pp4_b200_v1` (ServingStudioSim branch `pp-main`,
`doc/handoff/glm53_flash_pp4_b200_v1/`, frozen commit `7bb224a6`). The handoff's README, contract and milestone
files 2–7 are the specification; its `data/` holds the targets.

## Provenance (fixed before any code)

1. **Engine provenance: hybrid.** Our own small engine (scheduler, KV/prefix cache, DRAM/SSD tiers, pipeline
   runtime, server). Kernels and model-layer code are reused from the vLLM fork (`servingstudio-alignment` line,
   commit `892da0822f`, `vllm/models/glm5next`), imported as a library. Where a stage needs different behavior (the
   stage split, the mHC hand-off between stages, the kernel choices of `2_kernels.md`) `ppimpl/model/pp_patch.py`
   overrides it at run time; its docstring lists every override, why, and the test that proves it still applies.
2. **Runtime dependencies:** PyTorch, the vLLM fork as a kernel/layer library (its custom ops, FlashInfer, DeepGEMM,
   DeepSelect, TileLang, causal-conv1d as it pins them), safetensors, Uvicorn (the server is a bare ASGI app), pyzmq +
   msgspec (control messages), torch.distributed gloo (barriers; activations go over CUDA IPC, `ppimpl/runtime/README.md`). Nothing else without a note in this file.
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
| `ppimpl/` | the engine: `config.py`, `model/` (one stage on the vLLM fork), `runtime/` (controller, workers, transport), `engine/` (scheduler, KV, tiers), `serve/` (HTTP) |
| `bench/` | harnesses that run the handoff's milestones and write `measured/` (`python -m bench.<name>`); `bench/slurm/` the job scripts |
| `env/` | `make_env.sh` builds the environment, `check_env.py` checks it; versions in `ENV.md` |
| `tests/` | `pytest -q` runs the CPU tests; `-m gpu` the GPU tests |
| `measured/` | measurements in the handoff's `measured/` format; `issues.md` is the gap log |
| `lanes/` | per-lane git worktrees (ignored) |
