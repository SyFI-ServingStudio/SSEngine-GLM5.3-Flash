# pp-impl: GLM-5.3-Flash NVFP4 · PP4 on B200, real implementation

Implements the sim-to-real handoff `glm53_flash_pp4_b200_v2` (ServingStudioSim commit `5153041a`,
`doc/handoff/glm53_flash_pp4_b200_v2/`), the frozen handoff in use; v2 replaces `glm53_flash_pp4_b200_v1`
(frozen `7bb224a6`) with the whole-context prefix-hit rule, its session targets and a refit microbatch fit. The
handoff's README, contract and milestone files 2–7 are the specification; its `data/` holds the targets
(`bench/common.py` `HANDOFF`: `third_party/handoff`, or `PPIMPL_HANDOFF`).

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

## Reproduce

**Hardware and system.** One node with 4 × NVIDIA B200 in one NVLink domain (measured on HGX B200, every pair NV18),
driver 580 or newer (measured: 580.178.04), the CUDA 13 toolkit (nvcc, for the causal-conv1d build), git, curl,
unzip, [uv](https://docs.astral.sh/uv/), `cargo` (req-frontend), about 200 GB of disk for the weights. Slurm is
optional: the job scripts also run with `bash`. `measured/setup.md` records the node the measurements ran on.

**Environment** (`ENV.md` has the versions and sources). As the normal user, never as root:

```bash
env/make_env.sh            # .venv and third_party/ in this tree, from public sources; ends with env/check_env.py
```

**Weights.** `nvidia/GLM-5.3-Flash-NVFP4` at revision `da920bb0b9f4a06727223a349e55468e38352348` (public, ~191 GB),
into the Hugging Face cache:

```bash
export HF_HOME=/path/to/hf          # default ~/.cache/huggingface
.venv/bin/hf download nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348
```

**Settings.** Every path has a default; set a variable only to point elsewhere.

| Variable | Default | What |
|---|---|---|
| `PPIMPL_CHECKPOINT` | the snapshot above under `$HF_HOME/hub` | the checkpoint directory |
| `PPIMPL_SIM` | `third_party/ServingStudioSim` | ServingStudioSim at `820adda0`: the handoff and req-frontend |
| `PPIMPL_ENWIK9` | `third_party/enwik9` | the prompt corpus |
| `PPIMPL_PYTHON` | `.venv/bin/python` | the interpreter the job scripts use |
| `PPIMPL_TMP` | `$TMPDIR`, else `/tmp` | parent of each job's short `TMPDIR` |
| `PPIMPL_REQ_FRONTEND_BIN` | `build/req-frontend/release/session_runner` (built on first use) | the client binary |
| `PPIMPL_FI_TACTICS` | `ppimpl/data/flashinfer_tactics_b200.json` | the tuned FlashInfer tactics (`bench/tune_tactics.py`) |

**Run one milestone check end to end**: M3 (`4_iteration.md`), one 1k-token request through the 4-stage server,
driven by req-frontend, compared with the handoff's targets:

```bash
.venv/bin/python -m pytest -q                       # CPU tests (scheduler, KV, tiers, protocol, server on a fake model)
mkdir -p measured/logs
RUNS=one_1k sbatch -p <partition> --time=00:20:00 -o measured/logs/%j.out bench/slurm/runs.sbatch
#   without Slurm, on GPUs 0-3:  RUNS=one_1k bash bench/slurm/runs.sbatch
.venv/bin/python -m bench.report --summarize measured/runs_m3 --out measured    # measured/iteration.md
```

The run directory `measured/runs_m3/one_1k/` holds the server and client logs, `metrics.json` and `report.json`.
Model correctness against vLLM TP4 at the same prompt: `sbatch -p <partition> --time=00:30:00 --gres=gpu:4 -o
measured/logs/%j.out bench/slurm/chain.sbatch "chain vllm"`. The other milestones have their job scripts in
`bench/slurm/`, each with its usage at the top.
