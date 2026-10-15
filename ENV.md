# Environment

One shared environment for every lane, outside the worktrees:

| Path | What |
|---|---|
| `pp-impl/.venv` | Python 3.12 venv (uv). `import vllm` resolves to `third_party/vllm`. |
| `pp-impl/third_party/vllm` | Detached git worktree of the fork repository at `892da0822f`, plus the compiled outputs copied from a built tree of the same HEAD (no rebuild) |
| `pp-impl/third_party/sim_7bb224a6/profiling` | ServingStudioSim profiler package at the handoff's frozen commit (`git archive`). Input builders and references for the M1 checks. |
| `pp-impl/third_party/causal_conv1d_try/pkg` | Dao causal-conv1d 1.7.0 built from source for torch 2.13 (optional; only the profiler's config-4 callable uses it) |

All of these are ignored by git.

## Versions

From `env/check_env.py`:

| Library | Version |
|---|---|
| torch | 2.13.0+cu130 (CUDA 13.0) |
| cuBLAS | 13.1.1.3 |
| vLLM fork | 0.1.dev21920+g892da0822 (`third_party/vllm`, HEAD `892da0822f`) |
| FlashInfer | 0.7.0 |
| nvidia-cutlass-dsl | 4.7.1 |
| Triton | 3.7.1 |
| TileLang | 0.1.12 |
| DeepGEMM | 2.8.0, vendored in the fork (`vllm/third_party/deep_gemm`) |
| DeepSelect | the fork's `vllm._deepselect_C` (`torch.ops.deep_select.topk`) |
| FlashKDA | the fork's `vllm._flashkda_C` (`torch.ops._flashkda_C.fwd`) |
| causal-conv1d (Dao) | 1.7.0. The released cu13 wheel is built for torch 2.10 and fails against torch 2.13 (`undefined symbol c10::impl::cow::materialize_cow_storage`), so it is built from source. |

The fork's glm5next path needs no Dao causal-conv1d (it uses vLLM's Triton conv) and no FlashInfer nightly (it runs
KDA prefill with FlashKDA). The profiler row of config 5 used FlashInfer 0.7.2.dev20261008 with cutlass-dsl 4.8.0.
Installing it here would replace the fork's pinned 0.7.0, so it stays in its own profile env
(`profile_envs/flashinfer_kda`). The correctness check of that callable runs there (`bench/kernels_correctness.py
--kda-flashinfer`).

## Recreate

Prerequisites:
- A built checkout of the fork at `892da0822f` with its own uv venv. We use
  `/raid/kanzhu/ServingStudio/wt-flash-dflash2/alignment/profiler/vllm`, read only.
- `uv`, run as the normal user, never as root.
- The workspace `.env` sourced, so `TMPDIR` and `UV_CACHE_DIR` point at `/raid/kanzhu/ServingStudio/{tmp,uv-cache}`.
  Keep `TMPDIR` short: vLLM's zmq ipc path is `TMPDIR` + a uuid and must fit in 107 characters.

```bash
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
lanes/model/env/make_env.sh /raid/kanzhu/ServingStudio/pp-impl \
    /raid/kanzhu/ServingStudio/wt-flash-dflash2/alignment/profiler/vllm
```

`make_env.sh` does four things:
1. `git worktree add --detach third_party/vllm 892da0822f` from the fork repository. Then `rsync --files-from
   env/build_outputs.txt`: the 24 ignored build outputs of the built tree (`_C`, `_deepselect_C`, `_flashkda_C`,
   FlashMLA, the vendored DeepGEMM, FA2/FA3 and so on).
2. `uv venv --python /usr/bin/python3.12 .venv` and `uv pip install --offline -r env/requirements.lock`. The lock file
   is the reference venv's `uv pip freeze` without `-e vllm`.
3. Copies the reference venv's editable-vllm finder, `.pth` and dist-info, and rewrites their paths to
   `third_party/vllm`.
4. `git archive 7bb224a6 profiling` from `wt-pp-main` into `third_party/sim_7bb224a6`.

Then it runs `env/check_env.py`, which imports every library above and exits non-zero if a required one fails.

Dao causal-conv1d (optional), built from the v1.7.0 source against this venv's torch:

```bash
T=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try
curl -sL -o $T/src.tar.gz https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.7.0.tar.gz
mkdir -p $T/src && tar -xzf $T/src.tar.gz -C $T/src
CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=32 \
  uv pip install --python /raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python --no-build-isolation --no-deps \
  --no-cache --target $T/pkg $T/src/causal-conv1d-1.7.0
# use: PYTHONPATH=$T/pkg
```

## Use

```bash
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python ...
```

GPU work goes through Slurm (`bench/slurm/*.sbatch`); the scripts source `.env` and call the venv's Python directly.
