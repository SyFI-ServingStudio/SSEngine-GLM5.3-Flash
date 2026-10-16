# Environment

The version record of pp-impl's environment. `env/make_env.sh` builds it from public sources and `env/check_env.py`
checks it (README, "Reproduce").

## Layout

One environment for the repo and every lane worktree under `lanes/`, in the main tree (all ignored by git):

| Path | What |
|---|---|
| `.venv` | Python 3.12 venv (uv), the packages of `env/requirements.lock` |
| `third_party/vllm` | the vLLM fork at `892da0822f`, installed editable: `import vllm` resolves here |
| `third_party/handoff` | the frozen handoff in use: `doc/handoff/glm53_flash_pp4_b200_v2` of ServingStudioSim at `5153041a` (`PPIMPL_HANDOFF` overrides it) |
| `third_party/ServingStudioSim` | ServingStudioSim at `820adda0` (branch `pp-main`): req-frontend (`PPIMPL_SIM` overrides it) |
| `third_party/sim_7bb224a6/profiling` | the profiler package at the handoff's frozen commit `7bb224a6` (`git archive`): input builders and references for the M1/M2 checks |
| `third_party/enwik9` | the prompt corpus (`PPIMPL_ENWIK9` overrides it) |
| `third_party/wheels` | the causal-conv1d wheel built here, reused when the venv is rebuilt |
| `third_party/downloads` | downloads, each checked by sha256 |

The weights are in the Hugging Face cache (`HF_HOME`), or at `PPIMPL_CHECKPOINT` (`ppimpl/config.py`).

## Versions

From `env/check_env.py`, which fails on any other version:

| Library | Version | Source |
|---|---|---|
| Python | 3.12 | uv |
| torch | 2.13.0+cu130 (CUDA 13.0) | PyPI |
| cuBLAS | 13.1.1.3 | PyPI (torch's pin) |
| vLLM fork | 0.1.dev21920+g892da0822 (`third_party/vllm`, HEAD `892da0822f`) | https://github.com/serendipity-zk/vllm, branch `servingstudio-alignment` |
| FlashInfer | 0.7.2.dev20261008 (nightly, main at `73b63f88`): the build the KDA rows used; replaces the fork's pinned 0.7.0 | https://github.com/flashinfer-ai/whl/releases/tag/nightly-v0.7.2-20261008, sha256 in `env/requirements.lock` |
| nvidia-cutlass-dsl | 4.8.0 (the fork pins 4.7.1) | PyPI |
| Triton | 3.7.1 | PyPI |
| TileLang | 0.1.12 | PyPI |
| DeepGEMM | 2.8.0, vendored in the fork (`vllm/third_party/deep_gemm`) | the fork's native build |
| DeepSelect | the fork's `vllm._deepselect_C` (`torch.ops.deep_select.topk`) | the fork's native build |
| FlashKDA | the fork's `vllm._flashkda_C` (`torch.ops._flashkda_C.fwd`) | the fork's native build |
| causal-conv1d (Dao) | 1.7.0, built from the source tag against torch 2.13. The released cu13 wheel is built for torch 2.10 and fails to load with 2.13 (`undefined symbol c10::impl::cow::materialize_cow_storage`). | https://github.com/Dao-AILab/causal-conv1d, tag `v1.7.0` |

Two kernels follow the design rather than the fork (`measured/kernels_identity.md`, Decisions):
- KDA chunked prefill uses FlashInfer `recurrent_kda(backend="cute-dsl-persistent")`. That backend exists only in the
  nightly the rows used, so the venv carries FlashInfer 0.7.2.dev20261008 with cutlass-dsl 4.8.0 instead of the
  fork's pins (0.7.0 / 4.7.1). `uv pip check` reports exactly these two pins as mismatched. The fork imports and
  dispatches with it. MoE (30), `mm_fp4` (9, 11) and sparse MLA (25) then run on this FlashInfer too.
- The KDA short conv uses Dao causal-conv1d.

`env/requirements.lock` is the venv's `uv pip freeze` without the editable fork and causal-conv1d.

## How `make_env.sh` builds it

1. `uv venv --python 3.12 .venv`, then `uv pip install -r env/requirements.lock`. FlashInfer comes by URL with its
   sha256.
2. Clones the fork and checks out `892da0822f`. It installs it as the fork's `ALIGNMENT.md` says: editable
   (`uv pip install -e`) with `VLLM_USE_PRECOMPILED=1`. The native extensions come from the upstream vLLM wheel of the
   fork's base commit `04730e8` (`https://wheels.vllm.ai/04730e8…/vllm-0.30.1rc1.dev97+g04730e827-…x86_64.whl`,
   sha256 `a6eb62ee…`); the fork changes no native code against that base (`git diff 04730e8 892da0822f -- csrc cmake
   CMakeLists.txt setup.py rust` is empty). Every native file in that wheel is byte-identical to the build this
   repo's measurements ran on. To compile instead, see the fork's `docs/contributing/incremental_build.md`.
3. Builds causal-conv1d 1.7.0 from the source tag into `third_party/wheels/` (`uv build --no-build-isolation`, CUDA
   13 toolkit at `CUDA_HOME`, `MAX_JOBS` default 16: about 10 minutes) and installs it.
4. Clones ServingStudioSim at `820adda0` with the req-frontend submodule
   (https://github.com/uw-syfi/request-factory at `7f6aa9d8`), and `git archive 7bb224a6 profiling` into
   `third_party/sim_7bb224a6`.
5. Downloads enwik9 (http://mattmahoney.net/dc/enwik9.zip) and checks the zip's and the file's sha256.
6. Runs `env/check_env.py`.

Run it as the normal user, never as root, with a short `TMPDIR`: vLLM's zmq ipc path is `TMPDIR` + a uuid and must fit
in 107 characters (`bench/slurm/env.sh` makes one short TMPDIR per job under `PPIMPL_TMP`).

## Other tools

- `cargo` (Rust): `bench/run.py` builds req-frontend's `session_runner` on first use.
- `/usr/bin/ninja` 1.11.1 for FlashInfer's JIT: processes that share `~/.cache/flashinfer` with another ninja version
  hash build commands differently and rebuild every object. `check_env.py` warns when another ninja is first on PATH.
- The M1 kernel re-measure (`bench/slurm/kernels_*.sbatch`) runs the ServingStudioSim profiler from
  `third_party/ServingStudioSim` with that checkout's own uv project env (`uv sync` there) and its CUPTI extension
  (`MOESIM_CUPTI_BUILD_DIR`); see the profiler's README.
