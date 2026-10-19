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
| `ppimpl/` | the engine: `config.py`, `model/` (one stage on the vLLM fork), `runtime/` (controller, workers, transport), `engine/` (scheduler, KV), `kvpool/` (DRAM/SSD tiers behind HBM, TIERS.md §13), `serve/` (HTTP) |
| `bench/` | harnesses that run the handoff's milestones and write `measured/` (`python -m bench.<name>`); `bench/slurm/` the job scripts |
| `env/` | `make_env.sh` builds the environment, `check_env.py` checks it; versions in `ENV.md` |
| `tests/` | `pytest -q` runs the CPU tests; `-m gpu` the GPU tests |
| `measured/` | measurements in the handoff's `measured/` format; `issues.md` is the gap log |
| `lanes/` | per-lane git worktrees (ignored) |
| `old-wt/` | retired lane worktrees and snapshots, kept with their branches (ignored) |

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
| `PPIMPL_REQ_FRONTEND` | `third_party/req-frontend` | req-frontend's checkout at 8723d90 or later (the session slot release; a run fails on an older or modified checkout). Its `session_runner` is built into `build/req-frontend/release/` on first use and rebuilt when the checkout's commit changes; `run.json` records its source, commit and sha256 |
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

## Serve

`ppimpl serve [<model>]` starts the server the way `vllm serve <model>` does (`ppimpl/serve/launch.py`). `<model>`
defaults to `nvidia/GLM-5.3-Flash-NVFP4`, so `ppimpl serve` alone serves it; it can also be
a Hugging Face repo id or a local checkpoint directory, and only one model is served: `nvidia/GLM-5.3-Flash-NVFP4`
(`Glm5NextForConditionalGeneration`, NVFP4) as 4 pipeline stages on 4 GPUs. Any other model is refused from its
`config.json` before anything else is downloaded. A repo id is fetched into the Hugging Face cache (`HF_HOME`) on
first use (~191 GB), at the revision above unless `--revision` says otherwise.

```bash
uv pip install --python .venv/bin/python --no-deps -e .   # once: the `ppimpl` command (or .venv/bin/python -m ppimpl)
ppimpl serve                                                                 # nvidia/GLM-5.3-Flash-NVFP4, HBM only
ppimpl serve /path/to/checkpoint                                             # the same model from a local directory
ppimpl serve --dram-gb 50 --ssd-gb 128 --ssd-dir /nvme/ppimpl   # with DRAM and SSD tiers
```

The run config is the design setting of the M6 evaluation (`measured/eval_m6/summary.md`, "st10"): scheduler step
`good` (6_scheduler.md), the exact prefix cache (`keep_sessions`) and, with a tier, exclusive DRAM writes and a 10%
loading area (TIERS.md). The options:

| Option | Config (`ppimpl/config.py`) | Default |
|---|---|---|
| `<model>` | `checkpoint` (the snapshot directory), after the model check | — |
| `--host`, `--port` | `host`, `port` (the engine also takes the next 11 ports on 127.0.0.1) | 0.0.0.0, 8000 |
| `--served-model-name` | `served_model_name`: the OpenAI API's model name | `<model>` as given |
| `--max-model-len` | `max_model_len`: longest prompt + output the API takes | 1,048,576 (the model's) |
| `--dram-gb`, `--ssd-gb` | `tiers.dram_gb`, `tiers.ssd_gb` per GPU; either one turns tiers on | no tiers |
| `--ssd-dir` | `tiers.ssd_dir`: the SSD shard sets, made once with `python -m ppimpl.kvpool.shards create --dir DIR --gb <ssd-gb>` (TIERS.md §5) | `$PPIMPL_SSD_DIR` |
| `--run-dir` | `run_dir`: logs (`server_config.yaml`, `microbatches.jsonl`, `worker*.log`, ...) | `./run` |
| `--config YAML` | any `RunConfig` field, applied on top of the design setting; the options above win | — |

`python -m ppimpl serve --config <yaml>` without a model still takes the whole run config from the YAML (bench/).

The endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions` (a string or token-id prompt),
`POST /v1/chat/completions` (the checkpoint's chat template; `tools` and `chat_template_kwargs` pass through), and
req-frontend's `POST /inference/v1/generate`. **The engine computes prefill and one output token** (greedy; PLAN.md §1:
no decode path), so `max_tokens` must be 1 (the default here); sampling fields are ignored, and `n` > 1, `echo` and
`logprobs` are refused. The usage block reports the prompt tokens served from the prefix cache
(`prompt_tokens_details.cached_tokens`). The first start compiles FlashInfer's JIT kernels into `~/.cache` (minutes);
later starts reuse them.

## Docker

`docker/Dockerfile` builds the serving environment of `ENV.md` (`env/make_env.sh` with `STEPS="venv vllm conv1d"`:
CUDA 13.0 devel on Ubuntu 24.04, uv's Python 3.12, `env/requirements.lock`, the vLLM fork, causal-conv1d built from
source) with this repo installed; the bench-only parts (ServingStudioSim, req-frontend, enwik9) are not in it. Build
from the repo root (CPU only, ~10 min, 8 of them the causal-conv1d build; the image is 16.6 GB; `--build-arg MAX_JOBS=N` sets the causal-conv1d build's parallel nvcc jobs,
default 16):

```bash
docker build -f docker/Dockerfile -t ppimpl .
docker run --rm ppimpl check-env                  # env/check_env.py, no GPU needed
```

Run it as `vllm/vllm-openai` is run, with the subcommand first:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ppimpl-cache:/root/.cache \
  ppimpl serve --served-model-name glm53-flash

curl localhost:8000/health
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm53-flash", "prompt": "The capital of France is", "max_tokens": 1}'
```

- `--gpus`: 4 B200 in one NVLink domain; with more on the host, name four: `--gpus '"device=0,1,2,3"'`.
- `--ipc=host`: the stages pass activations over CUDA IPC and the intake processes share prompts through `/dev/shm`.
- `ppimpl-cache` keeps the JIT and autotune caches (FlashInfer, Triton, TileLang, DeepGEMM) across containers: the
  first start compiles FlashInfer's JIT kernels and is ready after 11-16 min, later ones after ~50 s (jobs 8495, 8512,
  8517; natively 65 s). Without the volume every start compiles again. The weights' cache is the host's, mounted over `/root/.cache/huggingface`.
- Tiers: `--dram-gb 50` pins that much host memory per GPU. The SSD tier reads and writes through io_uring, which
  Docker's default seccomp profile denies: run with `--security-opt seccomp=docker/seccomp.json` (the default profile
  plus the three io_uring syscalls; `docker/make_seccomp.py` writes it). Mount a directory for the shards and make its
  shard set once: `docker run --rm -v /nvme/ppimpl:/ssd ppimpl python -m ppimpl.kvpool.shards create --dir /ssd --gb
  128`, then serve with `-v /nvme/ppimpl:/ssd ... --ssd-gb 128 --ssd-dir /ssd`. (The tiers in a container are not
  measured yet; the GPU smoke test, `bench/slurm/docker_smoke.sbatch`, runs HBM only.)
- Logs go to `/workspace/run` in the container (`-v $PWD/run:/workspace/run` keeps them). `TMPDIR` is `/tmp`: vLLM's zmq
  ipc path is `TMPDIR` + a uuid and must fit in 107 characters, so the entry point refuses a `TMPDIR` over 64.
- `<image> <command>` runs any other command (`bash`, `python -m ...`).
