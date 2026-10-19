# ppimpl

A pipeline-parallel prefill server for GLM-5.3-Flash NVFP4 on 4×B200 with HBM, DRAM and SSD KV-cache tiers.

ppimpl serves one model, [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), as
4 pipeline stages on 4 NVIDIA B200 GPUs. Its exact prefix cache keeps whole sessions' context (KV and KDA state) in
HBM. Two optional tiers sit behind HBM: pinned host DRAM and NVMe SSD. The server is OpenAI-compatible
(`/v1/completions`, `/v1/chat/completions`). The design was chosen and sized with the
[ServingStudio](https://github.com/SyFI-ServingStudio/ServingStudioSim) simulator.

> [!IMPORTANT]
> **Prefill and one output token only.** The engine has no decode path. Each request returns exactly one token,
> picked greedily: `max_tokens` must be 1, and sampling parameters are ignored. ppimpl is meant to be the prefill
> side of a deployment whose decode runs elsewhere, or to measure time to first token (TTFT).

**Contents:** [Results](#results) · [Features](#features) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [API](#api) · [Docker](#docker) · [Performance](#performance) ·
[Architecture](#architecture) · [Development](#development) · [Reproduce](#reproduce) ·
[Troubleshooting](#troubleshooting) · [License](#license)

## Results

The M6 session workload is a closed loop of 400 concurrent sessions replaying the TraceLab v3 session trace for
20 minutes. The numbers below count requests whose first token comes in minutes 10–20. All systems ran on 4×B200
nodes of the same cluster, with the same client (req-frontend) and the same trace. TTFT is the time from submit to
first token.

| System (4×B200) | KV storage per GPU | Requests/s | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|
| **ppimpl**, PP4, full setting | HBM ~100 GB + DRAM 50 GB + SSD 128 GB | **40.51** | **0.61 / 1.52 / 60.43** |
| vLLM DP4/EP4 + vllm-router `cache_aware` | HBM only | 4.64 | 48.31 / 148.23 / 212.89 |
| vLLM TP4 | HBM only | 1.17 | 319.95 / 344.78 / 349.48 |

The vLLM runs use the same vLLM fork whose kernels ppimpl uses (`892da0822f`). With vLLM's CPU offload at 50 GB per
engine, DP4/EP4 gives 4.37 r/s in the same window. Every setting and window is in
[docs/results/eval_m6.md](docs/results/eval_m6.md), with N = 100 and 300, a simulator column, and the
smaller-HBM settings. The baselines' setup is in [docs/results/vllm_baselines.md](docs/results/vllm_baselines.md).

## Features

- **Pipeline parallelism over 4 GPUs.** Layers are split 12/11/11/11. Activations move between stages over CUDA IPC
  by the copy engine. Microbatches up to 8,192 tokens replay CUDA graphs.
- **Exact prefix cache.** Sessions are kept whole and evicted least recently used first. A context resumes only where
  both its KV and its KDA (linear-attention) state are saved.
- **KV tiers behind HBM.** Every GPU stores and moves a quarter of every context ("balanced storage"):
  - DRAM: pinned memory on the GPU's NUMA node.
  - SSD: O_DIRECT through Linux AIO (io_uring optional).
  - HBM evictions are written to DRAM and DRAM evictions to SSD (a victim chain).
  - Tier reads land in a reserved 10% loading area of HBM.
- **Scheduler.** It forms microbatches of up to 32,768 tokens, shortest remaining prefill first. Landed tier reads
  go first, and so does any request that has waited 60 s.
- **OpenAI-compatible server.** Completions and chat completions, with streaming. Prompts are text or token ids.
  Chat uses the checkpoint's own chat template. HTTP runs in 16 intake processes, and each response reports the
  prompt tokens served from cache.
- **Kernels reused, not copied.** Model layers and kernels come from a vLLM fork, used as a library: FlashInfer,
  DeepGEMM, DeepSelect, TileLang, causal-conv1d. No vLLM engine, scheduler or KV manager runs.

### Supported hardware and model

| | Supported | Tested |
|---|---|---|
| Model | `nvidia/GLM-5.3-Flash-NVFP4` (`Glm5NextForConditionalGeneration`, NVFP4). Any other model is refused at start. | revision `da920bb0b9f4a06727223a349e55468e38352348` (~191 GB) |
| GPUs | 4 × B200 in one NVLink domain | HGX B200, every pair NV18 ([docs/results/setup.md](docs/results/setup.md)) |
| Driver / CUDA | driver ≥ 580, CUDA 13.0 | 580.178.04 |
| OS | Linux x86_64 | Ubuntu 24.04 (the image); [docs/results/setup.md](docs/results/setup.md) has the host |
| SSD tier | a local file system on NVMe that supports O_DIRECT | md RAID0 of 8 NVMe |

## Quick start

### With Docker

Build the image from the repo root. The build needs no GPU, takes about 15 minutes, and makes a 17 GB image.
Then serve with one command:

```bash
docker build -f docker/Dockerfile -t ppimpl .
docker compose -f docker/compose.yaml up
```

or with `docker run`:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ppimpl-cache:/root/.cache \
  ppimpl
```

With no arguments, the image runs `ppimpl serve`, which serves `nvidia/GLM-5.3-Flash-NVFP4` with HBM only. If the
weights are not in the mounted Hugging Face cache, they are downloaded first (~191 GB). Options after the image
name go to `ppimpl serve`: `docker run ... ppimpl --dram-gb 50` runs `ppimpl serve --dram-gb 50`. With compose, put
them in `PPIMPL_ARGS` (see [Docker](#docker)).

The image ships its JIT kernels, so a fresh container is ready in about 60 seconds without compiling or downloading
anything (see [Docker](#docker) for the cold-start numbers, GPU selection, tiers and logs).

### From source

You need:

- 4 × B200 and NVIDIA driver ≥ 580.
- The CUDA 13 toolkit, with `nvcc` at `$CUDA_HOME` (default `/usr/local/cuda`). It builds causal-conv1d and
  FlashInfer's JIT kernels.
- `git`, `curl`, `libnuma1` (the DRAM tier), `liburing2` (only for the optional io_uring SSD backend),
  `/usr/bin/ninja` (Ubuntu `ninja-build`, for FlashInfer's JIT), `g++` and [uv](https://docs.astral.sh/uv/).
- A normal user account: `make_env.sh` refuses to run as root.
- A `TMPDIR` of at most 64 characters (see [Troubleshooting](#troubleshooting)).

```bash
STEPS="venv vllm conv1d install jit check" env/make_env.sh   # .venv, third_party/vllm, the `ppimpl` command and
                                                            # FlashInfer's JIT kernels; ~20 min

# optional: fetch the weights ahead of the first start (otherwise `ppimpl serve` downloads them)
.venv/bin/hf download nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348

.venv/bin/ppimpl serve                                     # port 8000, HBM only
```

`env/make_env.sh` builds the pinned environment from public sources and checks it with `env/check_env.py`. Its
versions are in [ENV.md](ENV.md). The weights go into the Hugging Face cache (`HF_HOME`, default
`~/.cache/huggingface`). `ppimpl serve` also takes a local checkpoint directory: `ppimpl serve /path/to/checkpoint`.

### Send requests

`GET /health` returns 200 once all four stages are ready, and 503 before that:

```bash
curl -i localhost:8000/health
```

```bash
curl localhost:8000/v1/models
```
```json
{"object":"list","data":[{"id":"nvidia/GLM-5.3-Flash-NVFP4","object":"model","created":1792395419,"owned_by":"ppimpl",
 "root":"/root/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348",
 "parent":null,"max_model_len":1048576}]}
```

```bash
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "nvidia/GLM-5.3-Flash-NVFP4", "prompt": "The capital of France is", "max_tokens": 1}'
```
```json
{"id":"cmpl-04c74e7cc0254e2995b45eca422e895b","created":1792395424,"model":"nvidia/GLM-5.3-Flash-NVFP4",
 "object":"text_completion","choices":[{"index":0,"text":" Paris","logprobs":null,"finish_reason":"length"}],
 "usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6,"prompt_tokens_details":{"cached_tokens":0}}}
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 2+2? Answer with one digit."}],
       "chat_template_kwargs": {"reasoning_effort": "low"}}'
```
```json
{"id":"chatcmpl-d7041c92b679491ab1b02c0df76fb580","created":1792395425,"model":"nvidia/GLM-5.3-Flash-NVFP4",
 "object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"logprobs":null,
 "finish_reason":"length"}],"usage":{"prompt_tokens":24,"completion_tokens":1,"total_tokens":25,
 "prompt_tokens_details":{"cached_tokens":0}}}
```

These responses come from the image's GPU smoke test (`benchmarks/jobs/docker_smoke.sbatch`), which serves with
`--served-model-name glm53-flash`. Here they are shown with the default model name. The `openai` Python client
works too: point `base_url` at `http://localhost:8000/v1`.

## Configuration

### Serve options

`ppimpl serve [<model>] [options]` (`ppimpl serve --help`). By default it runs the design setting of the M6
evaluation ("st10"):

- the scheduler of step `good`;
- the exact prefix cache;
- with a tier: exclusive DRAM writes (DRAM holds only what HBM evicted) and a 10% HBM loading area.

| Option | Default | Meaning |
|---|---|---|
| `<model>` | `nvidia/GLM-5.3-Flash-NVFP4` | Hugging Face repo id or local checkpoint directory; only this model is accepted |
| `--revision` | `da920bb0b9` (for the default model) | Hugging Face revision |
| `--host`, `--port` | `0.0.0.0`, `8000` | HTTP address. The engine also uses the next 11 ports (`port+1` … `port+11`) on 127.0.0.1 |
| `--served-model-name` | `<model>` as given | the model name of the API |
| `--max-model-len` | 1,048,576 | longest prompt + 1; also capped by what the KV pool holds |
| `--dram-gb` | no DRAM tier | DRAM tier per GPU, in GB (pinned host memory) |
| `--ssd-gb` | no SSD tier | SSD tier per GPU, in GB; must equal the shard size of the shard set |
| `--ssd-dir` | `$PPIMPL_SSD_DIR` | directory of the SSD shard sets |
| `--ssd-backend` | `aio` | the SSD tier's file I/O: `aio` (Linux AIO) or `uring` (io_uring; in Docker it needs a seccomp profile, see [Docker](#docker)) |
| `--run-dir` | `./run` | logs: `server.log`, `worker*.log`, `server_config.yaml`, `microbatches.jsonl`, … |
| `--config YAML` | — | any `RunConfig` field (below), applied on top of the design setting; command-line options win. Name the model with it (see below) |

**Any engine field through `--config`.** The YAML may set any field of `RunConfig` or of its nested
`TierConfig` / `SchedulerConfig` ([ppimpl/config.py](ppimpl/config.py)). Unknown fields are refused. For example,
to cap HBM's KV pool:

```yaml
# hbm20.yaml: HBM pool capped at 12,220,800 tokens (~20 GB of KV per GPU)
pool_tokens: 12220800
```

```bash
ppimpl serve nvidia/GLM-5.3-Flash-NVFP4 --config hbm20.yaml --dram-gb 20 --ssd-gb 128 --ssd-dir /nvme/ppimpl
```

With `--config`, name the model. `ppimpl serve --config run.yaml` without a model takes the whole run config from
the YAML instead of the design setting, and refuses `--revision` and the tier options. The benchmark runners run it
that way.

### Memory tiers

```bash
ppimpl serve --dram-gb 50                                            # HBM + DRAM
ppimpl serve --dram-gb 50 --ssd-gb 128 --ssd-dir /nvme/ppimpl       # HBM + DRAM + SSD

# the same with Docker compose; the host's shard directory is mounted at /ssd
PPIMPL_ARGS="--dram-gb 50 --ssd-gb 128 --ssd-dir /ssd" PPIMPL_SSD_DIR=/nvme/ppimpl \
  docker compose -f docker/compose.yaml up
```

**DRAM.** `--dram-gb G` pins G GB of host memory per GPU on that GPU's NUMA node: 4 × G GB in total, plus room for
the server's own processes.

**SSD.** The SSD tier reads and writes preallocated shard files with O_DIRECT and never creates or resizes them. Its
file I/O is Linux AIO by default; `--ssd-backend uring` uses io_uring instead (equal in an A/B on the M6 workload,
[docs/results/ssd_backend_ab.md](docs/results/ssd_backend_ab.md)). Make a shard set once, on a local NVMe file system:

```bash
python -m ppimpl.kvpool.shards create --dir /nvme/ppimpl --gb 128   # set0/shard{0..3}.bin, 128 GB each
python -m ppimpl.kvpool.shards list --dir /nvme/ppimpl              # sizes, and whether a server holds each set
```

- `--gb` must equal the server's `--ssd-gb`. A set with a missing shard or one of another size stops the start.
  `--force` rewrites shards at a new size.
- A server locks the first free set for as long as it runs. To run two servers on one 8-GPU host (GPUs 0–3 and 4–7),
  make two sets with `--sets 2`. Give the second server other GPUs and a port block that does not overlap:

  ```bash
  CUDA_VISIBLE_DEVICES=4,5,6,7 ppimpl serve --port 8100 --dram-gb 50 --ssd-gb 128 --ssd-dir /nvme/ppimpl
  ```

- Shard contents are never trusted across runs: every run starts with an empty index.

**Sizing.** Measured on the M6 workload (v3 session trace):

| Setting (per GPU) | HBM KV | DRAM | SSD | N = 400, minutes 10–20 |
|---|---|---|---|---|
| full | full pool, 61.26M tokens (~100 GB) | 50 GB | 128 GB | 40.51 r/s, 94% of reused contexts from HBM |
| tiers | 12.2M tokens (~20 GB) | 20 GB | 128 GB | 37.66 r/s, 47% from SSD |
| ssdheavy | 7.6M tokens (12.5 GB) | 12.5 GB | 128 GB | 36.71 r/s, 77% from SSD |
| HBM only | full pool | — | — | 12.97 r/s: contexts are lost and recomputed |

Source: [docs/results/eval_m6.md](docs/results/eval_m6.md).

- **Without SSD, HBM + DRAM must hold the peak retained context.** On the v3 trace at N = 400 the peak is about
  65M tokens. In a simulator scan, every N = 400 configuration with HBM + DRAM ≤ 57M tokens collapsed within
  20 minutes: hits fall and the GPUs spend their time recomputing. 62.3M and 65.3M held. Two real runs agree:
  25 GB HBM + 50 GB DRAM (48.6M tokens) collapsed, and 25 GB HBM + 75 GB DRAM (65.3M tokens) held. The eval's 20/20
  and 12.5/12.5 GB sizes collapse without SSD by minute 4–8. See
  [docs/results/v3_dram_only_collapse.md](docs/results/v3_dram_only_collapse.md).
- **With an SSD tier, 8-hour runs are stable in the simulator.** The runs use DRAM 150 GB per GPU and an 8 TB SSD
  tier, at 1,438–2,500 concurrent sessions, and model a decode that runs elsewhere. The hit rate stays at
  99.2–99.4%. The SSD holds at most 56% of its capacity after 8 h and never evicts. With DRAM 150 GB and no SSD,
  every configuration of the deployed rules collapses to under 1 r/s. See
  [docs/results/v3_8h_steady.md](docs/results/v3_8h_steady.md). The real engine has run 20-minute workloads, not 8 hours.

## API

| Endpoint | Body |
|---|---|
| `GET /health` | 200 when ready, 503 while starting or after a failure |
| `GET /v1/models` | the one served model, with `max_model_len` |
| `POST /v1/completions` | `prompt`: a string or a list of token ids (no batch of prompts) |
| `POST /v1/chat/completions` | `messages`, rendered by the checkpoint's chat template; `tools` and `chat_template_kwargs` are passed to the template |
| `POST /inference/v1/generate` | req-frontend's token endpoint: `{"token_ids": [...], "sampling_params": {...}, "stream": ...}` |

Limits:

- **One output token**, greedy:
  - `max_tokens` (or `max_completion_tokens`) must be 1 or absent; any other value is a 400.
  - `temperature`, `top_p` and the other sampling fields are accepted and ignored.
  - `finish_reason` is `stop` when the token is an end-of-sequence token, else `length`.
- **Refused with 400:** `n` > 1, `echo`, `logprobs`, an empty prompt, a token id outside the vocabulary, and a
  prompt longer than `max_model_len` - 1.
- **Wrong model name:** a `model` other than the served name is a 404. Leaving `model` out is allowed.
- **Streaming:** `"stream": true` sends one chunk, then a usage chunk if `stream_options.include_usage`, then
  `data: [DONE]`.
- **Usage:** `usage.prompt_tokens_details.cached_tokens` is the number of prompt tokens served from the prefix cache
  or a tier.
- **Errors** use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

## Docker

`docker/Dockerfile` builds the serving environment of [ENV.md](ENV.md) on `nvidia/cuda:13.0.2-devel-ubuntu24.04`,
running `env/make_env.sh` with `STEPS="venv vllm conv1d install jit"`, and installs ppimpl in `/opt/ppimpl`. The
benchmark-only parts (ServingStudioSim, req-frontend, enwik9) are left out. `--build-arg MAX_JOBS=N` sets the number
of parallel nvcc jobs (default 16). The entry point (`docker/entrypoint`) works as follows:

| Command | Runs |
|---|---|
| `<image>` (no arguments), or `<image> --option ...` | `ppimpl serve [options]`: `nvidia/GLM-5.3-Flash-NVFP4`, HBM only |
| `<image> serve [<model>] [options]` | `ppimpl serve` |
| `<image> check-env` | `env/check_env.py`; needs no GPU |
| `<image> <command>` | the command as is, e.g. `bash` or `python -m ppimpl.kvpool.shards ...` |

**Compose.** `docker compose -f docker/compose.yaml up` starts the server with every setting it needs: all GPUs,
`ipc: host`, port 8000, the host's Hugging Face cache, the `ppimpl-cache` volume and `cap_add: SYS_NICE` (for the
tiers, below). It reads these environment variables, all optional:

| Variable | Default | What |
|---|---|---|
| `PPIMPL_ARGS` | none | options of `ppimpl serve`, e.g. `"--dram-gb 50 --ssd-gb 128 --ssd-dir /ssd"` |
| `PPIMPL_IMAGE` | `ppimpl` | the image |
| `PPIMPL_PORT` | `8000` | the host port |
| `HF_CACHE` | `~/.cache/huggingface` | the host's Hugging Face cache (the weights) |
| `PPIMPL_SSD_DIR` | a volume | host directory of the SSD shard sets (fast NVMe), mounted at `/ssd` |

**JIT kernels in the image.** The image carries the kernels a start would otherwise compile, in
`/opt/ppimpl/jit-seed`. At each start the entry point copies whatever `/root/.cache` lacks from there, so an empty or
newly mounted cache volume is seeded.

- FlashInfer's JIT modules and the trtllm-gen cubins of the kernel families serving uses. `env/bake_jit.py` (step
  `jit`) builds them without a GPU.
- The GPU seed: the Triton, TileLang, DeepGEMM, CuTe-DSL and inductor kernels that one start on a B200 compiles. It is
  `docker/jit-seed/gpu.tar`, which is not in git and is optional: without it the image builds the same, and only the
  first start of a fresh container is slower. To make it, build the `base` target and run the GPU smoke test with
  `SEED_OUT` on 4 B200:

  ```bash
  docker build -f docker/Dockerfile --target base -t ppimpl:base .
  IMAGE=ppimpl:base CACHE=fresh OFFLINE=1 SEED_OUT=$PWD/docker/jit-seed/gpu.tar \
    sbatch -p <partition> --time=00:30:00 -o runs/logs/%j.out benchmarks/jobs/docker_smoke.sbatch
  ```

  Make it again when the environment or a kernel changes. A stale seed only costs compile time at the first start.

First start of a fresh container (empty cache volume), until `/health` returns 200:

| Image | Ready after |
|---|---|
| no JIT kernels baked | 932 s (FlashInfer compiles its modules and downloads cubins) |
| FlashInfer modules and cubins baked, no GPU seed | 287 s |
| with the GPU seed | ~60 s, also with `--network none` |
| any image, later starts with the same cache volume | ~50 s |

**Options.**

- **GPUs.** `--gpus all` needs exactly 4 B200 in one NVLink domain. On a larger host, name four:
  `--gpus '"device=0,1,2,3"'` (compose: `device_ids` in `docker/compose.yaml`).
- **`--ipc=host`.** The stages pass activations over CUDA IPC, and the intake processes share prompts through
  `/dev/shm`.
- **Caches.** The `ppimpl-cache` volume at `/root/.cache` keeps the JIT and autotune caches across containers. The
  host's Hugging Face cache is mounted over `/root/.cache/huggingface`.
- **Tiers.** Add `--cap-add SYS_NICE` (compose sets it). The DRAM tier and the staging area place their pages on the
  GPU's NUMA node with `mbind`, which Docker's default seccomp profile allows only with `CAP_SYS_NICE`. Nothing else is
  needed: the default SSD backend, Linux AIO, runs under Docker's default seccomp profile, and neither it nor the
  pinned DRAM tier counts against the memlock limit. `ppimpl serve` checks this at start and stops with the fix.
  - Mount the shard directory and create a set once:

  ```bash
  docker run --rm -v /nvme/ppimpl:/ssd ppimpl python -m ppimpl.kvpool.shards create --dir /ssd --gb 128
  docker run --gpus all --ipc=host -p 8000:8000 --cap-add SYS_NICE \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ppimpl-cache:/root/.cache -v /nvme/ppimpl:/ssd \
    ppimpl --dram-gb 50 --ssd-gb 128 --ssd-dir /ssd
  ```

  - **io_uring (optional).** `--ssd-backend uring` needs io_uring, which Docker's default seccomp profile denies since
    Docker 25, and no memlock limit, since io_uring counts the host memory it registers as locked. Run with
    `--security-opt seccomp=docker/seccomp.json --ulimit memlock=-1` (Docker's default profile plus the three
    io_uring syscalls; `docker/make_seccomp.py` writes it, and the image has a copy:
    `docker run --rm ppimpl cat /opt/ppimpl/docker/seccomp.json`), or with compose:
    `docker compose -f docker/compose.yaml -f docker/compose.uring.yaml up`.

  The GPU smoke test (`benchmarks/jobs/docker_smoke.sbatch` with `TIERS=1`) starts compose with a 20 GB DRAM and an 8 GB
  SSD tier and checks the answers. The tiers in a container are not measured under load yet.
- **Logs** go to `/workspace/run`; mount `-v $PWD/run:/workspace/run` to keep them. `TMPDIR` is `/tmp`; the entry
  point refuses one over 64 characters (see [Troubleshooting](#troubleshooting)).

## Performance

| Document | What it measures |
|---|---|
| [docs/results/eval_m6.md](docs/results/eval_m6.md) | **The evaluation.** ppimpl vs the simulator vs vLLM DP4/EP4 and TP4 at N = 100/300/400, every tier setting, and ablations |
| [docs/results/vllm_baselines.md](docs/results/vllm_baselines.md) | The vLLM baselines: flags, router, per-run rows |
| [docs/results/v3_dram_only_collapse.md](docs/results/v3_dram_only_collapse.md) | HBM + DRAM without SSD: when the prefix cache collapses (simulator scan + 2 real runs) |
| [docs/results/v3_8h_steady.md](docs/results/v3_8h_steady.md) | 8-hour steady state with and without SSD (simulator) |
| [docs/results/kda_ckpt_k_recompute_v3.md](docs/results/kda_ckpt_k_recompute_v3.md) | Extra prefill from keeping 1, 2 or 4 KDA states per context (simulator) |
| [docs/results/ssd_backend_ab.md](docs/results/ssd_backend_ab.md) | The SSD tier's backends: Linux AIO vs io_uring (disk only, bring-up, sessions) |
| [docs/results/setup.md](docs/results/setup.md) | The node and software the measurements ran on |

The bring-up measurements (single-request iteration times, prefix cache, scheduler, kernel timings and identity
against the profiler rows, correctness against vLLM TP4, known gaps from the design targets) are on the `dev`
branch under `research/measured/` (see [Branches](#branches)).

## Architecture

```
 clients ──HTTP :8000──▶ 16 intake processes ──zmq──▶ controller
                                                      scheduler · prefix cache · KV pool · tier index
                                                          │ microbatch plans        ▲ reports
                                                          ▼                         │
                         GPU 0: stage 0 ─▶ GPU 1: stage 1 ─▶ GPU 2: stage 2 ─▶ GPU 3: stage 3
                         (activations over CUDA IPC; each GPU stores and moves 1/4 of every context
                          to and from its own DRAM and SSD tiers)
```

- **Controller.** It owns scheduling, the prefix cache, the KV page pool and the tier index, and sends each stage
  its microbatch plans.
- **Stage workers.** Each one builds its layers of the model from the vLLM fork's `Glm5Next` modules and runs them
  inside vLLM's forward context. Block tables and state slots come from the controller.
  [ppimpl/model/pp_patch.py](ppimpl/model/pp_patch.py) lists every runtime override of the fork, with the reason
  for each and the test that checks it.
- **Tiers** live in [ppimpl/kvpool/](ppimpl/kvpool/). DRAM reads and writes are copy-engine batches. SSD I/O uses
  Linux AIO (or io_uring, with the host arenas as fixed buffers), reads ahead of writes.

Design documents:

- [docs/design/architecture.md](docs/design/architecture.md): the reuse boundary, process model, scheduling and logs.
- [docs/design/tiers.md](docs/design/tiers.md): the DRAM and SSD tiers.
- [ppimpl/runtime/README.md](ppimpl/runtime/README.md): the runtime and transport.
- [docs/sim_to_real/](docs/sim_to_real/index.html): from simulation to hardware, how the simulator's design became this
  engine ([English](docs/sim_to_real/en.html)).
- The design follows the ServingStudio sim-to-real handoff `glm53_flash_pp4_b200_v2`
  (ServingStudioSim `doc/handoff/glm53_flash_pp4_b200_v2/` at `5153041a`). Its milestone files (`1_contract.md` …
  `7_offloading.md`) are the specification the code refers to.

## Project layout

| Path | Content |
|---|---|
| `ppimpl/config.py` | `RunConfig`, `TierConfig`, `SchedulerConfig`: every engine setting |
| `ppimpl/serve/` | the CLI (`launch.py`), HTTP intake, OpenAI endpoints |
| `ppimpl/runtime/` | controller, stage workers, transport, dispatch |
| `ppimpl/engine/` | scheduler, KV pool and prefix cache |
| `ppimpl/kvpool/` | DRAM and SSD tiers, the SSD shard files |
| `ppimpl/model/` | one pipeline stage on the vLLM fork, and the patches to it |
| `benchmarks/` | the runners that reproduce the results (`python -m benchmarks.<name>`); `benchmarks/jobs/` holds the job scripts |
| `env/` | `make_env.sh`, `check_env.py`, `requirements.lock` |
| `docker/` | Dockerfile, compose files, entry point, the io_uring seccomp profile |
| `tests/` | CPU tests, plus GPU tests (`-m gpu`) |
| `docs/` | design ([docs/design/](docs/design/)), results ([docs/results/](docs/results/)), the sim-to-real story |

## Development

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q          # CPU tests: scheduler, KV, tiers, server on a fake model
```

- The CPU tests run the server end to end on a fake model. Hiding the GPUs keeps them off any GPU.
- `test_req_frontend_cpu` (in `tests/test_e2e_cpu.py`) drives the server with req-frontend's `session_runner`.
  It is skipped until that binary exists in `build/req-frontend/release/`. The first bench run builds it, which
  needs the bench environment (`STEPS="client" env/make_env.sh`) and `cargo`.
- The GPU tests need 4 GPUs: `.venv/bin/python -m pytest -q -m gpu`, through Slurm where there is one
  (`benchmarks/jobs/gtest.sbatch`, usage at its top).

## Reproduce

The evaluation needs the serving environment and the bench parts:

```bash
env/make_env.sh        # all steps: serving env + ServingStudioSim + req-frontend + enwik9 + check

# the handoff (traces, targets, microbatch fit) at its frozen commit; make_env.sh does not place it
mkdir -p third_party/handoff_src
git -C third_party/ServingStudioSim archive 5153041a doc/handoff/glm53_flash_pp4_b200_v2 | tar -x -C third_party/handoff_src
ln -s handoff_src/doc/handoff/glm53_flash_pp4_b200_v2 third_party/handoff
```

This also needs `unzip` and `cargo`: req-frontend's `session_runner` is built on first use. About 200 GB of disk
holds the weights. Slurm is optional, since the job scripts also run with `bash`. [docs/results/setup.md](docs/results/setup.md) records the
node.

**Settings.** Every path has a default; set a variable only to point elsewhere.

| Variable | Default | What |
|---|---|---|
| `PPIMPL_CHECKPOINT` | the snapshot above under `$HF_HOME/hub` | the checkpoint directory |
| `PPIMPL_HANDOFF` | `third_party/handoff` | the handoff directory: `doc/handoff/glm53_flash_pp4_b200_v2` of ServingStudioSim at `5153041a` (traces, targets, microbatch fit) |
| `PPIMPL_SIM` | `third_party/ServingStudioSim` | ServingStudioSim at `820adda0` |
| `PPIMPL_ENWIK9` | `third_party/enwik9` | the prompt corpus |
| `PPIMPL_REQ_FRONTEND` | `third_party/req-frontend` | req-frontend at `8723d90` or later. Its `session_runner` is built into `build/req-frontend/release/` and rebuilt when the commit changes |
| `PPIMPL_PYTHON` | `.venv/bin/python` | the interpreter of the job scripts |
| `PPIMPL_TMP` | `$TMPDIR`, else `/tmp` | parent of each job's short `TMPDIR` |
| `PPIMPL_SSD_DIR` | `<main tree>/tmp/ssd_tier` (job scripts) | the SSD shard sets |
| `PPIMPL_FI_TACTICS` | `ppimpl/data/flashinfer_tactics_b200.json` | the tuned FlashInfer tactics (`benchmarks/tune_tactics.py`) |

**The M6 evaluation.** One run per job (`benchmarks/jobs/m6.sbatch`, usage at its top). The trace is the TraceLab v3
session trace `sessions_2ps_v3.csv`, which is not in this repository. `m6.sbatch` reads its targets from
`third_party/handoff_v2_realistic` (the v2 handoff with regenerated targets) unless `PPIMPL_HANDOFF` is set:

```bash
python -m ppimpl.kvpool.shards create --dir $PPIMPL_SSD_DIR --gb 128       # once
sbatch -p <partition> --time=00:30:00 --mem=400G -o runs/m6/slurm/%j.out benchmarks/jobs/m6.sbatch \
  sched_good_n400 --trace <sessions_2ps_v3.csv> --dram-gb 50 --ssd-gb 128 \
  --tier dram_writes=hbm-evictions-exclusive --tier loading_area=0.10 --run-name st10_full_n400
uv pip install --python .venv/bin/python pyarrow                            # once: eval_table.py reads parquet
.venv/bin/python benchmarks/eval_table.py real:full=runs/m6/st10_full_n400
```

The vLLM baselines use `benchmarks/jobs/vllm_dpep.sbatch`, with its usage at the top. Outputs go to `runs/`. The
bring-up milestones (kernels, layers, single requests, prefix cache, scheduler, correctness against vLLM TP4) have
their harnesses and job scripts on the `dev` branch under `research/bench/`.

## Troubleshooting

**The first start takes minutes.** Without the baked kernels (a source install without the `jit` step, or an image
without the GPU seed), the first start compiles JIT kernels into `~/.cache` and FlashInfer downloads trtllm-gen cubins
from NVIDIA, which needs network access: up to 16 minutes. Later starts reuse the cache: about 50 s in Docker and 65 s
natively. In Docker, mount a volume at `/root/.cache` to keep what a start adds. On a host, use `/usr/bin/ninja`:
processes that share the cache but use another ninja rebuild every object, and `check_env.py` warns about this.

**`the KV tiers place host memory with mbind, which failed here: EPERM`.** In Docker, add `--cap-add SYS_NICE`
(compose sets it).

**The SSD backend fails to start.** `ppimpl serve` stops before loading anything:
- Linux AIO (default): `io_setup` EAGAIN means `sysctl fs.aio-max-nr` is used up; raise it.
- io_uring (`--ssd-backend uring`): Docker 25 and later deny it by default: add
  `--security-opt seccomp=docker/seccomp.json --ulimit memlock=-1` (or `docker/compose.uring.yaml`). On a host,
  `sysctl kernel.io_uring_disabled` must be 0, or 1 with your user in `kernel.io_uring_group`, and the memlock limit
  must cover the DRAM tier of all four stages.

**The DRAM tier fails to pin memory.** Check that the host has 4 × `--dram-gb` GB free, plus room for the server's
processes, and that no container or job memory limit is lower than that.

**`TMPDIR ... is over 64 characters`, or zmq "File name too long".** vLLM's zmq IPC path is `TMPDIR` plus a UUID and
must fit in 107 characters. Use a short `TMPDIR`, such as `/tmp`.

**`SSD shard set ...: shards ... the run needs 4 shards of N B`.** The shard size differs from `--ssd-gb`, or a shard
is missing. Recreate the set with the matching `--gb` (`--force` rewrites existing shards). **`every SSD shard set ...
is in use`**: another server holds every set. Create another set with `--sets K`.

**Address already in use.** The server uses `--port` and the next 11 ports. Pick a port block that is free for all 12.

**`unsupported model`.** Only `nvidia/GLM-5.3-Flash-NVFP4` is served; the check reads `config.json` before anything
else is downloaded.

**`max_tokens 16: this engine computes prefill and one output token`.** By design: see the note at the top.

## Acknowledgements

ppimpl builds on:

- [vLLM](https://github.com/vllm-project/vllm), through the fork
  [serendipity-zk/vllm](https://github.com/serendipity-zk/vllm) (branch `servingstudio-alignment`): model layers,
  weight loading, attention metadata, custom ops.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): MoE, FP4 GEMM, sparse MLA and KDA kernels. Its
  `_split_heads` search is ported to C++ in `ppimpl/native/kda_split_heads.cpp`.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and [TileLang](https://github.com/tile-ai/tilelang).
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d).
- [req-frontend (request-factory)](https://github.com/uw-syfi/request-factory): the benchmark client.
- [ServingStudioSim](https://github.com/SyFI-ServingStudio/ServingStudioSim): the simulator that set the design,
  its targets and the kernel profiler. It is under its own license.

## License

Apache License 2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) lists the files derived from ServingStudioSim, which its
copyright holders make available under Apache-2.0 here, and the FlashInfer and moby material. ServingStudioSim itself
and the other dependencies keep their own licenses.
