# AgentX evaluation runbook

How to run SemiAnalysis AgentX (InferenceX v3, prefill only) against SSEngine, vLLM and SGLang on GLM-5.3-Flash,
and compare them. Written for whoever (or whichever agent) runs the evaluation on their own machines. Launching is
left to you: on a Slurm cluster wrap each run in a job (`benchmarks/jobs/agentx.sbatch` is ours), on a plain Docker
host run them one after another. This page says what one run is, what to run, and what goes wrong.

## What one run is

`python -m benchmarks.agentx run` (module docstring in `benchmarks/agentx.py` has the details) does, on the GPUs it
can see:

1. starts one engine at one parallel layout (`--engine ssengine|vllm|sglang`, `--layout pp4|tp4|dp4-ep|...`),
   plus a cache-aware router in front of DP layouts;
2. waits for it, then runs the AgentX client (`agentx profile`, agentx-harness 1.0.7) at `--concurrency N` live
   session trees: warm-up, then `--duration` seconds (default 1800; the scenario requires >= 900) of measurement;
3. stops everything and writes a run directory (`runs/agentx/<engine>_<layout>_c<N>_<job>/`): `run.json` (exact
   commands, versions, image), `server.log`, `gpu_samples.csv`, `aiperf/` (one record per request) and
   `summary.json` (req/s, TTFT p50/p90/p99, prompt and computed tokens per GPU, cache hit).

Every request asks for one output token, so the runs compare prefill: TTFT = queueing + prefill.

Print the commands of a run without running anything: `python -m benchmarks.agentx argv <same options>`.

## Setup

Hardware: 4 GPUs per run (B200; GB200 should work for vLLM/SGLang, see ARM below), in one NVLink domain, and for
SSEngine's tiers ~4 x DRAM-GB of free host memory plus a local NVMe file system.

1. Weights: `nvidia/GLM-5.3-Flash-NVFP4` at revision `da920bb0b9f4a06727223a349e55468e38352348` (~191 GB). Pass
   the snapshot directory as `--checkpoint`; it must be visible inside the containers at the same path.
2. AgentX client: `STEPS=agentx env/make_env.sh` (a venv at `third_party/agentx-venv`). The first run downloads the
   dataset (`semianalysis_cc_traces_weka_062126_256k`) into `~/.cache/aiperf`.
3. Engines:

   | engine | how | x86_64 | aarch64 |
   |---|---|---|---|
   | SSEngine | from source (`--python .venv/bin/python`, README "From source") or its image (`--ssengine-image`) | `docker build -f docker/Dockerfile` | same Dockerfile built natively on aarch64 |
   | vLLM | upstream image: `--vllm-image docker.io/vllm/vllm-openai:v0.33.0` | yes | same tag (multi-arch) |
   | SGLang | image `docker.io/lmsysorg/sglang:v0.5.23-cu130-runtime` (default; `SSENGINE_SGLANG_IMAGE` overrides) | yes | same tag (multi-arch) |

   Without `--vllm-image` the harness runs our vLLM fork from `.venv` (what SSEngine's kernels come from); for a
   fair comparison use the upstream release.
4. Containers: on a plain Docker host set `AGENTX_DOCKER_ARGS="--gpus all -v /path/to/hf:/path/to/hf"` (every
   container gets it; add `-v <ssd dir>:<ssd dir>` for SSEngine's SSD tier from its image). On cayenne leave it empty:
   its docker wrapper adds the job's GPUs and `/scratch/hf`.
   Each engine image gets a persistent `/root/.cache` (JIT and autotune results) under `AGENTX_CACHE_DIR` (default
   `~/.cache/agentx-engines`); without it SGLang pp4 re-tunes its MoE kernels for ~40 min on every run.
5. SSEngine tiers: make the SSD shard files once (`python -m ssengine.kvpool.shards create --dir <nvme dir> --gb 128`;
   `--sets 2` to run two servers at once), then pass `--server-arg=--dram-gb --server-arg=100 --server-arg=--ssd-gb
   --server-arg=128 --server-arg=--ssd-dir --server-arg=<nvme dir>`.

### ARM (GB200 / GB300)

- vLLM and SGLang: the same image tags carry arm64 builds. Not run by us on ARM.
- SSEngine: `docker/Dockerfile` builds natively on aarch64 (`make_env.sh` picks the aarch64 vLLM
  wheel). Built and smoke-tested on a GB10 (sm_121), which cannot hold the model. Its serving kernels are built for
  sm_100a (DeepSelect, trtllm-gen MoE/FMHA cubins, the baked FlashInfer JIT): GB200 (sm_100) should run them; GB300
  (sm_103) is untested and likely needs those rebuilt. Run `docker run --gpus all <image> check-env` and a short
  smoke (`--smoke`) first.

## What to run

The matrix (4 GPUs per run):

| engine | layouts | notes |
|---|---|---|
| SSEngine | pp4 | with tiers: `--dram-gb 100 --ssd-gb 128` |
| vLLM v0.33.0 | tp4, dp4-ep (router) | `--mamba-cache-mode align` is set by the harness |
| SGLang v0.5.23 | tp4, dp4-ep (router), pp4 | |

Concurrency: 8 (smoke first), 64, 128, 256, 512, then higher for SSEngine until it saturates. One run takes
model load (SSEngine ~2 min, vLLM ~6 min, SGLang ~12 min) + warm-up + measurement.

Warm-up: the scenario sends one full-history primer per live session, then 10 more requests per lane, and starts
measuring only after all of them return (no time cap). At c512 that is ~5,600 requests, hours for an engine that
cannot keep the contexts. For c >= 256 pass `--warmup-per-lane 1` for every engine (the run is then stamped
`submission_valid: false`; compare only runs with the same warm-up, `compare` groups by it).

## Compare

```bash
python -m benchmarks.agentx compare runs/agentx/*/      # one table per (concurrency, warm-up), best req/s first
python -m benchmarks.agentx table runs/agentx/*/        # one row per run
```

Report per run: req/s, cache hit, TTFT p50/p90/p99, prompt and computed tokens/s per GPU, failed requests, and the
`build` column (image or commit). Keep the run directories.

## What goes wrong

- **One failed warm-up request aborts the whole run** (AgentX: `warmup_failure`). Check `client.log` for the first
  error, then `server.log`.
- **Connection reset by peer with a clean server log (vLLM).** vLLM closes idle keep-alive connections after 5 s,
  AIPerf reuses pooled ones for up to 300 s. The harness sets `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600` for vLLM.
- **A worker killed with no traceback.** On a shared cluster, check the host's GPU-idle killer (on cayenne
  `journalctl | grep idle_kill`): per-process NVML utilization can read low while the GPU is busy.
- **Weights not found in a container.** The container sees only mounted paths; pass the checkpoint by a path that is
  mounted at the same place (`AGENTX_DOCKER_ARGS -v`).
- **SSD tier start fails.** Shards are created once with `--gb` equal to `--ssd-gb`; a server locks one set; two
  servers at once need `--sets 2`.
- **Long TTFT tails under overload are expected.** SSEngine schedules shortest-remaining-work first and forces any
  request that waited 60 s; past saturation the forced requests queue among themselves (tails of ~2x 60 s).

## Reference numbers (cayenne, 4x B200, 2026-11-07)

Our runs, for a sanity check of a new setup (vLLM here is our fork, not v0.33.0):

| c / warm-up | engine | req/s | cache hit | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|---|
| 128 / 10 | SSEngine pp4 (DRAM 100 GB, SSD 128 GB per GPU) | 8.66 | 0.932 | 0.30 / 0.58 / 2.41 |
| 128 / 10 | SGLang dp4-ep | 5.99 | 0.841 | 6.9 / 29.9 / 99.8 |
| 128 / 10 | vLLM fork tp4 | 3.92 | 0.813 | 32.0 / 64.1 / 151 |
| 512 / 1 | SSEngine pp4 | 17.3 | 0.922 | 2.6 / 61 / 112 |
| 512 / 1 | SGLang dp4-ep | 1.73 | 0.379 | 109 / 478 / 558 |
| 512 / 1 | vLLM fork tp4 | 0.87 | 0.144 | 241 / 496 / 614 |
