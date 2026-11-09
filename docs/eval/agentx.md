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
3. stops everything and writes a run directory (`runs/agentx/<engine>_<layout>[_kv<N>][_ssd<M>]_c<N>_<job>/`;
   `<job>` is the Slurm job id, outside Slurm `local-<date>-<time>`; `--run-name` overrides): `run.json` (exact
   commands, versions, image), `server.log`, `client.log`, `gpu_samples.csv`, `aiperf/` (one record per request) and
   `summary.json` (req/s, TTFT p50/p90/p99, prompt and computed tokens per GPU, cache hit).

Before it creates the run directory it checks what the run needs (the AgentX client, vllm-router for vLLM DP layouts,
the `--ssd-dir` directory, free ports from `--port`, default 20060 outside Slurm) and stops with the fix if one is
missing.

Every request asks for one output token, so the runs compare prefill: TTFT = queueing + prefill.

Print the commands of a run without running anything: `python -m benchmarks.agentx argv <same options>`. They are
the ones `run` executes, `--smoke` defaults included; only the run directory and (without `--checkpoint`) the
checkpoint are placeholders.

Run the harness from the repo root with a Python that has numpy, msgspec and pyyaml: `.venv/bin/python` when SSEngine
is built from source, else `third_party/agentx-venv/bin/python` (setup step 2 installs them there). Every command
below is `<that python> -m benchmarks.agentx ...`.

## Setup

Hardware: 4 GPUs per run (B200; GB200 should work for vLLM/SGLang, see ARM below), in one NVLink domain, and for
SSEngine's tiers ~4 x DRAM-GB of free host memory plus a local NVMe file system.

1. Weights: `nvidia/GLM-5.3-Flash-NVFP4` at revision `da920bb0b9f4a06727223a349e55468e38352348` (~191 GB). Pass
   the snapshot directory as `--checkpoint`; it must be visible inside the containers at the same path.
2. AgentX client: `STEPS=agentx env/make_env.sh` (a venv at `third_party/agentx-venv`, which can also run the
   harness). The first run downloads the dataset (`semianalysis_cc_traces_weka_062126_256k`) into `~/.cache/aiperf`.
   For vLLM DP layouts also `STEPS=router env/make_env.sh` (vllm-router 0.1.15 in `tmp/vllm-router-venv`; PyPI has
   x86_64 and aarch64 wheels). The two steps need only `uv`; `STEPS="agentx router" env/make_env.sh` does both.
3. Engines:

   | engine | how | x86_64 | aarch64 |
   |---|---|---|---|
   | SSEngine | from source (`--python .venv/bin/python`, README "From source") or its image (`--ssengine-image`) | `docker build -f docker/Dockerfile` | same Dockerfile built natively on aarch64 |
   | vLLM | upstream image: `--vllm-image docker.io/vllm/vllm-openai:v0.33.0` | yes | same tag (multi-arch) |
   | SGLang | image `docker.io/lmsysorg/sglang:v0.5.23-cu130-runtime` (default; `SSENGINE_SGLANG_IMAGE` overrides) | yes | same tag (multi-arch) |

   Without `--vllm-image` the harness runs our vLLM fork from `.venv` (what SSEngine's kernels come from); for a
   fair comparison use the upstream release.
4. Containers: on a plain Docker host set
   `AGENTX_DOCKER_ARGS='--gpus "device=0,1,2,3" -v /path/to/hf:/path/to/hf'` (every container gets it). Name exactly
   the run's 4 GPUs: `--gpus all` on an 8-GPU host gives every container all 8. The harness mounts the run directory
   and the `--ssd-dir` directory itself. SSEngine from source takes its GPUs from `CUDA_VISIBLE_DEVICES=0,1,2,3`.
   `gpu_samples.csv` samples every GPU of the host; read the run's rows. Containers are named
   `agentx-<engine>-<job>`, so outside Slurm run one run at a time per host. On cayenne leave `AGENTX_DOCKER_ARGS`
   empty: its docker wrapper adds the job's GPUs and `/scratch/hf`.
   Each engine image gets a persistent `/root/.cache` (JIT and autotune results) under `AGENTX_CACHE_DIR` (default
   `~/.cache/agentx-engines`); without it SGLang pp4 re-tunes its MoE kernels for ~40 min on every run.
5. SSEngine tiers: make the SSD shard files once (`python -m ssengine.kvpool.shards create --dir <nvme dir> --gb 512`,
   4 x 512 GB; `--sets 2` to run two servers at once; from the image: `docker run -v <nvme dir>:<nvme dir> <image>
   python -m ssengine.kvpool.shards create ...`), then pass `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <nvme dir>`.
   SGLang and vLLM need no preparation, only a directory with room (see "Disk" below).

### Disk

Runs go one at a time, so one NVMe file system with ~2 TB free is enough if the SSEngine runs and the vLLM/SGLang SSD
runs do not overlap:

- SSEngine's shard set is 4 x 512 GB = 2 TB of files written in full when created, kept across runs and reused by
  every SSEngine run.
- SGLang's file backend writes up to 512 GB per rank (2 TB at 4 ranks) into `<ssd dir>/<run name>`; vLLM's fs tier
  writes there without a cap (67 GB in a 5-minute c8 smoke). The harness deletes that directory after each run.

So run all SSEngine tier runs first, then delete the shard set (`rm -r <nvme dir>/set0`) before the vLLM/SGLang SSD
runs, or keep both on a file system with ~4 TB free. Watch `df` during the vLLM SSD runs: we have not run one long
enough to fill a disk, so record the peak use, and treat a run that filled the disk as invalid.

### ARM (GB200)

- vLLM and SGLang: the same image tags carry arm64 builds. Not run by us on ARM.
- SSEngine: `docker/Dockerfile` builds natively on aarch64 (`make_env.sh` picks the aarch64 vLLM
  wheel). Built and smoke-tested on a GB10 (sm_121), which cannot hold the model. Its serving kernels are built for
  sm_100a (DeepSelect, trtllm-gen MoE/FMHA cubins, the baked FlashInfer JIT), which GB200 (sm_100) runs. Run
  `docker run --gpus all <image> check-env` and a short smoke (`--smoke`) first.

## What to run

The matrix (4 GPUs per run):

| engine | layouts | notes |
|---|---|---|
| SSEngine | pp4 | with tiers: `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <shard dir>` |
| SSEngine HBM only | pp4 | no tier flags: the counterpart of the vLLM/SGLang rows without offloading |
| SSEngine DRAM only | pp4 | `--host-kv-gb 100`: the counterpart of the vLLM/SGLang DRAM rows |
| vLLM v0.33.0 | tp4, dp4-ep (router) | `--mamba-cache-mode align` is set by the harness |
| vLLM v0.33.0 + CPU offloading | best layout of the row above | `--host-kv-gb 100` (native offloading) |
| SGLang v0.5.23 | tp4, dp4-ep (router), pp4 | |
| SGLang v0.5.23 + HiCache | best layout of the row above | `--host-kv-gb 100` (hierarchical cache) |
| vLLM / SGLang + DRAM + SSD | best layout | `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <nvme dir>` |

`--host-kv-gb N` gives every engine the same host DRAM for KV, N GB per GPU, in each engine's own flag: SSEngine
`--dram-gb N`, SGLang `--enable-hierarchical-cache --hicache-size N` (per rank), vLLM `--kv-offloading-size N x tp x pp
--kv-offloading-backend native` (one engine's total). `compare` shows it as `host KV GB/GPU`. Give such a job host
memory for 4 x N GB plus the engine (our SSEngine runs with N = 100 used ~490 GB; we request `--mem=600G`).

`--ssd-gb M --ssd-dir D` adds a disk tier below DRAM, M GB per GPU, on a local NVMe: SSEngine `--ssd-gb M --ssd-dir D`
(its shard set, made beforehand, see Setup); SGLang HiCache's `file` storage backend with `max_size` M GB per rank;
vLLM's tiering offloading with an `fs` secondary tier. vLLM v0.33.0's fs tier has no size cap, so for vLLM M only
switches it on (`compare` shows `no cap`). SGLang and vLLM write into `D/<run name>`, which the harness removes after
the run, so no run hits files a previous run wrote. Put D on the same NVMe as SSEngine's shards.

"Best layout": the layout of that engine with the highest req/s at c128 (the highest concurrency with the scenario's
own warm-up, so a valid submission); if two are within 5%, the one with the lower TTFT p90. Run the offloading rows
at the same concurrencies as the plain rows.

Concurrency: every configuration runs a smoke (c8 `--smoke`) and then c64, c128, c256, c512. All three SSEngine
configurations then go on with 768, 1024, 1536, ... (all with `--warmup-per-lane 1`) until they saturate: TTFT p90
reaches ~60 s (the 60 s forced-request rule, see below) or req/s grows < 5% over the previous step; always run at least
one step past c512 to show it. Our c512 run already had p90 61 s, but with a 128 GB SSD tier that was full; with
512 GB the knee may be later.

### Run list

Every run below, 4 GPUs each. Skip none: a missing cell leaves a comparison open. Order matters: phase 2 needs the
best layouts from phase 1, and on a ~2 TB disk the SSEngine shards are deleted between the phases (see "Disk").

| phase | configuration | flags | runs |
|---|---|---|---:|
| 1 | SSEngine pp4, DRAM + SSD | `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <shard dir>` | smoke, c64, c128, c256, c512, c768, ... until saturated (>= 6) |
| 1 | SSEngine pp4, DRAM only | `--host-kv-gb 100` | smoke, c64, c128, c256, c512, c768, ... until saturated (>= 6) |
| 1 | SSEngine pp4, HBM only | | smoke, c64, c128, c256, c512, c768, ... until saturated (>= 6) |
| 1 | vLLM v0.33.0 tp4 | `--vllm-image docker.io/vllm/vllm-openai:v0.33.0` | smoke, c64, c128, c256, c512 (5) |
| 1 | vLLM v0.33.0 dp4-ep | same | 5 |
| 1 | SGLang v0.5.23 tp4 | | 5 |
| 1 | SGLang v0.5.23 dp4-ep | | 5 |
| 1 | SGLang v0.5.23 pp4 | | 5 |
| | pick the best vLLM and the best SGLang layout (rule above); delete the shard set if the disk needs it | | |
| 2 | vLLM best layout, DRAM | `--host-kv-gb 100` | 5 |
| 2 | SGLang best layout, DRAM (HiCache) | `--host-kv-gb 100` | 5 |
| 2 | vLLM best layout, DRAM + SSD | `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <nvme dir>` | 5 |
| 2 | SGLang best layout, DRAM + SSD | `--host-kv-gb 100 --ssd-gb 512 --ssd-dir <nvme dir>` | 5 |

Total: 12 configurations, 12 smokes and at least 51 measured runs, so >= 63 runs. With the budget below that is about
57 hours of one 4-GPU node run after run (12 smokes x ~0.5 h + 51 x ~1 h); two nodes halve it. Re-run a failed run before
moving on; `compare` lists failed runs so none goes unnoticed.

### Budget

Per run: model load (SSEngine ~2 min, vLLM ~6 min, SGLang ~12 min; SGLang pp4 with an empty `AGENTX_CACHE_DIR`
~36 min of MoE tuning) + dataset load and warm-up (our runs: c128 5-12 min, c512 at 1/lane 12-32 min, longest for
the engine with the lowest hit rate) + `--duration` (1800 s; smoke 300 s). A Slurm time limit of 2.5 h covers every
cell; a smoke needs ~45 min. Host memory: the engine plus 4 x `--host-kv-gb` GB (`--mem=600G` at 100; 400G without
offloading).

Warm-up: the scenario sends one full-history primer per live session, then 10 more requests per lane, and starts
measuring only after all of them return (no time cap). At c512 that is ~5,600 requests, hours for an engine that
cannot keep the contexts. For c >= 256 pass `--warmup-per-lane 1` for every engine (the run is then stamped
`submission_valid: false`; compare only runs with the same warm-up, `compare` groups by it).

## Compare

```bash
python -m benchmarks.agentx compare runs/agentx/*/      # one table per (concurrency, warm-up), best req/s first
python -m benchmarks.agentx table runs/agentx/*/        # one row per run
```

Both skip (and name on stderr) directories without a `run.json`, such as a Slurm log directory under
`runs/agentx/`; `benchmarks/jobs/agentx.sbatch`'s example writes the job logs to `runs/agentx-slurm/` instead.

Report per run: req/s, cache hit, TTFT p50/p90/p99, prompt and computed tokens/s per GPU, failed requests, and the
`build` column (image or commit). Keep the run directories.

## What goes wrong

- **One failed warm-up request aborts the whole run** (AgentX: `warmup_failure`). Check `client.log` for the first
  error, then `server.log`.
- **Connection reset by peer with a clean server log (vLLM).** vLLM closes idle keep-alive connections after 5 s,
  AIPerf reuses pooled ones for up to 300 s. The harness sets `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600` for vLLM.
- **SGLang `Initialization failed. warmup error` at high concurrency.** SGLang answers `/v1/models` before its own
  start-up warm-up request returns; the harness waits for `/health` (503 until then) so the client cannot starve it.
- **A worker killed with no traceback.** On a shared cluster, check the host's GPU-idle killer (on cayenne
  `journalctl | grep idle_kill`): per-process NVML utilization can read low while the GPU is busy.
- **Weights not found in a container.** The container sees only mounted paths; pass the checkpoint by a path that is
  mounted at the same place (`AGENTX_DOCKER_ARGS -v`).
- **SSD tier start fails.** Shards are created once with `--gb` equal to `--ssd-gb`; a server locks one set; two
  servers at once need `--sets 2`.
- **Long TTFT tails under overload are expected.** SSEngine schedules shortest-remaining-work first and forces any
  request that waited 60 s; past saturation the forced requests queue among themselves (tails of ~2x 60 s).

## Reference numbers (cayenne, 4x B200, 2026-11-07)

Our runs, for a sanity check of a new setup (vLLM here is our fork, not v0.33.0; SGLang and vLLM without offloading).
SSEngine ran with a 128 GB SSD tier, which was full at c512; the matrix above uses 512 GB. All of them measured
`--duration 900` (the scenario's minimum), not the default 1800 s the matrix uses; the numbers should agree within
noise, but say which one you ran.

| c / warm-up | engine | req/s | cache hit | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|---|
| 128 / 10 | SSEngine pp4 (DRAM 100 GB, SSD 128 GB per GPU) | 8.66 | 0.932 | 0.30 / 0.58 / 2.41 |
| 128 / 10 | SGLang pp4 | 7.30 | 0.874 | 2.46 / 11.1 / 70.8 |
| 128 / 10 | SGLang dp4-ep | 5.99 | 0.841 | 6.9 / 29.9 / 99.8 |
| 128 / 10 | vLLM fork tp4 | 3.92 | 0.813 | 32.0 / 64.1 / 151 |
| 512 / 1 | SSEngine pp4 | 17.3 | 0.922 | 2.6 / 61 / 112 |
| 512 / 1 | SGLang dp4-ep | 1.73 | 0.379 | 109 / 478 / 558 |
| 512 / 1 | vLLM fork tp4 | 0.87 | 0.144 | 241 / 496 / 614 |
