# Setup record (`1_contract.md` §6 Setup)

One node, recorded 2026-10-15. Measured values only; anything not yet measured says so.

## Model

| | |
|---|---|
| Checkpoint | `nvidia/GLM-5.3-Flash-NVFP4`, snapshot `da920bb0b9f4a06727223a349e55468e38352348` (the only snapshot on disk) |
| Location | `$HF_HOME/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0…`, 33 safetensors shards, 191 GB |
| Text corpus | enwik9 at `<workspace>/data/corpus/enwik9`, 1,000,000,000 bytes, md5 `e206c3450ac99950df65bf70ef61a12d` (matches §2.2) |

## Serving code and kernel libraries

The engine (`SSEngine`, provenance in its `README.md`) and the correctness reference run in `.venv`,
declared final by the model lane at `lane-model` 8ec9465. It holds the vLLM fork checkout
`third_party/vllm`, commit `892da0822f` (`servingstudio-alignment` line, the same commit as
`wt-flash-dflash2/alignment/profiler/vllm`):

| Library | Version / commit |
|---|---|
| vLLM fork | `892da0822f` (`vllm 0.1.dev21920+g892da0822.precompiled`, editable) |
| PyTorch | 2.13.0, CUDA 13.0 |
| FlashInfer | `flashinfer-python` 0.7.2.dev20261008, git `73b63f88` (nightly; the fork venv pins 0.7.0 `4d75a33f`). JIT cache `~/.cache/flashinfer/0.7.2.dev20261008/100a` |
| DeepGEMM | vendored `vllm/third_party/deep_gemm` 2.8.0, built from `vllm-project/DeepGEMM@e1f418c2a4f20818221f6b0e578b4c2f634d4c3f` (has `mega_mhc`) |
| DeepSelect | `vllm/_deepselect_C.abi3.so`, `vllm-project/DeepSelect@d96d33afe1fab0d6066da49cdc91e64c2bee65ea` |
| FlashKDA / FlashMLA | `vllm-project/FlashKDA@b59532f1`, `vllm-project/FlashMLA@0eee43b1` (built into the fork; not in the design's path) |
| TileLang | 0.1.12 |
| Triton | 3.7.1 |
| CuTe DSL | `nvidia-cutlass-dsl` 4.8.0 (fork venv: 4.7.1) |
| cuBLAS / cuDNN / NCCL | 13.1.1.3 / 9.20.0.48 / 2.29.7 (cu13) |
| causal-conv1d (Dao-AILab) | 1.7.0, on `PYTHONPATH` from `third_party/causal_conv1d_try/pkg` (not in the fork venv, whose `glm5next` KDA layer calls vLLM's Triton `causal_conv1d_fn`) |
| transformers / safetensors | 5.19.0 / 0.8.0 |

The kernel rows (`data/kernel_specs.jsonl`) were measured by the ServingStudioSim profiler in other environments:
most backends in the `vllm_env` container (`vibesim-profiler-vllm:cu130-3f667d7e`, fork commit 3f667d7);
`dsa_topk_prefill/deep_select` and `mhc_fused_post_pre_rms_norm/deepgemm_mega_nonshifted` in the fork venv;
`kda_chunk_prefill/flashinfer_cute_persistent` with FlashInfer 0.7.2.dev20261008 (commit `73b63f88`, CuTe DSL 4.8.0,
Torch 2.11); `gdn_causal_conv_prefill/dao_channellast` with causal-conv1d 1.7.0 on Torch 2.10 (cu128). See
`kernels_nodecheck.md` for the same profiler re-run on this node.

## GPUs

| | |
|---|---|
| GPUs on the node | 8 × NVIDIA B200 (Blackwell), 183,359 MiB each, VBIOS 97.00.9A.00.0F |
| Driver / CUDA | 580.178.04 / 13.0 |
| Interconnect | every pair NV18 (18 bonded NVLinks), one NVLink domain; GPUs 0–3 on NUMA 0 (CPUs 0–55,112–167), 4–7 on NUMA 1 |
| A job's 4 GPUs | varies per job. Node-check job 7020 got host GPUs 0, 2, 3, 6 (bus 1B, 52, 61, D1): three on NUMA 0, one on NUMA 1, all pairs NV18. Reference job 7029 got host GPUs 1, 4, 5, 7 (bus 43, 9D, C3, DF) |

`nvidia-smi topo -m` (GPU part, whole node):

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity     NUMA
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  0-55,112-167     0
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  0-55,112-167     0
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  0-55,112-167     0
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  0-55,112-167     0
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  56-111,168-223   1
GPU5    NV18  NV18  NV18  NV18  NV18   X    NV18  NV18  56-111,168-223   1
GPU6    NV18  NV18  NV18  NV18  NV18  NV18   X    NV18  56-111,168-223   1
GPU7    NV18  NV18  NV18  NV18  NV18  NV18  NV18   X    56-111,168-223   1
```

The node is shared and scheduled by Slurm; a job gets whichever 4 cards are free.

**Not measured here** (they come from the engine, model lane): usable HBM per GPU, the largest stage's weights, and
the KV pool in tokens against 61,258,368.

## Host DRAM

| | |
|---|---|
| CPUs | 2 × Intel Xeon Platinum 8570, 224 hardware threads, 2 NUMA nodes |
| DRAM | 2,015 GiB total (`MemTotal` 2,113,450,468 kB); 1,484 GiB available at 01:44 (shared node, 48 users logged in) |
| Contract | 150 GB per GPU, 600 GB for 4 GPUs: fits in the node's DRAM |
| Pinned DRAM → HBM | 55.7 GB/s per GPU alone (each of 4 GPUs, 4 GiB pinned copies, job 7020); 4 GPUs at once: 55.7 / 48.5 / 52.2 / 55.7, sum 212.1 GB/s (`research/bench/nodecheck/h2d_bw.py`, dev branch) |

## Local SSD

| | |
|---|---|
| Device | `/raid`: `/dev/md1`, RAID 0 of 8 × KIOXIA KCMYDRUG3T84 NVMe (3.5 TB each, 27.9 TB), ext4. (`/` is a RAID 1 of 2 Samsung NVMe, 109 GB free; not used.) |
| Free space | **469 GB** (`/raid` 99% used, shared with other users) |
| Contract | 256 GB per GPU, 1 TB for 4 GPUs at 10 GB/s each (40 GB/s) |
| Sequential read, measured | see below |

`fio`, `O_DIRECT` (`--direct=1`, libaio), on a 64 GiB file on `/raid` written just before, each run one pass
(no rereads), at 01:48 with load average 18 on 224 threads:

| Pattern | Read bandwidth |
|---|---|
| 4 MiB blocks, queue depth 16, 4 jobs (16 GiB each) | 103 GB/s (95.8 GiB/s) |
| 1 MiB blocks, queue depth 32, 1 job (32 GiB) | 40.1 GB/s |
| 1 MiB blocks, queue depth 1, 1 job (16 GiB) | 3.8 GB/s |

A 16 GiB file read repeatedly for 15 s gave 50–94 GB/s for the parallel patterns and 3.7 GB/s at queue depth 1.
Writes (4 MiB, queue depth 16) ran at 17.6 GB/s. The array is shared: other users' I/O lowers what a run sees.
The striped array meets 40 GB/s for the node only with deep queues (≥ 32 outstanding MiB-sized reads); one
synchronous reader gets 3.8 GB/s.

## Differences from `1_contract.md` §1 (for `issues.md`)

| Item | Contract | Here | Affects |
|---|---|---|---|
| Local SSD capacity | 256 GB per GPU, 1 TB for the node | 469 GB free on the shared `/raid` | `7_offloading.md` runs that need the full SSD tier (`tiers`, `ssd_heavy`) |
| Local SSD bandwidth | 10 GB/s per GPU, 40 GB/s for the node | 40.1 GB/s needs ≥ 32 outstanding 1 MiB reads; 103 GB/s with 4 × qd16 4 MiB; 3.8 GB/s for one synchronous reader; shared with other users | SSD-tier read times (`7_offloading.md`) unless the reader keeps deep queues |
| Host DRAM bandwidth | 50 GB/s per GPU | 55.7 GB/s per GPU alone; 48.5–55.7 (sum 212.1) with 4 GPUs at once | none (meets it) |
| GPUs | 4 × B200 in one NVLink domain | met: all pairs NV18; Slurm picks the 4 cards per job, which may span both NUMA nodes | none |
| Kernel library versions | the rows' environments (above) | FlashInfer nightly 0.7.2.dev20261008 and CuTe DSL 4.8.0 instead of the fork's 0.7.0 / 4.7.1; Dao causal-conv1d added | kernel times: see `kernels.md` (re-measured in this env) and `kernels_nodecheck.md` |

Not measured by this lane: usable HBM per GPU, the largest stage's weights and the KV pool (model lane).
