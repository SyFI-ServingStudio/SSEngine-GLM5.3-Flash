# M6 design: prefix KV in DRAM and SSD (7_offloading.md)

Status: draft for review, nothing implemented. Branch lane-m6 from m5-done (a167593).

## 0. Summary

| Item | Choice |
|---|---|
| What is already there | `ppimpl/engine/tiers.py` (the per-tier LRU index, write-through/victim policy, read channels, read-wait bound) and the scheduler's read path (`_at_head`, `read_landed`, landed rounds first) are written and CPU-tested in replay. Nothing moves bytes yet: no worker executes a `TierOp`, the controller never sends them. M6 adds the byte path and the completion reports, and fixes the index to the real slot sizes. |
| Balanced storage | Every HBM byte range a context owns (each KV page of each DSA layer of each stage, each KDA state of each stage) is cut into 4 equal contiguous quarters; GPU q stores and moves quarter q of every range. Each GPU then holds and reads exactly 1/4 of every context (1,499 B/token + 36.9 MB on average, `7_offloading.md` §2), 3/4 of it to or from the other stages' HBM over NVLink, by GPU q's own copy engine. |
| Copy mechanism | Copy engine only (no SMs): one `cuMemcpyBatchAsync` per tier op per GPU (vLLM `ops.swap_blocks_batch`), source and destination lists built from a per-GPU pointer table of all 4 stages' KV views (CUDA IPC, opened once at start-up). |
| DRAM | Pinned host memory per GPU on that GPU's NUMA node (libnuma + `cudaHostRegister`, not torch's power-of-2 pinned allocator), cut into fixed slots of one KV block's quarter; same slot ids on all 4 GPUs, owned by the controller's index. |
| SSD | One preallocated file per GPU under `PPIMPL_SSD_DIR` on /raid (md RAID0 of 8 NVMe), same slot layout; O_DIRECT `preadv`/`pwritev` from a small thread pool per worker into/out of DRAM slots; written only when DRAM evicts (write-back, wt-pp-ssd-writeback 3247309c), read SSD -> DRAM slot (copy-up) -> HBM, pipelined per slot. No GDS (nvidia-fs is not loaded on this node). |
| Writes | Issued when a round's token is out; background stream, never waited on by compute; only new blocks + state. HBM cannot evict a session (and its next round cannot take over its blocks) until all 4 GPUs reported its write done. |
| Reads | As `7_offloading.md` §2 and the scheduler already does: HBM footprint reserved at read start, read-wait bound 1 s, one read at a time per tier per GPU, landed rounds admitted first in landing order. A read lands when all 4 GPUs report it. |
| Eviction | LRU whole sessions per tier (index unchanged); DRAM slots of a session being written back to SSD are freed only when the write-back is done. |
| New code (estimate) | worker tier executor ~250 lines, controller wiring ~80, index slot accounting ~60, pinned arena + SSD file ~120, tests ~250. |

## 1. Bytes per context (this engine's HBM layout)

The layout is vLLM's (`ppimpl/model/kv_tensors.py`): one tensor per layer, block-major, block = 8,576 tokens.

| Range | Per stage | Size |
|---|---|---|
| KV page | for each DSA layer: MLA page + indexer page of one block | 8,576 x 545 B per layer = 4.67 MB; stage 0/2/3: 3 layers (14.02 MB per block), stage 1: 2 (9.35 MB) |
| KDA state | for each KDA layer: recurrent fp32 [64, 128, 128] = 4 MiB + conv bf16 [3, 3 x 8192] = 144 KiB | 4.34 MB per layer; 9 / 9 / 8 / 8 layers = 39.1 / 39.1 / 34.7 / 34.7 MB |
| kpool tail | per DSA layer, bf16 [2, 4, 128] | 2 KiB (state block 5) |

Per GPU after quartering: one KV block = (3 x 14.02 + 9.35) / 4 = 12.86 MB = 8,576 x 1,499 B; one state = 36.9 MB.
The exact view list (names, row bytes, strides) is read from the stage's `StageKV` at start-up and checked against
these sums; a view whose block dimension is not dim 0 or whose row is not divisible by 4 stops the server.

## 2. Transfers

### 2.1 What existing systems do (read from source, 2026-10-17 clones; paths relative to each repo)

| System | H2D load mechanism | Granularity / layout | One host pool for several GPUs? | Disk | Best published GB/s (conditions) |
|---|---|---|---|---|---|
| LMCache, in-process | UVA gather kernel reading pinned host memory, grid tokens x layers x 2 (`csrc/cuda/mem_kernels.cu:669`) | 256-token chunk `[2, L, T, hd]`, layout converted | no, per worker (5 GB default) | file per chunk, O_DIRECT opt-in; cuFile | 50 GB/s load (arXiv 2510.09665 Table 5, 8 x H100 testbed) |
| LMCache, multiprocess | `cudaMemcpyBatchAsync` per chunk, copy engine only ("direct"), or staging + scatter kernel | one entry per (kv, layer, block); cost model 0.6 µs per entry, direct wins above ~34 KB | yes, one L1 per node via CUDA IPC; each rank's slice its own object | GDS slab, NIXL / fs | 55 GB/s CE line rate, H200 (`docs/design/v1/multiprocess/lmcache_driven_transfer_copy_paths.md:106`) |
| SGLang HiCache | `kernel`: 2-CTA (or 4-CTA TMA on sm90+) UVA copy kernels; `direct`: `cudaMemcpyBatchAsync` | layer_first `(2, L, tok, H, D)`, page_first `(2, tok, L, H, D)`, page_first_direct `(2, pages, L, page, H, D)`; layer-wise loads | no: "host memory cannot be pooled across instances" (`hicache_best_practices.mdx:11`); MLA: rank 0 holds, NCCL broadcast | file per page, buffered; Mooncake / 3FS / NIXL | 192 GB/s with 4 of 152 SMs on GB300 NVLink-C2C (PR #40278); ~50 GB/s with 2 CTAs on H200 Gen5 (Strata, arXiv 2508.18572) |
| vLLM (this fork) | `cuMemcpyBatchAsync` (`ops.swap_blocks_batch`), 0 SMs; 12-CTA Triton UVA kernel only for pages < 28 KB | per (layer, block), or a 2 MB cross-layer block | yes, one `/dev/shm` mmap, per-worker slots, each worker registers it (`kv_offload/cpu/shared_offload_region.py`) | file per block group, O_DIRECT, 16 + 16 threads; no GDS | ~50 GB/s one direction, H100 (vLLM blog "KV offloading connector") |
| Dynamo KVBM / NIXL | one memcpy per contiguous block, else `vectorized_copy` kernel (up to 1024 x 256) or batch memcpy | FullyContiguous `[layers, outer, inner]` per block | no, per worker | one fallocated O_DIRECT file per worker; NIXL POSIX (io_uring) / GDS | 25.2 GB/s plateau, GPU not stated (`lib/kvbm-kernels/README.md`) |
| Mooncake TE / Store | the host engine (e.g. SGLang) does H2D; TENT nvlink transport uses `cudaMemcpyBatchAsync` | 64 KB slices sprayed over NICs | Store is a cluster pool over the network | TENT gds / io_uring, O_DIRECT | 87-190 GB/s RoCE, 780 GB/s EFA (network, not H2D) |

- **"Only per-GPU pools":** half right. vLLM and LMCache multiprocess already share one pinned pool across the node's
  GPUs, but in every system each GPU still loads its own slice through its own link. None stripes one context over
  several GPUs' PCIe links and forwards over NVLink; that part is new here. Closest precedents: vLLM's replicated-MLA
  store rotation (D2H only) and SGLang's MLA rank-0 load + NCCL broadcast (one link).
- **"~30 GB/s":** no source caps any of them at ~30 GB/s on PCIe Gen5. The nearby numbers are Gen4 links (LMCache
  #4626: L20, raw 25.1 GB/s, LMCache 22.1; Dynamo's 25.2 GB/s plateau, GPU not stated), small fragmented copies on Gen5
  (Strata: HiCache 32-token pages at ~22% of Gen5, 10.8 GB/s sustained), or a network table (LMCache report Table 1,
  "1MB -> 30GBps" is RCCL between nodes). On Gen5 they publish 50-57 GB/s per GPU for large copies. Nobody publishes
  4 GPUs reading host memory at once.
- **What follows for us:** copy engine with large descriptors (>= ~1-2 MB) reaches line rate with 0 SMs; small or
  per-token copies are what fall to 10-30 GB/s. A UVA kernel needs only 1-4 CTAs for 50 GB/s (one CTA tops out at
  ~48-66 GB/s, its write port) but still takes SMs from compute that our stages keep busy.

### 2.2 This node

- 8 x B200, each on its own root port (PCIe Gen5 x16, 32 GT/s, behind a ConnectX-7 bridge; no shared switch uplink),
  GPUs 0-3 on NUMA node 0, 4-7 on node 1, all pairs NV18. Host: 2 sockets, ~1 TB per NUMA node.
- /raid: md RAID0 (512 KiB chunk, ext4) of 8 KIOXIA CM7 3.84 TB, each Gen5 x4; 4 behind a switch on each socket.
  nvidia-fs is installed but not loaded, so cuFile runs only in compat mode (POSIX), no GDS.
- fio, O_DIRECT, 00:46-00:49 (during job 7907):

  | Test | GB/s |
  |---|---|
  | seq read, io_uring qd 16, 1 job (bs 1-16 MiB) | 33.5-35.1 |
  | seq read, io_uring qd 16, 4 jobs (one file each) | 96.9-99.6 |
  | seq read, psync bs 4 MiB, 4 / 8 / 16 threads on one file | 34.0 / 52.9 / 73.9 |
  | seq write, io_uring qd 16 bs 4 MiB, 4 jobs | 53.8 |
  | 4 readers + 4 writers at once | 45.4 read + 25.8 write |

  The node target (4 x 10 GB/s read while ~4.5 GB/s is written back) is within this with 2x margin.

### 2.3 Host <-> GPU on this node (job 7916)

Pending: job 7916 (bench/tier_bw.py, 4 GPUs, 10 min) is queued behind other users' 8-GPU jobs. It measures, per
GPU and with 4 GPUs at once, NUMA local and remote: one cudaMemcpyAsync per transfer from 64 KiB to 1 GiB (H2D, D2H,
both directions together); `cuMemcpyBatchAsync` of scattered 64 KiB-8 MiB device pages (our read/write shape); the
vLLM Triton UVA kernel by CTA count (4-148); GPU g's copy engine writing into the other GPUs' HBM (the balanced
path), single copies and batched; a BF16 GEMM's slowdown while copies run; the time to pin 16 GiB.

### 2.4 Choices

Taken now, each with the 7916 number that confirms it or switches to the fallback:

| Choice | Why | Confirmed by (7916) | Fallback |
|---|---|---|---|
| Copy engine, one `cuMemcpyBatchAsync` per op per GPU (vLLM `ops.swap_blocks_batch`), 0 SMs | every system that reaches Gen5 line rate without SMs does this; our stages keep all SMs busy | batch H2D >= 45 GB/s per GPU at 512 KiB-2 MiB pieces, 1 and 4 GPUs; GEMM slowdown < 1% | pieces too small: diagonal assignment of whole pages (§3) instead of quarters |
| Balanced by quarters, GPU q's CE writes peer HBM over NVLink | exact 1/4 per link (1,499 B/token, 36.9 MB) | peer / peer_batch H2D per GPU within 5% of own-GPU H2D, 4 GPUs at once | two hops: H2D into a staging ring on GPU q, then a D2D peer copy (CePipe-style) |
| page_first host layout (a slot = one block's quarter of all layers and stages, contiguous) | one host run per slot; host side never the small piece | - | - |
| No layer-wise load pipelining | the round computes only after the whole prefix landed (target model); DRAM reads 1.7-16 ms | - | - |
| Host memory on the GPU's NUMA node, registered by `cudaHostRegister` (not torch pin_memory) | torch rounds pins to powers of 2; remote NUMA caps lower | local vs remote H2D | preferred-local allocation, remote spill logged |
| SSD: O_DIRECT `preadv`/`pwritev`, 8 threads per GPU, one preallocated file per GPU | fio: 8 psync threads give 53 GB/s on one file; no GDS on this node (nvidia-fs not loaded; it would also need DRAM copy-up anyway) | per-read latency measured at step 2 | io_uring via liburing if threads miss 10 GB/s per GPU |

Note: with pipeline-parallel stages, own-stage storage already sends about a quarter of a context through each link
(stage 0's share is 1,635 / 1,499 = 9% above the mean). Balanced storage buys that 9% and equal tier sizes; the
handoff makes it the design (§2), so step 1 builds it, and the own-stage variant stays a config for the record.

## 3. Balanced storage and the slot layout

- **Quarter rule.** For every range r (address a_r on stage s_r, n_r bytes): GPU q moves [a_r + q n_r / 4,
  a_r + (q+1) n_r / 4). One rule for KV pages and states, exact balance for every context (no "diagonal" block
  assignment, which is off by up to one stage-block for contexts of fewer than 4 x k blocks, and makes the state the
  stage-0 39.1 MB on 2 of the 4 GPUs).
- **Host layout per GPU (page_first).** A tier slot holds GPU q's quarter of one KV block of all 4 stages, packed
  stage by stage, layer by layer: 12.86 MB rounded up to 4 KiB (O_DIRECT). A context's state takes 3 slots
  (36.9 MB in 38.6 MB). A context of P tokens = ceil(P / 8,576) KV slots + 3 state slots. The same slot ids on
  every GPU: the controller's index allocates them, the workers only execute.
- **Pieces per op.** One KV slot = 22 device ranges (MLA + indexer page of each of the 11 DSA layers, over the 4
  stages), each a quarter page: 8,576 x 545 B / 4 = 1.17 MB per layer in total, split between MLA and indexer by
  their row sizes (read at start-up); the state = 34 recurrent quarters of 1 MiB + 34 conv quarters (36 KiB) + 11
  tails. A 512k read is 62 x 22 + 79 = 1,443 pieces in one `cuMemcpyBatchAsync` per GPU. Host side contiguous.
- **What it costs.** Block granularity: the last block is stored whole. Mean waste half a block per context per GPU
  (6.4 MB) plus 1.7 MB of state rounding: about 4% of the average session context (~100k tokens); a 32k read moves
  4 x 12.86 + 38.6 = 90 MB per GPU instead of 86 MB (+5%). Recorded as a difference (§7 "a context stored larger").
  The index charges real slots, so tier capacity is exact.

## 4. DRAM

- Per worker at start-up, on a side thread while the weights load: `numa_alloc_onnode(dram_bytes, gpu_node)`,
  `madvise(MADV_HUGEPAGE)`, first touch, `cudaHostRegister(portable | mapped)` (7916 times pinning 16 GiB; 150 GB is pinned once per server start, in
  parallel with the weight load). torch's pinned
  allocator rounds to a power of 2 (150 GB -> 256 GB), so it is not used.
- Sizes per GPU: 150 GB (full), 37.5 GB (Tiers), 18.75 GB (SSD-heavy); plus a 2 GiB staging area for SSD-only runs
  (DRAM tier off): SSD reads and writes pass through it, it serves no hits.
- Node: RealMemory 2,047,532 MB, two NUMA nodes of ~1 TB; GPUs 0-3 are on node 0, 4-7 on node 1. A full-memory run
  pins 600 GB on one node. Slurm `--mem`: 800G for full memory (today's session jobs get 400G; worker RSS ~17 GB),
  350G for Tiers, 300G for SSD-heavy. If the node lacks local memory, allocation falls back to the other node
  (preferred, not bound) and the split is logged; remote-node bandwidth is in §2.

## 5. Writes (HBM -> DRAM, and DRAM -> SSD)

- **When.** `Scheduler.finish` -> `Tiers.store` emits a `write` op with the round's new KV blocks (from the block that
  holds the previous context's last token) and its state blocks, and the destination slots. The controller sends
  tier ops to all 4 workers with the next plan (or on their own when no plan is due).
- **Worker.** A tier thread (not the launch thread) builds the pointer lists (numpy over the start-up pointer table)
  and calls `swap_blocks_batch` on a write stream; an event per op; the reporter thread sends `TierDone(op_id,
  stage, start_ns, end_ns)`. Copy engine only, no SM, no wait on the compute stream: the data is final (the round's
  last chunk passed every stage before stage 3 reported the token).
- **Safety.** Until all 4 `TierDone`s: the KvManager does not evict that kept session, and the session's next round
  does not take over its blocks (it waits in the queue, passed over like a read-bound round). Expected rare: a
  37 MB + new-KV write takes ~1-3 ms.
- **SSD write-back.** DRAM eviction emits `demote` (DRAM slots -> SSD slots). A worker I/O thread pool (8 threads)
  `pwritev`s the slots (O_DIRECT, 4 KiB aligned). The DRAM slots return to the free list on all-4 `TierDone`. DRAM
  evicts ahead to keep 2% free, so a write never waits for a write-back. SSD write load at N = 400: 1.12 GB/s per GPU
  average, 1.7 GB/s peak (§4 of the handoff).

## 6. Reads (DRAM/SSD -> HBM)

- Scheduler unchanged (`_at_head`): footprint reserved when the read starts, read-wait bound 1 s, the round out of the
  queue, landed first in landing order. Only the blocks HBM lacks (from the block holding HBM's last token) + state.
- **DRAM read.** One `swap_blocks_batch` per GPU on the read stream (H2D; the writes are D2H, the other PCIe
  direction). Reads of one tier on one GPU run one after another (one stream).
- **SSD read.** I/O pool `preadv`s the session's SSD slots into newly allocated DRAM slots (copy-up), and after each
  group of slots lands the tier thread enqueues their H2D: read time ~ max(SSD, H2D) + one slot. SSD-only runs use
  the staging area instead of DRAM slots.
- **Landing.** `read_landed` when all 4 `TierDone`s are in. The index's channel clock (when the tier is free, for the
  1 s bound) uses the configured bandwidth and is pushed to the real landing time when a read lands late.
- No layer-wise pipelining: a round computes only after its whole prefix landed (the target's model); reads are
  1.7-16 ms (DRAM) and 9-82 ms (SSD) against 52-57 ms of round-1 compute.

## 7. Eviction

Unchanged in the index: each tier LRU of whole sessions, a read refreshes. Slots instead of bytes. SSD evicts by LRU
into nothing (its slots are reused). HBM eviction waits for the session's write (§5).

## 8. Correctness

- `tests/test_tier_bytes_gpu.py` (GPU, 4 stages, real `StageModel`): prefill P = 40,000 (partial last block, chunk
  starts off KDA's grid) in session s, snapshot every KV page and state row of s on every stage, write to DRAM,
  evict s from HBM, fill its blocks with garbage, read back into different block ids: every byte equal. The same
  through SSD (DRAM off). Then round 1 of s (A = 1k) after the tier read vs after an HBM hit: logits bitwise equal
  (same microbatch shapes), token equal.
- `tests/test_tiers.py` (CPU): slot accounting, write-before-evict, takeover wait, demote frees DRAM only when done,
  SSD LRU eviction at a small SSD (slots reused, the evicted session found nowhere and computed again), landing order; the quarter rule's pointer lists (every byte of every range covered exactly once over the 4 GPUs).
- Bring-up: round-1 tokens against the no-cache reference as in M4 (`bench/prefix_check.py`).

## 9. Steps and runs

| Step | Code | Runs (each on a fresh server, SSD file deleted after) |
|---|---|---|
| 1a | pointer table, quarter rule, DRAM arena, write/read ops, TierDone, scheduler wiring | GPU test §8; tier bring-up DRAM (150 GB, pool 1,234,944 tokens): per-P round-1 TTFT and read time vs §3 |
| 1b | | sessions full N = 400/500 and h25d37/h12d19 (DRAM only, step 1) against the v2-realistic tier targets |
| 2a | SSD file, I/O pool, demote, copy-up, staging | GPU test through SSD; bring-up SSD only (128 GB) |
| 2b | | sessions full / h25d37 / h12d19 at N = 400/500 with SSD 128 GB (`*_ssd128`) |

## 10. Targets (v2-realistic, replace the handoff's numbers)

Computed by the coordinator with the realistic-input profile (ServingStudioSim wt-pp-hitrule
`logs/20261017_2_v2_realistic_tiers`: `tiers/`, `ssdsize/`, `tiercheck/`; metrics by the handoff's `data/metrics.py`).

**Tier bring-up** (`tiercheck/`, `data/tier_targets.py`), ms, read time in parentheses:

| P | Round 0 | HBM hit | No tier | DRAM (balanced) | SSD (balanced) | DRAM own stage | SSD own stage |
|---|---|---|---|---|---|---|---|
| 32k | 367.5 | 52.7 | 387.9 | 54.4 (1.72) | 61.3 (8.60) | 54.5 (1.85) | 61.9 (9.26) |
| 128k | 1,200.2 | 53.7 | 1,221.2 | 58.4 (4.67) | 77.1 (23.34) | 58.8 (5.07) | 79.1 (25.34) |
| 512k | 4,949.0 | 57.5 | 4,972.7 | 73.9 (16.46) | 139.8 (82.28) | 75.4 (17.93) | 147.1 (89.63) |

The read times do not depend on the compute profile and equal the handoff's.

**Sessions, SSD 128 GB per GPU** (`ssdsize/*_ssd128`; the user's setting): r/s, new tok/s, TTFT p50/p90/p99 (s),
rounds over 60 s.

| Memory | N = 400 | N = 500 |
|---|---|---|
| full | 26.10, 90.8k, 0.16/0.76/6.72, 127 | 28.37, 100.9k, 0.30/2.31/60.90, 753 |
| h25d37 (Tiers) | 26.06, 90.7k, 0.19/0.80/6.76, 121 | 28.12, 100.0k, 0.37/2.69/60.92, 765 |
| h12d19 (SSD-heavy) | 26.01, 90.7k, 0.20/0.84/6.88, 119 | 27.69, 98.8k, 0.39/3.37/60.93, 731 |

Step 1 (DRAM only, `tiers/*_dram`): full N = 400 is the same as full_ssd128 (26.10, 90.8k, 0.16/0.76/6.72, 127),
since at full memory an SSD no larger than DRAM holds nothing DRAM does not.

**SSD size differs from the handoff.** The user set 128 GB of SSD per GPU; `7_offloading.md` has 256 GB. At 128 GB the
SSD tier evicts (LRU, finished sessions first: the active set is ~73-87 GB per GPU, notes.md 2026-10-16), so the
runs compare against `*_ssd128`, and the SSD eviction path (slots freed, an evicted session computed again) is part
of every h25d37/h12d19 run and of the CPU tests (§8). Every run uses 128 GB (`TierConfig.ssd_gb` default; user,
2026-10-17); no 256 GB run.

**Memory per run (user, 2026-10-17: reserve, then release).** Only full-memory session runs ask for `--mem=800G` and
pin 600 GB (150 GB per GPU); the job ends, the memory is free. Bring-up: the handoff gives DRAM 150 GB, but the run
writes only ~38 GB per GPU (75 fillers of 256k at 430 MB plus 15 sessions at 5.7 GB), so a 48 GB arena per GPU holds
all of it and nothing is evicted, as at 150 GB: `--mem=300G`. h25d37 / h12d19: 37.5 / 18.75 GB per GPU,
`--mem=350G` / `300G`.

## 11. What is reused, from where, and why not the others

Decision (coordinator, user's rule "reuse what fits, change or write what does not"): reuse primitives, not whole
connectors. LMCache's and HiCache's connectors load each GPU's slice through its own link and are built around vLLM's
or SGLang's scheduler; wiring them to balanced striping and our scheduler would take more code than the parts below.

| Part | Reused | From | Why not the alternatives |
|---|---|---|---|
| Batched H2D/D2H copy, 0 SMs | `ops.swap_blocks_batch` (`cuMemcpyBatchAsync`, one driver call per op) | vLLM fork `vllm/_custom_ops.py`, `csrc/.../cache_kernels.cu`; used by `kv_offload/cpu/gpu_worker.py` | LMCache in-process and HiCache `kernel` use SM copy kernels (2-4 CTAs, or a grid over tokens x layers); our stages keep the SMs busy. LMCache MP and HiCache `direct` use the same CUDA call, so nothing to gain from taking their wrappers |
| Peer pointers to the other stages' KV | CUDA IPC export/open as `CePipe` already does | `ppimpl/runtime/transport.py` | LMCache MP shares GPU KV over CUDA IPC the same way; no new mechanism |
| Pinned host memory | libnuma first-touch on the GPU's node + `cudaHostRegister` (portable, mapped), chunked | the pattern of LMCache `csrc/cuda/mem_alloc.cpp` (mmap + mbind + register) and vLLM `kv_offload/cpu/shared_offload_region.py` (mmap + register in <= 64 GB chunks) | torch `pin_memory` rounds to a power of 2 (150 -> 256 GB); vLLM's region is one `/dev/shm` file with per-worker rows and no mbind (NUMA by first touch only): we need each GPU's slice bound to its node and need no shared file (each worker maps only its own slice; peer bytes go over NVLink, not through another worker's host memory) |
| Host layout | page_first: a slot holds one block of all layers (and all stages, quartered) contiguously | the idea of HiCache `page_first_direct` `(2, pages, L, page, H, D)` and vLLM 0.12's cross-layer block | HiCache's `layer_first` and per-token `page_first` give small copies (10.8 GB/s sustained in Strata's measurement); the HBM side stays vLLM's per-layer layout (changing it would touch every attention kernel), so pieces are per (layer, block) quarters there |
| O_DIRECT probe | `probe_o_direct` | vLLM fork `kv_offload/tiering/fs/io.py` | - |
| SSD file I/O | own: one preallocated file per GPU, slot-addressed `os.preadv` / `os.pwritev` with O_DIRECT from 8 threads (GIL released) | pattern of Dynamo KVBM's single fallocated O_DIRECT file per worker | vLLM fs tier and LMCache/HiCache file backends write one file per block or chunk (temp file + rename, or buffered I/O): 10k+ files per GPU and an open/close per 12.9 MB; `fs_io_C` loops serially in C, so it needs the same thread pool anyway. GDS (LMCache cuFile, NIXL) is not available (nvidia-fs not loaded) and would skip the DRAM copy-up the design requires |
| Index, LRU, read-wait bound, scheduler | own (`ppimpl/engine/tiers.py`, `scheduler.py`, already written for M5 replay) | - | the handoff's rules (whole-session LRU per tier, write-back to SSD, read-wait bound, landed rounds first) are not any of these systems' policies |

## 12. Differences from the target to record

Block-granular slots (§3: +4% space, +5% on a 32k read); real write traffic on the same links (D2H vs H2D, and SSD
reads vs write-back); host time to start and land a read (measured per read in `reads.jsonl`); SSD latency of small reads (a 32k
context is 90 MB per GPU; fio's one-shot reads of that size were dominated by fio's own start-up, so measured in
step 2 inside the engine).
