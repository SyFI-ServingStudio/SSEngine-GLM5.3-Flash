# M6 design: prefix KV in DRAM and SSD (7_offloading.md)

Status: steps 1a/2a code in, CPU-tested; two hops in (`hops = "two"`); interfaces under review (§13) before the
refactor into them. Branch lane-m6 from m5-done (a167593).

## 0. Summary

| Item | Choice |
|---|---|
| What is already there | `ppimpl/engine/tiers.py` (the per-tier LRU index, write-through/victim policy, read channels, read-wait bound) and the scheduler's read path (`_at_head`, `read_landed`, landed rounds first) are written and CPU-tested in replay. Nothing moves bytes yet: no worker executes a `TierOp`, the controller never sends them. M6 adds the byte path and the completion reports, and fixes the index to the real slot sizes. |
| Balanced storage | Every HBM byte range a context owns (each KV page of each DSA layer of each stage, each KDA state of each stage) is cut into 4 equal contiguous quarters; GPU q stores and moves quarter q of every range. Each GPU then holds and reads exactly 1/4 of every context (1,499 B/token + 36.9 MB on average, `7_offloading.md` §2), 3/4 of it to or from the other stages' HBM over NVLink in a second hop through a staging ring in GPU q's HBM (direct peer writes from host run at 2/3 of the rate, §2.3). |
| Copy mechanism | Copy engine only (no SMs): one `cuMemcpyBatchAsync` per tier op per GPU (vLLM `ops.swap_blocks_batch`), source and destination lists built from a per-GPU pointer table of all 4 stages' KV views (CUDA IPC, opened once at start-up). |
| DRAM | Pinned host memory per GPU on that GPU's NUMA node (libnuma + `cudaHostRegister`, not torch's power-of-2 pinned allocator), cut into fixed slots of one KV block's quarter; same slot ids on all 4 GPUs, owned by the controller's index. |
| SSD | One shard file per pool member, kept across runs: `$PPIMPL_SSD_DIR/set<k>/shard<q>.bin` (default `/raid/kanzhu/ServingStudio/pp-impl/tmp/ssd_tier`, md RAID0 of 8 NVMe; shard q = member q's quarter of every slot, §5), same slot layout; O_DIRECT, one io_uring per worker (fixed file, the host arenas as fixed buffers, one request per slot, 8 reads / 4 writes in the kernel, reads first; §2.5). Written only with DRAM's victims (§5); read SSD -> staging slots -> HBM, pipelined per slot. No GDS (nvidia-fs is not loaded on this node). |
| Writes | The victim chain HBM -> DRAM -> SSD (§5): a tier takes only what the level above lets go of. One write-back class (`kvpool/writeback.py`) for both levels: HBM writes its least recently used kept contexts ahead of need, keeping free + clean HBM above the largest demand of two write times (measured, no knob), so eviction finds them clean; DRAM demotes to SSD only at eviction. Background, never waited on by compute; only what the tier below lacks. HBM cannot evict a kept context (and its next round cannot take over its blocks) until all 4 GPUs reported its write done. |
| Reads | As `7_offloading.md` §2 and the scheduler already does: HBM footprint reserved at read start, read-wait bound 1 s, one read at a time per tier per GPU, landed rounds admitted first in landing order. A read lands when all 4 GPUs report it. |
| Eviction | LRU whole sessions per tier; a level evicts only entries the level below holds whole (else it waits, counted); DRAM slots of a session being demoted to SSD are reused only when the demote is done. |
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

Job 7916 (`bench/tier_bw.py`, 2026-10-17 03:30, load 10.3; the job's GPUs 0-3 were physical GPUs 4-7, NUMA node 1;
`measured/tier_bw/7916/tier_bw.jsonl`). GB/s per GPU; "4 GPUs" = all four copying at once, each from its own buffer.

| Path | 1 GPU | 4 GPUs |
|---|---|---|
| H2D, one copy, 64 KiB / 1 MiB / 16 MiB / 1 GiB, local node | 13.1 / 46.4 / 54.9 / 55.7 | - / 31.1 / 48.7 / 48.4 |
| D2H, same sizes | 13.7 / 46.1 / 54.9 / 55.6 | - / 31.1 / 40.3 / 38.5 |
| H2D + D2H at once (sum of both directions) | - | 1 MiB 37.2, 16 MiB 47.3, 1 GiB 47.8 |
| H2D / D2H, host buffer on the other node, 1 GiB | 55.6 / 55.6 | 27.2 / 19.4 |
| `cuMemcpyBatchAsync` H2D, pages of 64 KiB / 512 KiB / 2 MiB / 8 MiB | 41.5 / 52.7 / 54.3 / 54.7 | 29.1 / 44.2 / 47.6 / 47.2 |
| `cuMemcpyBatchAsync` D2H, same pages | 43.8 / 52.3 / 54.1 / 54.8 | 28.5 / 36.1 / 38.3 / 38.2 |
| SM kernel (vLLM UVA copy) H2D, 4 / 8 / 16 / 148 CTAs | 20.8 / 40.3 / 50.8 / 50.7 | 16 CTAs 46.9 |
| SM kernel D2H, 4 / 8 / 16 / 148 CTAs | 42.4 / 49.2 / 50.9 / 51.9 | 16 CTAs 39.2 |
| GPU g's copy engine, host -> all 4 GPUs' HBM (3/4 over NVLink), one copy of 2 / 64 MiB | - | H2D 19.1 / 20.1, D2H 16.5 / 24.6 |
| the same, batched 2 MiB pages | - | H2D 31.7, D2H 28.3 |

- **Line rate.** One GPU reaches 55.6 GB/s both ways from 16 MiB copies or 2 MiB batched pages; the 50 GB/s target
  holds for one GPU. Four GPUs on one NUMA node share the socket's host side: H2D 48.4 GB/s per GPU (193.8 in all,
  -13%), D2H 38.5 (154.1, -31%); reads are H2D, so the read target holds within 3%; writes run at ~38.
- **Both directions at once** sum to 47.8 GB/s per GPU, not 2 x 48: a write-through running during a read slows the
  read. Writes are 1.1-1.7 GB/s per GPU on average (§5), so the overlap is short; recorded per read in `tier_ops.jsonl`.
- **NUMA.** One GPU does not see the node (55.6 either way); four GPUs reading from the other node get 27.2 (H2D) and
  19.4 (D2H): host memory must be on the GPU's own node (§4 does this).
- **Piece size.** Batched pages of 512 KiB lose 7% at 4 GPUs (44.2 vs 47.6), 64 KiB pages 39%.
- **Copy engine vs SMs.** A BF16 GEMM 8192^3 on GPU 0 takes 0.682 ms alone, 0.677 ms while all four GPUs' copy
  engines run H2D, 0.717 ms (+5%) beside a 16-CTA copy kernel: copy engines it is.
- **Peer writes are slow.** GPU g's copy engine writing host data straight into the other GPUs' HBM reaches 19-20
  GB/s per GPU (single copies) and 31.7 (batched), against 47.6 into its own HBM. **Direct peer writes are out;
  balanced storage goes in two hops** (§2.4): host <-> GPU q's own HBM over its own PCIe link, then GPU q's HBM <->
  the stages' KV over NVLink (D2D, NV18, hundreds of GB/s), pipelined in chunks of slots.
- **Pinning.** 16 GiB on one node: first touch 6.45 s (one thread), `cudaHostRegister` 1.58 s; 150 GB per GPU:
  ~60 s of touch on one thread (4 threads in the worker, overlapping the model load) and ~15 s of registering.
- **GDS:** `gdscheck -p` reports no nvidia-fs (`measured/tier_bw/7916/gdscheck.txt`), as expected.

**Second hop (job 7943, the coordinator's probe `pp-impl/coord/d2d_probe.py`, output `coord/d2d_probe_7943.out`;
3 GPUs, physical 1-3 on NUMA node 0, host load 66).**

| Measurement | Result |
|---|---|
| Peer `copy_` in the kineto trace | "Memcpy PtoP (Device -> Device)", no kernel row: copy engine, 0 SMs |
| D2D GPU q -> the peers, one 64 MiB copy | 699.6 GB/s |
| D2D, all GPUs copying in a ring at once | 670.7 GB/s per GPU |
| D2D, 2 MiB pieces, one `copy_` each from Python | 54.8 GB/s (launch overhead; one batch call avoids it) |
| H2D alone / H2D while D2D runs at the H2D rate | 55.5 / 55.0 GB/s (D2D also 55.0): H2D unaffected |
| GEMM 8192^3 on the sending GPU: alone, with D2D at 700 GB/s, with H2D + D2D at 55 GB/s, alone again | 0.686 / 0.735 (+7%) / 0.713 (+4%) / 0.682 ms |
| the same on a receiving GPU | 0.691 / 0.728 (+5%) / 0.717 (+4%) / 0.710 ms |

- The second hop runs on copy engines. Its HBM traffic slows a GEMM by 5-7% at full D2D rate (HBM contention, not SMs);
  at the rate it actually runs (paced by the host hop, <= 55 GB/s) the slowdown is ~4%, against ~3% between the two
  "alone" runs of the same GPU: near noise.
**The engine's two-hop path in one process (job 7961, `bench/tier_hop2.py`, 4 GPUs on NUMA node 0, load 14.7).** Fake
KV laid out like GLM-5.3 PP4, four `TierEngine`s, each submitted from its own thread (as in the server, where each
worker submits its own share). Bytes equal for every setting. Per op on all 4 GPUs, p50 of 10, ms; "solo" is one
GPU's read alone, with its host submit time in parentheses. (Job 7951 drove the 4 engines from one thread, so its
numbers were mostly serialized submission; superseded by this table.)

| Ring | 32k read / write | 128k read / write | 512k read / write | solo read (submit) 32k / 128k / 512k |
|---|---|---|---|---|
| direct (pieces host <-> peers) | 6.58 / 7.45 | 12.85 / 12.71 | 33.9 / 31.7 | 2.69 (1.55) / 6.09 (4.84) / 18.6 (17.4) |
| depth 2, chunk 1 | 5.51 / 7.02 | 13.0 / 13.8 | 43.8 / 42.7 | 2.47 (0.82) / 5.20 (1.82) / 16.3 (5.67) |
| depth 2, chunk 3 | 3.15 / 3.95 | 7.15 / 12.7 | 18.9 / 38.6 | 2.57 (0.60) / 5.32 (1.16) / 16.1 (3.25) |
| **depth 2, chunk 4** | **2.85 / 4.00** | **5.55 / 11.5** | **16.9 / 38.7** | 2.07 (0.63) / 5.01 (1.16) / 15.7 (3.42) |
| depth 4, chunk 1 | 5.65 / 5.78 | 14.7 / 14.9 | 41.5 / 42.4 | 2.33 (0.83) / 5.27 (1.82) / 16.3 (5.64) |
| depth 4, chunk 3 | 3.26 / 3.95 | 6.10 / 12.0 | 18.1 / 38.8 | 2.19 (0.60) / 5.34 (1.17) / 16.0 (3.23) |
| depth 4, chunk 4 | 2.81 / 4.25 | 5.60 / 12.4 | 17.0 / 38.8 | 2.65 (0.52) / 5.43 (1.03) / 16.1 (2.94) |

`cuMemcpyBatchAsync` by piece size, GPU 0's stream, 512 MB in all (GB/s; host submit included):

| Piece | local | push 0 -> 1 | pull 1 -> 0 | host -> 0 | 4 GPUs push to all 4 / pull from all 4 (per GPU) |
|---|---|---|---|---|---|
| 128 KiB | 30.5 | 128 | 128 | 43.0 | 9.4 / 3.8 |
| 1 MiB | 240 | 461 | 456 | 53.5 | 84 / 25 |
| 4 MiB | 903 | 639 | 636 | 54.9 | 246 / 80 |
| 12.86 MB | 1,396 | 708 | 708 | 56.3 | 419 / 144 |

- **Defaults confirmed: depth 2, chunk 4 (206 MB per GPU).** Best or tied at every size; depth 4 adds nothing.
  512k reads run at 50.5 GB/s per GPU with all 4 GPUs reading (target 48.4 shared-socket, §2.3 table 1).
- **Host submission overlaps the transfer.** Solo 512k: 3.42 ms of submit inside a 15.7 ms read (53 GB/s). The
  submit is ~0.25 ms of Python (34 batch calls, 1,510 descriptors, pointers computed once per op) and the rest is
  the driver (~2.3 us per D2D descriptor, ~12 us per host <-> peer descriptor in direct mode), with the GIL
  released inside the call. Going below ~1 ms of submit would need fewer descriptors (2D/3D batched copies).
- **Writes gather by pulling over NVLink**, slow with 4 GPUs at once (pull 144 vs push 419 GB/s per GPU at 12.86
  MB): 512k writes 38.7 ms (~22 GB/s per GPU). Writes are background (1.1-1.7 GB/s needed per GPU, §5); they stay
  on pull while the bring-up counters (§5) show no delay.
- **GEMM 8192^3 beside two-hop reads at 39-42 GB/s per GPU:** GPU 0 0.767 alone / 0.795 with reads / 0.792 alone
  again ms; GPU 1 0.772 / 0.779 / 0.800. Within the alone-to-alone spread: no measurable slowdown.
- **Stage edges (`CePipe`, 256 MiB) beside reads:** alone 0.36-0.41 ms (654-740 GB/s), with reads 0.365-0.370 ms
  (725-735 GB/s); reads during the edges 46.9 GB/s per GPU. Neither slows the other.
- **Target gap.** Solo 32k read 2.07 ms (copy floor 1.73 ms: 90 MB at 52 GB/s) against 1.72; the server's 32k read
  is 3.58 ms (job 7955, §9): the rest is messaging, below.
- **4 KiB pages cost nothing (job 7991, 4 KiB / 2 MiB alternated twice, `PPIMPL_TIER_HUGEPAGE=0/1`; THP is
  `madvise` on this host, so 0 really is 4 KiB).** Solo 62-block reads 50.5 / 50.8 GB/s (4 KiB) against 50.7 / 50.2
  (2 MiB); 16 blocks 45.4 / 44.9 against 44.9 / 44.1; four GPUs reading and writing (rwdef, bytes checked) 44.0-44.9
  read / 21.3-21.6 write GB/s per GPU at 62 blocks in every variant. 4 KiB stays the default: MADV_HUGEPAGE made
  the first touch compact memory synchronously (~500 s for 150 GB on a node full of page cache, 7976).

### 2.4 Choices

Taken now, each with the 7916 number that confirms it or switches to the fallback:

| Choice | Why | Confirmed by (7916) | Fallback |
|---|---|---|---|
| Copy engine, one `cuMemcpyBatchAsync` per op per GPU (vLLM `ops.swap_blocks_batch`), 0 SMs | every system that reaches Gen5 line rate without SMs does this; our stages keep all SMs busy | **confirmed**: batch H2D 47.6 GB/s per GPU at 2 MiB pages with 4 GPUs (44.2 at 512 KiB); GEMM 0.677 vs 0.682 ms | - |
| Balanced by quarters, ~~GPU q's CE writes peer HBM over NVLink~~ **two hops** | exact 1/4 per link (1,499 B/token, 36.9 MB) | **switched**: peer_batch H2D 31.7 GB/s per GPU vs 47.6 into own HBM (-33%) | in use: per chunk of slots, one contiguous H2D per slot into a device staging ring on GPU q, then one D2D batch from the ring to the stages' pieces (writes: D2D gather, then D2H per slot); two streams, chunk i+1's H2D overlaps chunk i's D2D |
| page_first host layout (a slot = one block's quarter of all layers and stages, contiguous) | one host run per slot; host side never the small piece | - | - |
| No layer-wise load pipelining | the round computes only after the whole prefix landed (target model); DRAM reads 1.7-16 ms | - | - |
| Host memory on the GPU's NUMA node, registered by `cudaHostRegister` (not torch pin_memory) | torch rounds pins to powers of 2; remote NUMA caps lower | **confirmed**: 4 GPUs from the other node 27.2 / 19.4 GB/s vs 48.4 / 38.5 | preferred-local allocation, remote spill logged |
| SSD: O_DIRECT through one io_uring per GPU, one shard file per member | §2.5: at the same throughput as 8 `preadv` threads, half the CPU and half the p99 read latency under write-back; no GDS on this node (nvidia-fs not loaded) | **confirmed** (§2.5, 4 processes on node 0) | - |

### 2.5 SSD I/O: io_uring (bench/ssd_io.py, 2026-10-17)

`SsdTier` drives its file through `kvpool/uring.py`: one ring per worker (liburing-ffi through ctypes), the file
registered (IOSQE_FIXED_FILE) and every host arena it reads into or writes from (DRAM and staging) registered as
fixed buffers of at most 1 GiB, so I/O is READ_FIXED / WRITE_FIXED and the kernel pins nothing per request. A
request is one slot (cut only where a fixed buffer ends). Queued requests wait in a read queue and a write queue; at
most 8 reads and 4 writes are in the kernel at once, and free read room is filled first, so a write-back never fills
the device's queues in front of a read. One thread owns the ring (submit and wait in one call, GIL released) and
completes each slot's future; callers only queue. No fallback: a ring or a registration that fails stops the worker
at start-up. The file is O_DIRECT (buffered only where the file system has none, as tmpfs in CPU tests).

Benchmark: `bench/ssd_io.py`, the SSD tier's pattern without GPUs. 4 processes (one per GPU) bound to node 0, each
with its own 16 GB file of 12,857,344 B slots (GLM-5.3 PP4 balanced) and a host arena; a reader of whole sessions
(3-60 random slots, one session at a time) and a write-back of 3-60 slots at a time paced to a rate. 20 s per run,
per process (= per GPU):

| Mechanism | Read GB/s (max) + 3 GB/s writes | CPU cores | Read ms, 31-60 slots p50 / p99 | 3-10 slots p99 | Max + 5 GB/s writes: read GB/s, 31-60 p99 | 10 GB/s reads + 3 writes: cores, 31-60 p99 |
|---|---|---|---|---|---|---|
| 8 threads, `preadv` / `pwritev` of whole slots (before) | 15.2-16.1 | 1.11-1.15 | 26-27 / 100-115 | 65-79 | 11.0-11.5, 104-115 | 0.96-1.00, 81-87 |
| **io_uring, whole slots, 8 reads / 4 writes (chosen)** | **16.8-16.9** | **0.49-0.50** | 31-32 / **62-67** | **16-18** | **12.0-12.1, 71-74** | **0.31-0.35, 43-50** |
| io_uring, whole slots, 4 / 2 | 16.3-16.4 | 0.53 | 34-35 / 65-68 | 13-14 | | |
| io_uring, 2 MiB pieces, 32 / 8 | 16.1-16.3 | 0.60-0.62 | 34-35 / 61-67 | 13-16 | 11.3-11.5, 70-72 | 0.37-0.40, 43-56 |
| io_uring, 2 MiB, 16 / 4 | 15.7-15.8 | 0.53 | 34-36 / 62-65 | 12-15 | | |
| io_uring, 1 MiB, 64 / 16 | 16.2-16.3 | 0.62 | 31-33 / 61-62 | 14-17 | | |

- The node's array delivers ~65-67 GB/s of reads to 4 processes with any mechanism at depth >= 8 slots; io_uring
  adds ~5% throughput, halves the CPU (one thread, no per-request syscalls or pinning) and halves the read p99
  under write-back (reads go first, threads share one queue with writes). Writes held their rate in every run.
- Smaller pieces at a larger depth change nothing measurable; a whole slot per request is the fewest requests.
  Depth 8 / 4 is the smallest that keeps the maximum (4 / 2: -3%).
- Need (simulator, victim chain, N = 400-500): SSD reads 0.7-1.8 GB/s per GPU on average (10 s peaks 2.3-4.1), SSD
  writes ~1.1 GB/s: a fifth of what one GPU's share delivers with 5 GB/s of writes beside it.

Note: own-stage storage (each GPU stores and moves its own stage's bytes) is not balanced. At PP4, stages 0, 2 and 3
hold 3 DSA layers each (14.02 MB per block, 1,635 B/token: 9% above the mean of 1,499) and stage 1 holds 2 (9.35 MB,
27% below); the KDA states are 39.1 / 39.1 / 34.7 / 34.7 MB. So three links carry 9% more than the mean and one 27%
less, and the slowest link sets a read's landing time. At PP8 the 11 DSA layers split 2 or 1 per stage (mean 1.375),
so the busiest link carries at least 2 / 1.375 = 45% more than the mean. Balanced storage is exact for every PP degree
and every context; the code takes the PP degree as a parameter (n = `num_stages`), and own-stage stays a placement
option for comparison (§13).

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
  tails. A 512k read is 62 x 22 + 79 = 1,443 device pieces per GPU. Host side contiguous: with two hops (§2.3) the
  host <-> GPU q hop is one 12.86 MB copy per slot (line rate), and the 1,443 pieces are the D2D hop over NVLink.
- **Device staging (two hops).** One ring per direction in GPU q's HBM: ring bytes = 2 directions x depth x chunk,
  chunk = `ring_chunk_slots` x slot bytes. Defaults from the job 7961 sweep (§2.3): depth 2 (double buffering),
  chunk 4 slots (51.4 MB), so 2 x 2 x 51.4 = 206 MB per GPU. Why depth 2 is enough: one slot's host hop takes
  ~270 us (12.86 MB at 48 GB/s), its NVLink hop ~19 us (670 GB/s, job 7943), so with two chunks the host hop never
  waits for the ring; depth 4 measured no faster for reads. A context's state is cut into chunks like its KV, so a
  chunk can be smaller than the state (1 slot measured slower: more driver calls).
- **Which HBM the ring uses.** Ready = 7,144 KV blocks per stage is the contract (`1_contract.md` §1) and is not
  reduced: the controller stops the server if a stage reports fewer with tiers on. The KV pool is sized first
  (`blocks_that_fit`: the HBM free after the weights minus the 13.6 GB activation/workspace reserve, capped at
  7,144); the ring is allocated after the model, from what is left, and each worker prints that amount at start-up
  (`stage s tiers: ... ring ... of X GB free HBM after the model and KV pool`).
- **What it costs.** Block granularity: the last block is stored whole. Mean waste half a block per context per GPU
  (6.4 MB) plus 1.7 MB of state rounding: about 4% of the average session context (~100k tokens); a 32k read moves
  4 x 12.86 + 38.6 = 90 MB per GPU instead of 86 MB (+5%). Recorded as a difference (§7 "a context stored larger").
  The index charges real slots, so tier capacity is exact.

## 4. DRAM

- Per worker at start-up, on a side thread while the weights load: anonymous `mmap` (no page faulted),
  `mbind(MPOL_PREFERRED, gpu_node)`, first touch from 16 threads on the node's CPUs (4 KiB pages), then after the
  model `cudaHostRegister(portable | mapped)` in ~1 GiB chunks of whole slots (§14.1; 7995: touch 5.5-11.8 s hidden
  under the 65 s model load, register 12-13 s). torch's pinned
  allocator rounds to a power of 2 (150 GB -> 256 GB), so it is not used.
- Sizes per GPU: 150 GB (full), 37.5 GB (Tiers), 18.75 GB (SSD-heavy); plus a 2 GiB staging area for SSD-only runs
  (DRAM tier off): SSD reads and writes pass through it, it serves no hits.
- Node: RealMemory 2,047,532 MB, two NUMA nodes of ~1 TB; GPUs 0-3 are on node 0, 4-7 on node 1. A full-memory run
  pins 600 GB on one node. Slurm `--mem`: 800G for full memory (today's session jobs get 400G; worker RSS ~17 GB),
  350G for Tiers, 300G for SSD-heavy. If the node lacks local memory, allocation falls back to the other node
  (preferred, not bound) and the split is logged; remote-node bandwidth is in §2.

## 5. Writes: the victim chain HBM -> DRAM -> SSD

The user's policy (2026-10-17, as the simulator's victim DRAM, ServingStudioSim 4b93b396): a tier is written only
with what the level above lets go of, and only with what it lacks. HBM's kept contexts go to the fastest tier (DRAM,
or SSD when there is no DRAM), DRAM's entries to SSD; SSD evicts into nothing. No write-through: a round's context
reaches DRAM only on its way out of HBM.

- **One write-back policy for both levels** (`kvpool/writeback.py`, `WriteBack`). A level keeps whole entries in LRU
  order, each with a charge, and holds a fixed charge outside them. An entry is dirty until the level below holds it
  whole, writing while its write is in flight, then clean; only a clean entry is evicted without waiting.
  - Ahead of need (HBM; `step`, each scheduler step; user-approved option B, 2026-10-17). Invariant: demand never
    waits for a write. With C capacity, F fixed, D dirty and W writing charge, free + clean = C - F - D - W. A write
    lands within T after it starts (T: the longest write seen, from its start to landing); the writes in flight then
    answer demand of the last T, so free + clean covers the next T's demand if C - F - D >= R, R the largest demand
    seen in any window of 2T. Demand = growth of F + D + W from one step to the next (admissions; contexts the tier
    lost); a landing is not counted as supply. After each step the LRU dirty entries are written until
    C - F - D >= R. T and R are running maxima of what the run measured: no knob (`hbm_write_ahead` /
    `dram_write_ahead` are gone). T starts when the write is sent, after the demotes of the DRAM slots it reuses:
    counted from the issue, a burst of write-backs queued behind demotes raised T to 125-143 ms, R to 760-840 blocks
    and DRAM writes by 60% (`bench.replay writeback`, ssd_heavy); a write that does wait for a demote is the rare
    residual wait.
  - At eviction only (DRAM): SSD takes only what DRAM evicts; a demote of what SSD already holds whole moves
    nothing.
  - On demand (`make_room`): an eviction that lacks clean charge starts the writes of the LRU dirty entries beyond
    those in flight and counts a wait (`evict_waits`, `evict_writes`). When a write lands, the round that did not fit
    at the head of the order starts its tier read at once (`Scheduler.op_done`), not at the next forming.
  - The level below reports a write done (`done`) or that it lost the entry (`lost`: dirty again).
- **HBM level** (`BalancedKVPool.HbmLevel`): entries = the KV manager's kept contexts (charge: their blocks or
  tokens), fixed = running and held requests. A write is `write_back` -> `write` ops of what the tier lacks (from
  the block holding its last token, plus state); the kept context is pinned until all 4 GPUs reported it (its next
  round waits to take it over, `busy_waits`). A request that fits only by evicting contexts not yet written waits
  (`blocked_by_write`) while `make_room` writes them. `fits` does not count a resuming round's own kept context twice
  (in its footprint and as held back).
- **DRAM level** (`TierManager`, `_TierLevel`): entries = DRAM's entries (charge: slots), fixed 0, at eviction only.
  A write is a `demote` (DRAM slots -> SSD slots, one io_uring write per slot). DRAM evicts at once when it needs
  slots: a dirty victim's demote is emitted then, and the op that reuses the slots waits for it through `SlotGuard`
  (`evict_waits`). A member's SSD write thread does not wait for a write's or demote's file writes: many are in
  flight in the io_uring at once (reads first), each reported when its last slot is written.
- **What moves.** Only what the tier below lacks: a context that the tier holds whole (same content) is only
  refreshed there (`hbm_evicted`), a longer one keeps its full blocks' slots. A read back into HBM leaves the tier's
  copy in place (clean). When a session's next round is admitted, its entries that the round does not resume from
  are dropped (`drop_stale`).
- **Counters**, every run: `cache.json` `write_back` = {hbm: {ahead_writes, evict_writes, evict_waits, headroom (R),
  write_ms (T)}, dram: {...}} and `blocked_by_write`, `busy_waits`.
- **Byte mode** (replay): writes take no time, so each level writes at eviction (no write-ahead).
- **Why (2026-10-17, runs tiers_victim_*_n400 with the 0.9 watermarks).** Every admission wait was write-back, none
  real HBM pressure: ssd_heavy 848 rounds waited 202.7 s (p50 130 ms, p99 1.87 s), tiers 195 / 55 s, while the
  HBM -> DRAM path was busy 13% of the time. The watermark bounds F + D only, so free + clean was at most 10% of HBM
  minus the writes in flight (89 blocks in ssd_heavy) against up to 122-149 blocks of demand in one scheduling step;
  the round was then checked again only at the next forming (p50 48 ms later). 537 HBM writes also waited for a
  demote (p50 19 ms). `bench.replay writeback` (the real `KvManager` + `BalancedKVPool` on a run's logged
  admissions, ops timed by the measured write rates; old policy reproduces the measured DRAM / SSD write volume
  within 1%): waits old -> new, write GB per GPU DRAM / SSD:

  | run (N=400) | waits old | waits new | DRAM old -> new | SSD old -> new |
  |---|---|---|---|---|
  | full | 0 | 1 (14 ms) | 226 -> 187 | 154 -> 126 |
  | tiers | 127 (1.0 s) | 13 (0.6 s) | 2127 -> 2184 | 752 -> 702 |
  | ssd_heavy | 284 (2.5 s) | 4 (0.05 s) | 3029 -> 3188 | 1055 -> 1024 |

  The model has no feedback (a wait does not delay later rounds) and no forming delay, so the old policy's waits
  are far shorter there than measured. The new policy's waits all fall while R is still growing (R in tiers 31 ->
  68 -> 357 blocks, ssd_heavy 39 -> 95 -> 326, T 46 ms): a burst larger than any before.
- **SSD shard files, kept across runs** (user, 2026-10-17; `kvpool/shards.py`). Balanced storage gives member q a
  quarter of every slot, so an SSD file is a shard, not a GPU's KV cache: `$PPIMPL_SSD_DIR/set<k>/shard<q>.bin`, q =
  0..n-1, default `PPIMPL_SSD_DIR` = `/raid/kanzhu/ServingStudio/pp-impl/tmp/ssd_tier` (`bench/slurm/m6.sbatch`).
  The leader takes the first set whose `set<k>/lock` it can `flock` (k = 0, 1, ...) and holds it for the run, so two
  concurrent runs (GPUs 0-3 and 4-7) get different sets; member q opens `shard<q>.bin` of that set
  (`tiers.ssd_dir` in the workers' config is the set). A set is created once and explicitly, every shard written in
  full so no O_DIRECT write lands on an unwritten ext4 extent: `python -m ppimpl.kvpool.shards create --sets 1
  --shards 4 --gb 128` (`list` shows the sets and which are in use). A run never creates, resizes or deletes a
  shard: a missing shard or one of another size fails the start-up (`create --force` rewrites). Contents are never
  trusted: the slot index starts empty every run, so a run reads only slots it wrote itself
  (`test_kept_ssd_file_is_never_served_stale`). Disk kept: 128 GB per shard, 512 GB per 4-GPU set (~1 TB for two
  concurrent runs).
- **Worker.** A tier thread builds the pointer lists and copies on a write stream (copy engine only, no SM, no
  wait on the compute stream: a kept context's blocks are final). SSD writes and demotes run on the worker's SSD
  write thread through the io_uring (§2.5).

## 6. Reads (DRAM/SSD -> HBM)

- Scheduler unchanged (`_at_head`): footprint reserved when the read starts, read-wait bound 1 s, the round out of the
  queue, landed first in landing order. Only the blocks HBM lacks (from the block holding HBM's last token) + state.
- **DRAM read.** One `swap_blocks_batch` per GPU on the read stream (H2D; the writes are D2H, the other PCIe
  direction). Reads of one tier on one GPU run one after another (one stream).
- **SSD read.** The io_uring reads the session's SSD slots into staging slots (`staging_gb`, not DRAM: the victim
  chain puts nothing in DRAM that HBM did not let go of), and as each slot lands the tier thread enqueues its H2D:
  read time ~ max(SSD, H2D) + one slot.
- **Landing.** `read_landed` when all 4 `TierDone`s are in. The index's channel clock (when the tier is free, for the
  1 s bound) uses the configured bandwidth and is pushed to the real landing time when a read lands late.
- No layer-wise pipelining: a round computes only after its whole prefix landed (the target's model); reads are
  1.7-16 ms (DRAM) and 9-82 ms (SSD) against 52-57 ms of round-1 compute.

## 7. Eviction

Each tier LRU of whole sessions, a read refreshes. Slots instead of bytes. HBM and DRAM evict only what the level
below holds whole, written ahead (§5); SSD evicts by LRU into nothing (its slots are reused).

## 8. Correctness

- `tests/test_kvpool_bytes_gpu.py` (GPU, 4 stages, real `StageModel`): prefill P = 40,000 (partial last block, chunk
  starts off KDA's grid) in session s, snapshot every KV page and state row of s on every stage, write to DRAM,
  evict s from HBM, fill its blocks with garbage, read back into different block ids: every byte equal. The same
  through SSD (DRAM off). Then round 1 of s (A = 1k) after the tier read vs after an HBM hit: logits bitwise equal
  (same microbatch shapes), token equal.
- `tests/test_kvpool_slots.py`, `test_kvpool_manager.py` (CPU): slot accounting, write-before-evict, takeover wait,
  demote frees DRAM only when done, SSD LRU eviction at a small SSD (slots reused, the evicted session found nowhere
  and computed again), landing order; `test_kvpool_placement.py`: the quarter rule's pointer lists (every byte of
  every range covered exactly once over the 4 GPUs).
- Bring-up: round-1 tokens against the no-cache reference as in M4 (`bench/prefix_check.py`).

## 9. Steps and runs

| Step | Code | Runs (each on a fresh server; SSD shard sets kept, §5) |
|---|---|---|
| 1a | pointer table, quarter rule, DRAM arena, write/read ops, TierDone, scheduler wiring | GPU test §8; tier bring-up DRAM (150 GB, pool 1,234,944 tokens): per-P round-1 TTFT and read time vs §3 |
| 1b | | sessions full N = 400/500 and h25d37/h12d19 (DRAM only, step 1) against the v2-realistic tier targets |
| 2a | SSD file, I/O pool, demote, copy-up, staging | GPU test through SSD; bring-up SSD only (128 GB) |
| 2b | io_uring SSD tier (§2.5), victim chain (§5) | sessions full / tiers / ssd_heavy (§10) at N = 400/500 with SSD 128 GB |

Status (2026-10-17): the code of 1a and 2a is in and passes on CPU (`tests/test_tiers_slots.py`: index, op order,
pins; `tests/test_tier_copy.py`: quarters; `tests/test_tier_worker.py`: four engines' DRAM and SSD round trips,
demote, copy-up, staging; `tests/test_tiers_e2e_cpu.py`: the server with the fake model through DRAM and SSD).
`tests/test_tier_bytes_gpu.py` (§8, DRAM and SSD) passed on GPU: job 7937 (direct) and job 7944 (two hops). Since
the §13 refactor these are `tests/test_kvpool_*.py` (§13.4).

**Step 1a DRAM bring-up (job 7955, 2026-10-17 04:24, load 8.4 at start; two hops, ring depth 2 x chunk 4; DRAM 48
GB per GPU; the bring-up pool of 1,234,944 tokens, so Ready = 145 blocks and every round 1 reads from DRAM).**
Per P, mean of 5 sessions, ms; targets from `tier_targets.json` (`bench/tiers.py --target dram`):

| P | Round-1 TTFT (target) | Read (target) | Read on the GPUs | Send -> submit / done -> landed | TTFT - read vs HBM-hit target |
|---|---|---|---|---|---|
| 32k | 58.24 (54.4) | 3.58 (1.72) | 2.4 (35-40 GB/s per GPU) | 0.3 / 0.4 | 54.7 vs 52.7 |
| 128k | 63.77 (58.4) | 6.69 (4.67) | 5.2 (44-48 GB/s) | 0.4 / 0.4 | 57.1 vs 53.7 |
| 512k | 82.88 (73.9) | 17.71 (16.46) | 16.0 (52 GB/s) | 0.4 / 0.4 | 65.2 vs 57.5 |

- In the server each worker submits its own share, and a 512k read runs at line rate on every GPU (52 GB/s); the
  probe's lower rates (§2.3) came from submitting four engines from one thread.
- The read misses its target at 32k (+1.9 ms): messaging (controller -> worker, worker -> controller) and 2.4 ms of
  copies for 90 MB (two chunks: the first host hop cannot overlap anything, the last NVLink hop is a tail).
- **Where the read time goes (job 7966, lane-m6 b7730cb = main merged + per-op timing; two bring-ups in one
  allocation, load 6-11).** `tier_ops.jsonl` now carries each op's submit time and GPU span (timing events), and
  `bench.tiers` splits a read on the GPU that finished last: sent -> its tier thread took the op (recv), GPU span
  from a mark recorded as submission began (gpu; the submit overlaps it), GPU end -> its TierDone sent (notify:
  the done thread's wake from a blocking event), last TierDone -> landed in the scheduler (land). Mean of 5, ms:

  | P | Read (target) | recv | submit | gpu | notify | land | Round-1 TTFT p50 (target) |
  |---|---|---|---|---|---|---|---|
  | 32k | 3.55 (1.72) | 0.46 | 0.74 | 2.09 | 0.52 | 0.34 | 57.44 (54.4) |
  | 128k | 6.40 (4.67) | 0.46 | 1.41 | 5.00 | 0.39 | 0.36 | 62.18 (58.4) |
  | 512k | 17.97 (16.46) | 0.55 | 3.51 | 16.19 | 0.32 | 0.43 | 81.38 (73.9) |

  The copies are near the floor (32k: 2.09 vs 1.73 ms for 90 MB at 52 GB/s; 512k 16.2 ms = 53.8 GB/s); messaging
  is ~1.3 ms per read at every P (recv + notify + land). The worker's GIL switch interval is not it: 0.5 ms vs
  Python's 5 ms gave the same parts (32k recv 0.46 / 0.34, notify 0.52 / 0.49, land 0.34 / 0.37; read 3.55 /
  3.44 ms) and the same stage compute (32k 19.37 / 18.31 / 18.14 / 18.59 ms vs 19.36 / 18.31 / 18.14 / 18.60), so
  the switch was not kept. Each hop is a thread waking on a blocked wait (zmq poll, blocking CUDA event) on the
  worker's 4 logical CPUs (48 threads). `busy_waits`, `blocked_by_write`, `passed_over` 0 in both runs.
- **Not done (32k read is +1.8 ms over its target, 3.3% of round-1 TTFT, under the 10% bar); options if wanted:**
  (a) the done thread waits on a spinning (non-blocking-sync) event: est. -0.4 ms per read (notify), costs one of the
  worker's 4 logical CPUs while an op is in flight; (b) a GPU-side dependency: the stage workers wait on the read's
  completion through CUDA IPC events before the forward, so the plan goes out without waiting for every TierDone to
  reach the controller: est. -0.9 to -1.2 ms per read (notify + land + the plan's dispatch), a medium change (an IPC
  event pool per engine, exchanged at start-up, the plan naming the events). (c) 2D/3D batched copies (fewer
  descriptors, less submit time): kept as an alternative.
- **Attribution (job 7960: the same bring-up on the full pool, tiers on, every round 1 an HBM hit).** Ready = 7,144
  blocks on every stage with tiers on. Round-1 TTFT p50 on HBM hits: 54.99 / 57.53 / 65.79 ms at 32k / 128k / 512k
  (targets 52.7 / 53.7 / 57.5). Per-stage compute of round 1 is the same with and without the DRAM read (512k:
  21.76 / 19.67 / 21.14 / 21.64 ms vs 21.86 / 19.69 / 21.18 / 21.64), so the tiers do not slow compute, and the DRAM
  run's TTFT is the HBM-hit TTFT plus the read: 58.21 / 63.53 / 82.37 = HBM hit + 3.2 / 6.0 / 16.6 ms. Against this
  engine's HBM hit plus the target read time (56.71 / 62.20 / 82.25 ms) the DRAM tier is +2.6% / +2.1% / +0.1%. The
  rest of the gap to the DRAM target (512k: 82.37 vs 73.9) is the engine's own long-context round 1 (65.79 vs 57.5
  on HBM hits, +14%), an M5 compute gap, not M6.
- **Free HBM with the full pool and the ring (job 7960):** 15.25 / 38.27 / 4.24 / 2.24 GB on stages 0-3 after the
  model, CUDA graphs, the 7,144-block KV pool, before the 0.206 GB ring. Stage 3 has the least room (2.24 GB).
- `busy_waits` and `blocked_by_write` were 0 in 7960 (every write landed before HBM needed its blocks); 7955 predates
  the counters.

**Step 1b, the capacity-squeezed DRAM settings (jobs 7969 h25d37_dram and 7970 h12d19_dram, N = 400; run at the
same time, 7969 on GPUs 0-3 / NUMA 0, 7970 on GPUs 4-7 / NUMA 1).** Both settings are squeezed on purpose and fall
into a metastable collapse in the simulator and here alike: hits drop to 0 and every round recomputes its prompt
(~350 rounds per 2 min). Their r/s depends on the 2-minute window the collapse starts in, so they are compared per
window before the collapse, plus the collapse window (`bench.tiers --setting`; simulator from request_slo.parquet,
prefix_tiers_w0.csv, prefix_cache_event):

| Setting | Rounds per 2 min, sim / ours | Hit share before collapse | DRAM reads per 2 min, sim restores / ours | Collapse window |
|---|---|---|---|---|
| h25d37 | 1667 2796 2765 3417 3457 2404 1647 / 1665 2791 2745 3332 3194 2325 1110 | 0.83-0.91 / 0.83-0.90 | 9 155 226 587 852 796 430 / 10 156 228 566 847 762 357 | min 14-16 / min 12-16 (fewer rounds at 12-14) |
| h12d19 | 1666 2497 1674 1104 371 / 1664 2492 1757 2293 1216 348 | 0.83 0.84 0.79 0.55 / 0.83 0.83 0.79 0.83 0.60 | 78 482 534 330 0 / 83 520 669 1215 453 0 | min 6-8 / min 8-10 |

- h25d37 matches window by window until the collapse (HBM evictions 96-1047 vs 109-1072 per window); the collapse
  starts in the same windows, ours with fewer rounds at minute 12-14: r/s -5%, TTFT p90 +44%.
- h12d19 is +24% r/s because its collapse comes one window later: before it, ours evicts more from HBM (899 / 1077
  vs 758 / 862 per window) and serves more rounds from DRAM. A tipping point: read latency, admission order and
  write landing move it. Not chased further; M6 is judged on full_dram and the *_ssd128 sessions (SLO-meeting).
- Reads under load queue: a read with no other tier op in flight takes 5.9 ms p50 (recv 0.43, notify 0.32 ms, as
  idle), one behind 5-20 other reads (928 MB ahead) 37.9 ms; writes ahead of a read are 0 MB at the median. One
  receive thread submits every op in order (1.8-3.6 ms of submit per read under load) and a read's copy rate is
  ~27 GB/s by its GPU span: split by the per-batch trace from job 7976 on (`tier_trace_s*.jsonl`).

**Step 2a SSD bring-up, victim chain and io_uring (job 8073, lane-m6-ssd ceff3dc; GPUs 4-7, DRAM off, SSD 128 GB
shard set0, staging 8 GB; bring-up pool 1,234,944 tokens).** Per P, mean of 5 sessions, ms (`bench.tiers --target
ssd`):

| P | Round-1 TTFT (target) | SSD read, start -> landed (target) | Arrival -> read start |
|---|---|---|---|
| 32k | 61.6 (61.3) | 7.5 (8.6) | 2.5 |
| 128k | 79.8 (77.1) | 23.1 (23.3) | 2.2 |
| 512k | 189.9 (139.8) | 79.0 (82.3) | 52 |

- The SSD reads meet their targets at every P (io_uring, staging, H2D per slot as it lands).
- 512k misses by +50 ms, all before the read starts: its admission has to evict one kept context that is not yet
  written (`blocked_by_write` 313, HBM `evict_waits` 313, `ahead_writes` 44, `evict_writes` 60). The HBM
  watermark (0.9) keeps ~10% of this small pool clean, and a 512k round needs ~45% of it; in the session settings
  (pool 7.6-61M tokens) a 512k round is under 7% of the pool. (The watermarks were replaced by the measured
  headroom rule, §5.)

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

**Sessions, victim chain, SSD 128 GB per GPU** (user, 2026-10-17; simulator wt-pp-dram-victim
`logs/20261017_7_victim_chain/rows.jsonl`, SSD written only by DRAM evictions, `bench.tiers --setting full|tiers|ssd_heavy`): r/s, new tok/s, TTFT
p50/p90/p99 (s), rounds over 60 s; prompt tokens hit in HBM / DRAM / SSD, minutes 10-20, N = 400.

| Setting (HBM / DRAM GB per GPU) | `--pool-tokens` | N = 400 | N = 500 | Hit tokens N = 400 |
|---|---|---|---|---|
| full (100.18 / 50) | (full pool) | 26.15, 91.0k, 0.16/0.76/6.91, 121 | 28.82, 102.3k, 0.30/2.03/60.84, 708 | 98 / 0.6 / 0.1% |
| tiers (20 / 20) | 12,220,800 | 26.61, 92.2k, 0.19/0.66/4.83, 73 | 28.63, 101.8k, 0.37/2.56/60.86, 682 | 38 / 33 / 28% |
| ssd_heavy (12.5 / 12.5) | 7,632,640 | 26.34, 91.6k, 0.20/0.80/5.06, 89 | 28.04, 100.0k, 0.38/3.16/60.88, 708 | 19 / 29 / 50% |

Step 1 (DRAM only, `tiers/*_dram`): full N = 400 is the same as full_ssd128 (26.10, 90.8k, 0.16/0.76/6.72, 127),
since at full memory an SSD no larger than DRAM holds nothing DRAM does not.

**SSD size differs from the handoff.** The user set 128 GB of SSD per GPU; `7_offloading.md` has 256 GB. At 128 GB the
SSD tier evicts (LRU, finished sessions first: the active set is ~73-87 GB per GPU, notes.md 2026-10-16), so the
runs compare against 128 GB targets, and the SSD eviction path (slots freed, an evicted session computed again) is
part of every tiers / ssd_heavy run and of the CPU tests (§8). Every run uses 128 GB (`TierConfig.ssd_gb` default; user,
2026-10-17); no 256 GB run.

**Memory per run (user, 2026-10-17: reserve, then release).** Only full-memory session runs ask for `--mem=800G` and
pin 600 GB (150 GB per GPU); the job ends, the memory is free. Bring-up: the handoff gives DRAM 150 GB, but the run
writes only ~38 GB per GPU (75 fillers of 256k at 430 MB plus 15 sessions at 5.7 GB), so a 48 GB arena per GPU holds
all of it and nothing is evicted, as at 150 GB: `--mem=300G`. Victim settings: full / tiers / ssd_heavy pin 50 /
20 / 12.5 GB of DRAM plus 8 GB of staging per GPU: `--mem=400G` / `250G` / `220G`.

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
| SSD file I/O | own: one shard file per member (kept across runs, §5), slot-addressed O_DIRECT through one io_uring per GPU (liburing, fixed file and buffers; `kvpool/uring.py`, §2.5) | pattern of Dynamo KVBM's single fallocated O_DIRECT file per worker; io_uring as NIXL's POSIX backend | vLLM fs tier and LMCache/HiCache file backends write one file per block or chunk (temp file + rename, or buffered I/O): 10k+ files per GPU and an open/close per 12.9 MB; `fs_io_C` loops serially in C. GDS (LMCache cuFile, NIXL) is not available (nvidia-fs not loaded) |
| Index, LRU, read-wait bound, scheduler | own (`ppimpl/kvpool/manager.py`, was `engine/tiers.py`; `scheduler.py`, already written for M5 replay) | - | the handoff's rules (whole-session LRU per tier, write-back to SSD, read-wait bound, landed rounds first) are not any of these systems' policies |

## 12. Differences from the target to record

Block-granular slots (§3: +4% space, +5% on a 32k read); real write traffic on the same links (D2H vs H2D, and SSD
reads vs write-back); host time to start and land a read (measured per read in `reads.jsonl`); SSD latency of small reads (a 32k
context is 90 MB per GPU; fio's one-shot reads of that size were dominated by fio's own start-up, so measured in
step 2 inside the engine).

## 13. Interfaces (proposal 2026-10-17, approved; implemented in `ppimpl/kvpool/`, branch lane-m6-kvpool, §13.4)

One public object, `BalancedKVPool`, with five internal layers. Every layer has a small interface and runs on CPU in
tests. Names follow vLLM's offloading API where the concept is the same (`OffloadingManager.lookup` /
`prepare_load` / `prepare_store` / `complete_*`, `OffloadingWorker.transfer_async` / `get_finished`,
`register_kv_caches`), and SGLang HiCache's host pool (`HostKVCache.alloc` / `free` / `available_size`,
`HiCacheStorage` `batch_get` / `batch_set`).

### 13.1 Public API

```python
pool = BalancedKVPool(group, tiers=[HostTier(gb=150), SsdTier(gb=128, dir=...)], hops="two")
pool.register(stage_kv)              # once per process after the KV pool is allocated (collective)
h = pool.store(key, views)           # collective: all n processes, same key
h = pool.load(key, views)            # collective
pool.lookup(key, prompt=None) -> Hit(tier, tokens, wait_ns) | None
pool.close()
```

- **`group`.** The PP group: n stage processes (n = PP degree, any value) plus one leader that holds the index. In
  this engine the leader is the controller (where the scheduler runs). Without a controller, rank 0 is the leader.
- **`views` are block ids, not tensors.** `KVViews(kv_blocks, state_blocks, tokens, hbm_tokens=0)` names this
  process's blocks in the caches it registered. Tensors cannot be passed per call: GPU q writes into every stage's
  KV, so it needs the other processes' device addresses, and opening a CUDA IPC handle takes milliseconds. That is
  why `register` exchanges the handles once, the same split as vLLM's `register_kv_caches` plus per-call block
  ids. Each process passes only its own blocks; the block ids may differ between stages.
- **Collective semantics.** The n calls with the same key form one operation, matched by (key, seq), where seq
  counts that process's calls on the key. Every process must call in the same order per key, as with NCCL. Placement
  is hidden from the caller: GPU q moves 1/n of every process's bytes over its own PCIe link and passes it over
  NVLink to the stage that owns it. Capacity, LRU, DRAM eviction written back to SSD, and slot allocation are also
  hidden.
- **Two call sites, one operation.** A process calls `store` / `load` with its own views. The leader can also post
  the whole collective in one call, `pool.load(key, [views_0, ..., views_{n-1}])`. The member form costs one round
  trip to the leader before the copies start: the views go up, the plan comes back. That is ~0.1 ms over zmq,
  which is 6% of a 32k DRAM read (1.72 ms). This engine uses the leader form:
  - the controller already owns the block ids and decides when a round ends and when a read starts;
  - the workers run only the pool's data plane (their own threads, nothing in the compute loop).

  Both forms build the same plan.
- **Asynchronous.** `store` / `load` return a `Handle` at once and never block on I/O:
  - `h.status`: pending, done or failed;
  - `h.done()` and `h.wait(timeout)`;
  - `h.eta_ns`: for a load, the expected landing time from the tier's read channel;
  - `h.tier`, `h.tokens`, and `h.t_start` / `h.t_land`.

  A load is done when all n GPUs have reported their share, so completion is host-side. A CUDA event can only cover
  one GPU's share; the scheduler admits a round only after it has landed, so it does not need a GPU-side wait.
- **`lookup(key, prompt)`.** Returns the fastest tier that holds the longest context of `key` the prompt can resume
  from. The KDA state exists only at an entry's end, so `tokens` is either the entry's whole length (the prompt
  starts with the entry's content) or no hit. A hit inside an entry is not possible with one KDA state per context
  (`partial_hits` exists only so replay can reproduce the simulator). `wait_ns` is how long a load started now would
  wait for the tier's read channel (reads of one tier run one after another). `lookup` is advisory: `load` checks
  again.
- **The read-wait bound is the scheduler's, not the pool's** (`7_offloading.md`, Read-wait bound). The pool only
  reports `wait_ns`; the scheduler decides. A round whose read would wait more than 1 s is passed over: it stays in
  the queue, holds nothing, later rounds go first, and it is tried again at the next microbatch (today's
  `Scheduler._at_head` returning "passed"). `load` itself has no time limit.
- **Partial transfer.** `load(key, views)` with `views.tokens = T` (the hit) and `views.hbm_tokens = H` (what the
  views already hold from an HBM prefix hit) moves:
  - KV blocks floor(H / B) to ceil(T / B);
  - the whole state.

  `T` must equal the entry's length, otherwise the load fails (that context is not stored). An SSD load reads the
  whole entry from the file
  into DRAM (copy-up) and copies only [H, T) to HBM. `store(key, views)` with `views.tokens = T` replaces the key's
  entry. It writes only the blocks from floor(old / B) on plus the new state, where old is what the entry already
  holds and the new context still starts with. A context that does not extend the entry drops it and writes in full.
- **Errors.**
  - `load` of a key that no tier holds any more (evicted from every tier since the lookup): fails at once, no bytes
    moved, and the caller computes the prefix (`7_offloading.md`: a session found nowhere is computed again). A key
    that moved between tiers since the lookup (DRAM evicted it to SSD) is read from where it is now, with a new ETA.
    This is the only way a load fails besides the faults below.
  - `store` larger than a tier: goes to the next tier, or is not stored there. It never waits for space.
  - An SSD I/O error (`preadv` / `pwritev`) on any GPU: that op fails; the entry is dropped from that tier; a load's
    views hold garbage and the caller recomputes.
  - A CUDA error or a dead member: the group aborts, and every pending handle fails.
- **One key, concurrent ops.** Ops run in call order per key. Slot dependencies keep the bytes right:
  - a store then a load reads the stored bytes (RAW);
  - a load then a store does not overwrite the load's source slots before they are read (WAR). A store writes new
    slots from the last partial block on, so this rarely waits.
  - Eviction while an op is in flight: the slots return to the free list, but reuse waits for the op.

  The caller must not change a store's source views until `h.done()`; the scheduler keeps a kept session's blocks
  pinned until its write lands. The caller must not compute on a load's views before `h.done()`.

### 13.2 Internal layers

| Layer | Class and methods | Side, torch? | Today | Tests |
|---|---|---|---|---|
| 1. `KVLayout` (model-agnostic) | `KVLayout.from_stage_kv(stage, ctx, kv) -> list[ViewRange]`; `ViewRange(stage, name, kind: kv/state, base, stride, nbytes, group)`. Only "which bytes a block / a state owns". | member, reads torch tensors once | `tier_copy.stage_ranges`, `ViewRange` | ranges of the real `StageKV` (GPU test), fake stages (CPU) |
| 2. `Placement` (pure, numpy) | `BalancedPlacement(ranges_all_stages, n, q)`, `OwnStagePlacement(...)` for comparison: `slot_bytes`, `state_slots`, `pieces(views_by_stage, slots, state_slots, b_lo, b_hi) -> (dev, host_off, len)`. A slot = share q of one block of all stages, page_first. | member, no torch | `tier_copy.Layout` (one block-id list for all stages: becomes per stage) | `test_tier_copy.py` at n = 4 and 8 |
| 3. Tier backends | `HostTier`: NUMA-local pinned arena, `capacity_slots`, `ptr(slot)`. `SsdTier`: O_DIRECT file, `capacity_slots`, `read(slots, host_ptrs) -> futures`, `write(slots, host_ptrs) -> futures`. A new medium (CXL, remote) implements the same interface. Slot ids are allocated by layer 5, not here: they must be the same on all n GPUs. | member, no torch | `HostArena`, `SsdFile` | arena, file round trip, unlink |
| 4. `Mover` (one per GPU) | `Mover(device, hops="two" / "direct", ring_depth, ring_chunk_slots)`: `transfer_async(direction, host_ptrs, host_len, dev, ring_off, lens, ready=None) -> event`. Copy engines only, `cuMemcpyBatchAsync`. Two hops: host <-> own HBM ring (PCIe), ring <-> the stages (NVLink); the ring is allocated before the KV pool. Nothing above sees the ring. | member, torch / CUDA (CPU: memmove) | `TierEngine._batch` / `_move` | byte-equal through both hop modes (CPU), GPU byte test |
| 5. `TierManager` (pure policy) | `lookup(key, prompt) -> Hit`, `prepare_load(key, views_by_stage, now) -> (ops, eta)` (None if no tier holds the key), `wait_ns(tier, now)`, `prepare_store(key, views_by_stage, tokens, content) -> ops` (with demotes and drops from eviction), `ready_ops()`, `complete(op_id, ok)`. Per-tier LRU and slot tables, write-back to SSD on DRAM eviction, read channels (the read-wait bound stays in the scheduler). Byte mode (no slots) stays for replay. | leader, no torch | `engine/tiers.Tiers` | `test_tiers.py`, `test_tiers_slots.py` |

Glue: a `PoolMember` per process (today's `TierEngine` without the copy code) receives plans, maps slots to host
addresses through the backends, asks `Placement` for pieces, runs `Mover` and the SSD I/O, and reports `TierDone`.
The facade `BalancedKVPool` holds a `TierManager` on the leader and a `PoolMember` on each member. The scheduler
sees only `lookup` (with `wait_ns`), `load` -> `Handle` (with `eta_ns`), `store`, and the handle's completion. It
never sees shares, slots or the ring.

Proposed files: `ppimpl/kvpool/{__init__ (BalancedKVPool, KVViews, Hit, Handle), layout.py, placement.py,
backends.py, mover.py, manager.py, member.py}`; `engine/tiers.py`, `runtime/tier_copy.py` and
`runtime/tier_worker.py` go away. All existing tests move to the new layers, with no compatibility shims.

### 13.3 Data flow

- **DRAM-hit load.**
  1. The scheduler calls `lookup` and reserves the HBM blocks, then calls `pool.load(key, views)`.
  2. `TierManager.prepare_load` touches the LRU and emits a read op (slots, state slots, deps, eta).
  3. Once its deps are done, the op goes to all n members.
  4. Member q: `Placement.pieces` for its share; `Mover` copies one contiguous slot H2D into the ring, then
     batch-copies D2D from the ring to every stage's blocks; then reports done.
  5. When all n are done, `complete` runs and the handle is done. The scheduler admits landed rounds first.
- **Store.**
  1. At the end of a round, `pool.store(key, views)`.
  2. `prepare_store` keeps the entry's full blocks and allocates slots for the rest plus the state.
  3. If DRAM is full: the LRU victim gets a demote op, emitted before the write that reuses its slots.
  4. Member q: `Mover` batch-copies D2D from the stages to the ring, then D2H per slot.
  5. When all n are done, the caller's blocks are unpinned.
- **SSD write-back (demote).** Member q: `SsdTier.write(slots, DRAM slot pointers)` through its io_uring. When all n
  are done, the DRAM slots return to the free list.
- **SSD load.** `prepare_load` allocates staging slots (as built: never DRAM, §6). Member q:
  1. `SsdTier.read` of all the entry's slots into those host slots;
  2. as each chunk of slots lands, `Mover` copies its blocks in [b_lo, b_hi) to HBM;
  3. the state goes last;
  4. done.

### 13.4 As implemented (lane-m6-kvpool)

Files: `ppimpl/kvpool/__init__.py` (`BalancedKVPool`, `Handle`, `Hit`, `KVViews`), `layout.py` (`ViewRange`,
`from_stage_kv`; was `runtime/tier_copy.stage_ranges`), `placement.py` (`BalancedPlacement`; was `tier_copy.Layout`),
`backends.py` (`HostTier`, `SsdTier`, the NUMA/OOM helpers; was `runtime/tier_worker.HostArena`, `SsdFile`),
`mover.py` (`Mover`, `Segment`; was `TierEngine._batch` / `_move`), `manager.py` (`TierManager`, `TierStore`,
`SlotGuard`, `TierOp`; was `engine/tiers.Tiers`), `member.py` (`PoolMember`, `share_ranges`, `tier_move`; was
`TierEngine` and `controller.tier_move`). Tests: `tests/test_kvpool_{layout,placement,backends,mover,member,manager,
slots,e2e_cpu,bytes_gpu}.py`.

Behavior is unchanged: the slot-mode op stream (every emitted op with its slots, deps, blocks and landing time, the
ready order and the summaries) is identical to the old index's over 16 random runs of 400 steps (12,855 ops), and
`bench.replay` (sessions N = 400, 3 min, `--tiers` full / tiers / ssd_heavy / ssd_heavy with partial hits and
every-context writes, and `bringup tier`) writes byte-identical outputs. Log files and fields, the worker start-up
line (`resident GB by node`, `REMOTE SPILL`) and the OOM guards are the same code.

Where it differs from §13.1-§13.2, and why:
- **Leader form only.** `BalancedKVPool(cfg, partial_hits)` lives on the controller (in the scheduler). No member
  form (a process calling `store` / `load` with its own views), no `group` / `register` / `close` on the facade: the
  workers build a `PoolMember` directly and `PoolMember.attach` is `register` (CUDA IPC exchange). This engine has
  no use for the rest.
- **Calls.** `lookup(key, prompt, now_ns)` takes the caller's clock (the scheduler is a pure state machine);
  `load(key, hit, views, now_ns, req_id)` takes the `Hit` and checks only that its tier still holds the key, instead
  of comparing the prompt's content again on the control thread; `store(key, prompt, views)`
  takes the request for its tokens, content and declared prefix (replay). `Handle`: `status` pending / done,
  `done()`, `eta_ns`, `tier`, `tokens`, `t_start`, `t_land`; no `wait` and no `failed` (a member's error still fails
  the run, `WorkerError`), so `TierManager.complete(op_id, now_ns)` has no `ok`. It moves a read's channel to the
  real landing time (was `Tiers.landed_late`). A store's handle covers its writes from HBM; the scheduler pins the
  kept context once per pending store (was once per write op: the same pinned / not pinned at every point).
- **Layer 1** is a function, `layout.from_stage_kv`; no class around it.
- **Layer 2.** `pieces` takes one block list for every stage or one per stage; the wire (`TierMove.kv_blocks`,
  `state_blocks`) carries one list for all stages, as this engine has one allocation for all.
  `OwnStagePlacement` is not built: it needs per-GPU slot sizes in the index.
- **Layer 3.** `HostTier.ptr(slot)` once `register(slot_bytes)` fixed the slot size; slot counts stay
  `backends.num_slots(gb, slot_bytes)` (no `capacity_slots` attribute). `SsdTier(path, nbytes, slot_bytes,
  buffers)` opens its shard of the run's set (`shards.py`, §5) and owns its io_uring (`uring.UringFile`, §2.5);
  `buffers` are the host arenas to register.
- **Layer 4.** `Mover(device, slot_bytes, hops, ring_depth, ring_chunk_slots)`, `transfer_async(direction,
  segments, ready, tr) -> (event, bytes)`. A segment is host slot pointers plus the pieces in them; the mover cuts
  each segment into ring chunks (KV and state never share a chunk, as before) or, direct, puts all pieces in one
  batch. The ring is allocated after the KV pool, as §3 says (the §13.2 table's "before" was wrong).
- **Byte mode** (replay): no members; a store's handle is done at once, a load's lands at `eta_ns` on the caller's
  clock (replay pushes its landing at `Request.read_land_ns`).

### 13.5 Victim chain and write-back (lane-m6-ssd)

Tiering is invisible to the scheduler and KV manager beyond one hook object. The scheduler no longer writes: it hands
the KV manager the pool's HBM level once (`kv.lower = pool.hbm_level(kv)`, sessions kept only) and otherwise sees
`lookup` / `load` / `ready_ops` / `complete` as before. `store` and `TierManager.prepare_store` are gone.

```python
# KvManager -> the level below (BalancedKVPool.HbmLevel); None without tiers
lower.kept(key)                      # a kept context appeared or changed (release): dirty
lower.unkept(key)                    # it was taken over by its next round
lower.evicted(key, kept)             # HBM evicted it -> TierManager.hbm_evicted (refresh below, or byte mode: write)
lower.admitted(key, req)             # a round admitted -> TierManager.drop_stale
lower.evictable(key) -> bool         # clean: the tier below holds it whole
lower.held_back() -> float           # charge not evictable now (dirty or writing)
lower.make_room(short: float)        # a request fits only by evicting unwritten contexts: write them (it waits)

# BalancedKVPool (leader)
pool.hbm_level(kv) -> HbmLevel
pool.write_back(key, tokens, content, views: KVViews) -> Handle   # HbmLevel.write: ops to the fastest tier
pool.ready_ops() -> list[TierOp]     # HBM write-ahead, then ops whose deps are done (a write's start: T's start)
pool.complete(op_id, now_ns) -> (TierOp, Handle | None)           # a write's landing unpins and marks it clean
pool.write_back_stats() -> {"hbm": {...}, "dram": {...}}          # into cache.json

# TierManager (layer 5)
prepare_write_back(key, tokens, content, views=None) -> list[TierOp]
hbm_evicted(key, tokens, content) -> list[TierOp]
drop_stale(key, req) -> list[TierOp]
take_gone() -> list                  # keys the fastest tier lost -> HbmLevel.lost (dirty again)

# writeback.py: one class, two levels
class Level(Protocol):               # HbmLevel (HBM's kept contexts), _TierLevel (DRAM's entries)
    capacity: float
    def fixed(self) -> float: ...    # HBM: running requests; DRAM: 0
    def lru(self) -> Iterable: ...   # keys, least recently used first
    def write(self, key, entry=None) -> object | None: ...   # start the write below: a token, None if nothing moved
WriteBack(name, level, ahead, clock) # add, remove, done, lost, evictable, held_back, step, make_room, evicting
                                     # ahead: HBM True (headroom R(2T)), DRAM False (at eviction); clock: replay's
```

`HbmLevel` and `_TierLevel` are the only per-level code: what an entry is and how a write is issued (HBM: `write`
ops from the KV blocks, pinned until they land; DRAM: a `demote`, nothing if SSD holds the entry whole). The dirty /
writing / clean state, the headroom rule and the wait counters are `WriteBack`'s. HBM waits by not admitting the request; DRAM evicts at once and orders the slot
reuse after the demote (`evicting`).

cudaHostRegister stays in `attach` (`HostTier.register`: ~1 GiB chunks of whole slots from 4 threads): its chunks
must be whole slots, and the slot size is known only once the model is loaded, so it cannot run on the touch thread
while the model loads. The io_uring's buffer registration comes after it (its time is in the worker's start-up line).

### 13.6 Exclusive DRAM and staged loading (lane-m6-staged)

The real-engine counterpart of ServingStudioSim pp-staged (`prefix_fetch.rs`, `prefix_tiers.rs`, `dram_tier_writes:
hbm-evictions-exclusive`), k = 1 (one KDA state per entry).

- **Exclusive DRAM** (`tiers.dram_writes: hbm-evictions-exclusive`): a DRAM read releases its entry (`_release_read`;
  the slots' next writer waits for the read by `deps`); a round admitted on HBM's kept context releases the copy
  write-ahead made of it (`taken_over`). SSD stays inclusive (a read leaves its entry). An entry stored where the tier
  below holds it whole is clean.
- **Staged loading** (`tiers.loading_read_gb_s`, `loading_prefill_tok_s`): pending -> loading -> ready -> batch
  (`engine/scheduler.py` "Staged loading").
  - HBM's loading area lives in `KvManager` (`area`, `area_held`, `reads`): a set number of blocks counted held
    whether used or not (`hard + kept + max(area_held, area) <= capacity`, in `fits` and `_make_room`). A read holds
    its context's blocks there (`hold_read`) until admission takes them over (`allocate`).
  - DRAM's area lives in the DRAM `TierStore` (`area`, `area_used`): the same fraction of its slots, out of what
    entries may use. An SSD read is a `fetch` op (file -> area slots, reported when its last slot is in DRAM), then a
    `hop` (a DRAM read from those slots into HBM's area) once HBM's area has room, in fetch landing order; DRAM's
    area goes back when the hop lands. A full DRAM area stops the SSD queue (no direct SSD -> HBM bypass).
  - `Scheduler.load` walks the overdue queue, then the policy order; each area's queue stops at its first read that
    does not fit (a read larger than the area starts when it is empty). Admission takes landed reads first (overdue
    ones also in the force-schedule pass), then the queues, passing over rounds that still need a read. Rounds found
    ready in HBM and later read are counted (`rereads`).
  - Size (Little's law, `BalancedKVPool.loading_area_blocks`): area = read rate into HBM x one batch period
    (mb_tokens / prefill tokens/s). `bench/loading_area.py` measures both on a run of the setting: Fix C runs give
    ssd_heavy 3.30 GB/s x 0.306 s = 79 blocks (8.9% of 890), Tiers 2.47 GB/s x 0.314 s = 61 blocks (4.3% of 1,425).
  - **Eviction stays ahead** (the lesson of Fix D, where every eviction waited for its write: blocked_by_write
    44k): each level's write-back keeps the LRU entries holding `reserve` written (`WriteBack.reserve`, replacing the
    0.9 watermark): fixed + dirty <= capacity - reserve, fixed counting the loading area. The reserve is what one
    batch period frees: the area (the reads admission takes over) plus one microbatch's footprint
    (`blocks_for(mb_tokens)`); DRAM's the same in slot bytes. Admission evicts written contexts and does not wait for
    a write as long as a write lands within one batch period.

## 14. Hand-off (2026-10-17)

### 14.1 Host-memory OOM prevention (commit 0e5bc04)

**What happened (7996, full_dram N=500).** Three stages bound 150 GB arenas each to node 0 with
`numa_alloc_onnode` (strict MPOL_BIND). Another user's job held ~394 GB of shmem on node 0, so the node could not
hold 450 GB more; a strict bind does not fall back, and the kernel ran a global OOM
(`constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0`). It killed that job's vLLM worker (job 7999), then other users'
dbus/systemd/bash processes, then our workers. Host MemAvailable was 1,331 GB at precheck: the check was host-wide,
not per node. Not a cgroup OOM (MaxRSS 482 GB of 800G).

**Changes.**
- **Preferred, not bound.** `HostArena` (DRAM tier and SSD staging) and `bench/tier_bw.py` map anonymous memory
  with no page faulted in (`anon_map`), then `mbind(MPOL_PREFERRED, gpu_node)` (`prefer_node`), then touch.
  libnuma's `numa_alloc` faults every page under the current policy, so a policy set after it never applies
  (checked: `prefer:0` with every page on node 1). A short node now spills to the other node instead of OOM.
- **Remote spill is flagged.** The start-up line's `resident GB by node {..}` adds `REMOTE SPILL x GB off node n`
  above 1 GB.
- **OOM-first.** Every process that makes an arena sets `oom_score_adj` 1000 (no privilege needed), so an OOM takes
  our process before anyone else's.
- **Per-node precheck** (`bench/slurm/memguard.py precheck`, run by m6.sbatch). For each NUMA node of the job's GPUs
  (nvidia-smi bus id to sysfs `numa_node`), MemFree + file LRU pages (Active(file) + Inactive(file)) must exceed
  that node's arenas + 50 GB. Other users' shmem and mlocked memory are not counted as available. This is on top of
  host MemAvailable > pinned + 100 GB and PSI full avg10 < 5%. A failed check exits 0 with the reason.
- **Per-node watchdog** (`memguard watch`). From start-up on (not only the timed part), any of those nodes under
  50 GB of MemFree + file pages marks the run INVALID and kills it. The PSI > 20% for 30 s and D-state > 10 s rules
  still apply from t0.
- **cgroup memory.peak** is printed on the start-up line.

**Call sites audited.**

| Call site | Result |
| --- | --- |
| `HostArena` (DRAM) | Changed. |
| `HostArena` (SSD staging) | Changed: same class, same path. |
| `bench/tier_bw.py host_buffer` and `register` | Changed: the two strict binds. |
| Ring | HBM (`torch.empty` on the GPU); not host memory. |
| `cudaHostRegister` | Only `HostArena.register` and `tier_bw`; both now register preferred memory. |
| torch `pin_memory=True` (`model/metadata.py`, `model/fake.py`, `bench/nodecheck/h2d_bw.py`) | Small buffers with the default (local, unbound) policy. |

No `numa_alloc_onnode`, `MPOL_BIND` or `membind` is left in the lane.

**Verified.**
- CPU suite at 0e5bc04: 236 passed, 1 xfailed. New tests:
  - preferred placement follows the policy on each node, not the touching CPU;
  - `oom_score_adj` reads 1000;
  - cgroup peak is readable;
  - per-node precheck refuses 3 x 150 GB on a node with 400 GB and accepts 520 GB;
  - the watchdog stops a run on a short node before t0, even before the run dir exists.
- Outside the lane (scratch script, own process):
  - 10 GB arenas preferred to node 0 and to node 1 landed fully on each;
  - a 150 GB arena on node 0 touched in 4.9 s, all on node 0, PSI 0;
  - the real precheck for N=500 on GPUs (0,0,0,1) at 12:5x: node 0 738 GB (needs > 500), node 1 932 GB (needs > 200), start.

**Not verified yet.**
- The small-job test the user asked for has not run (8035 was submitted and cancelled at hand-off). It covers:
  - precheck refusing with a raised node threshold, inside a job;
  - a preferred 10 GB arena on a node whose MemFree we filled with our own preferred filler, landing on the other
    node without an OOM, then `cudaHostRegister` on it.

  The script is ready in the scratchpad (`oomcheck.py` and `oomcheck.sbatch`, 1 GPU, priority, 10 min, 300G). It
  fills the node with less MemFree, refuses to run unless the other node has > filler + 200 GB, and aborts on
  PSI > 5% or the other node under 150 GB. Copy it into the lane before use. Caveat: with
  `zone_reclaim_mode` 0 the spill happens at the node's low watermark, but kswapd also reclaims that node's page
  cache, so part of the arena may still land locally; record the split.
- The spill path at full scale (150 GB arenas on a short node) and the per-node watchdog in a real job.

### 14.2 Next, in order

1. Coordinator review of 0e5bc04: done 13:20 PDT 2026-10-17, no change needed.
   - `mbind` mask/maxnode, `anon_map`/`anon_unmap`, `oom_score_adj` in the arena-making workers, and the per-node
     sums are right. Shmem sits on the anon LRU, so `Active(file) + Inactive(file)` does not count other users'
     shmem as available.
   - The 50 GB per-node watchdog trips before a real spill: the kernel spills only at the node's low watermark, far
     under 50 GB. So with the precheck, a node that runs short stops our run; spill is the backstop, not the plan.
   - The small-job filler test of 14.1 is not run, by choice. Filling a node's MemFree to its watermark on this
     shared host makes the kernel reclaim other users' page cache on that node and pushes their allocations
     remote. That is the kind of interference the fix exists to prevent. MPOL_PREFERRED's fallback is kernel
     semantics, and the no-filler checks above cover our side. The scripts stay in coord/ (pp-impl orchestrator,
     untracked) if the user still wants it on an otherwise idle node.
2. Pre-submit check, posted to the coordinator before submitting:
   - per node, MemFree + file pages;
   - other users' shmem and mlock holders per node (cgroup `memory.stat`, read only);
   - PSI;
   - the precheck verdict.

   Then rerun full_dram N=500: `--mem=800G`, main with nice -500, `PPIMPL_TIER_TRACE=0`. Watch start-up and the
   resident-by-node split; stop and report at once if the watchdog fires. (N=400 passed as 7995.)
3. §13 refactor (user-approved): implement under `ppimpl/kvpool/`, delete the three old files, migrate every test,
   no compatibility shim. Done on lane-m6-kvpool (§13.4); GPU byte test, DRAM bring-up and full_dram N=400 not
   rerun yet.
4. CPU suite, DRAM bring-up, full_dram N=400 recheck.
5. SSD bring-up and the 128 GB SSD sessions (record df, exercise eviction; SSD shard sets kept, §5).

### 14.3 Recorded only, not to implement

- **Round-1 long-context host overhead.**
- **C++ Mover:** one GIL-free call per op. Estimated read p50 -1.5 to 2 ms (§9, 7979/7980); notify is unaffected.
- **Read notify path:** done-thread wake, GIL and send, 4.0-4.9 ms p50 under load against 0.41 ms idle. Options
  (a) spin event and (b) IPC-event dependency are in §9; a separate tier process per GPU would also remove it.


### 14.4 Page-cache reclaim: bind each checked piece (2026-10-17, after 8039)

- **8039** (full_dram N=500 on GPUs 4-7, NUMA 1): precheck passed on node 1 free + file 919 GB, but node 1 had ~240 GB
  free and ~678 GB page cache. A preferred fault does not reclaim (`zone_reclaim_mode` 0): once node 1 reached its
  low watermark the pages went to node 0, which had free memory. 365 of 600 GB landed on node 0 (REMOTE SPILL
  83-100 GB per arena); node 1's page cache fell only 678 -> 662 GB. Cancelled after start-up, nothing timed.
- **Change (user: reclaim the cache, there is no other way to get the node).** `HostTier.touch` binds each
  1 GiB piece (`MPOL_BIND`) to the GPU's node right before touching it, after checking that the node's free plus
  clean file pages (less Dirty and Writeback; shmem and mlocked are not on the file LRU) exceed the piece plus
  `TOUCH_RESERVE_GB` = 114 GB (50 GB headroom + 4 arenas x 16 threads x 1 GiB in flight). A bound fault makes the
  kernel reclaim that node's page cache. A short node fails the touch (`NodeShort`, raised from `attach`): no spill,
  no forced allocation. Other users lose only clean cached file pages (re-read from disk), never anonymous or shmem
  memory. memguard precheck counts the same clean file pages.
- 7996 differs: there 3 x 150 GB were bound at once to a node whose memory was another job's shmem, which cannot be
  reclaimed, with no check per piece.
