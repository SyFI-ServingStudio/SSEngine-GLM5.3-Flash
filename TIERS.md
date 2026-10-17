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
**The engine's two-hop path in one process (job 7951, `bench/tier_hop2.py`, 4 GPUs on NUMA node 0, load 10).** Fake KV
laid out like GLM-5.3 PP4, four `TierEngine`s driven from one Python thread (their submissions run one after
another; in the server each worker submits its own share in parallel). Bytes equal for every setting below. Per op
on all 4 GPUs, p50 of 10, ms; "submit" is the host time to submit the op on all 4 engines:

| Ring | 32k read / write | 128k read / write | 512k read / write | 512k submit |
|---|---|---|---|---|
| direct (pieces host <-> peers) | 7.43 / 7.27 | 20.6 / 20.6 | 71.0 / 71.5 | 70.0 |
| depth 2, chunk 1 | 4.36 / 5.88 | 10.9 / 17.3 | 38.1 / 61.1 | 28.2 |
| depth 2, chunk 3 | 3.69 / 6.15 | 8.77 / 17.4 | 27.7 / 59.6 | 16.2 |
| depth 2, chunk 4 | 3.37 / 5.06 | 8.25 / 15.6 | 25.6 / 59.9 | 11.7 |
| depth 4, chunk 1 | 4.85 / 5.00 | 12.0 / 15.8 | 35.3 / 59.7 | 24.8 |
| depth 4, chunk 3 | 3.81 / 4.86 | 8.61 / 13.3 | 27.2 / 55.5 | 15.8 |
| depth 4, chunk 4 | 3.53 / 5.19 | 8.18 / 13.0 | 26.1 / 54.5 | 13.5 |

`cuMemcpyBatchAsync` by piece size, GPU 0's stream, 512 MB in all (GB/s; host submit included):

| Piece | local | push 0 -> 1 | pull 1 -> 0 | host -> 0 | 4 GPUs push to all 4 / pull from all 4 (per GPU, submitted one after another) |
|---|---|---|---|---|---|
| 128 KiB | 30.5 | 125 | 127 | 42.8 | 9.4 / 3.8 |
| 1 MiB | 240 | 457 | 455 | 53.5 | 84 / 25 |
| 4 MiB | 903 | 639 | 634 | 54.9 | 249 / 80 |
| 12.86 MB | 1,396 | 708 | 708 | 56.3 | 419 / 144 |

- **Host submission is the read's main cost.** Direct mode is all submission (70.0 of 71.0 ms: ~12 us per peer
  piece on the host). Two hops with chunk 4: 11.7 ms of the 25.6 ms 512k read is the host submitting 4 engines one
  after another, ~2.9 ms per engine. Fewer, larger chunks submit faster, hence chunk 4. The server's DRAM bring-up
  measures the real per-worker read time (each worker submits in parallel).
- **Writes gather by pulling over NVLink, which is slow with 4 GPUs at once** (pull 144 vs push 419 GB/s per GPU at
  12.86 MB, 25 vs 84 at 1 MiB): 512k writes take 55-60 ms (~15 GB/s per GPU). Writes are background (1.1-1.7 GB/s
  needed per GPU on average, §5), so this is not a target miss by itself; it lengthens how long a written session's
  blocks stay pinned. The fix, if needed: each stage's copy engine pushes its pieces into the owner's ring.
- **Target gap.** 32k read 3.37 ms in the probe against 1.72 (balanced DRAM target), 512k 25.6 against 16.46. Both
  include the serialized submission above; the bring-up decides.

- Still to measure, in the first DRAM bring-up job (no separate job): the GEMM slowdown with the real
  `cuMemcpyBatchAsync` pipeline at the real rate, and the second hop beside the stage edges' `CePipe` sends (both on
  NVLink: does either slow the other).

### 2.4 Choices

Taken now, each with the 7916 number that confirms it or switches to the fallback:

| Choice | Why | Confirmed by (7916) | Fallback |
|---|---|---|---|
| Copy engine, one `cuMemcpyBatchAsync` per op per GPU (vLLM `ops.swap_blocks_batch`), 0 SMs | every system that reaches Gen5 line rate without SMs does this; our stages keep all SMs busy | **confirmed**: batch H2D 47.6 GB/s per GPU at 2 MiB pages with 4 GPUs (44.2 at 512 KiB); GEMM 0.677 vs 0.682 ms | - |
| Balanced by quarters, ~~GPU q's CE writes peer HBM over NVLink~~ **two hops** | exact 1/4 per link (1,499 B/token, 36.9 MB) | **switched**: peer_batch H2D 31.7 GB/s per GPU vs 47.6 into own HBM (-33%) | in use: per chunk of slots, one contiguous H2D per slot into a device staging ring on GPU q, then one D2D batch from the ring to the stages' pieces (writes: D2D gather, then D2H per slot); two streams, chunk i+1's H2D overlaps chunk i's D2D |
| page_first host layout (a slot = one block's quarter of all layers and stages, contiguous) | one host run per slot; host side never the small piece | - | - |
| No layer-wise load pipelining | the round computes only after the whole prefix landed (target model); DRAM reads 1.7-16 ms | - | - |
| Host memory on the GPU's NUMA node, registered by `cudaHostRegister` (not torch pin_memory) | torch rounds pins to powers of 2; remote NUMA caps lower | **confirmed**: 4 GPUs from the other node 27.2 / 19.4 GB/s vs 48.4 / 38.5 | preferred-local allocation, remote spill logged |
| SSD: O_DIRECT `preadv`/`pwritev`, 8 threads per GPU, one preallocated file per GPU | fio: 8 psync threads give 53 GB/s on one file; no GDS on this node (nvidia-fs not loaded; it would also need DRAM copy-up anyway) | per-read latency measured at step 2 | io_uring via liburing if threads miss 10 GB/s per GPU |

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
  chunk = `ring_chunk_slots` x slot bytes. Defaults from the job 7951 sweep (§2.3): depth 2 (double buffering),
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

Status (2026-10-17): the code of 1a and 2a is in and passes on CPU (`tests/test_tiers_slots.py`: index, op order,
pins; `tests/test_tier_copy.py`: quarters; `tests/test_tier_worker.py`: four engines' DRAM and SSD round trips,
demote, copy-up, staging; `tests/test_tiers_e2e_cpu.py`: the server with the fake model through DRAM and SSD).
`tests/test_tier_bytes_gpu.py` (§8, DRAM and SSD) has not run yet; no GPU run before the design is confirmed.

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

## 13. Interfaces (proposal for review, 2026-10-17; the refactor waits for approval)

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
- **SSD write-back (demote).** Member q: `SsdTier.write(slots, DRAM slot pointers)` with 8 I/O threads. When all n
  are done, the DRAM slots return to the free list.
- **SSD load.** `prepare_load` allocates DRAM slots for the copy-up (or staging slots). Member q:
  1. `SsdTier.read` of all the entry's slots into those host slots;
  2. as each chunk of slots lands, `Mover` copies its blocks in [b_lo, b_hi) to HBM;
  3. the state goes last;
  4. done.
