# KV tiers: DRAM and SSD behind HBM

The prefix cache keeps whole sessions in HBM. With `--dram-gb` and `--ssd-gb`, contexts that HBM lets go of move to
pinned host DRAM and then to NVMe SSD, and are read back when the session's next round arrives. The code is in
[ppimpl/kvpool/](../../ppimpl/kvpool/). Measured results: [../results/eval_m6.md](../results/eval_m6.md) and
[../results/ssd_backend_ab.md](../results/ssd_backend_ab.md).

## Bytes per context

The HBM layout is vLLM's: one tensor per layer, block-major, 8,576 tokens per block.

| Range | Size |
|---|---|
| KV page, per DSA layer | MLA page + indexer page of one block: 8,576 × 545 B = 4.67 MB. Stages 0, 2, 3 have 3 DSA layers, stage 1 has 2 |
| KDA state, per KDA layer | recurrent fp32 [64, 128, 128] (4 MiB) + conv bf16 (144 KiB) = 4.34 MB; 9 / 9 / 8 / 8 layers per stage |

## Balanced storage

Every byte range a context owns (each KV page of each DSA layer of each stage, each KDA state) is cut into 4 equal
contiguous quarters, and GPU q stores and moves quarter q of every range. Each GPU therefore holds and moves exactly a
quarter of every context, 12.86 MB per KV block and 36.9 MB of state, whatever the stage split.

- **Slots.** A tier slot holds GPU q's quarter of one KV block of all 4 stages, packed stage by stage and layer by
  layer, rounded up to 4 KiB for O_DIRECT. A context of P tokens takes ceil(P / 8,576) KV slots plus 3 state slots.
  The slot ids are the same on every GPU: the controller's index allocates them, the workers only execute.
- **Two hops.** The host ↔ GPU q copy is one contiguous slot copy at line rate. The pieces that belong to the other
  stages move between GPU q's staging ring and their own GPU over NVLink. A direct copy from host to another GPU's HBM
  measured a third slower.
- **Copy engine only.** Each tier op is one batched copy (`cuMemcpyBatchAsync`) per GPU; no SMs are used.
- **Placement** is in [placement.py](../../ppimpl/kvpool/placement.py) and [layout.py](../../ppimpl/kvpool/layout.py);
  the byte path in [mover.py](../../ppimpl/kvpool/mover.py) and [member.py](../../ppimpl/kvpool/member.py).

## DRAM

Per GPU, on that GPU's NUMA node ([backends.py](../../ppimpl/kvpool/backends.py) `HostTier`): an anonymous mapping,
`mbind(MPOL_PREFERRED)` to the GPU's node, first touch from 16 threads while the weights load (checking free memory
per 1 GiB so the host does not run out), then `cudaHostRegister` in ~1 GiB chunks. torch's pinned allocator is not
used: it rounds to a power of two.

## SSD

One preallocated shard file per GPU, kept across runs ([shards.py](../../ppimpl/kvpool/shards.py)):
`<ssd-dir>/set<k>/shard<q>.bin` holds GPU q's quarter of every slot. A server locks the first free set for as long as
it runs. Shard contents are never trusted: the slot index starts empty each run.

I/O is O_DIRECT, one request per slot, at most 8 reads and 4 writes in the kernel per GPU, reads first so a
write-back never queues in front of a read. Two backends (`tiers.ssd_backend`), equal in an A/B
([../results/ssd_backend_ab.md](../results/ssd_backend_ab.md)):

- `aio` (default, [aio.py](../../ppimpl/kvpool/aio.py)): Linux native AIO through raw syscalls, one context and four
  submit threads per GPU. Runs under Docker's default seccomp profile.
- `uring` ([uring.py](../../ppimpl/kvpool/uring.py)): one io_uring per GPU, with the file and the host arenas
  registered as fixed buffers. Half the CPU of AIO; Docker's default profile denies it.

The one named is set up or the start fails; nothing falls back to the other.

## Writes: the victim chain

A tier is written only with what the level above evicts: HBM's evicted contexts go to DRAM (or to SSD without DRAM),
DRAM's evictions go to SSD, and SSD evicts into nothing. Only what the level below lacks is written.

- **Exclusive DRAM** (the served setting, `tiers.dram_writes: hbm-evictions-exclusive`): HBM writes a context to DRAM
  at its eviction and frees the blocks at once; the blocks' next users wait for the copy. A DRAM read releases the
  DRAM entry. SSD stays inclusive.
- One write-back class ([writeback.py](../../ppimpl/kvpool/writeback.py)) serves both levels. An entry is dirty until
  the level below holds it whole, and only clean entries are evicted without waiting.
- The index is [manager.py](../../ppimpl/kvpool/manager.py) (`TierManager`); [../../ppimpl/kvpool/__init__.py](../../ppimpl/kvpool/__init__.py)
  is the facade the scheduler uses (`BalancedKVPool`).

## Reads and staged loading

A round whose context is in a tier is read back before it computes; there is no layer-wise pipelining.

- **Loading area.** 10% of HBM's KV pool (`tiers.loading_area: 0.10`) is reserved for reads. A read holds its blocks
  there until the round is admitted. An SSD read first lands in the same fraction of DRAM (a fetch), then moves to
  HBM's area (a hop).
- **Order.** Reads start in the scheduler's policy order while they fit their area; landed rounds are admitted first,
  in landing order; a round that waited 60 s is forced.
- A DRAM read is one batched host-to-device copy per GPU. An SSD read lands slot by slot in staging slots, and each
  slot's host-to-device copy starts as it lands.

## Eviction

Each tier evicts whole sessions, least recently used first; a read refreshes. HBM and DRAM evict only entries the
level below holds whole.

## Correctness

[tests/test_kvpool_bytes_gpu.py](../../tests/test_kvpool_bytes_gpu.py) (4 GPUs, real stage models) writes a context
to DRAM and to SSD, evicts it, overwrites its blocks, reads it back into other blocks and checks every byte, then
checks that the next round's logits are bitwise equal to an HBM hit. The CPU tests (`tests/test_kvpool_*.py`) cover
slot accounting, write-before-evict, the victim chain and the quarter rule's coverage of every byte.
