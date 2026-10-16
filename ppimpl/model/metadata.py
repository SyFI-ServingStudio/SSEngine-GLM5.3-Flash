"""A microbatch plan -> vLLM attention metadata for one stage, from our block tables and state blocks.

vLLM's own builder classes build the MLA, indexer and kpool-tail metadata (as `GPUModelRunner` does, one
`AttentionGroup` per (cache group, backend)); we only fill the `CommonAttentionMetadata` they take, from the plan:

- requests in the plan's chunk order, except that vLLM requires decode-class rows (query <= the sparse MLA reorder
  threshold, 256 tokens at 64 heads) first. When a plan breaks that order, every stage applies the same stable
  reordering, the hidden rows travel between stages in that order, and only stage 0 (token ids) and stage 3 (the
  rows that produce tokens) translate. `Batch.order` is the chunk order used.
- g0 (MLA + indexer K): block table in 64-token kernel blocks, logical block b -> b * 134 + j; slot mapping
  b * 8576 + pos % 8576 with b = kv_blocks[pos // 8576].
- g1 (kpool tail): one block per request, its state_blocks[5]; the builder computes its own slots.
- kpool pools a chunk starts inside (start % 4 != 0: prefix hits, odd chunk ends): `pp_patch.KPOOL_HEADS`, keyed by
  the indexer metadata's prefill slot mapping, read by the kpool insert override.
- KDA (g2..g6): `pp_patch.KdaPrefillMeta`, read by our KDA prefill forward (the fork's GDN metadata is not built:
  the `_forward` that reads it is replaced).

Everything is built on the host, packed into one pinned staging buffer and copied with one non-blocking H2D copy
on the compute stream (`Pack`), so building microbatch k+1 never waits for the GPU work of microbatch k.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import torch

from ppimpl.model import kv_tensors
from ppimpl.model.pp_patch import (KPOOL_HEADS, TOPK_SEGMENTS, KdaPrefillMeta, KpoolHeads, kpool_heads_all,
                                   kpool_heads_suffix, make_kda_plan)

DECODE_THRESHOLD = 256
KERNEL_BLOCK = 64
PAD_SLOT_ID = -1


@dataclass
class Batch:
    num_tokens: int
    order: list[int]                  # plan chunk indices in row order
    positions: torch.Tensor           # int64 [T], device
    token_ids: torch.Tensor | None    # int64 [T], device (stage 0)
    attn: dict                        # layer name -> metadata
    slots: dict                       # layer name -> slot mapping (MLA KV write)
    last_rows: torch.Tensor           # int64 rows whose token stage 3 returns, in plan chunk order of is_last chunks
    bucket: int | None = None         # CUDA graph bucket (padded token count) when the plan runs as a graph replay
    graph_rows: torch.Tensor | None = None   # int64 [2 or 3, bucket], device: `graph_rows` of this microbatch


_NP = {torch.int32: np.int32, torch.int64: np.int64, torch.bool: np.bool_}
PACK_ALIGN = 256          # every segment starts 256-byte aligned (the CUDA caching allocator's own granularity is 512)


def graph_rows(positions, slots, token_ids, bucket: int) -> np.ndarray:
    """What a captured stage graph reads of a microbatch, padded to its bucket: positions (pad 0), the g0 slot
    mapping (pad -1: the padded rows write no KV) and, on stage 0, the token ids (pad 0). Rows of one int64 array,
    copied into the graph's static buffer with one device copy before replay."""
    T = len(positions)
    assert T <= bucket, (T, bucket)
    rows = [positions, slots] + ([token_ids] if token_ids is not None else [])
    out = np.zeros((len(rows), bucket), dtype=np.int64)
    out[1, :] = PAD_SLOT_ID
    for i, r in enumerate(rows):
        out[i, :T] = r
    return out


class Pack:
    """All of one microbatch's host arrays in one pinned staging buffer, copied to the device in one non-blocking H2D
    copy; each array is a typed view of the staging buffer (host side) and of the device buffer (device side).
    `add` returns a handle; `copy` stages and issues the copy; `host(h)` / `dev(h)` return the views. The views
    keep both buffers alive (the caching host allocator holds the pinned block until the copy has completed)."""

    def __init__(self):
        self.items: list[tuple[int, np.ndarray, torch.dtype, tuple]] = []
        self.size = 0
        self.host_buf = self.dev_buf = None

    def add(self, a, dtype: torch.dtype) -> int:
        arr = np.ascontiguousarray(a, dtype=_NP[dtype])
        off = -(-self.size // PACK_ALIGN) * PACK_ALIGN
        self.items.append((off, arr, dtype, arr.shape))
        self.size = off + arr.nbytes
        return len(self.items) - 1

    def copy(self, device: torch.device, pin: bool = True) -> None:
        self.host_buf = torch.empty(max(self.size, 1), dtype=torch.uint8, pin_memory=pin)
        hv = self.host_buf.numpy()
        for off, arr, _, _ in self.items:
            hv[off:off + arr.nbytes] = arr.reshape(-1).view(np.uint8)
        self.dev_buf = self.host_buf.to(device, non_blocking=True)

    def _view(self, buf: torch.Tensor, h: int) -> torch.Tensor:
        off, arr, dtype, shape = self.items[h]
        return buf[off:off + arr.nbytes].view(dtype).view(shape)

    def host(self, h: int) -> torch.Tensor:
        return self._view(self.host_buf, h)

    def dev(self, h: int) -> torch.Tensor:
        return self._view(self.dev_buf, h)


class StageMetadata:
    def __init__(self, vc, kv: kv_tensors.StageKV, device: torch.device, num_kda_heads: int = 64):
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        from vllm.v1.worker.utils import AttentionGroup

        self.vc, self.kv, self.device = vc, kv, device
        self.block_tokens = kv.block_tokens
        assert self.block_tokens % KERNEL_BLOCK == 0
        self.kernel_per_block = self.block_tokens // KERNEL_BLOCK
        ctx = vc.compilation_config.static_forward_context
        self.groups: list[tuple[str, object]] = []    # ("g0" | "tail", AttentionGroup)
        for gid, group in enumerate(kv.config.kv_cache_groups):
            spec = group.kv_cache_spec
            if group.layer_names[0] in kv.tail_names:
                kind, kernel_bs = "tail", spec.block_size
            elif isinstance(spec, UniformTypeKVCacheSpecs):
                kind, kernel_bs = "g0", KERNEL_BLOCK
            else:
                continue   # KDA groups
            by_backend: dict = collections.defaultdict(list)
            for name in group.layer_names:
                layer_spec = spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
                by_backend[(ctx[name].get_attn_backend(), layer_spec)].append(name)
            for (backend, layer_spec), names in by_backend.items():
                ag = AttentionGroup(backend, names, layer_spec, gid)
                ag.create_metadata_builders(vc, device, kernel_bs)
                self.groups.append((kind, ag))
        self.kda_index = kv.kda_state_index()
        text = getattr(getattr(vc, "model_config", None), "hf_text_config", None)
        self.kpool = int(getattr(text, "index_kpool", 1) or 1)
        self.num_kda_heads = num_kda_heads
        self._plans: collections.OrderedDict = collections.OrderedDict()
        self._side = torch.cuda.Stream(device)

    # -- KDA work plan (FlashInfer persistent KDA): built once per distinct sequence offsets, on a side stream so
    #    its small synchronous H2D copies never wait for queued compute.
    def _kda_plan(self, offsets: tuple[int, ...]):
        plan = self._plans.get(offsets)
        if plan is None:
            with torch.cuda.stream(self._side):
                plan = make_kda_plan(list(offsets), self.num_kda_heads, self.device)
            self._side.synchronize()
            compute = torch.cuda.current_stream(self.device)
            for t in plan:
                if isinstance(t, torch.Tensor):
                    t.record_stream(compute)
            self._plans[offsets] = plan
            if len(self._plans) > 512:
                self._plans.popitem(last=False)
        else:
            self._plans.move_to_end(offsets)
        return plan

    def build(self, plan, with_tokens: bool, bucket: int | None = None) -> Batch:
        from vllm.v1.attention.backend import CommonAttentionMetadata

        chunks = plan.chunks
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].num_tokens > DECODE_THRESHOLD)
        R = len(order)
        q_lens = np.array([chunks[i].num_tokens for i in order], dtype=np.int64)
        starts = np.array([chunks[i].start for i in order], dtype=np.int64)
        seq_lens = starts + q_lens
        qsl = np.zeros(R + 1, dtype=np.int64)
        np.cumsum(q_lens, out=qsl[1:])
        T = int(qsl[-1])
        positions = np.concatenate([np.arange(s, e) for s, e in zip(starts, seq_lens)])

        # g0: kernel-block table and slot mapping
        B = self.block_tokens
        width = int(max(-(-int(s) // B) for s in seq_lens)) * self.kernel_per_block
        table = np.zeros((R, width), dtype=np.int64)
        slots = np.empty(T, dtype=np.int64)
        j = np.arange(self.kernel_per_block)
        for r, i in enumerate(order):
            blocks = np.asarray(chunks[i].kv_blocks, dtype=np.int64)
            n = -(-int(seq_lens[r]) // B)
            assert len(blocks) >= n, (chunks[i].req_id, len(blocks), n)
            table[r, : n * self.kernel_per_block] = (blocks[:n, None] * self.kernel_per_block + j).reshape(-1)
            pos = positions[qsl[r]:qsl[r + 1]]
            slots[qsl[r]:qsl[r + 1]] = blocks[pos // B] * B + pos % B
        tail_table = np.array([[chunks[i].state_blocks[kv_tensors.TAIL_STATE]] for i in order], dtype=np.int64)

        # every host array of this microbatch -> one pinned buffer -> one H2D copy (M3: was 16 pinned allocations,
        # 16 copies and 2 fills per build)
        pk = Pack()
        h_pos = pk.add(positions, torch.int64)

        def cm_arrays(block_table: np.ndarray, slot_mapping: np.ndarray) -> dict:
            return {"qsl": pk.add(qsl, torch.int32), "seq": pk.add(seq_lens, torch.int32),
                    "table": pk.add(block_table, torch.int32), "slots": pk.add(slot_mapping, torch.int64),
                    "prefill": pk.add(np.ones(R, dtype=np.bool_), torch.bool)}

        h_g0 = cm_arrays(table, slots)
        h_tail = cm_arrays(tail_table, np.full(T, PAD_SLOT_ID, dtype=np.int64))
        h_kda = [pk.add([chunks[i].state_blocks[g] for i in order], torch.int64)
                 for g in range(len(self.kv.kda_groups))]
        tok = (np.concatenate([np.asarray(chunks[i].token_ids, dtype=np.int64) for i in order]) if with_tokens
               else None)
        h_tok = None if tok is None else pk.add(tok, torch.int64)
        h_graph = None if bucket is None else pk.add(graph_rows(positions, slots, tok, bucket), torch.int64)
        row_end = {i: int(qsl[r + 1]) - 1 for r, i in enumerate(order)}
        h_last = pk.add([row_end[i] for i, c in enumerate(chunks) if c.is_last], torch.int64)
        # kpool pools a chunk starts inside (pp_patch "kpool indexer": prefix hits, odd chunk ends), for every
        # request, staged with the rest; the indexer's prefill requests are cut out once its metadata is built
        hk, hk_req = (kpool_heads_all(starts, q_lens, qsl[:-1], tail_table[:, 0], self.kpool) if self.kpool > 1
                      else (None, []))
        h_hk = None if hk is None else [pk.add(a, dt) for a, dt in
                                        zip(hk, (torch.int64, torch.bool, torch.int64, torch.int64))]
        pk.copy(self.device)
        pos_dev = pk.dev(h_pos)

        def common(h: dict) -> CommonAttentionMetadata:
            seq_cpu = pk.host(h["seq"])
            return CommonAttentionMetadata(
                query_start_loc=pk.dev(h["qsl"]), query_start_loc_cpu=pk.host(h["qsl"]),
                seq_lens=pk.dev(h["seq"]), num_reqs=R, num_actual_tokens=T,
                max_query_len=int(q_lens.max()), max_seq_len=int(seq_lens.max()),
                block_table_tensor=pk.dev(h["table"]), slot_mapping=pk.dev(h["slots"]), causal=True,
                positions=pos_dev, is_prefilling=pk.dev(h["prefill"]), seq_lens_cpu_upper_bound=seq_cpu)

        TOPK_SEGMENTS.clear()
        KPOOL_HEADS.clear()
        attn: dict = {}
        slot_map: dict = {}
        cm_g0 = common(h_g0)
        cm_tail = common(h_tail)
        heads: dict = {}
        for kind, ag in self.groups:
            cm = cm_g0 if kind == "g0" else cm_tail
            md = ag.get_metadata_builder().build(common_prefix_len=0, common_attn_metadata=cm)
            for name in ag.layer_names:
                attn[name] = md
                if kind == "g0":
                    slot_map[name] = cm_g0.slot_mapping
            if kind == "g0" and self.kpool > 1 and type(md).__name__ == "DeepseekV32IndexerMetadata":
                # device views of the staged heads (a pageable .to(device) here waited for the GPU to drain at
                # every unaligned start, and the host lost its lead: issues.md, "unaligned resume cost")
                nd = int(md.num_decodes)
                if nd not in heads:
                    h = kpool_heads_suffix(None if h_hk is None else [pk.dev(x) for x in h_hk], hk_req, nd,
                                           int(md.num_decode_tokens))
                    heads[nd] = None if h is None else KpoolHeads(*h)
                if heads[nd] is not None:
                    KPOOL_HEADS[md.slot_mapping[int(md.num_decode_tokens):].data_ptr()] = heads[nd]

        # KDA: one meta per state group (the sequences and plan are shared, the state blocks differ)
        offsets = tuple(int(v) for v in qsl)
        kda_plan = self._kda_plan(offsets)
        any_initial = bool((starts > 0).any())
        for g, names in enumerate(self.kv.kda_groups):
            blocks = [chunks[i].state_blocks[g] for i in order]
            meta = KdaPrefillMeta(
                num_tokens=T,
                seqs=[(int(qsl[r]), int(qsl[r + 1]), blocks[r], bool(starts[r] > 0)) for r in range(R)],
                slots=pk.dev(h_kda[g]), plan=kda_plan, any_initial=any_initial)
            for name in names:
                attn[name] = meta

        token_ids = None if h_tok is None else pk.dev(h_tok)
        last_rows = pk.dev(h_last)
        return Batch(T, order, pos_dev, token_ids, attn, slot_map, last_rows, bucket,
                     None if h_graph is None else pk.dev(h_graph))
