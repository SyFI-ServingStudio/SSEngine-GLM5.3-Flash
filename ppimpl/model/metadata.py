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
- KDA (g2..g6): `pp_patch.KdaPrefillMeta`, read by our KDA prefill forward (the fork's GDN metadata is not built:
  the `_forward` that reads it is replaced).

Everything is built on the host and copied with non-blocking H2D copies from pinned memory on the compute stream,
so building microbatch k+1 never waits for the GPU work of microbatch k.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import torch

from ppimpl.model import kv_tensors
from ppimpl.model.pp_patch import TOPK_SEGMENTS, KdaPrefillMeta, make_kda_plan

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


def _pinned(a: np.ndarray, dtype) -> torch.Tensor:
    return torch.from_numpy(a).to(dtype).pin_memory()


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

    def build(self, plan, with_tokens: bool) -> Batch:
        from vllm.v1.attention.backend import CommonAttentionMetadata

        chunks = plan.chunks
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].num_tokens > DECODE_THRESHOLD)
        R = len(order)
        q_lens = np.array([chunks[i].num_tokens for i in order], dtype=np.int64)
        starts = np.array([chunks[i].start for i in order], dtype=np.int64)
        assert all(s % 4 == 0 for s in starts), "kpool indexer: chunk starts must be pool-aligned"
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

        dev, nb = self.device, True
        qsl_cpu = _pinned(qsl, torch.int32)
        seq_cpu = _pinned(seq_lens, torch.int32)
        pos_dev = _pinned(positions, torch.int64).to(dev, non_blocking=nb)

        def common(block_table: np.ndarray, slot_mapping: np.ndarray) -> CommonAttentionMetadata:
            return CommonAttentionMetadata(
                query_start_loc=qsl_cpu.to(dev, non_blocking=nb), query_start_loc_cpu=qsl_cpu,
                seq_lens=seq_cpu.to(dev, non_blocking=nb), num_reqs=R, num_actual_tokens=T,
                max_query_len=int(q_lens.max()), max_seq_len=int(seq_lens.max()),
                block_table_tensor=_pinned(block_table, torch.int32).to(dev, non_blocking=nb),
                slot_mapping=_pinned(slot_mapping, torch.int64).to(dev, non_blocking=nb), causal=True,
                positions=pos_dev, is_prefilling=torch.ones(R, dtype=torch.bool, device=dev),
                seq_lens_cpu_upper_bound=seq_cpu)

        TOPK_SEGMENTS.clear()
        attn: dict = {}
        slot_map: dict = {}
        cm_g0 = common(table, slots)
        cm_tail = common(tail_table, np.full(T, PAD_SLOT_ID, dtype=np.int64))
        for kind, ag in self.groups:
            cm = cm_g0 if kind == "g0" else cm_tail
            md = ag.get_metadata_builder().build(common_prefix_len=0, common_attn_metadata=cm)
            for name in ag.layer_names:
                attn[name] = md
                if kind == "g0":
                    slot_map[name] = cm_g0.slot_mapping

        # KDA: one meta per state group (the sequences and plan are shared, the state blocks differ)
        offsets = tuple(int(v) for v in qsl)
        kda_plan = self._kda_plan(offsets)
        any_initial = bool((starts > 0).any())
        for g, names in enumerate(self.kv.kda_groups):
            blocks = [chunks[i].state_blocks[g] for i in order]
            meta = KdaPrefillMeta(
                num_tokens=T,
                seqs=[(int(qsl[r]), int(qsl[r + 1]), blocks[r], bool(starts[r] > 0)) for r in range(R)],
                slots=_pinned(np.asarray(blocks, dtype=np.int64), torch.int64).to(dev, non_blocking=nb),
                plan=kda_plan, any_initial=any_initial)
            for name in names:
                attn[name] = meta

        token_ids = None
        if with_tokens:
            ids = np.concatenate([np.asarray(chunks[i].token_ids, dtype=np.int64) for i in order])
            token_ids = _pinned(ids, torch.int64).to(dev, non_blocking=nb)
        row_end = {i: int(qsl[r + 1]) - 1 for r, i in enumerate(order)}
        last = [row_end[i] for i, c in enumerate(chunks) if c.is_last]
        last_rows = _pinned(np.asarray(last, dtype=np.int64), torch.int64).to(dev, non_blocking=nb)
        return Batch(T, order, pos_dev, token_ids, attn, slot_map, last_rows)
