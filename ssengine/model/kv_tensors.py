"""One stage's KV tensors: vLLM's GLM-5.3 hybrid layout for KV, backed by the controller's page space, with compact
KDA states (docs/design/kept_states.md).

`1_contract.md` §1 and `ssengine/runtime/README.md`: a block is 8,576 tokens of DSA KV, 67 pages of 128. vLLM computes
the KV layout for the model (`Platform._align_hybrid_block_size` -> block_size 8576;
`kv_cache_utils._get_kv_cache_groups_glm5_next`), so we reuse it to allocate:

- The cache groups are computed from all four stages' layer specs (the GLM-5 branch balances KDA groups across
  pipeline stages), then projected onto this stage, exactly as vLLM's `get_kv_cache_configs` does for PP workers.
  Global groups: g0 MLA + indexer K (one block table), g1 kpool tail, g2..g6 the 5 KDA groups. vLLM aliases the KDA
  and tail views onto the MLA and indexer tensors, a whole block per state.
- Block ids are the controller's: block b of every tensor is the same id on every stage; block 0 is the null block.
- States: vLLM's KDA and tail views are replaced by compact ones (`state_layout`): a request's state is R runs, each
  the first c pages of a block, `state_blocks` holds the runs' block ids, and each KDA layer's conv / recurrent state
  and each tail has a fixed (run, region, offset) over this stage's MLA and indexer tensors. `StageKV.kda_groups[r]` are the
  KDA layers whose state is in run r, `tail_run` the run of every tail.

Tensors are allocated by vLLM's `allocate_kv_cache` and bound by `bind_kv_cache_to_layers` (the MLA layer, indexer
caches and KDA layers read their views from `layer.kv_cache`); the state views are bound over them after.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ssengine import kernels
from ssengine.config import BLOCK_PAGES, CONTRACT_BLOCKS
from ssengine.model import state_layout

PREFIX = "language_model.model.layers"
NUM_LAYERS = 45
VLLM_KDA_GROUPS = 5       # vLLM's KDA cache groups (its own state layout, replaced by the compact one)
KERNEL_BLOCK = 64         # g0's kernel block: FlashInfer sparse MLA's page (metadata.KERNEL_BLOCK)


def layer_spec_names(i: int) -> list[str]:
    if i in kernels.DSA_LAYERS:
        return [f"{PREFIX}.{i}.self_attn.attn", f"{PREFIX}.{i}.self_attn.indexer.k_cache",
                f"{PREFIX}.{i}.self_attn.indexer.tail_cache"]
    return [f"{PREFIX}.{i}.self_attn"]


def all_stage_specs(local_spec: dict) -> dict:
    """Specs of all 45 layers, from one representative of each kind on this stage (every stage has both kinds)."""
    rep = {}
    for name, spec in local_spec.items():
        kind = name.rsplit(".", 1)[-1] if ".indexer." in name or name.endswith(".attn") else "kda"
        rep.setdefault(kind, spec)
    out = {}
    for i in range(NUM_LAYERS):
        for name in layer_spec_names(i):
            kind = name.rsplit(".", 1)[-1] if ".indexer." in name or name.endswith(".attn") else "kda"
            out[name] = rep[kind]
    return out


def design_pp_indices(num_layers: int, rank: int, size: int) -> tuple[int, int]:
    """Stage `rank`'s layers under the design's split [12, 11, 11, 11] (`kernels.SPLIT`)."""
    assert (num_layers, size) == (NUM_LAYERS, len(kernels.SPLIT)), (num_layers, size)
    start = sum(kernels.SPLIT[:rank])
    return start, start + kernels.SPLIT[rank]


@contextlib.contextmanager
def design_split():
    """vLLM's GLM-5 group function places layers with `vllm.distributed.utils.get_pp_indices` (imported at call
    time), which takes the split from VLLM_PP_LAYER_PARTITION as vllm.envs cached it, else vLLM's own default split.
    Within this block it gets the design's split instead."""
    from vllm.distributed import utils as dist_utils

    saved = dist_utils.get_pp_indices
    dist_utils.get_pp_indices = design_pp_indices
    try:
        yield
    finally:
        dist_utils.get_pp_indices = saved


def group_config_shim(vc, num_blocks: int | None):
    """What the GLM-5 group functions read of the VllmConfig, with the pipeline the design has (4 stages; the split
    comes from `design_split`)."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=len(kernels.SPLIT)),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: NUM_LAYERS),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(num_gpu_blocks_override=num_blocks, prefix_cache_retention_interval=None),
    )


def stage_kv_config(vc, local_spec: dict, num_blocks: int):
    """vLLM KVCacheConfig of this stage for `num_blocks` blocks (incl. the null block)."""
    from vllm.v1.core import kv_cache_utils as kcu

    shim = group_config_shim(vc, num_blocks)
    with design_split():
        groups = kcu._get_kv_cache_groups_glm5_next(shim, all_stage_specs(local_spec))
    proj = kcu._project_kv_cache_groups_to_worker(groups, local_spec)
    per_block = kcu._get_kv_cache_bytes_per_block(proj)
    return kcu.get_kv_cache_config_from_groups(shim, proj, per_block * num_blocks), per_block


def bytes_per_block(vc, local_spec: dict) -> int:
    from vllm.v1.core import kv_cache_utils as kcu

    with design_split():
        groups = kcu._get_kv_cache_groups_glm5_next(group_config_shim(vc, None), all_stage_specs(local_spec))
    return kcu._get_kv_cache_bytes_per_block(kcu._project_kv_cache_groups_to_worker(groups, local_spec))


@dataclass
class StageKV:
    config: object                  # vLLM KVCacheConfig
    num_blocks: int                 # incl. the null block
    bytes_per_block: int
    mla_names: list[str]            # g0 MLA layers
    indexer_names: list[str]        # g0 indexer K caches
    tail_names: list[str]           # g1
    kda_groups: list[list[str]]     # per state run r: the KDA layer prefixes whose state is in run r
    mla_spec: object
    indexer_spec: object
    tail_spec: object
    state: object = None            # state_layout.StateViews: the compact state views and the state copy
    tail_run: int = 0               # the run holding every kpool tail

    @property
    def geometry(self):
        return self.state.geo

    def state_at(self, block: int) -> list[int]:
        """A state with its runs in blocks block .. block + R - 1: warm-up shapes."""
        return [block + r for r in range(self.geometry.runs)]

    @property
    def block_tokens(self) -> int:
        return self.mla_spec.block_size

    @property
    def page_tokens(self) -> int:
        """The KV allocation unit (`config.PAGE_TOKENS`): the smallest token range both g0 kernels address through
        their page tables, lcm of the MLA kernel block (`KERNEL_BLOCK`) and the indexer's storage page
        (`storage_block_size`: DeepGEMM paged MQA's 32 or 64 pooled entries x index_kpool tokens)."""
        storage = getattr(self.indexer_spec, "storage_block_size", None) or KERNEL_BLOCK
        page = math.lcm(KERNEL_BLOCK, storage)
        assert self.block_tokens % page == 0, (self.block_tokens, page)
        return page

    def kda_state_index(self) -> dict[str, int]:
        return {name: g for g, names in enumerate(self.kda_groups) for name in names}


def _stage_of(local_spec: dict) -> int:
    first = min(int(n.split(".")[3]) for n in local_spec)
    return next(s for s in range(len(kernels.SPLIT)) if first in kernels.stage_layers(s))


def stage_pieces(local_spec: dict) -> list[state_layout.StagePieces]:
    """Every stage's KV regions (bytes per page) and state pieces from the layer specs (all stages': the geometry is
    global)."""
    import torch

    specs = all_stage_specs(local_spec)
    out = []
    for s in range(len(kernels.SPLIT)):
        regions, kda, tails = [], [], []
        for i in kernels.stage_layers(s):
            names = layer_spec_names(i)
            if i in kernels.DSA_LAYERS:
                mla, idx, tail = names
                for n in (mla, idx):
                    assert specs[n].page_size_bytes % BLOCK_PAGES == 0, (n, specs[n].page_size_bytes)
                    regions.append((n, specs[n].page_size_bytes // BLOCK_PAGES))
                tails.append((tail, specs[tail].page_size_bytes))
            else:
                sp = specs[names[0]]
                conv, rec = (math.prod(sh) * torch.empty(0, dtype=dt).element_size()
                             for sh, dt in zip(sp.shapes, sp.dtypes))
                kda.append((names[0], conv, rec))
        out.append(state_layout.StagePieces(tuple(regions), tuple(kda), tuple(tails)))
    return out


def state_geometry(local_spec: dict) -> state_layout.Geometry:
    return state_layout.geometry(stage_pieces(local_spec))


def bind_states(ctx: dict, kv: "StageKV", local_spec: dict) -> None:
    """Replace vLLM's KDA and tail views (a whole block per state) by the compact ones of `state_layout`."""
    geo = state_geometry(local_spec)
    places = geo.places[_stage_of(local_spec)]

    def view(name):
        v = ctx[name].kv_cache
        return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v

    regions = {}
    for name in kv.mla_names + kv.indexer_names:
        t = view(name)
        rows = t.shape[0] // kv.num_blocks
        stride = t.stride(0) * t.element_size() * rows          # vLLM's block stride (layers may share blocks)
        assert t[0].is_contiguous() and (rows == 1 or t.stride(0) * t.element_size() == t[0].numel()
                                         * t.element_size()), name
        assert stride >= local_spec[name].page_size_bytes, (name, stride, local_spec[name].page_size_bytes)
        regions[name] = (t, stride)
    pieces = {}
    kda = [n for g in kv.kda_groups for n in g]
    for name in kda:
        for i, t in enumerate(view(name)):          # (conv, recurrent)
            pieces[f"{name}#{i}"] = (tuple(t.shape[1:]), t.dtype)
    for name in kv.tail_names:
        t = view(name)
        pieces[name] = (tuple(t.shape[1:]), t.dtype)
    sv = state_layout.StateViews(regions, kv.num_blocks, geo, places, pieces)
    for name in kda:
        ctx[name].kv_cache = (sv.views[f"{name}#0"], sv.views[f"{name}#1"])
    for name in kv.tail_names:
        ctx[name].kv_cache = sv.views[name]
    kv.state = sv
    kv.kda_groups = [[n for n in kda if sv.run_of[f"{n}#1"] == r] for r in range(geo.runs)]
    runs = {sv.run_of[n] for n in kv.tail_names}
    assert len(runs) <= 1, runs
    kv.tail_run = runs.pop() if runs else 0


def resolve_layout(vc, static_ctx: dict, local_spec: dict):
    """The KV layout vLLM's engine core would resolve for this stage's backends (`resolve_kv_cache_layout` over
    `get_supported_kv_cache_layouts`), recorded on the cache config as the engine does."""
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout

    backends = {static_ctx[name].get_attn_backend() for name in local_spec}
    supported = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
    layout = resolve_kv_cache_layout(vc, [supported], local_spec.values())
    vc.cache_config.kv_cache_layout = layout.name
    return layout


def allocate(vc, static_ctx: dict, local_spec: dict, num_blocks: int, device: torch.device,
             ipc: bool = False) -> StageKV:
    """This stage's KV tensors, bound to its layers. `ipc`: allocated by plain cudaMalloc segments, which CUDA IPC can
    share (the tiers' peers copy them), also when the process runs with expandable segments (their VMM memory is not
    IPC-shareable); the setting is restored after, as vLLM's CuMemAllocator does around its pool."""
    from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec, UniformTypeKVCacheSpecs
    from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache_to_layers

    cfg, per_block = stage_kv_config(vc, local_spec, num_blocks)
    mla, idx, tail, kda = [], [], [], []
    kernel_block_sizes = []
    for group in cfg.kv_cache_groups:
        spec = group.kv_cache_spec
        inner = spec.kv_cache_specs[group.layer_names[0]] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
        if isinstance(inner, MambaSpec):
            kda.append(list(group.layer_names))
            kernel_block_sizes.append(inner.block_size)
        elif isinstance(inner, KpoolTailSpec):
            tail += group.layer_names
            kernel_block_sizes.append(inner.block_size)
        else:
            for name in group.layer_names:
                (idx if ".indexer." in name else mla).append(name)
            kernel_block_sizes.append(KERNEL_BLOCK)   # FlashInfer sparse MLA / indexer (select_common_block_size)
    assert len(kda) == VLLM_KDA_GROUPS, f"{len(kda)} KDA groups"
    layout = resolve_layout(vc, static_ctx, local_spec)
    if hasattr(cfg, "kv_cache_layout"):
        cfg.kv_cache_layout = layout.name
    expandable = ipc and "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if expandable:
        torch.cuda.empty_cache()
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    try:
        tensors = allocate_kv_cache(cfg, device, layout, kernel_block_sizes)
    finally:
        if expandable:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    bind_kv_cache_to_layers(tensors, static_ctx, 1, kv_cache_groups=cfg.kv_cache_groups)
    kv = StageKV(cfg, num_blocks, per_block, mla, idx, tail, kda,
                 local_spec[mla[0]], local_spec[idx[0]], local_spec[tail[0]] if tail else None)
    bind_states(static_ctx, kv, local_spec)
    return kv


def blocks_that_fit(per_block: int, free_bytes: int, reserve_bytes: int, pool_tokens: int | None,
                    block_tokens: int = 8576) -> int:
    """Blocks this stage can back: free HBM after weights minus the activation/workspace reserve, at most the
    contract's 7,144 (incl. the null block), at most `pool_tokens` worth when the run config caps the pool."""
    n = min(CONTRACT_BLOCKS, max(0, (free_bytes - reserve_bytes) // per_block))
    if pool_tokens is not None:
        n = min(n, pool_tokens // block_tokens + 1)
    return int(n)
