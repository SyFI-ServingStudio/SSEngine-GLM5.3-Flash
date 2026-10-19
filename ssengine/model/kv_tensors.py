"""One stage's KV tensors: vLLM's GLM-5.3 hybrid layout, backed by the controller's block-id space.

`1_contract.md` §1 and `ssengine/runtime/README.md`: a block is 8,576 tokens of DSA KV or one KDA layer's state; a
request holds ceil(prompt / 8576) KV blocks and 6 state blocks. vLLM computes exactly this layout for the model
(`Platform._align_hybrid_block_size` -> block_size 8576; `kv_cache_utils._get_kv_cache_groups_glm5_next`), so we
reuse it instead of re-deriving it:

- The cache groups are computed from all four stages' layer specs (the GLM-5 branch balances KDA groups across
  pipeline stages), then projected onto this stage, exactly as vLLM's `get_kv_cache_configs` does for PP workers.
  Global groups: g0 MLA + indexer K (one block table), g1 kpool tail (1 block per request), g2..g6 the 5 KDA groups
  (1 state block per request each).
- On a stage, KDA group g's i-th layer aliases the i-th DSA layer's MLA region; the tail aliases the indexer region.
- Block ids are the controller's: block b of every tensor is the same id on every stage; block 0 is the null block.
  A request's 6 state blocks map to g2..g6 (state_blocks[0..4]) and g1, the tail (state_blocks[5]).

Tensors are allocated by vLLM's `allocate_kv_cache` and bound by `bind_kv_cache_to_layers` (the MLA layer, indexer
caches and KDA layers read their views from `layer.kv_cache`).
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ssengine import kernels
from ssengine.config import CONTRACT_BLOCKS

PREFIX = "language_model.model.layers"
NUM_LAYERS = 45
STATE_BLOCKS = 6
KERNEL_BLOCK = 64         # g0's kernel block: FlashInfer sparse MLA's page (metadata.KERNEL_BLOCK)
TAIL_STATE = 5            # state_blocks[5] backs the kpool tail group


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
    kda_groups: list[list[str]]     # g2..g6, in order: KDA layer prefixes per state block 0..4
    mla_spec: object
    indexer_spec: object
    tail_spec: object

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
    assert len(kda) == STATE_BLOCKS - 1, f"{len(kda)} KDA groups"
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
    return StageKV(cfg, num_blocks, per_block, mla, idx, tail, kda,
                   local_spec[mla[0]], local_spec[idx[0]], local_spec[tail[0]] if tail else None)


def blocks_that_fit(per_block: int, free_bytes: int, reserve_bytes: int, pool_tokens: int | None,
                    block_tokens: int = 8576) -> int:
    """Blocks this stage can back: free HBM after weights minus the activation/workspace reserve, at most the
    contract's 7,144 (incl. the null block), at most `pool_tokens` worth when the run config caps the pool."""
    n = min(CONTRACT_BLOCKS, max(0, (free_bytes - reserve_bytes) // per_block))
    if pool_tokens is not None:
        n = min(n, pool_tokens // block_tokens + 1)
    return int(n)
