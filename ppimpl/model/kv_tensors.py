"""One stage's KV tensors: vLLM's GLM-5.3 hybrid layout, backed by the controller's block-id space.

`1_contract.md` §1 and `ppimpl/runtime/README.md`: a block is 8,576 tokens of DSA KV or one KDA layer's state; a
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

import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ppimpl import kernels

PREFIX = "language_model.model.layers"
NUM_LAYERS = 45
STATE_BLOCKS = 6
TAIL_STATE = 5            # state_blocks[5] backs the kpool tail group
CONTRACT_BLOCKS = 7144    # 1_contract.md §1, incl. the null block


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


def group_config_shim(vc, num_blocks: int | None):
    """What the GLM-5 group functions read of the VllmConfig, with the pipeline the design has (4 stages, split
    [12, 11, 11, 11]: the group function places layers with vLLM's `get_pp_indices`, which reads the split from
    VLLM_PP_LAYER_PARTITION)."""
    os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join(str(n) for n in kernels.SPLIT)
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
    groups = kcu._get_kv_cache_groups_glm5_next(shim, all_stage_specs(local_spec))
    proj = kcu._project_kv_cache_groups_to_worker(groups, local_spec)
    per_block = kcu._get_kv_cache_bytes_per_block(proj)
    return kcu.get_kv_cache_config_from_groups(shim, proj, per_block * num_blocks), per_block


def bytes_per_block(vc, local_spec: dict) -> int:
    from vllm.v1.core import kv_cache_utils as kcu

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


def allocate(vc, static_ctx: dict, local_spec: dict, num_blocks: int, device: torch.device) -> StageKV:
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
            kernel_block_sizes.append(64)   # FlashInfer sparse MLA / indexer kernel block (select_common_block_size)
    assert len(kda) == STATE_BLOCKS - 1, f"{len(kda)} KDA groups"
    layout = resolve_layout(vc, static_ctx, local_spec)
    if hasattr(cfg, "kv_cache_layout"):
        cfg.kv_cache_layout = layout.name
    tensors = allocate_kv_cache(cfg, device, layout, kernel_block_sizes)
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
