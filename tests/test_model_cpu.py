"""CPU checks of the model lane: kernel manifest, top-k table, trace check, per-stage weight selection."""

import json
from pathlib import Path

import pytest

from ppimpl import kernels as K

CKPT = Path("/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348")


def test_manifest_has_35_configs_in_order():
    assert [c.config for c in K.CONFIGS] == list(range(1, 36))
    assert sum(c.covered for c in K.CONFIGS) == 14
    for c in K.CONFIGS:
        assert c.names and c.layer_types and c.positions


def test_stage_layer_types_follow_split():
    assert [len(K.stage_layers(s)) for s in range(4)] == [12, 11, 11, 11]
    assert K.stage_layers(3).stop == 45
    assert {K.layer_type(i) for i in K.stage_layers(1)} == {"kda_moe", "dsa_moe"}
    assert sorted(i for i in range(45) if K.layer_type(i) == "dsa_moe") == list(range(3, 45, 4))
    assert "input_head" in K.stage_layer_types(0) and "output_head" in K.stage_layer_types(3)


def test_topk_choice_matches_the_faster_row_at_every_profiled_shape():
    rows = json.loads((Path(K.__file__).parent / "data" / "topk_rows.json").read_text())["rows"]
    assert len(rows) == 412
    for q, k, ds, vc in rows:
        if k <= 512:
            continue
        assert K.topk_backend(q, k) == ("deep_select" if ds < vc else "vllm_cuda"), (q, k)


def test_check_trace_flags_fallbacks_and_unknowns():
    names = ["nvjet_tst_128x64", "causal_conv1d_channellast_fwd_kernel", "_causal_conv1d_fwd_kernel", "weird_kernel_x"]
    r = K.check_trace(names, 1)
    assert (4, "_causal_conv1d_fwd_kernel") in r.forbidden
    assert 4 in r.matched and 2 in r.matched
    assert not r.ok


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not on this host")
def test_stage_weights_partition_the_text_model():
    from ppimpl.model.pp_patch import stage_weight_names

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    seen: dict[str, int] = {}
    for s in range(4):
        files = stage_weight_names(CKPT, K.stage_layers(s), s == 0, s == 3)
        for names in files.values():
            for n in names:
                assert n not in seen, n
                seen[n] = s
    text = {n for n in index if not n.startswith("model.visual.") and ".layers.45." not in n}
    assert set(seen) == text
    assert seen["model.language_model.embed_tokens.weight"] == 0
    assert seen["lm_head.weight"] == 3
    assert seen["model.language_model.layers.12.input_layernorm.weight"] == 1


def _specs_for_stage(stage: int):
    """The four layer specs a stage reports (values as vLLM computes them for this model, block size 8576)."""
    from dataclasses import replace

    import torch
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec, MLAAttentionSpec, SparseCacheRole, get_kv_quant_mode

    from ppimpl.model import kv_tensors

    B, kv = 8576, "fp8_e4m3"
    mla = MLAAttentionSpec(block_size=B, num_kv_heads=1, head_size=512, dtype=torch.float8_e4m3fn,
                           cache_dtype_str=kv, kv_quant_mode=get_kv_quant_mode(kv), is_index_group_leader=True)
    idx = replace(MLAAttentionSpec(block_size=B, num_kv_heads=1, head_size=132, dtype=torch.uint8,
                                   cache_role=SparseCacheRole.INDEXER), tokens_per_state=4)
    idx = replace(idx, storage_block_size=128)
    tail = KpoolTailSpec(block_size=4, num_kv_heads=2, head_size=128, head_size_v=0, dtype=torch.bfloat16,
                         sliding_window=4)
    mamba = MambaSpec(shapes=((3, 3 * 64 * 128), (64, 128, 128)), dtypes=(torch.bfloat16, torch.float32),
                      block_size=1048576, page_size_padded=B * 512, mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                      mamba_cache_mode="none")
    spec = {}
    for i in K.stage_layers(stage):
        for name in kv_tensors.layer_spec_names(i):
            spec[name] = mla if name.endswith(".attn") else idx if name.endswith("k_cache") else \
                tail if name.endswith("tail_cache") else mamba
    return spec


@pytest.mark.parametrize("stage", range(4))
def test_kv_layout_matches_the_contract(stage):
    from ppimpl.model import kv_tensors

    spec = _specs_for_stage(stage)
    per_block = kv_tensors.bytes_per_block(None, spec)
    n_dsa = sum(1 for i in K.stage_layers(stage) if i in K.DSA_LAYERS)
    assert per_block == 8576 * 545 * n_dsa          # 1_contract.md §1: 545 B per token per DSA layer
    assert kv_tensors.blocks_that_fit(per_block, int(149.84e9 - 49.66e9) + int(13.6e9), int(13.6e9), None) \
        == (7144 if n_dsa == 3 else 7144)          # capped at the contract's pool on every stage
    cfg, _ = kv_tensors.stage_kv_config(None, spec, 7144)
    kinds = [type(g.kv_cache_spec).__name__ for g in cfg.kv_cache_groups]
    assert kinds.count("MambaSpec") == 5 and len(kinds) == 7
    assert [n for g in cfg.kv_cache_groups for n in g.layer_names if n.endswith("tail_cache")] == \
        [n for n in spec if n.endswith("tail_cache")]
    local_kda = {n for n in spec if n.endswith(".self_attn")}
    grouped = {n for g in cfg.kv_cache_groups if type(g.kv_cache_spec).__name__ == "MambaSpec" for n in g.layer_names}
    assert grouped == local_kda


def test_kda_state_block_index_is_global():
    """A KDA layer's state block (state_blocks[g]) is the same group g whichever stage holds it."""
    from vllm.v1.kv_cache_interface import MambaSpec

    from ppimpl.model import kv_tensors

    index = {}
    for stage in range(4):
        cfg, _ = kv_tensors.stage_kv_config(None, _specs_for_stage(stage), 7144)
        kda = [g.layer_names for g in cfg.kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)]
        for g, names in enumerate(kda):
            for n in names:
                index[int(n.split(".")[3])] = g
    assert sorted(index) == [i for i in range(45) if i not in K.DSA_LAYERS]
    assert [i for i in sorted(index) if index[i] == 0] == [0, 6, 13, 20, 26, 33, 40]


def test_stage_reads_only_run_config_fields():
    """Every run-config field StageModel reads exists on RunConfig(model="glm53") (a misspelled field fails here
    instead of falling back), and the bench stand-in has the same fields."""
    from ppimpl.config import RunConfig
    from ppimpl.model.stage import bench_config, stage_settings

    cfg = RunConfig(model="glm53")
    st = stage_settings(cfg)
    assert st["mb_tokens"] == cfg.scheduler.mb_tokens and st["layer_split"] == (12, 11, 11, 11)
    one = RunConfig(model="glm53")
    one.scheduler.mb_tokens = 131072
    assert stage_settings(one)["mb_tokens"] == 131072
    assert stage_settings(bench_config(131072)).keys() == st.keys()


def test_ninja_check_warns_not_fails(monkeypatch):
    import shutil

    from ppimpl.model import stage

    monkeypatch.setattr(shutil, "which", lambda name: "/opt/venv/bin/ninja")
    import pytest
    with pytest.warns(UserWarning):
        assert stage.check_ninja() is False
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ninja")
    assert stage.check_ninja() is True
