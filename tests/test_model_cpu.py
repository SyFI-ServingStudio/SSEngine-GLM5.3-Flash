"""CPU checks of the model lane: kernel manifest, top-k table, trace check, per-stage weight selection."""

import json
from pathlib import Path

import pytest

from ssengine import kernels as K
from ssengine.config import CHECKPOINT

CKPT = Path(CHECKPOINT)


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
    names = ["nvjet_tst_128x64", "kda_conv_qkv_kernel", "kda_conv_state_kernel", "_causal_conv1d_fwd_kernel",
             "causal_conv1d_channellast_fwd_kernel", "weird_kernel_x"]
    r = K.check_trace(names, 1)
    assert (4, "_causal_conv1d_fwd_kernel") in r.forbidden
    assert (4, "causal_conv1d_channellast_fwd_kernel") in r.forbidden
    assert 4 in r.matched and 2 in r.matched
    assert not r.ok


def test_check_trace_assigns_each_kernel_to_its_own_config():
    """Real kernel names from a traced stage 0 (7940): each goes to the config that launches it, a kernel shared by two
    positions counts for both, and the Triton `kernel` of config 13 is listed as unverifiable, never matched."""
    names = [
        "fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen",
        "kernel_cutlass__pkd_TiledMMA_ThrLayoutVMNK11110000",
        "kda_conv_qkv_kernel",
        "mhc_post_tilelang_kernel",
        "void deep_gemm::sm100_mega_mhc_impl<4096u, 16u, 148u, false, true, false, 0u>(CUtensorMap_st)",
        "void moe::dev::routing::routingIndicesCoopKernel<moe::dev::routing::routingCustom::KernelParams<float, "
        "__nv_bfloat16, 384, 8, moe::dev::routing::TopKExpertSelect<moe::dev::routing::Foo> > >(Params)",
        "void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, int, int)",
        "void vllm::cvt_fp16_to_fp4_sf_major<__nv_bfloat16, false>(int, int, int, int, int)",
        "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100"
        "Sm100BlockScaledPersistentDenseGemmKernel",
        "sparse_mla_index_remap_kernel",
        "kernel",
        "void vllm::scaled_fp8_quant_kernel_strided_group_shape<c10::BFloat16, c10::Float8_e4m3fn, true, true>(int)",
    ]
    r = K.check_trace(names, 0)
    m = {c: set(v) for c, v in r.matched.items()}
    assert m[25] == {names[0]} and m[5] == {names[1]} and m[4] == {names[2]}
    assert m[33] == {names[3]} and m[7] == {names[4]} and names[5] in m[30]
    assert 20 not in m and 21 not in m                       # MoE TopKExpertSelect is not the indexer top-k
    assert m[8] == m[10] == {names[6]} and m[29] == {names[7]}
    assert m[9] == m[11] == {names[8]} and m[24] == {names[9]}
    assert 13 not in m and r.unverifiable == {13: ["kernel"]} and 13 not in r.missing
    assert "unverifiable by name: config 13" in r.summary()
    assert names[10] not in r.glue and names[11] in r.glue and not r.unknown
    assert 13 in K.check_trace(names[:10], 0).missing        # no candidate kernel at all: missing


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not on this host")
def test_stage_weights_partition_the_text_model():
    from ssengine.model.pp_patch import stage_weight_names

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

    from ssengine.model import kv_tensors

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
    from ssengine.model import kv_tensors

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

    from ssengine.model import kv_tensors

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
    from ssengine.config import RunConfig
    from ssengine.model.stage import bench_config, stage_settings

    cfg = RunConfig(model="glm53")
    st = stage_settings(cfg)
    assert st["mb_tokens"] == cfg.scheduler.mb_tokens and st["layer_split"] == (12, 11, 11, 11)
    one = RunConfig(model="glm53")
    one.scheduler.mb_tokens = 131072
    assert stage_settings(one)["mb_tokens"] == 131072
    assert stage_settings(bench_config(131072)).keys() == st.keys()


def test_ninja_check_warns_not_fails(monkeypatch):
    import shutil

    from ssengine.model import stage

    monkeypatch.setattr(shutil, "which", lambda name: "/opt/venv/bin/ninja")
    import pytest
    with pytest.warns(UserWarning):
        assert stage.check_ninja() is False
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ninja")
    assert stage.check_ninja() is True


def test_load_tactics_missing_file_is_reported(tmp_path):
    from ssengine.model.stage import load_tactics

    r = load_tactics(tmp_path / "none.json")
    assert r["source"] == "missing" and "does not exist" in r["reason"]


def test_kv_groups_place_layers_by_the_design_split(monkeypatch, caplog):
    """vLLM's GLM-5 group function counts KDA state groups per stage from `get_pp_indices`; it must see the design's
    split [12, 11, 11, 11], not vLLM's default [11, 11, 12, 11] (which logs "unevenly partitioned")."""
    from vllm.v1.core import kv_cache_utils as kcu
    from vllm.distributed import utils as dist_utils

    from ssengine.model import kv_tensors as kt

    seen, design = [], kt.design_pp_indices

    def spy(num_layers, rank, size):
        seen.append(design(num_layers, rank, size))
        return seen[-1]

    monkeypatch.setattr(kt, "design_pp_indices", spy)
    mla = [f"model.layers.{i}.self_attn" for i in range(45) if i in K.DSA_LAYERS]
    kda = [f"model.layers.{i}.linear_attn" for i in range(45) if i not in K.DSA_LAYERS]
    vllm_split = dist_utils.get_pp_indices
    with kt.design_split():
        n = kcu._pp_balanced_mamba_group_count(kt.group_config_shim(None, None), kda, mla)
    assert seen == [(0, 12), (12, 23), (23, 34), (34, 45)]
    assert dist_utils.get_pp_indices is vllm_split
    assert n == 5      # stage KDA/DSA [9/3, 9/2, 8/3, 8/3]: ceil(9/2)
    assert "unevenly" not in caplog.text
