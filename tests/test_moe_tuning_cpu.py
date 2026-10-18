"""ppimpl/model/moe_tuning.py: the MoE's serving tactic buckets and the served routing it is tuned on (CPU)."""

import json
import random

import pytest
import torch

from ppimpl.model import moe_tuning as mt
from ppimpl.model.stage import GRAPH_BUCKETS


def test_buckets_are_the_graph_buckets_then_powers_of_two():
    b = mt.serving_buckets(131072)
    assert b[:8] == (1, 2, 4, 8, 16, 32, 64, 128)
    assert {g for g in GRAPH_BUCKETS if g <= mt.TUNED_TOP} <= set(b)
    assert b[-5:] == (8192, 16384, 32768, 65536, 131072)
    assert mt.serving_buckets(8192)[-1] == 8192


@pytest.mark.parametrize("x,want", [(0, 1), (3, 4), (100, 128), (129, 256), (4096, 4096), (4097, 4352),
                                    (6000, 6144), (8191, 8192), (8193, 16384), (131072, 131072), (200000, 131072)])
def test_a_token_count_maps_to_the_bucket_at_or_above_it(x, want):
    assert mt.map_to_serving_bucket(x, 131072) == want
    assert mt.serving_bucket_mapper(131072) is mt.serving_bucket_mapper(131072)    # stable for FlashInfer's cache


def test_install_buckets_replaces_the_moe_tuning_bucket_functions():
    from flashinfer.fused_moe.shared import tuning as t

    saved = (t.get_hybrid_num_tokens_buckets, t.make_hybrid_bucket_mapper, mt._ORIG)
    try:
        mt._ORIG = None
        mt.install_buckets()
        assert 4352 in t.get_hybrid_num_tokens_buckets(131072)
        assert t.make_hybrid_bucket_mapper(131072)(4097) == 4352
    finally:
        t.get_hybrid_num_tokens_buckets, t.make_hybrid_bucket_mapper, mt._ORIG = saved


def test_scale_hist_keeps_rows_and_caps_at_tokens():
    h = [1088] + [100] * 287
    for tokens in (128, 4096, 6144, 8192):
        s = mt.scale_hist(h, tokens)
        assert sum(s) == tokens * 8 and max(s) <= tokens and len(s) == 288
    base = mt.scale_hist(h, 4096)
    assert mt.scale_hist(base, 8192) == [2 * x for x in base]      # exact doubling


def test_fold_is_the_sorted_mean_with_the_row_total_kept():
    a, b = [5, 0, 3], [1, 4, 3]
    f = mt.fold([a, b])
    assert f == [5, 3, 0]       # sorted columns (5, 3, 0) and (4, 3, 1): sums 9, 6, 1 // 2 = 4, 3, 0; 1 left -> first


def test_forced_logits_route_exactly_the_histogram_under_the_layer_bias():
    rnd = random.Random(0)
    tokens = 512
    w = [rnd.paretovariate(1.2) for _ in range(288)]
    h = mt.scale_hist([int(x * 100) + 1 for x in w], tokens)
    logits = mt.forced_logits(h, tokens, torch.float32, "cpu")
    bias = 12.4 + torch.rand(288) * 0.52            # the checkpoint's widest spread is 0.58
    ids = (logits.sigmoid() + bias).topk(8, dim=1).indices
    assert torch.bincount(ids.flatten(), minlength=288).tolist() == h
    assert all(len(set(r)) == 8 for r in ids.tolist())


def test_routing_profile_folds_each_row_count_and_profile_for_scales_the_nearest(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lines = [{"stage": 2, "mb": i, "counts": [mt.scale_hist([10 + j % 7 + (100 if j == i else 0) for j in range(288)],
                                                            4096)] * 2} for i in range(3)]
    lines.append({"stage": 2, "mb": 9, "counts": [mt.scale_hist([1] * 288, 128)]})
    (run / "experts_stage2.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    prof = mt.routing_profile([run])
    assert sorted(prof) == [128, 4096] and sum(prof[4096]) == 4096 * 8
    assert prof[4096] == sorted(prof[4096], reverse=True)
    h = mt.profile_for(prof, 6144)                  # from 4096 (>= 512 rows), scaled
    assert sum(h) == 6144 * 8
    assert mt.profile_for(prof, 100) == mt.scale_hist(prof[4096], 100)


def test_real_routing_swaps_the_logit_initializer_while_tuning_only():
    from flashinfer.fused_moe.shared import tuning as t

    orig = t.autotuner_initializer_rand
    prof = {4096: mt.scale_hist([3] * 288, 4096)}
    with mt.real_routing(prof) as used:
        x = t.autotuner_initializer_rand((256, 288), torch.float32, "cpu")
        assert x.shape == (256, 288)
        assert torch.bincount(x.topk(8, dim=1).indices.flatten(), minlength=288).tolist() == used[256]
    assert t.autotuner_initializer_rand is orig


def test_merge_fills_the_serving_buckets_from_the_base_and_takes_only_faster_ab_tactics(tmp_path):
    from bench.tune_tactics import merge

    op = "flashinfer::trtllm_fp4_block_scale_moe"
    key = lambda t: f"('{op}', 'MoERunner', (({t}, 4096), ({t}, 288)), ('Module', 8))"     # noqa: E731
    base = {"_metadata": {"v": 1}, "('fp4_gemm', 'X', ((8, 2),))": ["X", [1], []]}
    base.update({key(t): ["MoERunner", [t // 64, 1], []] for t in (128, 2560, 4096, 8192, 131072)})
    (tmp_path / "base.json").write_text(json.dumps(base))
    ab = {"cases": [
        {"tokens": 6144, "routing": "real", "tactics": [[[128, 41], 1.0], [[128, 1], 1.2]]},   # base 8192's [128, 1]
        {"tokens": 4352, "routing": "real", "tactics": [[[128, 1], 1.0], [[68, 1], 1.1]]},     # base already best
        {"tokens": 4608, "routing": "synth3", "tactics": [[[1, 1], 0.1], [[128, 1], 1.0]]}]}   # other routing
    (tmp_path / "ab.json").write_text(json.dumps(ab))
    res = merge(tmp_path / "base.json", [tmp_path / "ab.json"], "real", tmp_path / "out.json")
    out = json.loads((tmp_path / "out.json").read_text())
    assert out[key(6144)][1] == [128, 41] and out[key(4352)][1] == [128, 1] and out[key(4608)][1] == [128, 1]
    assert out[key(2304)][1] == [40, 1]                      # hybrid 2560's tactic
    assert set(res["changed"]) == {6144}
    assert all(out[k] == v for k, v in base.items())     # every base key unchanged
    assert {k for k in out if op in k} == {key(t) for t in mt.serving_buckets(131072)}
