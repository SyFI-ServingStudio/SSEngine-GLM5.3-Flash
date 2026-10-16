"""CPU tests of the M2 bench's trace attribution (bench/layers.py): scopes, ordinals, units, positions."""

import json

from bench import layers


def _ann(name, ts, dur, tid=1):
    return {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur, "tid": tid, "pid": 1}


def _launch(corr, ts, tid=1):
    return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1, "tid": tid, "pid": 1,
            "args": {"correlation": corr}}


def _kernel(corr, name, ts, dur):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "tid": 7, "pid": 0,
            "args": {"correlation": corr, "stream": 7}}


def test_trace_attribution(tmp_path):
    ev = [
        _ann("run0", 0, 1000),
        _ann("embed", 1, 5), _launch(1, 2),
        _ann("L0", 10, 400),
        _ann("hc_expand", 11, 3), _launch(2, 12),
        _ann("mhc_pre_op", 20, 5), _launch(3, 21),
        _ann("self_attn", 30, 100),
        _ann("in_proj_qkvbfg_a", 31, 5), _launch(4, 32),
        _ann("_forward", 40, 50), _launch(5, 41), _launch(6, 45), _launch(7, 50),
        _ann("mhc_fused_post_pre_op", 200, 5), _launch(8, 201),
        _ann("L1", 500, 400),
        _ann("mhc_fused_post_pre_op", 501, 5), _launch(9, 502),
        _ann("mhc_fused_post_pre_op", 600, 5), _launch(10, 601),
        _kernel(1, "vectorized_elementwise_kernel embedding", 100, 2),
        _kernel(2, "triton_hc_expand", 103, 3),
        _kernel(3, "tf32_hc_prenorm_gemm", 110, 10),
        _kernel(4, "nvjet_tst_128x256", 121, 20),
        _kernel(5, "causal_conv1d_fwd_kernel", 142, 5),
        _kernel(6, "vectorized_elementwise_kernel<FillFunctor<float>>", 148, 1),
        _kernel(7, "kernel_cutlass_kda_persistent", 150, 30),
        _kernel(8, "mega_mhc_kernel", 181, 4),
        _kernel(9, "mega_mhc_kernel", 300, 4),
        _kernel(10, "mega_mhc_kernel", 350, 4),
    ]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": ev}))
    runs = layers.parse_trace(path)
    assert list(runs) == [0]
    ks = runs[0]
    by_corr = {k["name"] + str(k["ts"]): k for k in ks}
    assert by_corr["triton_hc_expand103"]["unit"] == "hc_expand"
    assert by_corr["causal_conv1d_fwd_kernel142"]["scope"] == "self_attn#0/_forward#0"
    assert by_corr["mega_mhc_kernel350"]["scope"] == "mhc_fused_post_pre_op#1"
    assert by_corr["mega_mhc_kernel300"]["scope"] == "mhc_fused_post_pre_op#0"
    run = layers.summarize_run(ks)
    assert set(run) == {"input_head", "L0", "L1", "_forward"}
    assert run["input_head"]["span_us"] == 6        # 100 -> 106
    assert run["L0"]["span_us"] == 75               # 110 -> 185, hc_expand excluded
    pos = {k["name"] + str(k["ts"]): k["position"] for k in run["L0"]["kernels"]}
    assert pos["tf32_hc_prenorm_gemm110"] == "pp.first_kda_dense.attn_mhc_pre"
    assert pos["nvjet_tst_128x256121"] == "pp.first_kda_dense.kda.in_proj"
    assert pos["causal_conv1d_fwd_kernel142"] == "pp.first_kda_dense.kda.short_conv_prefill"
    assert pos["vectorized_elementwise_kernel<FillFunctor<float>>148"] == "pp.first_kda_dense.kda.state_gather"
    assert pos["kernel_cutlass_kda_persistent150"] == "pp.first_kda_dense.kda.chunk_prefill"
    assert pos["mega_mhc_kernel181"] == "pp.first_kda_dense.ffn_mhc_post_pre"
    pos1 = [k["position"] for k in run["L1"]["kernels"]]
    assert pos1 == ["pp.kda_dense.attn_mhc_post_pre", "pp.kda_dense.ffn_mhc_post_pre"]


def test_group_nth_rule():
    ks = [{"name": "nvjet_a", "scope": "self_attn#0/mla_attn#0/indexer#0", "ts": 1},
          {"name": "nvjet_b", "scope": "self_attn#0/mla_attn#0/indexer#0", "ts": 2},
          {"name": "nvjet_c", "scope": "self_attn#0/mla_attn#0/mla_attn#0", "ts": 3},
          {"name": "nvjet_d", "scope": "self_attn#0/mla_attn#0/mla_attn#0", "ts": 4},
          {"name": "elementwise_add", "scope": "mlp#0/experts#0", "ts": 5},
          {"name": "trtllm_moe_gemm", "scope": "mlp#0/experts#0", "ts": 6},
          {"name": "elementwise_mul", "scope": "mlp#0/experts#0", "ts": 7}]
    layers.assign_positions("dsa_moe", ks)
    assert [k["position"] for k in ks] == [
        "pp.dsa_moe.dsa.indexer.head_weights", "pp.dsa_moe.dsa.indexer.kpool_gate_score",
        "pp.dsa_moe.dsa.q_absorb", "pp.dsa_moe.dsa.v_up", "pp.dsa_moe.moe.input_glue",
        "pp.dsa_moe.moe.routed_rank0.fused_moe", "pp.dsa_moe.moe.combine_glue"]


def test_targets_and_stage_composition():
    t = layers.load_targets()
    assert t[(8192, "dsa_moe")]["layer_ms"] > 5.8
    for L in layers.LENGTHS:
        assert sum(t[(L, "dsa_moe")]["moe_per_expert_batches"]) == 8 * L
    # README stage 1 at 8k = 2 x dsa_moe + 9 x kda_moe
    s1 = sum(n * t[(8192, lt)]["layer_ms"] for lt, n in layers.STAGE_COMPOSITION[1].items())
    assert abs(s1 - 54.1) < 0.1
