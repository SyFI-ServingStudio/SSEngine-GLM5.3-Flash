"""Which KDA prefill kernel drifts with context: layers 0-2's recurrences on identical inputs, four kernels.

    python -m bench.kvaudit.kda_probe --length 131072      # 1 GPU

The audit (bench/kvaudit) finds ours vs vLLM's KDA states diverging with length in a few heads (layer 1 head 45:
0.03 at 8k, 0.28 at 128k, 0.90 at 512k; layer 2 head 56: 0.09, 0.42, 0.82) while the median head and ours vs ours
(other chunking) stay at noise. Ours runs FlashInfer persistent `recurrent_kda`; the fork runs FlashKDA. This runs
stage 0's layers 0-2 on the first L enwik9 tokens as one microbatch, captures each layer's exact inputs of our
FlashInfer call (q, k, v after our conv, raw gate, raw beta) and feeds the same inputs to:

- `flashinfer`: FlashInfer persistent `fwd` (ours, as captured)
- `flashkda`: FlashKDA `_flashkda_C.fwd` (the fork's prefill on SM100)
- `triton_chunk`: the fork's `chunk_kda_with_fused_gate` (its non-FlashKDA prefill)
- `recurrent`: the fork's `fused_recurrent_kda`, token by token with an FP32 state

and, on the heads named in --fp64-heads over all L tokens, a float64 PyTorch recurrence: the reference. Reports per
layer and kernel the output's and the final state's relative error against `recurrent` (all heads; 16 position
bins) and against float64 (the named heads). Writes measured/kvaudit/kda_probe_<job>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace as NS

import torch

from bench.kvaudit import common as C

JOB = os.environ.get("SLURM_JOB_ID", "local")
KDA_PROBE_LAYERS = (0, 1, 2)


class _Captured(Exception):
    pass


def rel(x: torch.Tensor, a: torch.Tensor) -> float:
    x, a = x.double(), a.double()
    return float((x - a).norm() / a.norm())


def bins(x: torch.Tensor, a: torch.Tensor, n: int = 16) -> list[float]:
    """Relative error per position bin (dim 0 = token)."""
    return [round(rel(xb, ab), 5) for xb, ab in zip(x.tensor_split(n), a.tensor_split(n))]


def fp64_head(q, k, v, g_raw, b_raw, a_log, dt_bias, lb: float, h: int):
    """KDA on head h, float64, from the kernel inputs [T, H, D]: q, k l2-normalized (q times D^-0.5); per key channel
    decay g = lb * sigmoid(exp(A_log) * (g_raw + dt_bias)); beta = sigmoid(b_raw).
    S <- diag(exp g) S;  S <- S + beta k (v - k^T S)^T;  o = q^T S.   S is [K, V]."""
    T, _, D = q.shape
    qq = torch.nn.functional.normalize(q[:, h].double(), dim=-1) * D ** -0.5
    kk = torch.nn.functional.normalize(k[:, h].double(), dim=-1)
    vv = v[:, h].double()
    gg = (lb * torch.sigmoid(a_log[h].double().exp() * (g_raw[:, h].double() + dt_bias[h].double()))).exp()
    bb = torch.sigmoid(b_raw[:, h].double()).tolist()   # host floats: no sync per step
    S = torch.zeros(D, D, dtype=torch.float64, device=q.device)
    out = torch.empty(T, D, dtype=torch.float64, device=q.device)
    for t in range(T):
        S.mul_(gg[t][:, None])
        S.addr_(kk[t], vv[t] - kk[t] @ S, alpha=bb[t])
        torch.mv(S.t(), qq[t], out=out[t])
    return out, S


def capture_inputs(m, L: int) -> list[dict]:
    """Run layers 0-2 on L tokens; per KDA layer, the inputs and output of our FlashInfer call."""
    import flashinfer.kda_kernels.persistent.forward as fip
    from vllm.forward_context import set_forward_context

    B = 8576
    nkv = -(-L // B)
    plan = NS(mb_id=0, chunks=[NS(req_id="p", start=0, num_tokens=L, kv_blocks=list(range(1, 1 + nkv)),
                                  state_blocks=list(range(1 + nkv, 7 + nkv)), is_last=True, token_ids=C.prompt(L))])
    caps: list[dict] = []
    orig = fip.fwd

    def capture(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None,
                cu_seqlens=None, plan=None):
        assert bool((initial_state == 0).all())
        orig(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=initial_state,
             final_state=final_state, cu_seqlens=cu_seqlens, plan=plan)
        caps.append(dict(q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(), lb=lower_bound,
                         cu=cu_seqlens.clone(), out=out.clone(), final=final_state[0].clone()))
        if len(caps) == len(KDA_PROBE_LAYERS):
            raise _Captured

    fip.fwd = capture
    try:
        with torch.inference_mode():
            b = m.prepare(plan)
            with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                x = m.embed(b.token_ids)
                residual = post = comb = None
                for layer in m.layers[:max(KDA_PROBE_LAYERS) + 1]:
                    x, residual, post, comb = layer(b.positions, x, residual, post, comb)
    except _Captured:
        torch.cuda.synchronize()
    finally:
        fip.fwd = orig
    assert len(caps) == len(KDA_PROBE_LAYERS), len(caps)
    return caps


def run_layer(attn, cap: dict, fp64_heads: list[int], dev) -> dict:
    import vllm._flashkda_C  # noqa: F401
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate, fused_recurrent_kda

    q, k, v, g, beta, lb, cu = (cap[n] for n in ("q", "k", "v", "g", "beta", "lb", "cu"))
    _, T, H, D = q.shape
    outs = {"flashinfer": (cap["out"].reshape(T, H, D), cap["final"])}
    zero = lambda: torch.zeros(1, H, D, D, dtype=torch.float32, device=dev)  # noqa: E731
    secs = {}
    with torch.inference_mode():
        t0 = time.time()
        ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, 1), dtype=torch.uint8, device=dev)
        o = torch.empty(1, T, H, D, dtype=q.dtype, device=dev)
        fin = torch.empty(1, H, D, D, dtype=torch.float32, device=dev)
        torch.ops._flashkda_C.fwd(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta, D ** -0.5, o,
                                  ws, attn.A_log.view(-1), attn.dt_bias.view(-1, D), lb, zero(), fin, cu, None, None)
        torch.cuda.synchronize()
        outs["flashkda"] = (o.reshape(T, H, D), fin[0])
        secs["flashkda"] = time.time() - t0
        del ws

        t0 = time.time()
        o, fin = chunk_kda_with_fused_gate(q=q, k=k, v=v, raw_g=g, beta=torch.sigmoid(beta.float()), A_log=attn.A_log,
                                           g_bias=attn.dt_bias, initial_state=zero(), output_final_state=True,
                                           use_qk_l2norm_in_kernel=True, cu_seqlens=cu, safe_gate=True,
                                           lower_bound=lb)
        torch.cuda.synchronize()
        outs["triton_chunk"] = (o.reshape(T, H, D), fin[0])
        secs["triton_chunk"] = time.time() - t0

        t0 = time.time()
        st = zero()
        o, _ = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=st, use_qk_l2norm_in_kernel=True,
                                   cu_seqlens=cu, ssm_state_indices=torch.zeros(1, dtype=torch.int32, device=dev),
                                   sigmoid_beta=True, a_log=attn.A_log, g_bias=attn.dt_bias, compute_gate=True,
                                   lower_bound=lb)
        torch.cuda.synchronize()
        outs["recurrent"] = (o.reshape(T, H, D), st[0])
        secs["recurrent"] = time.time() - t0

        res: dict = {"T": T, "H": H, "D": D, "seconds": {n: round(s, 2) for n, s in secs.items()}}
        ref_o, ref_s = outs["recurrent"]
        # the kernels' state layouts may differ ([K, V] vs [V, K]): pick each one's orientation against `recurrent`
        orient = {}
        for name, (o, s) in outs.items():
            t = rel(s.transpose(-1, -2), ref_s) < rel(s, ref_s)
            orient[name] = t
            s = s.transpose(-1, -2) if t else s
            ho = [rel(o[:, h], ref_o[:, h]) for h in range(H)]
            hs = [rel(s[h], ref_s[h]) for h in range(H)]
            res[name] = {"vs_recurrent": {
                "out_rel": round(rel(o, ref_o), 6), "out_bins": bins(o, ref_o), "state_rel": round(rel(s, ref_s), 6),
                "state_transposed": t, "head_out_rel_max": round(max(ho), 5), "head_out_worst": ho.index(max(ho)),
                "head_state_rel_max": round(max(hs), 5), "head_state_worst": hs.index(max(hs)),
                "heads": {h: {"out_rel": round(ho[h], 5), "state_rel": round(hs[h], 5),
                              "out_bins": bins(o[:, h], ref_o[:, h])} for h in fp64_heads}}}
        fl, fk = outs["flashinfer"], outs["flashkda"]
        fks = fk[1].transpose(-1, -2) if orient["flashkda"] != orient["flashinfer"] else fk[1]
        res["flashinfer_vs_flashkda"] = {
            "out_rel": round(rel(fl[0], fk[0]), 6), "out_bins": bins(fl[0], fk[0]), "state_rel": round(rel(fl[1], fks), 6),
            "heads": {h: {"out_rel": round(rel(fl[0][:, h], fk[0][:, h]), 5),
                          "state_rel": round(rel(fl[1][h], fks[h]), 5)} for h in fp64_heads}}

        a_log, dt_bias = attn.A_log.reshape(-1), attn.dt_bias.reshape(H, D)
        res["fp64"] = {}
        for h in fp64_heads:
            t0 = time.time()
            o64, s64 = fp64_head(q[0], k[0], v[0], g[0].reshape(T, H, D), beta[0].reshape(T, H), a_log, dt_bias, lb, h)
            r = {"seconds": round(time.time() - t0, 1),
                 "decay_mean": round(float((lb * torch.sigmoid(a_log[h].double().exp() * (
                     g[0].reshape(T, H, D)[:, h].double() + dt_bias[h].double()))).mean()), 5)}
            for name, (o, s) in outs.items():
                s = s.transpose(-1, -2) if orient[name] else s
                # `recurrent`'s layout is assumed [K, V]; the float64 state is [K, V]; report both orientations
                r[name] = {"out_rel": round(rel(o[:, h], o64), 6), "out_bins": bins(o[:, h], o64),
                           "state_rel": round(min(rel(s[h], s64), rel(s[h].t(), s64)), 6)}
            res["fp64"][h] = r
            print(f"[kda-probe] fp64 head {h}: {json.dumps(r)}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--fp64-heads", default="0:10,1:45,1:7,2:56",
                    help="layer:head pairs for the float64 reference (the audit's worst heads + a median one)")
    a = ap.parse_args()
    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    L, B = a.length, 8576
    # all of stage 0 (layers 0-11): a shorter range leaves a KDA cache group empty (StageMetadata indexes it)
    m = StageModel(bench_config(L), 0, dev, first=True, last=False,
                   num_blocks=1 + -(-L // B) + 6, max_model_len=L + B, warmup=False, cuda_graphs=False)
    heads: dict[int, list[int]] = {}
    for p in a.fp64_heads.split(","):
        layer, h = map(int, p.split(":"))
        heads.setdefault(layer, []).append(h)
    t0 = time.time()
    caps = capture_inputs(m, L)
    print(f"[kda-probe] L={L}: layers {KDA_PROBE_LAYERS} captured in {time.time() - t0:.1f} s; q "
          f"{tuple(caps[0]['q'].shape)} {caps[0]['q'].dtype}, g {tuple(caps[0]['g'].shape)} {caps[0]['g'].dtype}, "
          f"beta {tuple(caps[0]['beta'].shape)} {caps[0]['beta'].dtype}, lower bound {caps[0]['lb']}", flush=True)
    out = {"length": L, "job": JOB, "layers": {}}
    C.OUT.mkdir(parents=True, exist_ok=True)
    path = C.OUT / f"kda_probe_{JOB}.json"
    for i, layer in enumerate(KDA_PROBE_LAYERS):
        attn = m.layers[layer].self_attn
        assert m.layers[layer].layer_kind == "kda"
        r = run_layer(attn, caps[i], heads.get(layer, []), dev)
        out["layers"][layer] = r
        print(f"[kda-probe] layer {layer}: " + json.dumps({n: r[n]["vs_recurrent"] for n in
                                                          ("flashinfer", "flashkda", "triton_chunk")})[:2000], flush=True)
        path.write_text(json.dumps(out, indent=1))
        caps[i] = None
        torch.cuda.empty_cache()
    print(f"[kda-probe] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
