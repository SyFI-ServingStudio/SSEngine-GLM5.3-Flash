"""Which KDA prefill kernel is closer to the exact recurrence: layers 0-2 on identical inputs, against float64.

    python -m bench.kvaudit.kda_probe --length 131072 [--chunk 16384]      # 1 GPU

The audit (bench/kvaudit) finds ours vs vLLM's KDA states diverging with length in a few heads (layer 1 head 45:
0.03 at 8k, 0.28 at 128k, 0.90 at 512k; layer 2 head 56: 0.09, 0.42, 0.82). Meanwhile the median head and ours vs
ours (other chunking) stay at noise. Ours runs FlashInfer persistent `recurrent_kda` (the design row); the fork runs
FlashKDA.

This runs stage 0 on the first L enwik9 tokens in --chunk microbatches (ours is chunk-invariant bitwise). For each
of KDA layers 0-2 it captures our FlashInfer call's exact inputs: q, k, v after our conv, the raw gate and the raw
beta. Every kernel gets the same per-layer inputs (a fresh copy each, since a kernel may write its inputs; the
JSON reports which ones it wrote), so this measures the kernels and not their inputs. The kernels
run over the whole sequence from a zero state:

- `flashinfer`: FlashInfer persistent `fwd` (ours; its chunked outputs concatenated, its final state)
- `flashkda`: FlashKDA `_flashkda_C.fwd` (the fork's prefill on SM100)
- `triton_chunk`: the fork's `chunk_kda_with_fused_gate` (its non-FlashKDA prefill)
- the reference: the model's recurrence in float64, token by token, over all heads. It uses the model's bounded gate
  `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` (GLM-5.3 `linear_lower_bound`; part of the model, not a
  kernel clamp), beta = sigmoid(beta), q and k l2-normalized, scale D^-0.5.

It reports, per layer and kernel against float64, the output's relative error (all heads, 16 position bins, per
head) and the final state's relative error (all heads, per head). The --heads entries get their per-bin output
errors and decay statistics. Writes measured/kvaudit/kda_probe_<job>.json.
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
NBINS = 16


class ErrStats:
    """Streaming ||x - ref||^2 and ||ref||^2 per position bin and head; x, ref [T, H, D] fed in token order."""

    def __init__(self, T: int, H: int, dev):
        self.T, self.edges = T, [T * i // NBINS for i in range(NBINS + 1)]
        self.err = torch.zeros(NBINS, H, dtype=torch.float64, device=dev)
        self.ref = torch.zeros(NBINS, H, dtype=torch.float64, device=dev)

    def add(self, t0: int, x: torch.Tensor, ref: torch.Tensor) -> None:
        t1 = t0 + x.shape[0]
        for b in range(NBINS):
            lo, hi = max(t0, self.edges[b]), min(t1, self.edges[b + 1])
            if hi > lo:
                xs, rs = x[lo - t0:hi - t0].double(), ref[lo - t0:hi - t0].double()
                self.err[b] += (xs - rs).square().sum(dim=(0, 2))
                self.ref[b] += rs.square().sum(dim=(0, 2))

    def summary(self, heads: list[int]) -> dict:
        e, r = self.err, self.ref
        head = (e.sum(0) / r.sum(0)).sqrt()
        return {"out_rel": round(float((e.sum() / r.sum()).sqrt()), 6),
                "out_bins": [round(float(v), 5) for v in (e.sum(1) / r.sum(1)).sqrt()],
                "head_out_rel_max": round(float(head.max()), 5), "head_out_worst": int(head.argmax()),
                "head_out_rel_median": round(float(head.median()), 5),
                "heads": {h: {"out_rel": round(float(head[h]), 5),
                              "out_bins": [round(float(v), 5) for v in (e[:, h] / r[:, h]).sqrt()]} for h in heads}}


def capture_inputs(m, L: int, chunk: int) -> list[dict]:
    """Run stage 0's layers 0-2 on L tokens in `chunk`-token microbatches. Per KDA layer, our FlashInfer call's
    inputs (on the host, concatenated), its output (concatenated) and the final state."""
    import flashinfer.kda_kernels.persistent.forward as fip
    from vllm.forward_context import set_forward_context

    B = 8576
    nkv = -(-L // B)
    kv, st = list(range(1, 1 + nkv)), list(range(1 + nkv, 7 + nkv))
    toks = C.prompt(L)
    parts: list[dict] = [{n: [] for n in ("q", "k", "v", "g", "beta", "out")} for _ in KDA_PROBE_LAYERS]
    finals: list = [None] * len(KDA_PROBE_LAYERS)
    meta: dict = {}
    seen = [0]
    orig = fip.fwd

    def capture(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None,
                cu_seqlens=None, plan=None):
        i = seen[0]
        for n, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):    # before the call: inputs as given
            parts[i][n].append(x[0].to("cpu"))
        orig(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=initial_state,
             final_state=final_state, cu_seqlens=cu_seqlens, plan=plan)
        parts[i]["out"].append(out[0].to("cpu"))
        finals[i] = final_state[0].clone()
        meta.update(lb=float(lower_bound), scale=float(scale), dtypes={n: str(x.dtype) for n, x in (
            ("q", q), ("g", g), ("beta", beta), ("out", out))})
        seen[0] += 1

    fip.fwd = capture
    try:
        with torch.inference_mode():
            for s in range(0, L, chunk):
                n = min(chunk, L - s)
                plan = NS(mb_id=s // chunk, chunks=[NS(req_id="p", start=s, num_tokens=n, kv_blocks=kv,
                                                       state_blocks=st, is_last=s + n == L, token_ids=toks[s:s + n])])
                seen[0] = 0
                b = m.prepare(plan)
                # layers 0-2 run whole: each KDA layer writes its state back after the kernel call, and the next
                # microbatch starts from it
                with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                    x = m.embed(b.token_ids)
                    residual = post = comb = None
                    for layer in m.layers[:max(KDA_PROBE_LAYERS) + 1]:
                        x, residual, post, comb = layer(b.positions, x, residual, post, comb)
                assert seen[0] == len(KDA_PROBE_LAYERS), seen[0]
            torch.cuda.synchronize()
    finally:
        fip.fwd = orig
    caps = []
    for i in range(len(KDA_PROBE_LAYERS)):
        c = {n: torch.cat(v) for n, v in parts[i].items()}
        c.update(final=finals[i], **meta)
        caps.append(c)
        parts[i] = None
    return caps


def fp64_reference(q, k, v, g, beta, a_log, dt_bias, lb, scale, on_out, block: int = 2048, l2norm: bool = True):
    """The model's KDA recurrence in float64 over all heads, token by token: S [H, K, V];
    S <- diag(exp(gate)) S;  S <- S + beta k (v - S^T k)^T;  o = S^T q. Inputs [T, H, D] (beta [T, H]) on the GPU in
    their kernel dtypes; converted per `block` tokens. `on_out(t0, o_block)` receives the output in float32."""
    T, H, D = q.shape
    S = torch.zeros(H, D, D, dtype=torch.float64, device=q.device)
    A = a_log.double().exp().view(H, 1)
    bias = dt_bias.double().view(H, D)
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        qq, kk = q[t0:t1].double(), k[t0:t1].double()
        if l2norm:
            qq, kk = torch.nn.functional.normalize(qq, dim=-1), torch.nn.functional.normalize(kk, dim=-1)
        qq = qq * scale
        vv = v[t0:t1].double()
        dec = (lb * torch.sigmoid(A * (g[t0:t1].double() + bias))).exp()     # [n, H, K]
        bb = torch.sigmoid(beta[t0:t1].double())                              # [n, H]
        out = torch.empty(t1 - t0, H, D, dtype=torch.float64, device=q.device)
        for j in range(t1 - t0):
            S.mul_(dec[j].unsqueeze(-1))
            u = vv[j] - torch.bmm(kk[j].unsqueeze(1), S).squeeze(1)           # [H, V]
            S.baddbmm_((bb[j].unsqueeze(-1) * kk[j]).unsqueeze(-1), u.unsqueeze(1))
            out[j] = torch.bmm(qq[j].unsqueeze(1), S).squeeze(1)
        on_out(t0, out)
    return S


def run_layer(attn, cap: dict, heads: list[int], dev) -> dict:
    import vllm._flashkda_C  # noqa: F401
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate

    T, H, D = cap["q"].shape
    lb, scale = cap["lb"], cap["scale"]
    q, k, v, g = (cap[n].to(dev).unsqueeze(0) for n in ("q", "k", "v", "g"))
    beta = cap["beta"].to(dev).unsqueeze(0)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    zero = lambda: torch.zeros(1, H, D, D, dtype=torch.float32, device=dev)  # noqa: E731
    outs = {"flashinfer": (cap["out"], cap["final"])}       # output on the host
    secs: dict = {}
    mutated: dict = {}
    work: list = []

    def ins():
        """Fresh copies of the inputs for one kernel (a kernel may write its inputs); remembered for changed()."""
        work[:] = [x.clone() for x in (q, k, v, g, beta)]
        return work

    def changed() -> list[str]:
        torch.cuda.synchronize()
        return [n for n, a, b in zip("q k v g beta".split(), work, (q, k, v, g, beta)) if not torch.equal(a, b)]
    with torch.inference_mode():
        t0 = time.time()
        ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, 1), dtype=torch.uint8, device=dev)
        o = torch.empty(1, T, H, D, dtype=q.dtype, device=dev)
        fin = torch.empty(1, H, D, D, dtype=torch.float32, device=dev)
        torch.ops._flashkda_C.fwd(*ins(), scale, o, ws, attn.A_log.view(-1), attn.dt_bias.view(-1, D), lb,
                                  zero(), fin, cu, None, None)
        mutated["flashkda"] = changed()
        torch.cuda.synchronize()
        outs["flashkda"] = (o[0].cpu(), fin[0])
        secs["flashkda"] = time.time() - t0
        del ws, o

        t0 = time.time()
        qi, ki, vi, gi, bi = ins()
        o, fin = chunk_kda_with_fused_gate(q=qi, k=ki, v=vi, raw_g=gi, beta=torch.sigmoid(bi.float()), A_log=attn.A_log,
                                           g_bias=attn.dt_bias, scale=scale, initial_state=zero(),
                                           output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                                           safe_gate=True, lower_bound=lb)
        mutated["triton_chunk"] = changed()
        del qi, ki, vi, gi, bi
        torch.cuda.synchronize()
        outs["triton_chunk"] = (o.reshape(T, H, D).cpu(), fin[0])
        secs["triton_chunk"] = time.time() - t0
        del o

        t0 = time.time()
        stats = {n: ErrStats(T, H, dev) for n in outs}
        pair = ErrStats(T, H, dev)          # flashinfer vs flashkda

        def on_out(s: int, ref: torch.Tensor) -> None:
            e = s + ref.shape[0]
            got = {n: o[s:e].to(dev, non_blocking=True) for n, (o, _) in outs.items()}
            for n, x in got.items():
                stats[n].add(s, x, ref)
            pair.add(s, got["flashinfer"], got["flashkda"])

        S64 = fp64_reference(q[0], k[0], v[0], g[0].view(T, H, D), beta[0].view(T, H), attn.A_log.reshape(-1),
                             attn.dt_bias, lb, scale, on_out)
        secs["fp64"] = time.time() - t0

        a_log, bias = attn.A_log.reshape(-1).double(), attn.dt_bias.double().view(H, D)
        dec = torch.zeros(H, dtype=torch.float64, device=dev)
        for s in range(0, T, 8192):
            dec += (lb * torch.sigmoid(a_log.exp().view(H, 1) * (g[0, s:s + 8192].view(-1, H, D).double() + bias))
                    ).sum(dim=(0, 2))
        dec /= T * D                    # mean per-token log-decay per head (0 = no forgetting)

        res: dict = {"T": T, "H": H, "D": D, "lower_bound": lb, "scale": scale, "inputs_written": mutated,
                     "seconds": {n: round(s, 1) for n, s in secs.items()},
                     "mean_log_decay": {"min_abs_heads": [int(h) for h in dec.abs().argsort()[:6]],
                                        "heads": {h: round(float(dec[h]), 6) for h in heads}}}
        for n, (_, s) in outs.items():
            # state layouts may differ ([K, V] vs [V, K]): each kernel's orientation is the closer one
            hs = [min(float((s[h] - S64[h]).norm() / S64[h].norm()), float((s[h].t() - S64[h]).norm() / S64[h].norm()))
                  for h in range(H)]
            hs_t = torch.tensor(hs)
            tr = float((s.double().transpose(-1, -2) - S64).norm() / S64.norm())
            st_rel = min(float((s.double() - S64).norm() / S64.norm()), tr)
            r = stats[n].summary(heads)
            r.update(state_rel=round(st_rel, 6), state_transposed=tr < float((s.double() - S64).norm() / S64.norm()),
                     head_state_rel_max=round(float(hs_t.max()), 5), head_state_worst=int(hs_t.argmax()),
                     head_state_rel_median=round(float(hs_t.median()), 5))
            for h in heads:
                r["heads"][h]["state_rel"] = round(hs[h], 5)
            res[n] = r
        res["flashinfer_vs_flashkda"] = pair.summary(heads)
    return res


def selftest(dev, T: int = 1024, H: int = 64) -> dict:
    """Conventions check on random inputs: each kernel against float64 variants (q/k l2-normalized or not)."""
    import flashinfer.kda_kernels.persistent.forward as fip
    import vllm._flashkda_C  # noqa: F401
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate

    D, lb, scale = 128, -5.0, 128 ** -0.5
    gen = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
    q, k, v, g = (r(1, T, H, D).bfloat16() for _ in range(4))
    beta = r(1, T, H).bfloat16()
    a_log = (r(H) * 0.5).float()
    dt_bias = (r(H * D) * 0.5).float()
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    zero = lambda: torch.zeros(1, H, D, D, dtype=torch.float32, device=dev)  # noqa: E731
    outs = {}
    pristine = [x.clone() for x in (q, k, v, g, beta)]
    written = {}

    def changed():
        torch.cuda.synchronize()
        return [n for n, a, b in zip("q k v g beta".split(), (q, k, v, g, beta), pristine) if not torch.equal(a, b)]

    def restore():
        for a, b in zip((q, k, v, g, beta), pristine):
            a.copy_(b)

    o = torch.empty_like(q)
    st = zero()
    fip.fwd(q, k, v, g, beta, scale, o, a_log, dt_bias, lb, initial_state=st, final_state=st, cu_seqlens=cu,
            plan=fip.make_plan([0, T], H, dev))
    outs["flashinfer"] = (o.clone(), st[0].clone())
    written["flashinfer"] = changed()
    restore()
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, 1), dtype=torch.uint8, device=dev)
    o, fin = torch.empty_like(q), zero()
    torch.ops._flashkda_C.fwd(q, k, v, g, beta, scale, o, ws, a_log, dt_bias.view(-1, D), lb, zero(), fin, cu, None,
                              None)
    outs["flashkda"] = (o, fin[0])
    written["flashkda"] = changed()
    restore()
    o, fin = chunk_kda_with_fused_gate(q=q, k=k, v=v, raw_g=g, beta=torch.sigmoid(beta.float()), A_log=a_log,
                                       g_bias=dt_bias, scale=scale, initial_state=zero(), output_final_state=True,
                                       use_qk_l2norm_in_kernel=True, cu_seqlens=cu, safe_gate=True, lower_bound=lb)
    outs["triton_chunk"] = (o, fin[0])
    written["triton_chunk"] = changed()
    restore()
    # the float64 reference on the CPU, plain per-head loops (no batched GPU ops): checks the GPU path
    qc, kc, vc, gc, bc = (x[0].double().cpu() for x in (q, k, v, g, beta))
    Ac, bias = a_log.double().cpu().exp(), dt_bias.double().cpu().view(H, D)
    ref_cpu = torch.empty(T, H, D, dtype=torch.float64)
    S_cpu = torch.zeros(H, D, D, dtype=torch.float64)
    for h in range(H):
        qh = torch.nn.functional.normalize(qc[:, h], dim=-1) * scale
        kh = torch.nn.functional.normalize(kc[:, h], dim=-1)
        dec = (lb * torch.sigmoid(Ac[h] * (gc[:, h] + bias[h]))).exp()
        bh = torch.sigmoid(bc[:, h])
        Sh = torch.zeros(D, D, dtype=torch.float64)
        for t in range(T):
            Sh = dec[t][:, None] * Sh
            Sh = Sh + bh[t] * torch.outer(kh[t], vc[t, h] - kh[t] @ Sh)
            ref_cpu[t, h] = qh[t] @ Sh
        S_cpu[h] = Sh
    res = {"inputs_written": written, "norms": {"ref_cpu_out": float(ref_cpu.norm()), "ref_cpu_state": float(S_cpu.norm()),
                     **{f"{n}_out": float(o.double().norm()) for n, (o, _) in outs.items()},
                     **{f"{n}_state": float(s.double().norm()) for n, (_, s) in outs.items()}}}
    for norm in (True, False):
        got = []
        S = fp64_reference(q[0], k[0], v[0], g[0], beta[0], a_log, dt_bias, lb, scale,
                           lambda t0, o: got.append(o), l2norm=norm)
        ref = torch.cat(got)
        if norm:
            res["gpu_ref_vs_cpu_ref"] = {"out_rel": float((ref.cpu() - ref_cpu).norm() / ref_cpu.norm()),
                                         "state_rel": float((S.cpu() - S_cpu).norm() / S_cpu.norm())}
        for n, (o, s) in outs.items():
            o = o.reshape(T, H, D).double()
            s = s.double()
            res[f"{n} l2norm={norm}"] = {
                "out_rel": round(float((o - ref).norm() / ref.norm()), 6), "out_norm_ratio": round(float(o.norm() / ref.norm()), 4),
                "state_rel": round(min(float((s - S).norm() / S.norm()), float((s.transpose(-1, -2) - S).norm() / S.norm())), 6)}
    print(f"[kda-probe] selftest T={T}: {json.dumps(res, indent=0)}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--selftest-only", action="store_true", help="only the conventions check on random inputs")
    ap.add_argument("--heads", "--fp64-heads", dest="heads", default="0:10,1:45,1:7,2:56",
                    help="layer:head pairs reported per bin (the audit's worst heads + a median one)")
    a = ap.parse_args()
    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    C.OUT.mkdir(parents=True, exist_ok=True)
    path = C.OUT / f"kda_probe_{JOB}.json"
    out = {"length": a.length, "job": JOB, "selftest": selftest(dev), "layers": {}}
    path.write_text(json.dumps(out, indent=1))
    if a.selftest_only:
        return
    L, B = a.length, 8576
    chunk = min(a.chunk, L)
    # all of stage 0 (layers 0-11): a shorter range leaves a KDA cache group empty (StageMetadata indexes it)
    m = StageModel(bench_config(chunk), 0, dev, first=True, last=False, num_blocks=1 + -(-L // B) + 6,
                   max_model_len=L + B, warmup=False, cuda_graphs=False)
    heads: dict[int, list[int]] = {}
    for p in a.heads.split(","):
        layer, h = map(int, p.split(":"))
        heads.setdefault(layer, []).append(h)
    t0 = time.time()
    caps = capture_inputs(m, L, chunk)
    print(f"[kda-probe] L={L} chunk {chunk}: layers {KDA_PROBE_LAYERS} captured in {time.time() - t0:.1f} s; "
          f"q {tuple(caps[0]['q'].shape)}, dtypes {caps[0]['dtypes']}, lower bound {caps[0]['lb']}, "
          f"scale {caps[0]['scale']}", flush=True)
    out["chunk"] = chunk
    for i, layer in enumerate(KDA_PROBE_LAYERS):
        assert m.layers[layer].layer_kind == "kda"
        r = run_layer(m.layers[layer].self_attn, caps[i], heads.get(layer, []), dev)
        caps[i] = None
        torch.cuda.empty_cache()
        out["layers"][layer] = r
        brief = {n: {f: r[n][f] for f in ("out_rel", "state_rel", "head_out_rel_max", "head_out_worst",
                                          "head_state_rel_max", "head_state_worst")}
                 for n in ("flashinfer", "flashkda", "triton_chunk")}
        print(f"[kda-probe] layer {layer} vs float64 ({r['seconds']}): {json.dumps(brief)}", flush=True)
        for n in ("flashinfer", "flashkda"):
            print(f"[kda-probe] layer {layer} {n} out bins {r[n]['out_bins']}", flush=True)
        path.write_text(json.dumps(out, indent=1))
    print(f"[kda-probe] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
