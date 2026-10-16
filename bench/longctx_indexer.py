"""The DSA indexer's prefill kernels at long context, one GPU: our calls at the shapes a resumed chunk makes.

  python -m bench.longctx_indexer [--out measured/longctx_indexer.json]

For each (cached context P, computed tokens A): one request whose A query rows each score the (P + r + 1) // 4 pools
before them (kpool 4, `glm53_dsa_attn_local.rs`). Per call, in ms (CUDA events, L2 flushed with a 256 MiB write
before every launch, median of REPS):
- `mqa_logits`: `vllm.utils.deep_gemm.fp8_fp4_mqa_logits` (DeepGEMM sm100_mqa_logits, clean_logits=False, FP8 q/k,
  FP32 logits), split into calls as the fork splits it (`fork`, 512 MiB greedy) and as `pp_patch.install_indexer_split`
  does (`ours`);
- `topk`: DeepSelect top-512 on each call's logits (the backend `kernels.topk_backend` picks at these shapes);
- `canon_old` / `canon_new`: the previous one-program-per-row topk_canon kernel and the current one, on the top-k's
  answer; both must equal `canonical_topk_torch`, also on logits rounded to 0.1 (ties at the k-th value).
Compared with the handoff's L1 rows (kernel_specs.jsonl) at num_keys = 131,072 pools etc.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from bench.common import HANDOFF, MEASURED

REPS = 15
K = 512
H, D = 32, 128


def flush(buf):
    buf.zero_()


def timed(fn, buf, reps=REPS):
    ts = []
    for _ in range(reps):
        flush(buf)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def old_canon(logits, ks, ke, dst, k):
    """topk_canon before job 7404's fix: one program per row, two passes over the row."""
    import triton
    import triton.language as tl

    global _OLD
    if "_OLD" not in globals():
        @triton.jit
        def kern(logits_ptr, stride_l, ks_ptr, ke_ptr, dst_ptr, stride_d, K: tl.constexpr, KP: tl.constexpr,
                 BLOCK: tl.constexpr):
            r = tl.program_id(0).to(tl.int64)
            lo = tl.load(ks_ptr + r).to(tl.int32)
            hi = tl.load(ke_ptr + r).to(tl.int32)
            row = logits_ptr + r * stride_l
            drow = dst_ptr + r * stride_d
            offs_k = tl.arange(0, KP)
            sel = tl.load(drow + offs_k, mask=offs_k < K, other=-1)
            valid = sel >= 0
            v = tl.load(row + lo + sel, mask=valid, other=float("inf"))
            kth = tl.min(v, axis=0)
            n_valid = tl.sum(valid.to(tl.int32), axis=0)
            n_above = tl.zeros((), dtype=tl.int32)
            for c0 in range(lo, hi, BLOCK):
                cols = c0 + tl.arange(0, BLOCK)
                m = cols < hi
                x = tl.load(row + cols, mask=m, other=float("-inf"))
                n_above += tl.sum(((x > kth) & m).to(tl.int32), axis=0)
            need = n_valid - n_above
            pos = tl.zeros((), dtype=tl.int32)
            ties = tl.zeros((), dtype=tl.int32)
            for c0 in range(lo, hi, BLOCK):
                cols = c0 + tl.arange(0, BLOCK)
                m = cols < hi
                x = tl.load(row + cols, mask=m, other=float("-inf"))
                tie = ((x == kth) & m).to(tl.int32)
                take = ((x > kth) & m) | ((tie != 0) & (tl.cumsum(tie, axis=0) + ties <= need))
                slot = tl.cumsum(take.to(tl.int32), axis=0) - 1 + pos
                tl.store(drow + slot, cols - lo, mask=take)
                pos += tl.sum(take.to(tl.int32), axis=0)
                ties += tl.sum(tie, axis=0)
            tl.store(drow + offs_k, tl.full((KP,), -1, tl.int32), mask=(offs_k >= pos) & (offs_k < K))
        _OLD = kern
    _OLD[(dst.shape[0],)](logits, logits.stride(0), ks, ke, dst, dst.stride(0), K=k, KP=512, BLOCK=1024,
                          num_warps=4)


def rows_db():
    out = {}
    for line in open(HANDOFF / "data/kernel_specs.jsonl"):
        r = json.loads(line)
        if r["kind"] in ("dsa_mqa_logits_prefill", "dsa_topk_prefill"):
            s = r["spec"]
            out[(r["kind"], r["backend"], s["num_queries"], s["num_keys"])] = r["db_time_ms"]
    return out


def case(P: int, A: int, buf, db, dev) -> dict:
    import vllm._deepselect_C  # noqa: F401
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder as B

    from ppimpl.model import pp_patch, topk_canon

    N = (P + A) // 4
    g = torch.Generator(device=dev).manual_seed(P + A)
    q = torch.randn(A, H, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    kv = torch.randn(N, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    ksc = torch.rand(N, device=dev, generator=g) + 0.5
    w = torch.randn(A, H, device=dev, generator=g)
    r = torch.arange(A, device=dev, dtype=torch.int32)
    ks_all = torch.zeros(A, dtype=torch.int32, device=dev)
    ke_all = ((P + r + 1) // 4).to(torch.int32)
    split_fork = B.__dict__["_split_indexer_prefill_chunks"] if pp_patch._ORIG_INDEXER_SPLIT is None \
        else pp_patch._ORIG_INDEXER_SPLIT
    fork = [c[1] for c in split_fork.__func__(torch.tensor([N]), torch.tensor([A]), 1 << 40, 512 << 20)]
    ours = [c[1] for c in pp_patch.split_indexer_prefill_chunks([N], [A], 1 << 40)]
    res = {"P": P, "A": A, "pools": N, "fork_rows": [s.stop - s.start for s in fork],
           "ours_rows": [s.stop - s.start for s in ours]}

    def logits_of(s):
        return fp8_fp4_mqa_logits((q[s], None), (kv, ksc), w[s], ks_all[s], ke_all[s], clean_logits=False)

    for name, split in (("fork", fork), ("ours", ours)):
        calls = []
        for s in split:
            t_log = timed(lambda: logits_of(s), buf)
            lg = logits_of(s)
            ks, ke = ks_all[s].contiguous(), ke_all[s].contiguous()
            dst = torch.empty(s.stop - s.start, K, dtype=torch.int32, device=dev)
            t_topk = timed(lambda: pp_patch._deep_select(lg, ke, dst, K), buf)
            pp_patch._deep_select(lg, ke, dst, K)
            base = dst.clone()
            t_old = timed(lambda: (dst.copy_(base), old_canon(lg, ks, ke, dst, K)), buf) \
                - timed(lambda: dst.copy_(base), buf)
            t_new = timed(lambda: (dst.copy_(base), topk_canon.canonicalize(lg, ks, ke, dst, K)), buf) \
                - timed(lambda: dst.copy_(base), buf)
            calls.append({"rows": s.stop - s.start, "keys": int(lg.shape[1]), "mqa_logits_ms": t_log,
                          "topk_ms": t_topk, "canon_old_ms": t_old, "canon_new_ms": t_new})
            del lg
        res[name] = {"calls": calls, **{f"{k}_ms": sum(c[f"{k}_ms"] for c in calls)
                                        for k in ("mqa_logits", "topk", "canon_old", "canon_new")}}
    pools = 1 << round(N).bit_length() - 1 if N else 0
    res["rows_db"] = {"num_keys": pools,
                      "mqa_logits": db.get(("dsa_mqa_logits_prefill", "deepgemm_fp8", A, pools)),
                      "topk_deep_select": db.get(("dsa_topk_prefill", "deep_select", A, pools)),
                      "topk_vllm_cuda": db.get(("dsa_topk_prefill", "vllm_cuda", A, pools))}
    return res


def check_canon(dev) -> dict:
    """canon_new == canon_old == torch reference on the top-k of random and of tie-heavy logits."""
    from ppimpl.model import topk_canon

    out = {}
    for name, rnd in (("random", False), ("ties_0.1", True)):
        g = torch.Generator(device=dev).manual_seed(1)
        rows, width = 1024, 40000
        lg = torch.randn(rows, width, device=dev, generator=g)
        if rnd:
            lg = lg.round(decimals=1)
        r = torch.arange(rows, device=dev)
        ks = torch.where(r < rows // 2, 0, 300).to(torch.int32)
        ke = torch.minimum(ks + 100 + r * 37, torch.full_like(ks, width)).to(torch.int32)
        cols = torch.arange(width, device=dev)[None]
        masked = torch.where((cols >= ks[:, None]) & (cols < ke[:, None]), lg, float("-inf"))
        dst = masked.topk(K, dim=1).indices.to(torch.int32) - ks[:, None]
        dst[dst + ks[:, None] >= ke[:, None]] = -1
        ref = dst.clone()
        topk_canon.canonical_topk_torch(lg, ks, ke, ref, K)
        a, b = dst.flip(1).contiguous(), dst.clone()
        topk_canon.canonicalize(lg, ks, ke, a, K)
        old_canon(lg, ks, ke, b, K)
        out[name] = {"new_eq_ref": bool(torch.equal(a, ref)), "old_eq_ref": bool(torch.equal(b, ref))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MEASURED / "longctx_indexer.json"))
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    db = rows_db()
    out = {"canon_check": check_canon(dev), "cases": []}
    print(json.dumps(out["canon_check"]), flush=True)
    for P in (8192, 131072, 524288):
        for A in (1024, 8192):
            c = case(P, A, buf, db, dev)
            out["cases"].append(c)
            print(json.dumps({k: c[k] for k in ("P", "A", "pools", "fork_rows", "ours_rows", "rows_db")}))
            for nm in ("fork", "ours"):
                print(f"  {nm}: " + " ".join(f"{k}={c[nm][k + '_ms']:.3f}" for k in
                                             ("mqa_logits", "topk", "canon_old", "canon_new")), flush=True)
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
