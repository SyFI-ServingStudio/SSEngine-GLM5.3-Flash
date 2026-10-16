"""Compare the audit's dumps pairwise (bench/kvaudit/common.py `cases(L).pairs`), per layer and tensor.

    python -m bench.kvaudit.compare --length 8192 [--dumps DIR] [--device cuda] [--delete]

Per pair (x, reference) and layer, on dequantized values: relative L2 (||x - a|| / ||a||), max |x - a|, cosine,
the fraction of exactly equal bytes (fp8 tensors); per token (MLA) or pool (indexer) the relative error
e_i = ||x_i - a_i|| / ||a_i||, summarized by quantiles, 16 position bins, the worst positions, and the mean near
chunk starts / the resume start / the end against the overall mean (where the error concentrates). KDA: relative L2
of the recurrent state as stored and transposed (a layout check), per-head worst, conv state per q/k/v section.
Logits: KL(reference || x), top-1, top-5 overlap. Indexer top-k: per sampled row, the overlap of the selected pool
sets. Writes OUT/<L>/compare.json and OUT/<L>/samples.pt (a few cache rows per case), then with --delete removes the
dumps.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from bench.kvaudit import common as C

BINS = 16


def _f(x) -> float:
    return float(x)


def stats(x: torch.Tensor, a: torch.Tensor) -> dict:
    x, a = x.double().flatten(), a.double().flatten()
    d = x - a
    na = a.norm()
    return {"rel_l2": _f(d.norm() / na) if na > 0 else (0.0 if d.norm() == 0 else float("inf")),
            "max_abs": _f(d.abs().max()) if d.numel() else 0.0,
            "cos": _f(torch.nn.functional.cosine_similarity(x, a, dim=0)) if na > 0 else 1.0,
            "ref_rms": _f(a.square().mean().sqrt()) if a.numel() else 0.0}


def per_row(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    d = (x.float() - a.float()).norm(dim=1)
    n = a.float().norm(dim=1)
    return torch.where(n > 0, d / n.clamp_min(1e-30), torch.where(d > 0, torch.full_like(d, float("inf")), d))


def localize(e: torch.Tensor, starts: list[int], unit: int, extra: dict[str, int]) -> dict:
    """Where the per-row error e (rows = tokens if unit 1, pools if 4) concentrates."""
    n = e.numel()
    if n == 0:
        return {}
    fin = torch.isfinite(e)
    e = torch.where(fin, e, torch.zeros_like(e))
    mean = _f(e.mean())
    q = torch.quantile(e[:min(n, 1 << 24)].float(), torch.tensor([0.5, 0.99], device=e.device)).tolist()
    top = torch.topk(e, min(5, n))
    bins = [_f(b.mean()) for b in e.tensor_split(BINS)] if n >= BINS else []
    out = {"mean": mean, "p50": q[0], "p99": q[1], "max": _f(top.values[0]), "nonfinite": int((~fin).sum()),
           "worst_rows": [int(i) * unit for i in top.indices.tolist()], "worst": [_f(v) for v in top.values],
           "exact_rows": _f((e == 0).float().mean()), "bins": bins, "last64_mean": _f(e[-max(1, 64 // unit):].mean())}

    def win(points: list[int]) -> float | None:
        idx = sorted({p // unit + k for p in points for k in range(-max(1, 8 // unit), max(1, 8 // unit))
                      if 0 <= p // unit + k < n})
        return _f(e[torch.tensor(idx, device=e.device)].mean()) if idx else None

    if starts:
        out["chunk_start_mean"] = win(starts)
    for k, p in extra.items():
        lo = p // unit
        if 0 <= lo < n:
            out[f"{k}_mean"] = _f(e[lo:lo + max(1, 64 // unit)].mean())
    # block-level: 64-token kernel blocks (16 pools)
    g = 64 // unit
    if n >= 2 * g:
        blk = e[: n // g * g].view(-1, g).mean(1)
        med = _f(blk.median())
        out["block64_max_over_median"] = _f(blk.max()) / med if med > 0 else (0.0 if _f(blk.max()) == 0 else
                                                                              float("inf"))
        out["block64_worst"] = int(blk.argmax()) * 64
    return out


def fp8(u8: torch.Tensor) -> torch.Tensor:
    return u8.view(torch.float8_e4m3fn).float()


def chunk_starts(meta: dict) -> list[int]:
    """Our chunk starts (> 0) from the case's rounds; vLLM chunks at 16384 (after the filler when filler_first)."""
    if meta["side"] == "ours":
        out, s = [], 0
        for r in meta["rounds"]:
            for n in r:
                if s:
                    out.append(s)
                s += n
        return out
    first = 16384 - (C.FILLER if meta.get("filler_first") else 0)
    return list(range(first, meta["tokens"], 16384))


def resume_start(name: str) -> int | None:
    if name.startswith("o_prefix_"):
        return int(name.split("_")[2])
    return None


class Dumps:
    def __init__(self, root: Path, device: str):
        self.root, self.device = root, device
        self._meta: dict = {}

    def meta(self, case: str) -> dict:
        if case not in self._meta:
            p = self.root / case / "meta.json"
            self._meta[case] = json.loads(p.read_text())
        return self._meta[case]

    def layer(self, case: str, i: int) -> dict:
        d = torch.load(self.root / case / f"layer{i:02d}.pt", map_location="cpu")
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    def logprobs(self, case: str) -> torch.Tensor:
        return torch.load(self.root / case / "logprobs.pt", map_location="cpu").float()

    def topk(self, case: str) -> dict:
        d = self.root / case
        out = {}
        for p in [d / "topk.pt"] + sorted(d.glob("topk_rank*.pt")):
            if p.exists():
                out.update(torch.load(p, map_location="cpu"))
        return out


def cmp_dsa(x: dict, a: dict, starts: list[int], extra: dict) -> dict:
    T = x["mla"].shape[0]
    assert a["mla"].shape == x["mla"].shape, (x["mla"].shape, a["mla"].shape)
    xm, am = fp8(x["mla"]) * x["mla_scale"], fp8(a["mla"]) * a["mla_scale"]
    res = {"mla": {**stats(xm, am), "exact_bytes": _f((x["mla"] == a["mla"]).float().mean()),
                   "width": xm.shape[1],
                   "scale": [x["mla_scale"], a["mla_scale"]], "dtype": [x["mla_dtype"], a["mla_dtype"]],
                   "tokens": localize(per_row(xm, am), starts, 1, extra)}}
    xk = fp8(x["idx_k"]) * x["idx_s"][:, None]
    ak = fp8(a["idx_k"]) * a["idx_s"][:, None]
    res["indexer"] = {**stats(xk, ak), "exact_bytes": _f((x["idx_k"] == a["idx_k"]).float().mean()),
                      "scale_equal": _f((x["idx_s"] == a["idx_s"]).float().mean()), "pools": xk.shape[0],
                      "per_pool": localize(per_row(xk, ak), starts, C.KPOOL, extra)}
    v = T % C.KPOOL
    tx, ta = x["tail"].float(), a["tail"].float()
    tail = {"valid_slots": v, "shape": [x["tail_shape"], a["tail_shape"]],
            "all_slots_exact": bool(torch.equal(x["tail"], a["tail"]))}
    if v and list(tx.shape) == list(ta.shape) and tx.shape[-2] == C.KPOOL:
        tail.update(stats(tx[..., :v, :], ta[..., :v, :]))
        tail["exact_bytes"] = _f((x["tail"][..., :v, :] == a["tail"][..., :v, :]).float().mean())
    res["tail"] = tail
    return res


def cmp_kda(x: dict, a: dict) -> dict:
    xr, ar = x["rec"].float(), a["rec"].float()
    per_head = ((xr - ar).flatten(1).norm(dim=1) / ar.flatten(1).norm(dim=1).clamp_min(1e-30))
    rec = {**stats(xr, ar), "transposed_rel_l2": stats(xr.transpose(-1, -2), ar)["rel_l2"],
           "head_worst": int(per_head.argmax()), "head_worst_rel": _f(per_head.max()),
           "head_median_rel": _f(per_head.median()), "exact": _f((xr == ar).float().mean())}
    xc, ac = x["conv"], a["conv"]
    c = xc.shape[1] // 3
    conv = {**stats(xc.float(), ac.float()), "exact_bytes": _f((xc == ac).float().mean()),
            "section_rel_l2": [stats(xc[:, j * c:(j + 1) * c].float(), ac[:, j * c:(j + 1) * c].float())["rel_l2"]
                               for j in range(3)]}
    return {"rec": rec, "conv": conv}


def cmp_logprobs(x: torch.Tensor, a: torch.Tensor) -> dict:
    n = min(x.numel(), a.numel())
    x, a = x[:n], a[:n]
    pa = a.exp()
    m = torch.isfinite(a) & (pa > 0)
    xx = torch.where(torch.isfinite(x), x, torch.full_like(x, -1e4))
    kl = _f((pa[m] * (a[m] - xx[m])).sum())
    t5x, t5a = set(x.topk(5).indices.tolist()), set(a.topk(5).indices.tolist())
    ta = a.topk(2)
    return {"kl": kl, "top1": [int(x.argmax()), int(a.argmax())], "top1_equal": int(x.argmax()) == int(a.argmax()),
            "top5_overlap": len(t5x & t5a), "ref_margin": _f(ta.values[0] - ta.values[1]),
            "max_abs_top20": _f((x[a.topk(20).indices] - a[a.topk(20).indices]).abs().max())}


def cmp_topk(x: dict, a: dict) -> dict:
    out = {}
    for layer in sorted(set(x) & set(a)):
        px, ix = x[layer]
        pa, ia = a[layer]
        rows = []
        for p in sorted(set(px.tolist()) & set(pa.tolist())):
            sx = ix[(px == p).nonzero()[0, 0]]
            sa = ia[(pa == p).nonzero()[0, 0]]
            A = set((sx[sx >= 0] // C.KPOOL).tolist())
            Bs = set((sa[sa >= 0] // C.KPOOL).tolist())
            rows.append({"pos": int(p), "pools": [len(A), len(Bs)],
                         "overlap": len(A & Bs) / max(1, max(len(A), len(Bs)))})
        if rows:
            out[str(layer)] = {"min_overlap": min(r["overlap"] for r in rows),
                               "mean_overlap": sum(r["overlap"] for r in rows) / len(rows), "rows": rows}
    return out


def compare_pair(dumps: Dumps, x: str, a: str) -> dict:
    mx, ma = dumps.meta(x), dumps.meta(a)
    assert mx["tokens"] == ma["tokens"], (x, a)
    starts = sorted(set(chunk_starts(mx)) | set(chunk_starts(ma)))
    extra = {}
    rs = resume_start(x)
    if rs is not None:
        extra["resume"] = rs
    extra["end"] = max(0, mx["tokens"] - 64)
    out = {"x": x, "ref": a, "tokens": mx["tokens"], "chunk_starts": starts[:64], "layers": {}}
    for i in range(C.NUM_LAYERS):
        lx, la = dumps.layer(x, i), dumps.layer(a, i)
        out["layers"][str(i)] = cmp_dsa(lx, la, starts, extra) if i in C.DSA_LAYERS else cmp_kda(lx, la)
        del lx, la
    out["logprobs"] = cmp_logprobs(dumps.logprobs(x), dumps.logprobs(a))
    out["topk"] = cmp_topk(dumps.topk(x), dumps.topk(a))
    out["tokens_out"] = [mx.get("token"), ma.get("token")]
    out["meta"] = {"x": {k: mx.get(k) for k in ("job", "prefill_s", "replicas_equal", "replicas_differ",
                                                 "indexer_table_check")},
                   "ref": {k: ma.get(k) for k in ("job", "prefill_s", "replicas_equal", "replicas_differ")}}
    return out


def samples(dumps: Dumps, names: list[str]) -> dict:
    """A few rows per case: MLA tokens 0, mid, last; indexer pools 0, mid, last; KDA head 0 row 0 of one layer."""
    out = {}
    for c in names:
        d = dumps.layer(c, C.DSA_LAYERS[0])
        T = d["mla"].shape[0]
        rows = [0, T // 2, T - 1]
        npool = d["idx_k"].shape[0]
        out[c] = {"mla_rows": rows, "mla": d["mla"][rows].cpu(),
                  "pools": [0, npool // 2, npool - 1] if npool else [],
                  "idx_k": d["idx_k"][[0, npool // 2, npool - 1]].cpu() if npool else None,
                  "kda_rec_l0_h0_r0": dumps.layer(c, 0)["rec"][0, 0].cpu()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--dumps", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delete", action="store_true", help="remove the dumps after writing compare.json")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (shared host)")
    ap.add_argument("--tag", default="", help="suffix of the output directory (OUT/<length><tag>)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    root = a.dumps or C.DUMPS / str(a.length)
    dumps = Dumps(root, a.device)
    cs = C.cases(a.length)
    out_dir = C.OUT / f"{a.length}{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"length": a.length, "pairs": []}
    have = {p.name for p in root.iterdir() if (p / "meta.json").exists()}
    for x, ref, kind in cs.pairs:
        if x not in have or ref not in have:
            print(f"[kvaudit-compare] skip {x} vs {ref}: dump missing", flush=True)
            continue
        r = compare_pair(dumps, x, ref)
        r["kind"] = kind
        res["pairs"].append(r)
        worst = max(((i, v["mla"]["rel_l2"]) for i, v in r["layers"].items() if "mla" in v), key=lambda t: t[1])
        print(f"[kvaudit-compare] {kind:22s} {x} vs {ref}: worst MLA layer {worst[0]} rel {worst[1]:.3g}, "
              f"KL {r['logprobs']['kl']:.3g}", flush=True)
    (out_dir / "compare.json").write_text(json.dumps(res, indent=1))
    torch.save(samples(dumps, sorted(have)), out_dir / "samples.pt")
    print(f"[kvaudit-compare] wrote {out_dir / 'compare.json'}", flush=True)
    if a.delete:
        shutil.rmtree(root)
        print(f"[kvaudit-compare] removed {root}", flush=True)


if __name__ == "__main__":
    main()
