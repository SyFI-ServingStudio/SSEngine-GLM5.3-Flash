"""Tables of the KV-cache audit from compare.json files (research/bench/kvaudit/compare.py).

    python -m research.bench.kvaudit.report [--lengths 8192 32768 ...] [--out research/measured/kvaudit.md]

Per length: (1) every pair, the worst layer per tensor type; (2) per layer, ours vs vLLM against the two noise
baselines (vLLM alone vs with a filler first; ours whole vs chunked), each as an envelope over depth (a layer's
noise = the largest of it and every earlier layer of the same tensor); (3) flags: a layer/tensor whose audit error is
more than FACTOR x the larger noise baseline (and above FLOOR), and systematic patterns (block/pool concentration,
chunk-start / resume concentration, layout).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bench.kvaudit import common as C

FACTOR = 3.0
FLOOR = 1e-3


def g(d: dict, path: str, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


TENSORS = {  # name -> (layer kind, path to rel_l2 in a layer entry)
    "mla": ("dsa", "mla.rel_l2"),
    "indexer": ("dsa", "indexer.rel_l2"),
    "tail": ("dsa", "tail.rel_l2"),
    "kda_rec": ("kda", "rec.rel_l2"),
    "kda_conv": ("kda", "conv.rel_l2"),
}


def layers_of(kind: str):
    return C.DSA_LAYERS if kind == "dsa" else C.KDA_LAYERS


def worst(p: dict, tensor: str) -> tuple[int | None, float | None]:
    kind, path = TENSORS[tensor]
    vals = [(i, g(p["layers"][str(i)], path)) for i in layers_of(kind)]
    vals = [(i, v) for i, v in vals if v is not None]
    return max(vals, key=lambda t: t[1]) if vals else (None, None)


def fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.{nd}g}"
    return str(v)


def pair_table(res: dict) -> list[str]:
    rows = ["| pair (x vs reference) | kind | MLA rel L2 (layer) | MLA min exact bytes | indexer rel L2 (layer) | "
            "tail rel L2, valid slots | KDA state rel L2 (layer) | conv rel L2 (layer) | KL | top-1 | top-5 | "
            "top-k min overlap |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in res["pairs"]:
        cells = []
        for t in ("mla", "indexer", "tail", "kda_rec", "kda_conv"):
            i, v = worst(p, t)
            cells.append(f"{fmt(v)} ({i})" if i is not None else "—")
        exact = min(g(p["layers"][str(i)], "mla.exact_bytes") for i in C.DSA_LAYERS)
        tv = g(p["layers"][str(C.DSA_LAYERS[0])], "tail.valid_slots")
        lp = p["logprobs"]
        tk = [v["min_overlap"] for v in p["topk"].values()]
        rows.append(f"| {p['x']} vs {p['ref']} | {p['kind']} | {cells[0]} | {fmt(exact)} | {cells[1]} | "
                    f"{cells[2]}, {tv} | {cells[3]} | {cells[4]} | {fmt(lp['kl'])} | "
                    f"{'=' if lp['top1_equal'] else lp['top1']} | {lp['top5_overlap']}/5 | "
                    f"{fmt(min(tk)) if tk else '—'} |")
    return rows


def noise_of(res: dict, kind: str) -> dict:
    """tensor -> layer -> max rel_l2 over the pairs of `kind`."""
    out: dict = {}
    for p in res["pairs"]:
        if p["kind"] != kind:
            continue
        for t, (lk, path) in TENSORS.items():
            for i in layers_of(lk):
                v = g(p["layers"][str(i)], path)
                if v is not None:
                    out.setdefault(t, {})[i] = max(out.get(t, {}).get(i, 0.0), v)
    # envelope: the error grows with depth, and one noise sample per layer is itself noisy; a layer's noise is the
    # largest of its own and every earlier layer's of the same tensor
    for t, per in out.items():
        run = 0.0
        for i in sorted(per):
            run = max(run, per[i])
            per[i] = run
    return out


def layer_table(res: dict, pair: dict, nv: dict, no: dict) -> tuple[list[str], list[str]]:
    rows = [f"Per layer, `{pair['x']}` vs `{pair['ref']}`; noise: vLLM (filler first) / ours (chunked vs whole); "
            f"ratio = audit / max(noise).", "",
            "| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |",
            "|---|---|---|---|---|---|---|---|---|"]
    flags = []
    for t, (lk, path) in TENSORS.items():
        for i in layers_of(lk):
            e = pair["layers"][str(i)]
            v = g(e, path)
            if v is None:
                continue
            a, b = nv.get(t, {}).get(i), no.get(t, {}).get(i)
            n = max(x for x in (a, b, 0.0) if x is not None)
            ratio = v / n if n > 0 else (float("inf") if v > 0 else 1.0)
            flag = v > FLOOR and ratio > FACTOR
            sub = path.split(".")[0]
            cos = g(e, f"{sub}.cos")
            ex = g(e, f"{sub}.exact_bytes", g(e, f"{sub}.exact"))
            if flag:
                flags.append(f"{pair['x']} layer {i} {t}: rel {v:.3g} = {ratio:.1f}x noise ({n:.3g})")
            rows.append(f"| {i} | {t} | {fmt(v)} | {fmt(a)} | {fmt(b)} | {fmt(ratio, 2)} | {fmt(cos, 6)} | "
                        f"{fmt(ex)} | {'**above noise**' if flag else ''} |")
    return rows, flags


def patterns(p: dict) -> list[str]:
    out = []
    for i in C.DSA_LAYERS:
        e = p["layers"][str(i)]
        for sub, key in (("mla", "tokens"), ("indexer", "per_pool")):
            loc = g(e, f"{sub}.{key}") or {}
            mean = loc.get("mean") or 0
            if not mean:
                continue
            for k in ("chunk_start_mean", "resume_mean", "last64_mean"):
                v = loc.get(k)
                if v is not None and v > 3 * mean and v > FLOOR:
                    out.append(f"{p['x']} layer {i} {sub}: {k} {v:.3g} vs mean {mean:.3g}")
            b = loc.get("block64_max_over_median")
            if b is not None and b > 10 and loc.get("max", 0) > FLOOR:
                out.append(f"{p['x']} layer {i} {sub}: worst 64-token block x{b:.1f} the median "
                           f"(at {loc.get('block64_worst')})")
            if loc.get("nonfinite"):
                out.append(f"{p['x']} layer {i} {sub}: {loc['nonfinite']} non-finite rows")
    for i in C.KDA_LAYERS:
        r = p["layers"][str(i)]["rec"]
        if r["transposed_rel_l2"] < r["rel_l2"]:
            out.append(f"{p['x']} layer {i}: KDA state closer transposed ({r['transposed_rel_l2']:.3g} < "
                       f"{r['rel_l2']:.3g})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=C.MEASURED / "kvaudit.md")
    ap.add_argument("--header", type=Path, default=None, help="markdown put before the tables")
    a = ap.parse_args()
    lengths = a.lengths or sorted(int(p.name) for p in C.OUT.iterdir() if p.name.isdigit() and (p / "compare.json").exists())
    lines = [a.header.read_text().rstrip(), ""] if a.header else ["# KV-cache audit", ""]
    all_flags = []
    for L in lengths:
        res = json.loads((C.OUT / str(L) / "compare.json").read_text())
        nv, no = noise_of(res, "noise_vllm"), noise_of(res, "noise_ours")
        lines += [f"## {L:,} tokens", "", *pair_table(res), ""]
        for p in res["pairs"]:
            if p["kind"].startswith("audit"):
                rows, flags = layer_table(res, p, nv, no)
                pat = patterns(p)
                all_flags += [f"{L}: {f}" for f in flags + pat]
                lines += [f"<details><summary>{p['x']} vs {p['ref']}: {len(flags)} above noise, {len(pat)} "
                          f"patterns</summary>", "", *rows, "", *[f"- {x}" for x in pat], "", "</details>", ""]
    lines += ["## Flags", ""] + ([f"- {f}" for f in all_flags] or ["- none"])
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(all_flags)} flags")


if __name__ == "__main__":
    main()
