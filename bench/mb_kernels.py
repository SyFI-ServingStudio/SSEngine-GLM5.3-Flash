"""Per-kernel GPU time of chosen microbatches, per stage, from an Nsight capture of `bench/margin.py --nvtx`.

  nsys profile -o <out> python -m bench.margin <cases.json> --nvtx --out <res.json>
  nsys export --type sqlite -o <out>.sqlite <out>.nsys-rep
  python -m bench.mb_kernels <out>.sqlite <res>.mbs.json [--base CASE] [--top 15]

For every case, its is_last microbatches (one per repeat, the first repeat dropped as warm-up when there are more):
per stage the median compute (first kernel start to last kernel end), busy (union of kernels) and, per kernel name,
the median summed time per microbatch. With --base, each case's per-stage and per-kernel difference to that case.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bench.nsys_trace import load, stage_numbers


def summarize(db: Path, mbs: list[dict]) -> dict:
    per, _ = load(db)
    by_case: dict[str, list[int]] = defaultdict(list)
    for m in mbs:
        if m["is_last"]:
            by_case[m["case"]].append(m["mb"])
    out = {}
    for case, ids in by_case.items():
        ids = ids[1:] if len(ids) > 1 else ids
        stages = []
        for s in range(4):
            comp, busy, kern = [], [], defaultdict(list)
            for mb in ids:
                ks = per.get((s, mb), [])
                if not ks:
                    continue
                n = stage_numbers(ks)
                comp.append(n["compute_ms"])
                busy.append(n["busy_ms"])
                sums: dict[str, float] = defaultdict(float)
                for k in ks:
                    sums[k[2]] += (k[1] - k[0]) / 1e3
                for name in sums:
                    kern[name].append(sums[name])
            if not comp:
                stages.append({})
                continue
            stages.append({"compute_ms": statistics.median(comp), "busy_ms": statistics.median(busy),
                           "kernels_us": {k: statistics.median(v + [0.0] * (len(comp) - len(v)))
                                          for k, v in kern.items()}})
        out[case] = {"microbatches": ids, "stages": stages}
    return out


def report(res: dict, base: str | None, top: int) -> None:
    for case, r in res.items():
        print(f"== {case} ({len(r['microbatches'])} microbatches)")
        b = res.get(base) if base and case != base else None
        for s, st in enumerate(r["stages"]):
            if not st:
                continue
            line = f"  stage {s}: compute {st['compute_ms']:.3f} ms, busy {st['busy_ms']:.3f} ms"
            if b and b["stages"][s]:
                bs = b["stages"][s]
                line += (f"  (vs {base}: compute {st['compute_ms'] - bs['compute_ms']:+.3f}, "
                         f"busy {st['busy_ms'] - bs['busy_ms']:+.3f})")
            print(line)
            if b and b["stages"][s]:
                bk = b["stages"][s]["kernels_us"]
                diff = {k: st["kernels_us"].get(k, 0.0) - bk.get(k, 0.0) for k in set(st["kernels_us"]) | set(bk)}
                for k, d in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:top]:
                    print(f"    {d:+9.1f} us  {st['kernels_us'].get(k, 0.0):9.1f} us  {k[:110]}")
            else:
                for k, v in sorted(st["kernels_us"].items(), key=lambda kv: -kv[1])[:top]:
                    print(f"    {v:9.1f} us  {k[:110]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path)
    ap.add_argument("mbs", type=Path)
    ap.add_argument("--base")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = summarize(a.db, json.loads(a.mbs.read_text()))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    report(res, a.base, a.top)


if __name__ == "__main__":
    main()
