"""Compare per-stage GPU time by kernel class between nsys captures (bench/nsys_trace.py output).

  python bench/nsys_compare.py A=<run dir>[@mbs] B=<run dir>[@mbs] ...     mbs: a:b (microbatch ids, b exclusive)
  python bench/nsys_compare.py --copies <run dir>[@mbs]                      kernels overlapped by a CE copy vs not

Classes come from kernel short names (CLASSES, first match wins). Per stage and class: ms per microbatch (median
over the selected microbatches), and the change from the first capture. Also each stage's compute / busy / gaps.
`--copies` reads <run dir>/trace.sqlite: for each stage and kernel name, the mean duration of the instances that
overlap a copy touching that GPU (CePipe's peer memcpy, either end) and of those that do not.
"""

import json
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

CLASSES = [   # (class, substrings of the short name)
    ("nccl", ("nccl",)),
    ("moe_nvfp4", ("bmm_", "finalizeKernel", "routing", "act_and_mul", "cvt_fp16_to_fp4")),
    ("sparse_mla", ("fmha", "sparse_mla", "concat_and_cache_mla")),
    ("dsa_indexer", ("mqa_logits", "topK", "fwht", "_kpool", "_expand_pools", "indexer")),
    ("kda", ("pkd", "causal_conv1d", "layer_norm_gated", "kda", "chunk_")),
    ("mhc", ("mhc", "hc_prenorm")),
    ("dense_gemm", ("nvjet", "cutlass3x", "gemm")),
    ("elementwise", ("elementwise", "copy", "gather", "index", "quant", "Fill", "norm", "reduce", "cat")),
]


def kclass(name: str) -> str:
    for c, subs in CLASSES:
        if any(s in name for s in subs):
            return c
    return "other"


def parse_sel(arg: str):
    label, _, rest = arg.partition("=") if "=" in arg else (Path(arg).name, "", arg)
    path, _, mbs = rest.partition("@")
    lo, hi = (int(x) for x in mbs.split(":")) if mbs else (None, None)
    return label, Path(path), lo, hi


def per_mb_classes(run_dir: Path, lo, hi):
    """-> {stage: {class: [ms per selected mb]}}, {stage: {metric: [..]}}, selected mb ids. Reads trace.sqlite."""
    import nsys_trace
    per, _ = nsys_trace.load(run_dir / "trace.sqlite")
    nst = 1 + max(s for s, _ in per)
    mbs = sorted({mb for _, mb in per if (lo is None or mb >= lo) and (hi is None or mb < hi)
                  and all(per.get((s, mb)) for s in range(nst))})
    cls = defaultdict(lambda: defaultdict(list))
    met = defaultdict(lambda: defaultdict(list))
    for mb in mbs:
        for s in range(nst):
            ks = per[(s, mb)]
            acc = defaultdict(float)
            for k in ks:
                acc[kclass(k[2])] += (k[1] - k[0]) / 1e6
            for c in {c for c, _ in CLASSES} | {"other"}:
                cls[s][c].append(acc.get(c, 0.0))
            n = nsys_trace.stage_numbers(ks)
            for m in ("compute_ms", "busy_ms", "gaps_ms"):
                met[s][m].append(n[m])
    return cls, met, mbs


def compare(args):
    runs = []
    for a in args:
        label, path, lo, hi = parse_sel(a)
        cls, met, mbs = per_mb_classes(path, lo, hi)
        runs.append((label, cls, met, mbs))
        print(f"{label}: {path} microbatches {mbs[0]}..{mbs[-1]} ({len(mbs)})")
    base = runs[0]
    for s in sorted(base[1]):
        print(f"\nstage {s}" + "".join(f"  {r[0]:>16}" for r in runs))
        for m in ("compute_ms", "busy_ms", "gaps_ms"):
            vals = [st.median(r[2][s][m]) for r in runs]
            print(f"  {m:<14}" + "".join(f"  {v:7.2f} ({v - vals[0]:+5.2f})" for v in vals))
        order = sorted(base[1][s], key=lambda c: -st.median(base[1][s][c]))
        for c in order:
            vals = [st.median(r[1][s][c]) for r in runs]
            if max(vals) < 0.05:
                continue
            pct = "".join(f" {100 * (v / vals[0] - 1):+4.0f}%" if vals[0] > 0.05 else "" for v in vals[1:])
            print(f"  {c:<14}" + "".join(f"  {v:7.2f} ({v - vals[0]:+5.2f})" for v in vals) + pct)


def copies(arg):
    label, path, lo, hi = parse_sel(arg)
    import nsys_trace
    con = sqlite3.connect(path / "trace.sqlite")
    cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
    has_peer = {"srcDeviceId", "dstDeviceId"} <= cols
    q = "SELECT start, end, deviceId, bytes" + (", srcDeviceId, dstDeviceId" if has_peer else "") + \
        " FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes > 64e6"
    by_dev = defaultdict(list)
    for row in con.execute(q):
        devs = {row[2]} | ({row[4], row[5]} if has_peer else set())
        for d in devs:
            if d is not None:
                by_dev[d].append((row[0], row[1], row[3]))
    for d in by_dev:
        by_dev[d].sort()
        iv = by_dev[d]
        print(f"device {d}: {len(iv)} copies > 64 MB, median {st.median((e - s) / 1e6 for s, e, _ in iv):.3f} ms, "
              f"{st.median(b for *_, b in iv) / 1e6:.0f} MB")
    per, _ = nsys_trace.load(path / "trace.sqlite")
    stage_dev = {}
    kdev = {}   # kernel (start, end) -> device
    for start, end, dev in con.execute("SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kdev[(start, end)] = dev
    stats = defaultdict(lambda: [[], []])
    overlap_ms = defaultdict(float)
    for (s, mb), ks in per.items():
        if (lo is not None and mb < lo) or (hi is not None and mb >= hi):
            continue
        for k in ks:
            dev = kdev.get((k[0], k[1]))
            stage_dev[s] = dev
            iv = by_dev.get(dev, [])
            hit = any(a < k[1] and b > k[0] for a, b, _ in iv)
            if hit:
                overlap_ms[(s, mb)] += (k[1] - k[0]) / 1e6
            stats[(s, k[2])][1 if hit else 0].append((k[1] - k[0]) / 1e6)
    print("stage -> device:", dict(sorted(stage_dev.items())))
    for s in sorted({s for s, _ in stats}):
        print(f"\nstage {s}: kernels with instances both overlapped and not (mean ms, n)")
        rows = []
        for (s2, name), (no, yes) in stats.items():
            if s2 != s or not no or not yes:
                continue
            rows.append((st.mean(yes) * len(yes) - st.mean(no) * len(yes), name, st.mean(no), len(no), st.mean(yes),
                         len(yes)))
        for d, name, mn, nn, my, ny in sorted(rows, reverse=True)[:12]:
            print(f"  {name[:60]:<60} no {mn:7.3f} ({nn:4d})  overlapped {my:7.3f} ({ny:4d})  {100 * (my / mn - 1):+5.1f}%")


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--copies":
        return copies(a[1])
    compare(a)


if __name__ == "__main__":
    main()
