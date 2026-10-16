"""Is DeepSelect's top-k already canonical? One GPU.

  python -m bench.topk_determinism [--out measured/topk_determinism.json]

topk_canon.canonicalize makes the indexer's top-k deterministic: every key above the k-th largest value plus, among
the keys equal to it, the lowest indices, in ascending order. It reads every row's logits once more (at 512k + 8k,
3,584 rows x 133k pools: `_count_kernel` 0.31 ms per call, ~0.85 ms per DSA layer, job 7431). If DeepSelect
already returns that answer, the pass can be skipped on DeepSelect's rows. Checked here, per shape and data kind:

- `repeat`: 20 calls on the same logits give the same output (set and order), with sorted_index off and on;
- `canonical`: with sorted_index on, the output equals `canonical_topk_torch` (set and order);
- `batch`: a row's answer does not depend on the rows around it (each row alone == in the batch);
- the time of DeepSelect with sorted_index off, on, and of canonicalize (CUDA events, median of 15).

Data kinds: `randn` (fp32 normal), `ties` (rounded to 0.1: many keys equal to the k-th value), `few` (values in
{0, 1, 2}: most of a row ties), and `dirty` (randn, the columns past each row's end filled with +inf, as
clean_logits=False leaves them unspecified).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from bench.common import MEASURED

K = 512
SHAPES = [(1, 768), (33, 4096), (1024, 33024), (1024, 131328), (3584, 133120), (8192, 34816)]


def ds(logits, ends, k, sorted_index):
    out = torch.empty(logits.shape[0], k, dtype=torch.int32, device=logits.device)
    torch.ops.deep_select.topk(logits, k, None, ends, False, sorted_index, None, out, None, -1, float("-inf"),
                               False, False)
    return out


def timed(fn, reps=15):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def make(kind, rows, width, dev, g):
    lg = torch.randn(rows, width, device=dev, generator=g)
    if kind == "ties":
        lg = lg.round(decimals=1)
    elif kind == "few":
        lg = torch.randint(0, 3, (rows, width), device=dev, generator=g).float()
    # prefill-like row ends: row r of A query rows sees (P + r + 1) // 4 pools; here the last row sees the full width
    r = torch.arange(rows, device=dev)
    ends = (width - (rows - 1 - r) // 4).clamp(min=1).to(torch.int32)
    if kind == "dirty":
        cols = torch.arange(width, device=dev)[None]
        lg = torch.where(cols < ends[:, None], lg, float("inf"))
    return lg, ends


def main():
    import vllm._deepselect_C  # noqa: F401

    from ppimpl.model import topk_canon

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MEASURED / "topk_determinism.json"))
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    res = []
    for rows, width in SHAPES:
        for kind in ("randn", "ties", "few", "dirty"):
            lg, ends = make(kind, rows, width, dev, g)
            ks = torch.zeros(rows, dtype=torch.int32, device=dev)
            ref = torch.empty(rows, K, dtype=torch.int32, device=dev)
            ref.copy_(ds(lg, ends, K, False))
            topk_canon.canonical_topk_torch(lg, ks, ends, ref, K)
            row = {"rows": rows, "width": width, "kind": kind}
            for si in (False, True):
                first = ds(lg, ends, K, si)
                row[f"repeat_sorted{int(si)}"] = all(torch.equal(first, ds(lg, ends, K, si)) for _ in range(20))
                if si:
                    row["canonical"] = bool(torch.equal(first, ref))
                    row["canonical_rows_differ"] = int((first != ref).any(dim=1).sum())
                    alone = [ds(lg[i:i + 1].contiguous(), ends[i:i + 1], K, True) for i in
                             sorted({0, rows // 2, rows - 1})]
                    row["batch"] = all(torch.equal(x[0], first[i]) for x, i in
                                       zip(alone, sorted({0, rows // 2, rows - 1})))
                else:
                    # unsorted output: canonical as a set?
                    row["set_canonical_unsorted"] = bool(torch.equal(first.sort(dim=1).values,
                                                                     ref.sort(dim=1).values))
            row["ms_unsorted"] = timed(lambda: ds(lg, ends, K, False))
            row["ms_sorted"] = timed(lambda: ds(lg, ends, K, True))
            dst = ds(lg, ends, K, False)
            base = dst.clone()
            row["ms_canon"] = timed(lambda: (dst.copy_(base), topk_canon.canonicalize(lg, ks, ends, dst, K))) \
                - timed(lambda: dst.copy_(base))
            res.append(row)
            print(json.dumps(row), flush=True)
            del lg
            torch.cuda.empty_cache()
    ok = all(r["canonical"] and r["repeat_sorted1"] and r["batch"] for r in res)
    print(f"DeepSelect(sorted_index) canonical everywhere: {ok}", flush=True)
    Path(a.out).write_text(json.dumps({"canonical_everywhere": ok, "cases": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
