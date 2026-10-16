"""The indexer's decode top-k kernels against a torch reference, per backend and shape (1 GPU).

  python -m bench.topk_check [--out measured/topk_check/<job>.json]

Rows of pool logits as the decode path gives them (one row per decode token, next_n = 1, seq_lens [B, 1]; row r keeps
keys [0, seq_lens[r])), k = 512. Per backend (cooperative: up to 64 rows, persistent), shape and logits kind
(`randn`: no ties; `round`: rounded to 0.1, many ties at the k-th value), over 5 calls:
- `valid`: every call returns a valid top-k: min(k, L) distinct keys in [0, L), -1 after them, and no key left out
  is above the smallest key taken (ties at the k-th value may be any of them);
- `set_stable` / `order_stable`: the 5 calls return the same set / the same set in the same order;
- `canon_ok`: after `pp_patch.canonical_decode_topk` (kernel + `topk_canon.canonicalize_split`), every call equals
  `canonical_topk_torch`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from bench.common import MEASURED

JOB = os.environ.get("SLURM_JOB_ID", "local")
K = 512


def valid_topk(lg: torch.Tensor, ends: torch.Tensor, dst: torch.Tensor, k: int) -> list[int]:
    """Rows whose answer is not a valid top-k."""
    bad = []
    for r in range(dst.shape[0]):
        L = int(ends[r])
        n = min(k, L)
        got = dst[r].tolist()
        sel, rest = got[:n], got[n:]
        ok = all(0 <= i < L for i in sel) and len(set(sel)) == n and all(i == -1 for i in rest)
        if ok and L > n:
            row = lg[r, :L]
            mask = torch.ones(L, dtype=torch.bool, device=lg.device)
            mask[torch.tensor(sel, device=lg.device)] = False
            ok = bool(row[~mask].min() >= row[mask].max())
        if not ok:
            bad.append(r)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=MEASURED / "topk_check" / f"{JOB}.json")
    ap.add_argument("--calls", type=int, default=5)
    a = ap.parse_args()

    from ppimpl.model import pp_patch
    from ppimpl.model.topk_canon import canonical_topk_torch
    from vllm.model_executor.layers import indexer_topk
    from vllm.v1.worker.workspace import init_workspace_manager

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    init_workspace_manager(dev)
    pp_patch._ORIG_DECODE_TOPK = pp_patch._ORIG_DECODE_TOPK or indexer_topk.get_indexer_topk
    out = []
    for backend in ("cooperative", "persistent"):
        disp = indexer_topk.get_indexer_topk(backend)
        canon = pp_patch.canonical_decode_topk(backend)
        for rows in (1, 3, 16, 32, 64, 128):
            if backend == "cooperative" and rows > indexer_topk.AUTO_COOPERATIVE_MAX_ROWS:
                continue
            for width in (4096, 8192, 16384, 32768, 131072):
                for kind in ("randn", "round"):
                    g = torch.Generator(device=dev).manual_seed(rows * 1000 + width)
                    lg = torch.randn(rows, width, device=dev, generator=g)
                    if kind == "round":
                        lg = lg.round(decimals=1)
                    lo = max(1, width // 2)
                    seq = torch.linspace(lo, width, rows, device=dev).to(torch.int32).reshape(rows, 1)
                    ends = seq.reshape(-1)
                    cols = torch.arange(width, device=dev)[None]
                    ref = torch.where(cols < ends[:, None], lg, float("-inf")).topk(K, dim=1).indices.to(torch.int32)
                    ref[torch.arange(K, device=dev)[None] >= ends[:, None]] = -1
                    canonical_topk_torch(lg, torch.zeros_like(ends), ends, ref, K)   # a valid top-k -> canonical
                    calls, canon_ok, err = [], True, None
                    try:
                        for _ in range(a.calls):
                            d = torch.full((rows, K), -7, dtype=torch.int32, device=dev)
                            disp(lg, seq, 1, d, K, int(seq.max()))
                            calls.append(d.clone())
                            c = torch.full((rows, K), -7, dtype=torch.int32, device=dev)
                            canon(lg, seq, 1, c, K, int(seq.max()))
                            canon_ok &= bool(torch.equal(c, ref))
                    except RuntimeError as e:
                        err = str(e).splitlines()[0][:200]
                    if err:
                        line = {"backend": backend, "rows": rows, "width": width, "kind": kind, "error": err}
                    else:
                        bad = sorted({r for d in calls for r in valid_topk(lg, ends, d, K)})
                        sets = {tuple(map(tuple, d.sort(dim=1).values.tolist())) for d in calls}
                        orders = {tuple(map(tuple, d.tolist())) for d in calls}
                        line = {"backend": backend, "rows": rows, "width": width, "kind": kind,
                                "valid": not bad, "invalid_rows": bad[:8], "n_invalid_rows": len(bad),
                                "set_stable": len(sets) == 1, "order_stable": len(orders) == 1,
                                "canon_ok": canon_ok}
                    out.append(line)
                    print(json.dumps(line), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    bad = [x for x in out if x.get("error") or not x.get("valid") or not x.get("canon_ok")]
    print(f"[topk_check] {len(out)} cases; invalid or not canonical: {len(bad)}", flush=True)


if __name__ == "__main__":
    main()
