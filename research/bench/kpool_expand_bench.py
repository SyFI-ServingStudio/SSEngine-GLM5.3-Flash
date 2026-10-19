"""The indexer's prefill pool-to-token expansion: the fork's launches vs `ssengine.model.kpool_fast.expand_into` (1 GPU).

  python -m research.bench.kpool_expand_bench [--rows 4096] [--pools 512] [--pool 4] [--reps 200]

Fork: pool ids to int64, positions to int32 + 1, `expand_pools_and_append_tail`, copy into the top-k buffer. Ours:
one kernel into the buffer (BLOCK_COLS / warps swept). Buffer rows must be bitwise equal; CUDA-event times over
back-to-back reps, interleaved.
"""

import argparse
import json
import statistics


def main():
    import torch
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kops

    from ssengine.model import kpool_fast

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--pools", type=int, default=512)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    R, P, K = a.rows, a.pools, a.pool
    start = 40_000
    positions = torch.arange(start, start + R, device=dev, dtype=torch.int64)
    pool_topk = torch.randint(0, start // K, (R, P), generator=gen, device=dev, dtype=torch.int32)
    pool_topk[:, -7:] = -1
    width = P * K + K - 1
    buf_a = torch.full((R + 64, width + 8), -7, dtype=torch.int32, device=dev)
    buf_b = buf_a.clone()

    def fork():
        pool_ids = pool_topk.to(torch.int64)
        q_seq = positions.to(torch.int32) + 1
        expanded = kops.expand_pools_and_append_tail(pool_ids, q_seq, K)
        buf_a[:R, :expanded.shape[-1]] = expanded

    def ours(bc, warps):
        kpool_fast.BLOCK_COLS, kpool_fast.NUM_WARPS = bc, warps
        kpool_fast.expand_into(pool_topk, positions, K, buf_b[:R])

    variants = {"fork": fork}
    for bc in (256, 512, 1024, 2048):
        for warps in (2, 4, 8):
            variants[f"bc{bc}_w{warps}"] = (lambda bc=bc, warps=warps: ours(bc, warps))
    fork()
    ok = {}
    for k, f in variants.items():
        buf_b.fill_(-7)
        if k != "fork":
            f()
            ok[k] = bool(torch.equal(buf_a, buf_b))
    times = {k: [] for k in variants}
    for _ in range(5):
        for k, f in variants.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(20_000_000)
            e0.record()
            for _ in range(a.reps):
                f()
            e1.record()
            e1.synchronize()
            times[k].append(1e3 * e0.elapsed_time(e1) / a.reps)
    res = {k: {"us": round(statistics.median(v), 2), "bitwise": ok.get(k, True)} for k, v in times.items()}
    base = res["fork"]["us"]
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["us"]):
        print(f"[kpool_expand] {k:12} {v['us']:7.2f} us  x{v['us'] / base:.3f}  bitwise {v['bitwise']}", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
