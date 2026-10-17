"""KDA short conv of one layer: the per-sequence Dao calls the design prices (the reference) vs `ppimpl.model.kda_conv`
(two launches), bitwise and timed, on 1 GPU.

  python -m bench.kda_conv_bench [--reps 100] [--out x.json]

Cases: one 4,096-row sequence (fresh, and continuing a conv state), served-like multi-sequence microbatches with
initial states, sequences shorter than the conv width, 8,192 rows. q, k, v and the whole conv-state pool (paged as vLLM
binds it: blocks a page apart) must be bitwise equal to the Dao calls'. Then the KDA recurrent state gather / scatter
(ppimpl.model.kda_state) vs torch index_select / index_copy_. Times: CUDA events over `reps` back-to-back calls,
interleaved by variant, median of 5 rounds; the bound is reading and writing the bytes once at 8 TB/s.
"""

import argparse
import json
import statistics

import torch

from ppimpl.model import kda_conv

P = 8192                 # 64 heads x 128
C = 3 * P
CASES = {   # (first row, end row) are made from the lengths; (length, has initial state)
    "1x4096_fresh": [(4096, False)],
    "1x4096_init": [(4096, True)],
    "2_served_4096": [(3000, True), (1096, True)],
    "4_mixed_4096": [(1500, True), (1999, False), (500, True), (97, True)],
    "short_seqs": [(1, True), (2, True), (3, True), (5, False), (64, True), (65, True), (3956, True)],
    "1x8192_init": [(8192, True)],
    "6_served_8192": [(2048, True), (1500, True), (3000, False), (1000, True), (600, True), (44, True)],
}


def paged(raw: torch.Tensor, shape: tuple) -> torch.Tensor:
    """A pool as vLLM binds it: each block contiguous, blocks a page apart (the raw rows are longer than a block)."""
    n = 1
    for d in shape[1:]:
        n *= d
    return raw[:, :n].view(shape)


def pool_clone(pool: torch.Tensor) -> torch.Tensor:
    raw = torch.empty(pool.shape[0], pool.stride(0), dtype=pool.dtype, device=pool.device)
    p = paged(raw, tuple(pool.shape))
    p.copy_(pool)
    return p


def dao_loop(x, w, conv_state, seqs):
    import causal_conv1d_cuda
    T = x.shape[0]
    q, k, v = (torch.empty(T, P, dtype=x.dtype, device=x.device) for _ in range(3))
    for s, e, slot, has_init in seqs:
        state = conv_state[slot]
        init_all = state.clone() if has_init else None
        for j, out in enumerate((q, k, v)):
            cols = slice(j * P, (j + 1) * P)
            init = None if init_all is None else init_all[:, cols].t().unsqueeze(0)
            causal_conv1d_cuda.causal_conv1d_fwd(x[s:e, cols].t().unsqueeze(0), w[cols], None, None, init,
                                                 out[s:e].t().unsqueeze(0), state[:, cols].t().unsqueeze(0), True)
    return q, k, v


STATE_CASES = {"1_init": [1], "2_init": [1, 1], "2_mixed": [1, 0], "4_mixed": [1, 0, 1, 1], "6_init": [1] * 6}


def timed(fs: dict, reps: int) -> dict:
    times = {k: [] for k in fs}
    for _ in range(5):
        for k, f in fs.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(50_000_000)
            e0.record()
            for _ in range(reps):
                f()
            e1.record()
            e1.synchronize()
            times[k].append(1e3 * e0.elapsed_time(e1) / reps)
    return {k: round(statistics.median(v), 2) for k, v in times.items()}


def state_cases(dev, g, reps: int) -> dict:
    """KDA recurrent state gather (zeros for fresh requests) and scatter: the torch ops of pp_patch vs kda_state."""
    from ppimpl.model import kda_state
    pool0 = paged(torch.randn(40, 64 * 128 * 128 + 1024, generator=g, device=dev), (40, 64, 128, 128))
    out = {}
    for name, inits in STATE_CASES.items():
        N = len(inits)
        slots = torch.tensor([37 - 5 * i for i in range(N)], dtype=torch.int64, device=dev)
        init = torch.tensor(inits, dtype=torch.int32, device=dev)

        def ref_gather():
            st_ = pool0.index_select(0, slots)
            for i, h in enumerate(inits):
                if not h:
                    st_[i].zero_()
            return st_
        ours = kda_state.gather(pool0, slots, init)
        ok_g = torch.equal(ours, ref_gather())
        new = torch.randn(N, 64, 128, 128, generator=g, device=dev)
        p1, p2 = pool_clone(pool0), pool_clone(pool0)
        p1.index_copy_(0, slots, new)
        kda_state.scatter(new, slots, p2)
        ok_s = torch.equal(p1, p2)
        t = timed({"torch_gather": ref_gather, "ours_gather": lambda: kda_state.gather(pool0, slots, init),
                   "torch_scatter": lambda: p1.index_copy_(0, slots, new),
                   "ours_scatter": lambda: kda_state.scatter(new, slots, p2)}, reps)
        bound = round(2 * N * 4 * 2 ** 20 / 8e12 * 1e6, 1)
        out[name] = {"N": N, "bitwise_gather": ok_g, "bitwise_scatter": ok_s, "bound_us_each": bound, **t}
        print(f"[kda_state] {name}: N {N} bitwise gather {ok_g} scatter {ok_s} | bound {bound} us each | " +
              " ".join(f"{k} {v}" for k, v in t.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    w = (torch.randn(C, 4, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    pool0 = paged(torch.randn(32, 3 * C + 256, generator=g, device=dev).to(torch.bfloat16), (32, 3, C))
    configs = [(bt, bc, nw) for bt in (8, 16, 32, 64) for bc in (256, 512, 1024) for nw in (2, 4, 8)]
    res = {"gpu": torch.cuda.get_device_name(dev), "cases": {}}
    for name, lens in CASES.items():
        T = sum(n for n, _ in lens)
        xbuf = torch.randn(T, C + 4096, generator=g, device=dev).to(torch.bfloat16)   # q|k|v in front of more columns
        x = xbuf[:, :]
        seqs, at = [], 0
        for i, (n, init) in enumerate(lens):
            seqs.append((at, at + n, 3 + 2 * i, init))
            at += n
        ref_pool = pool_clone(pool0)
        rq, rk, rv = dao_loop(x, w, ref_pool, seqs)
        out = {"T": T, "seqs": len(seqs), "bound_us": round(2 * T * C * 2 / 8e12 * 1e6, 1), "variants": {}}
        variants = {"dao_loop": lambda pool: dao_loop(x, w, pool, seqs)}
        for bt, bc, nw in configs:
            kda_conv.BLOCK_T = bt
            bt_, st_ = kda_conv.tables(seqs)
            tabs = (torch.from_numpy(bt_).to(dev), torch.from_numpy(st_).to(dev))

            def ours(pool, bt=bt, bc=bc, nw=nw, tabs=tabs):
                kda_conv.BLOCK_T, kda_conv.BLOCK_C, kda_conv.NUM_WARPS = bt, bc, nw
                return kda_conv.conv_qkv(x, w, pool, tabs[0], tabs[1], P)
            pool = pool_clone(pool0)
            q, k, v = ours(pool)
            ok = all(torch.equal(a_, b_) for a_, b_ in ((q, rq), (k, rk), (v, rv))) and torch.equal(pool, ref_pool)
            out["variants"][f"bt{bt}_bc{bc}_w{nw}"] = {"bitwise": ok}
            variants[f"bt{bt}_bc{bc}_w{nw}"] = ours
        out["variants"]["dao_loop"] = {"bitwise": True}
        times = {k: [] for k in variants}
        pool = pool_clone(pool0)
        for _ in range(5):
            for k, f in variants.items():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(50_000_000)
                e0.record()
                for _ in range(a.reps):
                    f(pool)
                e1.record()
                e1.synchronize()
                times[k].append(1e3 * e0.elapsed_time(e1) / a.reps)
        for k in variants:
            out["variants"][k]["us"] = round(statistics.median(times[k]), 2)
        base = out["variants"]["dao_loop"]["us"]
        best = sorted(((v["us"], k) for k, v in out["variants"].items() if v["bitwise"] and k != "dao_loop"))
        print(f"[kda_conv] {name}: T {T} seqs {len(seqs)} bound {out['bound_us']} us | dao_loop {base} us | "
              + ", ".join(f"{k} {u} (x{u / base:.2f})" for u, k in best[:4])
              + f" | not bitwise: {[k for k, v in out['variants'].items() if not v['bitwise']]}", flush=True)
        res["cases"][name] = out
    try:
        res["state"] = state_cases(dev, g, a.reps)
    except Exception as e:     # the conv results above stand on their own
        print(f"[kda_state] failed: {e!r}", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
