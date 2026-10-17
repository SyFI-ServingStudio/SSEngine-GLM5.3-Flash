"""Two-hop tier copies on the GPUs, the engine's own code path (lanes/m6 TIERS.md §2.3 "Second hop").

One process drives the job's 4 GPUs with four `TierEngine`s (`ppimpl/runtime/tier_worker.py`, `hops = "two"`), each on
its own GPU, over fake per-stage KV laid out like GLM-5.3 PP4 (DSA layers 3/2/3/3 at 545 B/token each, split 4:1
between an MLA-like and an indexer-like view; KDA states 9/9/8/8 of 4 MiB + 144 KiB; a 2 KiB tail per DSA layer).
Measured, every result one JSON line in OUT/tier_hop2.jsonl:
- sweep: the ring's depth (2, 4) x chunk (1, 3 = the state, 4 slots) and direct, 32k / 128k / 512k contexts (4 / 16 /
  62 blocks): read and write time (p50 of 10) and bytes equal;
- write / read of a 512k context (62 blocks + state) on all 4 GPUs at once: GB/s per GPU (HBM bytes moved), and the
  bytes read back equal what was written;
- GEMM 8192^3 BF16 on GPU 0 and GPU 1: alone, while all 4 GPUs read back to back, alone again;
- stage edges: an 8,192-token microbatch of activations (8,192 x 32 KiB = 268 MB) copied GPU s -> s+1 by
  `cudaMemcpyAsync` on a copy stream, as `CePipe.send` does: time alone and during the reads, and the read rate while
  the edges copy.
usage: python -m bench.tier_hop2 --out DIR
"""

import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from ppimpl.config import HIDDEN_WIDTH, TierConfig
from ppimpl.runtime import tier_worker
from ppimpl.runtime.tier_copy import ViewRange
from ppimpl.runtime.transport import TierMove

B = 8576
DSA = [3, 2, 3, 3]
KDA = [9, 9, 8, 8]
MLA_B, IDX_B = 436, 109                  # B/token per DSA layer (545 in all; the 4:1 split is assumed)
REC_B, CONV_B, TAIL_B = 4 << 20, 144 << 10, 2 << 10
NB = 130                                 # KV blocks per GPU: two 62-block contexts and the null block
NST = 12                                 # state blocks (two contexts x 6)
CTX = 62                                 # blocks of a 512k context
EDGE_TOKENS = 8192


def fake_kv(n: int):
    """Per GPU s: its views and their ranges (stage s's KV)."""
    views, ranges = [], []
    for s in range(n):
        dev = torch.device("cuda", s)

        def add(t, name, kind, group=-1):
            views.append(t)
            ranges.append(ViewRange(s, name, kind, t.data_ptr(), t.stride(0) * t.element_size(),
                                    t[0].numel() * t.element_size(), group))
        for i in range(DSA[s]):
            add(torch.randint(0, 255, (NB, B * MLA_B), dtype=torch.uint8, device=dev), f"s{s}.mla{i}", "kv")
            add(torch.randint(0, 255, (NB, B * IDX_B), dtype=torch.uint8, device=dev), f"s{s}.idx{i}", "kv")
            add(torch.randint(0, 255, (NST, TAIL_B), dtype=torch.uint8, device=dev), f"s{s}.tail{i}", "state", 5)
        for i in range(KDA[s]):
            add(torch.randint(0, 255, (NST, CONV_B), dtype=torch.uint8, device=dev), f"s{s}.kda{i}#0", "state", i % 5)
            add(torch.randint(0, 255, (NST, REC_B), dtype=torch.uint8, device=dev), f"s{s}.kda{i}#1", "state", i % 5)
    return views, ranges


def peer_access(n: int):
    from cuda.bindings import runtime
    for g in range(n):
        torch.cuda.set_device(g)
        torch.zeros(1, device=g)
        for s in range(n):
            if s != g:
                err, = runtime.cudaDeviceEnablePeerAccess(s, 0)
                assert int(err) in (0, 704), f"peer access {g} -> {s}: {err}"    # 704: already enabled


class Probe:
    def __init__(self, out: Path, ring: dict):
        self.out, self.n = out, torch.cuda.device_count()
        assert self.n == 4, f"needs 4 GPUs, have {self.n}"
        self.f = open(out / "tier_hop2.jsonl", "a")
        peer_access(self.n)
        self.views, ranges = fake_kv(self.n)
        self.ranges = ranges
        tier_worker.share_ranges = lambda stage, n, r: (ranges, [])     # one process: no IPC
        self.eng = []
        self.build(ring)
        self.op = 0

    def build(self, ring: dict):
        """Four engines with TierConfig's ring settings `ring` (hops, ring_depth, ring_chunk_slots)."""
        for e in self.eng:
            e.close()
        self.ring = ring
        tc = TierConfig(dram_gb=4.0, ssd_gb=None, **ring)
        self.eng = []
        for g in range(self.n):
            e = tier_worker.TierEngine(NS(tiers=tc, num_stages=self.n, model="glm53", run_dir=str(self.out)), g,
                                       torch.device("cuda", g), None)
            e.attach(self.ranges)
            self.eng.append(e)
        self.state_slots = self.eng[0].layout.state_slots

    def log(self, **rec):
        print(json.dumps(rec), flush=True)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def move(self, op: str, kv: list[int], st: list[int], sub: list | None = None) -> tuple[float, int]:
        """One op of a len(kv)-block context on all GPUs at once (slots from 0, then the state); (seconds, HBM
        bytes per GPU)."""
        ctx = len(kv)
        slots, sslots = list(range(ctx)), list(range(ctx, ctx + self.state_slots))
        for d in range(self.n):
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        for e in self.eng:
            self.op += 1
            e.submit(TierMove(self.op, op, "dram", None, 0, ctx, 0, kv, st, slots, sslots, [], []))
        if sub is not None:
            sub.append(time.perf_counter() - t0)
        nbytes = 0
        for e in self.eng:
            _, ev, _, nb = e.q["read" if op == "read" else "write"].get()
            ev.synchronize()
            nbytes = nb
        return time.perf_counter() - t0, nbytes

    def rw(self, ctx: int | None = None, reps: int = 5, check: bool = True) -> dict:
        ctx = ctx or CTX
        src_kv, src_st = list(range(1, 1 + ctx)), list(range(6))
        dst_kv, dst_st = list(range(1 + ctx, 1 + 2 * ctx)), list(range(6, 12))
        self.move("write", src_kv, src_st)                      # warm-up
        self.move("read", dst_kv, dst_st)
        w = [self.move("write", src_kv, src_st) for _ in range(reps)]
        for v in self.views:                                    # destination scrambled, then read back
            v[dst_kv if v.shape[0] == NB else dst_st] = 0
        r = [self.move("read", dst_kv, dst_st) for _ in range(reps)]
        sub = []                                                # host time to submit one op on all 4 engines
        for _ in range(3):
            self.move("read", dst_kv, dst_st, sub)
        bad = []
        for v, vr in zip(self.views, self.ranges):
            if v.shape[0] == NB:
                diff = (v[src_kv] != v[dst_kv]).any(dim=1).nonzero().flatten().tolist()
            else:                                               # a state range moves its group's row only
                diff = [] if torch.equal(v[src_st[vr.group]], v[dst_st[vr.group]]) else [vr.group]
            if diff:
                bad.append([vr.name, diff[:8], len(diff)])
        equal = not bad
        best = lambda xs: max(nb / t for t, nb in xs) / 1e9
        med = lambda xs: sorted(t for t, _ in xs)[len(xs) // 2] * 1e3
        rec = dict(test="rw", ring=self.ring, ctx_blocks=ctx, gpus=self.n, hbm_bytes_per_gpu=r[0][1],
                   read_ms_p50=round(med(r), 3), write_ms_p50=round(med(w), 3),
                   read_gbps_per_gpu=round(best(r), 1), write_gbps_per_gpu=round(best(w), 1),
                   read_ms=[round(t * 1e3, 2) for t, _ in r], write_ms=[round(t * 1e3, 2) for t, _ in w],
                   submit_ms_4gpus=[round(t * 1e3, 2) for t in sub], bytes_equal=equal, bad=bad[:40])
        self.log(**rec)
        if check:
            assert equal, "two-hop round trip changed bytes"
        return rec

    def sweep(self):
        """Ring depth x chunk (TIERS.md §3): 32k (4 blocks), 128k (16) and 512k (62) reads and writes, 10 each; the
        default ring is rebuilt at the end. Byte equality is recorded per point, the probe stops after the sweep if
        any point failed."""
        default = self.ring
        recs = []
        for ring in [dict(hops="direct")] + [dict(ring_depth=d, ring_chunk_slots=c) for d in (2, 4) for c in (1, 3, 4)]:
            self.build(ring)
            for ctx in (4, 16, 62):
                recs.append(self.rw(ctx, reps=10, check=False))
        self.build(default)
        assert all(r["bytes_equal"] for r in recs), "a ring setting changed bytes (see the rw lines)"

    def pieces(self):
        """`cuMemcpyBatchAsync` of N pieces of S bytes (512 MB in all) on GPU 0's stream: local (0 -> 0), push (0 -> 1,
        GPU 0's copy engine writes peer HBM), pull (1 -> 0, it reads peer HBM), host -> 0; then all 4 GPUs at once,
        each pushing to the other 3 and itself (the read scatter's shape) and each pulling from all 4 (the write
        gather's shape). GB/s and microseconds per piece."""
        from vllm import _custom_ops
        total = 512 << 20
        buf = [torch.empty(total, dtype=torch.uint8, device=g) for g in range(self.n)]
        dst = [torch.empty(total, dtype=torch.uint8, device=g) for g in range(self.n)]
        host = self.eng[0].dram
        streams = [torch.cuda.Stream(g) for g in range(self.n)]

        def batch(g, srcs, dsts, size):
            with torch.cuda.stream(streams[g]):
                t = lambda x: torch.tensor(x, dtype=torch.int64)
                _custom_ops.swap_blocks_batch(t(srcs), t(dsts), t([size] * len(srcs)))

        def timed(jobs, reps=3):
            for d in range(self.n):
                torch.cuda.synchronize(d)
            best = None
            for _ in range(reps):
                t0 = time.perf_counter()
                for j in jobs:
                    j()
                for d in range(self.n):
                    torch.cuda.synchronize(d)
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            return best

        for size in (32 << 10, 128 << 10, 256 << 10, 1 << 20, 4 << 20, 12_857_344):
            n = total // size
            offs = [i * size for i in range(n)]
            res = {}
            cases = {"local": (buf[0].data_ptr(), dst[0].data_ptr()), "push": (buf[0].data_ptr(), dst[1].data_ptr()),
                     "pull": (buf[1].data_ptr(), dst[0].data_ptr())}
            if host is not None and host.nbytes >= total:
                cases["h2d"] = (host.ptr, dst[0].data_ptr())
            for name, (a, b) in cases.items():
                dt = timed([lambda a=a, b=b: batch(0, [a + o for o in offs], [b + o for o in offs], size)])
                res[name] = dict(gbps=round(total / dt / 1e9, 1), us_per_piece=round(dt / n * 1e6, 2))
            q = n // self.n                                     # all GPUs: g sends q pieces to each GPU
            push = [lambda g=g: batch(g, [buf[g].data_ptr() + i * size for i in range(q) for _ in range(self.n)],
                                      [dst[h].data_ptr() + (g * q // self.n + i // self.n) * size
                                       for i in range(q) for h in range(self.n)][:q * self.n], size)
                    for g in range(self.n)]
            dt = timed(push)
            res["all_push"] = dict(gbps_per_gpu=round(q * self.n * size / dt / 1e9, 1))
            pull = [lambda g=g: batch(g, [buf[h].data_ptr() + (g * q // self.n + i // self.n) * size
                                          for i in range(q) for h in range(self.n)][:q * self.n],
                                      [dst[g].data_ptr() + i * size for i in range(q) for _ in range(self.n)], size)
                    for g in range(self.n)]
            dt = timed(pull)
            res["all_pull"] = dict(gbps_per_gpu=round(q * self.n * size / dt / 1e9, 1))
            self.log(test="pieces", size=size, n=n, **res)
        del buf, dst

    def reads_loop(self, stop: threading.Event, done: list):
        dst_kv, dst_st = list(range(1 + CTX, 1 + 2 * CTX)), list(range(6, 12))
        while not stop.is_set():
            done.append(self.move("read", dst_kv, dst_st))

    def gemm(self):
        res = {}
        for g in (0, 1):
            a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=g)
            b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=g)
            cs = torch.cuda.Stream(g)

            def gemms(k=60):
                with torch.cuda.stream(cs):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(k):
                        a @ b
                    e.record()
                e.synchronize()
                return s.elapsed_time(e) / k

            gemms(10)
            alone = gemms()
            stop, done = threading.Event(), []
            t = threading.Thread(target=self.reads_loop, args=(stop, done))
            t.start()
            time.sleep(0.05)
            busy = gemms()
            stop.set()
            t.join()
            again = gemms()
            rate = sum(nb for _, nb in done) / sum(x for x, _ in done) / 1e9 if done else 0
            res[f"gpu{g}"] = dict(alone_ms=round(alone, 4), with_reads_ms=round(busy, 4),
                                  alone_again_ms=round(again, 4), reads=len(done), read_gbps_per_gpu=round(rate, 1))
        self.log(test="gemm_vs_two_hop_reads", **res)

    def edges(self):
        """Stage edges as CePipe sends them: one cudaMemcpyAsync s -> s+1 per microbatch on a copy stream."""
        nbytes = EDGE_TOKENS * HIDDEN_WIDTH * 2
        src = [torch.empty(nbytes, dtype=torch.uint8, device=s) for s in range(self.n - 1)]
        dst = [torch.empty(nbytes, dtype=torch.uint8, device=s + 1) for s in range(self.n - 1)]
        streams = [torch.cuda.Stream(s) for s in range(self.n - 1)]

        def send(k=20):
            evs = []
            for s in range(self.n - 1):
                with torch.cuda.stream(streams[s]):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(k):
                        dst[s].copy_(src[s], non_blocking=True)
                    b.record()
                evs.append((a, b))
            for a, b in evs:
                b.synchronize()
            return [round(a.elapsed_time(b) / k, 3) for a, b in evs]

        send(3)
        alone = send()
        stop, done = threading.Event(), []
        t = threading.Thread(target=self.reads_loop, args=(stop, done))
        t.start()
        time.sleep(0.05)
        busy = send()
        n_busy = len(done)
        stop.set()
        t.join()
        rate = sum(nb for _, nb in done) / sum(x for x, _ in done) / 1e9 if done else 0
        self.log(test="edges_vs_two_hop_reads", edge_bytes=nbytes, edge_ms_alone=alone, edge_ms_with_reads=busy,
                 edge_gbps_alone=[round(nbytes / m / 1e6, 1) for m in alone],
                 edge_gbps_with_reads=[round(nbytes / m / 1e6, 1) for m in busy], reads_during=n_busy,
                 read_gbps_per_gpu_during_edges=round(rate, 1))
        stop, done2 = threading.Event(), []                       # read rate with the edges copying throughout
        t = threading.Thread(target=self.reads_loop, args=(stop, done2))
        t.start()
        send(40)
        stop.set()
        t.join()
        rate2 = sum(nb for _, nb in done2) / sum(x for x, _ in done2) / 1e9 if done2 else 0
        self.log(test="reads_vs_edges", reads=len(done2), read_gbps_per_gpu=round(rate2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=62, help="blocks per context (62: 512k)")
    ap.add_argument("--tests", default="pieces,sweep,gemm,edges")
    ap.add_argument("--hops", default="two")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=4, help="slots per ring chunk")
    a = ap.parse_args()
    global CTX
    CTX = a.ctx
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ring = dict(hops=a.hops) if a.hops == "direct" else dict(ring_depth=a.depth, ring_chunk_slots=a.chunk)
    p = Probe(out, ring)
    for t in a.tests.split(","):
        t0 = time.time()
        getattr(p, t)()
        print(f"== {t} {time.time() - t0:.1f} s", flush=True)
    for e in p.eng:
        e.close()


if __name__ == "__main__":
    main()
