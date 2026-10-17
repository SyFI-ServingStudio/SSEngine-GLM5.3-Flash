"""Two-hop tier copies on the GPUs, the engine's own code path (lanes/m6 TIERS.md §2.3 "Second hop").

One process drives the job's 4 GPUs with four `TierEngine`s (`ppimpl/runtime/tier_worker.py`, `hops = "two"`), each on
its own GPU, over fake per-stage KV laid out like GLM-5.3 PP4 (DSA layers 3/2/3/3 at 545 B/token each, split 4:1
between an MLA-like and an indexer-like view; KDA states 9/9/8/8 of 4 MiB + 144 KiB; a 2 KiB tail per DSA layer).
Measured, every result one JSON line in OUT/tier_hop2.jsonl:
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
    def __init__(self, out: Path):
        self.out, self.n = out, torch.cuda.device_count()
        assert self.n == 4, f"needs 4 GPUs, have {self.n}"
        self.f = open(out / "tier_hop2.jsonl", "a")
        peer_access(self.n)
        self.views, ranges = fake_kv(self.n)
        tier_worker.share_ranges = lambda stage, n, r: (ranges, [])     # one process: no IPC
        tc = TierConfig(dram_gb=4.0, ssd_gb=None, hops="two")
        self.eng = []
        for g in range(self.n):
            e = tier_worker.TierEngine(NS(tiers=tc, num_stages=self.n, model="glm53", run_dir=str(out)), g,
                                       torch.device("cuda", g), None)
            e.attach(ranges)
            self.eng.append(e)
        self.state_slots = self.eng[0].layout.state_slots
        self.op = 0

    def log(self, **rec):
        print(json.dumps(rec), flush=True)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def move(self, op: str, kv: list[int], st: list[int]) -> tuple[float, int]:
        """One op of a CTX-block context on all GPUs at once (slots 0..CTX+state); (seconds, HBM bytes per GPU)."""
        slots, sslots = list(range(CTX)), list(range(CTX, CTX + self.state_slots))
        for d in range(self.n):
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        for e in self.eng:
            self.op += 1
            e.submit(TierMove(self.op, op, "dram", None, 0, CTX, 0, kv, st, slots, sslots, [], []))
        nbytes = 0
        for e in self.eng:
            _, ev, _, nb = e.q["read" if op == "read" else "write"].get()
            ev.synchronize()
            nbytes = nb
        return time.perf_counter() - t0, nbytes

    def rw(self):
        src_kv, src_st = list(range(1, 1 + CTX)), list(range(6))
        dst_kv, dst_st = list(range(1 + CTX, 1 + 2 * CTX)), list(range(6, 12))
        self.move("write", src_kv, src_st)                      # warm-up
        self.move("read", dst_kv, dst_st)
        w = [self.move("write", src_kv, src_st) for _ in range(5)]
        for v in self.views:                                    # destination scrambled, then read back
            v[dst_kv if v.shape[0] == NB else dst_st] = 0
        r = [self.move("read", dst_kv, dst_st) for _ in range(5)]
        equal = all(torch.equal(v[src_kv], v[dst_kv]) if v.shape[0] == NB else torch.equal(v[src_st], v[dst_st])
                    for v in self.views)
        best = lambda xs: max(nb / t for t, nb in xs) / 1e9
        self.log(test="rw_512k", gpus=self.n, hbm_bytes_per_gpu=r[0][1], write_gbps_per_gpu=round(best(w), 1),
                 read_gbps_per_gpu=round(best(r), 1), read_ms=[round(t * 1e3, 2) for t, _ in r],
                 write_ms=[round(t * 1e3, 2) for t, _ in w], bytes_equal=equal)
        assert equal, "two-hop round trip changed bytes"

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
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Probe(out)
    for t in ("rw", "gemm", "edges"):
        t0 = time.time()
        getattr(p, t)()
        print(f"== {t} {time.time() - t0:.1f} s", flush=True)
    for e in p.eng:
        e.close()


if __name__ == "__main__":
    main()
