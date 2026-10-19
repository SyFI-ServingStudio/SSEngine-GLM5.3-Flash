"""Host <-> GPU transfer bandwidth for the DRAM tier design (7_offloading.md §2, lanes/m6 TIERS.md "Transfers").

Pinned host memory (libnuma on a chosen node + cudaHostRegister, as the tier will allocate it) against the job's
GPUs, one test at a time, every result one JSON line in OUT/tier_bw.jsonl and a line on stdout:
- ce:     one cudaMemcpyAsync per transfer (copy engine), H2D / D2H, sizes 64 KiB .. 1 GiB, NUMA local / remote,
          one GPU alone, and all GPUs at once (H2D, D2H, both directions together);
- batch:  cuMemcpyBatchAsync (vLLM's swap_blocks_batch) of scattered device pages (64 KiB .. 8 MiB) to/from
          contiguous host memory, the tier's shape;
- sm:     an SM kernel that loads pinned host memory over UVA and stores to HBM (vLLM's triton swap kernel), by
          number of CTAs, to price SM-based gather in SMs;
- peer:   GPU g's copy engine moving host memory on g's NUMA node into every GPU's HBM (balanced storage: each GPU's
          link carries a quarter of a context, the bytes land on the stage that owns them), and back;
- gemm:   a BF16 GEMM loop alone, with all-GPU CE H2D running, and with the SM kernel running (the copies' cost to
          compute).
usage: python -m research.bench.tier_bw --out DIR [--quick]
"""

import argparse
import ctypes
import json
import os
import threading
import time
from pathlib import Path

import torch

GiB, MiB, KiB = 1 << 30, 1 << 20, 1 << 10
H2D, D2H = "h2d", "d2h"


def cudart():
    import nvidia.cu13 as cu  # the runtime torch itself loads
    lib = ctypes.CDLL(str(Path(list(cu.__path__)[0]) / "lib" / "libcudart.so.13"))
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    return lib


RT = cudart()
NUMA = ctypes.CDLL("libnuma.so.1")


def alloc_on(nbytes: int, node: int) -> int:
    """nbytes preferred (not bound) on `node`: a strict bind OOM-kills other processes on a full node."""
    from ppimpl.kvpool.backends import anon_map, oom_first, prefer_node
    oom_first()
    ptr = anon_map(nbytes)
    prefer_node(NUMA, ptr, nbytes, node)
    return ptr


def check(rc, what):
    if rc not in (0, 704):    # 704: peer access already enabled
        raise RuntimeError(f"{what}: cuda error {rc}")


def host_buffer(nbytes: int, node: int) -> torch.Tensor:
    """Pinned host memory on NUMA node `node`: first-touched there, registered portable (every GPU sees it pinned)."""
    ptr = alloc_on(nbytes, node)
    ctypes.memset(ptr, 1, nbytes)
    check(RT.cudaHostRegister(ptr, nbytes, 3), "cudaHostRegister")   # portable | mapped
    return torch.frombuffer((ctypes.c_uint8 * nbytes).from_address(ptr), dtype=torch.uint8)


def gpu_numa(dev: int) -> int:
    p = torch.cuda.get_device_properties(dev)
    bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text())


def run_parallel(fns):
    """Run one closure per GPU in its own thread (each enqueues its copies); return the wall seconds from a common
    start to the last GPU's end (every stream synchronized)."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    barrier = threading.Barrier(len(fns) + 1)
    out = [None] * len(fns)

    def body(i, fn):
        barrier.wait()
        out[i] = fn()

    ts = [threading.Thread(target=body, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    return time.perf_counter() - t0, out


class Bench:
    def __init__(self, out: Path, quick: bool):
        self.out = out
        self.quick = quick
        self.n = torch.cuda.device_count()
        self.node = [gpu_numa(d) for d in range(self.n)]
        nodes = sorted(set(int(x.name[4:]) for x in Path("/sys/devices/system/node").glob("node[0-9]*")))
        self.remote = {d: next(n for n in nodes if n != self.node[d]) for d in range(self.n)}
        self.hb = 2 * GiB
        self.host = {}       # (dev, "local"/"remote") -> pinned buffer on that NUMA node
        for d in range(self.n):
            self.host[d, "local"] = host_buffer(self.hb, self.node[d])
            self.host[d, "remote"] = host_buffer(GiB, self.remote[d])
        self.dev = [torch.empty(self.hb, dtype=torch.uint8, device=d) for d in range(self.n)]
        self.streams = [torch.cuda.Stream(d) for d in range(self.n)]
        self.streams2 = [torch.cuda.Stream(d) for d in range(self.n)]
        for g in range(self.n):
            torch.cuda.set_device(g)
            check(RT.cudaSetDevice(g), "cudaSetDevice")     # this library's own current device, if it is not torch's
            for s in range(self.n):
                if s != g:
                    check(RT.cudaDeviceEnablePeerAccess(s, 0), "cudaDeviceEnablePeerAccess")
        self.f = open(out / "tier_bw.jsonl", "a")

    def log(self, **rec):
        rec["t"] = time.strftime("%H:%M:%S")
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        print(json.dumps(rec), flush=True)

    # --- copy-engine closures ---------------------------------------------------------------------------------

    def ce_job(self, g, direction, size, total, numa="local", dst_dev=None, stream=None):
        """GPU g's stream copies `total` bytes in `size` transfers between host (`numa` of g) and device `dst_dev`
        (default g), by cudaMemcpyAsync on g's stream: g's copy engine moves them."""
        host = self.host[g, numa]
        dev = self.dev[g if dst_dev is None else dst_dev]
        st = stream or self.streams[g]
        span = min(host.numel(), dev.numel()) // size * size
        reps = max(1, total // size)

        def fn():
            torch.cuda.set_device(g)
            check(RT.cudaSetDevice(g), "cudaSetDevice")
            hp, dp = host.data_ptr(), dev.data_ptr()
            for i in range(reps):
                off = (i * size) % span
                src, dst = (hp + off, dp + off) if direction == H2D else (dp + off, hp + off)
                check(RT.cudaMemcpyAsync(dst, src, size, 4, st.cuda_stream), "cudaMemcpyAsync")
            return reps * size
        return fn

    def measure(self, fns):
        wall, nbytes = run_parallel(fns)
        return sum(nbytes) / wall / 1e9, wall

    # --- tests ------------------------------------------------------------------------------------------------

    def ce_single(self):
        sizes = [64 * KiB, 256 * KiB, MiB, 4 * MiB, 16 * MiB, 64 * MiB, 256 * MiB, GiB]
        for g in range(self.n):
            for numa in ("local", "remote"):
                for direction in (H2D, D2H):
                    for size in (sizes if g == 0 and numa == "local" else [16 * MiB, GiB]):
                        total = max(size, min(4 * GiB, 4096 * size))
                        self.measure([self.ce_job(g, direction, size, total, numa)])   # warm
                        bw, wall = self.measure([self.ce_job(g, direction, size, total, numa)])
                        self.log(test="ce", gpus=[g], numa=numa, dir=direction, size=size, gb_s=round(bw, 2),
                                 per_gpu_gb_s=round(bw, 2), wall_s=round(wall, 4))

    def ce_all(self):
        for numa in ("local", "remote"):
            for size in [MiB, 16 * MiB, GiB]:
                total = 4 * GiB
                for direction in (H2D, D2H, "bidir"):
                    if direction == "bidir":
                        fns = [self.ce_job(g, H2D, size, total, numa) for g in range(self.n)] + \
                              [self.ce_job(g, D2H, size, total, numa, stream=self.streams2[g]) for g in range(self.n)]
                    else:
                        fns = [self.ce_job(g, direction, size, total, numa) for g in range(self.n)]
                    self.measure(fns)
                    bw, wall = self.measure(fns)
                    self.log(test="ce", gpus=list(range(self.n)), numa=numa, dir=direction, size=size,
                             gb_s=round(bw, 2), per_gpu_gb_s=round(bw / self.n, 2), wall_s=round(wall, 4))

    def batch_job(self, g, direction, page, total, dst_devs=None):
        """cuMemcpyBatchAsync of `total` bytes as `page`-sized pieces: host contiguous, device pages scattered
        (every other page slot), on GPU g's stream. dst_devs: the pages' devices, round robin (peer pages)."""
        from vllm import _custom_ops as ops
        n = total // page
        devs = dst_devs or [g]
        host = self.host[g, "local"]
        slots = torch.randperm(self.hb // page // len(devs))[:n // len(devs) + 1]
        hptr = torch.arange(n, dtype=torch.int64) * page + host.data_ptr()
        dptr = torch.tensor([self.dev[devs[i % len(devs)]].data_ptr() + int(slots[i // len(devs)]) * page
                             for i in range(n)], dtype=torch.int64)
        sizes = torch.full((n,), page, dtype=torch.int64)
        src, dst = (hptr, dptr) if direction == H2D else (dptr, hptr)

        def fn():
            torch.cuda.set_device(g)
            with torch.cuda.stream(self.streams[g]):
                ops.swap_blocks_batch(src, dst, sizes)
            return n * page
        return fn

    def batch(self):
        for page in [64 * KiB, 512 * KiB, 2 * MiB, 8 * MiB]:
            for direction in (H2D, D2H):
                for gpus in ([0], list(range(self.n))):
                    fns = [self.batch_job(g, direction, page, GiB) for g in gpus]
                    self.measure(fns)
                    bw, wall = self.measure(fns)
                    self.log(test="batch", gpus=gpus, numa="local", dir=direction, size=page, gb_s=round(bw, 2),
                             per_gpu_gb_s=round(bw / len(gpus), 2), wall_s=round(wall, 4))

    def sm_job(self, g, direction, page, total, ctas, chunk=8192):
        from vllm.v1.kv_offload.cpu.swap_blocks_triton import _swap_blocks_kernel
        n = total // page
        host = self.host[g, "local"]
        hptr = torch.arange(n, dtype=torch.int64) * page + host.data_ptr()
        dptr = torch.arange(n, dtype=torch.int64) * page + self.dev[g].data_ptr()
        src, dst = (hptr, dptr) if direction == H2D else (dptr, hptr)
        src, dst = src.to(g), dst.to(g)
        sizes = torch.full((n,), page, dtype=torch.int64, device=g)

        def fn():
            torch.cuda.set_device(g)
            with torch.cuda.stream(self.streams[g]):
                _swap_blocks_kernel[(ctas,)](src, dst, sizes, n, BYTES_PER_CHUNK=chunk)
            return n * page
        return fn

    def sm(self):
        for direction in (H2D, D2H):
            for ctas in [4, 8, 16, 32, 64, 148]:
                fns = [self.sm_job(0, direction, 2 * MiB, GiB, ctas)]
                self.measure(fns)
                bw, wall = self.measure(fns)
                self.log(test="sm", gpus=[0], numa="local", dir=direction, size=2 * MiB, ctas=ctas,
                         gb_s=round(bw, 2), per_gpu_gb_s=round(bw, 2), wall_s=round(wall, 4))
            for ctas in [16, 32]:
                fns = [self.sm_job(g, direction, 2 * MiB, GiB, ctas) for g in range(self.n)]
                self.measure(fns)
                bw, wall = self.measure(fns)
                self.log(test="sm", gpus=list(range(self.n)), numa="local", dir=direction, size=2 * MiB, ctas=ctas,
                         gb_s=round(bw, 2), per_gpu_gb_s=round(bw / self.n, 2), wall_s=round(wall, 4))

    def peer(self):
        """Balanced storage: GPU g's copy engine reads host memory on g's node and writes a quarter to each GPU's
        HBM (3 of 4 over NVLink), all GPUs at once; and back (D2H from every GPU's HBM into g's host memory)."""
        for size in [2 * MiB, 64 * MiB]:
            for direction in (H2D, D2H):
                fns = []
                for g in range(self.n):
                    jobs = [self.ce_job(g, direction, size, GiB // self.n, "local", dst_dev=s) for s in range(self.n)]
                    fns.append(lambda jobs=jobs: sum(j() for j in jobs))
                self.measure(fns)
                bw, wall = self.measure(fns)
                self.log(test="peer", gpus=list(range(self.n)), numa="local", dir=direction, size=size,
                         gb_s=round(bw, 2), per_gpu_gb_s=round(bw / self.n, 2), wall_s=round(wall, 4))
        for direction in (H2D, D2H):    # the same through cuMemcpyBatchAsync with pages on all GPUs
            fns = [self.batch_job(g, direction, 2 * MiB, GiB, dst_devs=list(range(self.n))) for g in range(self.n)]
            try:
                self.measure(fns)
                bw, wall = self.measure(fns)
                self.log(test="peer_batch", gpus=list(range(self.n)), numa="local", dir=direction, size=2 * MiB,
                         gb_s=round(bw, 2), per_gpu_gb_s=round(bw / self.n, 2), wall_s=round(wall, 4))
            except Exception as e:  # noqa: BLE001
                self.log(test="peer_batch", dir=direction, error=repr(e))

    def register(self):
        """Time to make 16 GiB of pinned host memory on GPU 0's node: allocate, first touch, cudaHostRegister (the
        tier pins 150 GB per GPU at start-up)."""
        t0 = time.perf_counter()
        ptr = alloc_on(16 * GiB, self.node[0])
        ctypes.memset(ptr, 0, 16 * GiB)
        t1 = time.perf_counter()
        check(RT.cudaHostRegister(ptr, 16 * GiB, 3), "cudaHostRegister")
        t2 = time.perf_counter()
        self.log(test="register", gib=16, touch_s=round(t1 - t0, 2), register_s=round(t2 - t1, 2))

    def gemm(self):
        """GPU 0: 8192^3 BF16 GEMMs, alone; with every GPU's CE H2D running; with the SM kernel (16 CTAs) running."""
        torch.cuda.set_device(0)
        a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=0)
        b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=0)
        cs = torch.cuda.Stream(0)

        def gemms(k=40):
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
        res = dict(alone_ms=round(alone, 3))
        for name, fns in [("ce_h2d_all", [self.ce_job(g, H2D, 64 * MiB, 32 * GiB) for g in range(self.n)]),
                          ("sm16_h2d", [self.sm_job(0, H2D, 2 * MiB, GiB, 16) for _ in range(8)])]:
            torch.cuda.synchronize(0)
            t = threading.Thread(target=lambda: run_parallel(fns) if name.startswith("ce") else [f() for f in fns])
            t.start()
            time.sleep(0.05)
            res[name + "_ms"] = round(gemms(), 3)
            t.join()
            for d in range(self.n):
                torch.cuda.synchronize(d)
        self.log(test="gemm", **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", default="ce_single,ce_all,batch,sm,peer,gemm,register")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = Bench(out, a.quick)
    b.log(test="setup", gpus=b.n, numa=b.node, remote=b.remote, visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    for t in a.tests.split(","):
        t0 = time.time()
        getattr(b, t)()
        print(f"== {t} {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
