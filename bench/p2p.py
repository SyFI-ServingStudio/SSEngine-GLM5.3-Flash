"""Activation transfer between two GPUs: NCCL send/recv (the runtime's path, ppimpl/runtime/transport.py) and a
peer copy (copy engine), per prompt size, against the iteration targets' send times.

  python bench/p2p.py [--out FILE]        on 2 GPUs (Slurm); NCCL_* variables in the environment apply

Per size: the receive is posted first, then the send; times are CUDA events around the NCCL op on the sender (the
op waits for the receiver, which is already waiting), min and median over the iterations.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BYTES_PER_TOKEN = 16384 * 2          # ppimpl.config.HIDDEN_WIDTH bf16 columns
TOKENS = [1024, 2048, 4096, 8192, 16384]
ITERS = 30


def nccl_rank(rank: int, port: int, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    cpu = dist.new_group(backend="gloo")
    # The first send/recv creates the pair's communicator, which needs both ranks inside NCCL at once.
    t = torch.zeros(1, device=rank)
    dist.send(t, 1) if rank == 0 else dist.recv(t, 0)
    torch.cuda.synchronize()
    out = {}
    for T in TOKENS:
        buf = torch.ones(T, BYTES_PER_TOKEN // 2, dtype=torch.bfloat16, device=rank)
        times = []
        for i in range(ITERS + 3):
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            if rank == 1:
                work = dist.irecv(buf, 0)
                dist.barrier(group=cpu)          # the receive is posted before the send starts
                work.wait()
            else:
                dist.barrier(group=cpu)
                time.sleep(0.002)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                work = dist.isend(buf, 1)
                work.wait()
                b.record()
                b.synchronize()
                if i >= 3:
                    times.append(a.elapsed_time(b))
        if rank == 0:
            out[T] = times
    if rank == 0:
        q.put(out)
    dist.barrier(group=cpu)
    dist.destroy_process_group()


def peer_copy() -> dict:
    out = {}
    for T in TOKENS:
        src = torch.ones(T, BYTES_PER_TOKEN // 2, dtype=torch.bfloat16, device=0)
        dst = torch.empty_like(src, device=1)
        times = []
        for i in range(ITERS + 3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(torch.cuda.current_stream(0))
            dst.copy_(src, non_blocking=True)
            b.record(torch.cuda.current_stream(0))
            b.synchronize()
            if i >= 3:
                times.append(a.elapsed_time(b))
        out[T] = times
    return out


def summary(times: dict) -> dict:
    return {T: {"min_ms": min(t), "p50_ms": statistics.median(t),
                "GBps_at_p50": T * BYTES_PER_TOKEN / statistics.median(t) / 1e6} for T, t in times.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=29611)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=nccl_rank, args=(r, a.port, q)) for r in range(2)]
    for p in procs:
        p.start()
    nccl = q.get()
    for p in procs:
        p.join()
    res = {"env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
           "nccl": summary(nccl), "peer_copy": summary(peer_copy())}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
