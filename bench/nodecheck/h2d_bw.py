"""Pinned host DRAM -> HBM copy bandwidth, one GPU at a time then all visible GPUs at once.

Reads 4 GiB per copy from a pinned buffer on the GPU's NUMA node (as the process is
launched), 10 timed copies after 2 warm-ups. Prints GB/s (1e9 B/s).
"""

import threading

import torch

SIZE = 4 << 30
REPS = 10


def bw(dev, results, key):
    torch.cuda.set_device(dev)
    host = torch.empty(SIZE, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)
    dst = torch.empty(SIZE, dtype=torch.uint8, device=dev)
    s = torch.cuda.Stream(dev)
    with torch.cuda.stream(s):
        for _ in range(2):
            dst.copy_(host, non_blocking=True)
        s.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(s)
        for _ in range(REPS):
            dst.copy_(host, non_blocking=True)
        end.record(s)
        end.synchronize()
    results[key] = SIZE * REPS / (start.elapsed_time(end) / 1e3) / 1e9


n = torch.cuda.device_count()
single = {}
for d in range(n):
    bw(d, single, d)
    print(f"h2d single gpu{d}: {single[d]:.1f} GB/s")
together = {}
threads = [threading.Thread(target=bw, args=(d, together, d)) for d in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("h2d concurrent: " + ", ".join(f"gpu{d} {together[d]:.1f}" for d in range(n))
      + f"; sum {sum(together.values()):.1f} GB/s")
