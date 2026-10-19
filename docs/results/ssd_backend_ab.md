# SSD backend: Linux AIO vs io_uring

The SSD tier reads and writes its shard files with O_DIRECT, one request per slot (12.86 MB), at most 8 reads and 4
writes in the kernel per GPU, reads first ([../design/tiers.md](../design/tiers.md)). Two backends implement that:
io_uring (one ring per GPU, file and host arenas registered as fixed buffers) and Linux native AIO (one context per
GPU, raw syscalls, an owner thread and 4 submit threads). Docker's default seccomp profile (Docker ≥ 25) denies
io_uring and allows AIO. **Result: equal throughput and latency at the engine's loads and equal session metrics; AIO
uses ~1.8× the CPU. AIO is the default.**

Node: 4 × B200 of an HGX B200 node ([setup.md](setup.md)); the SSD file system is an md RAID0 of 8 NVMe drives.

## Disk only

The SSD tier's I/O pattern without GPUs (`research/bench/ssd_io.py` on the `dev` branch): 4 processes (one per GPU)
on NUMA node 0, each with its own 16 GB file of 12,857,344 B slots and a host arena; a reader of whole sessions (3–60
random slots, one session at a time) and a write-back of 3–60 slots at a time paced to a rate. 20 s per run, mean of
the 4 processes, two runs per arm in alternating order. Read GB/s per process, CPU cores, read ms by session size
p50 / p99, write ms per slot p50 / p99.

| Load per process | Backend | Read GB/s | Cores | 3–10 slots | 31–60 slots | Write ms / slot |
|---|---|---|---|---|---|---|
| reads max + 3 GB/s writes | io_uring | 16.58 | 0.43 | 6.5 / 16.2 | 32.1 / 63.8 | 0.79 / 2.26 |
| | aio | 16.27 | 0.76 | 6.4 / 15.3 | 33.8 / 63.3 | 0.89 / 2.41 |
| reads max + 5 GB/s writes | io_uring | 11.89 | 0.41 | 9.0 / 18.8 | 47.5 / 73.4 | 1.21 / 2.76 |
| | aio | 11.30 | 0.70 | 9.2 / 17.7 | 49.6 / 75.8 | 1.58 / 3.09 |
| 10 GB/s reads + 3 GB/s writes | io_uring | 10.16 | 0.29 | 3.1 / 10.3 | 16.6 / 48.4 | 0.60 / 1.76 |
| | aio | 10.23 | 0.55 | 3.4 / 11.9 | 16.6 / 52.5 | 0.64 / 2.62 |
| 4 GB/s reads + 1 GB/s writes | io_uring | 4.03 | 0.11 | 2.5 / 4.8 | 12.7 / 24.7 | 0.39 / 1.14 |
| | aio | 4.03 | 0.21 | 2.4 / 4.6 | 9.7 / 28.8 | 0.40 / 0.87 |
| reads max, no writes | io_uring | 24.05 | 0.52 | 5.0 / 8.5 | 23.3 / 32.0 | |
| | aio | 24.09 | 0.97 | 4.6 / 8.6 | 23.3 / 32.6 | |
| writes max, no reads | io_uring | (12.61 written) | 0.30 | | | 1.01 / 1.75 |
| | aio | (12.49 written) | 0.42 | | | 1.03 / 1.66 |

- At the loads the engine runs (0.7–4 GB/s of reads per GPU in the simulator's victim-chain runs at N = 400–500),
  throughput, read latency and write-back rate are the same.
- At the array's limit with writes beside the reads, AIO read 2–5% less in these two runs; over all five runs of that
  load the arms overlap (io_uring 16.14–16.58 and 10.85–11.89 GB/s, AIO 15.72–16.27 and 11.04–11.37).
- AIO pins each piece's pages in `io_submit` (~0.4 ms of CPU per slot, about twice the ring's). One thread submitting
  serially doubled a light-load read's latency (31–60 slots p50 21.5–23.7 ms vs 12.2–12.7); 4 submit threads bring it
  to the ring's. 8 or 12 change nothing.

## Engine

**SSD bring-up** (job 8529, one session at a time, mean of 5 sessions): SSD read 8.41 / 26.27 / 93.6 ms (io_uring) vs
8.43 / 26.18 / 95.3 ms (AIO) at 32k / 128k / 512k tokens; round-1 TTFT 62.6 / 84.5 / 206.4 vs 62.8 / 84.4 / 207.5 ms.

**Sessions**, the M6 workload ([eval_m6.md](eval_m6.md)) at N = 300 with the "tiers" setting (HBM 12.2M tokens + DRAM
20 GB + SSD 128 GB per GPU), minutes 10–20. Both arms ran at once on the two GPU sets of one 8-GPU node (set A = GPUs
0–3, set B = 4–7), then swapped. SSD read / write: a tier op's time from sent to landed, all sizes.

| Job | Set | Backend | SM MHz (s0–s3) | r/s | TTFT p50/p90/p99 (s) | > 60 s | Hit H/D/S % | s0 busy | SSD read ms p50/p99 | SSD write ms p50/p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8530 | A | io_uring | 1796/1896/1886/1884 | 34.07 | 0.19/0.44/3.92 | 21 | 45.0/33.1/21.8 | 0.873 | 15.2 / 52.9 | 17.3 / 58.1 |
| 8530 | B | aio | 1640/1653/1626/1597 | 32.85 | 0.23/0.72/5.72 | 52 | 42.3/35.3/22.3 | 0.873 | 15.2 / 54.9 | 18.9 / 66.3 |
| 8532 | A | aio | 1664/1641/1562/1580 | 32.91 | 0.23/0.66/4.93 | 43 | 43.0/34.5/22.5 | 0.865 | 14.9 / 61.0 | 18.0 / 59.9 |
| 8532 | B | io_uring | 1638/1649/1629/1597 | 32.83 | 0.23/0.70/6.35 | 51 | 42.4/35.0/22.5 | 0.873 | 15.3 / 62.3 | 18.5 / 72.1 |
| 8534 | A | aio | 1662/1637/1560/1580 | 32.83 | 0.23/0.71/5.43 | 46 | 42.3/35.2/22.4 | 0.863 | 15.0 / 54.4 | 18.5 / 61.7 |
| 8534 | B | io_uring | 1635/1653/1894/1890 | 33.12 | 0.21/0.63/5.36 | 40 | 43.0/34.8/22.1 | 0.885 | 15.1 / 55.1 | 18.5 / 62.9 |

The runs differ by their GPUs' clocks, not by the backend: the four runs whose GPUs sat at the power cap (~1600 MHz)
give 32.83–32.91 r/s with either backend; the GPUs that ran uncapped (~1890 MHz: all of set A in 8530, stages 2–3 in
8534) give 34.07 and 33.12. SSD reads and writes take the same time on both.

## Decision

AIO is the default (`tiers.ssd_backend: aio`): equal on reads, write-backs and session metrics, and it runs in Docker
with the default seccomp profile. io_uring stays available as `--ssd-backend uring`; in Docker it needs
[docker/seccomp.json](../../docker/seccomp.json) and no memlock limit (README, "Docker").
