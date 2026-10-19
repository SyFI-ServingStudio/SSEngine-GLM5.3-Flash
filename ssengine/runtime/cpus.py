"""CPU sets: each worker on a few whole cores of its GPU's NUMA node, the controller and the intake processes on the
rest of the job's CPUs, so other processes on a loaded host (and our own threads) do not preempt the stage that
launches the kernels.

Every worker computes the same assignment from the affinity it inherits from the controller (the job's CPUs), so no
messages are needed; the controller then reads the workers' sets back and takes what is left."""

import glob
import os
from pathlib import Path

WORKER_CORES = 2   # physical cores per worker, both hyperthreads: main thread, reporter, gloo, zmq, CUDA driver
CONTROLLER_CORES = 2   # with intake processes: the controller's own, the intakes get the rest


def _siblings(cpu: int) -> tuple[int, ...]:
    p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return tuple(_parse(p.read_text())) if p.exists() else (cpu,)


def _parse(lst: str) -> list[int]:
    out = []
    for part in lst.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def _node_of_cpu(cpu: int) -> int:
    nodes = glob.glob(f"/sys/devices/system/cpu/cpu{cpu}/node*")
    return int(Path(nodes[0]).name[4:]) if nodes else 0


def _gpu_node(index: int) -> int:
    import torch
    p = torch.cuda.get_device_properties(index)
    f = Path(f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/numa_node")
    n = int(f.read_text()) if f.exists() else -1
    return max(n, 0)


def worker_cpus(stage: int, num_stages: int) -> set[int] | None:
    """This worker's CPUs, or None when the job has too few whole cores to give every worker its own."""
    allowed = os.sched_getaffinity(0)
    cores = sorted({s for c in allowed if (s := _siblings(c)) and set(s) <= allowed})
    if len(cores) < num_stages * WORKER_CORES + 1:            # keep at least one core for the controller
        return None
    nodes = [_gpu_node(s) for s in range(num_stages)]
    used: set[tuple[int, ...]] = set()
    for s in range(num_stages):
        free = [c for c in cores if c not in used]
        local = [c for c in free if _node_of_cpu(c[0]) == nodes[s]]
        pick = (local if len(local) >= WORKER_CORES else free)[:WORKER_CORES]
        used.update(pick)
        if s == stage:
            return {cpu for c in pick for cpu in c}
    return None


def split_cores(cpus: set[int], k: int) -> tuple[set[int], set[int]]:
    """The first k whole cores of `cpus`, and the rest (or the same k cores when nothing is left)."""
    cores = sorted({tuple(c for c in _siblings(cpu) if c in cpus) for cpu in cpus})
    first = {cpu for c in cores[:k] for cpu in c}
    return first, (cpus - first) or first


def pin_process(cpus: set[int]):
    """Every thread of this process (os.sched_setaffinity(0) moves only the calling one); later threads inherit."""
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except ProcessLookupError:
            pass


def cpu_list(cpus) -> str:
    """{0, 1, 2, 112, 113} -> "0-2,112-113" (the kernel's cpulist format)."""
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def threads(pid: int | str = "self") -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except FileNotFoundError:
        return 0
