"""The host media (ppimpl/kvpool/backends.py) on CPU: the arena, the SSD file's slot round trip and its unlink, and
the OOM guards of TIERS.md §14.1 (preferred NUMA placement, oom_score_adj, residency per node, cgroup peak)."""

import ctypes
import os

import numpy as np

from ppimpl.kvpool import backends as bk


def test_arena_is_plain_memory_on_cpu():
    a = bk.HostTier(3 << 20, None, False)
    assert a.nbytes == 4 << 20 and a.base % 4096 == 0
    ctypes.memset(a.base + a.nbytes - 1, 7, 1)
    a.register(1 << 20)
    assert a.ptr(3) == a.base + (3 << 20)


def test_ssd_slots_round_trip_and_file_is_unlinked(tmp_path):
    sb = 8192
    f = bk.SsdTier(str(tmp_path / "d" / "ssd.bin"), 6 * sb, sb)
    assert os.path.getsize(f.path) == 6 * sb
    src = np.random.default_rng(1).integers(0, 255, 3 * sb + 4096, dtype=np.uint8)
    dst = np.zeros_like(src)
    p0, p1 = (x.ctypes.data + (-x.ctypes.data) % 4096 for x in (src, dst))
    for fut in f.write([5, 0, 2], [p0 + k * sb for k in range(3)]):
        fut.result()
    for fut in f.read([2, 5, 0], [p1 + k * sb for k in (2, 0, 1)]):
        fut.result()
    a, b = (np.frombuffer((ctypes.c_char * (3 * sb)).from_address(p), dtype=np.uint8) for p in (p0, p1))
    assert np.array_equal(a, b)
    f.close()
    assert not os.path.exists(f.path)


def test_numa_resident_gb(tmp_path):
    """The arena's residency per node from numa_maps: only mappings starting inside the arena, in their page size."""
    maps = tmp_path / "numa_maps"
    maps.write_text("7f0000000000 bind:1 anon=100 dirty=100 N1=100 kernelpagesize_kB=4\n"
                    "7f1000000000 bind:1 anon=262144 dirty=262144 N0=6 N1=262138 kernelpagesize_kB=4\n"
                    "7f1040000000 bind:1 anon=10 N1=10 kernelpagesize_kB=2048\n"
                    "7f2000000000 default file=/x N0=5 kernelpagesize_kB=4\n")
    got = bk.numa_resident_gb(0x7f1000000000, 0x1000000000, str(maps))
    assert got == {0: round(6 * 4096 / 1e9, 2), 1: round((262138 * 4096 + 10 * (2 << 20)) / 1e9, 2)}


def test_node_cpus():
    """The touch threads' CPUs: the job's CPUs on the arena's node, or all of the job's when the node has few."""
    node_cpus = bk.node_cpus
    node0 = set(bk._cpulist(open("/sys/devices/system/node/node0/cpulist").read()))
    assert node_cpus(None, 0) is None and node_cpus({1, 2}, None) == {1, 2}
    few = set(sorted(node0)[:3]) | {10_000}
    assert node_cpus(few, 0) == few                     # fewer than 8 on the node: all of them
    many = set(sorted(node0)[:12]) | {10_000}
    assert node_cpus(many, 0) == set(sorted(node0)[:12])


def test_prefer_node_is_preferred_not_bound():
    """The arena's policy is MPOL_PREFERRED (7996: a strict bind OOM-killed other users' processes on a full node)."""
    nodes = sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d[4:].isdigit())
    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    n = 64 << 20
    for node in nodes:                      # each node in turn: placement follows the policy, not the touching CPU
        _prefer_and_check(numa, n, node)


def _prefer_and_check(numa, n, node):
    ptr = bk.anon_map(n)
    try:
        bk.prefer_node(numa, ptr, n, node)
        ctypes.memset(ptr, 0, n)
        line = next(x for x in open("/proc/self/numa_maps") if int(x.split()[0], 16) == ptr)
        assert f"prefer:{node}" in line and "bind:" not in line
        assert bk.numa_resident_gb(ptr, n).get(node, 0) > 0.06
    finally:
        bk.anon_unmap(ptr, n)


def test_oom_first_and_cgroup_peak():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "from ppimpl.kvpool.backends import oom_first; oom_first(); "
                          "print(open('/proc/self/oom_score_adj').read())"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1000"
    pk = bk.cgroup_peak_gb()
    assert pk is None or pk > 0
