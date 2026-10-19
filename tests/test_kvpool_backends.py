"""The host media (ssengine/kvpool/backends.py) on CPU: the arena, the SSD file's slot round trip, lock and reuse, and
the OOM guards of TIERS.md §14.1 (preferred NUMA placement, oom_score_adj, residency per node, cgroup peak)."""

import ctypes
import os

import numpy as np
import pytest

from ssengine.kvpool import backends as bk
from ssengine.kvpool import shards


def test_arena_is_plain_memory_on_cpu():
    a = bk.HostTier(3 << 20, None, False)
    assert a.nbytes == 4 << 20 and a.base % 4096 == 0
    ctypes.memset(a.base + a.nbytes - 1, 7, 1)
    a.register(1 << 20)
    assert a.ptr(3) == a.base + (3 << 20)


@pytest.mark.parametrize("backend", ["uring", "aio"])
def test_ssd_slots_round_trip_and_file_is_kept(tmp_path, backend):
    sb = 8192
    src = np.random.default_rng(1).integers(0, 255, 3 * sb + 4096, dtype=np.uint8)
    dst = np.zeros_like(src)
    p0, p1 = (x.ctypes.data + (-x.ctypes.data) % 4096 for x in (src, dst))
    path = str(tmp_path / "shard0.bin")
    with pytest.raises(RuntimeError, match="missing"):
        bk.SsdTier(path, 6 * sb, sb, [(p0, 3 * sb)])
    shards.create_shard(path, 6 * sb)
    with pytest.raises(RuntimeError, match="the run needs"):                  # never resized by a run
        bk.SsdTier(path, 8 * sb, sb, [(p0, 3 * sb)])
    with pytest.raises(ValueError, match="SSD backend"):
        bk.SsdTier(path, 6 * sb, sb, [(p0, 3 * sb)], "threads")
    f = bk.SsdTier(path, 6 * sb, sb, [(p0, 3 * sb), (p1, 3 * sb)], backend)
    assert type(f.io) is bk.SSD_BACKENDS[backend] and backend in f.info
    for fut in f.write([5, 0, 2], [p0 + k * sb for k in range(3)]):
        fut.result()
    for fut in f.read([2, 5, 0], [p1 + k * sb for k in (2, 0, 1)]):
        fut.result()
    a, b = (np.frombuffer((ctypes.c_char * (3 * sb)).from_address(p), dtype=np.uint8) for p in (p0, p1))
    assert np.array_equal(a, b)
    f.close()
    assert os.path.getsize(path) == 6 * sb and not os.path.exists(path + ".tmp")


def test_shard_sets_are_taken_whole_one_run_each(tmp_path):
    """Runs take the first free set (flock on set<k>/lock); a set with a missing or wrong-sized shard fails; create
    makes only what is missing and refuses another size without --force."""
    nb = shards.shard_bytes(0.0001)
    assert nb % 4096 == 0
    with pytest.raises(RuntimeError, match="no SSD shard set"):
        shards.ShardSet(str(tmp_path), 4, nb)
    assert len(shards.create(str(tmp_path), 2, 4, 0.0001)) == 8
    assert shards.create(str(tmp_path), 2, 4, 0.0001) == []                # kept, not rewritten
    a = shards.ShardSet(str(tmp_path), 4, nb)
    b = shards.ShardSet(str(tmp_path), 4, nb)
    assert (a.k, b.k) == (0, 1)
    with pytest.raises(RuntimeError, match="in use by another run"):
        shards.ShardSet(str(tmp_path), 4, nb)
    with pytest.raises(RuntimeError, match="in use by a run"):
        shards.create(str(tmp_path), 1, 4, 0.0002, force=True)
    a.close()
    with pytest.raises(RuntimeError, match="the run needs 4 shards"):
        shards.ShardSet(str(tmp_path), 4, shards.shard_bytes(0.0002))
    with pytest.raises(RuntimeError, match="--force"):
        shards.create(str(tmp_path), 1, 4, 0.0002)
    os.unlink(shards.shard_path(str(tmp_path / "set0"), 2))
    with pytest.raises(RuntimeError, match="2 \\(missing\\)"):
        shards.ShardSet(str(tmp_path), 4, nb)
    b.close()


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
    """prefer_node sets MPOL_PREFERRED: the arena's policy until a checked piece is bound by the touch."""
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
    out = subprocess.run([sys.executable, "-c", "from ssengine.kvpool.backends import oom_first; oom_first(); "
                          "print(open('/proc/self/oom_score_adj').read())"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1000"
    pk = bk.cgroup_peak_gb()
    assert pk is None or pk > 0


def _nodes():
    return sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d[4:].isdigit())


def test_touch_binds_each_piece_to_the_node(monkeypatch):
    """The touch binds every piece to the arena's node (a bound fault reclaims that node's page cache; a preferred one
    leaves for the other node, 8039), and the whole arena lands there."""
    monkeypatch.setattr(bk, "TOUCH_CHUNK", 16 << 20)
    monkeypatch.setattr(bk, "TOUCH_RESERVE_GB", 0.0)
    for node in _nodes():
        a = bk.HostTier(64 << 20, node, True)
        try:
            a.touch()
            assert a.touch_error is None
            lines = [x for x in open("/proc/self/numa_maps") if a.base <= int(x.split()[0], 16) < a.base + a.nbytes]
            assert lines and all(f"bind:{node}" in x for x in lines)
            assert a.node_gb() == {node: a.node_gb()[node]} and a.node_gb()[node] > 0.06
        finally:
            bk.anon_unmap(a.base, a.nbytes)


def test_touch_on_a_short_node_fails_without_placing():
    """A node that cannot cover a piece plus the reserve fails the touch (NodeShort in touch_error) and binds nothing
    more: no spill, no forced allocation."""
    node = _nodes()[0]
    a = bk.HostTier(64 << 20, node, True)
    try:
        old = bk.TOUCH_RESERVE_GB
        bk.TOUCH_RESERVE_GB = bk.node_reclaimable_gb(node) + 1e6
        try:
            a.touch()
        finally:
            bk.TOUCH_RESERVE_GB = old
        assert isinstance(a.touch_error, bk.NodeShort)
        assert sum(a.node_gb().values()) == 0
    finally:
        bk.anon_unmap(a.base, a.nbytes)


def test_node_reclaimable_counts_free_and_clean_file_pages(tmp_path):
    d = tmp_path / "node0"
    d.mkdir()
    (d / "meminfo").write_text("Node 0 MemFree: 1000000 kB\nNode 0 Active(file): 300000 kB\n"
                               "Node 0 Inactive(file): 700000 kB\nNode 0 Dirty: 100000 kB\nNode 0 Writeback: 0 kB\n"
                               "Node 0 Shmem: 5000000 kB\n")
    assert bk.node_reclaimable_gb(0, str(tmp_path)) == pytest.approx(1_900_000 * 1024 / 1e9)
