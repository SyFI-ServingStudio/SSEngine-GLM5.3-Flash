"""The SSD tier's file I/O backends on CPU, io_uring (ppimpl/kvpool/uring.py) and Linux AIO (aio.py), the same tests
for both: requests cut into pieces at the piece size and at a buffer's end, byte-exact round trips, reads ahead of
queued writes, errors on the request's future, and memory outside the buffers refused."""

import ctypes
import os

import numpy as np
import pytest

from ppimpl.kvpool import aio, uring

BACKENDS = {"uring": uring.UringFile, "aio": aio.AioFile}


def aligned(n: int, seed: int | None = None):
    raw = np.zeros(n + 4096, np.uint8) if seed is None else np.random.default_rng(seed).integers(0, 255, n + 4096,
                                                                                                    dtype=np.uint8)
    p = raw.ctypes.data + (-raw.ctypes.data) % 4096
    return raw, p, np.frombuffer((ctypes.c_char * n).from_address(p), dtype=np.uint8)


@pytest.fixture(params=sorted(BACKENDS))
def File(request):
    return BACKENDS[request.param]


@pytest.fixture
def fd(tmp_path):
    fd = os.open(tmp_path / "f.bin", os.O_RDWR | os.O_CREAT, 0o600)
    os.posix_fallocate(fd, 0, 1 << 20)
    yield fd
    os.close(fd)


def test_round_trip_in_pieces_across_buffers(fd, File):
    n = 96 << 10
    _a, pa, va = aligned(n, 1)
    _b, pb, vb = aligned(n)
    # three buffers: the write source in two halves (a request crosses their boundary), the read target whole
    u = File(fd, [(pa, n // 2), (pa + n // 2, n // 2), (pb, n)], 16 << 10, 2, 1)
    try:
        u.submit("w", 64 << 10, pa, n).result(timeout=10)
        u.submit("r", 64 << 10, pb, n).result(timeout=10)
        assert np.array_equal(va, vb)
        assert u.stats == {"r": n, "w": n}
    finally:
        u.close()


def test_reads_go_before_queued_writes(fd, File):
    """With the write depth full, a read queued after many writes completes before the last of them."""
    n = 32 << 10
    _a, pa, _va = aligned(64 * n, 2)
    u = File(fd, [(pa, 64 * n)], n, 4, 1)
    order = []
    try:
        ws = [u.submit("w", i * n, pa + i * n, n) for i in range(16)]
        for i, w in enumerate(ws):
            w.add_done_callback(lambda _f, i=i: order.append(("w", i)))
        r = u.submit("r", 0, pa + 32 * n, n)
        r.add_done_callback(lambda _f: order.append(("r", 0)))
        for f in ws + [r]:
            f.result(timeout=10)
        assert order.index(("r", 0)) < order.index(("w", 15))
    finally:
        u.close()


def test_error_reaches_the_future_and_unregistered_memory_is_refused(tmp_path, File):
    n = 16 << 10
    _a, pa, _va = aligned(n)
    fd = os.open(tmp_path / "ro.bin", os.O_RDONLY | os.O_CREAT, 0o600)
    u = File(fd, [(pa, n)], n, 1, 1)
    try:
        with pytest.raises(OSError):
            u.submit("w", 0, pa, n).result(timeout=10)       # write to a read-only fd: EBADF on the future
        with pytest.raises(ValueError):
            u.submit("r", 0, pa + n, 4096)
    finally:
        u.close()
        os.close(fd)


def test_bad_settings_fail_loudly(fd, File):
    _a, pa, _va = aligned(8192)
    with pytest.raises(ValueError):
        File(fd, [(pa, 8192)], 1000, 1, 1)


def test_aio_probe_and_unaligned_o_direct_fails_on_the_future(tmp_path):
    """io_setup is allowed here; an O_DIRECT piece at an unaligned file offset fails its request, not the thread."""
    assert aio.probe() == 0
    _a, pa, _va = aligned(8192)
    from ppimpl.kvpool.backends import open_direct
    p = tmp_path / "d.bin"
    p.write_bytes(bytes(1 << 16))
    fd, direct = open_direct(str(p))
    u = aio.AioFile(fd, [(pa, 8192)], 4096, 1, 1)
    try:
        if direct:
            with pytest.raises(OSError):
                u.submit("r", 100, pa, 4096).result(timeout=10)   # EINVAL: not sector-aligned
        u.submit("r", 0, pa, 8192).result(timeout=10)
        assert u.error is None and u.stats["r"] == 8192
    finally:
        u.close()
        os.close(fd)
