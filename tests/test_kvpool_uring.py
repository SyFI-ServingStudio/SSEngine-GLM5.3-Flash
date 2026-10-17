"""The SSD tier's io_uring (ppimpl/kvpool/uring.py) on CPU: requests cut into pieces at the piece size and at a fixed
buffer's end, byte-exact round trips, reads ahead of queued writes, errors on the request's future, and memory outside
the registered buffers refused."""

import ctypes
import os

import numpy as np
import pytest

from ppimpl.kvpool import uring


def aligned(n: int, seed: int | None = None):
    raw = np.zeros(n + 4096, np.uint8) if seed is None else np.random.default_rng(seed).integers(0, 255, n + 4096,
                                                                                                    dtype=np.uint8)
    p = raw.ctypes.data + (-raw.ctypes.data) % 4096
    return raw, p, np.frombuffer((ctypes.c_char * n).from_address(p), dtype=np.uint8)


@pytest.fixture
def fd(tmp_path):
    fd = os.open(tmp_path / "f.bin", os.O_RDWR | os.O_CREAT, 0o600)
    os.posix_fallocate(fd, 0, 1 << 20)
    yield fd
    os.close(fd)


def test_round_trip_in_pieces_across_buffers(fd):
    n = 96 << 10
    _a, pa, va = aligned(n, 1)
    _b, pb, vb = aligned(n)
    # three buffers: the write source in two halves (a request crosses their boundary), the read target whole
    u = uring.UringFile(fd, [(pa, n // 2), (pa + n // 2, n // 2), (pb, n)], 16 << 10, 2, 1)
    try:
        u.submit("w", 64 << 10, pa, n).result(timeout=10)
        u.submit("r", 64 << 10, pb, n).result(timeout=10)
        assert np.array_equal(va, vb)
        assert u.stats == {"r": n, "w": n}
    finally:
        u.close()


def test_reads_go_before_queued_writes(fd):
    """With the write depth full, a read queued after many writes completes before the last of them."""
    n = 32 << 10
    _a, pa, _va = aligned(64 * n, 2)
    u = uring.UringFile(fd, [(pa, 64 * n)], n, 4, 1)
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


def test_error_reaches_the_future_and_unregistered_memory_is_refused(tmp_path):
    n = 16 << 10
    _a, pa, _va = aligned(n)
    fd = os.open(tmp_path / "ro.bin", os.O_RDONLY | os.O_CREAT, 0o600)
    u = uring.UringFile(fd, [(pa, n)], n, 1, 1)
    try:
        with pytest.raises(OSError):
            u.submit("w", 0, pa, n).result(timeout=10)       # write to a read-only fd: EBADF on the future
        with pytest.raises(ValueError):
            u.submit("r", 0, pa + n, 4096)
    finally:
        u.close()
        os.close(fd)


def test_bad_settings_fail_loudly(fd):
    _a, pa, _va = aligned(8192)
    with pytest.raises(ValueError):
        uring.UringFile(fd, [(pa, 8192)], 1000, 1, 1)
