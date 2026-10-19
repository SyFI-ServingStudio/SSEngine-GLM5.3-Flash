"""io_uring for the SSD tier (layer 3, `backends.SsdTier`; lanes/m6 TIERS.md §2.4, §5.1): one ring per member.

liburing (liburing-ffi.so.2, the distribution's liburing2 package) through ctypes; no fallback: a ring that cannot be
set up, or host memory that cannot be registered, fails the member at start-up.

- **Fixed file and buffers.** The file is registered (IOSQE_FIXED_FILE, index 0), and so is every host arena the tier
  reads into or writes from, as fixed buffers of at most 1 GiB (the kernel's limit per buffer): I/O is
  READ_FIXED / WRITE_FIXED, so the kernel pins nothing per request.
- **Pieces.** A request (one slot) is cut into pieces of at most `piece_bytes`, and at a fixed buffer's end.
- **Reads first.** Pieces wait in two queues; at most `read_depth` read pieces and `write_depth` write pieces are in
  the kernel at once, and free read room is filled before write room. A write-back cannot fill the device's queues
  in front of a read.
- **One thread** owns the ring: it moves queued pieces into the submission queue, submits and waits in one call
  (`io_uring_submit_and_wait`, the GIL released), and completes a request's future when its last piece is in. Other
  threads only queue and wake it through an eventfd whose read is always armed in the ring.
"""

from __future__ import annotations

import bisect
import ctypes
import errno
import os
import threading
import time
from collections import deque
from concurrent.futures import Future

IOSQE_FIXED_FILE = 1
MAX_BUFFER = 1 << 30          # kernel limit of one registered buffer
_WAKE = 0                     # user_data of the eventfd read

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL("liburing-ffi.so.2", use_errno=True)
        vp, u32, i32, u64 = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_uint64
        for name, res, args in (
                ("io_uring_queue_init", i32, [u32, vp, u32]),
                ("io_uring_queue_exit", None, [vp]),
                ("io_uring_register_buffers", i32, [vp, vp, u32]),
                ("io_uring_register_files", i32, [vp, ctypes.POINTER(i32), u32]),
                ("io_uring_get_sqe", vp, [vp]),
                ("io_uring_prep_read_fixed", None, [vp, i32, vp, u32, u64, i32]),
                ("io_uring_prep_write_fixed", None, [vp, i32, vp, u32, u64, i32]),
                ("io_uring_prep_read", None, [vp, i32, vp, u32, u64]),
                ("io_uring_sqe_set_flags", None, [vp, u32]),
                ("io_uring_sqe_set_data64", None, [vp, u64]),
                ("io_uring_submit_and_wait", i32, [vp, u32]),
                ("io_uring_peek_batch_cqe", u32, [vp, ctypes.POINTER(vp), u32]),
                ("io_uring_cq_advance", None, [vp, u32])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _lib = L
    return _lib


class _Cqe(ctypes.Structure):
    _fields_ = [("user_data", ctypes.c_uint64), ("res", ctypes.c_int32), ("flags", ctypes.c_uint32)]


def _check(ret: int, what: str):
    if ret < 0:
        raise OSError(-ret, f"{what}: {os.strerror(-ret)}")


def probe() -> int:
    """0 if this process may set up an io_uring ring, else the errno of io_uring_setup (EPERM: a seccomp profile or
    kernel.io_uring_disabled denies it, as Docker's default profile does since 25; ENOSYS: the kernel has none)."""
    L = lib()
    ring = ctypes.create_string_buffer(512)
    ret = L.io_uring_queue_init(1, ring, 0)
    if ret < 0:
        return -ret
    L.io_uring_queue_exit(ring)
    return 0


class UringFile:
    """Reads and writes of byte ranges of one file into and out of registered host memory, on one ring and one
    thread. `buffers`: (address, bytes) of the host memory every request lies in."""

    def __init__(self, fd: int, buffers: list[tuple[int, int]], piece_bytes: int, read_depth: int,
                 write_depth: int, name: str = "tier-uring"):
        if piece_bytes <= 0 or piece_bytes % 4096 or read_depth < 1 or write_depth < 1:
            raise ValueError(f"io_uring piece {piece_bytes} B (a multiple of 4 KiB), depths {read_depth}/{write_depth}")
        L = lib()
        self.piece_bytes, self.depth = piece_bytes, {"r": read_depth, "w": write_depth}
        self.entries = 1 << (read_depth + write_depth + 1).bit_length()
        self._ring = ctypes.create_string_buffer(512)          # struct io_uring (216 B in liburing 2.5)
        _check(L.io_uring_queue_init(self.entries, self._ring, 0), "io_uring_queue_init")
        self._open = True
        try:
            _check(L.io_uring_register_files(self._ring, (ctypes.c_int * 1)(fd), 1), "io_uring_register_files")
            spans = sorted((a + o, min(MAX_BUFFER, n - o)) for a, n in buffers for o in range(0, n, MAX_BUFFER))
            iov = (ctypes.c_uint64 * (2 * len(spans)))(*[x for s in spans for x in s])   # struct iovec[]
            t0 = time.perf_counter()
            _check(L.io_uring_register_buffers(self._ring, iov, len(spans)), "io_uring_register_buffers "
                   f"({len(spans)} buffers, {sum(n for _, n in spans) / 1e9:.1f} GB; ENOMEM: the locked-memory "
                   "limit, `ulimit -l`, is below it: in Docker run with --ulimit memlock=-1)")
        except BaseException:
            L.io_uring_queue_exit(self._ring)
            raise
        self.register_s = time.perf_counter() - t0
        self._starts = [a for a, _ in spans]
        self._ends = [a + n for a, n in spans]
        self.registered_bytes = sum(n for _, n in spans)
        self._q = {"r": deque(), "w": deque()}       # pieces: (request, file offset, address, bytes)
        self._live: dict[int, tuple] = {}            # user_data -> piece in the kernel
        self._in = {"r": 0, "w": 0}
        self._next = 1
        self._stop = False
        self._efd = os.eventfd(0, os.EFD_CLOEXEC)
        self._ebuf = ctypes.create_string_buffer(8)
        self._cqes = (ctypes.c_void_p * 256)()
        self.error: BaseException | None = None
        self.stats = {"r": 0, "w": 0}                 # bytes completed
        self._t = threading.Thread(target=self._loop, name=name, daemon=True)
        self._t.start()

    # --- callers ------------------------------------------------------------------------------------------------

    def submit(self, kind: str, offset: int, addr: int, nbytes: int) -> Future:
        """kind "r" (file -> addr) or "w" (addr -> file); a future done when every piece completed."""
        if self.error is not None:
            raise RuntimeError("io_uring thread failed") from self.error
        req = [Future(), 0, None]                   # future, pieces left, first error
        pieces = []
        o = 0
        while o < nbytes:
            a = addr + o
            i = bisect.bisect_right(self._starts, a) - 1
            if i < 0 or a >= self._ends[i]:
                raise ValueError(f"io_uring: {a:#x} is in no registered buffer")
            n = min(self.piece_bytes, nbytes - o, self._ends[i] - a)
            pieces.append((req, offset + o, a, n, i))
            o += n
        req[1] = len(pieces)
        self._q[kind].extend(pieces)
        os.eventfd_write(self._efd, 1)
        return req[0]

    def close(self):
        if not self._open:
            return
        self._stop = True
        os.eventfd_write(self._efd, 1)
        self._t.join()
        lib().io_uring_queue_exit(self._ring)
        os.close(self._efd)
        self._open = False

    # --- the ring's thread ------------------------------------------------------------------------------------------

    def _loop(self):
        L = lib()
        try:
            self._arm()
            while True:
                for k in ("r", "w"):              # reads take free room first
                    q = self._q[k]
                    while q and self._in[k] < self.depth[k]:
                        self._prep(k, q.popleft())
                if self._stop and not self._live:
                    break
                ret = L.io_uring_submit_and_wait(self._ring, 1)
                if ret == -errno.EINTR:
                    continue
                _check(ret, "io_uring_submit_and_wait")
                n = L.io_uring_peek_batch_cqe(self._ring, self._cqes, len(self._cqes))
                for j in range(n):
                    c = _Cqe.from_address(self._cqes[j])
                    if c.user_data == _WAKE:
                        if c.res < 0 and c.res != -errno.EINTR:
                            _check(c.res, "eventfd read")
                        self._arm()
                    else:
                        self._done(c.user_data, c.res)
                L.io_uring_cq_advance(self._ring, n)
        except BaseException as e:   # noqa: BLE001 - every waiting future fails with it
            self.error = e
            for p in [*self._live.values(), *self._q["r"], *self._q["w"]]:
                if not p[0][0].done():
                    p[0][0].set_exception(e)

    def _sqe(self):
        s = lib().io_uring_get_sqe(self._ring)
        if not s:       # cannot happen: entries > both depths + the eventfd read
            raise RuntimeError("io_uring submission queue full")
        return s

    def _arm(self):
        L = lib()
        s = self._sqe()
        L.io_uring_prep_read(s, self._efd, self._ebuf, 8, 0)
        L.io_uring_sqe_set_data64(s, _WAKE)

    def _prep(self, kind: str, piece):
        L = lib()
        _req, off, a, n, buf = piece
        s = self._sqe()
        (L.io_uring_prep_read_fixed if kind == "r" else L.io_uring_prep_write_fixed)(s, 0, a, n, off, buf)
        L.io_uring_sqe_set_flags(s, IOSQE_FIXED_FILE)
        ud = self._next
        self._next += 1
        L.io_uring_sqe_set_data64(s, ud)
        self._live[ud] = (kind, piece)
        self._in[kind] += 1

    def _done(self, ud: int, res: int):
        kind, (req, off, _a, n, _b) = self._live.pop(ud)
        self._in[kind] -= 1
        if res != n and req[2] is None:
            req[2] = OSError(-res if res < 0 else errno.EIO, f"io_uring {'read' if kind == 'r' else 'write'} at "
                             f"{off}: " + (os.strerror(-res) if res < 0 else f"{res} of {n} B"))
        self.stats[kind] += max(res, 0)
        req[1] -= 1
        if req[1] == 0:
            if req[2] is None:
                req[0].set_result(None)
            else:
                req[0].set_exception(req[2])
