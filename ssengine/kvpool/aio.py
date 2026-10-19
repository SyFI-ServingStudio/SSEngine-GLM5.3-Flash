"""Linux native AIO for the SSD tier (layer 3, `backends.SsdTier`, `tiers.ssd_backend: aio`; TIERS.md §2.5): one
context per member, the same requests, pieces and queues as `uring.UringFile`.

The kernel's io_setup / io_submit / io_getevents / io_destroy, called as raw syscalls through ctypes (libc's
`syscall`; no libaio): Docker's default seccomp profile allows them, while it denies io_uring_* (Docker >= 25). No
fallback: a context that cannot be set up fails the member at start-up, and the backend is only ever the one the
config names.

- **No registration.** AIO has no fixed files or buffers: each piece is an IOCB_CMD_PREAD / PWRITE of the fd at a host
  address. `buffers` only bound the addresses a request may use and where pieces are cut, as the ring's fixed buffers
  do.
- **Pieces.** A request (one slot) is cut into pieces of at most `piece_bytes`, and at a buffer span's end (spans of
  at most 1 GiB, as the ring's): the same pieces as io_uring.
- **Reads first.** Pieces wait in two queues; at most `read_depth` read pieces and `write_depth` write pieces are in
  the kernel at once, and free read room is filled before write room.
- **One thread** owns the context and its bookkeeping. Every iocb carries IOCB_FLAG_RESFD, so its completion adds to
  an eventfd that callers also write to when they queue: the thread blocks only in that eventfd's read (GIL released),
  then reaps every completed event without waiting (io_getevents, zero timeout) and hands what the free room allows to
  the submit threads. One wait covers both new work and completions, as the ring's armed eventfd read does.
- **Submit threads.** io_submit pins a piece's user pages and builds its bios before it returns (fixed buffers spare
  io_uring the pinning): ~0.4 ms of CPU per 12.9 MB slot, about twice the ring's. One thread doing it serially set
  the latency of a light-load read (research/bench/ssd_io.py, 4 GB/s reads + 1 GB/s writes per process: 31-60 slots p50
  21.5 ms vs 12.3 ms on the ring); `SUBMITTERS` threads, each submitting one iocb at a time, bring it to the ring's
  (9.5 ms). A refused iocb goes back to the owner, which fails its request.
"""

from __future__ import annotations

import bisect
import ctypes
import errno
import os
import platform
import queue
import threading
from collections import deque
from concurrent.futures import Future

from ssengine.kvpool.uring import MAX_BUFFER

# syscall numbers: x86_64, and the generic table (aarch64)
_NR = {"x86_64": {"io_setup": 206, "io_destroy": 207, "io_getevents": 208, "io_submit": 209},
       "aarch64": {"io_setup": 0, "io_destroy": 1, "io_submit": 2, "io_getevents": 4}}
IOCB_CMD_PREAD, IOCB_CMD_PWRITE = 0, 1
IOCB_FLAG_RESFD = 1
SUBMITTERS = 4                # io_submit threads per context (see above)

_libc = None


def libc():
    global _libc
    if _libc is None:
        if platform.machine() not in _NR:
            raise OSError(errno.ENOSYS, f"Linux AIO: no syscall numbers for {platform.machine()}")
        L = ctypes.CDLL(None, use_errno=True)
        L.syscall.restype = ctypes.c_long
        _libc = L
    return _libc


def _sys(name: str, *args) -> int:
    """The syscall's result, -errno on failure (as liburing returns)."""
    L = libc()
    ret = L.syscall(ctypes.c_long(_NR[platform.machine()][name]), *args)
    return -ctypes.get_errno() if ret < 0 else ret


class _Iocb(ctypes.Structure):      # struct iocb (linux/aio_abi.h, little endian), 64 B
    _fields_ = [("aio_data", ctypes.c_uint64), ("aio_key", ctypes.c_uint32), ("aio_rw_flags", ctypes.c_int32),
                ("aio_lio_opcode", ctypes.c_uint16), ("aio_reqprio", ctypes.c_int16), ("aio_fildes", ctypes.c_uint32),
                ("aio_buf", ctypes.c_uint64), ("aio_nbytes", ctypes.c_uint64), ("aio_offset", ctypes.c_int64),
                ("aio_reserved2", ctypes.c_uint64), ("aio_flags", ctypes.c_uint32), ("aio_resfd", ctypes.c_uint32)]


class _Event(ctypes.Structure):     # struct io_event, 32 B
    _fields_ = [("data", ctypes.c_uint64), ("obj", ctypes.c_uint64), ("res", ctypes.c_int64), ("res2", ctypes.c_int64)]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _check(ret: int, what: str):
    if ret < 0:
        raise OSError(-ret, f"{what}: {os.strerror(-ret)}")


def probe() -> int:
    """0 if this process may set up an AIO context, else the errno of io_setup (EPERM: a seccomp profile denies it;
    EAGAIN: fs.aio-max-nr is used up; ENOSYS: the kernel has no AIO)."""
    ctx = ctypes.c_ulong(0)
    ret = _sys("io_setup", ctypes.c_uint(1), ctypes.byref(ctx))
    if ret < 0:
        return -ret
    _sys("io_destroy", ctx)
    return 0


class AioFile:
    """Reads and writes of byte ranges of one file into and out of host memory, on one AIO context, its owner thread
    and `submitters` submit threads (the interface of `uring.UringFile`). `buffers`: (address, bytes) of the host
    memory every request lies in."""

    def __init__(self, fd: int, buffers: list[tuple[int, int]], piece_bytes: int, read_depth: int,
                 write_depth: int, name: str = "tier-aio", submitters: int = SUBMITTERS):
        if piece_bytes <= 0 or piece_bytes % 4096 or read_depth < 1 or write_depth < 1 or submitters < 1:
            raise ValueError(f"aio piece {piece_bytes} B (a multiple of 4 KiB), depths {read_depth}/{write_depth}, "
                             f"{submitters} submit threads")
        self.fd, self.piece_bytes, self.depth = fd, piece_bytes, {"r": read_depth, "w": write_depth}
        self.entries = read_depth + write_depth
        self._ctx = ctypes.c_ulong(0)
        _check(_sys("io_setup", ctypes.c_uint(self.entries), ctypes.byref(self._ctx)), "io_setup")
        self._open = True
        spans = sorted((a + o, min(MAX_BUFFER, n - o)) for a, n in buffers for o in range(0, n, MAX_BUFFER))
        self.register_s = 0.0                        # nothing to register (as the ring's fields, for SsdTier.info)
        self._starts = [a for a, _ in spans]
        self._ends = [a + n for a, n in spans]
        self.registered_bytes = sum(n for _, n in spans)
        self._q = {"r": deque(), "w": deque()}       # pieces: (request, file offset, address, bytes, span)
        self._live: dict[int, tuple] = {}            # aio_data -> (kind, piece, iocb) in the kernel
        self._in = {"r": 0, "w": 0}
        self._next = 1
        self._stop = False
        self._efd = os.eventfd(0, os.EFD_CLOEXEC)    # completions (IOCB_FLAG_RESFD) and callers' wake-ups
        self._events = (_Event * self.entries)()
        self._zero = _Timespec(0, 0)
        self.error: BaseException | None = None
        self.stats = {"r": 0, "w": 0}                 # bytes completed
        self._subq: queue.SimpleQueue = queue.SimpleQueue()   # aio_data of iocbs to submit
        self._failed: deque = deque()               # (aio_data, -errno) of iocbs io_submit refused
        self._subs = [threading.Thread(target=self._submitter, name=f"{name}-sub{i}", daemon=True)
                      for i in range(submitters)]
        for t in self._subs:
            t.start()
        self._t = threading.Thread(target=self._loop, name=name, daemon=True)
        self._t.start()

    # --- callers ------------------------------------------------------------------------------------------------

    def submit(self, kind: str, offset: int, addr: int, nbytes: int) -> Future:
        """kind "r" (file -> addr) or "w" (addr -> file); a future done when every piece completed."""
        if self.error is not None:
            raise RuntimeError("aio thread failed") from self.error
        req = [Future(), 0, None]                   # future, pieces left, first error
        pieces = []
        o = 0
        while o < nbytes:
            a = addr + o
            i = bisect.bisect_right(self._starts, a) - 1
            if i < 0 or a >= self._ends[i]:
                raise ValueError(f"aio: {a:#x} is in no tier buffer")
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
        for _ in self._subs:
            self._subq.put(None)
        for t in self._subs:
            t.join()
        _sys("io_destroy", self._ctx)
        os.close(self._efd)
        self._open = False

    # --- the context's thread -----------------------------------------------------------------------------------

    def _loop(self):
        try:
            while True:
                while self._failed:
                    self._done(*self._failed.popleft())
                for k in ("r", "w"):              # reads take free room first
                    q = self._q[k]
                    while q and self._in[k] < self.depth[k]:
                        self._subq.put(self._prep(k, q.popleft()))
                if self._stop and not self._live:
                    break
                os.eventfd_read(self._efd)        # blocks (GIL released) until a completion or a caller's wake-up
                if self._live:
                    n = _sys("io_getevents", self._ctx, ctypes.c_long(0), ctypes.c_long(self.entries),
                             self._events, ctypes.byref(self._zero))
                    if n == -errno.EINTR:
                        continue
                    _check(n, "io_getevents")
                    for j in range(n):
                        e = self._events[j]
                        self._done(e.data, e.res)
        except BaseException as e:   # noqa: BLE001 - every waiting future fails with it
            self.error = e
            for p in [*self._live.values(), *self._q["r"], *self._q["w"]]:
                p = p[1] if isinstance(p[0], str) else p
                if not p[0][0].done():
                    p[0][0].set_exception(e)

    def _prep(self, kind: str, piece) -> int:
        _req, off, a, n, _span = piece
        cb = _Iocb(aio_lio_opcode=IOCB_CMD_PREAD if kind == "r" else IOCB_CMD_PWRITE, aio_fildes=self.fd,
                   aio_buf=a, aio_nbytes=n, aio_offset=off, aio_flags=IOCB_FLAG_RESFD, aio_resfd=self._efd)
        ud = self._next
        self._next += 1
        cb.aio_data = ud
        self._live[ud] = (kind, piece, cb)          # the iocb lives until its event is reaped
        self._in[kind] += 1
        return ud

    def _submitter(self):
        """io_submit one iocb at a time (`_subq`; None stops); a refused one goes to `_failed` and wakes the owner."""
        while (ud := self._subq.get()) is not None:
            ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(self._live[ud][2]))
            while (ret := _sys("io_submit", self._ctx, ctypes.c_long(1), ptrs)) == -errno.EINTR:
                pass
            if ret < 0:
                self._failed.append((ud, ret))
                os.eventfd_write(self._efd, 1)

    def _done(self, ud: int, res: int):
        kind, (req, off, _a, n, _b), _cb = self._live.pop(ud)
        self._in[kind] -= 1
        if res != n and req[2] is None:
            req[2] = OSError(-res if res < 0 else errno.EIO, f"aio {'read' if kind == 'r' else 'write'} at "
                             f"{off}: " + (os.strerror(-res) if res < 0 else f"{res} of {n} B"))
        self.stats[kind] += max(res, 0)
        req[1] -= 1
        if req[1] == 0:
            if req[2] is None:
                req[0].set_result(None)
            else:
                req[0].set_exception(req[2])
