"""Shared-memory hand-off of long prompts from an intake process to the controller.

A 512k-token prompt is 2.1 MB of int32 ids. Sent as a zmq frame over tcp it took 2-4 ms from the intake's send to
the controller's receive (CPU fake server, 512k bring-up sessions); a small frame takes 0.4-0.8 ms. Instead, each
intake owns one file in /dev/shm (`Slab`), parses long prompts' ids into it and sends only (offset, count); the
controller maps the same file (`SlabReader`) and copies the ids out on receipt. The intake frees the range when the
request is answered (the controller copied it long before). Prompts below `SLAB_FROM` bytes, and any that do not
fit, go as a zmq frame as before.

The file is created before the intake's ready message, which carries its name; the controller maps it when it
takes that message, and the intake unlinks the name once every intake is ready (the mappings stay valid), so a
killed server leaves nothing in /dev/shm after start-up.
"""

from __future__ import annotations

import mmap
import os
import uuid

import numpy as np

SLAB_BYTES = 32 << 20      # 8M ids per intake in flight: 7 prompts of 1M tokens
SLAB_FROM = 1 << 18        # ids bytes (64k tokens) from which a prompt goes through the slab
ALIGN = 4096


class Slab:
    """Intake side: a first-fit allocator over one shared file."""

    def __init__(self, size: int = SLAB_BYTES):
        self.name = f"ppimpl-intake-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = f"/dev/shm/{self.name}"
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            self.mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.buf = np.frombuffer(self.mm, np.uint8)
        self.buf[:] = 0            # fault every page in now, not on a request's path
        self.size = size
        self.used: dict[int, int] = {}    # offset -> bytes

    def unlink(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def alloc(self, nbytes: int) -> int | None:
        """Offset of a free range of nbytes, or None."""
        nbytes = -(-nbytes // ALIGN) * ALIGN
        at = 0
        for off in sorted(self.used):
            if off - at >= nbytes:
                break
            at = off + self.used[off]
        if at + nbytes > self.size:
            return None
        self.used[at] = nbytes
        return at

    def free(self, offset: int) -> None:
        self.used.pop(offset, None)

    def ids(self, offset: int, n: int) -> np.ndarray:
        return self.buf[offset:offset + 4 * n].view(np.int32)


class SlabReader:
    """Controller side: the intakes' files, mapped read-only by name."""

    def __init__(self):
        self.maps: dict[str, mmap.mmap] = {}

    def attach(self, name: str) -> None:
        fd = os.open(f"/dev/shm/{name}", os.O_RDONLY)
        try:
            # populated: no page faults on the first copy out (2 MB at 512k tokens: ~1 ms of faults)
            self.maps[name] = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE, prot=mmap.PROT_READ)
        finally:
            os.close(fd)

    def copy(self, name: str, offset: int, n: int) -> np.ndarray:
        return np.frombuffer(self.maps[name], np.int32, n, offset).copy()
