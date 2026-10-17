"""The SSD tier's shard files, kept across runs (user, 2026-10-17; lanes/m6 TIERS.md §5).

Balanced storage (`placement.py`) gives member q its quarter (1/n) of every slot, so an SSD file holds a shard, not
a GPU's KV cache: `<PPIMPL_SSD_DIR>/set<k>/shard<q>.bin`, q = 0..n-1. A run takes a whole set:

- **Exclusive.** The leader takes the first set whose `set<k>/lock` it can `flock` (k = 0, 1, ...) and holds the lock
  for the run's lifetime, so concurrent runs (GPUs 0-3 and 4-7) get different sets. Member q opens `shard<q>.bin` of
  that set (`TierConfig.ssd_dir` is the set's directory in the workers' config).
- **Created once, explicitly.** `python -m ppimpl.kvpool.shards create --sets K --shards N --gb G` writes every shard
  in full (zeros; `.tmp` renamed when complete), so no O_DIRECT write ever lands on an unwritten ext4 extent. A run
  never creates, resizes or deletes a shard: a set with a missing shard or one of another size fails the start-up.
- **Never trusted.** A shard's bytes are never read as data: the slot index starts empty every run, so a run reads
  only slots it wrote itself.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import mmap
import os
import threading
import time

from ppimpl.kvpool.backends import open_direct


def shard_bytes(gb: float) -> int:
    """A shard of `gb` GB per member: whole 4 KiB pages (a tier of `num_slots(gb, slot_bytes)` slots fits)."""
    return int(gb * 1e9) // 4096 * 4096


def shard_path(set_dir: str, q: int) -> str:
    return os.path.join(set_dir, f"shard{q}.bin")


def check_set(set_dir: str, n: int, nbytes: int):
    bad = [(q, os.path.getsize(f) if os.path.exists(f) else None) for q in range(n)
           if not os.path.exists(f := shard_path(set_dir, q)) or os.path.getsize(f) != nbytes]
    if bad:
        raise RuntimeError(f"SSD shard set {set_dir}: shards " + ", ".join(
            f"{q} ({'missing' if b is None else f'{b} B'})" for q, b in bad) + f", the run needs {n} shards of "
            f"{nbytes} B; create them with `python -m ppimpl.kvpool.shards create` (TIERS.md §5)")


class ShardSet:
    """The leader's hold on one set: the first set under `base` whose lock is free, with all n shards of `nbytes`."""

    def __init__(self, base: str, n: int, nbytes: int):
        sets = sorted((int(d[3:]), d) for d in os.listdir(base) if d.startswith("set") and d[3:].isdigit()) \
            if os.path.isdir(base) else []
        if not sets:
            raise RuntimeError(f"no SSD shard set in {base}: create one with `python -m ppimpl.kvpool.shards create`")
        busy = []
        for k, d in sets:
            fd = os.open(os.path.join(base, d, "lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                busy.append(k)
                continue
            try:
                check_set(os.path.join(base, d), n, nbytes)
            except BaseException:
                os.close(fd)
                raise
            self.k, self.dir, self._fd = k, os.path.join(base, d), fd
            return
        raise RuntimeError(f"every SSD shard set in {base} is in use by another run (sets {busy} locked); create "
                           "another with `python -m ppimpl.kvpool.shards create --sets K`")

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def create_shard(path: str, nbytes: int) -> float:
    """Write one shard in full (zeros, 64 MiB at a time) and rename it into place; seconds taken."""
    t0 = time.perf_counter()
    tmp = path + ".tmp"
    fd, _ = open_direct(tmp, os.O_CREAT | os.O_TRUNC)
    chunk = 64 << 20
    buf = mmap.mmap(-1, chunk)                                 # page-aligned zeros
    try:
        os.posix_fallocate(fd, 0, nbytes)
        for off in range(0, nbytes, chunk):
            n = min(chunk, nbytes - off)
            if os.pwritev(fd, [memoryview(buf)[:n]], off) != n:
                raise OSError(errno.EIO, f"short write creating {tmp} at {off}")
        os.fsync(fd)
    finally:
        os.close(fd)
        buf.close()
    os.rename(tmp, path)
    return time.perf_counter() - t0


def create(base: str, sets: int, n: int, gb: float, force: bool = False) -> list[str]:
    """Sets 0..sets-1 with n shards of `gb` GB each: missing (or, with `force`, every) shard written in full, one
    thread per shard, under the set's lock (never a set a run holds). Existing shards of another size fail."""
    nbytes, made, errs = shard_bytes(gb), [], []
    for k in range(sets):
        d = os.path.join(base, f"set{k}")
        os.makedirs(d, exist_ok=True)
        fd = os.open(os.path.join(d, "lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{d} is in use by a run") from None
            todo = []
            for q in range(n):
                f = shard_path(d, q)
                if force or not os.path.exists(f):
                    todo.append(f)
                elif os.path.getsize(f) != nbytes:
                    raise RuntimeError(f"{f} is {os.path.getsize(f)} B, not {nbytes}: pass --force to rewrite it")

            def run(f):
                try:
                    print(f"{f}: {nbytes / 1e9:.1f} GB in {create_shard(f, nbytes):.0f} s", flush=True)
                except BaseException as e:   # noqa: BLE001 - re-raised below
                    errs.append(e)
            ts = [threading.Thread(target=run, args=(f,)) for f in todo]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            if errs:
                raise errs[0]
            made += todo
        finally:
            os.close(fd)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["create", "list"])
    ap.add_argument("--dir", default=os.environ.get("PPIMPL_SSD_DIR"), help="default $PPIMPL_SSD_DIR")
    ap.add_argument("--sets", type=int, default=1, help="create: sets 0..K-1")
    ap.add_argument("--shards", type=int, default=4, help="create: shards per set (the PP degree)")
    ap.add_argument("--gb", type=float, default=128.0, help="create: GB per shard (TierConfig.ssd_gb)")
    ap.add_argument("--force", action="store_true", help="create: rewrite existing shards (another size)")
    a = ap.parse_args()
    if not a.dir:
        raise SystemExit("--dir or PPIMPL_SSD_DIR")
    if a.cmd == "create":
        create(a.dir, a.sets, a.shards, a.gb, a.force)
    for d in sorted(os.listdir(a.dir)) if os.path.isdir(a.dir) else []:
        p = os.path.join(a.dir, d)
        fd = os.open(os.path.join(p, "lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = "free"
        except BlockingIOError:
            state = "in use"
        os.close(fd)
        shards = sorted(f for f in os.listdir(p) if f.startswith("shard"))
        print(f"{p} ({state}): " + ", ".join(f"{f} {os.path.getsize(os.path.join(p, f)) / 1e9:.1f} GB" for f in shards))


if __name__ == "__main__":
    main()
