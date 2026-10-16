"""Request body -> (request fields, int32 token ids), fast for long prompts; the JSON API is unchanged.

A 512k-token prompt is a 3.3 MB body. Decoding it with msgspec (`token_ids` as msgspec.Raw, which still scans every
element) took 4.9 ms and `np.fromstring` on the array 14.2 ms (login host), on the arrival -> first-dispatch path
(job 7404: 27-30 ms at 512k). Here:

1. Find the top-level `"token_ids"` key with `bytes.find` and its array's `[`.
2. Parse the array with a compiled loop (`_parse_ints`, numba, GIL released), cut at commas into `PARSE_THREADS`
   segments parsed in parallel for long arrays.
3. Decode the body with the array replaced by `[]` (a few hundred bytes) with msgspec, which checks everything else.

Anything this path does not take for granted sends the request down the old path (`parse_slow`), which gives the
same answer or the same error: the key not found, found twice, after a backslash; anything but optional whitespace,
non-negative JSON ints below 2**31 and commas inside the brackets; the remainder not a valid request.
"""

from __future__ import annotations

import concurrent.futures
import os
import warnings

import msgspec
import numpy as np

try:
    import numba
except ImportError:          # the old path only
    numba = None

FAST = os.environ.get("PPIMPL_FAST_IDS", "1") != "0"   # 0: always the old path (A/B of the intake phases)
KEY = b'"token_ids"'
PARALLEL_FROM = 1 << 18      # bytes of array from which the parse is split over threads
PARSE_THREADS = 4

if numba is not None:
    @numba.njit(nogil=True, cache=True, boundscheck=False)
    def _parse_ints(buf, i, stop, out):
        """Ints from buf[i]: `int (ws* , ws* int)*` up to the comma at `stop` or the closing `]` (stop < 0).
        Returns (count, index of that comma or `]`), or (-1, index of the error)."""
        n = 0
        L = buf.shape[0]
        while True:
            while i < L and (buf[i] == 32 or buf[i] == 10 or buf[i] == 13 or buf[i] == 9):
                i += 1
            v = 0
            nd = 0
            while i < L:
                c = buf[i]
                if c < 48 or c > 57:
                    break
                v = v * 10 + (c - 48)
                i += 1
                nd += 1
                if nd > 10:
                    return -1, i
            if nd == 0 or v >= 2147483648 or (nd > 1 and buf[i - nd] == 48):   # JSON: no leading zeros
                return -1, i
            out[n] = v
            n += 1
            while i < L and (buf[i] == 32 or buf[i] == 10 or buf[i] == 13 or buf[i] == 9):
                i += 1
            if i >= L:
                return -1, i
            if buf[i] == 44:
                if i == stop:
                    return n, i
                i += 1
            elif buf[i] == 93 and stop < 0:
                return n, i
            else:
                return -1, i


_pool: concurrent.futures.ThreadPoolExecutor | None = None


def warm() -> None:
    """Compile (or load from numba's cache) the parse loop before the first request."""
    if numba is not None:
        _parse_ints(np.frombuffer(b"1,2]", np.uint8), 0, -1, np.empty(2, np.int32))


def _parse_array(raw: bytes, start: int) -> tuple[np.ndarray, int] | None:
    """The int array whose `[` is at raw[start]: (int32 ids, index of its `]`), or None if it is not a plain one."""
    buf = np.frombuffer(raw, np.uint8)
    end_guess = raw.find(b"]", start)
    if end_guess < 0:
        return None
    span = end_guess - start
    if span < 64 and not raw[start + 1:end_guess].strip():
        return np.zeros(0, np.int32), end_guess
    if span < PARALLEL_FROM:
        out = np.empty(span // 2 + 1, np.int32)
        n, end = _parse_ints(buf, start + 1, -1, out)
        return None if n < 0 else (out[:n], end)
    # segments cut at commas: [s_k, c_k) with c_k a comma (the last ends at `]`)
    cuts = []
    for k in range(1, PARSE_THREADS):
        c = raw.find(b",", start + span * k // PARSE_THREADS, end_guess)
        if c < 0 or (cuts and c <= cuts[-1]):
            break
        cuts.append(c)
    starts = [start + 1] + [c + 1 for c in cuts]
    stops = cuts + [-1]
    outs = [np.empty((((c if c >= 0 else end_guess) - s) // 2 + 1), np.int32) for s, c in zip(starts, stops)]
    global _pool
    if _pool is None:
        _pool = concurrent.futures.ThreadPoolExecutor(PARSE_THREADS - 1, thread_name_prefix="parse")
    futs = [_pool.submit(_parse_ints, buf, s, c, o) for s, c, o in zip(starts[1:], stops[1:], outs[1:])]
    res = [_parse_ints(buf, starts[0], stops[0], outs[0])] + [f.result() for f in futs]
    if any(n < 0 for n, _ in res) or any(e != c for (_, e), c in zip(res[:-1], cuts)):
        return None
    return np.concatenate([o[:n] for o, (n, _) in zip(outs, res)]), res[-1][1]


def parse_fast(raw: bytes, rest_decoder: msgspec.json.Decoder):
    """(request with token_ids == [], int32 ids) or None (use the old path)."""
    if numba is None or not FAST:
        return None
    i = raw.find(KEY)
    if i <= 0 or raw[i - 1] == 0x5C:
        return None
    j = i + len(KEY)
    while j < len(raw) and raw[j] in b" \t\r\n":
        j += 1
    if j >= len(raw) or raw[j] != 0x3A:          # ':'
        return None
    j += 1
    while j < len(raw) and raw[j] in b" \t\r\n":
        j += 1
    if j >= len(raw) or raw[j] != 0x5B:          # '['
        return None
    got = _parse_array(raw, j)
    if got is None:
        return None
    ids, end = got
    head, tail = raw[:j], raw[end + 1:]
    if KEY in head[:i] or KEY in tail:
        return None
    try:
        body = rest_decoder.decode(head + b"[]" + tail)
    except (msgspec.ValidationError, msgspec.DecodeError):
        return None
    if body.token_ids:
        return None
    return body, ids


def parse_slow_ids(raw) -> np.ndarray:
    """A JSON array of ints (msgspec.Raw) -> int32 array; ValueError for anything else."""
    text = bytes(raw)
    if text[:1] != b"[":
        raise ValueError("token_ids: not a list")
    inner = text[1:-1]
    if not inner.strip():
        return np.zeros(0, np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")    # numpy warns (and stops early) on what is not an int list
        try:
            ids = np.fromstring(inner, dtype=np.int64, sep=",")
        except (DeprecationWarning, ValueError) as e:
            raise ValueError(f"token_ids: {e}") from None
    if len(ids) != inner.count(b",") + 1 or (len(ids) and (ids.min() < 0 or ids.max() >= 2**31)):
        raise ValueError("token_ids: not a list of int32 ids")
    return ids.astype(np.int32)
