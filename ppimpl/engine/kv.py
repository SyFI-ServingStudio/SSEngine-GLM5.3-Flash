"""The block pool and the exact prefix cache (ppimpl/runtime/README.md "Block-id space"; 5_prefix_cache.md §2).

One block-id space for all stages. A request holds ceil(L / 8576) KV blocks (its token positions in order) and 6
state blocks (KDA), reserved whole when it is admitted; no preemption.

Prefix cache (phase B, `keep_sessions`): when a request finishes, its blocks stay as its session's **kept** context:
the KV of its prompt and the KDA state after the prompt's last token. The session's next round takes them over
(ownership moves, nothing is shared) and computes only the rest of its prompt. The hit is exact: the whole kept
context, not rounded to a block. A kept context resumes only at its end, where its KDA state is, so a prompt that does
not start with the whole kept context gets no hit (`resumable_tokens`). Kept sessions use the same pool as running
requests and yield to them: when a request needs room, kept sessions are evicted least recently used first, a whole
session at a time.

Accounting. `blocks` (the real pool): a request costs blocks_for(prompt), a kept session the same. `tokens` (replay
only, to reproduce the targets): the simulator's token ledger, a running request costs prompt + 1 + 51,456 tokens and
a kept session prompt + 51,456, against the pool's token count; block ids then come from an unbounded virtual pool.
"""

import hashlib
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass

from ppimpl.config import BLOCK_TOKENS, NULL_BLOCK, STATE_BLOCKS

STATE_TOKENS = STATE_BLOCKS * BLOCK_TOKENS     # 51,456: what the 6 state blocks cost in tokens


def kv_blocks_for(tokens: int) -> int:
    return -(-tokens // BLOCK_TOKENS)


def blocks_for(tokens: int) -> int:
    """A request's footprint in blocks: its DSA KV blocks plus the KDA state blocks."""
    return kv_blocks_for(tokens) + STATE_BLOCKS


def pool_size(ready_blocks: list[int], pool_tokens: int | None = None) -> int:
    """Usable blocks: the smallest stage's report minus the null block, optionally capped by a token count."""
    n = min(ready_blocks) - 1
    if pool_tokens is not None:
        n = min(n, pool_tokens // BLOCK_TOKENS)
    if n < 1:
        raise ValueError(f"KV pool has no usable block (ready={ready_blocks}, pool_tokens={pool_tokens})")
    return n


def prefix_digest(token_ids, n: int) -> bytes:
    """Content key of a prompt's first n tokens: a kept context matches a new prompt when the digests agree."""
    return hashlib.blake2b(array("i", token_ids[:n]).tobytes(), digest_size=16).digest()


def resumable_tokens(tokens: int, digest: bytes | None, req, partial: bool = False) -> int:
    """How much of a kept context of `tokens` tokens a request can resume from: all of it if the prompt starts with
    it and goes beyond it, else 0 (its KDA state is at its end only).

    The request says what it shares with its session's previous context either as `prefix_tokens` (the trace's
    declared prefix, replay) or by its `token_ids` (server). `partial` is the simulator's rule, a hit of
    min(kept, declared) even inside a kept context; it is not realizable with one KDA state per context and exists
    only to reproduce the targets in replay.
    """
    if tokens <= 0:
        return 0
    if req.token_ids is not None and digest is not None:
        ok = tokens < req.num_tokens and prefix_digest(req.token_ids, tokens) == digest
        return tokens if ok else 0
    shared = req.prefix_tokens
    if partial:
        hit = min(tokens, shared)
        return hit - 1 if hit >= req.num_tokens else hit
    return tokens if tokens <= shared and tokens < req.num_tokens else 0


class BlockPool:
    """Free list over block ids 1..num_blocks (0 is the null block). `virtual`: no limit (token accounting)."""

    def __init__(self, num_blocks: int, virtual: bool = False):
        self.num_blocks = num_blocks
        self.virtual = virtual
        self._free = deque(range(1, num_blocks + 1))
        self._next = num_blocks + 1

    @property
    def num_free(self) -> int:
        return len(self._free)

    def take(self, n: int) -> list[int]:
        if self.virtual and n > len(self._free):
            self._free.extend(range(self._next, self._next + n - len(self._free)))
            self._next += n
        assert n <= len(self._free)
        return [self._free.popleft() for _ in range(n)]

    def give(self, blocks: list[int]):
        assert NULL_BLOCK not in blocks
        self._free.extend(blocks)


@dataclass
class Allocation:
    kv_blocks: list[int]
    state_blocks: list[int]

    @property
    def num_blocks(self) -> int:
        return len(self.kv_blocks) + len(self.state_blocks)


@dataclass(eq=False)
class Kept:
    """A kept session: its last prompt's KV and the KDA state at the prompt's end."""
    session: object
    tokens: int
    alloc: Allocation
    digest: bytes | None


class KvManager:
    """Per-request allocation over the pool, plus the kept sessions. Pure state; the scheduler drives it."""

    def __init__(self, num_blocks: int, keep_sessions: bool = False, accounting: str = "blocks",
                 partial_hits: bool = False):
        assert accounting in ("blocks", "tokens")
        self.accounting = accounting
        self.keep_sessions = keep_sessions
        self.partial_hits = partial_hits
        self.pool = BlockPool(num_blocks, virtual=accounting == "tokens")
        self.capacity = num_blocks if accounting == "blocks" else num_blocks * BLOCK_TOKENS
        self.hard = 0                                  # charge of running and held requests
        self.kept_charge = 0
        self.kept: OrderedDict[object, Kept] = OrderedDict()   # least recently used first
        self.held: dict[str, Allocation] = {}          # running and held requests
        self._charge: dict[str, int] = {}
        self.stats = dict(admitted=0, hits=0, hit_tokens=0, evicted_sessions=0, evicted_tokens=0, dropped_stale=0,
                          peak_hard=0)

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    # --- charges ------------------------------------------------------------------------------------------------

    def footprint(self, prompt_tokens: int) -> int:
        """What a running request of `prompt_tokens` costs, in the pool's unit."""
        if self.accounting == "blocks":
            return blocks_for(prompt_tokens)
        return prompt_tokens + 1 + STATE_TOKENS

    def kept_cost(self, tokens: int) -> int:
        return blocks_for(tokens) if self.accounting == "blocks" else tokens + STATE_TOKENS

    def fits_ever(self, prompt_tokens: int) -> bool:
        return self.footprint(prompt_tokens) <= self.capacity

    def fits(self, req) -> bool:
        """Whether the request's whole footprint fits now, kept sessions evicted as needed."""
        return self.hard + self.footprint(req.num_tokens) <= self.capacity

    @property
    def used_tokens(self) -> int:
        """Pool tokens in use by running/held requests and kept sessions (blocks × 8,576 in block accounting)."""
        u = self.hard + self.kept_charge
        return u * BLOCK_TOKENS if self.accounting == "blocks" else u

    # --- lookups ------------------------------------------------------------------------------------------------

    def peek_hit(self, req) -> int:
        """The HBM hit the request would get if it were admitted now."""
        if req.session is None or not self.keep_sessions:
            return 0
        k = self.kept.get(req.session)
        return 0 if k is None else resumable_tokens(k.tokens, k.digest, req, self.partial_hits)

    # --- allocation ---------------------------------------------------------------------------------------------

    def allocate(self, req) -> int:
        """Reserve the request's whole footprint (the caller checked `fits`); take over its session's kept context
        when it resumes from it. Returns the hit: tokens already in the allocated KV."""
        assert self.fits(req), req.req_id
        k = self.kept.pop(req.session, None) if req.session is not None else None
        hit = 0
        if k is not None:
            self.kept_charge -= self.kept_cost(k.tokens)
            hit = resumable_tokens(k.tokens, k.digest, req, self.partial_hits)
            if hit == 0:
                self.pool.give(k.alloc.kv_blocks + k.alloc.state_blocks)     # superseded: the prompt left it
                self.stats["dropped_stale"] += 1
        charge = self.footprint(req.num_tokens)
        self.hard += charge
        self.stats["peak_hard"] = max(self.stats["peak_hard"], self.hard)
        n_kv = kv_blocks_for(req.num_tokens)
        if hit:
            alloc = k.alloc
            if len(alloc.kv_blocks) > n_kv:                 # partial hit (simulator rule): the rest goes back
                self.pool.give(alloc.kv_blocks[n_kv:])
                del alloc.kv_blocks[n_kv:]
            need = n_kv - len(alloc.kv_blocks)
            self._make_room(need)
            alloc.kv_blocks += self.pool.take(need)
        else:
            self._make_room(n_kv + STATE_BLOCKS)
            alloc = Allocation(self.pool.take(n_kv), self.pool.take(STATE_BLOCKS))
        self.held[req.req_id] = alloc
        self._charge[req.req_id] = charge
        self.stats["admitted"] += 1
        if hit:
            self.stats["hits"] += 1
            self.stats["hit_tokens"] += hit
        return hit

    def admit(self, req) -> Allocation | None:
        """Phase-A entry point: allocate if it fits now, else None. Sets req.cached to the hit."""
        if not self.fits(req):
            return None
        req.cached = self.allocate(req)
        return self.held[req.req_id]

    def _make_room(self, blocks_needed: int):
        """Evict kept sessions, least recently used first, until the hard charge fits and `blocks_needed` blocks are
        free."""
        while self.kept and (self.hard + self.kept_charge > self.capacity
                             or (not self.pool.virtual and self.pool.num_free < blocks_needed)):
            _, k = self.kept.popitem(last=False)
            self.kept_charge -= self.kept_cost(k.tokens)
            self.pool.give(k.alloc.kv_blocks + k.alloc.state_blocks)
            self.stats["evicted_sessions"] += 1
            self.stats["evicted_tokens"] += k.tokens
        assert self.pool.virtual or self.pool.num_free >= blocks_needed

    def release(self, req):
        """The request is done (its token is out). With `keep_sessions`, its prompt's KV and final KDA state stay as
        its session's kept context; otherwise its blocks are freed."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        if not (self.keep_sessions and req.session is not None):
            self.pool.give(alloc.kv_blocks + alloc.state_blocks)
            return
        old = self.kept.pop(req.session, None)       # the session's entry went to this request; a stray one goes
        if old is not None:
            self.kept_charge -= self.kept_cost(old.tokens)
            self.pool.give(old.alloc.kv_blocks + old.alloc.state_blocks)
        digest = prefix_digest(req.token_ids, req.num_tokens) if req.token_ids is not None else None
        self.kept[req.session] = Kept(req.session, req.num_tokens, alloc, digest)
        self.kept_charge += self.kept_cost(req.num_tokens)
        self._make_room(0)

    def drop(self, req):
        """Give back a held allocation without keeping anything (a cancelled request)."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        self.pool.give(alloc.kv_blocks + alloc.state_blocks)
