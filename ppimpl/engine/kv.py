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

Matching (server, `content_keys`): a kept context is found by content, not by a session id. Its key is its first
`SESSION_KEY_TOKENS` token ids (all of them if it is shorter); a new prompt looks up the key of its own first tokens
(`find_session`), and the hit then needs the prompt to start with the whole kept context, compared token by token
(`resumable_tokens`). req-frontend starts every session at its own corpus offset, so sessions share no first tokens
(5_prefix_cache.md §2). Two prompts that share their first tokens but not the whole kept context get no hit, and the
newer context replaces the older one.

Accounting. `blocks` (the real pool): a request costs blocks_for(prompt), a kept session the same. `tokens` (replay
only, to reproduce the targets): the simulator's token ledger, a running request costs prompt + 1 + 51,456 tokens and
a kept session prompt + 51,456, against the pool's token count; block ids then come from an unbounded virtual pool.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from ppimpl.config import BLOCK_TOKENS, NULL_BLOCK, STATE_BLOCKS

STATE_TOKENS = STATE_BLOCKS * BLOCK_TOKENS     # 51,456: what the 6 state blocks cost in tokens
SESSION_KEY_TOKENS = 64                        # content key of a kept context: its first 64 token ids


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


def common_prefix(content, token_ids) -> int | None:
    """Tokens a kept context and a prompt share from the start (None without the ids). A numpy compare of the whole
    overlap: for misses only, not on a full hit's path."""
    if content is None or token_ids is None:
        return None
    n = min(len(content), len(token_ids))
    diff = np.flatnonzero(np.asarray(content[:n]) != np.asarray(token_ids[:n], dtype=np.int32))
    return int(diff[0]) if len(diff) else n


def prefix_content(token_ids, n: int) -> np.ndarray:
    """A prompt's first n token ids (int32, a view when token_ids is already an int32 array): what a kept context is
    compared with."""
    return np.asarray(token_ids[:n], dtype=np.int32)


def session_key(token_ids, n: int | None = None) -> bytes:
    """Content key of a context: its first SESSION_KEY_TOKENS token ids (of its first n tokens)."""
    n = len(token_ids) if n is None else n
    return prefix_content(token_ids, min(n, SESSION_KEY_TOKENS)).tobytes()


def resumable_tokens(tokens: int, content: np.ndarray | None, req, partial: bool = False) -> int:
    """How much of a kept context of `tokens` tokens a request can resume from: all of it if the prompt starts with
    it and goes beyond it, else 0 (its KDA state is at its end only).

    The request says what it shares with its session's previous context either as `prefix_tokens` (the trace's
    declared prefix, replay) or by its `token_ids` (server). `partial` is the simulator's rule, a hit of
    min(kept, declared) even inside a kept context; it is not realizable with one KDA state per context and exists
    only to reproduce the targets in replay.
    """
    if tokens <= 0:
        return 0
    if req.token_ids is not None and content is not None:
        ok = tokens < req.num_tokens and np.array_equal(prefix_content(req.token_ids, tokens), content)
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
    content: np.ndarray | None          # its token ids (server); None in replay
    token: int | None = None            # the output token its prompt got (server)


class KvManager:
    """Per-request allocation over the pool, plus the kept sessions. Pure state; the scheduler drives it."""

    def __init__(self, num_blocks: int, keep_sessions: bool = False, accounting: str = "blocks",
                 partial_hits: bool = False, content_keys: bool = False):
        assert accounting in ("blocks", "tokens")
        self.accounting = accounting
        self.keep_sessions = keep_sessions
        self.content_keys = content_keys               # server: kept contexts keyed by session_key, not req.session
        self._short = Counter()                        # content keys: lengths of kept contexts under the key size
        self.partial_hits = partial_hits
        self.pool = BlockPool(num_blocks, virtual=accounting == "tokens")
        self.capacity = num_blocks if accounting == "blocks" else num_blocks * BLOCK_TOKENS
        self.hard = 0                                  # charge of running and held requests
        self.kept_charge = 0
        self.kept: OrderedDict[object, Kept] = OrderedDict()   # least recently used first
        self.held: dict[str, Allocation] = {}          # running and held requests
        self._charge: dict[str, int] = {}
        self.stats = dict(admitted=0, hits=0, hit_tokens=0, evicted_sessions=0, evicted_tokens=0, dropped_stale=0,
                          peak_hard=0, preemptions=0, repeats=0,    # no preemption: the count stays 0
                          blocked_by_write=0)   # fits() checks a pinned kept context (tier write in flight) turned down
        self.evicted: dict[object, int] = {}           # key -> tokens of a kept context evicted and not replaced since
        self.pinned: Counter = Counter()               # kept contexts a tier write still copies from (not evictable)

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
        """Whether the request's whole footprint fits now, kept sessions evicted as needed (not the pinned ones)."""
        pinned = sum(self.kept_cost(self.kept[k].tokens) for k in self.pinned if k in self.kept)
        need = self.hard + self.footprint(req.num_tokens)
        if need + pinned <= self.capacity:
            return True
        if pinned and need <= self.capacity:      # only a tier write's pin keeps it out (write-before-evict)
            self.stats["blocked_by_write"] += 1
        return False

    # --- pins: a kept context a tier write copies from ------------------------------------------------------------

    def kept_key(self, req):
        """The key `release` keeps the request's context under."""
        return session_key(req.token_ids) if self.content_keys else req.session

    def pin(self, key):
        self.pinned[key] += 1

    def unpin(self, key):
        self.pinned[key] -= 1
        if self.pinned[key] <= 0:
            del self.pinned[key]

    def busy(self, req) -> bool:
        """The request would take over (or give back) its session's kept context while a tier write still copies
        from it: it waits."""
        return req.session is not None and req.session in self.pinned and req.session in self.kept

    @property
    def used_tokens(self) -> int:
        """Pool tokens in use by running/held requests and kept sessions (blocks × 8,576 in block accounting)."""
        u = self.hard + self.kept_charge
        return u * BLOCK_TOKENS if self.accounting == "blocks" else u

    # --- lookups ------------------------------------------------------------------------------------------------

    def find_session(self, token_ids) -> bytes:
        """Content keys: the key of the kept context this prompt may continue (its own key if none is kept)."""
        n = len(token_ids)
        key = session_key(token_ids)
        if key not in self.kept:
            for m in sorted(m for m, c in self._short.items() if c and m < n):
                short = session_key(token_ids, m)
                if short in self.kept:
                    return short
        return key

    def _hit(self, k: Kept, req) -> int:
        """resumable_tokens, remembered per (request, kept context): the scheduler asks again at every microbatch
        while the request waits, and comparing a long prompt costs ~0.1 ms per call."""
        memo = req.hit_memo
        if memo is not None and memo[0] is k:
            return memo[1]
        hit = resumable_tokens(k.tokens, k.content, req, self.partial_hits)
        req.hit_memo = (k, hit)
        return hit

    def peek_hit(self, req) -> int:
        """The HBM hit the request would get if it were admitted now."""
        if req.session is None or not self.keep_sessions:
            return 0
        k = self.kept.get(req.session)
        return 0 if k is None else self._hit(k, req)

    def shared_prefix(self, req) -> int:
        """Tokens the prompt shares with its session's previous context, as the trace declares it: the HBM hit, else
        what it shares with a kept context it does not extend (superseded), else the length of its evicted context.
        The load budget's backlog counts a waiting round by its tokens beyond this (the simulator counts a round's
        new tokens, not the prefix it recomputes after an eviction). For the server, which has no declared prefix."""
        if req.session is None or not self.keep_sessions:
            return 0
        k = self.kept.get(req.session)
        if k is not None:
            hit = self._hit(k, req)
            return hit or min(common_prefix(k.content, req.token_ids) or 0, req.num_tokens - 1)
        return min(self.evicted.get(req.session, 0), req.num_tokens - 1)

    def repeat_token(self, req) -> int | None:
        """A prompt identical to a kept context: nothing new to compute from its KDA state, and the answer is the
        token that prompt already got (greedy). Returns it and refreshes the context, or None. req-frontend's prefix
        preflight sends one prompt twice and requires the second to report cached tokens."""
        if req.session is None or not self.keep_sessions or req.token_ids is None:
            return None
        k = self.kept.get(req.session)
        if k is None or k.token is None or k.tokens != req.num_tokens \
                or not np.array_equal(prefix_content(req.token_ids, k.tokens), k.content):
            return None
        self.kept.move_to_end(req.session)
        self.stats["repeats"] += 1
        return k.token

    def _unkeep(self, k: Kept):
        self.kept_charge -= self.kept_cost(k.tokens)
        if self.content_keys and k.tokens < SESSION_KEY_TOKENS:
            self._short[k.tokens] -= 1

    # --- allocation ---------------------------------------------------------------------------------------------

    def allocate(self, req) -> int:
        """Reserve the request's whole footprint (the caller checked `fits`); take over its session's kept context
        when it resumes from it. Returns the hit: tokens already in the allocated KV."""
        assert self.fits(req), req.req_id
        k = self.kept.pop(req.session, None) if req.session is not None else None
        hit = 0
        evicted_before = self.stats["evicted_sessions"]
        info = {"hit_kind": "none_first_round", "kept_tokens": 0, "common_prefix_tokens": None}
        if k is not None:
            self._unkeep(k)
            hit = self._hit(k, req)
            info["kept_tokens"] = k.tokens
            if hit == 0:
                self.pool.give(k.alloc.kv_blocks + k.alloc.state_blocks)     # superseded: the prompt left it
                self.stats["dropped_stale"] += 1
                info["hit_kind"] = "superseded"
                info["common_prefix_tokens"] = common_prefix(k.content, req.token_ids)
            else:
                info["hit_kind"] = "full" if hit == k.tokens else "partial"   # partial: the simulator's rule (replay)
                info["common_prefix_tokens"] = hit
        elif req.session is not None and req.session in self.evicted:
            info["hit_kind"] = "evicted"
            info["evicted_context_tokens"] = self.evicted[req.session]
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
        info["evicted_sessions_at_admit"] = self.stats["evicted_sessions"] - evicted_before
        req.admit_info = info
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
            key = next((x for x in self.kept if x not in self.pinned), None)
            if key is None:
                break
            k = self.kept.pop(key)
            self._unkeep(k)
            self.evicted[key] = k.tokens
            self.pool.give(k.alloc.kv_blocks + k.alloc.state_blocks)
            self.stats["evicted_sessions"] += 1
            self.stats["evicted_tokens"] += k.tokens
        assert self.pool.virtual or self.pool.num_free >= blocks_needed

    def release(self, req):
        """The request is done (its token is out). With `keep_sessions`, its prompt's KV and final KDA state stay as
        its session's kept context (returns its key); otherwise its blocks are freed."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        if not (self.keep_sessions and req.session is not None):
            self.pool.give(alloc.kv_blocks + alloc.state_blocks)
            return None
        key = self.kept_key(req)
        old = self.kept.pop(key, None)               # the session's entry went to this request; a stray one goes
        if old is not None:
            self._unkeep(old)
            self.pool.give(old.alloc.kv_blocks + old.alloc.state_blocks)
        content = prefix_content(req.token_ids, req.num_tokens) if req.token_ids is not None else None
        self.kept[key] = Kept(key, req.num_tokens, alloc, content, req.output_token)
        self.evicted.pop(key, None)
        self.kept_charge += self.kept_cost(req.num_tokens)
        if self.content_keys and req.num_tokens < SESSION_KEY_TOKENS:
            self._short[req.num_tokens] += 1
        self._make_room(0)
        return key

    def drop(self, req):
        """Give back a held allocation without keeping anything (a cancelled request)."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        self.pool.give(alloc.kv_blocks + alloc.state_blocks)
