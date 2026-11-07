"""The page pool and the exact prefix cache (ssengine/runtime/README.md "Block-id space"; 5_prefix_cache.md §2;
docs/design/kept_states.md).

One page space for all stages: a block is 67 pages of 128 tokens (`config.PAGE_TOKENS`). A request holds ceil(L / 128)
KV **pages** (its token positions in order) and its KDA **state**: R runs of c pages (`StateShape`, the compact layout
of `model/state_layout.py`: c = 64, R = 5 for GLM-5.3-Flash, 320 pages), reserved whole when it is admitted; no
preemption. Page p is page p % 67 of block p // 67 (`PagePool`): the kernels address KV through page tables at that
granularity, so a context costs its tokens rounded up to 128, not to a block of 8,576 (the simulator charges by the
token: ServingStudioSim `build_pipeline_head_worker.rs`, "Context is charged by the token").

Prefix cache (phase B, `keep_sessions`): when a request finishes, its pages stay as its session's **kept** context:
the KV of its prompt and the KDA state after the prompt's last token (**end**). With `kept_states` 2 (k2) it also keeps
the state it resumed from (**inner**: the previous round's prompt end), so a prompt that rewrites the tail of the
previous context still resumes from there. The session's next round takes them over (ownership moves, nothing is
shared) and computes only the rest of its prompt from the deepest kept state its prompt extends (`resumable_tokens`;
the hit is exact, not rounded to a block). With k2 a resumed round computes on a fresh state, the one it resumed
from copied into it before its first chunk (copy-on-resume: `Allocation.base`, `Chunk.state_from`), and the other
kept state of the context is given back. Kept sessions use the same pool as running requests and yield to them: when
a request needs room, kept sessions are evicted least recently used first, a whole session (both states) at a time.

Tiers (`lower`, set by the scheduler: the KV pool's `HbmLevel`, dev:research/TIERS.md §5). The pool is the level below
HBM: it hears of every kept context (`kept`, `unkept`, `evicted`) and admission (`admitted`), writes kept contexts
back ahead of eviction (KV and every kept state), and says which may be evicted now (`evictable`: written whole below)
and how much charge is not yet (`held_back`, `make_room`). A request that fits only once writes land waits
(`blocked_by_write`).

Matching (server, `content_keys`): a kept context is found by its whole content, not by a session id. Every kept
state point (end and inner) is indexed by the chained hash of its whole `HASH_BLOCK_TOKENS`-token blocks
(`block_hashes`: block i's hash covers tokens [0, 64 (i + 1)), as vLLM's prefix cache does); the tokens past its last
whole block are compared directly. A new prompt walks its own block hashes and takes the deepest state point, in HBM
or a tier below, that it starts with (`find_session`); a prompt identical to a context finds it too (`repeat_token`).
Each context has its own key: a prompt that resumes one keeps its next context under the same key (one key per
lineage, which the tiers replace and drop by, `drop_stale`); any other prompt gets a new key. Contexts that share
their first tokens (agent trees under one system prompt, sibling subagents) therefore no longer replace one another:
each stays until it is resumed or evicted.

Loading area (staged loading, `area` > 0; the scheduler's docstring). `area` pages are held for tier reads whether
used or not: running requests and kept contexts never use them (`hard + kept + max(area_held, area) <= capacity`, in
`fits` and `_make_room`). A read holds its context's pages and one state there (`hold_read`, `area_held`) until its
request is admitted, which takes them over (`allocate`; with k2 the read's state becomes the base and a fresh one is
taken); a read larger than the whole area starts when the area is empty, once kept contexts make the room. A read
whose room is held only by kept contexts not yet written below waits (`area_waits`), never an admission.

Accounting. `pages` (the real pool): a request costs ceil(prompt / 128) + states x c x R pages (2 states while a k2
round runs resumed, else 1), a kept session its KV and its kept states the same way; the charge is exactly the pages
the pool hands out (`check_pool` in the tests). The pool can hand out the charge whenever it is free, except that a
state's runs need free run slots: when there are too few (the free pages are spread over partly used blocks), an
admission that needs new states evicts more kept contexts (their states free run slots) or waits for a running
request to finish (`fits`, `_make_room`). `tokens` (replay only, to reproduce the targets): the simulator's token
ledger, a running request costs prompt + 1 + 51,456 tokens and a kept session prompt + 51,456 per state, against the
pool's token count; pages then come from an unbounded virtual pool.
"""

import functools
import hashlib
import itertools
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

from ssengine.config import BLOCK_PAGES, BLOCK_TOKENS, NULL_BLOCK, PAGE_TOKENS, STATE_BLOCKS

STATE_TOKENS = STATE_BLOCKS * BLOCK_TOKENS     # 51,456: a state in the simulator's token ledger (vLLM's 6 blocks)
HASH_BLOCK_TOKENS = 64                         # content index: tokens per chained block hash


@dataclass(frozen=True)
class StateShape:
    """A KDA state in the page space: `runs` runs of `run_pages` consecutive pages, each inside one block at a page
    offset that is a multiple of run_pages (docs/design/kept_states.md)."""
    runs: int
    run_pages: int

    @property
    def pages(self) -> int:
        return self.runs * self.run_pages


@functools.cache
def default_state() -> StateShape:
    """GLM-5.3-Flash's compact state at the design split (the workers report theirs in `Ready`)."""
    from ssengine.model.state_layout import glm53_geometry

    g = glm53_geometry()
    return StateShape(g.runs, g.run_pages)


def kv_blocks_for(tokens: int) -> int:
    """Blocks (8,576 tokens) a context of `tokens` spans: the tiers' slots are per block."""
    return -(-tokens // BLOCK_TOKENS)


def kv_pages_for(tokens: int) -> int:
    return -(-tokens // PAGE_TOKENS)


def pages_for(tokens: int, states: int = 1, state: StateShape | None = None) -> int:
    """A context's footprint in pages: its DSA KV pages plus its KDA states' pages."""
    return kv_pages_for(tokens) + states * (state or default_state()).pages


def block_pages(blocks) -> list[int]:
    """The pages of whole blocks, in order (a context laid out block by block, as the benches build them)."""
    return [b * BLOCK_PAGES + i for b in blocks for i in range(BLOCK_PAGES)]


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


def block_hashes(token_ids, n: int | None = None) -> list[bytes]:
    """Chained hashes of the first n tokens' whole HASH_BLOCK_TOKENS-token blocks, with the empty prefix's first:
    entry i covers tokens [0, HASH_BLOCK_TOKENS * i)."""
    n = len(token_ids) if n is None else n
    arr = prefix_content(token_ids, n)
    out = [b""]
    h = b""
    for i in range(n // HASH_BLOCK_TOKENS):
        h = hashlib.blake2b(h + arr[i * HASH_BLOCK_TOKENS:(i + 1) * HASH_BLOCK_TOKENS].tobytes(), digest_size=16).digest()
        out.append(h)
    return out


def resumable_tokens(tokens: int, content: np.ndarray | None, req, partial: bool = False, inner: int = 0) -> int:
    """How much of a kept context of `tokens` tokens a request can resume from: all of it if the prompt starts with
    it and goes beyond it; else, with an `inner` kept state (k2), `inner` tokens if the prompt starts with those and
    goes beyond them; else 0 (KDA states exist only at those points).

    The request says what it shares with its session's previous context either as `prefix_tokens` (the trace's
    declared prefix, replay) or by its `token_ids` (server). `partial` is the simulator's rule, a hit of
    min(kept, declared) even inside a kept context; it is not realizable with one KDA state per context and exists
    only to reproduce the targets in replay.
    """
    if tokens <= 0:
        return 0
    if req.token_ids is not None and content is not None:
        for t in (tokens, inner):
            if 0 < t < req.num_tokens and np.array_equal(prefix_content(req.token_ids, t), content[:t]):
                return t
        return 0
    shared = req.prefix_tokens
    if partial:
        hit = min(tokens, shared)
        return hit - 1 if hit >= req.num_tokens else hit
    for t in (tokens, inner):
        if 0 < t <= shared and t < req.num_tokens:
            return t
    return 0


class PagePool:
    """The free pages over blocks 1..num_blocks (block 0 is the null block); page p is page p % BLOCK_PAGES of block
    p // BLOCK_PAGES. KV takes pages, a state `state.runs` runs, each the first `state.run_pages` pages of a block, named
    by the block id (a layer's blocks are strided in vLLM's shared allocation, so a run never crosses a block and a
    block holds one run). `virtual`: no limit (token accounting).

    Rules: a state's runs come first from partly used blocks whose run pages are all free (the fullest first), then
    from whole free blocks (the oldest first); the rest of such a block is ordinary pages. KV pages come from partly
    used blocks first, the fullest first (lowest page first within a block), and a whole free block is split only when
    no partly used block has a free page, so free pages outside whole blocks come only from frees, and blocks fill up
    and empty out whole. Whole free blocks are handed out in the order they became whole (the oldest first, as the free
    list of blocks was: a block an eviction just freed, which a tier write may still copy from, is reused last). An
    admission takes its state before its KV pages, so the pages never take the run slots the state was counted on."""

    def __init__(self, num_blocks: int, virtual: bool = False, state: StateShape | None = None):
        self.num_blocks = num_blocks
        self.virtual = virtual
        self.state = state or default_state()
        assert self.state.run_pages <= BLOCK_PAGES, self.state
        self._used = np.zeros(num_blocks + 1, dtype=np.int64)       # pages in use per block
        self._used[NULL_BLOCK] = BLOCK_PAGES
        self._free = np.ones((num_blocks + 1) * BLOCK_PAGES, dtype=bool)
        self._free[:BLOCK_PAGES] = False
        self._whole: OrderedDict[int, None] = OrderedDict.fromkeys(range(1, num_blocks + 1))
        self._split: list[set[int]] = [set() for _ in range(BLOCK_PAGES)]   # partly used blocks by free pages
        self.num_free = num_blocks * BLOCK_PAGES                    # free pages

    @property
    def num_whole(self) -> int:
        return len(self._whole)

    def _partial_slots(self):
        """Partly used blocks whose run pages are all free, the fullest first."""
        c = self.state.run_pages
        for f in range(c, BLOCK_PAGES):
            for b in sorted(self._split[f]):
                if self._free[b * BLOCK_PAGES:b * BLOCK_PAGES + c].all():
                    yield b

    def free_runs(self, at_least: int | None = None) -> int:
        """Free run slots (counted up to `at_least` when given: enough to know)."""
        n = self.num_whole
        if at_least is not None and n >= at_least:
            return n
        for _ in self._partial_slots():
            n += 1
            if at_least is not None and n >= at_least:
                break
        return n

    def can_take(self, pages: int, states: int) -> bool:
        """Whether `pages` KV pages and `states` states can be handed out now."""
        if self.virtual:
            return True
        runs = states * self.state.runs
        return self.num_free - runs * self.state.run_pages >= pages and (runs == 0 or self.free_runs(runs) >= runs)

    def _grow(self, blocks: int):
        n = self.num_blocks
        self._used = np.r_[self._used, np.zeros(blocks, dtype=np.int64)]
        self._free = np.r_[self._free, np.ones(blocks * BLOCK_PAGES, dtype=bool)]
        self._whole.update(OrderedDict.fromkeys(range(n + 1, n + 1 + blocks)))
        self.num_blocks += blocks
        self.num_free += blocks * BLOCK_PAGES

    def _set_used(self, b: int, used: int):
        old = self._used[b]
        if 0 < old < BLOCK_PAGES:
            self._split[BLOCK_PAGES - old].discard(b)
        elif old == 0:
            self._whole.pop(b, None)
        self._used[b] = used
        if 0 < used < BLOCK_PAGES:
            self._split[BLOCK_PAGES - used].add(b)
        elif used == 0:
            self._whole[b] = None

    def take_state(self) -> list[int]:
        """One state: the block of each of its runs."""
        R, c = self.state.runs, self.state.run_pages
        if self.virtual and self.free_runs(R) < R:
            self._grow(max(R, self.num_blocks))
        out: list[int] = []

        def mark(b: int):
            base = b * BLOCK_PAGES
            self._free[base:base + c] = False
            self._set_used(b, int(self._used[b]) + c)
            out.append(b)

        for b in list(itertools.islice(self._partial_slots(), R)):
            mark(b)
        while len(out) < R:
            assert self._whole, (R, len(out))
            mark(next(iter(self._whole)))
        self.num_free -= R * c
        return out

    def take_pages(self, n: int) -> list[int]:
        if self.virtual and n > self.num_free:
            self._grow(max(-(-(n - self.num_free) // BLOCK_PAGES), self.num_blocks))
        assert n <= self.num_free, (n, self.num_free)
        out: list[int] = []
        need = n
        for f in range(1, BLOCK_PAGES):                 # partly used blocks, the fullest first
            if need == 0:
                break
            for b in sorted(self._split[f]):
                if need == 0:
                    break
                base = b * BLOCK_PAGES
                got = (np.flatnonzero(self._free[base:base + BLOCK_PAGES])[:need] + base).tolist()
                self._free[got] = False
                self._set_used(b, self._used[b] + len(got))
                out += got
                need -= len(got)
        while need:                                     # then whole blocks, the oldest free first
            b = next(iter(self._whole))
            k = min(need, BLOCK_PAGES)
            base = b * BLOCK_PAGES
            self._free[base:base + k] = False
            self._set_used(b, k)
            out += range(base, base + k)
            need -= k
        self.num_free -= n
        return out

    def give_pages(self, pages: list[int]):
        if not len(pages):
            return
        p = np.asarray(pages, dtype=np.int64)
        assert not self._free[p].any() and (p >= BLOCK_PAGES).all()
        self._free[p] = True
        blocks, counts = np.unique(p // BLOCK_PAGES, return_counts=True)
        for b, c in zip(blocks.tolist(), counts.tolist()):
            self._set_used(b, int(self._used[b]) - c)
        self.num_free += len(p)

    def run_pages(self, runs: list[int]) -> list[int]:
        """The pages of a state's runs (their blocks)."""
        c = self.state.run_pages
        return [p for b in runs for p in range(b * BLOCK_PAGES, b * BLOCK_PAGES + c)]

    def give_state(self, runs: list[int]):
        assert all(b != NULL_BLOCK for b in runs)
        self.give_pages(self.run_pages(runs))

    def give(self, alloc: "Allocation"):
        self.give_pages(alloc.kv_pages)
        self.give_state(alloc.state_blocks)
        self.give_state(alloc.base)


@dataclass
class Allocation:
    kv_pages: list[int]                 # KV pages from token 0, PAGE_TOKENS each
    state_blocks: list[int]             # its KDA state: the block of each run (StateShape)
    run_pages: int = BLOCK_PAGES        # pages per run
    base: list[int] = field(default_factory=list)   # k2: the state it resumed from (kept as the context's inner)
    copy_pending: bool = False          # k2: its first chunk copies `base` into `state_blocks` (Chunk.state_from)

    @property
    def num_pages(self) -> int:
        return len(self.kv_pages) + (len(self.state_blocks) + len(self.base)) * self.run_pages

    @property
    def read_state(self) -> list[int]:
        """Where a tier read puts the state it resumes from: the base (k2), else the state itself."""
        return self.base or self.state_blocks


@dataclass(eq=False)
class Kept:
    """A kept session: its last prompt's KV, the KDA state at the prompt's end (`alloc.state_blocks`) and, with k2,
    the state its round resumed from (`inner` tokens, `alloc.base`)."""
    session: object
    tokens: int
    alloc: Allocation
    content: np.ndarray | None          # its token ids (server); None in replay
    token: int | None = None            # the output token its prompt got (server)
    inner: int = 0                      # k2: tokens of the inner kept state (0: none)

    @property
    def states(self) -> int:
        return 1 + bool(self.inner)


class KvManager:
    """Per-request allocation over the pool, plus the kept sessions. Pure state; the scheduler drives it.

    `kept_states`: 1 (k1: a kept context keeps the state at its end; a resumed round computes on it in place) or 2
    (k2: also the state it resumed from; copy-on-resume). `state`: the compact state's shape (the workers'
    `Ready`; default GLM-5.3-Flash's)."""

    def __init__(self, num_blocks: int, keep_sessions: bool = False, accounting: str = "pages",
                 partial_hits: bool = False, content_keys: bool = False, kept_states: int = 1,
                 state: StateShape | None = None):
        assert accounting in ("pages", "tokens")
        assert kept_states in (1, 2), kept_states
        self.accounting = accounting
        self.keep_sessions = keep_sessions
        self.k2 = kept_states == 2 and keep_sessions
        self.content_keys = content_keys               # server: contexts found by content (find_session)
        self._index: dict[bytes, set] = {}             # content keys: hash of a kept state point's whole blocks ->
                                                       # keys of contexts with a state point in the block after them
                                                       # (kept, in a tier below, or evicted; pruned lazily in
                                                       # find_session)
        self._new_keys = itertools.count(1)            # content keys: keys of new lineages
        self.partial_hits = partial_hits
        self.state = state or default_state()
        self.pool = PagePool(num_blocks, virtual=accounting == "tokens", state=self.state)
        self.capacity = num_blocks * BLOCK_PAGES if accounting == "pages" else num_blocks * BLOCK_TOKENS
        self.state_cost = self.state.pages if accounting == "pages" else STATE_TOKENS
        self.hard = 0                                  # charge of running and held requests
        self.kept_charge = 0
        self.kept: OrderedDict[object, Kept] = OrderedDict()   # least recently used first
        self.held: dict[str, Allocation] = {}          # running and held requests
        self._charge: dict[str, int] = {}
        self.stats = dict(admitted=0, hits=0, hit_tokens=0, evicted_sessions=0, evicted_tokens=0, dropped_stale=0,
                          peak_hard=0, preemptions=0, repeats=0,    # no preemption: the count stays 0
                          inner_hits=0,         # k2: hits on a kept context's inner state
                          blocked_by_write=0,   # fits() checks turned down only because kept contexts are not yet
                                                # written to the tier below (an HBM eviction waits for its write)
                          area_waits=0,         # hold_read turned down only for that reason (a read waits)
                          states_short=0)       # checks whose charge fit but whose states found too few free run
                                                # slots (more kept contexts are evicted, or the request waits)
        self.evicted: dict[object, int] = {}           # key -> tokens of a kept context evicted and not replaced since
        self.pinned: Counter = Counter()               # kept contexts a tier write still copies from (not evictable;
                                                       # their next round waits to take them over, `busy`)
        # The tiers below HBM (kvpool.HbmLevel: kept contexts are written there before they are evicted), or None:
        # it is told about kept, taken-over and evicted contexts and decides which kept context may be evicted now.
        self.lower = None
        self.area = 0                                  # loading area in pages (staged loading), held for reads
        self.area_held = 0                             # pages the reads hold in it
        self.reads: dict[str, Allocation] = {}         # req id -> the pages and state its read holds, until admitted
        # Byte gate (staged loading, as the simulator's held area): reads start while their bytes in the tiers' units
        # (`read_bytes(tokens)`: KV and one state as a tier stores them) fit `area_bytes`; their pages are held
        # physically, the area counting max(pages held, area). None: the gate is the pages themselves.
        self.read_bytes = None
        self.area_bytes = 0
        self.area_used_bytes = 0
        self._read_charge: dict[str, int] = {}

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    def _alloc(self, kv_pages: list[int], state_blocks: list[int]) -> Allocation:
        return Allocation(kv_pages, state_blocks, self.state.run_pages)

    # --- charges ------------------------------------------------------------------------------------------------

    def footprint(self, prompt_tokens: int, states: int = 1) -> int:
        """What a running request of `prompt_tokens` holding `states` states costs, in the pool's unit."""
        if self.accounting == "pages":
            return kv_pages_for(prompt_tokens) + states * self.state_cost
        return prompt_tokens + 1 + states * self.state_cost

    def kept_cost(self, tokens: int, states: int = 1) -> int:
        kv = kv_pages_for(tokens) if self.accounting == "pages" else tokens
        return kv + states * self.state_cost

    def kept_cost_of(self, key) -> int:
        k = self.kept[key]
        return self.kept_cost(k.tokens, k.states)

    def fits_ever(self, prompt_tokens: int) -> bool:
        return self.footprint(prompt_tokens, 1 + self.k2) <= self.capacity

    def area_charge(self, area_held: int | None = None) -> int:
        """What the loading area counts as held: its reads' pages, at least the whole area."""
        return max(self.area_held if area_held is None else area_held, self.area)

    def area_fits(self, tokens: int) -> bool:
        """Whether a read of a `tokens`-long context may start in the loading area now: it fits, or the area is empty
        (a read larger than the whole area). With the byte gate, in the tiers' bytes (`read_bytes`)."""
        if self.read_bytes is not None:
            return self.area_used_bytes == 0 or self.area_used_bytes + self.read_bytes(tokens) <= self.area_bytes
        c = self.kept_cost(tokens)
        return self.area_held == 0 or self.area_held + c <= self.area

    def _states_free(self, states: int, give_back: int = 0, skip=None) -> bool | None:
        """Whether `states` states find free run slots once `_make_room` has run (the pages are, when the charge
        fits: the charge is what the pool hands out): they are free now, or with the `give_back` states the admission
        gives back first, or once kept contexts but `skip` that may be evicted give theirs back. None: only kept
        contexts not yet written below could."""
        R = self.state.runs
        need = (states - give_back) * R
        if self.pool.virtual or need <= 0 or self.pool.free_runs(need) >= need:
            return True
        self.stats["states_short"] += 1
        need -= self.pool.free_runs()
        later = False
        for x, k in self.kept.items():
            if x == skip:
                continue
            if x not in self.pinned and (self.lower is None or self.lower.evictable(x)):
                need -= k.states * R
                if need <= 0:
                    return True
            else:
                later = True
        return None if later else False

    def _admission(self, req, rd, own, read: bool = False) -> tuple[int, int, int]:
        """(states the admitted request holds, states it takes from the pool, states it gives back first). A request
        takes over its read's state or, on a hit, its kept context's (k1: computes on it; k2: keeps it as its base and
        takes a fresh one, giving back the context's other state); a superseded kept context is given back. With k2
        a round that will read a tier (`read`) holds a base for the read besides its own state."""
        hit = self._hit(own, req) if own is not None and rd is None else 0
        back = own.states if own is not None and (rd is not None or not hit) else 0
        if not self.k2:
            if rd is not None or hit:
                return 1, 0, back
            return 1, 1, back
        if rd is not None:
            return 2, 1, back
        if hit:
            return 2, 1, own.states - 1
        return (2, 2, back) if read else (1, 1, back)

    def fits(self, req, read: bool = False) -> bool:
        """Whether the request's whole footprint fits now, kept sessions evicted as needed (only those the lower level
        lets go: written there). If it fits only by evicting ones not yet written, the lower level starts their
        writes (`make_room`) and the request waits. A request whose read holds pages in the loading area takes them
        over. When the charge fits without evicting but its new states find too few free run slots, more kept
        contexts are evicted for them (`_make_room`); with none to evict, the request waits for a running one to end.
        `read`: the request will read its context from a tier (k2: it holds a base state for it)."""
        rd = self.reads.get(req.req_id)
        own_k = self.kept.get(req.session) if req.session is not None else None   # taken over (or given back) at
        held_states, take, back = self._admission(req, rd, own_k, read)
        need = self.hard + self.footprint(req.num_tokens, held_states) \
            + self.area_charge(self.area_held - (0 if rd is None else rd.num_pages))
        own = 0 if own_k is None else self.kept_cost(own_k.tokens, own_k.states)  # admission: not counted twice
        if need + self.kept_charge - own <= self.capacity:
            ok = self._states_free(take, back, skip=req.session)
            if ok is None:                        # only writes still to land free run slots for its states
                self.stats["blocked_by_write"] += 1
                self.lower.make_room(take * self.state_cost)
            return bool(ok)
        held = 0 if self.lower is None else self.lower.held_back()
        if own and self.lower is not None and not self.lower.evictable(req.session):
            held -= own
        if need + held <= self.capacity:
            return True
        if held and need <= self.capacity:        # only writes still to land keep it out (write-before-evict)
            self.stats["blocked_by_write"] += 1
            self.lower.make_room(need + held - self.capacity)
        return False

    def hold_read(self, req, tokens: int) -> Allocation | None:
        """Hold the pages and one state of a `tokens`-long context in the loading area for the request's read (the
        caller checked `area_fits`), kept sessions evicted as needed. None if that needs kept contexts not yet written
        below: their writes start (`make_room`) and the read waits."""
        assert self.accounting == "pages" and req.req_id not in self.reads
        c = self.kept_cost(tokens)
        base = self.hard + self.area_charge(self.area_held + c)
        if base + self.kept_charge > self.capacity:
            held = 0 if self.lower is None else self.lower.held_back()
            if base + held > self.capacity:
                if held and base <= self.capacity:
                    self.stats["area_waits"] += 1
                    self.lower.make_room(base + held - self.capacity)
                return None
        else:
            ok = self._states_free(1)
            if not ok:
                if ok is None:
                    self.stats["area_waits"] += 1
                    self.lower.make_room(self.state_cost)
                return None
        self.area_held += c
        if self.read_bytes is not None:
            self._read_charge[req.req_id] = b = self.read_bytes(tokens)
            self.area_used_bytes += b
        n_kv = kv_pages_for(tokens)
        self._make_room(n_kv, 1)
        st = self.pool.take_state()
        alloc = self._alloc(self.pool.take_pages(n_kv), st)
        self.reads[req.req_id] = alloc
        return alloc

    def _read_gone(self, req_id: str, alloc: Allocation):
        self.area_held -= alloc.num_pages
        self.area_used_bytes -= self._read_charge.pop(req_id, 0)

    def drop_read(self, req):
        """Give back what the request's read holds in the loading area (a read that could not start)."""
        alloc = self.reads.pop(req.req_id)
        self._read_gone(req.req_id, alloc)
        self.pool.give(alloc)

    # --- pins: a kept context a tier write copies from ------------------------------------------------------------

    def kept_key(self, req):
        """The key `release` keeps the request's context under: its session's (content keys: the lineage it resumed or
        started; a new one if another request already kept a context under that key)."""
        if self.content_keys and (req.session is None or req.session in self.kept):
            return next(self._new_keys)
        return req.session

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
        """Pool tokens in use by running/held requests and kept sessions (pages x 128 in page accounting)."""
        u = self.hard + self.kept_charge
        return u * PAGE_TOKENS if self.accounting == "pages" else u

    # --- lookups ------------------------------------------------------------------------------------------------

    def find_session(self, token_ids):
        """Content keys: the key of the context, in HBM or a tier below, with the deepest kept state point (its end,
        or its inner state with k2) that this prompt starts with (or equals, for `repeat_token`); else that of the
        longest evicted context whose whole blocks it starts with (the backlog counts its tokens as shared,
        `shared_prefix`); else a new key."""
        n = len(token_ids)
        arr = prefix_content(token_ids, n)
        best, best_tokens, evicted, evicted_tokens = None, -1, None, -1
        for i, h in enumerate(block_hashes(arr)):
            keys = self._index.get(h)
            if not keys:
                continue
            lo = i * HASH_BLOCK_TOKENS
            for key in list(keys):
                held = self._held_points(key)
                live = [x for x in held if lo <= x[0] < lo + HASH_BLOCK_TOKENS]
                if not live and not (key in self.evicted and lo <= self.evicted[key] < lo + HASH_BLOCK_TOKENS):
                    keys.discard(key)                  # its point here was resumed, replaced or dropped
                    continue
                for tokens, content, end in live:      # an inner point resumes only with a token after it
                    if best_tokens < tokens and (tokens < n or (end and tokens == n)) and content is not None \
                            and np.array_equal(arr[lo:tokens], content[lo:tokens]):
                        best, best_tokens = key, tokens
                if not live and evicted_tokens < self.evicted[key] <= n:
                    evicted, evicted_tokens = key, self.evicted[key]
            if not keys:
                del self._index[h]
        if best is not None:
            return best
        return evicted if evicted is not None else next(self._new_keys)

    def _held_points(self, key) -> list:
        """(tokens, content, is the end) of every kept state point of the key: its HBM context's end and inner, and its
        entries' in the tiers below."""
        out = []
        k = self.kept.get(key)
        if k is not None:
            out.append((k.tokens, k.content, True))
            if k.inner:
                out.append((k.inner, k.content, False))
        if self.lower is not None:
            for t in self.lower.pool.manager.tiers:
                e = t.entries.get(key)
                if e is not None:
                    out.append((e.tokens, e.content, True))
                    if e.inner:
                        out.append((e.inner, e.content, False))
        return out

    def _index_points(self, key, content, points):
        for t in points:
            if t:
                h = block_hashes(content, t)[-1]
                self._index.setdefault(h, set()).add(key)

    def _hit(self, k: Kept, req) -> int:
        """resumable_tokens, remembered per (request, kept context): the scheduler asks again at every microbatch
        while the request waits, and comparing a long prompt costs ~0.1 ms per call."""
        memo = req.hit_memo
        if memo is not None and memo[0] is k:
            return memo[1]
        hit = resumable_tokens(k.tokens, k.content, req, self.partial_hits, k.inner)
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
            if hit == k.tokens:
                return hit
            return max(hit, min(common_prefix(k.content, req.token_ids) or 0, req.num_tokens - 1))
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

    def _unkeep(self, k: Kept, evicted: bool = False):
        self.kept_charge -= self.kept_cost(k.tokens, k.states)
        if self.lower is not None:
            if evicted:
                self.lower.evicted(k.session, k)
            else:
                self.lower.unkept(k.session)

    # --- allocation ---------------------------------------------------------------------------------------------

    def allocate(self, req, read: bool = False) -> int:
        """Reserve the request's whole footprint (the caller checked `fits`); take over its session's kept context
        when it resumes from it. Returns the hit: tokens already in the allocated KV. `read`: a tier read will fill
        its context (`Allocation.read_state`; with k2 a base state is taken for it)."""
        assert self.fits(req, read), req.req_id
        k = self.kept.pop(req.session, None) if req.session is not None else None
        rd = self.reads.pop(req.req_id, None)
        held_states, take, _ = self._admission(req, rd, k, read)
        hit = 0
        evicted_before = self.stats["evicted_sessions"]
        info = {"hit_kind": "none_first_round", "kept_tokens": 0, "common_prefix_tokens": None}
        if rd is not None:                              # its read landed in the loading area: it takes the pages
            self._read_gone(req.req_id, rd)
            info["hit_kind"] = "tier"
            if k is not None:                           # a shorter kept context the read superseded
                self._unkeep(k)
                self.pool.give(k.alloc)
                self.stats["dropped_stale"] += 1
                k = None
        elif k is not None:
            self._unkeep(k)
            hit = self._hit(k, req)
            info["kept_tokens"] = k.tokens
            if hit == 0:
                self.pool.give(k.alloc)                                    # superseded: the prompt left it
                self.stats["dropped_stale"] += 1
                info["hit_kind"] = "superseded"
                info["common_prefix_tokens"] = common_prefix(k.content, req.token_ids)
            else:
                info["hit_kind"] = ("full" if hit == k.tokens else "inner" if hit == k.inner
                                    else "partial")             # partial: the simulator's rule (replay)
                info["common_prefix_tokens"] = hit
        elif req.session is not None and req.session in self.evicted:
            info["hit_kind"] = "evicted"
            info["evicted_context_tokens"] = self.evicted[req.session]
        charge = self.footprint(req.num_tokens, held_states)
        self.hard += charge
        self.stats["peak_hard"] = max(self.stats["peak_hard"], self.hard)
        n_kv = kv_pages_for(req.num_tokens)
        if rd is not None:
            alloc = rd
            if self.k2:                                 # the read's state is the base, computed on in a copy
                self._make_room(0, 1)
                alloc.base, alloc.state_blocks = alloc.state_blocks, self.pool.take_state()
                alloc.copy_pending = True
            need = n_kv - len(alloc.kv_pages)
            self._make_room(need, 0)
            alloc.kv_pages += self.pool.take_pages(need)
        elif hit:
            alloc = k.alloc
            if self.k2:                                 # copy-on-resume: the resumed state is the base
                end, inner = alloc.state_blocks, alloc.base
                base = end if hit == k.tokens else inner
                other = inner if base is end else end
                self.pool.give_state(other)
                self._make_room(0, 1)
                alloc.base, alloc.state_blocks = base, self.pool.take_state()
                alloc.copy_pending = True
                if hit == k.inner:
                    self.stats["inner_hits"] += 1
            if len(alloc.kv_pages) > n_kv:                  # partial hit (simulator rule): the rest goes back
                self.pool.give_pages(alloc.kv_pages[n_kv:])
                del alloc.kv_pages[n_kv:]
            need = n_kv - len(alloc.kv_pages)
            self._make_room(need, 0)
            alloc.kv_pages += self.pool.take_pages(need)
        else:
            self._make_room(n_kv, take)
            alloc = self._alloc([], self.pool.take_state())
            if take == 2:                               # k2, a tier read follows: its base
                alloc.base, alloc.copy_pending = self.pool.take_state(), True
            alloc.kv_pages = self.pool.take_pages(n_kv)
        assert alloc.num_pages == charge or self.accounting == "tokens", (alloc.num_pages, charge)
        self.held[req.req_id] = alloc
        self._charge[req.req_id] = charge
        if self.lower is not None and req.session is not None:
            self.lower.admitted(req.session, req, hit)   # after this admission's evictions, as the simulator
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

    def _make_room(self, pages: int, states: int):
        """Evict kept sessions, least recently used first, until the hard charge fits and the pool can hand out
        `pages` KV pages and `states` states."""
        while self.kept and (self.hard + self.kept_charge + self.area_charge() > self.capacity
                             or not self.pool.can_take(pages, states)):
            key = next((x for x in self.kept if x not in self.pinned
                        and (self.lower is None or self.lower.evictable(x))), None)
            if key is None:
                break
            k = self.kept.pop(key)
            self._unkeep(k, evicted=True)
            self.evicted[key] = k.tokens
            self.pool.give(k.alloc)
            self.stats["evicted_sessions"] += 1
            self.stats["evicted_tokens"] += k.tokens
        assert self.pool.can_take(pages, states), (pages, states, self.pool.num_free, self.pool.free_runs())

    def release(self, req):
        """The request is done (its token is out). With `keep_sessions`, its prompt's KV and final KDA state stay as
        its session's kept context, with k2 also the state it resumed from (its base, at `req.cached`); returns its
        key. Otherwise its pages are freed."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        alloc.copy_pending = False
        if not (self.keep_sessions and req.session is not None):
            self.pool.give(alloc)
            return None
        inner = req.cached if alloc.base and req.cached > 0 else 0
        if alloc.base and not inner:                 # a base with nothing in it (its read was gone): not kept
            self.pool.give_state(alloc.base)
            alloc.base = []
        key = self.kept_key(req)
        old = self.kept.pop(key, None)               # the session's entry went to this request; a stray one goes
        if old is not None:
            self._unkeep(old)
            self.pool.give(old.alloc)
        content = prefix_content(req.token_ids, req.num_tokens) if req.token_ids is not None else None
        k = Kept(key, req.num_tokens, alloc, content, req.output_token, inner)
        self.kept[key] = k
        self.evicted.pop(key, None)
        self.kept_charge += self.kept_cost(k.tokens, k.states)
        if self.lower is not None:
            self.lower.kept(key)
        if self.content_keys and content is not None:
            self._index_points(key, content, (k.tokens, k.inner))
        self._make_room(0, 0)
        return key

    def drop(self, req):
        """Give back a held allocation without keeping anything (a cancelled request)."""
        alloc = self.held.pop(req.req_id)
        self.hard -= self._charge.pop(req.req_id)
        self.pool.give(alloc)
