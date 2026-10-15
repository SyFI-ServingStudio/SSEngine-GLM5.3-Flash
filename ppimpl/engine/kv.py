"""The block pool: one block-id space for all stages (ppimpl/runtime/README.md, "Block-id space").

Phase A: a request holds ceil(L / 8576) KV blocks and 6 state blocks from admission to completion; admission is in
arrival order (a request that does not fit holds back the ones behind it); nothing is kept after completion.
Phase B adds the exact prefix cache (shared, ref-counted KV blocks + kept sessions with LRU eviction) on top of
`BlockPool`, behind the same `admit` / `release` calls.
"""

from collections import deque
from dataclasses import dataclass

from ppimpl.config import BLOCK_TOKENS, NULL_BLOCK, STATE_BLOCKS


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


class BlockPool:
    """Free list over block ids 1..num_blocks (0 is the null block)."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = deque(range(1, num_blocks + 1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def take(self, n: int) -> list[int]:
        assert n <= len(self._free)
        return [self._free.popleft() for _ in range(n)]

    def give(self, blocks: list[int]):
        assert NULL_BLOCK not in blocks
        self._free.extend(blocks)


@dataclass
class Allocation:
    kv_blocks: list[int]
    state_blocks: list[int]


class KvManager:
    """Per-request accounting over the pool, and arrival-order admission."""

    def __init__(self, num_blocks: int):
        self.pool = BlockPool(num_blocks)
        self.held: dict[str, Allocation] = {}

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    def fits_ever(self, prompt_tokens: int) -> bool:
        return blocks_for(prompt_tokens) <= self.pool.num_blocks

    def admit(self, req_id: str, prompt_tokens: int) -> Allocation | None:
        """Allocate the whole footprint, or None if it does not fit now."""
        need = blocks_for(prompt_tokens)
        if need > self.pool.num_free:
            return None
        alloc = Allocation(self.pool.take(kv_blocks_for(prompt_tokens)), self.pool.take(STATE_BLOCKS))
        self.held[req_id] = alloc
        return alloc

    def release(self, req_id: str):
        alloc = self.held.pop(req_id)
        self.pool.give(alloc.kv_blocks)
        self.pool.give(alloc.state_blocks)
