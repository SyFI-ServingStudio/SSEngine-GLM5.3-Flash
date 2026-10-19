"""pp_patch.split_indexer_prefill_chunks: the fork's indexer prefill split with query rows cut in multiples of 512."""

from ssengine.model.pp_patch import split_indexer_prefill_chunks

MB = 1 << 20


def rows(chunks):
    return [(c[1].stop - c[1].start) for c in chunks]


def test_512k_context_1k_append_is_one_call_and_never_leaves_a_sliver():
    # 525,312 tokens: 131,328 pools per row
    assert rows(split_indexer_prefill_chunks([131328], [1024], 1 << 30)) == [1024]
    # under the fork's 512 MiB budget: 2 x 512 rows, not 1,022 + 2
    assert rows(split_indexer_prefill_chunks([131328], [1024], 1 << 30, 512 * MB)) == [512, 512]


def test_512k_context_8k_append_in_512_row_multiples():
    c = split_indexer_prefill_chunks([133120], [8192], 1 << 30)
    assert rows(c) == [3584, 3584, 1024]
    assert c[0][1] == slice(0, 3584) and c[-1][1] == slice(7168, 8192)
    assert all(x[0] == slice(0, 1) for x in c)


def test_small_requests_are_packed_as_the_fork_does():
    # three short prompts fit one call; request_offset shifts the request slice
    c = split_indexer_prefill_chunks([2048, 4096, 1024], [2048, 4096, 1024], 1 << 30, request_offset=2)
    assert c == [(slice(2, 5), slice(0, 7168))]


def test_budget_below_one_step_keeps_the_forks_cut():
    # fewer than 512 rows fit: rows as the budget allows, unaligned
    assert rows(split_indexer_prefill_chunks([1 << 20], [1000], 1 << 30, 1 << 30)) == [256, 256, 256, 232]
