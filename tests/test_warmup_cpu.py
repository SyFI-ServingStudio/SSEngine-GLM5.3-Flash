"""CPU tests of what keeps Triton compiles out of a run: the g0 table width classes (ppimpl/model/metadata.py), the
served shapes warm-up runs (ppimpl/model/stage.py)."""

from types import SimpleNamespace as NS

from ppimpl.model.metadata import StageMetadata
from ppimpl.model.stage import StageModel

B = 8576
KPB = B // 64
CAP = 124 * KPB          # vLLM's width at max_model_len 1,048,576 (123 blocks, aligned to 128 tokens)


def width(blocks: int, cap=CAP) -> int:
    return StageMetadata.table_width(NS(kernel_per_block=KPB, table_cap=cap), blocks)


def test_table_width_is_a_power_of_2_of_blocks_capped():
    assert [width(b) // KPB for b in (1, 2, 3, 4, 5, 8, 9, 62, 64, 65, 123)] == [1, 2, 4, 4, 8, 8, 16, 64, 64, 124, 124]
    assert width(5, cap=None) == 8 * KPB
    assert len({width(b) for b in range(1, 124)}) == 8


def fake_stage(num_blocks=7143, max_tokens=8192, mml=1_048_576):
    return NS(kv=NS(block_tokens=B), num_blocks=num_blocks, max_tokens=max_tokens,
              vc=NS(model_config=NS(max_model_len=mml)))


def test_served_shapes_cover_every_table_width_and_decode_mix():
    m = fake_stage()
    shapes = [specs for _, specs in StageModel._warm_served_shapes(m)]
    decode_only = [s for s in shapes if len(s) == 1 and s[0][1] == 1]
    widths = {width(-(-(st + n) // B)) for st, n in (s[0] for s in decode_only)}
    assert widths == {width(b) for b in range(1, 124)}                       # all 8 classes
    assert {sum(n == 1 for _, n in s) for s in shapes if len(s) > 1} >= {0, 1, 2, 3}
    assert {n % 4 for s in shapes for st, n in s if len(s) == 1 and st == 8192} >= {2, 3}
    assert {len(s) for s in shapes} >= set(range(1, 10))
    for s in shapes:
        assert sum(n for _, n in s) <= m.max_tokens and all(st + n <= 1_048_576 for st, n in s)
        chunks = StageModel._warm_chunks(m, s)
        used = [b for c in chunks for b in c.kv_blocks + c.state_blocks]
        assert len(used) == len(set(used)) and min(used) >= 1 and max(used) < m.num_blocks


def test_served_shapes_skip_what_a_small_pool_cannot_hold():
    m = fake_stage(num_blocks=64)
    for _, s in StageModel._warm_served_shapes(m):
        assert sum(-(-(st + n) // B) + 6 for st, n in s) <= 63

