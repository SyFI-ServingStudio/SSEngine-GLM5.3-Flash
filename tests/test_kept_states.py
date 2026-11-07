"""Compact KDA states and k2 (docs/design/kept_states.md): the state geometry, the state views and their copy, two kept
states per context, matching the deepest kept state, copy-on-resume and the tiers' two states."""

import numpy as np
import pytest
import torch

from ssengine.config import BLOCK_PAGES, PAGE_TOKENS, TierConfig
from ssengine.engine.kv import KvManager, default_state
from ssengine.engine.scheduler import Request
from ssengine.kvpool.manager import KVViews, TierManager
from ssengine.model import state_layout


def ids_of(seed: int, n: int) -> list[int]:
    return np.random.default_rng(seed).integers(1, 150_000, n).tolist()


def check_pool(kv):
    allocs = list(kv.held.values()) + [k.alloc for k in kv.kept.values()] + list(kv.reads.values())
    used = sum(a.num_pages for a in allocs)
    assert used + kv.pool.num_free == kv.num_blocks * BLOCK_PAGES
    assert kv.hard + kv.kept_charge + kv.area_held == used
    pages = [p for a in allocs for p in a.kv_pages] + [p for a in allocs for p in kv.pool.run_pages(a.state_blocks
                                                                                                    + a.base)]
    assert len(set(pages)) == len(pages) and min(pages, default=BLOCK_PAGES) >= BLOCK_PAGES
    assert all(0 < b <= kv.pool.num_blocks for a in allocs for b in a.state_blocks + a.base)


def serve(kv, ids, name):
    """Find, admit, finish one round (the server's path); returns (the request, its hit)."""
    r = Request(name, len(ids), session=kv.find_session(ids), token_ids=ids)
    assert kv.fits(r)
    r.cached = kv.allocate(r)
    r.alloc = kv.held[name]
    alloc = r.alloc
    kv.release(r)
    return r, alloc


# --- geometry and views ---------------------------------------------------------------------------------------------

def test_glm53_geometry_is_five_runs_of_64_pages():
    """A 4 MiB recurrent state is exactly 64 MLA pages (512 B x 128 tokens); stage 1 (2 DSA, 9 KDA layers) needs 5 runs,
    the others 3; tails share one run; a layer's conv and recurrent state share a run."""
    g = state_layout.glm53_geometry()
    assert (g.run_pages, g.runs, g.pages) == (64, 5, 320) and default_state().pages == 320
    assert [len({p.run for p in places.values()}) for places in g.places] == [3, 5, 3, 3]
    for places in g.places:
        assert len({p.run for n, p in places.items() if n.endswith(".tail")}) == 1
        for n in places:
            if n.endswith("#0"):
                assert places[n].run == places[n[:-1] + "1"].run
    # within a (run, region) no two pieces overlap and all fit the region
    st = state_layout.glm53_pieces()
    for s, places in enumerate(g.places):
        size = {f"{n}#0": c for n, c, _ in st[s].kda} | {f"{n}#1": r for n, _, r in st[s].kda} | dict(st[s].tails)
        cap = {n: g.run_pages * pb for n, pb in st[s].regions}
        spans = {}
        for n, p in places.items():
            assert p.offset % state_layout.ALIGN == 0 and p.offset + size[n] <= cap[p.region]
            spans.setdefault((p.run, p.region), []).append((p.offset, p.offset + size[n]))
        for v in spans.values():
            v.sort()
            assert all(a[1] <= b[0] for a, b in zip(v, v[1:]))


def test_geometry_picks_the_fewest_pages_and_needs_every_piece_to_fit():
    one = state_layout.StagePieces((("m", 1000), ("i", 100)), (("k", 50, 5000),), (("t", 20),))
    g = state_layout.geometry([one])
    assert (g.run_pages, g.runs) == (5, 1)                  # 5,000 B fits 5 pages of the 1,000 B region
    with pytest.raises(ValueError):
        state_layout.geometry([state_layout.StagePieces((("m", 10),), (("k", 1, 10_000),), ())])


def _stage_views(stage: int, num_blocks: int, device="cpu"):
    """CPU stand-ins for a stage's MLA / indexer tensors laid out as vLLM's `allocate_kv_cache` does (one backing
    buffer; layer l of block b at l x layer_bytes + b x block_stride) and its state views."""
    g = state_layout.glm53_geometry()
    st = state_layout.glm53_pieces()[stage]
    layer = [BLOCK_PAGES * pb for _, pb in st.regions]
    block_stride = sum(layer)
    buf = torch.zeros(num_blocks * block_stride, dtype=torch.uint8, device=device)
    regions, off = {}, 0
    for (n, pb), lb in zip(st.regions, layer):
        regions[n] = (buf[off:], block_stride)
        off += lb
    pieces = {}
    for n, c, r in st.kda:
        pieces[f"{n}#0"] = ((3, 3 * 64 * 128), torch.bfloat16)
        pieces[f"{n}#1"] = ((64, 128, 128), torch.float32)
    for n, _ in st.tails:
        pieces[n] = ((2, 4, 128), torch.bfloat16)
    sv = state_layout.StateViews(regions, num_blocks, g, g.places[stage], pieces)
    return sv, regions, buf, block_stride


@pytest.mark.parametrize("stage", [0, 1])
def test_state_views_address_blocks_and_copy_a_whole_state(stage):
    """Each piece's row b is at its region's byte b x block_stride + offset, inside the region's first c pages of block
    b; the copy (copy-on-resume) makes the state at runs D equal to S's, piece by piece, leaves S, and writes only
    D's run pages."""
    nb = 12
    sv, regions, buf, bs = _stage_views(stage, nb)
    st = state_layout.glm53_pieces()[stage]
    pb = dict(st.regions)
    c = sv.geo.run_pages
    for name, v in sv.views.items():
        p = sv.geo.places[stage][name]
        assert v.stride(0) * v.element_size() == bs and v.shape[0] == nb
        assert v.data_ptr() - regions[p.region][0].data_ptr() == p.offset
        assert p.offset + v[0].numel() * v.element_size() <= c * pb[p.region]
    src, dst = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    gen = torch.Generator().manual_seed(stage)
    for name, v in sv.views.items():
        v[src[sv.run_of[name]]].copy_(torch.randn(v.shape[1:], generator=gen).to(v.dtype))
    before = buf.clone()
    sv.copy([(src, dst)])
    for name, v in sv.views.items():
        r = sv.run_of[name]
        assert torch.equal(v[dst[r]], v[src[r]]) and v[src[r]].abs().sum() > 0
    # only bytes inside the first c pages of the destination blocks' regions changed
    allowed = torch.zeros_like(buf, dtype=torch.bool)
    for n, (t, _) in regions.items():
        lo = t.data_ptr() - buf.data_ptr()
        for b in dst:
            allowed[lo + b * bs:lo + b * bs + c * pb[n]] = True
    assert not ((buf != before) & ~allowed).any()


def test_specs_give_the_same_geometry_as_the_constants():
    """The workers derive the geometry from vLLM's specs (`kv_tensors.state_geometry`); the fake model and the KV
    manager's default use the constants: they agree."""
    from test_model_cpu import _specs_for_stage

    from ssengine.model import kv_tensors

    for stage in range(4):
        g = kv_tensors.state_geometry(_specs_for_stage(stage))
        c = state_layout.glm53_geometry()
        assert (g.run_pages, g.runs) == (c.run_pages, c.runs)
        assert sorted(p.run for p in g.places[stage].values()) == sorted(p.run for p in c.places[stage].values())


# --- k2 in the KV manager --------------------------------------------------------------------------------------------

def k2(blocks=200, **kw):
    return KvManager(blocks, keep_sessions=True, content_keys=True, kept_states=2, **kw)


def test_a_resumed_round_keeps_the_state_it_resumed_from_as_inner():
    kv = k2()
    a = ids_of(0, 5000)
    r0, al0 = serve(kv, a, "r0")
    assert r0.cached == 0 and not al0.base and kv.kept[r0.session].inner == 0     # a first round: one state
    end0 = list(al0.state_blocks)
    b = a + ids_of(1, 3000)
    r1 = Request("r1", len(b), session=kv.find_session(b), token_ids=b)
    assert r1.session == r0.session and kv.peek_hit(r1) == 5000
    r1.cached = kv.allocate(r1)
    al1 = kv.held["r1"]
    assert r1.cached == 5000 and al1.base == end0 and al1.state_blocks != end0 and al1.copy_pending
    assert kv._charge["r1"] == al1.num_pages == 63 + 2 * 320                   # two states while it runs
    check_pool(kv)
    kv.release(r1)
    k = kv.kept[r1.session]
    assert (k.tokens, k.inner) == (8000, 5000) and k.alloc.base == end0 and not k.alloc.copy_pending
    assert kv.kept_charge == 63 + 2 * 320
    check_pool(kv)


def test_diverging_after_the_previous_prompt_end_hits_there():
    """The k2 case: the prompt diverges before the kept context's end but after the previous round's prompt end
    (AgentX re-renders the tail): it resumes from the inner state, takes it as its base (copy-on-resume) and the end
    state goes back to the pool."""
    kv = k2()
    a = ids_of(0, 5000)
    serve(kv, a, "r0")
    b = a + ids_of(1, 3000)
    r1, al1 = serve(kv, b, "r1")
    key = r1.session
    end1, inner1 = list(al1.state_blocks), list(al1.base)
    assert kv.kept[key].alloc.state_blocks == end1
    c = b[:7000] + ids_of(2, 2500)                         # diverges at 7,000: inside (5,000, 8,000)
    assert kv.find_session(c) == key
    r2 = Request("r2", len(c), session=key, token_ids=c)
    assert kv.peek_hit(r2) == 5000 and kv.shared_prefix(r2) == 7000
    free_before = kv.pool.num_free
    r2.cached = kv.allocate(r2)
    al2 = kv.held["r2"]
    assert r2.cached == 5000 and r2.admit_info["hit_kind"] == "inner" and kv.stats["inner_hits"] == 1
    assert al2.base == inner1 and al2.copy_pending and set(al2.state_blocks).isdisjoint(inner1)
    assert kv.pool.num_free == free_before - (len(al2.kv_pages) - 63)   # end1 back, a fresh state taken (end1's)
    check_pool(kv)
    kv.release(r2)
    k = kv.kept[key]
    assert (k.tokens, k.inner) == (9500, 5000) and k.alloc.base == inner1           # still the same inner point
    check_pool(kv)
    # a prompt that extends the whole new context resumes from its end, the inner state goes back
    d = c + ids_of(3, 100)
    r3, al3 = serve(kv, d, "r3")
    assert r3.cached == 9500 and r3.admit_info["hit_kind"] == "full"
    assert (kv.kept[key].tokens, kv.kept[key].inner) == (9600, 9500)
    check_pool(kv)


def test_diverging_before_the_inner_state_misses_and_keeps_the_context():
    kv = k2()
    a = ids_of(0, 5000)
    serve(kv, a, "r0")
    b = a + ids_of(1, 3000)
    r1, _ = serve(kv, b, "r1")
    c = b[:4000] + ids_of(2, 3000)                         # diverges before the inner point (5,000)
    s = kv.find_session(c)
    assert s != r1.session
    r2, al2 = serve(kv, c, "r2")
    assert r2.cached == 0 and not al2.base and r1.session in kv.kept               # a sibling: both kept
    check_pool(kv)


def test_a_prompt_equal_to_the_inner_point_does_not_take_the_context():
    """An inner point resumes only with a token after it: a prompt that ends exactly there gets a new key and leaves
    the context kept (the end point instead is a repeat)."""
    kv = k2()
    a = ids_of(0, 5000)
    serve(kv, a, "r0")
    r1, _ = serve(kv, a + ids_of(1, 3000), "r1")
    assert kv.find_session(list(a)) != r1.session


def test_deepest_point_wins_across_contexts():
    """Two lineages over the same first tokens: the prompt takes the context whose kept point (end or inner) it
    extends the furthest."""
    kv = k2()
    a = ids_of(0, 5000)
    serve(kv, a, "x0")
    rx, _ = serve(kv, a + ids_of(1, 3000), "x1")           # x: end 8,000, inner 5,000
    ry, _ = serve(kv, a[:2000] + ids_of(2, 4000), "y0")    # y: end 6,000 (shares 2,000 with x)
    c = a + ids_of(1, 3000)[:2500] + ids_of(3, 50)          # extends x's inner (5,000), not its end
    assert kv.find_session(c) == rx.session
    d = a[:2000] + ids_of(2, 4000) + ids_of(4, 10)          # extends y's end (6,000)
    assert kv.find_session(d) == ry.session


def test_eviction_frees_both_states_and_k1_runs_in_place():
    kv = k2(blocks=30)                                      # 2,010 pages
    a = ids_of(0, 5000)
    serve(kv, a, "r0")
    r1, _ = serve(kv, a + ids_of(1, 3000), "r1")            # kept: 63 + 640 pages
    big = ids_of(9, 130_000)                                # 1,016 + 320: evicts r1's context, both states
    r2, _ = serve(kv, big, "r2")
    assert r1.session not in kv.kept and kv.stats["evicted_sessions"] == 1
    check_pool(kv)
    one = KvManager(200, keep_sessions=True, content_keys=True)          # k1: no base, no copy, the same state
    _, al0 = serve(one, a, "r0")
    st = list(al0.state_blocks)
    r1 = Request("r1", 8000, session=one.find_session(a + ids_of(1, 3000)), token_ids=a + ids_of(1, 3000))
    r1.cached = one.allocate(r1)
    assert one.held["r1"].state_blocks == st and not one.held["r1"].base and not one.held["r1"].copy_pending


def test_random_k2_sessions_keep_the_charge_equal_to_the_pages_held():
    """Random lineages that extend, rewrite their tail (inner hits) or branch, on a small pool: every admission `fits`
    allows succeeds and the charge stays exactly the pages held."""
    rng = np.random.default_rng(0)
    kv = k2(blocks=60)
    ctx: dict[int, list[int]] = {}
    running = []
    for step in range(500):
        if running and rng.random() < 0.5:
            r = running.pop(int(rng.integers(len(running))))
            kv.release(r)
            ctx[r.lineage] = r.token_ids
        else:
            lin = int(rng.integers(12))
            if any(r.lineage == lin for r in running):
                continue
            prev = ctx.get(lin, [])
            cut = len(prev) - int(rng.integers(0, 400)) if prev and rng.random() < 0.4 else len(prev)
            ids = prev[:max(cut, 0)] + rng.integers(1, 150_000, int(rng.integers(1, 6000))).tolist()
            r = Request(f"r{step}", len(ids), session=kv.find_session(ids), token_ids=ids)
            r.lineage = lin
            if kv.fits(r):
                r.cached = kv.allocate(r)
                r.alloc = kv.held[r.req_id]
                running.append(r)
        check_pool(kv)
        assert kv.hard + kv.kept_charge <= kv.capacity
    assert kv.stats["inner_hits"] > 5 and kv.stats["hits"] > 50


# --- the tiers keep both states, a read moves one ----------------------------------------------------------------

def test_tier_entry_keeps_both_states_and_a_read_moves_the_one_resumed_from():
    S = 3
    m = TierManager(TierConfig(dram_gb=10.0, ssd_gb=None, slot_bytes=100_000_000, state_slots=S))
    m.run_pages = 64
    content = np.asarray(ids_of(0, 20_000), dtype=np.int32)
    R = 5
    end, inner = list(range(1, 6)), list(range(6, 11))
    views = KVViews(list(range(1000, 1000 + 157)), end + inner, 20_000, inner=12_000)
    ops = m.prepare_write_back("k", 20_000, content, views, 12_000)
    w = [o for o in ops if o.op == "write"][0]
    e = m.tiers[0].entries["k"]
    assert e.inner == 12_000 and len(e.state) == 2 * S and w.state_slots == e.state
    assert w.state_blocks == [end + inner]
    assert m.tiers[0].used == m.tiers[0].charge(20_000, 2)
    # a prompt diverging between the two points resumes from the inner state; the read moves only that state
    p = Request("p", 18_000, token_ids=content[:15_000].tolist() + ids_of(5, 3000))
    hit = m.lookup("k", p)
    assert hit.tokens == 12_000
    ops, _ = m.prepare_load("k", hit, KVViews(list(range(2000, 2094)), [20], 12_000, 0), 0)
    rd = [o for o in ops if o.op == "read"][0]
    assert rd.state_slots == e.state[S:] and rd.token_end == 12_000
    q = Request("q", 21_000, token_ids=content.tolist() + ids_of(6, 1000))
    assert m.lookup("k", q).tokens == 20_000
    ops, _ = m.prepare_load("k", m.lookup("k", q), KVViews([], [20], 20_000, 0), 0)
    assert [o for o in ops if o.op == "read"][0].state_slots == e.state[:S]
    # HBM pages a write reads: the KV pages and every page of both states' runs
    busy = m._touches(w)[1]
    first = [b * BLOCK_PAGES for b in end + inner]
    assert ("hbm", first[0] + 63) in busy and ("hbm", first[9] + 63) in busy and ("hbm", first[0] + 64) not in busy


def test_placement_moves_k_states_state_by_state():
    from ssengine.kvpool.layout import ViewRange
    from ssengine.kvpool.placement import BalancedPlacement

    pb = 65_536
    ranges = [ViewRange(0, "mla", "kv", 1 << 30, BLOCK_PAGES * pb, BLOCK_PAGES * pb),
              ViewRange(0, "a#1", "state", (1 << 30) + 0, BLOCK_PAGES * pb, 64 * pb, group=0),
              ViewRange(0, "b#1", "state", (1 << 30) + 0, BLOCK_PAGES * pb, 64 * pb, group=1)]
    P = BalancedPlacement(ranges, 4, 0)
    S = P.state_slots
    one = P.pieces([], [10, 11], [], list(range(S)), 0, 0, 0)
    two = P.pieces([], [10, 11, 20, 21], [],
                   list(range(2 * S)), 0, 0, 0)
    n = len(one[0])
    assert len(two[0]) == 2 * n and (two[0][:n] == one[0]).all() and (two[1][:n] == one[1]).all()
    assert (two[0][n:] == one[0] + 10 * BLOCK_PAGES * pb).all()          # the second state's runs
    assert (two[1][n:] == one[1] + S * P.slot_bytes).all()               # in the next S slots
    assert PAGE_TOKENS == 128
