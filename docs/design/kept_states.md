# KDA states: compact allocation and two kept states per context (k2)

GLM-5.3-Flash has 34 KDA (linear attention) layers. A cached context can be resumed only where a KDA recurrent state
was saved. This page covers two things: how a state is laid out in HBM (compact), and which states a kept context
keeps (k2). Tiers below HBM: [tiers.md](tiers.md).

## Compact state allocation

**Before.** vLLM's hybrid layout gave every request 6 whole 8,576-token blocks for its state: 5 KDA groups plus the
kpool tail. Each block is 67 pages of 128 tokens, so a state cost 402 pages, about 51k tokens of KV. The real state
is 34 x 4.34 MB plus 11 kpool tails of 2 KiB, about 148 MB over 4 GPUs.

**Page geometry.** A page is 128 tokens of every KV tensor on a stage. vLLM's `allocate_kv_cache` puts all of a
stage's layers in one buffer: layer l of block b starts at `l * layer_bytes + b * block_stride`, so within a block a
layer's 67 pages are contiguous (65,536 B per MLA page, 4,224 B per indexer page on this model), and blocks are
`block_stride` apart. So the first c pages of a block give, in every KV tensor of the stage, one contiguous region of
c x page_bytes bytes. State pieces are placed into these regions:

- the recurrent state and the conv state of each KDA layer;
- the kpool tail of each DSA layer.

**Runs.** A state is R runs. A run is the first c pages of a block, named by the block id; the other 67 - c pages of
the block are ordinary KV pages. On each stage, every piece has a fixed place: (run, region tensor, byte offset). Its
view is `as_strided(rows = blocks, row stride = block_stride)` from the region's start plus the offset, so row b is
the piece of the state whose run is block b. A request's `state_blocks` are its runs' blocks. The kernels already
address states as `pool[slot]` with any row stride (`kda_state`, `kda_conv` and the kpool tail seed, whose
`TAIL_BLOCK_ELEMS = stride(0)`). Only the views and the slot ids change. A block holds at most one run: with the
blocks strided, two runs of one block would need two row strides.

**Rules for placement.**

- The tails of a stage share one run: the tail metadata has one block table per microbatch.
- A KDA layer's conv state and recurrent state are in the same run: they share the sequence table.
- Within those rules, each piece goes, in layer order, into the first run with room, in the region of that run with the least room left that holds it (best fit). Pieces are 16-byte aligned.
- The block ids are global (the same id on all 4 stages), so R is the maximum over stages.
- c runs from 1 to 67. The geometry chosen is the one with the fewest pages per state (c x R), ties going to the
  smaller R. Every stage computes it from all stages' layer specs, so all stages agree. The workers report (c, R) in
  `Ready` and the controller checks that they match.

**This model.** A 4 MiB recurrent state fills exactly 64 MLA pages, so c >= 64. Stage 1 has 2 DSA layers and 9 KDA
layers, so it needs 5 runs. The result is c = 64, R = 5: 320 pages per state instead of 402. Stage 1 sets the size:
its real bytes alone (9 x 4.34 MB over 2 x 69,760 B per page) need 280 pages. Any layout with global page ids pays
for stage 1 on every stage.

**Pool.** `PagePool` hands out a run as a block whose first c pages are free:

- First from partly used blocks, fullest first; then from whole free blocks, oldest first.
- The rest of a block that holds runs (here 3 pages) is ordinary KV pages.
- KV pages are taken after the state, so they never take the last free slot that a state needs.

A request is charged kv_pages + states x c x R pages. `fits` and `_make_room` check free runs where they checked
whole blocks.

**Tiers.** The state ranges are the piece views (stride = block_stride, group = run). Placement and the movers are
unchanged: a state's device address is `base + run_block x stride`. Slot sizes are unchanged too: tiers already
store real bytes. `hbm_busy` takes a state as the pages of its runs.

## k2: two kept states per context

**Problem.** A kept context has one KDA state, at its end (k1). A prompt that rewrites a few hundred tokens before
that end (AgentX re-renders the previous turn's tail) misses the whole context.

**What is kept.** A kept context keeps two states:

- **end**: the state at its prompt end (as before);
- **inner**: the state its round resumed from, which is the previous round's prompt end. It is kept only when the
  round had a hit.

Both are on the same lineage key, and the KV of [0, end) holds both.

**Copy-on-resume.** KDA updates a state in place, so a round that resumes from a kept state X never computes on X:

- Admission takes a fresh working state W. The request's allocation holds X as its **base** and W as its state.
- The first chunk carries `state_from = X`. Each stage, in stream order before that chunk, copies X's pieces into W
  (`StageModel.copy_states`: one `_foreach_copy_` of 9 to 21 pieces, about 37 MB per stage).
- The prefill computes on W. When the round ends, the new kept context is end = W at its prompt end, inner = X at
  the point it resumed from.
- A round that resumes from a tier reads the tier's state into X (the read target) and then copies the same way.

**Matching.** `find_session` indexes every kept state point (end and inner, in HBM and in tiers) by the chained hash
of its whole 64-token blocks, and takes the deepest point the prompt extends with at least one new token. The other
state of the context it resumes from is given back at admission:

- resumed at end: inner is freed, and the old end becomes the new inner;
- resumed at inner: end is freed, and the new inner stays at the same point.

A request that resumes holds 2 states while it runs. A first round holds 1. With `kept_states: 1` the engine runs
exactly as k1 does: in place, no copy.

**Eviction and supersession.**

- An HBM eviction frees the context whole, both states. A tier write moves the KV plus both states. A tier entry
  keeps both states, so a later round can resume from either. A tier evicts an entry whole.
- A tier read moves only the state the round resumes from. In exclusive DRAM, the other state goes with the
  released entry.
- When a round of the lineage is admitted (`drop_stale`), a tier entry stays only if the round resumes from that
  entry's end. Then its KV stays a prefix of the new context, and the new context's write-back keeps its blocks.
  Otherwise the entry is dropped. The new context then keeps the states that matter for its next round: its end, and
  the point it resumed from.
- A context replaced on its key (a stray second round of the lineage) is freed whole.

**Cost.**

- HBM: one more state per kept context that had a hit, and per running resumed request (320 pages).
- Tiers: one more state per entry, 3 state slots of 12.86 MB per GPU.
- Copy: about 37 MB D2D per stage per resumed round, at HBM bandwidth (tens of µs).

In the simulator, k2 under st10 / DRAM 150 GB cut extra recompute from 13-25% to 7-10% and raised r/s by 6-14%
(ServingStudioSim pp-state-bytes, logs/20261106_0_state_bytes).

**Replay.** Replay reproduces k1 targets and keeps `kept_states: 1`. Its token ledger is unchanged: a state still
costs the simulator's 51,456 tokens there.
