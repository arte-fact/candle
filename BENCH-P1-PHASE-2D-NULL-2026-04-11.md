# P1 Phase 2d — null result

**Date**: 2026-04-11
**Goal**: close the 6.3× per-call gap vs turbo on Q4_1 MMQ (15.8 ms vs 2.5 ms) via LDS staging + multi-warp blocks
**Result**: **null / regression on all three attempts**. The existing v2 row-per-thread framework is at its ceiling; closing the gap requires a structural rewrite (different work distribution), which is deferred.

## What was tried and what happened

**Baseline (post-P5)**: Q4_1 `mul_mat_q4_1_gfx906_v2_tile32`, 1 warp/block, 64×32 output tile, ~15.8 ms/call, **328 t/s prefill on qwen35-9B**.

### Attempt 1: 2-warp block with row split (no LDS)

- Block: 2 warps × 64 threads = 128 threads
- Tile: 128 rows × 32 cols (warp 0 → rows 0-63, warp 1 → rows 64-127)
- Grid halved (576 → 288 blocks); each block packs 2 SIMDs instead of 1

**Measured**: 329.27 t/s (v2) vs 328.48 t/s (v3) — **0.2 % diff, within noise**.

**What this tells us**: block scheduling / CU dispatch overhead is **not** the bottleneck. The 1-warp-per-block shape is already fine; packing 2 warps per block gives no benefit on its own.

### Attempt 2: 2-warp block with column split + X-tile LDS staging

- Block: 2 warps × 64 threads
- Tile: 64 rows × 32 cols (warp 0 → cols 0-15, warp 1 → cols 16-31)
- Both warps use the SAME 64 rows of X; warp 0 loads X into LDS once per K block via `__syncthreads()`, both warps read from LDS in the compute phase
- LDS footprint: ~1.3 KB/block (64 rows × 20 bytes of Q4_1 block header)

**Measured**: **60 t/s** at tile16, **40 t/s** at tile32, **28 t/s** at tile64 — **5–12× slower than v2**.

**What this tells us**:
- `__syncthreads()` barriers are expensive: 2 per K block × 128 K blocks × 288 blocks = ~73 k barriers per matmul.
- During the load phase warp 1 idles (single warp loading), halving SIMD utilisation for that window.
- More importantly: **LDS added no benefit because each row is only used by one thread.** In the row-per-thread distribution, LDS can only cache data a thread was going to load anyway. Cross-warp X reuse would matter only if multiple threads needed the same X row — which they don't in this work layout.
- gfx906's L1 already handles the cross-block X reuse that the LDS staging was supposed to amortise (blocks on the same CU share L1).

### Attempt 3: explicit `int4` vectorised X load

- One-line change: `const int4 q4_vec = *((const int4 *) bx->qs)` instead of 4 sequential `int` loads
- Forces a single `flat_load_dwordx4` instruction for the 16-byte X quant block

**Measured**: **288–322 t/s** over 3 runs (first run a thermal outlier; runs 2–3 within noise of the 328 baseline).

**What this tells us**: the compiler was **already emitting `dwordx4`** for the `__restrict__` + `#pragma unroll` scalar loop. No new instructions to issue.

## Why the v2 framework is at its ceiling

The 6.3× gap to turbo is structural. It lives in levers that are incompatible with row-per-thread work distribution:

1. **Cross-warp sharing of both X and Y tiles.** Turbo's kernel has multiple warps *per block* that collaborate on tile loads and **each thread owns a small 2-D output sub-tile** (e.g., 2 rows × 4 cols). Every thread needs to load data from multiple rows, which is where LDS amortises loads. Our row-per-thread model loads each row exactly once regardless of LDS.
2. **Y-tile staging in a packed MMQ layout** (`block_q8_1_mmq`). Turbo repacks Y during quantisation so tile loads into LDS are coalesced; we use standard `block_q8_1` which has a per-block header that doesn't vectorise as cleanly.
3. **Async double-buffered K iteration** where warps prefetch the next K tile while computing on the current one. Requires the tile-LDS model to work.
4. **DPP / matrix-cooperative instructions** (`v_dot*` variants with cross-lane ops). Not available on gfx906 in the way CDNA/RDNA have them, but turbo's gfx906 path still uses the dp4a-only code with wider tiles and LDS.

Implementing any of (1) requires rewriting the kernel from scratch with a different output decomposition — closer to a classic GPU GEMM kernel. That's a **1000+ LOC effort** (matching roughly the scope of `load_tiles_q4_1` + `vec_dot_q4_1_q8_1_dp4a` + `mul_mat_q_process_tile` from turbo's mmq.cuh, minus the stream-K/fixup machinery).

## Recommendation

**Defer Phase 2d indefinitely** unless the end-to-end prefill gap becomes the critical path. The current state is:

| Metric | Pre-P0 | Post-P5 | Turbo |
|---|---:|---:|---:|
| Prefill (pp1149) | 78.60 t/s | **~328 t/s** (73 % of turbo) | 450 t/s |
| Total GPU time | 17.4 s | **7.1 s** | 2.7 s |
| Gap vs turbo | 5.7× | **2.6×** | 1× |

**Next move**: **P4 — fused norm+silu+quantize**. Pointwise ops are now at 913 ms vs turbo's 134 ms = 6.8× ratio — the biggest disproportionate category left. Expected ~750 ms savings for ~300 LOC of kernel work. That's ~10 % of our total GPU time versus Phase 2d's uncertain multi-day rewrite.

## Files touched in this investigation

All reverted; working tree is back to commit `f7124bc9` (post-P5). This doc is the only artifact.
