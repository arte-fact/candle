Q3 — Flash-attention v2 (BR=4, LDS-tiled), 2026-04-11
====================================================

Goal
----
Rewrite the P2 BR=1 flash-attention kernel to use:
- BR=4 (four Q rows per block) so four wavefronts share a single
  LDS K/V tile, amortising K/V loads 4×.
- LDS K/V staging with BC chosen per-D to fit ≤32 KiB/block and
  leave room for 2 blocks/CU.
- D ∈ {64, 128, 256} to cover gemma4's head_dim=256 in addition to
  TinyLlama (64) and qwen-class (128).

Target: collapse the 10,816 rocblas_attn dispatches (627 ms) on
gemma4 into ~2,688 flash-attn calls (one per layer per forward) and
reclaim the ~15 % GPU-time share the Cijk_* family currently owns.

What landed
-----------
Infrastructure, default OFF:
- `candle-hip-kernels/src/flash_attn_v2.cu` — new kernel file.
  Templated on `<int D, int BR, int BC>` with three extern-C
  entry points:
  - `flash_attn_v2_fwd_d64_f32`  (BR=4, BC=64 → 32 KiB LDS)
  - `flash_attn_v2_fwd_d128_f32` (BR=4, BC=32 → 32 KiB LDS)
  - `flash_attn_v2_fwd_d256_f32` (BR=4, BC=16 → 32 KiB LDS)
  Each lane owns `D_PER_LANE = D / WARP_SIZE` elements of the D
  axis; the inner loop is a scalar dot product + 6-shuffle warp
  reduce + __expf online softmax.
- `candle-core/src/hip_backend/flash_attn.rs` — new launcher
  `flash_attn_v2_fused(q, k, v, mask, scale)`. Same additive-mask
  stride convention as v1. Gates on `L_q >= FLASH_V2_MIN_L_Q = 4`.
- `candle-hip-kernels/src/lib.rs` — new `Id::FlashAttnV2` module.
- `candle-core/src/hip_backend/mod.rs` — re-export.
- `candle-transformers/src/models/quantized_blocks/attention.rs`:
  - Added `CANDLE_FLASH_ATTN_V2_ENABLE` opt-in gate in
    `gqa_attention`. Forces `q`, `k`, `v`, `mask` contiguous before
    the call (KvCache narrow views are strided when the cache
    buffer is over-allocated).
  - Oracle regression test `hip_flash_attn_v2_matches_cpu_oracle`
    covers 11 shapes: D ∈ {64, 128, 256}, B ∈ {1, 2},
    L_q ∈ {4, 5, 7, 8, 16, 32}, L_k up to 64, four mask kinds
    (None, (1,1,1,Lk), (1,1,Lq,Lk), (B,1,Lq,Lk)). All pass
    max_abs < 2e-4.

Default OFF (same as v1)
------------------------
Both v1 and v2 are opt-in via environment variables:
- `CANDLE_FLASH_ATTN_V1_ENABLE=1` — BR=1 scalar (TinyLlama 17x
  regression, documented in BENCH-P2-FLASH-ATTN-v1).
- `CANDLE_FLASH_ATTN_V2_ENABLE=1` — BR=4 LDS-tiled (this doc).

Why v2 also loses
-----------------
**Measurement: v2 ENABLED vs rocBLAS (default) on 1000-tok prompts,
64-token decode, clean 3-run median:**

| Model        | rocBLAS pp | v2 pp | Δ     | rocBLAS tg | v2 tg | Δ    |
|--------------|-----------:|------:|------:|-----------:|------:|-----:|
| TinyLlama    |       3430 |   790 | −77 % |        167 |   168 | flat |
| gemma4-E4B   |        882 |   611 | −31 % |       35.3 |  35.8 | flat |
| qwen3.5-9B   |        447 |   397 | −11 % |       40.3 |  40.1 | flat |

Decodes are flat because `L_q=1 < 4` drops through to rocBLAS.
Prefills regress uniformly.

Per-call cost analysis (TinyLlama, D=64):
- Grid: ceil(1065 / 4) × 32 heads × 1 batch = 8,544 blocks per layer
- Per block: 4 warps × 1065 j-iterations × ~30 cycles/j = 128 k
  wave-cycles
- All layers: 8,544 × 128 k × 22 = 24 G wave-cycles
- 60 CUs × 4 SIMDs × (1 inst / 4 cycles) = 60 wave-insts/cycle
- At 1.5 GHz: 24 G / 60 / 1.5e9 = 267 ms of compute per prefill
- Measured: ~350 ms per v2 prefill, suggesting we're close to
  compute-bound at this design.

rocBLAS does 1 prefill in ~150 ms (from 3430 t/s × 1065 prompt
tokens). **rocBLAS is ~2.3× faster than my compute-bound estimate.**
That means rocBLAS achieves higher instructions-per-cycle per SIMD
than the scalar dot-product + warp-reduce design, via:

1. **Dense FMA unrolling inside a tile.** A GEMM tile computes
   (tile_m × tile_k) × (tile_k × tile_n) via a 2D loop with
   dozens of in-register accumulators. No per-k warp reduce.
2. **LDS double-buffering.** Load next tile while computing the
   current one. My kernel has `__syncthreads()` between chunks
   which serialises load + compute.
3. **Bank-conflict-free LDS patterns.** My cooperative load might
   still hit some bank contention across the 4 warps.
4. **Vectorized loads.** rocBLAS uses `v_load_dwordx4` style.
   My scalar `k_lds[j*D + lane]` reads are 4 bytes per lane.

The bigger issue: the **warp reduce per j** is a hard critical
path. It's 6 shuffles + 6 FMAs = ~24 cycles minimum with pipeline
latency, and every output depends on it sequentially. Even with
perfect occupancy this caps the per-(q, k) throughput at ~1/24 per
cycle per wave. Compare with a GEMM's 1 FMA per cycle per lane
(0.2 inst per k per m).

Path to a competitive v3
------------------------
Real flash-attention on scalar f32 gfx906 needs to avoid the
per-j warp reduce. Two options:

1. **Q-tile + K-tile GEMM-style inner block.**
   Instead of one Q row per warp, tile (BR_Q × BR_K) outputs in
   registers per warp. Each lane accumulates partial sums across
   multiple (q, k) pairs. Warp reduce happens once per tile, not
   once per k. This is what turbo's attention kernels do.

2. **Use rocBLAS for the matmuls and fuse only the softmax/scale.**
   The Q0c `masked_softmax_scale` kernel already does this for the
   between-matmul chain. The remaining lever is killing the `k.t()`
   contiguous() in `gqa_attention` (via rocBLAS GemmOp::Trans if
   Tensile picks a fast kernel for it — see the earlier note at
   `candle-transformers/src/models/quantized_blocks/attention.rs:
   gqa_attention` about the 7% regression that killed this idea
   last time). Maybe a different approach: pre-transpose K at
   KV-cache append time so it's already in the fast layout.

Option 2 is lower-risk and doesn't require a new kernel. The 7%
regression from the `.contiguous()` drop might have been an
artefact of a particular Tensile heuristic; worth re-measuring
now that gqa_attention is all-fused internally.

Status
------
- v2 kernel + launcher + oracle test: landed.
- Gate: default OFF (`CANDLE_FLASH_ATTN_V2_ENABLE` opt-in).
- Roadmap Q3: **parked.** Future work needs the Q-tile GEMM-style
  kernel or the KV-cache layout change described above.
- Correctness: 11/11 oracle test cases pass within 2e-4.
- Performance: not competitive; documented in this file.
