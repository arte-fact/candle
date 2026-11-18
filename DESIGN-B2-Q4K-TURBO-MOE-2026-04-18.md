# B2 design — Q4_K support in turbo_moe MMQ (physical gather-by-expert)

**Date**: 2026-04-18
**Target**: `indexed_moe_forward_q4k_q8_1` currently owns 81.6 % of Qwen3.6 prefill GPU time (10.4 s / 12.75 s). Turbo does the equivalent in ~0.36 s via batched `mul_mat_q<Q4_K, 64>` after physical token reorder. **Single remaining pp lever.**

## Situation

The candle codebase already has a **physical gather-by-expert MMQ path** (`indexed_moe_forward_turbo_moe` in `candle-core/src/quantized/hip.rs:1286`). It:
1. Calls `quantize_q8_1_mmq_gather` to build `expert_bounds` + `ids_dst` and physically sort activations by routed-expert.
2. Launches a per-expert batched MMQ kernel (`mul_mat_{q4_0,q4_1,q8_0}_turbo_moe_x{8,16,32,64}_*`).
3. Falls back to `indexed_moe_forward` (the per-token gemv path) when the dtype isn't one of the three supported.

**Hard-coded dtype gate at `hip.rs:1307-1312`**:
```rust
let (dtype_tag, mmq_y_rows): (&str, usize) = match w_dtype {
    GgmlDType::Q4_0 => ("q4_0", 128),
    GgmlDType::Q4_1 => ("q4_1", 128),
    GgmlDType::Q8_0 => ("q8_0", 64),
    _ => return Ok(None),
};
```

Q4_K (and Q6_K for lm_head) fall through to the slow indexed path. **All the machinery for the fast path already exists** — we just need one more dtype.

## Required building blocks (most already present)

| Piece | Status | Location |
|---|---|---|
| `load_tiles_q4_K` (cuh-style 16-bit unpacked tile) | ✅ exists in candle | `quantized.cu` — already used by per-token Q4_K MMQ (P2 kernel `mul_mat_q4_K_gfx906_v2f_tile*`) |
| `vec_dot_q4_K_q8_1_impl_mmq` | ✅ exists | `quantized.cu:2024` |
| `block_q4_K` struct | ✅ exists | shared utils |
| `MMQ_Y_Q4K` tile constant | ✅ can reuse P2's choice (tile_n=32 best) | `quantized.cu` |
| `moe_dtype_traits<MOE_Q4_K>` specialization | ❌ to add | `mmq_turbo.cu:1031` |
| `mul_mat_q4_K_turbo_moe_*` extern kernel entries | ❌ to add | `mmq_turbo.cu` (via existing macro) |
| `quantize_q8_1_mmq_gather` produces Q8_1 in correct layout | ✅ already works | `mmq_turbo.cu` — dtype-agnostic Y side |
| Rust match arm for `Q4K` | ❌ to add | `hip.rs:1307` |

## Complication: Q4_K superblock size is 256, not 32

`MMQ_TILE_NE_K = 32` currently. Q4_K is a 256-element superblock with 8 internal sub-blocks of 32 elems each, sharing one pair of scales (6-bit packed in 12 bytes) and one `dm` (dmin, dscale) half2.

Two options:
- **(a)** One K-tile per superblock (tile_ne_k=256). Doubles the LDS x-tile size (144 bytes × mmq_y rows). At mmq_y=64 and mmq_x=32, that's 64×256/8 × 4 + 64×8/4 × 4 = 8 KB + 512 B — OK, fits LDS.
- **(b)** Keep tile_ne_k=32 and load 1/8 of a superblock at a time. Means each k-tile needs to know which sub-block's scales to use. Matches what upstream llama.cpp does (see its `load_tiles_q4_K`, which generates per-sub-block scales inside the tile load). This is the chosen path in P2.

Recommend (b) to minimise divergence from the existing P2 kernel and the turbo upstream layout.

## Risks and unknowns

1. **x_sc tile size**: Q4_K needs `mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8` int-slots for packed scales (see upstream `MMQ_DP4A_TXS_Q4_K`). Must extend the `shared_buf` layout in `mul_mat_q_turbo_moe_impl` to allocate this third alias. Currently it has `tile_y + x_qs + x_df_or_dm`.

2. **Expert weight stride bytes**: Q4_K is 144 bytes per 256 elements = 0.5625 bytes/elem vs Q8_0's 34/32 = 1.0625. The `weight_expert_stride_bytes = (n*k)/qk * block_bytes` formula already handles this via `qk`, but needs `qk=256` (not 32) to be passed through correctly.

3. **Correctness regression risk on Q4_0/Q8_0** if shared-buf layout changes. Mitigation: make x_sc conditional on `Tr::has_scales_tile` (new traits field) so existing dtypes unaffected.

4. **dp4a vec_dot Q4_K register pressure**. The upstream impl uses `vec_dot_q4_K_q8_1_mma` for MFMA-capable (MI200+). gfx906 has no MFMA — we must use the `_dp4a` variant. Candle's existing `vec_dot_q4_K_q8_1_impl_mmq` (line 2024) is the dp4a form; can reuse directly.

5. **Q6_K (lm_head)**: 388 ms / 9 calls = 43 ms/call is the LOGIT matmul at end of forward. Not MoE. Out of scope for B2. Address separately (probably via dispatcher selecting the already-landed `mul_mat_q6_K_gfx906_*` warp-coop or a batched MMQ variant).

## Implementation phases

### B2.1 — Scaffold the Q4_K dtype in turbo_moe (~0.5 day)
- Add `MOE_Q4_K = 3` enum variant.
- Add `moe_dtype_traits<MOE_Q4_K>`: mmq_y=64 (conservative), x_qs_ints (same as Q8_0 path, doubled K), block_bytes=144, plus new `has_scales_tile=true`.
- Extend `shared_buf` layout in `mul_mat_q_turbo_moe_impl` to conditionally allocate x_sc.
- No kernel body changes yet — template instantiation fails to link.

**Build check**: template compiles but no kernel body is instantiated. Rust side unchanged.

### B2.2 — Q4_K load_tiles + vec_dot path in the template (~1 day)
- Inside the `if constexpr` ladders: for `DT == MOE_Q4_K`, call a new `load_tiles_q4_K_moe()` adapted from upstream mmq.cuh:2137 (per-sub-block scales into x_sc).
- Inner dp4a loop: call `vec_dot_q4_K_q8_1_impl_mmq` (candle's existing port at `quantized.cu:2024`), passing x_sc slot.
- Register macro `DEFINE_TURBO_MOE_KERNEL(Q4K, MOE_Q4_K, q4_K, 8, check)` × (mmq_x ∈ {8,16,32,64}) × (check ∈ {_check, ""}) = 8 extern kernels.

### B2.3 — Rust dispatch + correctness test (~0.5 day)
- Add `GgmlDType::Q4K => ("q4_K", 64)` in `hip.rs:1307` match.
- Write `candle-core/examples/hip_q4k_moe_test.rs`: synthesise (tokens=64, topk=4, n_experts=16, hidden=2048, ffn=512) inputs, run both `indexed_moe_forward` and the turbo_moe path, assert max_abs_rel < 3e-3.
- Run on a 4-expert toy MoE to verify expert routing arithmetic.

### B2.4 — Benchmark + ship (~0.5 day)
- Rerun the A1 profile on Qwen3.6 pp512 with and without `CANDLE_MMQ_TURBO_MOE=0` toggle.
- Expected: `indexed_moe_forward_q4k_q8_1` drops to near-zero, replaced by `mul_mat_q4_K_turbo_moe_x32_*` at ~0.3-0.5 s.
- Update `bench/qwen36_3way_findings.md` with post-B2 profile.
- Expected candle pp512 improvement: 178.93 → ~400-500 t/s if full 29× MoE delta closes. Honest floor: 250 t/s (2× current), the remaining 10.4 − 0.4 = 10 s being far from fully recovered on the first try.

## Anti-plan (what NOT to do)

- **Do not** rewrite `indexed_moe_forward` itself. The turbo_moe path makes it obsolete for MoE workloads where `avg_tokens_per_expert ≥ 4` (the gate condition already at line 1298). Keep indexed_moe as the decode-time fallback.
- **Do not** touch the gather kernel `quantize_q8_1_mmq_gather`. It's dtype-agnostic on the Y side.
- **Do not** add Q6_K in the same PR. Separate B3 task: lm_head usually doesn't benefit from the gather-by-expert path anyway (it's a dense matmul, not MoE).
- **Do not** bundle this with TP+EP port (Q3.6.7). TP+EP is a separate architectural axis.

## Effort & risk

- **Effort**: ~2-3 days including bench validation.
- **Risk**: medium. The kernel body is the sharpest part (x_sc allocation + per-sub-block scale fetch inside the hot K-loop). Fallback exists: `CANDLE_MMQ_TURBO_MOE=0` disables the new path if correctness regresses.
- **Measured anchor (single data point)**: 1.57× on Qwen3-Coder-Next-Q4_0 (qwen3next hybrid, 512 experts, topk=10). turbo_moe ON = 234.83 t/s, turbo_moe OFF = 149.18 t/s. **NOT a ceiling** — see below.
- **Retracted claim**: the earlier "HBM bandwidth caps at ~2×" was wrong on the numbers. The HBM floor for Qwen3.6 Q4_K pp512 MoE weight read is 81.5 ms (= 6 279 t/s) on 4×MI50 at 1 TB/s each — 35× faster than the observed 178 t/s. We are **not** bandwidth-bound.
- **Actual driver of the MMVQ→MMQ win**: arithmetic intensity. MMVQ does ~2 FLOP/byte (bandwidth-bound). MMQ with a 64×32 tile reuses each weight register across 32 activation columns, giving ~32× higher compute density on the matmul portion. On dp4a-bound gfx906 (26.5 TOPs int8), the matmul itself has up to ~10× headroom before hitting the compute ceiling.
- **Why Qwen3.6 might see MORE than 1.57×**: avg tokens/expert is 16 for Qwen3.6 (vs 10 for qwen3next). Better MMQ tile fill → closer to theoretical tile utilisation → likely *larger* speedup, not smaller. Direction is positive; magnitude is unknowable without porting and measuring.
- **Why the speedup will still be well below 10×**: non-MoE kernels (GDN step, attention, rmsnorm, quantize, copy) are ~18 % of current candle GPU time and unaffected by B2. Amdahl-limited. Rough upper envelope if the MoE portion collapses 10×: 12.75 s → (10.4/10 + 2.35) = 3.4 s, i.e. up to ~3.8× pp on the whole prefill. Realistic mid-point 2-3×.
- **Honest stance**: 1.57× is a floor suggested by one different-arch data point. True payoff must be measured, not extrapolated from the µs table or the Q4_0 A/B.

## Out of scope / follow-up

- **Q3.6-C attention parity**: defer per A1 findings.
- **Q3.6-B2-followup**: Q6_K turbo_moe for lm_head (separate task).
- **MXFP4 variants**: qwen3-coder-next uses MXFP4 shared expert; current path is the O2 MMVQ kernel. Not in B2 scope.
- **Fused gate_up turbo_moe**: turbo fuses gate+up into one matmul with 2×ffn output rows. Separate optimization after B2 lands.
