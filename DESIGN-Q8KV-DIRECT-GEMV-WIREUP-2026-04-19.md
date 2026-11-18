# Direct Q8-KV GEMV wire-up into StandardAttention

**Status**: P1.1 plan review — partially verified, awaiting live P0.1/P0.2/P0.3/P0.5 numbers.
**Date**: 2026-04-19.

## Verified findings (source-level, no live data yet)

| Fact-check | Method | Result |
|---|---|---|
| P0.6 fused_ffn_decode_q4_0 dead | Grep `fused_ffn` in `candle-transformers/` | **CONFIRMED** — zero callers. Drops off the priority list. |
| P0.4 No L2 prefetch in candle MMQ | Grep `prefetch` in `mmq_turbo.cu` + turbo source read | **CONFIRMED** — candle source line 21: `// Drop L2-prefetch: add back in stage 3 if useful.` Turbo has `gfx906_prefetch_y_tile_v4` (`mmq-prefetch.cuh`). |
| gfx906_sgpr_broadcast unused | Grep | **CONFIRMED** — defined `gfx906_primitives.cuh:285,292`, zero call sites. |
| amdgpu_waves_per_eu missing | Grep | **CONFIRMED** — 0 matches under `candle-hip-kernels/`. |
| Q8 GEMV orphaned in StandardAttention | Code read | **CONFIRMED** — `quantized_blocks/attention.rs:1731-1734` does `dequantize_k/v` + transpose + contiguous; direct wrapper `gqa_attention_decode_q8` is only called from `quantized_llama.rs:630`. |

## Pending (need live data)

- **P0.1 baseline perf** — gemma4-E4B / qwen3.5-9B / qwen3-coder-next × `{CANDLE_KV_Q8=0,1}` × `pp512,tg64`.
- **P0.2 rocprofv3** — prove `dequantize_q8_0_kv_f32` fires per-layer, measure % of decode.
- **P0.3 correctness** — run `hip_q8_attn_pipeline_test`, `hip_q8_v_attn_test`, `hip_q8_qk_attn_test`, `hip_q8_decode_loop_test`.
- **P0.5 MMVQ occupancy** — PMC `SQ_WAVES` / `SQ_BUSY_CYCLES`.

## Chosen first target: wire `gqa_attention_decode_q8` into `StandardAttention`

### Why this first

- Kernels + wrapper already exist and are decode-only safe (`q8_kv_cache.rs:549`).
- Existing test `hip_q8_attn_pipeline_test` already validates Q8 vs F32-dequant with 5e-5 × r_max tolerance.
- Scope is localized to `candle-transformers/src/models/quantized_blocks/attention.rs:1709-1749`.
- Expected payoff scales linearly with layer count × decode tokens × KV-slab size.
- **Does NOT touch** prefill, LM-head, MoE, or TP/EP paths — rollback is flipping one branch.

### Current code shape (attention.rs:1709-1749)

```rust
let q8_route = if tq_route.is_none() && self.use_q8_kv {
    ... init cache ...
    cache.append(&k, &v)?;
    let k_canonical = cache.dequantize_k()?;    // HBM write of N_kv×D F32
    let v_canonical = cache.dequantize_v()?;    // HBM write of N_kv×D F32
    let k_t = k_canonical.transpose(2,3)?.contiguous()?;   // HBM re-read+write
    Some((k_t, v_canonical))
} else { None };
...
let (k_t, v) = ... tq_route / q8_route / kv_cache.append ...;
let attn_output = gqa_attention_k_transposed(&q, &k_t, &v, mask, self.attn_scale)?;
```

### Proposed shape

```rust
// Decode fast path: L_q == 1 and Q8 cache active.
#[cfg(feature = "hip")]
let q8_decode_direct = if self.use_q8_kv && seq_len == 1 && tq_route.is_none() {
    use candle::hip_backend::q8_kv_cache::{gqa_attention_decode_q8, QKvCache};
    // ... ensure self.q8_cache is Some (same init as current q8_route)
    let cache = self.q8_cache.as_mut().unwrap();
    cache.append(&k, &v)?;
    let l_k = cache.current_seq_len();
    // Mask shape contract: gqa_attention_decode_q8 wants optional (1,1,1,L_k).
    // When `mask` is None (full-attention decode at L_q=1), pass None.
    let out = gqa_attention_decode_q8(&q, cache, l_k, mask, self.attn_scale)?;
    Some(out)
} else { None };

// Existing q8_route gated to prefill only
let q8_route = if tq_route.is_none() && self.use_q8_kv && q8_decode_direct.is_none() {
    ... existing body ...
};

let attn_output = if let Some(out) = q8_decode_direct {
    out
} else {
    let (k_t, v) = ... ;
    gqa_attention_k_transposed(&q, &k_t, &v, mask, self.attn_scale)?
};
```

### Risks

1. **Mask shape mismatch**: `gqa_attention_decode_q8` expects `(1,1,1,L_k)` additive F32. StandardAttention's caller passes `mask` whose shape varies by model. Must inspect one per model arch (gemma4 SWA/global, qwen3.5 standard, qwen3-coder-next SWA). If shape wrong → bail in the wrapper; if wrong silently → correctness hole.
2. **Head dim coverage**: kernel supports D=64/128/256/512. Gemma4-E4B global layers are D=512 ✓; SWA D=256 ✓; Qwen3.5 D=128 ✓; Qwen3-coder-next D=256 ✓. **Safe for all current targets**, but add a `matches!(head_dim, 64|128|256|512)` guard as a defensive skip rather than bail.
3. **BxH_q×1×L_k scores scaling**: wrapper applies `affine(scale, 0.0)` inside — re-check `self.attn_scale` matches `scale` contract.
4. **Integration with G2/G3 graph replay**: this adds 2 new kernel launches per layer (`gqa_decode_gemv_qk_q8` + `gqa_decode_gemv_v_q8`) and drops ≥2 (dequant_k/v). Graph-capture plan will need a re-record. Flag as follow-up, not blocker.

### Correctness gate (P2.2)

For each of (gemma4-E4B, qwen3.5-9B, qwen3-coder-next) at temp=0, seed=fixed:
1. Baseline (dequant path): generate 64 tokens from a fixed prompt.
2. New path: same prompt, direct-GEMV.
3. Assert identical tokens OR logit max_abs_diff < 5e-4.

If tokens differ: first-difference index + logits dump.

### Rollback

Env `CANDLE_Q8_KV_DIRECT_GEMV=0` forces the old dequant path. Default **on** only after all 3 models pass.

### Integration test consolidation (P2.4)

Add to `candle-core/tests/` (or as a `#[test]` in existing q8_kv module):
- `q8_direct_gemv_matches_dequant_pipeline` — uses the same tolerance (5e-5 × r_max) as `hip_q8_attn_pipeline_test`, but wrapped in `#[cfg(feature="hip")] #[test]`.
- `q8_direct_gemv_env_optout` — with `CANDLE_Q8_KV_DIRECT_GEMV=0`, verify dequant kernel fires (hard to test without rocprof; settle for confirming output byte-equal to pre-change build).
- End-to-end: smoke-test via `cargo test --features hip -p candle-transformers quantized_blocks::attention::tests` if an existing test harness fits; if not, add a minimal one.

## What we're NOT doing in this change

- No MMQ L2-prefetch port (that's a separate, bigger effort with correctness-sweep needed).
- No `waves_per_eu` annotations (pure perf, should be one-at-a-time with PMC verification).
- No `gfx906_sgpr_broadcast` application (deferred).
- No `fused_ffn_decode_q4_0` wire-up (dead code, out of scope).
- No flash_attn_v2 D=512 bank-conflict fix.

Each of those gets its own DESIGN doc after this one lands.

## Go/no-go gate for P2.1

Proceed iff ALL of:
- P0.1 baseline captured (reproducible numbers)
- P0.2 confirms dequant kernel is non-trivial fraction of decode time (≥10%)
- P0.3 all 4 Q8 correctness examples exit 0 on current build
- User signs off on this doc
