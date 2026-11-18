# P4 — Direct Q8 GEMV into gemma4 decode path

**Status**: P4.2 plan, awaiting sign-off.
**Target**: `candle-transformers/src/models/quantized_gemma4.rs` — the Q8 arm at lines 1891-1894 that currently calls `canonical_k_f32()` + `canonical_v_f32()`.

## P4.0+P4.1 findings (measured)

Gemma4-26B-A4B, 4 GPU, `CANDLE_KV_Q8=1`, prompt="explain quantum gravity", sample-len=32:

| Kernel | Calls | Pct of decode kernel time | Role |
|---|---|---|---|
| `dequantize_q8_0_kv_f32` | 2040 | **0.95%** | Dequant K+V, every FullAttn layer × decode step |
| `gqa_decode_mv_fast_d256_f32` | 800 | **0.86%** | SWA attention fused Q·K+softmax+V·attn, F32 |
| `gqa_decode_mv_fast_d512_f32` | 160 | **0.34%** | Global attention fused, F32 |
| `softmax_f32` | 1052 | 9.60% | Mostly MoE router (A4B is MoE), not attention |
| `gqa_decode_gemv_*_q8_*` | 0 | — | Direct-GEMV kernels NOT FIRING (confirm wire-up target) |

## Uncertainty to flag

Unlike GatedAttention (which used `gqa_attention_k_transposed` → rocBLAS GEMM + separate softmax), gemma4 already uses a **fused F32 attention kernel** `gqa_decode_mv_fast_d{256,512}_f32` that does Q·K + softmax + V·attn in one launch.

Replacing with `gqa_attention_decode_q8` wrapper decomposes into: `gqa_decode_gemv_qk_q8` → `affine` → `max_keepdim` → `exp` → `sum_keepdim` → `broadcast_div` → `gqa_decode_gemv_v_q8`. That's **7 kernel launches** instead of 2 (dequant + fused_mv_fast). So this change trades:

- **Saved**: dequant cost (0.95%) + transient F32 KV slab allocation + reading F32 K/V through HBM once.
- **Paid**: extra kernel launches (launch overhead ~5-10 µs × 5 extra kernels × FullAttn layers × decode steps).
- **Saved again**: `gqa_decode_mv_fast` takes F32 K/V as input — without it we skip the F32 K/V read which is also the biggest HBM bandwidth line item.

Net could go either way on gemma4; **this IS an experiment, not a guaranteed win**. We must measure both directions.

## Proposed change

In `quantized_gemma4.rs`, near line 1891 where the Q8 dequant-and-extract happens:

1. Add a decode-only fast path branch: when `seq_len_q == 1` + cache is Q8 + `head_dim ∈ {64,128,256,512}` + env opt-in allows, call `gqa_attention_decode_q8(q, &q8_cache, l_k, mask, scale)` directly.
2. Return the pre-`finish_attn` result so the existing `finish_attn` + post-attn norm + residual remain untouched.
3. Fall back to the existing dequant path for prefill (`seq_len_q > 1`), TQ caches, or unsupported head_dims.

### Complication: multi-device placement

Gemma4 is pipeline-parallel-capable. The Q8 cache lives on `layer_device` of layer `src` (possibly != layer `il`'s device). The q must be on the same device as the cache. The existing code handles this at line 1897 (`k.to_device(&layer_device)` after extraction). For direct-GEMV, we must place q on the cache's device before calling the wrapper.

Helper signature:

```rust
#[cfg(feature = "hip")]
fn gemma4_q8_decode_direct(
    &self,
    il: usize,
    has_kv: bool,
    kv_source_idx: usize,
    q_for_attn: &Tensor,       // already through prepare_q or shared_qkv
    attention_mask: Option<&Tensor>,
    seq_len_q: usize,
) -> Result<Option<Tensor>> {
    if seq_len_q != 1 { return Ok(None); }
    if std::env::var("CANDLE_Q8_KV_DIRECT_GEMV").ok().as_deref() == Some("0") {
        return Ok(None);
    }
    let src = if has_kv { il } else { kv_source_idx };
    let any_cache = self.kv_caches[src].as_ref().expect("KV cache must exist");
    let q8_cache = match any_cache.as_q8() {
        Some(c) => c,
        None => return Ok(None),          // not Q8, fall back
    };
    let head_dim = q_for_attn.dim(candle::D::Minus1)?;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        return Ok(None);
    }
    let l_k = q8_cache.current_seq_len();
    if l_k == 0 { return Ok(None); }
    // Move q to the cache's device if needed (matches existing k/v
    // to_device at line 1897).
    let cache_dev = /* derive from q8_cache — see below */;
    let q = if device_eq(q_for_attn.device(), cache_dev) {
        q_for_attn.clone()
    } else {
        q_for_attn.to_device(cache_dev)?
    };
    // Mask on same device.
    let mask = match attention_mask {
        Some(m) if !device_eq(m.device(), cache_dev) => Some(m.to_device(cache_dev)?),
        Some(m) => Some(m.clone()),
        None => None,
    };
    use candle::hip_backend::q8_kv_cache::gqa_attention_decode_q8;
    let out = gqa_attention_decode_q8(&q, q8_cache, l_k, mask.as_ref(), self.layers[il].attn.attn_scale)?;
    Ok(Some(out))
}
```

Need to expose `QKvCache`'s device — check if `dev()` getter exists (likely `QKvCache::dev()` or similar, otherwise add it).

### Integration points (two different attention dispatches in gemma4::forward)

Lines 1944-2011 have TWO branches that both do attention:
- `shared_qkv` present → uses `q_precomputed` directly
- `k_is_canonical` no shared_qkv → calls `self.layers[il].attn.prepare_q(&x_norm, index_pos)` inline

Strategy: thread the Q8 short-circuit into BOTH branches by computing `q8_direct_attn: Option<Tensor>` right after the q tensor is materialized and before the k_use/v_use extraction. When Some, skip the branch-specific attention dispatch and go straight to `finish_attn`.

Cleanest placement: put the short-circuit inside each branch at its topmost "q is ready" point, since `q` materializes at different places in the two branches.

## Correctness protocol (P4.4)

1. temp=0 diff on gemma4-26B-A4B, prompt="explain quantum gravity", 32 tokens, opt-out vs default. Expect coherent output in both; drift acceptable per float-reduction-order rules.
2. rocprofv3: with opt-in, `gqa_decode_mv_fast_*` call count should **drop** (replaced by `gqa_decode_gemv_*_q8_*`), `dequantize_q8_0_kv_f32` should also drop to prefill-only (~few calls).
3. IT test extension in `bench/test_q8_direct_gemv_it.sh`: add a gemma4 coherence + kernel-dispatch section that toggles `CANDLE_Q8_KV_DIRECT_GEMV={0,1}`.

## Rollback + failure modes

Env: existing `CANDLE_Q8_KV_DIRECT_GEMV=0` forces dequant path (same flag as GatedAttention wire-up — unified surface).

Failure modes: same as P1.2. Head-dim guard + env opt-out make it defensive.

## Go/no-go

The per-kernel payoff math is ambiguous (see "Uncertainty" above). Decision:
- Implement behind the same env flag.
- Measure on gemma4-26B-A4B.
- If **regression >1% tg** → keep the flag off by default for this model, or gate by arch.
- If **win or neutral** → default on, ship.

**Go or tweak?**
