# P1.2 — Implementation plan: direct Q8 GEMV into GatedAttention decode

**Parent**: `DESIGN-Q8KV-DIRECT-GEMV-WIREUP-2026-04-19.md`
**Target**: `candle-transformers/src/models/quantized_blocks/attention.rs` — `GatedAttention::forward` (lines 1616-1762).
**Opt-out**: `CANDLE_Q8_KV_DIRECT_GEMV=0` (default on).

## Scope

One function change. No new kernels. No dispatcher changes. No data structure changes.

The `GatedAttention` struct already owns a `q8_cache: Option<QKvCache>` (attention.rs:1545). The kernel entrypoint `gqa_attention_decode_q8` is already imported-able from `candle_core::hip_backend::q8_kv_cache`. The wire-up is a single branch added inside `forward`, plus a gating tweak on the existing `q8_route`.

## Preconditions (activate fast path iff ALL hold)

1. `cfg!(feature = "hip")`
2. `self.tq_variant.is_none()` — TurboQuant takes precedence (unchanged)
3. `self.use_q8_kv == true` — Q8 KV is opted in
4. `seq_len == 1` — decode only; the wrapper is explicitly decode-only (`q8_kv_cache.rs:544`)
5. `matches!(self.head_dim, 64 | 128 | 256 | 512)` — kernel coverage
6. `std::env::var("CANDLE_Q8_KV_DIRECT_GEMV").ok().as_deref() != Some("0")` — opt-out escape hatch

Prefill (seq_len > 1) stays on the existing `dequantize_k/v` path.

## Code change (concrete diff sketch)

### New helper (inside `impl GatedAttention`, near `forward`)

```rust
#[cfg(feature = "hip")]
fn q8_decode_direct(
    &mut self,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    offset: usize,
    seq_len: usize,
) -> Result<Option<Tensor>> {
    // Preconditions (see doc).
    if self.tq_variant.is_some() || !self.use_q8_kv || seq_len != 1 {
        return Ok(None);
    }
    if !matches!(self.head_dim, 64 | 128 | 256 | 512) {
        return Ok(None);
    }
    if std::env::var("CANDLE_Q8_KV_DIRECT_GEMV").ok().as_deref() == Some("0") {
        return Ok(None);
    }
    use candle::hip_backend::q8_kv_cache::{gqa_attention_decode_q8, QKvCache};
    let hip_dev = match k.device() {
        candle::Device::Hip(d) => d.clone(),
        _ => return Ok(None),       // wrong device — let the outer path bail
    };
    let (b_sz_q, _, _, _) = k.dims4()?;
    if self.q8_cache.is_none() {
        let t_max = kv_max_seq_len();
        self.q8_cache = Some(QKvCache::new(
            &hip_dev, b_sz_q, self.n_kv_head, t_max, self.head_dim,
        )?);
    }
    if offset == 0 {
        self.q8_cache.as_mut().unwrap().reset();
    }
    let cache = self.q8_cache.as_mut().unwrap();
    cache.append(k, v)?;
    let l_k = cache.current_seq_len();
    let out = gqa_attention_decode_q8(q, cache, l_k, mask, self.attn_scale)?;
    Ok(Some(out))
}

#[cfg(not(feature = "hip"))]
fn q8_decode_direct(
    &mut self,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
    _offset: usize,
    _seq_len: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}
```

### Forward body changes (attention.rs:~1670-1749)

1. Immediately after `let k = k.contiguous()?; let v = v.contiguous()?;` (line 1668), call the helper and save `q8_direct_out: Option<Tensor>`.

2. Gate the existing `q8_route` block (line 1710) on `q8_direct_out.is_none()`:
   ```rust
   let q8_route = if tq_route.is_none() && self.use_q8_kv && q8_direct_out.is_none() {
       ... existing body unchanged ...
   };
   ```
   This prevents the dequant body from running when the fast path has already appended to the cache and produced `out`.

3. Replace the final attn_output computation (line 1742-1749):
   ```rust
   let attn_output = if let Some(out) = q8_direct_out {
       out
   } else {
       let (k_t, v) = if let Some(pair) = tq_route {
           pair
       } else if let Some(pair) = q8_route {
           pair
       } else {
           self.kv_cache.append(&k, &v)?
       };
       gqa_attention_k_transposed(&q, &k_t, &v, mask, self.attn_scale)?
   };
   ```

Downstream code (gate sigmoid, reshape, `self.wo.forward`) is unchanged.

## Invariants preserved

| Invariant | Reasoning |
|---|---|
| KV cache is appended exactly once per forward | Fast path appends via `cache.append` OR existing q8_route body appends, never both (gated by `q8_direct_out.is_none()`). |
| Prefill path unchanged | `seq_len != 1` short-circuits the helper. Existing dequant path runs. |
| TQ takes precedence over Q8 | Helper bails if `self.tq_variant.is_some()`. Matches existing behavior. |
| Output shape `(B, H_q, 1, D)` | `gqa_attention_decode_q8` returns exactly this. Same as `gqa_attention_k_transposed` output for L_q=1. |

## Test protocol (P2.2)

### Unit test — new `attention.rs` test module

```rust
#[test]
#[cfg(feature = "hip")]
fn q8_direct_gemv_matches_dequant_pipeline() {
    // Mirror hip_q8_attn_pipeline_test structure but drive through
    // GatedAttention::forward end-to-end: build a minimal GatedAttention
    // instance, run forward() with CANDLE_Q8_KV_DIRECT_GEMV=1 vs "0"
    // on the same inputs, assert max_abs_diff < 5e-5 × r_max.
}
```

Risk: building a minimal `GatedAttention` requires `QMatMul` + `RmsNorm` + `RotaryEmbedding` + `KvCache`. If the setup is prohibitive, fall back to:

### Alt test — end-to-end model, deterministic

```bash
CANDLE_KV_Q8=1 CANDLE_Q8_KV_DIRECT_GEMV=0 ./target/release/examples/quantized-qwen35 \
  --model /artefact/models/Qwen3.5-9B-Q4_1.gguf --prompt "..." --sample-len 32 --temperature 0 > /tmp/dequant.txt

CANDLE_KV_Q8=1 CANDLE_Q8_KV_DIRECT_GEMV=1 ./target/release/examples/quantized-qwen35 \
  --model /artefact/models/Qwen3.5-9B-Q4_1.gguf --prompt "..." --sample-len 32 --temperature 0 > /tmp/direct.txt

diff /tmp/dequant.txt /tmp/direct.txt
```

Expected: identical output, OR max first-difference at token N where N is large and both outputs remain coherent (floating-point reduction-order noise can diverge sampling after many tokens).

### Perf regression test

rocprofv3 kernel trace with `CANDLE_KV_Q8=1 CANDLE_Q8_KV_DIRECT_GEMV=1`:
- `dequantize_q8_0_kv_f32` should **drop out** of the kernel trace for decode steps.
- `gqa_decode_gemv_qk_q8_d{...}_f32` and `gqa_decode_gemv_v_q8_d{...}_f32` should **appear** with call counts matching (FullAttn layer count × decode-step count).

## Correctness failure modes and responses

| Failure | Diagnosis | Action |
|---|---|---|
| Output shape mismatch after `gqa_attention_decode_q8` | wrapper contract not matched | bail with clear error, disable fast path for that layer's shape |
| Mask broadcast error inside wrapper | mask shape isn't `(1,1,1,L_k)` | dump mask.dims() and fix at call site, not in wrapper |
| max_abs_diff > 5e-5 but tokens match | reduction-order float noise | acceptable, document |
| Tokens diverge early (within first ~8) | real bug | revert; do not ship |
| Works on qwen3.5 but not qwen3-moe | arch-specific issue | add arch guard; narrow scope |

## P2.1 execution order

1. Add helper `q8_decode_direct` (one new function, #[cfg(feature="hip")] variant + stub).
2. Modify `forward` body in three spots (compute `q8_direct_out`, gate `q8_route`, branch on attn_output).
3. Compile-check: `cargo build --release -p candle-transformers --features hip`.
4. Run existing Q8 correctness examples (should still pass — they don't exercise the new path).
5. Run end-to-end diff test on Qwen3.5-9B (both env settings).
6. Run rocprofv3 trace: verify dequant drops, direct-GEMV appears.
7. Measure pp512 + tg64 with `CANDLE_Q8_KV_DIRECT_GEMV={0,1}`; expect tg parity or modest gain at short context, more at longer.

## P2.4 IT consolidation

Add to `candle-transformers/tests/` (or existing attention test module):
- `q8_kv_direct_gemv_vs_dequant_coherent` — end-to-end model: temp=0, 32-token generation, assert identical or tolerated drift vs opt-out.
- `q8_kv_direct_gemv_head_dim_guard` — construct GatedAttention with head_dim=77 (unsupported), verify it falls back gracefully (no panic, dequant path runs).
- `q8_kv_direct_gemv_offset_zero_resets_cache` — sequential forwards with offset=0 in the middle reset the Q8 cache.

## Go/no-go for P2.1

Proceed iff user signs off on this doc. No code changes until then.
