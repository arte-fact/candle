# F4: LDS-Q-tile fused Q8-KV attention — design

**Status:** M1 (correctness harness + minimal kernel)
**Date:** 2026-04-20
**Context:** P10 (2026-04-19) shipped a single-Q dp4a fused Q8 attention kernel.
Correctness passed (2e-4 abs err) but perf was 4× slower than F32-fused because
Q-quant overhead didn't amortize over a single Q row. Retro concluded that a
structural **multi-Q tile** port is needed — the "turbo fattn-q8" approach.

## Insight (what P10 missed)

Decode has one Q token per sequence. But GQA has multiple Q heads sharing one
KV head (n_rep = H_q / H_kv). Those Q heads read the SAME KV cache row. Tile
them together:

- One block per (batch, kv_head). Grid: (1, H_kv, B).
- Each block handles ncols = n_rep Q heads in one pass.
- Load K/V tile into LDS ONCE; reuse across all ncols Q heads.
- Q-quant per-Q-row: once per block; amortises over L_k iters × ncols rows.
- dp4a Q·K inner product: 4 int8 FMAs → 1 sdot4.

Model GQA ratios (n_rep):
- Devstral-24B: H_q=32, H_kv=8 → n_rep=4
- Qwen3.6-35B: H_q=32, H_kv=4 → n_rep=8
- Gemma-4-31B (FullAttn): H_q=32, H_kv=8 → n_rep=4

## Kernel contract (M1)

Inputs:
- `q`: F32 `(B, H_q, 1, D)`
- `k_cache`: Q8 blocks `(B, H_kv, T_max, D/32) × {half d, int8 qs[32]}`
- `v_cache`: same layout
- `mask`: F32 `(B, L_k)` or null
- `out`: F32 `(B, H_q, 1, D)`

Fixed constants for M1:
- D = 128 (Devstral, Qwen3.5, gemma4 FullAttn all fit)
- ncols = 4 (Devstral GQA ratio); M2 adds ncols=8 for Qwen3.6

Grid: `(1, H_kv, B)`. Block: 256 threads (4 warps).

## Kernel structure

```
Load Q tile (ncols × D) into registers-per-warp + quantize to Q8 per row:
  For each Q row r in [0, ncols):
    max_abs = warp-reduce max |q[r,:]|
    q_scale[r] = max_abs / 127.0
    q_int8[r,d] = round(q[r,d] / q_scale[r])

For each K-block chunk (t in [0, L_k) step nbatch_fa):
  // Load K tile (nbatch_fa rows × D) into LDS
  Load K tile to LDS.

  For each k in [0, nbatch_fa):
    // Q·K inner product (dp4a over D/4 int32 words)
    For each Q row r:
      s[r, t+k] = q_scale[r] * k_scale[k] * sum_d(dp4a(q_int8[r,d], k_int8[k,d]))
    // mask + online softmax update per r

  // V accumulation (stays F32 for now in M1 — matches P10 choice)
  Load V tile to LDS (dequant on load).
  For each Q row r, each k: o[r,:] += attn[r,k] * v[k,:]

Write out[r] = o[r] / l_sum[r]
```

## What M1 delivers

1. Design note (this file).
2. Minimal `gqa_decode_fattn_tile_q8_d128_n4_f32` kernel in
   `candle-hip-kernels/src/attn_q8_kv.cu`.
3. Correctness harness in `candle-core/examples/hip_q8_fattn_tile_test.rs`:
   compares against F32 dequant reference at L_k ∈ {64, 256, 1024},
   max_abs_err < 5e-3.
4. **No perf claim. No model wire-in.** That's M2+.

## What M1 does NOT do

- Multiple D values (M3).
- V-dp4a path (risks M2 decision gate).
- Batched-K LDS staging beyond naive: M1 uses simplest tile-K (one row at a time in LDS); M2 will compare full nbatch_fa staging.
- Model wire-in (M4, after perf gate passes).

## Exit criteria

- Kernel builds and runs.
- `hip_q8_fattn_tile_test` passes max_abs_err < 5e-3 at all three L_k values.
- Design note committed.

## Risk & kill criteria

M2 has the perf gate: if dp4a tile < F32-fused at L_k≥1024 after reasonable
tuning, F4 is killed and memoryed as a second null. Do NOT wire in until M2
clears.
