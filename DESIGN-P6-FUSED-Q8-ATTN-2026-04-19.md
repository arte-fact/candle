# P6 — Fused Q8 attention decode kernel (gemma4)

**Goal**: Write `gqa_decode_mv_fast_q8_d{256,512}_f32` — a single-launch kernel that reads Q8_0 KV blocks directly and does Q·K + online-softmax + V·attn in one pass. Replaces the current dequant→`gqa_decode_mv_fast_d{256,512}_f32` chain on gemma4 and also beats the 7-launch `gqa_attention_decode_q8` wrapper.

## Why this is the right target

Measured in P4 (`DESIGN-P4-GEMMA4-Q8-DIRECT-GEMV-2026-04-19.md`):
- Dequant + fused F32: tg=50.56 (baseline on gemma4-26B-A4B Q8 KV 4-GPU).
- Wrapper (qk_gemv → softmax chain → v_gemv, 7 launches): tg=46.13 (-8.8%).

The wrapper regressed because we traded **1 fused kernel** for **7 small kernels**. A fused Q8 kernel gets the best of both: no dequant detour, and a single launch.

## Structural mirror of the F32 fused kernel

Template from `flash_attn_v2.cu:857` (`gqa_decode_mv_fast_impl`):

```
Block: 64 threads (single wave). Grid: (1, H_q, B).
Per-lane regs: Q as F2_PER_LANE float2 (=D/128 float2 values), o_reg scalar[2*F2_PER_LANE].
Online softmax: m_i (running max), l_i (running sum).
Main loop: for t in [0, L_k):
    Load K row + V row (D floats each from HBM).
    Per-thread partial Q·K dot.
    Warp-reduce → s_t (scalar, same on all lanes).
    Scale + optional mask.
    m_new = max(m_i, s_t); alpha = exp(m_i - m_new); p = exp(s_t - m_new).
    o_reg = alpha * o_reg + p * V_elements.
    l_i = alpha * l_i + p; m_i = m_new.
Finalize: o = o_reg / l_i.
```

## Q8 adaptation

Changes from F32:
- Block is 64 threads, single wave. **Same as F32.**
- Per iteration, load `block_q8_0_kv` K and V blocks (34 B each vs 64 B for F16 or 128 B for F32 at D=256). Unpack int8 + multiply by half-scale.
- Per-lane covers `F_PER_LANE = D / BLOCK_SIZE` D-positions. At D=256, that's 4; at D=512, 8.
- Coalescing: lane `l` reads `block_q8_0[l / 32].qs[l % 32]` for each of its i-iters at d = l + i * BLOCK_SIZE. Lanes 0..31 of iter 0 read a single block's 32 qs consecutively → natural coalesce.
- Block scale `__half2float(blk->d)` — loaded per lane redundantly, but the compiler uses the scalar cache for the half broadcast. Could later `gfx906_sgpr_broadcast_f32` explicitly (from the earlier audit).

### Sketch

```c
template <int D, int BLOCK_SIZE = 64>
static __device__ __forceinline__ void gqa_decode_mv_fast_q8_impl(
    const float * q,             // (B, H_q, 1, D)
    const void  * k_cache,       // (B, H_kv, T_max, D/32) block_q8_0_kv
    const void  * v_cache,       // (B, H_kv, T_max, D/32) block_q8_0_kv
    const float * mask,          // optional (1, 1, 1, L_k) or broadcast
    float       * out,           // (B, H_q, 1, D)
    int B, int H_q, int H_kv, int L_k, float scale, int n_rep,
    int mask_b_stride, int L_k_iter, int T_max)
{
    constexpr int F_PER_LANE     = D / BLOCK_SIZE;
    constexpr int BLOCKS_PER_ROW = D / 32;
    static_assert(D % 32 == 0 && D % BLOCK_SIZE == 0);

    const int h_q = blockIdx.y;
    const int b   = blockIdx.z;
    const int lane = threadIdx.x;
    const int h_kv = (n_rep > 1) ? h_q / n_rep : h_q;

    // Q in registers (F32)
    const float * q_ptr = q + ((int64_t)(b*H_q)+h_q)*D;
    float q_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) q_reg[i] = q_ptr[lane + i*BLOCK_SIZE];

    // Running output, online softmax
    float o_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) o_reg[i] = 0.0f;
    float m_i = -INFINITY, l_i = 0.0f;

    const block_q8_0_kv * k_base = (const block_q8_0_kv *)k_cache
        + (int64_t)(b*H_kv + h_kv) * T_max * BLOCKS_PER_ROW;
    const block_q8_0_kv * v_base = (const block_q8_0_kv *)v_cache
        + (int64_t)(b*H_kv + h_kv) * T_max * BLOCKS_PER_ROW;
    const float * mask_row = mask ? mask + (int64_t)b * mask_b_stride : nullptr;
    const int L_k_eff = L_k_iter >= 0 ? L_k_iter : L_k;

    for (int t = 0; t < L_k_eff; ++t) {
        const block_q8_0_kv * k_row = k_base + (int64_t)t * BLOCKS_PER_ROW;
        const block_q8_0_kv * v_row = v_base + (int64_t)t * BLOCKS_PER_ROW;

        float k_vals[F_PER_LANE], v_vals[F_PER_LANE];
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            const int d    = lane + i * BLOCK_SIZE;
            const int bd   = d >> 5;           // block index in row
            const int lane_in_blk = d & 31;    // position in block
            const float ks = __half2float(k_row[bd].d);
            const float vs = __half2float(v_row[bd].d);
            k_vals[i] = ks * (float)k_row[bd].qs[lane_in_blk];
            v_vals[i] = vs * (float)v_row[bd].qs[lane_in_blk];
        }

        // Partial Q·K
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) partial += q_reg[i] * k_vals[i];
        float s_t = warp_reduce_sum(partial) * scale;
        if (mask_row) s_t += mask_row[t];

        const float m_new = fmaxf(m_i, s_t);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_t - m_new);
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            o_reg[i] = alpha * o_reg[i] + p * v_vals[i];
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    const float inv_l = gfx906_rcp(l_i);
    float * o_ptr = out + ((int64_t)(b*H_q)+h_q)*D;
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        o_ptr[lane + i*BLOCK_SIZE] = o_reg[i] * inv_l;
    }
}
```

## Correctness test

New example `hip_q8_fused_attn_test.rs`:
- Pick B=1, H_q=8, H_kv=1 or 2, D=256, L_k=256.
- Populate F32 q + F32 k,v in [−1, 1].
- Pack k,v into QKvCache. Dequant to reference F32. Run the F32 `gqa_decode_mv_fast_d256_f32` to get the F32-dequant reference.
- Run the new fused Q8 kernel on the same QKvCache.
- Assert `max_abs_err < 5e-5 × r_max` (same tolerance as `hip_q8_attn_pipeline_test`).

## Rust dispatcher

New `pub fn gqa_decode_mv_fast_q8_d{256,512}_f32(q, cache, l_k, mask, scale, l_k_iter) -> Result<Tensor>` in `q8_kv_cache.rs`. Signature matches `gqa_decode_gemv_v_q8` but outputs the full (B, H_q, 1, D) attn vs scores.

## Integration

Modify `gemma4_try_q8_decode_direct` to prefer the fused kernel when D ∈ {256, 512}. Fall back to `gqa_attention_decode_q8` wrapper for D=64/128 (and default-off elsewhere).

Flip gemma4 default to **ON** once measurements confirm win.

## Risk / fallback

Risk: online-softmax numerical drift across Q8 rounding + reduction order. Tolerance at 5e-5 should hold; if not, use two-pass softmax (costs L_k extra exp + writeback) as fallback.

Rollback: same env flag `CANDLE_Q8_KV_DIRECT_GEMV=0` disables.

## Scope for this session

- Write the kernel, dispatcher, and correctness test.
- Measure per-kernel µs vs dequant+fast on gemma4-A4B.
- Only if win → wire into gemma4 + bench end-to-end.
