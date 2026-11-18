// Q8_0 quantize-append + dequantize kernels for the Q8 KV cache.
//
// Storage layout (matches ggml block_q8_0):
//     struct { half d; int8_t qs[32]; }   // 34 bytes per 32-element block
// Cache shape:  (B, H_kv, T_max, D/32) of block_q8_0.
// Row stride:   (D/32) * 34 bytes, D-contiguous within a row.
//
// Kernel C (V·attn gemv) and Kernel A (Q·K gemv) both want D-contiguous reads
// per t, so K and V use the same layout; no transpose variant needed.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "gfx906_primitives.cuh"

#define QK8_0 32

typedef struct {
    __half  d;
    int8_t  qs[QK8_0];
} block_q8_0_kv;

static_assert(sizeof(block_q8_0_kv) == sizeof(__half) + QK8_0,
              "block_q8_0_kv has unexpected padding");

// ---------------------------------------------------------------------------
// quantize_q8_0_append_f32
//
// Packs an F32 (B, H, T_new, D) tensor into Q8_0 blocks at token offset
// `t_offset` in a (B, H, T_max, D/32) cache.
//
// Grid:  (D / 32, H * T_new, B)
// Block: 32 threads  (one per element in a QK8_0 block)
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_q8_0_append_f32(
    const float * __restrict__ x,        // (B, H, T_new, D)
    void        * __restrict__ cache,    // (B, H, T_max, D/32) block_q8_0
    const int B,
    const int H,
    const int T_new,
    const int D,
    const int T_max,
    const int t_offset
) {
    const int tid   = threadIdx.x;          // 0..31
    const int ib_d  = blockIdx.x;           // which D/32 block along D
    const int bh_t  = blockIdx.y;           // H * T_new flattened
    const int b     = blockIdx.z;

    const int t_new = bh_t / H;
    const int h     = bh_t % H;
    const int d     = ib_d * QK8_0 + tid;

    const int64_t x_idx =
        ((int64_t)((b * H + h) * T_new + t_new)) * D + d;
    const float xi = x[x_idx];

    // amax reduction across the 32-thread block.
    float amax = fabsf(xi);
    amax = gfx906_half_warp_reduce_max_dpp(amax);

    const float d_scale = amax / 127.0f;
    const float rd = d_scale == 0.0f ? 0.0f : 1.0f / d_scale;
    const int8_t q = (int8_t)roundf(xi * rd);

    const int t_abs = t_offset + t_new;
    const int blocks_per_row = D / QK8_0;

    block_q8_0_kv * blk = (block_q8_0_kv *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    blk->qs[tid] = q;
    if (tid == 0) {
        blk->d = __float2half(d_scale);
    }
}

// ---------------------------------------------------------------------------
// F2 (2026-04-20): quantize_q8_0_append_rope_i_f32
//
// Fused version of `quantize_q8_0_append_f32` that applies interleaved
// (rope_i) rotary embedding BEFORE quantisation. Replaces the 2-kernel
// chain `rope_i_f32 → quantize_q8_0_append_f32` with one launch.
//
// Layout assumptions (matches candle's `rope_i`):
//   - cos/sin buffers are (max_seq, head_dim / 2) F32 row-major.
//   - Position indexing: for the append, absolute position is
//     `t_offset + t_new`. Rope index = t_abs * (head_dim / 2) + (d / 2).
//   - Interleaved pairing: (d_even, d_odd) rotated together. Each thread
//     holds one D element; partner is `tid ^ 1` via `__shfl_xor`.
//
// This kernel only supports **full rope** (rope_dim == D). For partial
// rope (rope_dim < D) the caller should fall back to the decomposed path.
//
// Grid:  (D / 32, H * T_new, B)
// Block: 32 threads
extern "C" __global__ void quantize_q8_0_append_rope_i_f32(
    const float * __restrict__ x,        // (B, H, T_new, D)
    const float * __restrict__ cos_tbl,  // (max_seq, D/2)
    const float * __restrict__ sin_tbl,  // (max_seq, D/2)
    void        * __restrict__ cache,    // (B, H, T_max, D/32) block_q8_0
    const int B,
    const int H,
    const int T_new,
    const int D,
    const int T_max,
    const int t_offset
) {
    const int tid   = threadIdx.x;
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_new = bh_t / H;
    const int h     = bh_t % H;
    const int d     = ib_d * QK8_0 + tid;

    const int64_t x_idx =
        ((int64_t)((b * H + h) * T_new + t_new)) * D + d;
    const float xi = x[x_idx];

    // Rope: fetch partner's value via warp shuffle (pair = tid^1).
    const float xi_partner = __shfl_xor(xi, 1, 32);

    const int t_abs = t_offset + t_new;
    const int pair_idx = d / 2;
    // cos/sin row for this position: t_abs * (D/2).
    const int cs_off = t_abs * (D / 2) + pair_idx;
    const float c = cos_tbl[cs_off];
    const float s = sin_tbl[cs_off];

    // Interleaved rope:
    //   d_even' = d_even * c - d_odd  * s
    //   d_odd'  = d_even * s + d_odd  * c
    float xi_rot;
    if ((tid & 1) == 0) {
        xi_rot = xi * c - xi_partner * s;
    } else {
        xi_rot = xi_partner * s + xi * c;
    }

    // Quantise across 32-thread block (amax reduce) and write.
    float amax = fabsf(xi_rot);
    amax = gfx906_half_warp_reduce_max_dpp(amax);

    const float d_scale = amax / 127.0f;
    const float rd = d_scale == 0.0f ? 0.0f : 1.0f / d_scale;
    const int8_t q = (int8_t)roundf(xi_rot * rd);

    const int blocks_per_row = D / QK8_0;
    block_q8_0_kv * blk = (block_q8_0_kv *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    blk->qs[tid] = q;
    if (tid == 0) {
        blk->d = __float2half(d_scale);
    }
}

// ---------------------------------------------------------------------------
// dequantize_q8_0_kv_f32
//
// Dequantises a contiguous slab of the Q8 cache back to F32. Used for the
// prefill fallback path (Phase 4) — avoids having to port a fused Q8 tile
// kernel for prefill. Writes (B, H, T_sel, D) F32.
//
// Grid:  (D / 32, H * T_sel, B)
// Block: 32 threads
// ---------------------------------------------------------------------------
extern "C" __global__ void dequantize_q8_0_kv_f32(
    const void  * __restrict__ cache,    // (B, H, T_max, D/32) block_q8_0
    float       * __restrict__ out,      // (B, H, T_sel, D)
    const int B,
    const int H,
    const int T_sel,
    const int D,
    const int T_max,
    const int t_offset
) {
    const int tid   = threadIdx.x;
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_rel = bh_t / H;
    const int h     = bh_t % H;
    const int t_abs = t_offset + t_rel;
    const int d     = ib_d * QK8_0 + tid;

    const int blocks_per_row = D / QK8_0;
    const block_q8_0_kv * blk = (const block_q8_0_kv *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    const float scale = __half2float(blk->d);
    const float v = scale * (float)blk->qs[tid];

    const int64_t out_idx =
        ((int64_t)((b * H + h) * T_sel + t_rel)) * D + d;
    out[out_idx] = v;
}
