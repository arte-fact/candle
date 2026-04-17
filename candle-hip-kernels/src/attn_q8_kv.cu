// Q8_0 KV-cache attention kernels for the decomposed decode path.
//
// Paired with candle-core/src/hip_backend/q8_kv_cache.rs and the
// F32-KV kernels in flash_attn_v2.cu (gqa_decode_gemv_{qk,v}_b128).
//
// Cache layout: (B, H_kv, T_max, D/32) of block_q8_0 = {half d; int8 qs[32]}.
// D-contiguous within a row → coalesced per-t loads across the warp.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "gfx906_primitives.cuh"

#define QK8_0 32

typedef struct {
    __half  d;
    int8_t  qs[QK8_0];
} block_q8_0_kv;

// ---------------------------------------------------------------------------
// gqa_decode_gemv_v_q8_b128_impl — V·attn gemv with Q8_0 V cache.
//
// Mirror of gqa_decode_gemv_v_b128_impl in flash_attn_v2.cu, but reads
// V from a packed Q8_0 slab. Each block owns 128 consecutive output d's
// (4 Q8 blocks along D). Each thread owns ONE output d.
// ---------------------------------------------------------------------------
template <int D>
static __device__ __forceinline__ void gqa_decode_gemv_v_q8_impl(
    const float * __restrict__ attn,        // (B, H_q, 1, L_k) post-softmax
    const void  * __restrict__ v_cache,     // (B, H_kv, T_max, D/32) Q8 blocks
    float       * __restrict__ out,         // (B, H_q, 1, D)
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    constexpr int BLOCK_SIZE = 128;
    constexpr int D_CHUNK    = BLOCK_SIZE;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;
    static_assert(D % D_CHUNK == 0, "D must be a multiple of D_CHUNK");

    const int d_chunk_idx = blockIdx.x;
    const int h_q         = blockIdx.y;
    const int b           = blockIdx.z;
    const int tid         = threadIdx.x;

    const int h_kv         = (n_rep > 1) ? (h_q / n_rep) : h_q;
    const int d            = d_chunk_idx * D_CHUNK + tid;
    // d → (block-index along D, lane within that 32-elem block)
    const int block_id_d   = d >> 5;        // d / 32
    const int lane_in_blk  = d & 31;        // d % 32

    const int64_t attn_off = ((int64_t)(b * H_q) + h_q) * L_k;
    const int64_t out_off  = ((int64_t)(b * H_q) + h_q) * D + (int64_t)d;

    const float * attn_ptr = attn + attn_off;
    float       * out_ptr  = out + out_off;

    // Pointer to the row of Q8 blocks for (b, h_kv) — step along t by
    // `BLOCKS_PER_ROW` blocks per token.
    const block_q8_0_kv * v_row_base =
        (const block_q8_0_kv *) v_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + block_id_d;

    // Preload attn row into LDS (same reason as the F32 kernel:
    // amortise per-t HBM load across 128 threads).
    extern __shared__ float attn_lds[];
    for (int t = tid; t < L_k; t += BLOCK_SIZE) {
        attn_lds[t] = attn_ptr[t];
    }
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < L_k; ++t) {
        const block_q8_0_kv * blk = v_row_base + (int64_t)t * BLOCKS_PER_ROW;
        const float scale = __half2float(blk->d);
        const float v_val = scale * (float)blk->qs[lane_in_blk];
        acc += attn_lds[t] * v_val;
    }

    *out_ptr = acc;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_v_q8_d128_f32(
    const float * __restrict__ attn, const void * __restrict__ v_cache,
    float * __restrict__ out,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_v_q8_impl<128>(attn, v_cache, out, L_k, T_max, H_q, H_kv, n_rep);
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_v_q8_d256_f32(
    const float * __restrict__ attn, const void * __restrict__ v_cache,
    float * __restrict__ out,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_v_q8_impl<256>(attn, v_cache, out, L_k, T_max, H_q, H_kv, n_rep);
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_v_q8_d512_f32(
    const float * __restrict__ attn, const void * __restrict__ v_cache,
    float * __restrict__ out,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_v_q8_impl<512>(attn, v_cache, out, L_k, T_max, H_q, H_kv, n_rep);
}
