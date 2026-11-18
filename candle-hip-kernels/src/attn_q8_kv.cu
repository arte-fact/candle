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

// Local dp4a helper (quantized.cu defines its own; keep a lightweight
// mirror here so attn_q8_kv.cu doesn't need to include quantized.cu).
static __device__ __forceinline__ int attn_q8_dp4a(const int a, const int b, int c) {
#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__))
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i) r += (int)a8[i] * (int)b8[i];
    return r;
#endif
}

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

// ---------------------------------------------------------------------------
// D=64 V·attn variant. Uses a 64-thread block (one per output d) since
// D=64 < 128. Each block owns ALL 64 output d positions for one (h_q, b).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64, 2)
gqa_decode_gemv_v_q8_d64_f32(
    const float * __restrict__ attn, const void * __restrict__ v_cache,
    float * __restrict__ out,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    constexpr int D = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;    // = 2

    const int h_q = blockIdx.y;
    const int b   = blockIdx.z;
    const int tid = threadIdx.x;

    const int h_kv        = (n_rep > 1) ? (h_q / n_rep) : h_q;
    const int d           = tid;                 // one thread per d
    const int block_id_d  = d >> 5;              // 0 or 1
    const int lane_in_blk = d & 31;

    const int64_t attn_off = ((int64_t)(b * H_q) + h_q) * L_k;
    const int64_t out_off  = ((int64_t)(b * H_q) + h_q) * D + (int64_t)d;
    const float * attn_ptr = attn + attn_off;
    float       * out_ptr  = out + out_off;

    const block_q8_0_kv * v_row_base =
        (const block_q8_0_kv *) v_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + block_id_d;

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

// ---------------------------------------------------------------------------
// gqa_decode_gemv_qk_q8_impl — Q·K gemv with Q8_0 K cache.
//
// Mirror of gqa_decode_gemv_qk_b128_impl in flash_attn_v2.cu, reading K
// from a packed Q8_0 slab (B, H_kv, T_max, D/32). Grid: (L_k, H_q, B);
// each 128-thread block dots Q[b,h_q,:] · K[b,h_kv,t,:].
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float q8_warp_reduce_sum(float x) {
    #pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}

template <int D>
static __device__ __forceinline__ void gqa_decode_gemv_qk_q8_impl(
    const float * __restrict__ q,          // (B, H_q, 1, D)
    const void  * __restrict__ k_cache,    // (B, H_kv, T_max, D/32) Q8
    float       * __restrict__ scores,     // (B, H_q, 1, L_k)
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    constexpr int BLOCK_SIZE     = 128;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;
    constexpr int F_PER_THREAD   = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static_assert(D % QK8_0 == 0, "D must be multiple of QK8_0");

    const int t   = blockIdx.x;            // K-sequence position
    const int h_q = blockIdx.y;
    const int b   = blockIdx.z;
    const int tid = threadIdx.x;

    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    const float * q_ptr = q + ((int64_t)(b * H_q) + h_q) * D;
    const block_q8_0_kv * k_row_t = (const block_q8_0_kv *) k_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + (int64_t)t * BLOCKS_PER_ROW;

    float partial = 0.0f;
    #pragma unroll
    for (int i = 0; i < F_PER_THREAD; ++i) {
        const int d = i * BLOCK_SIZE + tid;
        if (d < D) {
            const int blk_d = d >> 5;          // d / 32
            const int lane  = d & 31;          // d % 32
            const block_q8_0_kv * kblk = k_row_t + blk_d;
            const float k_scale = __half2float(kblk->d);
            const float k_val   = k_scale * (float)kblk->qs[lane];
            partial += q_ptr[d] * k_val;
        }
    }

    // 64-lane wave reduce + 2-wave LDS combine (same pattern as the
    // F32 kernel in flash_attn_v2.cu).
    partial = q8_warp_reduce_sum(partial);
    __shared__ float smem[2];
    const int wave_id      = tid >> 6;
    const int lane_in_wave = tid & 63;
    if (lane_in_wave == 0) {
        smem[wave_id] = partial;
    }
    __syncthreads();
    if (tid == 0) {
        const float final_sum = smem[0] + smem[1];
        scores[((int64_t)(b * H_q) + h_q) * L_k + t] = final_sum;
    }
}

// D=64 Q·K variant. BLOCK_SIZE=64 (single wave, one thread per d).
// Uses only a wave-local DPP reduce — no cross-wave LDS combine needed
// since the whole block fits in one wave.
extern "C" __global__ void __launch_bounds__(64, 2)
gqa_decode_gemv_qk_q8_d64_f32(
    const float * __restrict__ q, const void * __restrict__ k_cache,
    float * __restrict__ scores,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    constexpr int D = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;    // = 2

    const int t   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int b   = blockIdx.z;
    const int tid = threadIdx.x;

    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    const float * q_ptr = q + ((int64_t)(b * H_q) + h_q) * D;
    const block_q8_0_kv * k_row_t = (const block_q8_0_kv *) k_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + (int64_t)t * BLOCKS_PER_ROW;

    const int d           = tid;
    const int blk_d       = d >> 5;
    const int lane_in_blk = d & 31;
    const block_q8_0_kv * kblk = k_row_t + blk_d;
    const float k_scale = __half2float(kblk->d);
    const float k_val   = k_scale * (float)kblk->qs[lane_in_blk];
    float partial = q_ptr[d] * k_val;

    // 64-lane wave-local reduction (whole block fits in one wave).
    partial = q8_warp_reduce_sum(partial);
    if (tid == 0) {
        scores[((int64_t)(b * H_q) + h_q) * L_k + t] = partial;
    }
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_qk_q8_d128_f32(
    const float * __restrict__ q, const void * __restrict__ k_cache,
    float * __restrict__ scores,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_qk_q8_impl<128>(q, k_cache, scores, L_k, T_max, H_q, H_kv, n_rep);
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_qk_q8_d256_f32(
    const float * __restrict__ q, const void * __restrict__ k_cache,
    float * __restrict__ scores,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_qk_q8_impl<256>(q, k_cache, scores, L_k, T_max, H_q, H_kv, n_rep);
}

extern "C" __global__ void __launch_bounds__(128, 1)
gqa_decode_gemv_qk_q8_d512_f32(
    const float * __restrict__ q, const void * __restrict__ k_cache,
    float * __restrict__ scores,
    int L_k, int T_max, int H_q, int H_kv, int n_rep)
{
    gqa_decode_gemv_qk_q8_impl<512>(q, k_cache, scores, L_k, T_max, H_q, H_kv, n_rep);
}

// ---------------------------------------------------------------------------
// gqa_decode_mv_fast_q8_impl — SINGLE-KERNEL fused Q·K + online-softmax + V·attn
// over a Q8_0 KV cache. Mirrors the F32 `gqa_decode_mv_fast_impl` in
// flash_attn_v2.cu but loads int8 blocks directly: no F32 dequant alloc,
// no multi-kernel dispatch. Targets gemma4 decode where the F32 fused
// kernel already beats rocBLAS.
//
// Block: 64 threads (single wave). Grid: (1, H_q, B).
//
// Per-lane layout (v2, 2026-04-19):
//   Lane `l` owns the CONTIGUOUS d range [F_PER_LANE·l, F_PER_LANE·l+F_PER_LANE).
//   With F_PER_LANE ≤ 32 (= QK8_0) for all supported D, each lane's range
//   fits entirely within ONE block_q8_0 → a single scale half-load covers
//   all F_PER_LANE elements per K per t (and per V per t). Contrast with
//   v1's strided layout d ∈ {l, l+64, l+128, l+192} which crossed 4 blocks
//   per lane per t and burned VGPRs on 4 live scales + 4 byte-indices —
//   measured 61 VGPR → 4 waves/SIMD, 5× slower than dequant+F32-fused.
//   v2 expected VGPR ≤ 35 → 7 waves/SIMD (still less than F32's 8 but
//   closer). Plus compiler should merge 4 sequential byte loads into a
//   single int32 load for the qs array.
// ---------------------------------------------------------------------------
template <int D>
static __device__ __forceinline__ void gqa_decode_mv_fast_q8_impl(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep,
    int mask_b_stride, int L_k_iter)
{
    constexpr int BLOCK_SIZE     = 64;
    constexpr int F_PER_LANE     = D / BLOCK_SIZE;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;
    static_assert(D % QK8_0 == 0, "D must be multiple of QK8_0");
    static_assert(D % BLOCK_SIZE == 0, "D must be multiple of 64");
    static_assert(F_PER_LANE <= QK8_0, "F_PER_LANE must fit in one block");

    const int h_q  = blockIdx.y;
    const int b    = blockIdx.z;
    const int lane = threadIdx.x;
    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    // Contiguous-per-lane layout. Each lane's F_PER_LANE d values live in
    // a single block_q8_0 → one scale load per K per t, ditto V.
    const int d_base   = F_PER_LANE * lane;
    const int blk_d    = d_base / QK8_0;
    const int byte_off = d_base - blk_d * QK8_0;

    const float * q_ptr = q + ((int64_t)(b * H_q) + h_q) * D + d_base;
    float q_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        q_reg[i] = q_ptr[i];
    }

    float o_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) o_reg[i] = 0.0f;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const block_q8_0_kv * k_base = (const block_q8_0_kv *) k_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + blk_d;
    const block_q8_0_kv * v_base = (const block_q8_0_kv *) v_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + blk_d;

    const float * mask_row = (mask != nullptr)
        ? mask + (int64_t)b * (int64_t)mask_b_stride
        : nullptr;

    const int L_k_eff = (L_k_iter >= 0) ? L_k_iter : L_k;

    for (int t = 0; t < L_k_eff; ++t) {
        const block_q8_0_kv * k_blk = k_base + (int64_t)t * BLOCKS_PER_ROW;
        const block_q8_0_kv * v_blk = v_base + (int64_t)t * BLOCKS_PER_ROW;

        // ONE scale per lane per t per K (and per V) — shared across all
        // F_PER_LANE elements of the lane. v1 had F_PER_LANE separate
        // half-loads per t which burned VGPRs on 4 live scales.
        const float ks = __half2float(k_blk->d);
        const float vs = __half2float(v_blk->d);

        // F_PER_LANE contiguous int8 bytes — compiler should merge to
        // one int32 load at F_PER_LANE==4 (D=256) or one int16 at
        // F_PER_LANE==2 (D=128).
        float k_vals[F_PER_LANE];
        float v_vals[F_PER_LANE];
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            k_vals[i] = ks * (float)k_blk->qs[byte_off + i];
            v_vals[i] = vs * (float)v_blk->qs[byte_off + i];
        }

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            partial += q_reg[i] * k_vals[i];
        }
        float s_t = gfx906_warp_reduce_sum(partial) * scale;
        if (mask_row != nullptr) s_t += mask_row[t];

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
    float * o_ptr = out + ((int64_t)(b * H_q) + h_q) * D + d_base;
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        o_ptr[i] = o_reg[i] * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_q8_d128_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_q8_impl<128>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_q8_d256_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_q8_impl<256>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_q8_d512_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_q8_impl<512>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride, L_k_iter);
}

// ---------------------------------------------------------------------------
// gqa_decode_mv_fast_q8_dp4a_impl — v3 with INT8 dp4a Q·K path.
//
// P10 / P6 v3 (2026-04-19): the v2 kernel reaches matching occupancy (29
// VGPR = 8 waves/SIMD at D=256) but loses 4× per-kernel to F32-fused
// because every inner-loop iteration pays int8→float + scale_mul per K
// element. Match turbo's `fattn-q8.cuh` strategy: quantise Q to int8 once
// at kernel entry (single warp-wide scale), then Q·K uses dp4a int8×int8
// → int32 accumulator. Single float convert at block boundary.
//
// Layout (F_PER_LANE == 4 case, i.e. D=256):
//   Lane l owns d ∈ [4l .. 4l+3] contiguous → 4 int8 bytes = 1 int32 per
//   lane per t. One dp4a per lane per t. Per-warp reduce → single float
//   per step.
//
// V·attn stays F32 (post-softmax probs aren't amenable to int8 without
// per-row scale, and the V read dominates if we requantised each step).
//
// Supported D: 256 and 512 only (F_PER_LANE multiple of 4 for clean
// int32 loads). D=128 falls back to the v2 path via the Rust dispatcher.
// ---------------------------------------------------------------------------
template <int D>
static __device__ __forceinline__ void gqa_decode_mv_fast_q8_dp4a_impl(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep,
    int mask_b_stride, int L_k_iter)
{
    constexpr int BLOCK_SIZE     = 64;
    constexpr int F_PER_LANE     = D / BLOCK_SIZE;
    constexpr int BLOCKS_PER_ROW = D / QK8_0;
    constexpr int INTS_PER_LANE  = F_PER_LANE / 4;  // 1 for D=256, 2 for D=512
    static_assert(F_PER_LANE % 4 == 0, "dp4a variant needs F_PER_LANE multiple of 4");
    static_assert(F_PER_LANE <= QK8_0, "F_PER_LANE must fit in one block");

    const int h_q  = blockIdx.y;
    const int b    = blockIdx.z;
    const int lane = threadIdx.x;
    const int h_kv = (n_rep > 1) ? (h_q / n_rep) : h_q;

    // Contiguous-per-lane layout, same as v2.
    const int d_base   = F_PER_LANE * lane;
    const int blk_d    = d_base / QK8_0;
    const int byte_off = d_base - blk_d * QK8_0;

    // ---- Step 1: Load Q and compute PER-BLOCK scale. ----
    // Each K block covers 32 D-elements. Q_scale is per-block so the dp4a
    // int32 accumulator can be converted to float with a tight scale —
    // single warp-wide scale gave ~2e-3 abs error (too loose for softmax
    // on closely-spaced scores).
    const float * q_ptr = q + ((int64_t)(b * H_q) + h_q) * D + d_base;
    float q_reg[F_PER_LANE];
    float q_max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        q_reg[i] = q_ptr[i];
        q_max_abs = fmaxf(q_max_abs, fabsf(q_reg[i]));
    }
    // Reduce max|q| within this lane's K block. LANES_PER_BLOCK = QK8_0 /
    // F_PER_LANE (8 for D=256, 4 for D=512). Use __shfl_xor with width =
    // LANES_PER_BLOCK so the shuffle stays inside each block's lane group.
    constexpr int LANES_PER_BLOCK = QK8_0 / F_PER_LANE;
    static_assert(LANES_PER_BLOCK == 4 || LANES_PER_BLOCK == 8,
                  "dp4a variant only supports LANES_PER_BLOCK ∈ {4, 8}");
    if constexpr (LANES_PER_BLOCK >= 2) {
        q_max_abs = fmaxf(q_max_abs, __shfl_xor(q_max_abs, 1, LANES_PER_BLOCK));
    }
    if constexpr (LANES_PER_BLOCK >= 4) {
        q_max_abs = fmaxf(q_max_abs, __shfl_xor(q_max_abs, 2, LANES_PER_BLOCK));
    }
    if constexpr (LANES_PER_BLOCK >= 8) {
        q_max_abs = fmaxf(q_max_abs, __shfl_xor(q_max_abs, 4, LANES_PER_BLOCK));
    }
    const float q_scale = q_max_abs / 127.0f;
    const float q_scale_inv = q_max_abs > 0.0f ? 127.0f / q_max_abs : 0.0f;

    // ---- Step 2: Quantise Q to int8 packed as int32 per INTS_PER_LANE. ----
    int q_packed[INTS_PER_LANE];
    #pragma unroll
    for (int p = 0; p < INTS_PER_LANE; ++p) {
        int word = 0;
        #pragma unroll
        for (int b_idx = 0; b_idx < 4; ++b_idx) {
            const int i = p * 4 + b_idx;
            const int q8 = __float2int_rn(q_reg[i] * q_scale_inv);
            // Clamp to int8 range; round-to-nearest can produce 128.
            const int q8c = q8 > 127 ? 127 : (q8 < -128 ? -128 : q8);
            word |= (q8c & 0xff) << (8 * b_idx);
        }
        q_packed[p] = word;
    }

    // ---- Step 3: Online-softmax main loop. V·attn stays F32. ----
    float o_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) o_reg[i] = 0.0f;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const block_q8_0_kv * k_base = (const block_q8_0_kv *) k_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + blk_d;
    const block_q8_0_kv * v_base = (const block_q8_0_kv *) v_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW
        + blk_d;

    const float * mask_row = (mask != nullptr)
        ? mask + (int64_t)b * (int64_t)mask_b_stride
        : nullptr;
    const int L_k_eff = (L_k_iter >= 0) ? L_k_iter : L_k;

    for (int t = 0; t < L_k_eff; ++t) {
        const block_q8_0_kv * k_blk = k_base + (int64_t)t * BLOCKS_PER_ROW;
        const block_q8_0_kv * v_blk = v_base + (int64_t)t * BLOCKS_PER_ROW;

        const float ks = __half2float(k_blk->d);
        const float vs = __half2float(v_blk->d);

        // Load K as INTS_PER_LANE int32 words (int8 quads) and dp4a.
        int acc_int = 0;
        #pragma unroll
        for (int p = 0; p < INTS_PER_LANE; ++p) {
            // Safe: block_q8_0_kv.qs is int8[32], aligned; byte_off is a
            // multiple of 4 for supported (D,F_PER_LANE) (lane * 4 offset).
            const int k_word = *reinterpret_cast<const int*>(&k_blk->qs[byte_off + p * 4]);
            acc_int = attn_q8_dp4a(q_packed[p], k_word, acc_int);
        }

        // Per-lane partial: apply PER-BLOCK scales (q_scale and ks are the
        // scales for THIS lane's block) before warp reduction. This composes
        // per-block scales correctly across the full Q·K row dot.
        const float lane_partial = (float)acc_int * (q_scale * ks);
        float s_t = gfx906_warp_reduce_sum(lane_partial) * scale;
        if (mask_row != nullptr) s_t += mask_row[t];

        const float m_new = fmaxf(m_i, s_t);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p_w   = gfx906_fast_exp(s_t - m_new);

        // V·attn: keep F32 path (same as v2 — probs aren't int8-friendly).
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            const float v_elem = vs * (float)v_blk->qs[byte_off + i];
            o_reg[i] = alpha * o_reg[i] + p_w * v_elem;
        }
        l_i = alpha * l_i + p_w;
        m_i = m_new;
    }

    const float inv_l = gfx906_rcp(l_i);
    float * o_ptr = out + ((int64_t)(b * H_q) + h_q) * D + d_base;
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        o_ptr[i] = o_reg[i] * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_q8_dp4a_d256_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_q8_dp4a_impl<256>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_q8_dp4a_d512_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_q8_dp4a_impl<512>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride, L_k_iter);
}

// ---------------------------------------------------------------------------
// F4-M1: LDS-tile-Q fused Q8-KV attention kernel.
//
// Structural difference from gqa_decode_mv_fast_q8_impl (single-Q):
//   - Block handles ONE (batch, kv_head); tiles NCOLS Q heads together
//     (the GQA group that shares this KV head).
//   - Grid: (1, H_kv, B)   block: NCOLS × WARP_SIZE
//   - Warp w (∈ [0,NCOLS)) is assigned to Q head h_q = h_kv*n_rep + w.
//   - Each warp lane owns F_PER_LANE = D/WARP_SIZE contiguous d values.
//   - K and V HBM loads are issued by every warp but hit L1 after the
//     first warp: 4× fewer distinct CU dispatches + shared L1 lines.
//
// M1 scope: D=128, NCOLS=4 ONLY (Devstral/Gemma-31B GQA ratio).
// No dp4a yet — float Q·K math. Correctness-only. Perf is M2's decision.
// ---------------------------------------------------------------------------
template <int D, int NCOLS>
static __device__ __forceinline__ void gqa_decode_fattn_tile_q8_impl(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride)
{
    static_assert(D == 128, "F4-M1: D=128 only");
    static_assert(NCOLS == 4, "F4-M1: NCOLS=4 only");
    constexpr int WARP_SIZE_F = 64;
    constexpr int F_PER_LANE  = D / WARP_SIZE_F;   // = 2
    constexpr int BLOCKS_PER_ROW = D / QK8_0;      // = 4 Q8 blocks per D row

    const int b    = blockIdx.z;
    const int h_kv = blockIdx.y;
    const int tid  = threadIdx.x;
    const int w    = tid / WARP_SIZE_F;            // 0..NCOLS-1 → Q row
    const int lane = tid % WARP_SIZE_F;            // 0..63

    // NCOLS must match n_rep (host-side guard); otherwise skip stale warps.
    if (w >= n_rep) return;
    const int h_q = h_kv * n_rep + w;

    const int d_base   = F_PER_LANE * lane;        // 0..126 (even)
    const int blk_d    = d_base / QK8_0;           // 0..3
    const int byte_off = d_base - blk_d * QK8_0;

    // Load this warp's Q row into registers.
    const float * q_ptr = q + ((int64_t)(b * H_q) + h_q) * D + d_base;
    float q_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) q_reg[i] = q_ptr[i];

    // Accumulators (per-warp online softmax state).
    float o_reg[F_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) o_reg[i] = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // KV cache base pointer for (b, h_kv). SAME ptr across all warps in this
    // block — L1 coalesces the 4-warp repeat loads.
    const block_q8_0_kv * k_base = (const block_q8_0_kv *) k_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW + blk_d;
    const block_q8_0_kv * v_base = (const block_q8_0_kv *) v_cache
        + (int64_t)(b * H_kv + h_kv) * T_max * BLOCKS_PER_ROW + blk_d;

    const float * mask_row = (mask != nullptr)
        ? mask + (int64_t)b * (int64_t)mask_b_stride
        : nullptr;

    for (int t = 0; t < L_k; ++t) {
        const block_q8_0_kv * k_blk = k_base + (int64_t)t * BLOCKS_PER_ROW;
        const block_q8_0_kv * v_blk = v_base + (int64_t)t * BLOCKS_PER_ROW;

        const float ks = __half2float(k_blk->d);
        const float vs = __half2float(v_blk->d);

        float k_vals[F_PER_LANE];
        float v_vals[F_PER_LANE];
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            k_vals[i] = ks * (float)k_blk->qs[byte_off + i];
            v_vals[i] = vs * (float)v_blk->qs[byte_off + i];
        }

        // Q·K partial for this warp's Q row.
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < F_PER_LANE; ++i) {
            partial += q_reg[i] * k_vals[i];
        }
        float s_t = gfx906_warp_reduce_sum(partial) * scale;
        if (mask_row != nullptr) s_t += mask_row[t];

        // Online softmax update (per-warp state — independent across warps).
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
    float * o_ptr = out + ((int64_t)(b * H_q) + h_q) * D + d_base;
    #pragma unroll
    for (int i = 0; i < F_PER_LANE; ++i) {
        o_ptr[i] = o_reg[i] * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
gqa_decode_fattn_tile_q8_d128_n4_f32(
    const float * __restrict__ q,
    const void  * __restrict__ k_cache,
    const void  * __restrict__ v_cache,
    const float * __restrict__ mask,
    float       * __restrict__ out,
    int H_q, int H_kv, int L_k, int T_max,
    float scale, int n_rep, int mask_b_stride)
{
    gqa_decode_fattn_tile_q8_impl<128, 4>(q, k_cache, v_cache, mask, out,
        H_q, H_kv, L_k, T_max, scale, n_rep, mask_b_stride);
}
