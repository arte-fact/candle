// Flash-attention forward — minimal tile kernel for gfx906.
//
// Replaces the rocBLAS-based `matmul + softmax + mask + matmul` chain
// in candle-transformers'
// `quantized_blocks/attention.rs::gqa_attention` with a single-pass
// online-softmax kernel. Targets llama-class dense models where
// attention is the dominant category (TinyLlama, gemma4). For
// qwen35-9B the savings are marginal (rocBLAS attention already
// collapsed via the GQA zero-copy reshape in P1 Phase 1).
//
// Phase 1 scope:
//   - f32 Q, K, V, output
//   - GQA via explicit `n_rep = n_head / n_kv_head`, llama-style
//     division (`q_head / n_rep → kv_head`)
//   - Optional additive mask (causal / sliding)
//   - Head dim `D` as a template parameter; this file instantiates
//     `D = 64` (TinyLlama / gemma4) and `D = 128` (qwen35/llama-70B)
//   - BR = 1 row of Q per warp, BC = 64 cols of K/V per iteration
//   - Block = 1 Wave64 (64 threads). Grid = (L_q, n_head, B).
//
// Algorithm — online softmax (flash-attention-1):
//
//     m_i = -inf, l_i = 0, O_i = 0
//     for each K/V chunk of BC cols:
//         load K_tile, V_tile into LDS (collaboratively by the warp)
//         for each j in 0..BC:
//             s_j = scale * (q · K_tile[j]) + mask[j]     // warp reduce
//             m_new = max(m_i, s_j)
//             alpha = exp(m_i - m_new)
//             p     = exp(s_j - m_new)
//             O_i   = alpha * O_i + p * V_tile[j]          // per-d update
//             l_i   = alpha * l_i + p
//             m_i   = m_new
//     O_i /= l_i
//     write O_i to out
//
// Each lane owns one output d (lane_id in [0, D)). The warp-reduce
// across d lanes gives the per-j scalar s_j, broadcast to all lanes.
// Per-lane state: m_i, l_i, o_d, q_d. LDS holds the current K/V tile.

#include "compatibility.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

static __device__ __forceinline__ float flash_warp_reduce_sum_f32(float x) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}

template <int D>
static __device__ __forceinline__ void flash_attn_fwd_impl(
    const float * __restrict__ q,    // (B, n_head, L_q, D)
    const float * __restrict__ k,    // (B, n_kv_head, L_k, D)
    const float * __restrict__ v,    // (B, n_kv_head, L_k, D)
    const float * __restrict__ mask, // (B|1, 1, L_q|1, L_k) or null
    float * __restrict__ out,        // (B, n_head, L_q, D)
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int mask_l_q_stride) {

    static_assert(D == 64 || D == 128, "only D=64 or D=128 supported");
    static_assert(D <= WARP_SIZE || D == 2 * WARP_SIZE,
                  "D must be WARP_SIZE or 2*WARP_SIZE");

    const int q_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int lane  = threadIdx.x;

    if (q_idx >= L_q || h_idx >= n_head) {
        return;
    }

    // llama-style GQA: q_head / n_rep = kv_head.
    const int h_kv = n_rep > 1 ? (h_idx / n_rep) : h_idx;

    // Per-(b, head, q_idx) slice base pointers.
    const int64_t q_offset = ((int64_t)(b_idx * n_head) + h_idx) * L_q * D + (int64_t)q_idx * D;
    const int64_t kv_offset = ((int64_t)(b_idx * n_kv_head) + h_kv) * L_k * D;
    const int64_t out_offset = ((int64_t)(b_idx * n_head) + h_idx) * L_q * D + (int64_t)q_idx * D;

    const float * q_ptr = q + q_offset;
    const float * k_ptr = k + kv_offset;
    const float * v_ptr = v + kv_offset;
    float *       o_ptr = out + out_offset;

    // Mask ptr: caller passes `mask_b_stride` (0 if the mask is
    // broadcast across the B dim, L_q*L_k otherwise) and
    // `mask_l_q_stride` (0 if the mask has L_q=1 broadcast,
    // `L_k` otherwise). Shapes supported:
    //   (1, 1,  1,  L_k) → mask_b_stride=0, mask_l_q_stride=0
    //   (1, 1, L_q, L_k) → mask_b_stride=0, mask_l_q_stride=L_k
    //   (B, 1,  1,  L_k) → mask_b_stride=L_k, mask_l_q_stride=0
    //   (B, 1, L_q, L_k) → mask_b_stride=L_q*L_k, mask_l_q_stride=L_k
    const float * mask_row_ptr = nullptr;
    if (mask != nullptr) {
        mask_row_ptr = mask + (int64_t)b_idx * (int64_t)mask_b_stride
                            + (int64_t)q_idx * (int64_t)mask_l_q_stride;
    }

    // --- Load Q row into per-lane registers. ---
    // For D = 64, one float per lane exactly.
    // For D = 128, two floats per lane (d0 = lane, d1 = lane + 64).
    float q_reg0 = 0.0f, q_reg1 = 0.0f;
    float o_reg0 = 0.0f, o_reg1 = 0.0f;
    if (D == 64) {
        q_reg0 = q_ptr[lane];
    } else {
        // D == 128
        q_reg0 = q_ptr[lane];
        q_reg1 = q_ptr[lane + WARP_SIZE];
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // No LDS staging: each K/V element is used exactly once per (b,h,q)
    // block so LDS buys nothing, and 32 KiB LDS/block would cap occupancy
    // at 2 waves/CU on gfx906. Direct global loads coalesce across the
    // 64 lanes of the wave (1 cache line per row per dim slab) and free
    // us to run many more concurrent wavefronts per CU.
    //
    // Iterate j over all K/V rows directly; scalar inner loop, no
    // chunking, no LDS.
    for (int j = 0; j < L_k; ++j) {
        // Dot product q · K[j] across D lanes, reduce within the wave.
        float partial;
        if (D == 64) {
            partial = q_reg0 * k_ptr[j * D + lane];
        } else {
            partial = q_reg0 * k_ptr[j * D + lane]
                    + q_reg1 * k_ptr[j * D + lane + WARP_SIZE];
        }
        float s_j = flash_warp_reduce_sum_f32(partial);
        s_j *= scale;

        // Additive mask (caller passes -inf for blocked positions).
        if (mask_row_ptr != nullptr) {
            s_j += mask_row_ptr[j];
        }

        // Online softmax update. `__expf` is the fast hardware variant
        // (identical to expf for the values we see, ~10× faster).
        const float m_new = fmaxf(m_i, s_j);
        const float alpha = __expf(m_i - m_new);
        const float p     = __expf(s_j - m_new);

        if (D == 64) {
            o_reg0 = alpha * o_reg0 + p * v_ptr[j * D + lane];
        } else {
            o_reg0 = alpha * o_reg0 + p * v_ptr[j * D + lane];
            o_reg1 = alpha * o_reg1 + p * v_ptr[j * D + lane + WARP_SIZE];
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    // --- Final normalisation and write back. ---
    const float inv_l = 1.0f / l_i;
    if (D == 64) {
        o_ptr[lane] = o_reg0 * inv_l;
    } else {
        o_ptr[lane]             = o_reg0 * inv_l;
        o_ptr[lane + WARP_SIZE] = o_reg1 * inv_l;
    }
}

extern "C" __global__ void flash_attn_fwd_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride) {
    flash_attn_fwd_impl<64>(q, k, v, mask, out, B, n_head, n_kv_head,
                             L_q, L_k, scale, n_rep,
                             mask_b_stride, mask_l_q_stride);
}

extern "C" __global__ void flash_attn_fwd_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride) {
    flash_attn_fwd_impl<128>(q, k, v, mask, out, B, n_head, n_kv_head,
                              L_q, L_k, scale, n_rep,
                              mask_b_stride, mask_l_q_stride);
}
