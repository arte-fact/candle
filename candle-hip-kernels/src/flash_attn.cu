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
#include "gfx906_primitives.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Use DPP-fused warp reduction from gfx906_primitives.cuh
static __device__ __forceinline__ float flash_warp_reduce_sum_f32(float x) {
    return gfx906_warp_reduce_sum(x);
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
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_j - m_new);

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
    const float inv_l = gfx906_rcp(l_i);
    if (D == 64) {
        o_ptr[lane] = o_reg0 * inv_l;
    } else {
        o_ptr[lane]             = o_reg0 * inv_l;
        o_ptr[lane + WARP_SIZE] = o_reg1 * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 1) flash_attn_fwd_d64_f32(
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

extern "C" __global__ void __launch_bounds__(64, 1) flash_attn_fwd_d128_f32(
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

// ============================================================================
// Decode attention with strided V — for KvCache pre-allocated buffers.
// ============================================================================
//
// Same as flash_attn_fwd_impl but V is accessed via explicit strides:
//   V[head][j][d] = v_ptr[head * v_head_stride + j * v_stride_j + d * v_stride_d]
//
// This handles V stored as (D, max_T) per head with:
//   v_stride_j = 1, v_stride_d = max_T (transposed storage)
// OR V stored as (T, D) per head with:
//   v_stride_j = D, v_stride_d = 1 (standard storage)
//
// K is always standard layout (D, T) per head with k_ptr[j * D + d].
// K may also be transposed: k_ptr[d * max_T + j] with k_stride_j and k_stride_d.

template <int D>
static __device__ __forceinline__ void flash_attn_decode_strided_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d) {

    static_assert(D == 64 || D == 128, "only D=64 or D=128 supported");

    const int q_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int lane  = threadIdx.x;

    if (q_idx >= L_q || h_idx >= n_head) return;

    const int h_kv = n_rep > 1 ? (h_idx / n_rep) : h_idx;

    const int64_t q_offset = ((int64_t)(b_idx * n_head) + h_idx) * L_q * D + (int64_t)q_idx * D;
    const int64_t out_offset = q_offset;

    const float * q_ptr = q + q_offset;
    // K and V use explicit strides per head.
    const float * k_base = k + ((int64_t)(b_idx * n_kv_head) + h_kv) * k_head_stride;
    const float * v_base = v + ((int64_t)(b_idx * n_kv_head) + h_kv) * v_head_stride;
    float *       o_ptr = out + out_offset;

    const float * mask_row_ptr = nullptr;
    if (mask != nullptr) {
        mask_row_ptr = mask + (int64_t)b_idx * (int64_t)mask_b_stride
                            + (int64_t)q_idx * (int64_t)mask_l_q_stride;
    }

    float q_reg0 = 0.0f, q_reg1 = 0.0f;
    float o_reg0 = 0.0f, o_reg1 = 0.0f;
    if (D == 64) {
        q_reg0 = q_ptr[lane];
    } else {
        q_reg0 = q_ptr[lane];
        q_reg1 = q_ptr[lane + WARP_SIZE];
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int j = 0; j < L_k; ++j) {
        float partial;
        if (D == 64) {
            partial = q_reg0 * k_base[j * k_stride_j + lane * k_stride_d];
        } else {
            partial = q_reg0 * k_base[j * k_stride_j + lane * k_stride_d]
                    + q_reg1 * k_base[j * k_stride_j + (lane + WARP_SIZE) * k_stride_d];
        }
        float s_j = flash_warp_reduce_sum_f32(partial);
        s_j *= scale;

        if (mask_row_ptr != nullptr) {
            s_j += mask_row_ptr[j];
        }

        const float m_new = fmaxf(m_i, s_j);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_j - m_new);

        if (D == 64) {
            o_reg0 = alpha * o_reg0 + p * v_base[j * v_stride_j + lane * v_stride_d];
        } else {
            o_reg0 = alpha * o_reg0 + p * v_base[j * v_stride_j + lane * v_stride_d];
            o_reg1 = alpha * o_reg1 + p * v_base[j * v_stride_j + (lane + WARP_SIZE) * v_stride_d];
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    const float inv_l = gfx906_rcp(l_i);
    if (D == 64) {
        o_ptr[lane] = o_reg0 * inv_l;
    } else {
        o_ptr[lane]             = o_reg0 * inv_l;
        o_ptr[lane + WARP_SIZE] = o_reg1 * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 1) flash_attn_decode_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d) {
    flash_attn_decode_strided_impl<64>(q, k, v, mask, out, B, n_head, n_kv_head,
        L_q, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d);
}

extern "C" __global__ void __launch_bounds__(64, 1) flash_attn_decode_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int n_head, int n_kv_head, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d) {
    flash_attn_decode_strided_impl<128>(q, k, v, mask, out, B, n_head, n_kv_head,
        L_q, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d);
}

// ============================================================================
// Split-K decode flash attention
// ----------------------------------------------------------------------------
// The single-warp-per-head decode kernel uses only n_head workgroups (32 for
// TinyLlama) which leaves half of gfx906's 60 CUs idle. Split-K partitions
// the L_k dimension across multiple workgroups per head, then a lightweight
// merge kernel combines the per-chunk (o, m, l) partials via the log-sum-exp
// trick.
//
// Phase 1: grid=(num_chunks, n_head, B), block=(64, 1, 1)
//   - Each workgroup handles K/V[chunk_start..chunk_end] for head h, batch b.
//   - Writes partial output (D floats) + m_i + l_i to partial_out.
//
// Phase 2 (merge): grid=(n_head, B, 1), block=(64, 1, 1)
//   - Reads partials from all chunks for this (h, b).
//   - Merges: m_final = max(m_c), l_final = sum(l_c * exp(m_c - m_final)),
//     o_final = sum(o_c * exp(m_c - m_final)) / l_final.
//
// Partial buffer layout: [chunk, B, n_head, D+2] (row-major on last 3 dims).
// The +2 slots store m and l. Indexed as:
//   partial_out[((chunk*B + b)*n_head + h) * (D+2) + slot]
// ============================================================================

template<int D>
static __device__ __forceinline__ void flash_attn_decode_split_k_phase1_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int /*mask_l_q_stride — L_q=1, unused*/,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {

    static_assert(D == 64 || D == 128, "only D=64 or D=128 supported");

    const int chunk_idx = blockIdx.x;
    const int h_idx     = blockIdx.y;
    const int b_idx     = blockIdx.z;
    const int lane      = threadIdx.x;

    if (h_idx >= n_head || chunk_idx >= num_chunks) return;

    // Chunk boundaries along L_k.
    const int chunk_size = (L_k + num_chunks - 1) / num_chunks;
    const int j_start    = chunk_idx * chunk_size;
    const int j_end      = j_start + chunk_size < L_k ? j_start + chunk_size : L_k;
    if (j_start >= L_k) {
        // Empty chunk (more chunks than positions): write sentinel partials.
        const int64_t base = ((((int64_t)chunk_idx * B) + b_idx) * n_head + h_idx) * (D + 2);
        if (D == 64) {
            partial_out[base + lane] = 0.0f;
        } else {
            partial_out[base + lane]            = 0.0f;
            partial_out[base + lane + WARP_SIZE] = 0.0f;
        }
        if (lane == 0) {
            partial_out[base + D]     = -INFINITY;
            partial_out[base + D + 1] = 0.0f;
        }
        return;
    }

    const int h_kv = n_rep > 1 ? (h_idx / n_rep) : h_idx;

    // L_q = 1 for decode. Q offset = (b*n_head + h) * D.
    const int64_t q_offset = ((int64_t)(b_idx * n_head) + h_idx) * D;
    const float * q_ptr = q + q_offset;
    const float * k_base = k + ((int64_t)(b_idx * n_kv_head) + h_kv) * k_head_stride;
    const float * v_base = v + ((int64_t)(b_idx * n_kv_head) + h_kv) * v_head_stride;

    const float * mask_row_ptr = mask ? (mask + (int64_t)b_idx * (int64_t)mask_b_stride) : nullptr;

    float q_reg0, q_reg1 = 0.0f;
    float o_reg0 = 0.0f, o_reg1 = 0.0f;
    if (D == 64) {
        q_reg0 = q_ptr[lane];
    } else {
        q_reg0 = q_ptr[lane];
        q_reg1 = q_ptr[lane + WARP_SIZE];
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int j = j_start; j < j_end; ++j) {
        float partial;
        if (D == 64) {
            partial = q_reg0 * k_base[j * k_stride_j + lane * k_stride_d];
        } else {
            partial = q_reg0 * k_base[j * k_stride_j + lane * k_stride_d]
                    + q_reg1 * k_base[j * k_stride_j + (lane + WARP_SIZE) * k_stride_d];
        }
        float s_j = flash_warp_reduce_sum_f32(partial);
        s_j *= scale;
        if (mask_row_ptr) s_j += mask_row_ptr[j];

        const float m_new = fmaxf(m_i, s_j);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_j - m_new);
        if (D == 64) {
            o_reg0 = alpha * o_reg0 + p * v_base[j * v_stride_j + lane * v_stride_d];
        } else {
            o_reg0 = alpha * o_reg0 + p * v_base[j * v_stride_j + lane * v_stride_d];
            o_reg1 = alpha * o_reg1 + p * v_base[j * v_stride_j + (lane + WARP_SIZE) * v_stride_d];
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    // Write partials (unnormalized). Merge kernel normalizes at the end.
    const int64_t base = ((((int64_t)chunk_idx * B) + b_idx) * n_head + h_idx) * (D + 2);
    if (D == 64) {
        partial_out[base + lane] = o_reg0;
    } else {
        partial_out[base + lane]             = o_reg0;
        partial_out[base + lane + WARP_SIZE] = o_reg1;
    }
    if (lane == 0) {
        partial_out[base + D]     = m_i;
        partial_out[base + D + 1] = l_i;
    }
}

template<int D>
static __device__ __forceinline__ void flash_attn_decode_split_k_merge_impl(
    const float * __restrict__ partial_in,
    float * __restrict__ out,
    int B, int n_head, int num_chunks) {

    static_assert(D == 64 || D == 128, "only D=64 or D=128 supported");

    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int lane  = threadIdx.x;

    // Two passes: first find m_final, then accumulate l_final + o_final.
    float m_final = -INFINITY;
    for (int c = 0; c < num_chunks; ++c) {
        const int64_t base = ((((int64_t)c * B) + b_idx) * n_head + h_idx) * (D + 2);
        const float m_c = partial_in[base + D];
        m_final = fmaxf(m_final, m_c);
    }

    float l_final = 0.0f;
    float o_final0 = 0.0f, o_final1 = 0.0f;
    for (int c = 0; c < num_chunks; ++c) {
        const int64_t base = ((((int64_t)c * B) + b_idx) * n_head + h_idx) * (D + 2);
        const float m_c = partial_in[base + D];
        const float l_c = partial_in[base + D + 1];
        const float w   = gfx906_fast_exp(m_c - m_final);
        l_final += w * l_c;
        if (D == 64) {
            o_final0 += w * partial_in[base + lane];
        } else {
            o_final0 += w * partial_in[base + lane];
            o_final1 += w * partial_in[base + lane + WARP_SIZE];
        }
    }

    const float inv_l = gfx906_rcp(l_final);
    const int64_t out_base = ((int64_t)b_idx * n_head + h_idx) * D;
    if (D == 64) {
        out[out_base + lane] = o_final0 * inv_l;
    } else {
        out[out_base + lane]             = o_final0 * inv_l;
        out[out_base + lane + WARP_SIZE] = o_final1 * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 2) flash_attn_decode_split_k_phase1_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {
    flash_attn_decode_split_k_phase1_impl<64>(q, k, v, mask, partial_out,
        B, n_head, n_kv_head, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d, num_chunks);
}

extern "C" __global__ void __launch_bounds__(64, 2) flash_attn_decode_split_k_phase1_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {
    flash_attn_decode_split_k_phase1_impl<128>(q, k, v, mask, partial_out,
        B, n_head, n_kv_head, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d, num_chunks);
}

extern "C" __global__ void __launch_bounds__(64, 2) flash_attn_decode_split_k_merge_d64_f32(
    const float * __restrict__ partial_in,
    float * __restrict__ out,
    int B, int n_head, int num_chunks) {
    flash_attn_decode_split_k_merge_impl<64>(partial_in, out, B, n_head, num_chunks);
}

extern "C" __global__ void __launch_bounds__(64, 2) flash_attn_decode_split_k_merge_d128_f32(
    const float * __restrict__ partial_in,
    float * __restrict__ out,
    int B, int n_head, int num_chunks) {
    flash_attn_decode_split_k_merge_impl<128>(partial_in, out, B, n_head, num_chunks);
}

// ============================================================================
// LDS-tiled split-K decode flash attention
// ----------------------------------------------------------------------------
// Combines split-K (grid=num_chunks × n_head × B → ~120+ workgroups for gfx906)
// with LDS tiling of K and V. The cooperative tile load serializes per-thread
// but issues coalesced stride=1 loads from global memory, amortising the
// uncoalesced K-layout over many reuse from LDS. Same partial layout + merge
// kernel as the plain split-K variant.
// ============================================================================

template<int D, int BC, int NW = 1>
static __device__ __forceinline__ void flash_attn_decode_split_k_lds_phase1_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int /*mask_l_q_stride*/,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {

    // Only LDS-tile K (its stride pattern is uncoalesced across lanes).
    // V with natural layout (stride[3]=1) is already coalesced → read direct.
    // NW = number of warps per block (1 or 2). With NW=2 all 128 threads
    // cooperate on the K tile load; only warp 0 does the compute.
    static_assert(D == 64 || D == 128, "only D=64 or D=128 supported");
    static_assert(D % WARP_SIZE == 0, "D must be multiple of WARP_SIZE");
    static_assert(NW == 1 || NW == 2, "NW must be 1 or 2");
    constexpr int D_PER_LANE = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = NW * WARP_SIZE;
    constexpr int K_TILE_ELEMS = BC * D;

    const int chunk_idx = blockIdx.x;
    const int h_idx     = blockIdx.y;
    const int b_idx     = blockIdx.z;
    const int lane      = threadIdx.x;
    const int warp      = threadIdx.y;  // 0 or 1
    const int tid       = warp * WARP_SIZE + lane;

    if (h_idx >= n_head || chunk_idx >= num_chunks) return;

    const int chunk_size = (L_k + num_chunks - 1) / num_chunks;
    const int j_start    = chunk_idx * chunk_size;
    const int j_end      = j_start + chunk_size < L_k ? j_start + chunk_size : L_k;

    const int64_t base_out = ((((int64_t)chunk_idx * B) + b_idx) * n_head + h_idx) * (D + 2);

    if (j_start >= L_k) {
        if (warp == 0) {
            #pragma unroll
            for (int i = 0; i < D_PER_LANE; ++i) {
                partial_out[base_out + lane + i * WARP_SIZE] = 0.0f;
            }
            if (lane == 0) {
                partial_out[base_out + D]     = -INFINITY;
                partial_out[base_out + D + 1] = 0.0f;
            }
        }
        return;
    }

    const int h_kv = n_rep > 1 ? (h_idx / n_rep) : h_idx;
    const int64_t q_offset = ((int64_t)(b_idx * n_head) + h_idx) * D;
    const float * q_ptr = q + q_offset;
    const float * k_base = k + ((int64_t)(b_idx * n_kv_head) + h_kv) * k_head_stride;
    const float * v_base = v + ((int64_t)(b_idx * n_kv_head) + h_kv) * v_head_stride;
    const float * mask_row_ptr = mask ? (mask + (int64_t)b_idx * (int64_t)mask_b_stride) : nullptr;

    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
    if (warp == 0) {
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            q_reg[i] = q_ptr[lane + i * WARP_SIZE];
            o_reg[i] = 0.0f;
        }
    }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    __shared__ float k_lds[K_TILE_ELEMS];

    for (int j_tile = j_start; j_tile < j_end; j_tile += BC) {
        // K tile load — all NW warps cooperate.
        for (int idx = tid; idx < K_TILE_ELEMS; idx += THREADS_PER_BLOCK) {
            const int j_local = idx / D;
            const int d       = idx % D;
            const int j       = j_tile + j_local;
            k_lds[idx] = (j < j_end) ? k_base[j * k_stride_j + d * k_stride_d] : 0.0f;
        }
        __syncthreads();

        // Only warp 0 does the compute (L_q=1, 1 row of Q).
        if (warp == 0) {
            const int j_tile_end = j_tile + BC < j_end ? j_tile + BC : j_end;
            for (int j = j_tile; j < j_tile_end; ++j) {
                const int j_local = j - j_tile;

                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    partial += q_reg[i] * k_lds[j_local * D + lane + i * WARP_SIZE];
                }
                float s_j = flash_warp_reduce_sum_f32(partial) * scale;
                if (mask_row_ptr) s_j += mask_row_ptr[j];

                const float m_new = fmaxf(m_i, s_j);
                const float alpha = gfx906_fast_exp(m_i - m_new);
                const float p     = gfx906_fast_exp(s_j - m_new);
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    const int d_eff = lane + i * WARP_SIZE;
                    const float v_el = v_base[j * v_stride_j + d_eff * v_stride_d];
                    o_reg[i] = alpha * o_reg[i] + p * v_el;
                }
                l_i = alpha * l_i + p;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            partial_out[base_out + lane + i * WARP_SIZE] = o_reg[i];
        }
        if (lane == 0) {
            partial_out[base_out + D]     = m_i;
            partial_out[base_out + D + 1] = l_i;
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) flash_attn_decode_split_k_lds_phase1_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {
    // BC=64, NW=2 warps per block. K LDS = 64*64*4 = 16 KiB/block.
    // 4 blocks/CU via LDS, 2 warps × 4 blocks = 8 waves/CU = 2 waves/SIMD.
    flash_attn_decode_split_k_lds_phase1_impl<64, 64, 2>(q, k, v, mask, partial_out,
        B, n_head, n_kv_head, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d, num_chunks);
}

extern "C" __global__ void __launch_bounds__(128, 4) flash_attn_decode_split_k_lds_phase1_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ partial_out,
    int B, int n_head, int n_kv_head, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_l_q_stride,
    int k_head_stride, int k_stride_j, int k_stride_d,
    int v_head_stride, int v_stride_j, int v_stride_d,
    int num_chunks) {
    flash_attn_decode_split_k_lds_phase1_impl<128, 32, 2>(q, k, v, mask, partial_out,
        B, n_head, n_kv_head, L_k, scale, n_rep, mask_b_stride, mask_l_q_stride,
        k_head_stride, k_stride_j, k_stride_d,
        v_head_stride, v_stride_j, v_stride_d, num_chunks);
}
