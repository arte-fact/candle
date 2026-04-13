// Flash-attention v2 — BR=4 LDS-tiled prefill kernel for gfx906.
//
// **Why v2.** The v1 kernel in `flash_attn.cu` uses BR=1 (one Q row per
// Wave64) and directly streams K/V through global memory. On TinyLlama
// prefill shapes it was ~17× slower per call than rocBLAS because every
// K/V row is re-read once per Q row, and occupancy is capped by the
// serial j-loop. v2 fixes both:
//
//   - **BR=4.** One thread block owns a contiguous 4-row slab of Q and
//     issues 4 Wave64 warps (`blockDim.y = BR = 4`, `blockDim.x =
//     WARP_SIZE = 64`, total 256 threads per block). Each warp works on
//     one Q row independently, but all four share the LDS K/V tile ->
//     4× amortisation of K/V loads from global.
//   - **LDS K/V tile.** `BC` rows of K and V are staged in LDS at a
//     time. BC is picked per-D so the tile fits in ≤32 KiB LDS (two
//     tiles / block × BR blocks / CU remains within the gfx906 LDS
//     budget).
//   - **D in {64, 128, 256}.** D=256 is the gemma4 case. Each lane
//     owns `D_PER_LANE = D / WARP_SIZE` elements of the D axis via a
//     manual unrolled loop.
//   - **Additive mask via stride broadcast.** No in-kernel sliding-
//     window logic — gemma4 builds a full additive mask with -inf for
//     positions outside the window, exactly what this kernel needs.
//     Same (mask_b_stride, mask_lq_stride) convention as v1.
//
// **Scope.**
//   - Prefill path only. Decode (L_q = 1) falls back to rocBLAS: BR=4
//     can't tile a single Q row, and the grid would collapse to n_head
//     blocks — under-utilising gfx906's 60 CUs.
//   - f32 only.
//   - llama-convention GQA: `h_kv = h_q / n_rep`.
//
// Algorithm — the classic flash-attention-1 online softmax per Q row.
// Online state (m_i, l_i, o_reg[D_PER_LANE]) lives in registers across
// iterations of the outer K-chunk loop.

#include "compatibility.cuh"
#include "gfx906_primitives.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Use DPP-fused warp reduction from gfx906_primitives.cuh
static __device__ __forceinline__ float v2_warp_reduce_sum(float x) {
    return gfx906_warp_reduce_sum(x);
}

// -------------------------------------------------------------------
// Generic BR=4 LDS-tiled flash-attention forward.
// -------------------------------------------------------------------
//
// Template params:
//   D        = head dim (64, 128, 256)
//   BR       = rows of Q per block (fixed at 4)
//   BC       = K/V rows per LDS tile (picked per-D; see wrappers below)
//
// K_TRANS: if true, K is (B, H_kv, D, L_k) instead of (B, H_kv, L_k, D).
// v_stride_j / v_stride_d: strides for V access. If both are 0, use standard layout.
//   Standard V (L_k, D): v[j*D + d]       → v_stride_j=D, v_stride_d=1
//   Transposed V (D, T): v[d*max_T + j]   → v_stride_j=1, v_stride_d=max_T
// L_k is the allocated/stride size along the K sequence dim (used for
// K's stride when K_TRANS=true, for the kv_offset base, and as the upper
// bound on mask row indexing). L_k_iter is the number of REAL K/V entries
// to attend to — the kernel iterates chunks [0 .. L_k_iter) and writes
// -INFINITY for rows >= L_k_iter. When L_k_iter == L_k (the default),
// behavior is identical to the pre-change kernel. When L_k_iter < L_k
// (G2 dynamic l_k: the K buffer is padded to L_k for stable replay while
// only the first L_k_iter positions carry real token state), the kernel
// skips the padded tail entirely — cutting flash-attn work from O(L_k_pad)
// to O(L_k_real) each decode step.
template <int D, int BR, int BC, bool K_TRANS = false>
static __device__ __forceinline__ void flash_attn_fwd_v2_impl(
    const float * __restrict__ q,     // (B, H_q, L_q, D) contiguous
    const float * __restrict__ k,     // (B, H_kv, L_k, D) or (B, H_kv, D, L_k) if K_TRANS
    const float * __restrict__ v,     // (B, H_kv, ...) with strides
    const float * __restrict__ mask,  // additive f32 or nullptr
    float * __restrict__ out,         // (B, H_q, L_q, D)
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int mask_lq_stride,
    int v_head_stride = 0, int v_stride_j = 0, int v_stride_d = 0,
    int L_k_iter = -1)
{
    static_assert(D == 64 || D == 128 || D == 256 || D == 512,
                  "only D=64, D=128, D=256, D=512 supported");
    static_assert(D % WARP_SIZE == 0,
                  "D must be a multiple of WARP_SIZE");
    static_assert(BR >= 1 && BR <= 8, "BR must be 1..8");
    static_assert(BC >= 8 && BC <= 64, "BC must be 8..64");

    constexpr int D_PER_LANE = D / WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = BR * WARP_SIZE;

    // Block / thread identity.
    const int q_tile = blockIdx.x;   // covers BR consecutive Q rows
    const int h_idx  = blockIdx.y;
    const int b_idx  = blockIdx.z;
    const int lane   = threadIdx.x;  // 0..WARP_SIZE-1 (D axis)
    const int warp   = threadIdx.y;  // 0..BR-1        (Q-row axis)
    const int tid    = warp * WARP_SIZE + lane;  // flat block-local tid

    const int q_idx = q_tile * BR + warp;

    // llama-style GQA: q_head / n_rep = kv_head.
    const int h_kv = (n_rep > 1) ? (h_idx / n_rep) : h_idx;

    // Per-slice base pointers.
    const int64_t q_offset =
        ((int64_t)(b_idx * H_q) + h_idx) * (int64_t)L_q * D + (int64_t)q_idx * D;
    // K and V head offsets. K uses the standard or K_TRANS layout.
    // V uses explicit strides if v_head_stride > 0, else standard L_k*D.
    const int64_t kv_offset =
        ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)L_k * D;
    // V head offset: use explicit stride if provided, else same as K.
    const int64_t v_offset = (v_head_stride > 0)
        ? ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)v_head_stride
        : kv_offset;
    // V element strides: default to standard (j*D + d).
    const int vs_j = (v_stride_j > 0) ? v_stride_j : D;
    const int vs_d = (v_stride_d > 0) ? v_stride_d : 1;
    const int64_t out_offset = q_offset;

    const float * q_ptr = q + q_offset;
    const float * k_ptr = k + kv_offset;
    const float * v_ptr = v + v_offset;
    float *       o_ptr = out + out_offset;

    // Additive mask row pointer. Row index is driven by (b, q_idx)
    // via the caller-provided strides:
    //   (1, 1,  1,  L_k) -> mask_b_stride = 0,    mask_lq_stride = 0
    //   (1, 1, L_q, L_k) -> mask_b_stride = 0,    mask_lq_stride = L_k
    //   (B, 1,  1,  L_k) -> mask_b_stride = L_k,  mask_lq_stride = 0
    //   (B, 1, L_q, L_k) -> mask_b_stride = L_q*L_k, mask_lq_stride = L_k
    const float * mask_row = nullptr;
    if (mask != nullptr) {
        mask_row = mask
                 + (int64_t)b_idx * (int64_t)mask_b_stride
                 + (int64_t)q_idx * (int64_t)mask_lq_stride;
    }

    // ---- Per-lane register state (Q / O / softmax denom). ----
    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
    const bool q_in_range = (q_idx < L_q);
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; ++i) {
        q_reg[i] = q_in_range ? q_ptr[lane + i * WARP_SIZE] : 0.0f;
        o_reg[i] = 0.0f;
    }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ---- Shared LDS K/V tile. All BR warps read the same tile. ----
    __shared__ float k_lds[BC * D];
    __shared__ float v_lds[BC * D];

    constexpr int TILE_ELEMS = BC * D;
    // Each thread loads TILE_ELEMS / THREADS_PER_BLOCK elements per tile.
    // For D=64,  BC=64, BR=4: 4096 / 256 = 16 loads per thread.
    // For D=128, BC=32, BR=4: 4096 / 256 = 16 loads per thread.
    // For D=256, BC=16, BR=4: 4096 / 256 = 16 loads per thread.
    constexpr int LOADS_PER_THREAD =
        (TILE_ELEMS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const int L_k_effective = (L_k_iter >= 0) ? L_k_iter : L_k;
    const int n_chunks = (L_k_effective + BC - 1) / BC;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int k_start = chunk * BC;

        // ---- Cooperative load of K/V tile into LDS. ----
        // Threads stride through [0, BC*D) in steps of THREADS_PER_BLOCK.
        #pragma unroll
        for (int il = 0; il < LOADS_PER_THREAD; ++il) {
            const int idx = tid + il * THREADS_PER_BLOCK;
            if (idx < TILE_ELEMS) {
                const int j = idx / D;
                const int d = idx % D;
                const int row = k_start + j;
                const bool valid = (row < L_k_effective);
                // K layout: standard (L_k, D) → k[row*D + d]
                //           transposed (D, L_k) → k[d*L_k + row]
                if (K_TRANS) {
                    k_lds[idx] = valid ? k_ptr[d * L_k + row] : 0.0f;
                } else {
                    k_lds[idx] = valid ? k_ptr[row * D + d] : 0.0f;
                }
                v_lds[idx] = valid ? v_ptr[row * vs_j + d * vs_d] : 0.0f;
            }
        }
        __syncthreads();

        // ---- Inner j loop: online softmax + output update. ----
        // Each warp works on its own Q row independently. Warps don't
        // interfere — each has private `m_i`, `l_i`, `o_reg`.
        if (q_in_range) {
            #pragma unroll 4
            for (int j = 0; j < BC; ++j) {
                const int row = k_start + j;
                // Dot product q · K_lds[j]. Partial-sum across the D
                // axis via per-lane contribution, then warp-reduce
                // within the single Wave64 warp.
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    partial += q_reg[i] * k_lds[j * D + lane + i * WARP_SIZE];
                }
                float s_j = v2_warp_reduce_sum(partial) * scale;

                // Additive mask.
                if (mask_row != nullptr && row < L_k_effective) {
                    s_j += mask_row[row];
                }
                // Out-of-bounds K rows contribute nothing (K/V tile
                // has zero padding for row >= L_k_effective, but mask
                // isn't indexable beyond the true attended range).
                if (row >= L_k_effective) {
                    s_j = -INFINITY;
                }

                // Online softmax update.
                const float m_new = fmaxf(m_i, s_j);
                const float alpha = gfx906_fast_exp(m_i - m_new);
                const float p     = gfx906_fast_exp(s_j - m_new);

                #pragma unroll
                for (int i = 0; i < D_PER_LANE; ++i) {
                    o_reg[i] = alpha * o_reg[i]
                             + p * v_lds[j * D + lane + i * WARP_SIZE];
                }
                l_i = alpha * l_i + p;
                m_i = m_new;
            }
        }
        __syncthreads();  // ensure all warps finished reading LDS before next load
    }

    // ---- Final normalisation and write back. ----
    if (q_in_range) {
        const float inv_l = gfx906_rcp(l_i);
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            o_ptr[lane + i * WARP_SIZE] = o_reg[i] * inv_l;
        }
    }
}

// -------------------------------------------------------------------
// Extern "C" wrappers. Choose BC per-D so LDS fits under ~32 KiB/block
// (allows 2 blocks per CU by LDS; register/occupancy is the other limit).
// -------------------------------------------------------------------

// D=64  : LDS = 2 * BC * 64 * 4 = 512*BC bytes. BC=64 -> 32 KiB.
extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<64, 4, 64>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

// D=128 : LDS = 2 * BC * 128 * 4 = 1024*BC bytes. BC=32 -> 32 KiB.
extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<128, 4, 32>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

// D=256 : LDS = 2 * BC * 256 * 4 = 2048*BC bytes. BC=16 -> 32 KiB.
extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_d256_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<256, 4, 16, false>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

// ---- K-transposed variants: K is (B, H_kv, D, L_k) ----
// Used when the KV cache stores K pre-transposed (Attack C).

extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_kt_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,  // (B, H_kv, D, L_k) transposed
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<64, 4, 64, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_kt_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<128, 4, 32, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_kt_d256_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride)
{
    flash_attn_fwd_v2_impl<256, 4, 16, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride);
}

// ---- K+V transposed with strided V: for KvCache decode ----
// Both K and V stored as (B, H_kv, D, max_T) with seq on last dim.
// K uses K_TRANS template. V uses runtime strides.

extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_ktvs_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d)
{
    flash_attn_fwd_v2_impl<64, 4, 64, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d);
}

extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_ktvs_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d)
{
    flash_attn_fwd_v2_impl<128, 4, 32, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d);
}

// D=256 strided variant — needed for Gemma4-E4B SWA layers (key_length_swa=256).
// BC=16 keeps LDS under 32 KiB (2 * 16 * 256 * 4 = 32 KiB per block).
extern "C" __global__ void __launch_bounds__(256, 2) flash_attn_v2_fwd_ktvs_d256_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d)
{
    flash_attn_fwd_v2_impl<256, 4, 16, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d);
}

// D=512 strided variant — needed for Gemma4-E4B GLOBAL layers
// (`key_length` = 512 on E4B). BC=8 keeps LDS at 32 KiB
// (2 * 8 * 512 * 4 = 32 KiB per block — same budget as d=256/BC=16).
// Halves the j-axis tile so each block does twice as many BC iterations
// for a fixed L_k, in exchange for staying within the gfx906 LDS limit.
extern "C" __global__ void __launch_bounds__(256, 1) flash_attn_v2_fwd_ktvs_d512_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d)
{
    flash_attn_fwd_v2_impl<512, 4, 8, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d);
}

// ---- Dynamic-L_k_iter variants for G2 replay. ----
// Same strided-V ktvs kernels as above, but accept an extra L_k_iter
// parameter naming how many real K positions to process. The K buffer
// is still sized to L_k (so its stride along D stays stable across
// replays), but iteration stops at L_k_iter. Under G2 replay,
// L_k_iter is captured as a per-token Counter arg (+1 per decode
// step), so the replayed plan processes O(real_l_k) entries instead
// of O(pad).
extern "C" __global__ void __launch_bounds__(256, 2)
flash_attn_v2_fwd_ktvs_dyn_d64_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter)
{
    flash_attn_fwd_v2_impl<64, 4, 64, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(256, 2)
flash_attn_v2_fwd_ktvs_dyn_d128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter)
{
    flash_attn_fwd_v2_impl<128, 4, 32, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(256, 2)
flash_attn_v2_fwd_ktvs_dyn_d256_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter)
{
    flash_attn_fwd_v2_impl<256, 4, 16, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(256, 1)
flash_attn_v2_fwd_ktvs_dyn_d512_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_q, int L_k,
    float scale, int n_rep, int mask_b_stride, int mask_lq_stride,
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter)
{
    flash_attn_fwd_v2_impl<512, 4, 8, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter);
}
