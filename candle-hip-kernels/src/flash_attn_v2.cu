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
// K strides: when `k_head_stride > 0`, K is read via explicit per-head
// and per-element strides (same pattern as V). This lets the caller pass
// a narrow'd (non-contiguous) K view — e.g. gemma4's decode path where K
// is stored in the KvCache as (B, H_kv, D, max_T) and the per-call
// narrow to `L_k` keeps stride-along-D equal to `max_T`, not `L_k`.
// Skipping the .contiguous() call saves a ~100 KiB/layer copy per token.
//
// When all three are 0 (the default), the kernel falls back to the
// contiguous K_TRANS layout K[d * L_k + row] (stride-along-D = L_k,
// stride-along-seq = 1).
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
    int L_k_iter = -1,
    int k_head_stride = 0, int k_stride_d = 0, int k_stride_j = 0)
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
    // K head offset: use explicit stride when provided (non-contiguous K view),
    // else fall back to contiguous L_k*D (K_TRANS) / L_k*D (standard).
    const int64_t kv_offset =
        ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)L_k * D;
    const int64_t k_offset = (k_head_stride > 0)
        ? ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)k_head_stride
        : kv_offset;
    // V head offset: use explicit stride if provided, else same as K (contiguous).
    const int64_t v_offset = (v_head_stride > 0)
        ? ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)v_head_stride
        : kv_offset;
    // K element strides: default to the contiguous K_TRANS layout
    //   K[d * L_k + row] (stride-along-D = L_k, stride-along-seq = 1).
    // V element strides: default to standard (j*D + d).
    const int ks_d = (k_stride_d > 0) ? k_stride_d : (K_TRANS ? L_k : D);
    const int ks_j = (k_stride_j > 0) ? k_stride_j : (K_TRANS ? 1 : 1);
    const int vs_j = (v_stride_j > 0) ? v_stride_j : D;
    const int vs_d = (v_stride_d > 0) ? v_stride_d : 1;
    const int64_t out_offset = q_offset;

    const float * q_ptr = q + q_offset;
    const float * k_ptr = k + k_offset;
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
                // K layout:
                //   standard (L_k, D) contiguous → k[row*D + d]
                //   transposed (D, L_k) default  → k[d*L_k + row]
                //   transposed + explicit strides → k[d*ks_d + row*ks_j]
                if (K_TRANS) {
                    k_lds[idx] = valid ? k_ptr[d * ks_d + row * ks_j] : 0.0f;
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
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter,
    int k_head_stride, int k_stride_d, int k_stride_j)
{
    flash_attn_fwd_v2_impl<64, 4, 64, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter,
        k_head_stride, k_stride_d, k_stride_j);
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
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter,
    int k_head_stride, int k_stride_d, int k_stride_j)
{
    flash_attn_fwd_v2_impl<128, 4, 32, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter,
        k_head_stride, k_stride_d, k_stride_j);
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
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter,
    int k_head_stride, int k_stride_d, int k_stride_j)
{
    flash_attn_fwd_v2_impl<256, 4, 16, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter,
        k_head_stride, k_stride_d, k_stride_j);
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
    int v_head_stride, int v_stride_j, int v_stride_d, int L_k_iter,
    int k_head_stride, int k_stride_d, int k_stride_j)
{
    flash_attn_fwd_v2_impl<512, 4, 8, true>(
        q, k, v, mask, out, B, H_q, H_kv, L_q, L_k,
        scale, n_rep, mask_b_stride, mask_lq_stride,
        v_head_stride, v_stride_j, v_stride_d, L_k_iter,
        k_head_stride, k_stride_d, k_stride_j);
}

// ============================================================================
// Phase P — mat-vec decode attention
// ----------------------------------------------------------------------------
// For L_q=1 decode with K, V both stored row-major (B, H_kv, T_pad, D) so D
// is the CONTIGUOUS axis. This lets the warp's 64 lanes read K[t, 0..64]
// in a single coalesced VMEM transaction, matching llama.cpp-turbo's
// `mul_mat_vec_f<float,float,...>` pattern.
//
// Grid: (1, H_q, B)     — one block per (batch, q_head); L_q=1 so q_idx=0.
// Block: (WARP_SIZE, 1, 1) — 64 threads, one warp.
//
// Each thread owns D / WARP_SIZE elements along D. D_PER_LANE=4 for D=256,
// D_PER_LANE=8 for D=512. Q and O live entirely in registers (D/WARP_SIZE
// floats each). No LDS for K/V tiles — direct global reads per t position,
// coalesced across the warp.
//
// Online softmax (flash-attn v2 style): for each t, compute dot Q·K[t],
// update running max m_i, output accumulator o_reg, and normaliser l.
// Final: o_reg /= l.
//
// L_k_iter < L_k is supported (defaults to L_k): for G2 dynamic l_k.
template <int D>
static __device__ __forceinline__ void gqa_decode_mv_impl(
    const float * __restrict__ q,       // (B, H_q, 1, D) contiguous
    const float * __restrict__ k,       // (B, H_kv, T_pad, D) row-major
    const float * __restrict__ v,       // (B, H_kv, T_pad, D) row-major
    const float * __restrict__ mask,    // optional (1,1,1,T_pad) or nullptr
    float * __restrict__ out,           // (B, H_q, 1, D)
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int L_k_iter)
{
    static_assert(D == 64 || D == 128 || D == 256 || D == 512,
                  "D must be 64, 128, 256, or 512");
    static_assert(D % WARP_SIZE == 0, "D must be multiple of WARP_SIZE");
    constexpr int D_PER_LANE = D / WARP_SIZE;

    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int lane  = threadIdx.x;

    // llama-style GQA: q head h maps to kv head h / n_rep.
    const int h_kv = (n_rep > 1) ? (h_idx / n_rep) : h_idx;

    // Per-head pointers. L_q=1 so no q_idx offset in Q.
    const int64_t q_off   = ((int64_t)(b_idx * H_q) + h_idx) * D;
    const int64_t kv_off  = ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)L_k * D;
    const int64_t out_off = q_off;

    const float * q_ptr = q + q_off;
    const float * k_ptr = k + kv_off;
    const float * v_ptr = v + kv_off;
    float       * o_ptr = out + out_off;

    // Mask row pointer. For (1,1,1,T_pad): mask_b_stride=0 → shared.
    // For (B,1,1,T_pad): mask_b_stride=T_pad.
    const float * mask_row = nullptr;
    if (mask != nullptr) {
        mask_row = mask + (int64_t)b_idx * (int64_t)mask_b_stride;
    }

    // Phase Q2 — vectorised float2 loads (8 bytes per VMEM txn) to match
    // llama.cpp-turbo's `mul_mat_vec_f` pattern in `ggml-cuda/mmvf.cu:132`.
    //
    // Per-lane element ownership under float2:
    //   lane L, pair i=0..PAIRS_PER_LANE-1 owns the two f32 elements at
    //     d = 2*(L + i*WARP_SIZE) + {0, 1}
    //   i.e. for D=256, D_PER_LANE=4, PAIRS=2:
    //     lane 0: d ∈ {0, 1, 128, 129}
    //     lane 1: d ∈ {2, 3, 130, 131}
    //     ...
    //   This differs from the scalar layout (lane owns d=lane, lane+64, ...)
    //   but the dot product is commutative in d — correctness is preserved
    //   as long as Q and K/V use the SAME partitioning.
    //
    // For D=64 (D_PER_LANE=1) we keep scalar reads: no float2 benefit.
    constexpr int PAIRS_PER_LANE = D_PER_LANE / 2;
    constexpr bool USE_FLOAT2 = (D_PER_LANE >= 2);

    float q_reg[D_PER_LANE];
    float o_reg[D_PER_LANE];
    if constexpr (USE_FLOAT2) {
        const float2 * q_ptr2 = reinterpret_cast<const float2 *>(q_ptr);
        #pragma unroll
        for (int i = 0; i < PAIRS_PER_LANE; ++i) {
            const float2 qq = q_ptr2[lane + i * WARP_SIZE];
            q_reg[2*i]     = qq.x;
            q_reg[2*i + 1] = qq.y;
            o_reg[2*i]     = 0.0f;
            o_reg[2*i + 1] = 0.0f;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            q_reg[i] = q_ptr[lane + i * WARP_SIZE];
            o_reg[i] = 0.0f;
        }
    }

    // Online softmax state.
    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int L_k_effective = (L_k_iter >= 0) ? L_k_iter : L_k;

    // Main loop over T positions. Each iteration:
    //   1. float2-coalesced read of K[t, d_lane..d_lane+D_PER_LANE*WARP_SIZE)
    //   2. Per-lane partial dot, warp-reduce → scalar s_t
    //   3. Apply scale + mask
    //   4. Online softmax update
    //   5. float2-coalesced read of V[t, d_lane], MAC into o_reg
    for (int t = 0; t < L_k_effective; ++t) {
        const float * k_row = k_ptr + (int64_t)t * D;
        const float * v_row = v_ptr + (int64_t)t * D;

        // Issue both K and V loads together so V VMEM latency overlaps
        // the warp-reduce + softmax-exp critical path (same pattern as
        // gqa_decode_mv_fast_impl; see its comment for details).
        float k_reg[D_PER_LANE];
        float v_reg[D_PER_LANE];
        if constexpr (USE_FLOAT2) {
            const float2 * k_row2 = reinterpret_cast<const float2 *>(k_row);
            const float2 * v_row2 = reinterpret_cast<const float2 *>(v_row);
            #pragma unroll
            for (int i = 0; i < PAIRS_PER_LANE; ++i) {
                const float2 kk = k_row2[lane + i * WARP_SIZE];
                const float2 vv = v_row2[lane + i * WARP_SIZE];
                k_reg[2*i]     = kk.x;
                k_reg[2*i + 1] = kk.y;
                v_reg[2*i]     = vv.x;
                v_reg[2*i + 1] = vv.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < D_PER_LANE; ++i) {
                k_reg[i] = k_row[lane + i * WARP_SIZE];
                v_reg[i] = v_row[lane + i * WARP_SIZE];
            }
        }

        // Per-lane partial dot product using pre-loaded K.
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            partial += q_reg[i] * k_reg[i];
        }
        float s_t = v2_warp_reduce_sum(partial) * scale;

        if (mask_row != nullptr) {
            s_t += mask_row[t];
        }

        // Online softmax.
        const float m_new = fmaxf(m_i, s_t);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_t - m_new);

        // O update using pre-loaded V.
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            o_reg[i] = alpha * o_reg[i] + p * v_reg[i];
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    // Normalise and write back. Match the float2 ownership layout used
    // for Q/K/V throughout the kernel.
    const float inv_l = gfx906_rcp(l_i);
    if constexpr (USE_FLOAT2) {
        float2 * o_ptr2 = reinterpret_cast<float2 *>(o_ptr);
        #pragma unroll
        for (int i = 0; i < PAIRS_PER_LANE; ++i) {
            float2 oo;
            oo.x = o_reg[2*i]     * inv_l;
            oo.y = o_reg[2*i + 1] * inv_l;
            o_ptr2[lane + i * WARP_SIZE] = oo;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            o_ptr[lane + i * WARP_SIZE] = o_reg[i] * inv_l;
        }
    }
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_d64_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_impl<64>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                           scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_d128_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_impl<128>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                            scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_d256_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_impl<256>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                            scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_d512_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_impl<512>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                            scale, n_rep, mask_b_stride, L_k_iter);
}

// ============================================================================
// Phase Q2 — fast mat-vec decode attention
// ----------------------------------------------------------------------------
// Same semantics as gqa_decode_mv_impl above, but:
//   * Single-warp block (BLOCK_SIZE = WARP_SIZE = 64) — no cross-warp LDS
//     sync, no __syncthreads in the T loop.
//   * float2 vectorised loads: each thread reads 8 bytes per K/V access,
//     halving the VMEM instruction count vs the scalar kernel.
//   * ggml-mmvf-style thread→element mapping: thread `lane` owns the
//     float2 pairs at positions `lane, lane+WARP_SIZE, ...` in the
//     float2-indexed space. Adjacent lanes read adjacent 8-byte
//     chunks → perfect coalescing on gfx906's `global_load_dwordx2` path.
//
// First iteration of Phase Q2 tried a 128-thread block with cross-warp
// LDS reduce; that *regressed* to 46 t/s (vs 54 for the scalar kernel)
// because two `__syncthreads` per T iteration dominated the runtime
// at our low occupancy (E4B has only n_head=2-8 blocks per layer,
// already CU-starved on MI50's 60 CUs — making sync latency matter more
// than VMEM transaction count). The single-warp float2 version keeps
// the warp-local reduction unchanged and just swaps scalar for vector
// loads.
//
// Only defined for D=256 and D=512 — smaller heads use the original
// scalar kernel (benchmarks showed negligible difference at those sizes).
template <int D, int BLOCK_SIZE>
static __device__ __forceinline__ void gqa_decode_mv_fast_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep,
    int mask_b_stride, int L_k_iter)
{
    static_assert(D % 2 == 0, "D must be even for float2 loads");
    static_assert(BLOCK_SIZE == WARP_SIZE, "Q2 fast impl: single-warp block only");
    static_assert(D / 2 >= BLOCK_SIZE, "D/2 must be >= BLOCK_SIZE");
    static_assert((D / 2) % BLOCK_SIZE == 0, "D/2 must be a multiple of BLOCK_SIZE");
    constexpr int NCOLS2 = D / 2;                 // float2 positions in one row
    constexpr int F2_PER_LANE = NCOLS2 / BLOCK_SIZE;

    const int h_idx = blockIdx.y;
    const int b_idx = blockIdx.z;
    const int lane  = threadIdx.x;

    // llama-style GQA: q head → kv head.
    const int h_kv = (n_rep > 1) ? (h_idx / n_rep) : h_idx;

    const int64_t q_off  = ((int64_t)(b_idx * H_q) + h_idx) * D;
    const int64_t kv_off = ((int64_t)(b_idx * H_kv) + h_kv) * (int64_t)L_k * D;

    const float2 * q2       = (const float2 *)(q + q_off);
    const float2 * k2_base  = (const float2 *)(k + kv_off);
    const float2 * v2_base  = (const float2 *)(v + kv_off);
    float *        o_ptr    = out + q_off;

    const float * mask_row = (mask != nullptr)
        ? mask + (int64_t)b_idx * (int64_t)mask_b_stride
        : nullptr;

    // Per-thread Q registers: F2_PER_LANE float2 values.
    float2 q_reg[F2_PER_LANE];
    #pragma unroll
    for (int i = 0; i < F2_PER_LANE; ++i) {
        q_reg[i] = q2[lane + i * BLOCK_SIZE];
    }
    // Output accumulators, 2×F2_PER_LANE scalars (online-softmax running o).
    float o_reg[2 * F2_PER_LANE];
    #pragma unroll
    for (int i = 0; i < 2 * F2_PER_LANE; ++i) o_reg[i] = 0.0f;

    // Online softmax scalars — per-lane, same value after warp reduce.
    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int L_k_effective = (L_k_iter >= 0) ? L_k_iter : L_k;

    for (int t = 0; t < L_k_effective; ++t) {
        const float2 * k_row = k2_base + (int64_t)t * NCOLS2;
        const float2 * v_row = v2_base + (int64_t)t * NCOLS2;

        // Issue K and V loads together so V latency overlaps QK reduce +
        // online-softmax exponentials. V is independent of the reduce result,
        // so the compiler is free to schedule its VMEM before the reduce.
        // (Note: K double-buffering across T was tried; it regressed because
        // the extra VGPR state dropped occupancy from 4→3 waves/EU, negating
        // the latency hiding. The compiler already schedules next-T K loads
        // via forward speculation when occupancy is high enough.)
        float2 k_vals[F2_PER_LANE];
        float2 v_vals[F2_PER_LANE];
        #pragma unroll
        for (int i = 0; i < F2_PER_LANE; ++i) {
            k_vals[i] = k_row[lane + i * BLOCK_SIZE];
            v_vals[i] = v_row[lane + i * BLOCK_SIZE];
        }

        // Per-thread partial dot over this K row.
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < F2_PER_LANE; ++i) {
            partial += q_reg[i].x * k_vals[i].x + q_reg[i].y * k_vals[i].y;
        }

        // Warp-wide reduce (no __syncthreads — single-warp block).
        float s_t = v2_warp_reduce_sum(partial) * scale;
        if (mask_row != nullptr) s_t += mask_row[t];

        const float m_new = fmaxf(m_i, s_t);
        const float alpha = gfx906_fast_exp(m_i - m_new);
        const float p     = gfx906_fast_exp(s_t - m_new);

        // V·attn accumulate using the pre-loaded V values.
        #pragma unroll
        for (int i = 0; i < F2_PER_LANE; ++i) {
            o_reg[2*i]     = alpha * o_reg[2*i]     + p * v_vals[i].x;
            o_reg[2*i + 1] = alpha * o_reg[2*i + 1] + p * v_vals[i].y;
        }
        l_i = alpha * l_i + p;
        m_i = m_new;
    }

    // Normalise + write back.
    const float inv_l = gfx906_rcp(l_i);
    #pragma unroll
    for (int i = 0; i < F2_PER_LANE; ++i) {
        const int base = 2 * (lane + i * BLOCK_SIZE);
        o_ptr[base]     = o_reg[2*i]     * inv_l;
        o_ptr[base + 1] = o_reg[2*i + 1] * inv_l;
    }
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_d256_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_impl<256, 64>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                                     scale, n_rep, mask_b_stride, L_k_iter);
}

extern "C" __global__ void __launch_bounds__(64, 1)
gqa_decode_mv_fast_d512_f32(
    const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ mask,
    float * __restrict__ out,
    int B, int H_q, int H_kv, int L_k,
    float scale, int n_rep, int mask_b_stride, int L_k_iter)
{
    gqa_decode_mv_fast_impl<512, 64>(q, k, v, mask, out, B, H_q, H_kv, L_k,
                                     scale, n_rep, mask_b_stride, L_k_iter);
}
