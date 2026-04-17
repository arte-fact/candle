// Signed Walsh-Hadamard Transform for TurboQuant KV preconditioning.
//
// Ported from llama.cpp-turbo `turbo-quant.cu::kernel_turbo_wht`.
// Forward transform: y = s_second ⊙ FWHT(s_first ⊙ x) / sqrt(128)
// Backward (inverse) transform: x = s_first ⊙ FWHT(s_second ⊙ y) / sqrt(128)
//   (FWHT is self-inverse when normalised; sign flips are self-inverse.)
//
// Each 128-element group is one block (64 threads, one wavefront).
// Expects input shape with the last dim a multiple of 128 (head_dim=128
// or 256; larger head_dims are processed as multiple 128-element groups).

#include <hip/hip_runtime.h>
#include <stdint.h>

// K-rotation signs (seed=42 in turbo)
__constant__ float WHT_S1_K[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};
__constant__ float WHT_S2_K[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// V-rotation signs (seed=12345 in turbo — independent rotation for V)
__constant__ float WHT_S1_V[128] = {
     1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1,
    -1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,
     1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,
     1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,
    -1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
     1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,
    -1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1
};
__constant__ float WHT_S2_V[128] = {
    -1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,
     1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,
    -1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
    -1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,
    -1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,
     1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,
     1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1, 1
};

// Core kernel: applies (s_second ⊙ FWHT(s_first ⊙ x)) / sqrt(128).
// Grid: (n_groups, 1, 1). Block: 64.
// `n_groups = n_elements / 128` — each group is one 128-elem vector.
template <const float * S_FIRST, const float * S_SECOND>
static __device__ __forceinline__ void turbo_wht_impl(
    const float * __restrict__ src,
    float       * __restrict__ dst,
    const int64_t n_groups
) {
    __shared__ float x[128];
    const int64_t group_idx = blockIdx.x;
    if (group_idx >= n_groups) return;

    const int tid = threadIdx.x;                    // 0..63
    const float * in  = src + group_idx * 128;
    float       * out = dst + group_idx * 128;

    x[tid]      = in[tid]      * S_FIRST[tid];
    x[tid + 64] = in[tid + 64] * S_FIRST[tid + 64];
    __syncthreads();

    // Cooperative FWHT, 7 butterfly stages.
    for (int h = 1; h < 128; h *= 2) {
        const int pair_idx = tid;
        const int block_size = h * 2;
        const int block_idx = pair_idx / h;
        const int within = pair_idx % h;
        const int j = block_idx * block_size + within;
        if (j + h < 128) {
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }

    const float inv = 0.08838834764831845f;         // 1 / sqrt(128)
    out[tid]      = x[tid]      * inv * S_SECOND[tid];
    out[tid + 64] = x[tid + 64] * inv * S_SECOND[tid + 64];
}

// K-forward: s1_k → FWHT → s2_k → *1/sqrt(128)
extern "C" __global__ void __launch_bounds__(64, 1) turbo_wht_k_forward(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) {
    turbo_wht_impl<WHT_S1_K, WHT_S2_K>(src, dst, n_groups);
}

// K-inverse: s2_k → FWHT → s1_k → *1/sqrt(128)  (swap of the two sign arrays)
extern "C" __global__ void __launch_bounds__(64, 1) turbo_wht_k_inverse(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) {
    turbo_wht_impl<WHT_S2_K, WHT_S1_K>(src, dst, n_groups);
}

// V-forward: s1_v → FWHT → s2_v → *1/sqrt(128)
extern "C" __global__ void __launch_bounds__(64, 1) turbo_wht_v_forward(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) {
    turbo_wht_impl<WHT_S1_V, WHT_S2_V>(src, dst, n_groups);
}

// V-inverse: s2_v → FWHT → s1_v → *1/sqrt(128)
extern "C" __global__ void __launch_bounds__(64, 1) turbo_wht_v_inverse(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) {
    turbo_wht_impl<WHT_S2_V, WHT_S1_V>(src, dst, n_groups);
}

// ============================================================================
// WHT-256 variant for head_dim=256 archs (qwen3-coder-next, gemma4-E4B).
// ============================================================================
//
// Sign tables are the 128-element ones repeated with a sign flip on the
// second half — a cheap way to get an orthogonal 256-elem preconditioning
// without hand-tuning new sign arrays. The 8-stage butterfly (1 more than
// WHT-128) handles the 2× size. One block per 256-elem group, 128 threads.

template <const float * S1_128, const float * S2_128>
static __device__ __forceinline__ void turbo_wht_256_impl(
    const float * __restrict__ src,
    float       * __restrict__ dst,
    const int64_t n_groups
) {
    __shared__ float x[256];
    const int64_t group_idx = blockIdx.x;
    if (group_idx >= n_groups) return;

    const int tid = threadIdx.x;                 // 0..127
    const float * in  = src + group_idx * 256;
    float       * out = dst + group_idx * 256;

    // Build effective 256-elem sign arrays from 128-elem bases: first
    // half uses the base; second half uses the base with a lane-wise
    // XOR sign (flips signs on even-indexed lanes). The resulting 256-
    // element pattern is guaranteed distinct from either base.
    const float s1_lo = S1_128[tid];
    const float s1_hi = (tid & 1) ? S1_128[tid] : -S1_128[tid];
    const float s2_lo = S2_128[tid];
    const float s2_hi = (tid & 1) ? -S2_128[tid] : S2_128[tid];

    x[tid]       = in[tid]       * s1_lo;
    x[tid + 128] = in[tid + 128] * s1_hi;
    __syncthreads();

    // 8 butterfly stages (h = 1,2,4,...,128).
    for (int h = 1; h < 256; h *= 2) {
        const int pair_idx = tid;                // 0..127 butterflies/stage
        const int block_size = h * 2;
        const int block_idx = pair_idx / h;
        const int within = pair_idx % h;
        const int j = block_idx * block_size + within;
        if (j + h < 256) {
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalise by 1 / sqrt(256) = 0.0625
    const float inv = 0.0625f;
    out[tid]       = x[tid]       * inv * s2_lo;
    out[tid + 128] = x[tid + 128] * inv * s2_hi;
}

extern "C" __global__ void __launch_bounds__(128, 1) turbo_wht_k_forward_256(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) { turbo_wht_256_impl<WHT_S1_K, WHT_S2_K>(src, dst, n_groups); }

extern "C" __global__ void __launch_bounds__(128, 1) turbo_wht_k_inverse_256(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) { turbo_wht_256_impl<WHT_S2_K, WHT_S1_K>(src, dst, n_groups); }

extern "C" __global__ void __launch_bounds__(128, 1) turbo_wht_v_forward_256(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) { turbo_wht_256_impl<WHT_S1_V, WHT_S2_V>(src, dst, n_groups); }

extern "C" __global__ void __launch_bounds__(128, 1) turbo_wht_v_inverse_256(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_groups
) { turbo_wht_256_impl<WHT_S2_V, WHT_S1_V>(src, dst, n_groups); }
