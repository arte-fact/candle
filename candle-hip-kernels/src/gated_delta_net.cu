// Gated Delta Net fused recurrent step kernel.
//
// Replaces the tensor-op chain in candle-transformers'
// `delta_net_step_vectorized` — a single fused HIP kernel that consumes
// q, k, v, gate, beta, and an in-place state (S_v × S_v) and produces
// the attention output row for L consecutive tokens.
//
// Ported from llamacpp-turbo's
// `ggml/src/ggml-cuda/gated_delta_net.cu::gated_delta_net_cuda<S_v, false>`.
// Kept minimal: Wave64 (WARP_SIZE=64), KDA=false (scalar gate per head),
// single template instantiation at S_v=128 to cover the qwen35 / qwen3next
// family. Extending to S_v∈{16,32,64} later is mechanical.
//
// Layout (row-major, contiguous):
//   q, k, v:    (B, H, L, S_v)   — elem at [b, h, t, i] = (b*H + h)*L*S_v + t*S_v + i
//   gate, beta: (B, H, L)        — one scalar per (b, h, t)
//   state_in:   (B, H, S_v, S_v) — row-major: state[i, j] at (b*H + h)*S_v*S_v + i*S_v + j
//   state_out:  (B, H, S_v, S_v) — same layout; written fresh by the kernel.
//                                  state_in and state_out may alias (same
//                                  pointer). The kernel loads state_in into
//                                  registers at the start, does the L-step
//                                  recurrence, and writes state_out at the
//                                  end, so aliasing is safe even though the
//                                  Rust side treats them as distinct buffers
//                                  for borrow-check ergonomics.
//   attn_out:   (B, H, L, S_v)   — row-major, same layout as q
//
// Convention (matches candle's existing `delta_net_step_vectorized`):
//   state[i, j] means "S matrix row i, col j" — stored row-major.
//   Per token:
//     state[i, j] *= exp(gate)
//     sk[j]       = Σ_i state[i, j] * k[i]
//     delta[j]    = (v[j] - sk[j]) * beta
//     state[i, j] += k[i] * delta[j]                            (outer update)
//     attn[j]     = Σ_i state[i, j] * q[i]                      (fused read-back)
//
// Each warp owns one output column `col` across the full token loop; the
// S_v rows of state[*, col] are sharded across the warp's 64 lanes (2 rows
// per lane at S_v=128). State stays register-resident across all L tokens
// in one kernel invocation.
//
// Grid:  (H, B, ceil(S_v / num_warps))
// Block: (WARP_SIZE, num_warps, 1) where num_warps = 4
// One warp = one output column. num_warps=4 blocks per SIMD lets 4 columns
// of the same (b, h) share instruction issue.

#include "compatibility.cuh"
#include "gfx906_primitives.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Use DPP-fused warp reduction from gfx906_primitives.cuh
static __device__ __forceinline__ float warp_reduce_sum_f32(float x) {
    return gfx906_warp_reduce_sum(x);
}

template <int S_v>
static __device__ __forceinline__ void gated_delta_net_step_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep) {

    // `H` is the number of V heads (== state heads == output heads).
    // Q and K have `H / n_rep` heads (the GQA K/V head count), and each
    // group of `n_rep` V heads shares one Q head and one K head.
    // `n_rep == 1` reduces to the original no-GQA case.
    //
    // Phase 3: eliminating the caller's GQA expand→reshape (which
    // materialised Q and K as ~7680 ucopy_f32 kernels per forward pass
    // on qwen35-9B decode) by teaching the kernel to do the head
    // broadcast implicitly via `h_idx / n_rep`.

    constexpr int warp_size     = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of WARP_SIZE");

    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int col   = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane  = threadIdx.x;

    if (col >= S_v) {
        return;
    }

    // V / state / attn_out / gate / beta are indexed by (b, h_idx).
    // Q / K are indexed by (b, h_idx % H_kv) — the GQA-shared head.
    //
    // The caller expanded Q/K via `unsqueeze(1) + expand(rep, h_kv) +
    // reshape(h_v)` which places the rep axis OUTER and the h_kv axis
    // INNER. After reshape the flat h_v index maps to source h_kv
    // as `h_kv_src = h_v % h_kv` (the "tile" pattern, matching
    // ggml_repeat_4d). This is NOT `h_idx / n_rep` — that would be
    // the interleave pattern from `repeat_interleave`.
    const int H_kv    = H / n_rep;
    const int h_kv    = h_idx % H_kv;
    const int bh      = b_idx * H    + h_idx;
    const int bh_kv   = b_idx * H_kv + h_kv;
    const float * q_bh        = q         + (int64_t)bh_kv * L * S_v;
    const float * k_bh        = k         + (int64_t)bh_kv * L * S_v;
    const float * v_bh        = v         + (int64_t)bh    * L * S_v;
    const float * gate_bh     = gate      + (int64_t)bh    * L;
    const float * beta_bh     = beta      + (int64_t)bh    * L;
    const float * state_in_bh = state_in  + (int64_t)bh    * S_v * S_v;
    float *       state_out_bh = state_out + (int64_t)bh   * S_v * S_v;
    float *       attn_bh      = attn_out  + (int64_t)bh   * L * S_v;

    // P23: state stored TRANSPOSED (col is outer, i is inner) so that a warp
    // reading column `col` hits `S_v` contiguous floats (8 cachelines at S_v=128)
    // instead of `S_v` cachelines strided by 512 B. Matches llama.cpp's layout
    // at `ggml-cuda/gated_delta_net.cu:42-54`. Since the state is zero-init in
    // Rust (`GdnState::init_*`) and only ever touched by this kernel, flipping
    // both reads and writes symmetrically preserves semantics without any
    // Rust-side change.
    const float * state_in_col  = state_in_bh  + (int64_t)col * S_v;
    float *       state_out_col = state_out_bh + (int64_t)col * S_v;

    // Load column `col` of state_in into registers:
    //   s_shard[r] = state[col, i] with i = r * warp_size + lane.
    // Warp reads `S_v` contiguous floats → coalesced.
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = state_in_col[i];
    }

    for (int t = 0; t < L; t++) {
        // Per-token scalars. All lanes read the same memory location —
        // scalar-broadcast via GCN L1.
        const float g_val    = gfx906_fast_exp(gate_bh[t]);
        const float beta_val = beta_bh[t];

        // Cache the per-token row shard of k and q in registers. Each lane
        // owns rows_per_lane elements — indexed the same way as s_shard.
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_bh[t * S_v + i];
            q_reg[r] = q_bh[t * S_v + i];
        }

        // Apply decay: state[col, i] *= g_val (exact original op order —
        // preserves numerical parity with the CPU reference).
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= g_val;
        }

        // sk[col] = Σ_i state[col, i] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = warp_reduce_sum_f32(kv_shard);

        // delta[col] = (v[col] - sk[col]) * beta
        const float v_col     = v_bh[t * S_v + col];
        const float delta_col = (v_col - kv_col) * beta_val;

        // Fused: state[col, i] += k[i] * delta_col
        //        attn[col]     = Σ_i state[col, i] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]   += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_reduce_sum_f32(attn_partial);

        // One writer per column. Matches upstream convention —
        // avoids 64-way write conflicts.
        if (lane == 0) {
            attn_bh[t * S_v + col] = attn_col;
        }
    }

    // Write state column to state_out (same transposed layout — contiguous write).
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        state_out_col[i] = s_shard[r];
    }
}

// Single template instantiation exposed to HIP module loader.
// S_v=128 covers qwen35 / qwen3next / any model using the default
// ssm.state_size from the GGUF metadata. Adding more S_v values later
// is mechanical — copy the extern "C" wrapper below with a new name.
//
// Phase 3: the kernel now accepts an `n_rep` parameter and indexes
// Q/K by `h_idx / n_rep` to do GQA head broadcast implicitly. The
// caller passes Q/K at `H / n_rep` heads (the KV head count) and
// V/state/gate/beta at `H` heads (the V/output head count).
// When `n_rep == 1` the behaviour is identical to the pre-Phase-3
// kernel.
extern "C" __global__ void __launch_bounds__(256, 1) gated_delta_net_step_s128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep) {
    gated_delta_net_step_impl<128>(q, k, v, gate, beta, state_in, state_out, attn_out, B, H, L, n_rep);
}
