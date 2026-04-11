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

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

static __device__ __forceinline__ float warp_reduce_sum_f32(float x) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
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
    int B, int H, int L) {

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

    // (b, h) slice base offsets.
    const int bh = b_idx * H + h_idx;
    const float * q_bh        = q         + (int64_t)bh * L * S_v;
    const float * k_bh        = k         + (int64_t)bh * L * S_v;
    const float * v_bh        = v         + (int64_t)bh * L * S_v;
    const float * gate_bh     = gate      + (int64_t)bh * L;
    const float * beta_bh     = beta      + (int64_t)bh * L;
    const float * state_in_bh = state_in  + (int64_t)bh * S_v * S_v;
    float *       state_out_bh = state_out + (int64_t)bh * S_v * S_v;
    float *       attn_bh      = attn_out  + (int64_t)bh * L * S_v;

    // Load column `col` of state_in into registers:
    //   s_shard[r] = state_in[i, col] with i = r * warp_size + lane.
    // The loads are strided by S_v (one per row); they fire once per warp
    // per kernel invocation and are amortized over the entire L-step loop.
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = state_in_bh[i * S_v + col];
    }

    for (int t = 0; t < L; t++) {
        // Per-token scalars. All lanes read the same memory location —
        // scalar-broadcast via GCN L1.
        const float g_val    = expf(gate_bh[t]);
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

        // Apply decay: state[i, col] *= g_val
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= g_val;
        }

        // sk[col] = Σ_i state[i, col] * k[i]
        // Each lane sums its owned rows, then warp-reduces.
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = warp_reduce_sum_f32(kv_shard);

        // delta[col] = (v[col] - sk[col]) * beta. v[col] is a scalar the
        // whole warp reads via L1 broadcast.
        const float v_col     = v_bh[t * S_v + col];
        const float delta_col = (v_col - kv_col) * beta_val;

        // Fused: state[i, col] += k[i] * delta_col
        //        attn[col]     = Σ_i state[i, col] * q[i]
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

    // Write state column to state_out.
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        state_out_bh[i * S_v + col] = s_shard[r];
    }
}

// Single template instantiation exposed to HIP module loader.
// S_v=128 covers qwen35 / qwen3next / any model using the default
// ssm.state_size from the GGUF metadata. Adding more S_v values later
// is mechanical — copy the extern "C" wrapper below with a new name.
extern "C" __global__ void gated_delta_net_step_s128_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L) {
    gated_delta_net_step_impl<128>(q, k, v, gate, beta, state_in, state_out, attn_out, B, H, L);
}
