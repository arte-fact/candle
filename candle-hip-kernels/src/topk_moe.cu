// topk_moe.cu — Phase T1.2: fused softmax + top-k + (optional) normalize for MoE routing.
//
// Replaces the 7-step Rust chain in
// `candle-transformers/src/models/quantized_gemma4.rs:1762-1779`:
//   softmax → to_device(Cpu) → arg_sort_last_dim → narrow → to_device(Hip)
//          → gather → sum_keepdim → broadcast_div
// with a single kernel launch per layer.  Per T1.3 finding, this is the
// real driver of the 0.50× turbo gap on multi-device MoE — the matmul
// kernels are equivalent algorithm; only orchestration differs.
//
// Ported from `/artefact/llamacpp-turbo/.../ggml-cuda/topk-moe.cu:79+`.
// Stripped to the gemma4 case: with_norm=true, no bias, no sigmoid, no
// delayed_softmax, no clamp.  Output layout is COMPACT
// (weights/ids stride is n_expert_used, not n_experts) — different from
// turbo's packed layout but matches what gemma4's downstream
// `indexed_moe_forward` expects.
//
// Block: 4 wavefronts × 64 threads = 256 threads.
//        Each wavefront processes one row (n_rows is tokens × seq_len).
// Grid:  n_rows / 4 blocks.
//
// Template params:
//   N_EXPERTS  — compile-time number of experts (must be multiple of 64
//                or ≤ 64).  Instantiated for 128 (gemma4-26B-A4B,
//                qwen3moe), 64 (qwen3-coder-30B-A3B), 32 (smaller MoEs).

#include "compatibility.cuh"
#include "gfx906_primitives.cuh"

#include <hip/hip_runtime.h>
#include <float.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// In-warp softmax: each lane holds `EXPERTS_PER_THREAD` slots of the
// distribution.  Output is in-place.
template <int EXPERTS_PER_THREAD>
static __device__ __forceinline__ void softmax_warp_inplace_64(
    float (&vals)[EXPERTS_PER_THREAD], const int n_experts, const int lane
) {
    float max_val = -INFINITY;
    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; i++) {
        const int idx = lane + i * WARP_SIZE;
        if (idx < n_experts) {
            max_val = fmaxf(max_val, vals[i]);
        }
    }
    max_val = gfx906_warp_reduce_max(max_val);

    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; i++) {
        const int idx = lane + i * WARP_SIZE;
        if (idx < n_experts) {
            const float v = expf(vals[i] - max_val);
            vals[i] = v;
            sum += v;
        } else {
            vals[i] = 0.f;
        }
    }
    sum = gfx906_warp_reduce_sum(sum);
    const float inv_sum = 1.f / sum;
    #pragma unroll
    for (int i = 0; i < EXPERTS_PER_THREAD; i++) {
        vals[i] *= inv_sum;
    }
}

// Specialised: with_norm=true, no bias.  Compact output stride.
//
// Args:
//   logits         : [n_rows × N_EXPERTS] f32, row-major (input)
//   weights_out    : [n_rows × n_expert_used] f32, row-major (output)
//   ids_out        : [n_rows × n_expert_used] u32, row-major (output)
//   n_rows         : token count
//   n_expert_used  : top-k value (caller's `num_experts_used`)
template <int N_EXPERTS>
static __device__ __forceinline__ void topk_moe_norm_f32_n(
    const float * __restrict__ logits,
    float       * __restrict__ weights_out,
    unsigned int* __restrict__ ids_out,
    const int n_rows,
    const int n_expert_used
) {
    const int lane    = threadIdx.x;
    const int warp_in = threadIdx.y;
    const int row     = blockIdx.x * blockDim.y + warp_in;
    if (row >= n_rows) return;

    const float * row_logits = logits      + (size_t)row * N_EXPERTS;
    float       * row_w      = weights_out + (size_t)row * n_expert_used;
    unsigned int* row_ids    = ids_out     + (size_t)row * n_expert_used;

    constexpr int EPT = (N_EXPERTS > WARP_SIZE) ? (N_EXPERTS / WARP_SIZE) : 1;
    float wt[EPT];

    // Initialise + load.
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        const int idx = lane + i * WARP_SIZE;
        wt[i] = (N_EXPERTS % WARP_SIZE == 0 || idx < N_EXPERTS) ? row_logits[idx] : -INFINITY;
    }

    // Softmax over n_experts.
    softmax_warp_inplace_64<EPT>(wt, N_EXPERTS, lane);

    // NaN sanitise — keeps argmax deterministic.
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (__isnanf(wt[i])) wt[i] = -FLT_MAX;
    }

    // Top-k argmax loop.  Each iteration: warp-reduce-max(wt), assign the
    // winner, mark winner slot -INFINITY for the next round.
    float wt_sum = 0.f;
    for (int k = 0; k < n_expert_used; k++) {
        // Per-thread argmax across its EPT slots.
        float max_val    = wt[0];
        int   max_expert = lane;
        #pragma unroll
        for (int i = 1; i < EPT; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((N_EXPERTS % WARP_SIZE == 0 || expert < N_EXPERTS) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
        // Warp-reduce: pick the winning (val, expert).  Tie-break by
        // smaller expert id for determinism (matches turbo's policy).
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val_x    = __shfl_xor(max_val, mask, WARP_SIZE);
            const int   expert_x = __shfl_xor(max_expert, mask, WARP_SIZE);
            if (val_x > max_val || (val_x == max_val && expert_x < max_expert)) {
                max_val    = val_x;
                max_expert = expert_x;
            }
        }
        // Mark winner -INF in its owning lane's slot.
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        // Lane 0 of the warp writes both arrays at index k.
        if (lane == 0) {
            row_w[k]   = max_val;
            row_ids[k] = (unsigned int)max_expert;
            wt_sum    += max_val;
        }
    }

    // Normalise: lane 0 has the sum; broadcast inv_sum across the warp,
    // then each lane scales the writes it owns.  Since lane 0 wrote
    // every slot, we re-write here.
    if (lane == 0) {
        const float inv_sum = 1.f / fmaxf(wt_sum, 1e-20f);
        for (int k = 0; k < n_expert_used; k++) {
            row_w[k] *= inv_sum;
        }
    }
}

// Explicit instantiations.  Add more here as new MoE models land.
extern "C" __launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_norm_f32_n128(
    const float * logits,
    float       * weights_out,
    unsigned int* ids_out,
    const int n_rows,
    const int n_expert_used
) {
    topk_moe_norm_f32_n<128>(logits, weights_out, ids_out, n_rows, n_expert_used);
}

extern "C" __launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_norm_f32_n64(
    const float * logits,
    float       * weights_out,
    unsigned int* ids_out,
    const int n_rows,
    const int n_expert_used
) {
    topk_moe_norm_f32_n<64>(logits, weights_out, ids_out, n_rows, n_expert_used);
}

extern "C" __launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_norm_f32_n512(
    const float * logits,
    float       * weights_out,
    unsigned int* ids_out,
    const int n_rows,
    const int n_expert_used
) {
    topk_moe_norm_f32_n<512>(logits, weights_out, ids_out, n_rows, n_expert_used);
}
