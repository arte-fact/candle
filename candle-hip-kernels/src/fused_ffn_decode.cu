// fused_ffn_decode.cu — Persistent decode kernel for the FFN path.
//
// Fuses: rmsnorm → gate matmul → up matmul → silu*gate → down matmul → residual add
// into a single kernel launch, eliminating 7 separate kernel launches per layer.
//
// For TinyLlama decode (hidden=2048, intermediate=8704), this saves ~154 launches
// per token (7 per layer × 22 layers).
//
// Design:
//   - Block: WARP_SIZE threads (64 on gfx906)
//   - Each thread handles hidden_dim/WARP_SIZE elements (32 for hidden=2048)
//   - Hidden state stays in registers between operations
//   - Weight data streamed from global memory per mat-vec
//   - LDS used for warp reductions (rmsnorm) and intermediate storage
//
// Quantized mat-vec: each mat-vec reads the ENTIRE weight matrix row-by-row.
// For Q4_0 weights: each row = hidden_dim * 18 / 32 bytes.
// The kernel accumulates partial dot products per-thread, then the thread
// that owns the output element collects the result.
//
// Limitations:
//   - Only Q4_0 weights supported initially
//   - hidden_dim must be a multiple of WARP_SIZE (64)
//   - intermediate_dim must be a multiple of WARP_SIZE (64)

#include "compatibility.cuh"
#include "gfx906_primitives.cuh"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define QK4_0_LOCAL 32

// block_q4_0 layout: { half d; uint8_t qs[16]; } = 18 bytes
struct block_q4_0_local {
    half d;
    uint8_t qs[QK4_0_LOCAL / 2];
};

static __device__ __forceinline__ int local_dp4a(const int a, const int b, int c) {
#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__))
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// Fused FFN decode kernel for Q4_0 weights.
//
// Parameters:
//   h:            (hidden_dim,) f32 — input hidden state (also output, in-place)
//   ffn_norm_w:   (hidden_dim,) f32 — rmsnorm weight
//   w_gate:       block_q4_0 * — gate projection weights (intermediate × hidden)
//   w_up:         block_q4_0 * — up projection weights (intermediate × hidden)
//   w_down:       block_q4_0 * — down projection weights (hidden × intermediate)
//   hidden_dim:   int
//   intermediate: int
//   eps:          float — rmsnorm epsilon
//
// Grid: (1, 1, 1) — single block, single wavefront
// Block: (WARP_SIZE, 1, 1) = (64, 1, 1)
//
extern "C" __global__ void __launch_bounds__(64, 1)
fused_ffn_decode_q4_0(
    float * __restrict__ h,
    const float * __restrict__ ffn_norm_w,
    const void * __restrict__ w_gate,
    const void * __restrict__ w_up,
    const void * __restrict__ w_down,
    const int hidden_dim,
    const int intermediate_dim,
    const float eps)
{
    const int tid = threadIdx.x;

    // How many hidden elements per thread
    const int h_per_thread = hidden_dim / WARP_SIZE;
    // How many intermediate elements per thread
    const int i_per_thread = intermediate_dim / WARP_SIZE;

    // ---- Step 1: Load hidden state into registers ----
    float h_reg[32]; // max 32 elements per thread (hidden_dim up to 2048)
    for (int i = 0; i < h_per_thread; ++i) {
        h_reg[i] = h[tid + i * WARP_SIZE];
    }

    // Save residual
    float residual[32];
    for (int i = 0; i < h_per_thread; ++i) {
        residual[i] = h_reg[i];
    }

    // ---- Step 2: RMSNorm ----
    float sum_sq = 0.0f;
    for (int i = 0; i < h_per_thread; ++i) {
        sum_sq += h_reg[i] * h_reg[i];
    }
    sum_sq = gfx906_warp_reduce_sum(sum_sq);
    const float rms_scale = rsqrtf(sum_sq / hidden_dim + eps);

    // Apply norm + weight
    float normed[32];
    for (int i = 0; i < h_per_thread; ++i) {
        normed[i] = h_reg[i] * rms_scale * ffn_norm_w[tid + i * WARP_SIZE];
    }

    // ---- Step 3: Gate and Up mat-vec ----
    // For each output element of gate/up, ALL threads cooperate:
    // Each thread computes partial dot product over its h_per_thread elements,
    // then warp-reduce gives the full dot product.
    //
    // Output: gate_val[i_per_thread], up_val[i_per_thread]
    // These are stored in LDS because intermediate_dim > WARP_SIZE.

    // Dynamic shared memory — caller passes
    // `2 * intermediate_dim * sizeof(float)` as the launch shared_mem_bytes.
    // This avoids the gfx906 64 KiB static-LDS hard cap when the kernel is
    // compiled with a generous max (intermediate_dim up to 8192).
    extern __shared__ float dyn_smem[];
    float * gate_lds = dyn_smem;
    float * up_lds   = dyn_smem + intermediate_dim;

    const int blocks_per_row = hidden_dim / QK4_0_LOCAL;
    const block_q4_0_local * gate_q4 = (const block_q4_0_local *) w_gate;
    const block_q4_0_local * up_q4   = (const block_q4_0_local *) w_up;

    // Process each output row of gate and up projections
    for (int out_row = 0; out_row < intermediate_dim; ++out_row) {
        const block_q4_0_local * g_row = gate_q4 + out_row * blocks_per_row;
        const block_q4_0_local * u_row = up_q4   + out_row * blocks_per_row;

        float g_sum = 0.0f;
        float u_sum = 0.0f;

        // Each thread processes its assigned blocks of the row
        // Thread tid handles elements [tid*h_per_thread .. (tid+1)*h_per_thread)
        // which spans blocks tid*h_per_thread/32 ..
        for (int i = 0; i < h_per_thread; ++i) {
            const int elem = tid + i * WARP_SIZE;
            const int block_idx = elem / QK4_0_LOCAL;
            const int elem_in_block = elem % QK4_0_LOCAL;

            // Dequantize Q4_0
            const block_q4_0_local * gb = &g_row[block_idx];
            const block_q4_0_local * ub = &u_row[block_idx];

            const float g_d = __half2float(gb->d);
            const float u_d = __half2float(ub->d);

            int g_q, u_q;
            if (elem_in_block < 16) {
                g_q = (gb->qs[elem_in_block] & 0x0F) - 8;
                u_q = (ub->qs[elem_in_block] & 0x0F) - 8;
            } else {
                g_q = (gb->qs[elem_in_block - 16] >> 4) - 8;
                u_q = (ub->qs[elem_in_block - 16] >> 4) - 8;
            }

            g_sum += normed[i] * (g_d * g_q);
            u_sum += normed[i] * (u_d * u_q);
        }

        // Warp reduce to get full dot products
        g_sum = gfx906_warp_reduce_sum(g_sum);
        u_sum = gfx906_warp_reduce_sum(u_sum);

        // Thread 0 writes result
        if (tid == 0) {
            // SiLU(gate) * up — fused with the matmul output
            const float silu_g = g_sum / (1.0f + gfx906_fast_exp(-g_sum));
            gate_lds[out_row] = silu_g * u_sum;
        }
    }

    __syncthreads(); // All intermediate values ready in LDS

    // ---- Step 4: Down mat-vec ----
    // Read intermediate from LDS, multiply by down projection
    const block_q4_0_local * down_q4 = (const block_q4_0_local *) w_down;
    const int down_blocks_per_row = intermediate_dim / QK4_0_LOCAL;

    for (int i = 0; i < h_per_thread; ++i) {
        const int out_elem = tid + i * WARP_SIZE;
        const block_q4_0_local * d_row = down_q4 + out_elem * down_blocks_per_row;

        float d_sum = 0.0f;
        for (int ib = 0; ib < down_blocks_per_row; ++ib) {
            const float d_scale = __half2float(d_row[ib].d);
            for (int j = 0; j < 16; ++j) {
                const int elem_lo = ib * 32 + j;
                const int elem_hi = ib * 32 + j + 16;
                const int q_lo = (d_row[ib].qs[j] & 0x0F) - 8;
                const int q_hi = (d_row[ib].qs[j] >> 4) - 8;
                d_sum += gate_lds[elem_lo] * (d_scale * q_lo);
                d_sum += gate_lds[elem_hi] * (d_scale * q_hi);
            }
        }

        // ---- Step 5: Residual add + write back ----
        h[out_elem] = residual[i] + d_sum;
    }
}
