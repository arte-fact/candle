// TurboQuant KV-cache kernels (TQ2_0, TQ3_0, TQ4_0) for gfx906.
//
// Ported from llama.cpp-turbo (`ggml-cuda/cpy-utils.cuh` for quant,
// `ggml-cuda/dequantize.cuh` + `ggml-turbo-quant.c` for dequant).
// Centroid tables are Lloyd-Max codebooks designed for unit-norm,
// rotated (WHT+dense) Gaussian-like distributions. WHT + dense rotation
// are NOT applied here — round-trip tests use synthetic Gaussian data so
// the codebook centres match. Real-model use needs WHT/rotation wired in
// as a separate phase (quality is garbage without them on non-Gaussian
// K/V distributions).
//
// Block layouts (match turbo's `ggml-common.h`):
//   TQ2_0: 32 values, 10 bytes   (fp16 norm + 8 B 2-bit indices)
//   TQ3_0: 32 values, 16 bytes   (fp16 norm + 8 B low-2 + 4 B sign + 2 B pad)
//   TQ4_0: 128 values, 68 bytes  (2*fp16 norm + 48 B 3-bit + 16 B signs)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Block structs (byte-identical to turbo's)
// -----------------------------------------------------------------------------
#define QK_TURBO2 32
#define QK_TURBO3 32
#define QK_TURBO4 128

typedef struct {
    __half  norm;                      // 2 B
    uint8_t qs[QK_TURBO2 / 4];         // 8 B (2-bit indices, 4 per byte)
} block_turbo2_0;                       // 10 B
static_assert(sizeof(block_turbo2_0) == 10, "turbo2_0 block size");

typedef struct {
    __half  norm;                      // 2 B
    uint8_t qs[QK_TURBO3 / 4];         // 8 B (low 2 bits of 3-bit index)
    uint8_t signs[QK_TURBO3 / 8];      // 4 B (high bit of 3-bit index)
    uint8_t pad[2];                     // 2 B padding → 16 B GDDR alignment
} block_turbo3_0;                       // 16 B
static_assert(sizeof(block_turbo3_0) == 16, "turbo3_0 block size");

typedef struct {
    __half  norm;                      // 2 B
    __half  rnorm;                     // 2 B (residual norm for QJL correction)
    uint8_t qs[QK_TURBO4 * 3 / 8];     // 48 B (3-bit indices, packed)
    uint8_t signs[QK_TURBO4 / 8];      // 16 B (1-bit QJL signs)
} block_turbo4_0;                       // 68 B
static_assert(sizeof(block_turbo4_0) == 68, "turbo4_0 block size");

// -----------------------------------------------------------------------------
// Centroid + midpoint tables (Lloyd-Max codebooks for rotated unit-Gaussian)
// -----------------------------------------------------------------------------
__constant__ float TB2_CENTROIDS[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
__constant__ float TB2_MIDPOINTS[3] = {
    -0.086728f, 0.0f, 0.086728f
};

__constant__ float TB3_CENTROIDS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
__constant__ float TB3_MIDPOINTS[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// TQ4 QJL sign arrays (seed=1042 in turbo). 2*128 signs for the residual correction.
__constant__ float QJL_SIGNS1[128] = {
     1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,
     1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,
     1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,
    -1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,
    -1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1
};
__constant__ float QJL_SIGNS2[128] = {
     1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
     1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,
     1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,
     1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,
    -1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1
};

// =============================================================================
// QUANTIZE kernels — one block_turboN_0 written per (B, H, T_new, D_block)
// =============================================================================
//
// Grid: (D / QK, H * T_new, B). Block: QK threads (32 for TQ2/TQ3, 128 for TQ4).
// Each thread owns one value. Norm is reduced inside the block.

// ---- TQ2_0 quantize-append ----
//
// Strategy: each thread computes its own 2-bit index. Indices shared via
// LDS. Thread 0 serialises the byte pack — avoids atomic alignment
// issues on the 10-byte block stride.
extern "C" __global__ void quantize_turbo2_0_append_f32(
    const float * __restrict__ x,       // (B, H, T_new, D) F32
    void        * __restrict__ cache,   // (B, H, T_max, D/QK_TURBO2) block_tq2_0
    const int B, const int H, const int T_new, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;          // 0..31
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_new = bh_t / H;
    const int h     = bh_t % H;
    const int d     = ib_d * QK_TURBO2 + tid;

    const int64_t x_idx = ((int64_t)((b * H + h) * T_new + t_new)) * D + d;
    const float xi = x[x_idx];

    float sq = xi * xi;
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor(sq, off, 32);
    }
    const float norm    = sqrtf(sq);
    const float invnorm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    const float val     = xi * invnorm;

    uint8_t idx = 0;
    idx += (val >= TB2_MIDPOINTS[0]);
    idx += (val >= TB2_MIDPOINTS[1]);
    idx += (val >= TB2_MIDPOINTS[2]);
    idx &= 0x3;

    __shared__ uint8_t sidx[QK_TURBO2];
    sidx[tid] = idx;
    __syncthreads();

    if (tid == 0) {
        const int blocks_per_row = D / QK_TURBO2;
        const int t_abs = t_offset + t_new;
        block_turbo2_0 * blk = (block_turbo2_0 *)cache
            + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
            + ib_d;
        blk->norm = __float2half(norm);
        for (int bi = 0; bi < (QK_TURBO2 / 4); ++bi) {
            uint8_t byte = 0;
            byte |= (sidx[bi * 4 + 0] & 0x3) << 0;
            byte |= (sidx[bi * 4 + 1] & 0x3) << 2;
            byte |= (sidx[bi * 4 + 2] & 0x3) << 4;
            byte |= (sidx[bi * 4 + 3] & 0x3) << 6;
            blk->qs[bi] = byte;
        }
    }
}

// ---- TQ2_0 dequantize ----
extern "C" __global__ void dequantize_turbo2_0_kv_f32(
    const void  * __restrict__ cache,
    float       * __restrict__ out,       // (B, H, T_sel, D)
    const int B, const int H, const int T_sel, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_rel = bh_t / H;
    const int h     = bh_t % H;
    const int t_abs = t_offset + t_rel;
    const int d     = ib_d * QK_TURBO2 + tid;

    const int blocks_per_row = D / QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[tid >> 2] >> ((tid & 3) << 1)) & 0x3;
    const float v = TB2_CENTROIDS[idx] * norm;

    const int64_t out_idx = ((int64_t)((b * H + h) * T_sel + t_rel)) * D + d;
    out[out_idx] = v;
}

// ---- TQ3_0 quantize-append ----
extern "C" __global__ void quantize_turbo3_0_append_f32(
    const float * __restrict__ x,
    void        * __restrict__ cache,   // (B, H, T_max, D/QK_TURBO3) block_tq3_0
    const int B, const int H, const int T_new, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_new = bh_t / H;
    const int h     = bh_t % H;
    const int d     = ib_d * QK_TURBO3 + tid;

    const int64_t x_idx = ((int64_t)((b * H + h) * T_new + t_new)) * D + d;
    const float xi = x[x_idx];

    float sq = xi * xi;
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor(sq, off, 32);
    }
    const float norm    = sqrtf(sq);
    const float invnorm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    const float val     = xi * invnorm;

    // 3-bit index via 7 midpoint comparisons.
    uint8_t idx = 0;
    idx += (val >= TB3_MIDPOINTS[0]);
    idx += (val >= TB3_MIDPOINTS[1]);
    idx += (val >= TB3_MIDPOINTS[2]);
    idx += (val >= TB3_MIDPOINTS[3]);
    idx += (val >= TB3_MIDPOINTS[4]);
    idx += (val >= TB3_MIDPOINTS[5]);
    idx += (val >= TB3_MIDPOINTS[6]);

    __shared__ uint8_t sidx3[QK_TURBO3];
    sidx3[tid] = idx;
    __syncthreads();

    if (tid == 0) {
        const int blocks_per_row = D / QK_TURBO3;
        const int t_abs = t_offset + t_new;
        block_turbo3_0 * blk = (block_turbo3_0 *)cache
            + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
            + ib_d;
        blk->norm = __float2half(norm);
        blk->pad[0] = 0;
        blk->pad[1] = 0;
        // Low 2 bits into qs[] (4 indices per byte).
        for (int bi = 0; bi < (QK_TURBO3 / 4); ++bi) {
            uint8_t byte = 0;
            byte |= (sidx3[bi * 4 + 0] & 0x3) << 0;
            byte |= (sidx3[bi * 4 + 1] & 0x3) << 2;
            byte |= (sidx3[bi * 4 + 2] & 0x3) << 4;
            byte |= (sidx3[bi * 4 + 3] & 0x3) << 6;
            blk->qs[bi] = byte;
        }
        // High 1 bit into signs[] (8 indices per byte).
        for (int bi = 0; bi < (QK_TURBO3 / 8); ++bi) {
            uint8_t byte = 0;
            for (int bit = 0; bit < 8; ++bit) {
                if ((sidx3[bi * 8 + bit] >> 2) & 0x1) byte |= (1u << bit);
            }
            blk->signs[bi] = byte;
        }
    }
}

// ---- TQ3_0 dequantize ----
extern "C" __global__ void dequantize_turbo3_0_kv_f32(
    const void  * __restrict__ cache,
    float       * __restrict__ out,
    const int B, const int H, const int T_sel, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_rel = bh_t / H;
    const int h     = bh_t % H;
    const int t_abs = t_offset + t_rel;
    const int d     = ib_d * QK_TURBO3 + tid;

    const int blocks_per_row = D / QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[tid >> 2] >> ((tid & 3) << 1)) & 0x3;
    const uint8_t hi1  = (blk->signs[tid >> 3] >> (tid & 7)) & 0x1;
    const uint8_t idx  = low2 | (hi1 << 2);
    const float v = TB3_CENTROIDS[idx] * norm;

    const int64_t out_idx = ((int64_t)((b * H + h) * T_sel + t_rel)) * D + d;
    out[out_idx] = v;
}

// =============================================================================
// TQ4_0 — 128-element blocks, 3-bit PolarQuant + 1-bit QJL residual.
// =============================================================================
//
// Quantize is structured in two halves (j∈[0,64) and j∈[64,128)); the
// residual uses QJL_SIGNS1/2 to add/subtract a scaled correction term.
// The bit-stream for 3-bit indices crosses byte boundaries — we pack
// sequentially from thread 0 upward using atomicOr + per-thread bit
// positions.
//
// Reference: turbo `ggml-turbo-quant.c::quantize_row_turbo4_0_ref`
// (too long to inline here — match the math exactly).

extern "C" __global__ void quantize_turbo4_0_append_f32(
    const float * __restrict__ x,
    void        * __restrict__ cache,   // (B, H, T_max, D/QK_TURBO4) block_tq4_0
    const int B, const int H, const int T_new, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;          // 0..127
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_new = bh_t / H;
    const int h     = bh_t % H;
    const int d     = ib_d * QK_TURBO4 + tid;

    const int64_t x_idx = ((int64_t)((b * H + h) * T_new + t_new)) * D + d;
    const float xi = x[x_idx];

    // Two-stage reduction: wave64 fold + 2-wave LDS combine.
    float sq = xi * xi;
    for (int off = 32; off > 0; off >>= 1) {
        sq += __shfl_xor(sq, off, 64);
    }
    __shared__ float ssq[2];
    const int wave = tid >> 6;
    const int lane = tid & 63;
    if (lane == 0) ssq[wave] = sq;
    __syncthreads();
    const float norm = sqrtf(ssq[0] + ssq[1]);
    const float invnorm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    // MSE step: nearest-centroid 3-bit index on the normalised value.
    const float val = xi * invnorm;
    uint8_t idx = 0;
    idx += (val >= TB3_MIDPOINTS[0]);
    idx += (val >= TB3_MIDPOINTS[1]);
    idx += (val >= TB3_MIDPOINTS[2]);
    idx += (val >= TB3_MIDPOINTS[3]);
    idx += (val >= TB3_MIDPOINTS[4]);
    idx += (val >= TB3_MIDPOINTS[5]);
    idx += (val >= TB3_MIDPOINTS[6]);
    const uint8_t idx3 = idx & 0x7;

    // QJL residual: difference between the original and MSE reconstruction,
    // projected onto a fixed random ±1 basis per half.
    const float mse_recon = TB3_CENTROIDS[idx3];
    const float resid     = val - mse_recon;
    const float sign_lo   = QJL_SIGNS1[tid];
    const float sign_hi   = QJL_SIGNS2[tid];

    // QJL scale: project residual onto sign basis and take sign of the
    // projection. Using a 64-lane reduction across the block for each
    // half (turbo splits d in two contiguous halves).
    const float proj_lo = resid * sign_lo;
    const float proj_hi = resid * sign_hi;
    float s_lo = proj_lo, s_hi = proj_hi;
    for (int off = 32; off > 0; off >>= 1) {
        s_lo += __shfl_xor(s_lo, off, 64);
        s_hi += __shfl_xor(s_hi, off, 64);
    }
    __shared__ float slo[2], shi[2];
    if (lane == 0) { slo[wave] = s_lo; shi[wave] = s_hi; }
    __syncthreads();
    const float proj_sum_lo = slo[0] + slo[1];
    const float proj_sum_hi = shi[0] + shi[1];
    // Sign bit for this thread: 1 if projection(+resid) along its own
    // sign aligns with the block sum (simplified — exact math in turbo
    // uses a per-half norm estimate).
    const uint8_t s_bit = ((tid < 64 ? proj_sum_lo : proj_sum_hi)
                            * (tid < 64 ? sign_lo : sign_hi)) > 0.0f;

    // rnorm: magnitude of the residual projection averaged over the block.
    const float rnorm_val = fabsf(tid < 64 ? proj_sum_lo : proj_sum_hi)
                            * (1.0f / 128.0f);

    // Exchange per-thread (idx3, s_bit) via LDS. Thread 0 serialises
    // the byte-stream pack of 3-bit indices + 1-bit signs.
    __shared__ uint8_t sidx4[QK_TURBO4];
    __shared__ uint8_t ssign4[QK_TURBO4];
    sidx4[tid] = idx3;
    ssign4[tid] = s_bit;
    __syncthreads();

    if (tid == 0) {
        const int blocks_per_row = D / QK_TURBO4;
        const int t_abs = t_offset + t_new;
        block_turbo4_0 * blk = (block_turbo4_0 *)cache
            + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
            + ib_d;
        blk->norm  = __float2half(norm);
        blk->rnorm = __float2half(rnorm_val);
        // Zero qs and signs, then bit-stream pack.
        for (int k = 0; k < (QK_TURBO4 * 3 / 8); ++k) blk->qs[k] = 0;
        for (int k = 0; k < (QK_TURBO4 / 8); ++k) blk->signs[k] = 0;
        for (int j = 0; j < QK_TURBO4; ++j) {
            const int bo = j * 3;
            const int bi = bo >> 3;
            const int bp = bo & 7;
            const uint16_t bits = (uint16_t)(sidx4[j] & 0x7) << bp;
            blk->qs[bi] |= (uint8_t)(bits & 0xff);
            if ((bp + 3) > 8 && bi + 1 < (QK_TURBO4 * 3 / 8)) {
                blk->qs[bi + 1] |= (uint8_t)(bits >> 8);
            }
            if (ssign4[j]) {
                blk->signs[j >> 3] |= (uint8_t)(1u << (j & 7));
            }
        }
    }
}

// ---- TQ4_0 dequantize (direct port of turbo's `dequantize_turbo4_0`) ----
extern "C" __global__ void dequantize_turbo4_0_kv_f32(
    const void  * __restrict__ cache,
    float       * __restrict__ out,
    const int B, const int H, const int T_sel, const int D,
    const int T_max, const int t_offset
) {
    const int tid   = threadIdx.x;          // 0..127
    const int ib_d  = blockIdx.x;
    const int bh_t  = blockIdx.y;
    const int b     = blockIdx.z;

    const int t_rel = bh_t / H;
    const int h     = bh_t % H;
    const int t_abs = t_offset + t_rel;
    const int d     = ib_d * QK_TURBO4 + tid;

    const int blocks_per_row = D / QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)cache
        + (int64_t)((b * H + h) * T_max + t_abs) * blocks_per_row
        + ib_d;

    const float norm  = __half2float(blk->norm);
    const float rnorm = __half2float(blk->rnorm);
    // qjl_scale matches turbo: 1.2533141 / 128 * rnorm
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;

    // Unpack 3-bit index across byte boundary.
    const int bo = tid * 3;
    const int bi = bo >> 3;
    const int bp = bo & 7;
    uint16_t raw = (uint16_t)blk->qs[bi];
    if (bi + 1 < (QK_TURBO4 * 3 / 8)) {
        raw |= (uint16_t)blk->qs[bi + 1] << 8;
    }
    const uint8_t idx = (raw >> bp) & 0x7;

    // QJL sign (1 or -1)
    const float s = (blk->signs[tid >> 3] & (1u << (tid & 7))) ? 1.0f : -1.0f;

    const float v = (TB3_CENTROIDS[idx] + s * qjl_scale) * norm;

    const int64_t out_idx = ((int64_t)((b * H + h) * T_sel + t_rel)) * D + d;
    out[out_idx] = v;
}
