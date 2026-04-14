// MMQ turbo port — Stage 2.
//
// Reproduces turbo's Q4_0 MMQ kernel geometry and compute:
//   mmq_y  = 128  (output rows per WG)
//   mmq_x  ∈ {8,16,32,64}  (output cols per WG)
//   nwarps = 4    (cooperating waves)
//   wg     = 64×4 = 256 threads
//   LDS    = ~20-40 KB depending on mmq_x
//
// Port-source lines (turbo ggml-cuda/mmq.cuh, 2026-04-14):
//   309-368  load_tiles_q4_0
//   371-425  vec_dot_q4_0_q8_1_dp4a
//   3530-3621  mul_mat_q_process_tile (K-loop scaffolding)
//   3638-3730  mul_mat_q main kernel (regular-matmul, non-CDNA branch)
//
// Differences vs turbo:
//   - Drop MoE (ids_dst, expert_bounds): we use identity col mapping.
//   - Drop multi-channel / multi-sample: single matmul.
//   - Drop stream-K: use xy-tiling branch (non-CDNA).
//   - Drop MMA path: gfx906 has no MFMA, dp4a only.
//   - Drop L2-prefetch: add back in stage 3 if useful.
//
// Shared Y layout: turbo's `block_q8_1_mmq` (144 B/block, 128 f32s → 4×
// half2 ds + 128 int8 qs), stored (k_big_block, col) row-major.
// Candle's existing `quantize_q8_1` output (per-(row,k_block) 36 B blocks)
// is NOT compatible; we ship a new `quantize_q8_1_mmq_q4_0` kernel here.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ================================================================
// Block layouts
// ================================================================

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))   // 4 ints per Q4_0 block
typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 2 + QK4_0 / 2, "block_q4_0 size");

// Q4_1: nibbles [0..15] (no centering offset like Q4_0); per-block has
// dm = (delta, min) packed as half2 at byte 0..3, qs at byte 4..19.
// Total 20 bytes; qs is 4-byte aligned (vs Q4_0's 2-byte).
#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))   // 4 ints per Q4_1 block
typedef struct {
    half2   dm;                       // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];
} block_q4_1;
static_assert(sizeof(block_q4_1) == 4 + QK4_1 / 2, "block_q4_1 size");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))   // 8 ints per Q8_1 block

// Turbo's "mmq" Q8_1 block: 128 f32s packed as DS4 header + 128 int8 qs.
// Byte 0..15  : half2 ds4[4]  (one (d, d*sum) per 32-elem sub-block)
// Byte 16..143: int8  qs[128] (quantized values)
struct block_q8_1_mmq {
    half2  ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "block_q8_1_mmq size");
#define Q8_1_MMQ_BYTES 144
#define Q8_1_MMQ_INTS  (Q8_1_MMQ_BYTES / 4)  // 36

// ================================================================
// Tile geometry
// ================================================================

#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#ifndef MMQ_NWARPS
#define MMQ_NWARPS 4
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define MMQ_TILE_NE_K 32                                 // K elements per vec_dot call
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)  // = 36 ints/col per Y half-tile
#define MMQ_ITER_K    256                                // K elements consumed per outer iter
#define VDR_Q4_0_Q8_1_MMQ 4

// LDS layout sizes (DP4A path, per turbo MMQ_DP4A_TXS_Q4_0):
//   x_qs: mmq_y*(MMQ_TILE_NE_K + 1)                 ints   (33 per row, +1 bank pad)
//   x_df: mmq_y*(MMQ_TILE_NE_K/QI4_0) + mmq_y/QI4_0 floats (8 + 32)
//   y_qs: mmq_x * MMQ_TILE_Y_K                      ints   (36 per col)
#define X_QS_INTS (MMQ_Y * (MMQ_TILE_NE_K + 1))
#define X_DF_FLTS (MMQ_Y * (MMQ_TILE_NE_K / QI4_0) + MMQ_Y / QI4_0)

// Pad Y tile size up to a multiple of nwarps*warp_size to match turbo's
// cooperative-load stride.
#define GGML_PAD(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

// ================================================================
// Helpers
// ================================================================

static __device__ __forceinline__ int get_int_b2(const void * x, int i32) {
    // Match turbo vecdotq.cuh:23 — two aligned uint16_t loads, OR'd.
    // Used for Q4_0 whose qs starts at a 2-byte (not 4-byte) offset.
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = (int)x16[2*i32 + 0] <<  0;
    x32     |= (int)x16[2*i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, int i32) {
    // Match turbo vecdotq.cuh:32 — direct int load. Used for Q4_1 whose qs
    // starts at a 4-byte offset (after the 4-byte dm half2 header).
    return ((const int *) x)[i32];
}

// gfx906 dp4a: v_dot4_i32_i8.  Same intrinsic turbo uses.
static __device__ __forceinline__ int dp4a_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Port of turbo vecdotq.cuh:138-157 (vec_dot_q4_0_q8_1_impl<vdr=4>).
static __device__ __forceinline__ float q4_0_q8_1_dp4a_4(
    const int * v,          // 4 ints of Q4_0 qs (packed nibbles)
    const int * u,          // 8 ints of Q8_1 qs
    const float d4,         // Q4_0 delta
    const half2 ds8) {      // Q8_1 (d, sum_raw)

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = dp4a_sdot4(vi0, u[2*i + 0], sumi);
        sumi = dp4a_sdot4(vi1, u[2*i + 1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    // "- (8*vdr/QI4_0) * ds8f.y" = "- 8 * ds8f.y" for vdr=4, QI4_0=4.
    return d4 * (sumi * ds8f.x - 8.0f * ds8f.y);
}

// Port of turbo vecdotq.cuh:162-190 (vec_dot_q4_1_q8_1_impl<vdr=4>).
// Q4_1 has no -8 nibble offset; instead applies (d4d8 * sumi + m4s8) with
// the bias term coming from Q4_1's per-block min.
static __device__ __forceinline__ float q4_1_q8_1_dp4a_4(
    const int * v,          // 4 ints of Q4_1 qs (packed nibbles)
    const int * u,          // 8 ints of Q8_1 qs
    const half2 dm4,        // Q4_1 (delta, min)
    const half2 ds8) {      // Q8_1 (d, sum_raw)

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = dp4a_sdot4(vi0, u[2*i + 0], sumi);
        sumi = dp4a_sdot4(vi1, u[2*i + 1], sumi);
    }
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    // "QI8_1/(vdr*QR4_1) = 8/(4*2) = 1" so the divisor drops out.
    return sumi * (dm4f.x * ds8f.x) + (dm4f.y * ds8f.y);
}

// ================================================================
// Y prequantization: f32 → block_q8_1_mmq laid out (big_block, col).
// One WG per (big_block, col).  Block = 128 threads = 2 waves.
// Each thread owns 1 of 128 K-elements of its big-block for one col.
//
// This single kernel serves both Q4_0 and Q4_1 (both use turbo's DS4
// Q8_1 layout — turbo mmq.cuh:62-69 `mmq_get_q8_1_ds_layout`).
// The kept-original symbol name is `quantize_q8_1_mmq_q4_0` for
// backward-compat with the M2 Rust dispatcher; the Q4_1 dispatcher
// reuses this symbol.
// ================================================================

extern "C" __global__ void quantize_q8_1_mmq_q4_0(
    const float * __restrict__ x,    // [total_b, ncols] row-major (col-dim is K)
    void *        __restrict__ vy,   // [n_big_blocks * total_b] blocks of 144 B
    const int ncols,                 // K dim
    const int total_b) {             // batch (= ncols_y)

    const int b   = blockIdx.x;      // big-block index along K
    const int c   = blockIdx.y;      // output col (batch row)
    const int tid = threadIdx.x;     // 0..127
    const int sub = tid / 32;        // 0..3
    const int lane_in_sub = tid % 32;

    const int ki = b * 128 + tid;    // global K index
    const float xi = (ki < ncols) ? x[c * ncols + ki] : 0.0f;

    // Reduce max|xi| and sum(xi) within the 32-lane sub-block.
    // HIP __shfl_xor with width=32 keeps the lanes partitioned.
    float amax = fabsf(xi);
    float ssum = xi;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        ssum = ssum + __shfl_xor(ssum, m, 32);
    }

    const float d = amax / 127.0f;
    const int8_t q =
        (amax == 0.0f) ? (int8_t)0
                       : (int8_t)__float2int_rn(xi / d);

    uint8_t * y_bytes = (uint8_t *) vy
                        + ((size_t)(b * total_b + c)) * Q8_1_MMQ_BYTES;

    // qs[tid] is just the per-thread int8.
    ((int8_t *) (y_bytes + 16))[tid] = q;

    // Lane 0 of each sub-block writes the (d, sum) half2.  ds8.y is the
    // *raw* f32 sum of the sub-block (turbo quantize.cu:346 —
    // `ds4[iqs/32] = make_half2(d, sum)`).  The vec_dot formula uses it
    // as `d4 * (sumi*d_y - 8*sum_raw)` where 8*sum_raw ≈ 8*d_y*Σy_int8
    // reverses the Q4_0 nibble-to-int8 offset.
    if (lane_in_sub == 0) {
        half2 * ds = (half2 *) y_bytes;
        ds[sub] = __floats2half2_rn(d, ssum);
    }
}

// ================================================================
// load_tiles_q4_0
// Port of turbo mmq.cuh:309-368 (DP4A path; MMA path stripped).
// 4 waves cooperate to load MMQ_Y=128 rows × 8 Q4_0 blocks into LDS.
// ================================================================

template <bool need_check>
static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x,      // raw Q4_0 byte pointer
    int *        __restrict__ x_qs,   // LDS x_qs, MMQ_Y * (MMQ_TILE_NE_K + 1) ints
    float *      __restrict__ x_df,   // LDS x_df (scales)
    const int kbx0,                   // starting Q4_0 block index for this tile's K-window
    const int i_max,                  // row upper bound (for need_check)
    const int stride_row_x) {         // Q4_0 blocks per X row

    // tile_x row = MMQ_Y=128.  Each K-iter (ITER_K=256 elements = 8 Q4_0 blocks)
    // loads 32 ints per row = 8 blocks × 4 ints/block.  threads_per_row = 32.
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);  // = 32
    constexpr int nrows           = WARP_SIZE / threads_per_row; // = 2
    const int txi  = threadIdx.x % threads_per_row;            // 0..31
    const int kbx  = txi / QI4_0;                              // 0..7 (block within K-iter)
    const int kqsx = txi % QI4_0;                              // 0..3 (int within block)

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        // Each thread loads row i. Per-warp: rows [i0 + y*nrows .. +nrows-1].
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride_row_x + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
    }

    // Scales x_df.  blocks_per_tile_x_row = 8 (one Q4_0 block per K-iter step=32).
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;   // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;          // 0..7

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride_row_x + kbxd;
        x_df[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + kbxd] = __half2float(bxi->d);
    }
}

// ================================================================
// vec_dot_q4_0_q8_1_dp4a
// Port of turbo mmq.cuh:371-425 (DP4A branch).
// Consumes one Q8_1_mmq block (128 K-elements) worth of y per call.
// k00 ∈ {0, MMQ_TILE_NE_K} selects first/second half of the outer K-iter.
// ================================================================

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int *   __restrict__ x_qs,
    const float * __restrict__ x_df,
    const int *   __restrict__ tile_y,
    float *       __restrict__ sum,
    const int k00) {

    // y_qs and y_ds alias tile_y: qs starts at int offset 4 (after 16 B ds header
    // within each per-col Q8_1_mmq block), ds is at base as half2.
    const int   * y_qs = tile_y + 4;
    const half2 * y_ds = (const half2 *) tile_y;

    // Note: turbo explicitly does NOT unroll k01 (register pressure).
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                // kyqs: byte-aligned offset within the Q8_1_mmq block's qs,
                // interleaved as (k01/2 within 4-int sub) * QI8_1 + local.
                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2))
                               + (k01 / 2) % (QI8_1 / 2);

                // Gather u[0..7] via the vectorized int4 load turbo uses:
                //   vec0 = y_qs[j*MMQ_TILE_Y_K + kyqs + 0..3]
                //   vec1 = y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_0 + 0..3]
                const int4 vec0 = *((const int4 *) &y_qs[j * MMQ_TILE_Y_K + kyqs]);
                const int4 vec1 = *((const int4 *) &y_qs[j * MMQ_TILE_Y_K + kyqs + QI4_0]);
                int u[2 * VDR_Q4_0_Q8_1_MMQ];
                u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
                u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;

                const float d4 =
                    x_df[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + k0 / (QR4_0 * QI4_0)];
                const half2 ds8 = y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1];

                const int * v = &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / QR4_0];
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE] +=
                    q4_0_q8_1_dp4a_4(v, u, d4, ds8);
            }
        }
    }
}

// ================================================================
// mul_mat_q4_0_turbo_impl — template body, K-loop, write-back.
// ================================================================

template <int mmq_x, bool need_check>
static __device__ void mul_mat_q4_0_turbo_impl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,      // packed block_q8_1_mmq, (big_block, col)
    float *      __restrict__ dst,
    const int ncols_x,                 // K dim
    const int nrows_x,                 // M (output rows)
    const int ncols_y,                 // real total_b (for write guard)
    const int stride_col_y,            // total_b_padded (Y stride in cols)
    const int stride_row_x,            // Q4_0 blocks per X row
    const int nrows_dst) {             // dst row stride

    // Shared memory: tile_y first (dynamic, sized by host), then tile_x.
    extern __shared__ int shared_buf[];
    int *   tile_y = shared_buf;
    int *   x_qs   = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);
    float * x_df   = (float *) (x_qs + X_QS_INTS);

    // Output tile coords.
    const int it = blockIdx.x;   // row-tile index
    const int jt = blockIdx.y;   // col-tile index

    // Per-thread accumulator: mmq_x*mmq_y / (nwarps*warp_size) floats.
    constexpr int sum_slots = mmq_x * MMQ_Y / (MMQ_NWARPS * WARP_SIZE);
    float sum[sum_slots];
#pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    // Y base pointer: this tile's starting col. sz = Q8_1_MMQ_INTS.
    // Y stride is `stride_col_y` (total_b_padded), not ncols_y.
    const int * y_base = (const int *) vy + jt * mmq_x * Q8_1_MMQ_INTS;

    // K-loop iterates in units of blocks_per_iter = MMQ_ITER_K / QK4_0 = 8 Q4_0 blocks.
    constexpr int blocks_per_iter = MMQ_ITER_K / QK4_0;  // = 8
    const int kb0_stop = ncols_x / QK4_0;

    const int i_max = nrows_x - it * MMQ_Y - 1;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        // Load X tile (128 rows × 8 Q4_0 blocks = 256 K-elements per row).
        load_tiles_q4_0<need_check>(
            (const char *) vx,
            x_qs, x_df,
            it * MMQ_Y * stride_row_x + kb0,
            (need_check ? i_max : 0),
            stride_row_x);

        // Load Y first half: mmq_x cols × 1 Q8_1_mmq block (128 K-elements).
        // No bounds check — tile_y LDS is GGML_PAD-rounded up (writes hit
        // padded region, harmless), Y buffer is sized n_big_blocks *
        // total_b_padded * 144 B so reads stay in-bounds.
        const int * by0 =
            y_base + (size_t)stride_col_y * (kb0 / 4) * Q8_1_MMQ_INTS;
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();

        vec_dot_q4_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, 0);

        __syncthreads();

        // Load Y second half.
        const int * by1 =
            y_base + (size_t)stride_col_y * ((kb0 / 4) * Q8_1_MMQ_INTS + Q8_1_MMQ_INTS);
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by1[l];
        }
        __syncthreads();

        vec_dot_q4_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    // Write-back (identity col mapping, dst col-major with stride=nrows_dst).
    // dst[col * nrows_dst + row] = accumulator.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col_g = jt * mmq_x + j;
        if (col_g >= ncols_y) return;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row_g = it * MMQ_Y + i;
            if (need_check && row_g >= nrows_x) continue;
            dst[(size_t)col_g * nrows_dst + row_g] =
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

// ================================================================
// load_tiles_q4_1
// Port of turbo mmq.cuh:425-484 (DP4A path).
// Same shape as load_tiles_q4_0 except qs is 4-byte aligned (use
// get_int_b4) and the per-block scale storage is half2 dm not float d.
// ================================================================

template <bool need_check>
static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x,
    int *        __restrict__ x_qs,
    half2 *      __restrict__ x_dm,
    const int kbx0,
    const int i_max,
    const int stride_row_x) {

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1);  // = 32
    constexpr int nrows           = WARP_SIZE / threads_per_row; // = 2
    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i * stride_row_x + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;   // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i * stride_row_x + kbxd;
        x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

// ================================================================
// vec_dot_q4_1_q8_1_dp4a
// Port of turbo mmq.cuh:486-539 (DP4A branch).  Same K-walk as Q4_0 but
// uses x_dm (half2) and the Q4_1 dot formula.
// ================================================================

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int *   __restrict__ x_qs,
    const half2 * __restrict__ x_dm,
    const int *   __restrict__ tile_y,
    float *       __restrict__ sum,
    const int k00) {

    const int   * y_qs = tile_y + 4;
    const half2 * y_ds = (const half2 *) tile_y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1 * VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2))
                               + (k01 / 2) % (QI8_1 / 2);

                const int4 vec0 = *((const int4 *) &y_qs[j * MMQ_TILE_Y_K + kyqs]);
                const int4 vec1 = *((const int4 *) &y_qs[j * MMQ_TILE_Y_K + kyqs + QI4_1]);
                int u[2 * VDR_Q4_0_Q8_1_MMQ];
                u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
                u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;

                const half2 dm4 =
                    x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + k0 / (QR4_1 * QI4_1)];
                const half2 ds8 = y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1];

                const int * v = &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / QR4_1];
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE] +=
                    q4_1_q8_1_dp4a_4(v, u, dm4, ds8);
            }
        }
    }
}

// ================================================================
// mul_mat_q4_1_turbo_impl — Q4_1 main kernel (same K-loop scaffold).
// LDS: tile_y first, then x_qs, then x_dm.  x_dm storage is the same
// number of bytes as Q4_0's x_df because both occupy mmq_y*8 + mmq_y/4
// 4-byte slots (float for Q4_0, half2 for Q4_1).
// ================================================================

template <int mmq_x, bool need_check>
static __device__ void mul_mat_q4_1_turbo_impl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float *      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int stride_col_y,
    const int stride_row_x,
    const int nrows_dst) {

    extern __shared__ int shared_buf[];
    int   * tile_y = shared_buf;
    int   * x_qs   = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);
    half2 * x_dm   = (half2 *) (x_qs + X_QS_INTS);

    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    constexpr int sum_slots = mmq_x * MMQ_Y / (MMQ_NWARPS * WARP_SIZE);
    float sum[sum_slots];
#pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    const int * y_base = (const int *) vy + jt * mmq_x * Q8_1_MMQ_INTS;
    constexpr int blocks_per_iter = MMQ_ITER_K / QK4_1;  // = 8
    const int kb0_stop = ncols_x / QK4_1;
    const int i_max = nrows_x - it * MMQ_Y - 1;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_q4_1<need_check>(
            (const char *) vx,
            x_qs, x_dm,
            it * MMQ_Y * stride_row_x + kb0,
            (need_check ? i_max : 0),
            stride_row_x);

        const int * by0 =
            y_base + (size_t)stride_col_y * (kb0 / 4) * Q8_1_MMQ_INTS;
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();

        vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, 0);

        __syncthreads();

        const int * by1 =
            y_base + (size_t)stride_col_y * ((kb0 / 4) * Q8_1_MMQ_INTS + Q8_1_MMQ_INTS);
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by1[l];
        }
        __syncthreads();

        vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col_g = jt * mmq_x + j;
        if (col_g >= ncols_y) return;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row_g = it * MMQ_Y + i;
            if (need_check && row_g >= nrows_x) continue;
            dst[(size_t)col_g * nrows_dst + row_g] =
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

// ================================================================
// extern "C" entry points: one per (dtype × mmq_x × need_check).
// ================================================================

#define MMQ_TURBO_EXPORT(MMQ_X, CHECK, SUFFIX)                           \
    extern "C" __global__                                                \
    __launch_bounds__(WARP_SIZE * MMQ_NWARPS, 2)                         \
    void mul_mat_q4_0_turbo_x##MMQ_X##_##SUFFIX(                         \
        const void * __restrict__ vx,                                    \
        const void * __restrict__ vy,                                    \
        float      * __restrict__ dst,                                   \
        const int ncols_x, const int nrows_x,                            \
        const int ncols_y, const int stride_col_y,                       \
        const int stride_row_x, const int nrows_dst) {                   \
        mul_mat_q4_0_turbo_impl<MMQ_X, CHECK>(                           \
            vx, vy, dst, ncols_x, nrows_x, ncols_y,                      \
            stride_col_y, stride_row_x, nrows_dst);                      \
    }                                                                     \
    extern "C" __global__                                                \
    __launch_bounds__(WARP_SIZE * MMQ_NWARPS, 2)                         \
    void mul_mat_q4_1_turbo_x##MMQ_X##_##SUFFIX(                         \
        const void * __restrict__ vx,                                    \
        const void * __restrict__ vy,                                    \
        float      * __restrict__ dst,                                   \
        const int ncols_x, const int nrows_x,                            \
        const int ncols_y, const int stride_col_y,                       \
        const int stride_row_x, const int nrows_dst) {                   \
        mul_mat_q4_1_turbo_impl<MMQ_X, CHECK>(                           \
            vx, vy, dst, ncols_x, nrows_x, ncols_y,                      \
            stride_col_y, stride_row_x, nrows_dst);                      \
    }

MMQ_TURBO_EXPORT( 8, false, unchecked)
MMQ_TURBO_EXPORT( 8, true,  checked)
MMQ_TURBO_EXPORT(16, false, unchecked)
MMQ_TURBO_EXPORT(16, true,  checked)
MMQ_TURBO_EXPORT(32, false, unchecked)
MMQ_TURBO_EXPORT(32, true,  checked)
MMQ_TURBO_EXPORT(64, false, unchecked)
MMQ_TURBO_EXPORT(64, true,  checked)

#undef MMQ_TURBO_EXPORT
