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

// Q8_0: 32 int8 qs + half d, no -8 offset (already signed).
#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))   // 8 ints per Q8_0 block
typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 2 + QK8_0, "block_q8_0 size");

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
// Q8_0 uses a smaller M-axis tile (64 rows/block) because x_qs is 2× larger
// per row than Q4_0 (8 ints/block vs 4 after nibble-pack), so MMQ_Y=128 would
// push total LDS to 46 720 B/block — 2 blocks/CU would overflow gfx906's
// 64 KB LDS and silently drop to 1 block/CU (half occupancy). With
// MMQ_Y_Q8=64, x_qs shrinks to 16 640 B and 2 blocks/CU fits in 55 936 B.
#define MMQ_Y_Q8 64
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
#define VDR_Q8_0_Q8_1_MMQ 8

// LDS layout sizes (DP4A path, per turbo MMQ_DP4A_TXS_Q4_0):
//   x_qs: mmq_y*(MMQ_TILE_NE_K + 1)                 ints   (33 per row, +1 bank pad)
//   x_df: mmq_y*(MMQ_TILE_NE_K/QI4_0) + mmq_y/QI4_0 floats (8 + 32)
//   y_qs: mmq_x * MMQ_TILE_Y_K                      ints   (36 per col)
#define X_QS_INTS (MMQ_Y * (MMQ_TILE_NE_K + 1))
#define X_DF_FLTS (MMQ_Y * (MMQ_TILE_NE_K / QI4_0) + MMQ_Y / QI4_0)

// Q8_0 uses MMQ_Y_Q8=64 (see note above).  At 64-row tile:
//   x_qs = 64 * 65     = 4160 ints (16 640 B)
//   x_df = 64 * 8 + 16 =  528 flts ( 2 112 B)
#define X_QS_INTS_Q8_0 (MMQ_Y_Q8 * (2 * MMQ_TILE_NE_K + 1))
#define X_DF_FLTS_Q8_0 (MMQ_Y_Q8 * (2 * MMQ_TILE_NE_K / QI8_0) + MMQ_Y_Q8 / (QI8_0 / 2))

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

// Packed 4×int8 saturated subtract. NVIDIA intrinsic, not available on HIP.
// Used by Q6_K dequant: combines 4-bit-low | 2-bit-high then subtracts 32.
static __device__ __forceinline__ int vsubss4_moe(int a, int b) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int va = (int8_t)(a >> (i * 8));
        int vb = (int8_t)(b >> (i * 8));
        int diff = va - vb;
        diff = max(diff, -128);
        diff = min(diff,  127);
        result |= (((uint32_t)(uint8_t)(int8_t)diff) << (i * 8));
    }
    return result;
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

// Port of turbo vecdotq.cuh:266-278 (vec_dot_q8_0_q8_1_impl<float, vdr=8>).
// Q8_0 has no nibble split: straight 8 dp4a calls on 8 ints of v and u,
// scaled by d_x * d_y.
static __device__ __forceinline__ float q8_0_q8_1_dp4a_8(
    const int * v,          // 8 ints of Q8_0 qs
    const int * u,          // 8 ints of Q8_1 qs
    const float d_x,        // Q8_0 delta
    const float d_y) {      // Q8_1 delta
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumi = dp4a_sdot4(v[i], u[i], sumi);
    }
    return d_x * d_y * (float) sumi;
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

    // (Shared tail with the gather variant below.)
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
    ((int8_t *) (y_bytes + 16))[tid] = q;
    if (lane_in_sub == 0) {
        half2 * ds = (half2 *) y_bytes;
        ds[sub] = __floats2half2_rn(d, ssum);
    }
}

// ================================================================
// MoE gather-quantize: same DS4 Q8_1 output, but the `c` grid-dim
// indexes a BUCKET POSITION (sorted-by-expert) rather than a token.
// Each thread reads the input row through `ids_src1[c]` so the output
// tensor comes out **physically sorted**: all tokens for expert 0
// first, then expert 1, etc. — exactly what llama.cpp does
// (quantize.cu:195 `i01 = ids ? ids[i1] : i1`).  The MMQ-MoE kernel
// can then read Y stride-1 with no indirection in the hot K-loop.
// ================================================================

extern "C" __global__ void quantize_q8_1_mmq_gather(
    const float * __restrict__ x,        // [n_tokens_src, ncols] row-major
    const int   * __restrict__ ids_src1, // [total_pos] bucket_pos -> source token
    void *        __restrict__ vy,       // [n_big_blocks * total_pos] blocks of 144 B
    const int ncols,                     // K dim
    const int total_pos) {               // bucket count (= n_tokens * topk)

    const int b   = blockIdx.x;
    const int c   = blockIdx.y;          // bucket position (0..total_pos)
    const int tid = threadIdx.x;
    const int sub = tid / 32;
    const int lane_in_sub = tid % 32;

    const int src_row = ids_src1[c];     // gather source token for this bucket pos
    const int ki = b * 128 + tid;
    const float xi = (ki < ncols) ? x[src_row * ncols + ki] : 0.0f;

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
                        + ((size_t)(b * total_pos + c)) * Q8_1_MMQ_BYTES;
    ((int8_t *) (y_bytes + 16))[tid] = q;
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
    const int kb_remaining,           // # of valid blocks in this K-iter (1..8)
    const int stride_row_x) {         // Q4_0 blocks per X row

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);  // = 32
    constexpr int nrows           = WARP_SIZE / threads_per_row; // = 2
    const int txi  = threadIdx.x % threads_per_row;            // 0..31
    const int kbx  = txi / QI4_0;                              // 0..7 (block within K-iter)
    const int kqsx = txi % QI4_0;                              // 0..3 (int within block)

    // K-OOB handling: for kbx >= kb_remaining, clamp the read to the last
    // valid block. The qs values land in tile_x positions that the next
    // load (x_df) zeros out via dm/d=0 for the same gk, so the contribution
    // to vec_dot is 0 regardless of the (possibly-duplicated) qs.
    const int kbx_safe = (kbx < kb_remaining) ? kbx : (kb_remaining - 1);

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride_row_x + kbx_safe;
        const int qs0 = get_int_b2(bxi->qs, kqsx);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;   // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;          // 0..7
    const bool kbxd_valid = (kbxd < kb_remaining);
    const int kbxd_safe = kbxd_valid ? kbxd : 0;

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride_row_x + kbxd_safe;
        const float d = kbxd_valid ? __half2float(bxi->d) : 0.0f;
        x_df[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + kbxd] = d;
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
    const int n_blocks_x = ncols_x / QK4_0;              // real blocks per row
    // Round kb0 stop up so we cover any partial tail.  load_tiles handles
    // OOB blocks via kb_remaining; vec_dot's contribution from OOB is 0
    // because d/d_y is zeroed for OOB.
    const int kb0_stop = (n_blocks_x + blocks_per_iter - 1)
                         / blocks_per_iter * blocks_per_iter;

    const int i_max = nrows_x - it * MMQ_Y - 1;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0)
                                                 : blocks_per_iter;
        // Load X tile (128 rows × 8 Q4_0 blocks = 256 K-elements per row).
        load_tiles_q4_0<need_check>(
            (const char *) vx,
            x_qs, x_df,
            it * MMQ_Y * stride_row_x + kb0,
            (need_check ? i_max : 0),
            kb_remaining,
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
    const int kb_remaining,           // # of valid blocks in this K-iter (1..8)
    const int stride_row_x) {

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1);  // = 32
    constexpr int nrows           = WARP_SIZE / threads_per_row; // = 2
    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;

    const int kbx_safe = (kbx < kb_remaining) ? kbx : (kb_remaining - 1);

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i * stride_row_x + kbx_safe;
        const int qs0 = get_int_b4(bxi->qs, kqsx);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;   // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    const bool kbxd_valid = (kbxd < kb_remaining);
    const int kbxd_safe = kbxd_valid ? kbxd : 0;

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i * stride_row_x + kbxd_safe;
        const half2 dm = kbxd_valid ? bxi->dm : __floats2half2_rn(0.0f, 0.0f);
        x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + kbxd] = dm;
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
    const int n_blocks_x = ncols_x / QK4_1;
    const int kb0_stop = (n_blocks_x + blocks_per_iter - 1)
                         / blocks_per_iter * blocks_per_iter;
    const int i_max = nrows_x - it * MMQ_Y - 1;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0)
                                                 : blocks_per_iter;
        load_tiles_q4_1<need_check>(
            (const char *) vx,
            x_qs, x_dm,
            it * MMQ_Y * stride_row_x + kb0,
            (need_check ? i_max : 0),
            kb_remaining,
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
// load_tiles_q8_0
// Port of turbo mmq.cuh:695-776 (DP4A path; software-pipeline stripped).
// Each thread loads 2 Q8_0 blocks (kbx and kbx+4) into LDS because
// each Q8_0 block is 8 ints (vs Q4_0's 4 after nibble packing).
// ================================================================

template <bool need_check>
static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x,
    int *        __restrict__ x_qs,
    float *      __restrict__ x_df,
    const int kbx0,
    const int i_max,
    const int kb_remaining,         // 1..8 valid Q8_0 blocks this K-iter
    const int stride_row_x) {

    constexpr int threads_per_row = 32;
    constexpr int nrows           = WARP_SIZE / threads_per_row;  // = 2
    const int txi  = threadIdx.x % threads_per_row;               // 0..31
    const int kbx  = txi / QI8_0;                                 // 0..3 (lo block)
    const int kqsx = txi % QI8_0;                                 // 0..7 (int within block)

    // Each thread covers 2 blocks per outer K-iter: kbx (lo half) and
    // kbx+4 (hi half).  K-OOB: clamp the read to the last valid block
    // if either is past kb_remaining; zero-out their scales in the d-loop.
    const int kbx_lo_safe = (kbx < kb_remaining) ? kbx : (kb_remaining - 1);
    const int kbx_hi      = kbx + MMQ_TILE_NE_K / QI8_0;          // kbx + 4
    const int kbx_hi_safe = (kbx_hi < kb_remaining) ? kbx_hi : (kb_remaining - 1);

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y_Q8; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q8_0 * bxi_lo = (const block_q8_0 *) x + kbx0 + i * stride_row_x + kbx_lo_safe;
        const block_q8_0 * bxi_hi = (const block_q8_0 *) x + kbx0 + i * stride_row_x + kbx_hi_safe;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + 0             + txi] = get_int_b2(bxi_lo->qs, kqsx);
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi_hi->qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = 2 * MMQ_TILE_NE_K / QI8_0;     // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;                // 0..7
    const bool kbxd_valid = (kbxd < kb_remaining);
    const int kbxd_safe = kbxd_valid ? kbxd : 0;

#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y_Q8; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i * stride_row_x + kbxd_safe;
        const float d = kbxd_valid ? __half2float(bxi->d) : 0.0f;
        x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + kbxd] = d;
    }
}

// ================================================================
// vec_dot_q8_0_q8_1_dp4a
// Port of turbo mmq.cuh:982-1012 (DP4A branch).
// Reads d_y from the half2 low-half because our quantize writes the
// DS4 layout (half2(d, sum)) — Q8_0 only needs d, so we __low2float it.
// ================================================================

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int *   __restrict__ x_qs,
    const float * __restrict__ x_df,
    const int *   __restrict__ tile_y,
    float *       __restrict__ sum,
    const int k00) {

    const int   * y_qs = tile_y + 4;
    const half2 * y_ds = (const half2 *) tile_y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < MMQ_Y_Q8; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int * v = &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0];
                const int * u = &y_qs[j * MMQ_TILE_Y_K + (k0 % MMQ_TILE_NE_K)];

                const float d_x =
                    x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + k0 / QI8_0];
                const int sub = (k0 / QI8_1) % (MMQ_TILE_NE_K / QI8_1);
                const float d_y = __low2float(y_ds[j * MMQ_TILE_Y_K + sub]);

                sum[(j0 / MMQ_NWARPS) * (MMQ_Y_Q8 / WARP_SIZE) + i0 / WARP_SIZE] +=
                    q8_0_q8_1_dp4a_8(v, u, d_x, d_y);
            }
        }
    }
}

// ================================================================
// mul_mat_q8_0_turbo_impl
// ================================================================

template <int mmq_x, bool need_check>
static __device__ void mul_mat_q8_0_turbo_impl(
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
    int *   tile_y = shared_buf;
    int *   x_qs   = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);
    float * x_df   = (float *) (x_qs + X_QS_INTS_Q8_0);

    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    constexpr int sum_slots = mmq_x * MMQ_Y_Q8 / (MMQ_NWARPS * WARP_SIZE);
    float sum[sum_slots];
#pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    const int * y_base = (const int *) vy + jt * mmq_x * Q8_1_MMQ_INTS;
    constexpr int blocks_per_iter = MMQ_ITER_K / QK8_0;  // = 8
    const int n_blocks_x = ncols_x / QK8_0;
    const int kb0_stop = (n_blocks_x + blocks_per_iter - 1)
                         / blocks_per_iter * blocks_per_iter;
    const int i_max = nrows_x - it * MMQ_Y_Q8 - 1;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0)
                                                 : blocks_per_iter;
        load_tiles_q8_0<need_check>(
            (const char *) vx,
            x_qs, x_df,
            it * MMQ_Y_Q8 * stride_row_x + kb0,
            (need_check ? i_max : 0),
            kb_remaining,
            stride_row_x);

        const int * by0 =
            y_base + (size_t)stride_col_y * (kb0 / 4) * Q8_1_MMQ_INTS;
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();

        vec_dot_q8_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, 0);

        __syncthreads();

        const int * by1 =
            y_base + (size_t)stride_col_y * ((kb0 / 4) * Q8_1_MMQ_INTS + Q8_1_MMQ_INTS);
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by1[l];
        }
        __syncthreads();

        vec_dot_q8_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col_g = jt * mmq_x + j;
        if (col_g >= ncols_y) return;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y_Q8; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row_g = it * MMQ_Y_Q8 + i;
            if (need_check && row_g >= nrows_x) continue;
            dst[(size_t)col_g * nrows_dst + row_g] =
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y_Q8 / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

// ================================================================
// MoE bucket prep (M6a): three tiny helpers that turn `topk_ids`
// (shape [n_tokens, topk], each entry ∈ [0, n_experts)) into:
//   counts        [n_experts]       # tokens routed to each expert
//   expert_bounds [n_experts + 1]   exclusive prefix sum of counts
//   ids_src1      [n_tokens*topk]   bucket_pos  -> source token idx
//   ids_dst       [n_tokens*topk]   bucket_pos  -> flat dst idx
//                                   = token*topk + slot
//
// `bucket_pos` walks experts in order: for expert e, positions
// [expert_bounds[e], expert_bounds[e+1]) hold the bucket of tokens
// routed to e (in whatever atomic-order the scatter produces —
// shape-equivalent to llama.cpp's mmid.cu).
// ================================================================

// Kernel 1: atomicAdd count per expert.  Grid (ceil(total/256), 1, 1).
extern "C" __global__ void moe_bucket_count(
    const unsigned int * __restrict__ topk_ids,
    const int total_pos,
    int * __restrict__ counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pos) return;
    atomicAdd(&counts[topk_ids[tid]], 1);
}

// Kernel 2: single-block Hillis-Steele scan (n_experts ≤ 512 for gemma4).
// Input  : counts[n_experts]
// Output : expert_bounds[n_experts + 1] = exclusive prefix sum (bounds[0]=0,
//          bounds[i+1] = counts[0]+..+counts[i]).
// Block  : 512 threads.
extern "C" __global__ void moe_bucket_prefix(
    const int * __restrict__ counts,
    const int n_experts,
    int * __restrict__ expert_bounds) {
    __shared__ int tmp[512];
    const int tid = threadIdx.x;
    tmp[tid] = (tid < n_experts) ? counts[tid] : 0;
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        int v = (tid >= off) ? tmp[tid - off] : 0;
        __syncthreads();
        tmp[tid] += v;
        __syncthreads();
    }
    if (tid == 0) expert_bounds[0] = 0;
    if (tid < n_experts) expert_bounds[tid + 1] = tmp[tid];
}

// Kernel 3: scatter bucket positions.  `cursor` must be pre-zeroed.
// For each topk entry (token, slot) assigned to expert e:
//   bp = expert_bounds[e] + atomicAdd(&cursor[e], 1)
//   ids_src1[bp] = source row (see below)
//   ids_dst [bp] = token * topk + slot
//
// `input_dim1` selects the source addressing mode:
//   input_dim1 == 1       : input is [tokens, 1, k] shared across topk
//                           (gate/up pre-expansion). src_row = token.
//   input_dim1 == topk    : input is [tokens, topk, k] per-slot
//                           (down, where silu_mul(gate,up) already expanded).
//                           src_row = token * topk + slot = tid.
extern "C" __global__ void moe_bucket_scatter(
    const unsigned int * __restrict__ topk_ids,
    const int total_pos,       // = n_tokens * topk
    const int topk,
    const int input_dim1,
    const int * __restrict__ expert_bounds,
    int * __restrict__ cursor,
    int * __restrict__ ids_src1,
    int * __restrict__ ids_dst) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pos) return;
    const int token = tid / topk;
    const int slot  = tid % topk;
    const int e     = (int) topk_ids[tid];
    const int bp    = expert_bounds[e] + atomicAdd(&cursor[e], 1);
    ids_src1[bp] = (input_dim1 == 1) ? token : (token * input_dim1 + slot);
    ids_dst [bp] = token * topk + slot;
}

// ================================================================
// mul_mat_q_turbo_moe_impl — generic MoE MMQ kernel template (M7).
// One template parameterised on dtype tag DT ∈ {MOE_Q4_0, MOE_Q4_1, MOE_Q8_0}.
// Grid   : (ceil(nrows_x / MMQ_Y), ntx_max, n_experts)
// Block  : (WARP_SIZE, MMQ_NWARPS, 1) = (64, 4, 1)
//
// Design mirrors llama.cpp's mul_mat_q (mmq.cuh:3594-3647 non-CDNA path,
// 3608-3631 MoE branch):
//   - Y is physically sorted by expert by the gather-quantize step
//     (quantize_q8_1_mmq_gather above), laid out `[n_big_blocks, total_pos]`
//     with bucket_pos walking experts in prefix-sum order.
//   - Per block: read expert_bounds[zt] / [zt+1] to find this expert's
//     bucket range, skip if empty or tile past bucket, pre-load the
//     ids_dst indices for this tile into LDS.
//   - Hot K-loop reads Y stride-1 from `vy + (col_low + jt*mmq_x) * sz`.
//   - Write-back scatters via ids_dst_shared[j] to `[tokens, topk, n]`.
//
// NOTE on the Y-load bound guard: `mmq_x * MMQ_TILE_Y_K` (e.g. 8*36=288)
// is not always a multiple of nwarps*warp_size (e.g. 4*64=256), so the
// cooperative load has a tail iteration where `l >= mmq_x*MMQ_TILE_Y_K`.
// Per-thread bound check skips the overshoot so we don't walk past
// total_pos into foreign memory.
//
// Per-dtype specialisation (MMQ_Y, x-tile LDS size, load_tiles, vec_dot,
// x_scales type) is picked by `if constexpr` — the compiler produces one
// specialised kernel per (DT, mmq_x, need_check) triple from a single body.
// ================================================================

// Q4_K support structures. Copied from quantized.cu (separate TU).
#define QK_K_Q4K_MOE  256
#define QR4_K_Q4K_MOE 2
#define QI4_K_Q4K_MOE (QK_K_Q4K_MOE / (4 * QR4_K_Q4K_MOE))   // = 32
#define VDR_Q4_K_Q8_1_MMQ_MOE 8

typedef struct {
    half2   dm;
    uint8_t scales[12];               // 3 * 256 / 64 = 12
    uint8_t qs[QK_K_Q4K_MOE / 2];     // 128 bytes (256 × 4-bit)
} block_q4_K_moe;
static_assert(sizeof(block_q4_K_moe) == 4 + 12 + 128,
              "block_q4_K_moe size mismatch (expected 144)");

// Port of turbo mmq.cuh:2127 — unpack a 4-int Q4_K scales blob to a packed int
// holding either 4 sub-scales or 4 sub-mins, depending on ksc (0,1 → sc; 2,3 → m).
static __device__ __forceinline__ int unpack_scales_q45_K_moe(
    const int * scales, const int ksc) {
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F)
         | ((scales[ksc / 2]                >> (2 * (ksc % 2)))         & 0x30303030);
}

// Port of candle quantized.cu:2024 — scalar vec_dot_q4_K_q8_1_impl_mmq.
// Reused by the MoE MMQ inner loop for one 32-elem K tile (covering 2 Q8_1
// sub-blocks via the i ∈ {0,1} outer, which select nibble-half of v).
static __device__ __forceinline__ float q4_K_q8_1_dp4a_impl(
    const int   * __restrict__ v,     // 8 ints of X (nibble-packed)
    const int   * __restrict__ u,     // 8 ints × 2 Q8_1 sub-blocks worth of Y
    const uint8_t * __restrict__ sc,  // 2 sc values (one per sub)
    const uint8_t * __restrict__ m,   // 2 m values
    const half2 & dm4,                // per-row dm (dscale, dmin)
    const half2 * __restrict__ ds8) { // 2 half2 ds values (one per Q8_1 sub)

    // Uses the d8 * Σq8 bias form (matches candle's vec_dot_q4_K_q8_1_impl_vmmq
    // and the B2 single-warp kernel), NOT `ds8.y * m`. Q8_1_MMQ stores
    // ds.y = Σ x_real while MMVQ reference uses d8 * Σq8 for the bias —
    // these diverge by ~1% quantization error. Reproducing the approximate
    // form keeps the template path bit-exact against the MMVQ reference.
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int sumi_d = 0;
        int sumi_y = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int u_j = u[i * 8 + j];
            sumi_d = dp4a_sdot4((v[j] >> (4 * i)) & 0x0F0F0F0F, u_j, sumi_d);
            sumi_y = dp4a_sdot4(0x01010101, u_j, sumi_y);
        }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.x * ( m[i] * sumi_y);
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

enum MoeDType : int {
    MOE_Q4_0 = 0,
    MOE_Q4_1 = 1,
    MOE_Q8_0 = 2,
    MOE_Q4_K = 3,  // C3-redo 2026-04-19: integrated into template behind
                   // CANDLE_MMQ_Q4K_TEMPLATE env flag. B2 single-warp
                   // kernel remains the default fallback until the
                   // template path passes full-model soak.
};

template <MoeDType DT> struct moe_dtype_traits;
template <> struct moe_dtype_traits<MOE_Q4_0> {
    static constexpr int  mmq_y        = MMQ_Y;
    static constexpr int  x_qs_ints    = MMQ_Y * (MMQ_TILE_NE_K + 1);
    static constexpr int  block_bytes  = sizeof(block_q4_0);    // 18
    static constexpr bool half2_scales = false;                 // uses float x_df
    static constexpr bool is_kquant    = false;
    static constexpr int  x_sc_ints    = 0;  // unused for flat-block dtypes
    static constexpr int  qk           = QK8_0;                 // 32 elems/block
    // Per-superblock (=256 elems) K-iter; here each "block" is 32 elems.
    static constexpr int  elems_per_outer_iter = MMQ_ITER_K;    // 256
};
template <> struct moe_dtype_traits<MOE_Q4_1> {
    static constexpr int  mmq_y        = MMQ_Y;
    static constexpr int  x_qs_ints    = MMQ_Y * (MMQ_TILE_NE_K + 1);
    static constexpr int  block_bytes  = sizeof(block_q4_1);    // 20
    static constexpr bool half2_scales = true;                  // uses half2 x_dm
    static constexpr bool is_kquant    = false;
    static constexpr int  x_sc_ints    = 0;
    static constexpr int  qk           = QK8_0;                 // 32
    static constexpr int  elems_per_outer_iter = MMQ_ITER_K;
};
template <> struct moe_dtype_traits<MOE_Q8_0> {
    static constexpr int  mmq_y        = MMQ_Y_Q8;              // 64, not 128
    static constexpr int  x_qs_ints    = MMQ_Y_Q8 * (2 * MMQ_TILE_NE_K + 1);
    static constexpr int  block_bytes  = sizeof(block_q8_0);    // 34
    static constexpr bool half2_scales = false;
    static constexpr bool is_kquant    = false;
    static constexpr int  x_sc_ints    = 0;
    static constexpr int  qk           = QK8_0;                 // 32
    static constexpr int  elems_per_outer_iter = MMQ_ITER_K;
};
// C3-redo: Q4_K plugged into the generic template. Native block is a
// 256-elem superblock, so blocks_per_iter = elems_per_outer_iter / qk = 1.
// Needs dm (half2 per row), sc tile (4 ints per row for 8 sub-scales/mins),
// in addition to the standard 32-int qs tile.
template <> struct moe_dtype_traits<MOE_Q4_K> {
    static constexpr int  mmq_y        = MMQ_Y_Q8;              // 64
    static constexpr int  x_qs_ints    = MMQ_Y_Q8 * (MMQ_TILE_NE_K + 1);
    static constexpr int  block_bytes  = sizeof(block_q4_K_moe); // 144
    static constexpr bool half2_scales = true;
    static constexpr bool is_kquant    = true;
    static constexpr int  x_sc_ints    = MMQ_Y_Q8 * (MMQ_TILE_NE_K / 8) + MMQ_Y_Q8 / 8; // 264
    static constexpr int  qk           = QK_K_Q4K_MOE;          // 256
    static constexpr int  elems_per_outer_iter = MMQ_ITER_K;    // 256 = 1 superblock
};

// ================================================================
// load_tiles_q4_K — port of turbo mmq.cuh:2137 (DP4A branch).
// Loads one Q4_K superblock (256 elems = 32 ints × 64 rows of qs,
// 64 half2 of dm, 8 (sc,m) pairs × 64 rows of scales) into LDS.
// ================================================================

template <bool need_check>
static __device__ __forceinline__ void load_tiles_q4_K_moe(
    const char * __restrict__ x,       // raw Q4_K byte pointer (expert slab)
    int *        __restrict__ x_qs,    // MMQ_Y_Q8 * (MMQ_TILE_NE_K+1) ints
    half2 *      __restrict__ x_dm,    // MMQ_Y_Q8 half2
    int *        __restrict__ x_sc,    // MMQ_Y_Q8 * (MMQ_TILE_NE_K/8) + MMQ_Y_Q8/8 ints
    const int kbx0,                    // starting Q4_K block index for this K-iter
    const int i_max,                   // row upper bound (for need_check)
    const int stride_row_x) {          // Q4_K blocks per X row = k/256

    constexpr int mmq_y = MMQ_Y_Q8;

    // qs load: 32 ints per row. threads_per_row=32, so 2 rows per warp.
    constexpr int threads_per_row = 32;                 // = MMQ_TILE_NE_K
    constexpr int nrows           = WARP_SIZE / threads_per_row;  // 2
    const int txi = threadIdx.x % threads_per_row;

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) i = min(i, i_max);
        const block_q4_K_moe * bxi =
            (const block_q4_K_moe *) x + kbx0 + i * stride_row_x;
        const int * qs_src = (const int *) bxi->qs;
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs_src[txi];
    }

    // dm load: 1 half2 per row. nwarps*warp_size threads handle mmq_y rows.
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += MMQ_NWARPS * WARP_SIZE) {
        int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % mmq_y;
        if (need_check) i = min(i, i_max);
        const block_q4_K_moe * bxi =
            (const block_q4_K_moe *) x + kbx0 + i * stride_row_x;
        x_dm[i] = bxi->dm;
    }

    // Scales load: one int-sized scales8 per (row, ksc ∈ 0..3). ksc=0,1 give
    // sc values; ksc=2,3 give m values. Packed as 4 bytes each.
    constexpr int ksc_per_row   = MMQ_TILE_NE_K / 8;   // 4
    constexpr int rows_per_warp = WARP_SIZE / ksc_per_row;  // 16
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / ksc_per_row) % mmq_y;
        if (need_check) i = min(i, i_max);
        const block_q4_K_moe * bxi =
            (const block_q4_K_moe *) x + kbx0 + i * stride_row_x;
        const int ksc = threadIdx.x % ksc_per_row;       // 0..3
        const int * scales = (const int *) bxi->scales;
        const int scales8  = unpack_scales_q45_K_moe(scales, ksc);
        x_sc[i * ksc_per_row + i / 8 + ksc] = scales8;
    }
}

// ================================================================
// vec_dot_q4_K_q8_1_dp4a_moe — port of turbo mmq.cuh:2247.
// Processes MMQ_TILE_NE_K=32 X-elements per call (= 2 Q4_K sub-blocks
// worth). Called twice per outer K-iter (k00=0, k00=MMQ_TILE_NE_K) to
// cover the full 256-elem superblock × 2 halves = 8 sub-blocks = 256 ✗
// wait — 2 calls × 32 elems = 64 elems. Need 4 calls? No:
// k00=0 covers k0 ∈ {0,16}: impl-call reads x_qs[k0/2]=x_qs[0] and x_qs[8],
// each impl covers 32 K-elems × 2 (low+high nibbles of the 32 ints loaded).
// Actually v[j] = x_qs[k0/2 + j] and impl processes v[0..8] twice
// (nibble-low + nibble-high) = 64 K-elems per impl-call. 2 impl-calls per
// vec_dot = 128. 2 vec_dot calls per outer = 256. ✓
// ================================================================

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a_moe(
    const int   * __restrict__ x_qs,
    const half2 * __restrict__ x_dm,
    const int   * __restrict__ x_sc,
    const int   * __restrict__ tile_y,
    float       * __restrict__ sum,
    const int k00) {

    constexpr int mmq_y = MMQ_Y_Q8;
    const int   * y_qs = tile_y + 4;              // 4 ints of ds header first
    const half2 * y_ds = (const half2 *) tile_y;

    // k01 steps by QR4_K*VDR = 2*8 = 16. 32/16 = 2 iterations.
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K_Q4K_MOE * VDR_Q4_K_Q8_1_MMQ_MOE) {
        const int k0 = k00 + k01;

        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
            const int j = j0 + threadIdx.y;

            #pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                // sc/m pointers: base + 2*(k01/16) selects which sub-block
                // pair (0-1 or 2-3) of the currently-active 32-elem k chunk.
                const uint8_t * sc =
                    (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32]
                    + 2 * (k01 / 16);

                const int   * v  = &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2];
                const int   * u  = &y_qs[j * MMQ_TILE_Y_K + k01];
                const half2 * ds = &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1];

                sum[(j0 / MMQ_NWARPS) * (mmq_y / WARP_SIZE) + i0 / WARP_SIZE] +=
                    q4_K_q8_1_dp4a_impl(v, u, sc, sc + 8, x_dm[i], ds);
            }
        }
    }
}

template <MoeDType DT, int mmq_x, bool need_check>
static __device__ void mul_mat_q_turbo_moe_impl(
    const void  * __restrict__ vx,
    const void  * __restrict__ vy,
    const int   * __restrict__ ids_dst,
    const int   * __restrict__ expert_bounds,
    float       * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,                    // = n (rows per expert slab)
    const int stride_col_y,               // = total_pos (big-block stride in Y)
    const int stride_row_x,               // = blocks per X row = k/32
    const int stride_expert_blocks,       // = nrows_x * stride_row_x
    const int nrows_dst) {                // = n

    using Tr = moe_dtype_traits<DT>;
    constexpr int MMQ_Y_R   = Tr::mmq_y;
    constexpr int X_QS_T    = Tr::x_qs_ints;
    // x_dm/x_df alias the same LDS bytes. Q4_K needs less here (64 half2 for 64
    // rows) than the Q4_0/Q4_1 case (MMQ_Y * 8 floats), but the reservation
    // below is sized for the larger use so all specializations fit.
    constexpr int X_DM_RESERVE = MMQ_Y_R * 8 + MMQ_Y_R / 4;

    extern __shared__ int shared_buf[];
    int   * tile_y = shared_buf;
    int   * x_qs   = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);
    float * x_df   = (float *) (x_qs + X_QS_T);
    half2 * x_dm   = (half2 *) (x_qs + X_QS_T);
    // x_sc follows the dm reserve. Only used for K-quants (Tr::is_kquant).
    // For non-K-quant instantiations this pointer is never dereferenced.
    int   * x_sc   = (int *)(x_qs + X_QS_T) + X_DM_RESERVE;

    const int expert   = blockIdx.z;
    const int col_low  = expert_bounds[expert];
    const int col_high = expert_bounds[expert + 1];
    const int col_diff = col_high - col_low;

    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    if (col_diff <= 0)          return;
    if (jt * mmq_x >= col_diff) return;

    __shared__ int ids_dst_shared[64];
    {
        const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
        if (tid < mmq_x) {
            const int local_j = jt * mmq_x + tid;
            ids_dst_shared[tid] =
                (local_j < col_diff) ? ids_dst[col_low + local_j] : -1;
        }
    }
    __syncthreads();

    // Weight base for this expert's row slab.
    const char * vx_expert = (const char *) vx
        + (size_t) expert * stride_expert_blocks * Tr::block_bytes;

    const int * y_base = (const int *) vy + (col_low + jt * mmq_x) * Q8_1_MMQ_INTS;

    constexpr int sum_slots = mmq_x * MMQ_Y_R / (MMQ_NWARPS * WARP_SIZE);
    float sum[sum_slots];
#pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    // Per-outer-iter K-elements is uniform (MMQ_ITER_K=256). What varies by
    // dtype is the number of NATIVE blocks that covers that: 8 mini-blocks
    // for Q4_0/Q4_1/Q8_0, 1 superblock for Q4_K. stride_row_x is expressed
    // in the native unit.
    constexpr int blocks_per_iter = Tr::elems_per_outer_iter / Tr::qk;
    const int n_blocks_x = ncols_x / Tr::qk;
    const int n_outer    = (n_blocks_x + blocks_per_iter - 1) / blocks_per_iter;
    const int i_max = nrows_x - it * MMQ_Y_R - 1;

    constexpr int TILE_Y_INTS =
        GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);

    for (int outer = 0; outer < n_outer; ++outer) {
        const int kb0 = outer * blocks_per_iter;
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0)
                                                 : blocks_per_iter;
        const int kbx0 = it * MMQ_Y_R * stride_row_x + kb0;
        const int i_max_arg = need_check ? i_max : 0;

        if constexpr (DT == MOE_Q4_0) {
            load_tiles_q4_0<need_check>(
                vx_expert, x_qs, x_df, kbx0, i_max_arg, kb_remaining, stride_row_x);
        } else if constexpr (DT == MOE_Q4_1) {
            load_tiles_q4_1<need_check>(
                vx_expert, x_qs, x_dm, kbx0, i_max_arg, kb_remaining, stride_row_x);
        } else if constexpr (DT == MOE_Q4_K) {
            load_tiles_q4_K_moe<need_check>(
                vx_expert, x_qs, x_dm, x_sc, kbx0, i_max_arg, stride_row_x);
        } else {
            load_tiles_q8_0<need_check>(
                vx_expert, x_qs, x_df, kbx0, i_max_arg, kb_remaining, stride_row_x);
        }

        // One outer iter covers 2 Q8_1 big_blocks (2 × 128 = 256 K-elems).
        // For Q4_0/Q4_1/Q8_0 this is kb0/4..kb0/4+1; for Q4_K it's 2*kb0,
        // 2*kb0+1. Both reduce to `outer * 2` uniformly.
        const int y_big_base = outer * 2;

        const int * by0 =
            y_base + (size_t) stride_col_y * y_big_base * Q8_1_MMQ_INTS;
#pragma unroll
        for (int l0 = 0; l0 < TILE_Y_INTS; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if (l < mmq_x * MMQ_TILE_Y_K) {
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        if constexpr (DT == MOE_Q4_0) {
            vec_dot_q4_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, 0);
        } else if constexpr (DT == MOE_Q4_1) {
            vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, 0);
        } else if constexpr (DT == MOE_Q4_K) {
            vec_dot_q4_K_q8_1_dp4a_moe<mmq_x>(x_qs, x_dm, x_sc, tile_y, sum, 0);
        } else {
            vec_dot_q8_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, 0);
        }
        __syncthreads();

        const int * by1 =
            y_base + (size_t) stride_col_y * (y_big_base + 1) * Q8_1_MMQ_INTS;
#pragma unroll
        for (int l0 = 0; l0 < TILE_Y_INTS; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if (l < mmq_x * MMQ_TILE_Y_K) {
                tile_y[l] = by1[l];
            }
        }
        __syncthreads();
        if constexpr (DT == MOE_Q4_0) {
            vec_dot_q4_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, MMQ_TILE_NE_K);
        } else if constexpr (DT == MOE_Q4_1) {
            vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, MMQ_TILE_NE_K);
        } else if constexpr (DT == MOE_Q4_K) {
            vec_dot_q4_K_q8_1_dp4a_moe<mmq_x>(x_qs, x_dm, x_sc, tile_y, sum, MMQ_TILE_NE_K);
        } else {
            vec_dot_q8_0_q8_1_dp4a<mmq_x>(x_qs, x_df, tile_y, sum, MMQ_TILE_NE_K);
        }
        __syncthreads();
    }

    // Scatter write-back: dst[ ids_dst_shared[j] * nrows_dst + row_g ].
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j       = j0 + threadIdx.y;
        const int local_j = jt * mmq_x + j;
        if (local_j >= col_diff) continue;
        const int dst_base = ids_dst_shared[j];
        if (dst_base < 0) continue;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y_R; i0 += WARP_SIZE) {
            const int i     = i0 + threadIdx.x;
            const int row_g = it * MMQ_Y_R + i;
            if (need_check && row_g >= nrows_x) continue;
            dst[(size_t) dst_base * nrows_dst + row_g] =
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y_R / WARP_SIZE) + i0 / WARP_SIZE];
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
    }                                                                     \
    extern "C" __global__                                                \
    __launch_bounds__(WARP_SIZE * MMQ_NWARPS, 2)                         \
    void mul_mat_q8_0_turbo_x##MMQ_X##_##SUFFIX(                         \
        const void * __restrict__ vx,                                    \
        const void * __restrict__ vy,                                    \
        float      * __restrict__ dst,                                   \
        const int ncols_x, const int nrows_x,                            \
        const int ncols_y, const int stride_col_y,                       \
        const int stride_row_x, const int nrows_dst) {                   \
        mul_mat_q8_0_turbo_impl<MMQ_X, CHECK>(                           \
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

// M7: MoE MMQ externs for Q4_0 / Q4_1 / Q8_0.  (No ids_src1 here — the
// gather happened at quantize time; Y is already expert-sorted.)
#define MMQ_TURBO_MOE_EXPORT_ONE(DTYPE_TAG, DT_ENUM, MMQ_X, CHECK, SUFFIX)   \
    extern "C" __global__                                                   \
    __launch_bounds__(WARP_SIZE * MMQ_NWARPS, 2)                            \
    void mul_mat_##DTYPE_TAG##_turbo_moe_x##MMQ_X##_##SUFFIX(               \
        const void  * __restrict__ vx,                                      \
        const void  * __restrict__ vy,                                      \
        const int   * __restrict__ ids_dst,                                 \
        const int   * __restrict__ expert_bounds,                           \
        float       * __restrict__ dst,                                     \
        const int ncols_x, const int nrows_x,                               \
        const int stride_col_y, const int stride_row_x,                     \
        const int stride_expert_blocks, const int nrows_dst) {              \
        mul_mat_q_turbo_moe_impl<DT_ENUM, MMQ_X, CHECK>(                    \
            vx, vy, ids_dst, expert_bounds, dst,                            \
            ncols_x, nrows_x, stride_col_y, stride_row_x,                   \
            stride_expert_blocks, nrows_dst);                               \
    }

#define MMQ_TURBO_MOE_EXPORT(MMQ_X, CHECK, SUFFIX)                          \
    MMQ_TURBO_MOE_EXPORT_ONE(q4_0, MOE_Q4_0, MMQ_X, CHECK, SUFFIX)          \
    MMQ_TURBO_MOE_EXPORT_ONE(q4_1, MOE_Q4_1, MMQ_X, CHECK, SUFFIX)          \
    MMQ_TURBO_MOE_EXPORT_ONE(q8_0, MOE_Q8_0, MMQ_X, CHECK, SUFFIX)          \
    MMQ_TURBO_MOE_EXPORT_ONE(q4_K_tpl, MOE_Q4_K, MMQ_X, CHECK, SUFFIX)

MMQ_TURBO_MOE_EXPORT( 8, false, unchecked)
MMQ_TURBO_MOE_EXPORT( 8, true,  checked)
MMQ_TURBO_MOE_EXPORT(16, false, unchecked)
MMQ_TURBO_MOE_EXPORT(16, true,  checked)
MMQ_TURBO_MOE_EXPORT(32, false, unchecked)
MMQ_TURBO_MOE_EXPORT(32, true,  checked)
MMQ_TURBO_MOE_EXPORT(64, false, unchecked)
MMQ_TURBO_MOE_EXPORT(64, true,  checked)

#undef MMQ_TURBO_MOE_EXPORT
#undef MMQ_TURBO_MOE_EXPORT_ONE

// ================================================================
// B2: Q4_K turbo_moe — dedicated per-row single-warp MMQ kernel.
//
// Not integrated into the main 4-warp template because the C3 attempt
// at integrating it had an unresolved stability bug on certain prompt
// shapes. This B2 single-warp kernel is slower per-call than turbo's
// production kernel (8.57ms vs 1.58ms, 5.4× gap) but is stable and
// delivers +2.3× over the indexed_moe baseline on Qwen3.6 pp512.
//
// Block: 64 threads (single warp), each thread owns one full output
// row across all TILE_N activation cols. Y read directly from the
// gather buffer (144-byte big-blocks) — no LDS tile. Register
// pressure is high (188 VGPR at TILE_N=32 → 1 wavefront/SIMD), but
// the kernel still beats indexed_moe by a large margin because
// batched weight reuse across tokens trumps occupancy loss.
// ================================================================

template <int TILE_N, bool need_check>
static __device__ void mul_mat_q4_K_turbo_moe_impl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    const int  * __restrict__ ids_dst,
    const int  * __restrict__ expert_bounds,
    float      * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_col_y,
    const int stride_row_x,             // = ncols_x / QK_K_Q4K_MOE (k/256)
    const int stride_expert_blocks,
    const int nrows_dst) {

    const int tid     = threadIdx.x;      // 0..63
    const int it      = blockIdx.x;
    const int jt      = blockIdx.y;
    const int expert  = blockIdx.z;

    const int col_low  = expert_bounds[expert];
    const int col_high = expert_bounds[expert + 1];
    const int col_diff = col_high - col_low;
    if (col_diff <= 0)             return;
    if (jt * TILE_N >= col_diff)   return;

    __shared__ int ids_dst_shared[64];
    if (tid < TILE_N) {
        const int local_j = jt * TILE_N + tid;
        ids_dst_shared[tid] =
            (local_j < col_diff) ? ids_dst[col_low + local_j] : -1;
    }
    __syncthreads();

    const int row        = it * WARP_SIZE + tid;
    const bool row_valid = (row < nrows_x);

    const block_q4_K_moe * x_exp = (const block_q4_K_moe *)
        ((const char *) vx + (size_t) expert * stride_expert_blocks
                               * (size_t) sizeof(block_q4_K_moe));

    const int blocks_per_row = stride_row_x;

    float sums[TILE_N] = {0.0f};

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};
        const block_q4_K_moe * bx = nullptr;
        if (row_valid) {
            bx = &x_exp[row * blocks_per_row + ib];
            const float2 dmf = __half22float2(bx->dm);
            super_d    = dmf.x;
            super_dmin = dmf.y;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int ksc = j;
                // Inline unpack of the 6-bit sc/m pair (same as turbo's
                // get_scale_min_k4): for j<4 use direct bytes; for j≥4
                // use bits from scales[j-4] and scales[j] combined.
                if (ksc < 4) {
                    sub_sc[j] = bx->scales[ksc]     & 63;
                    sub_m [j] = bx->scales[ksc + 4] & 63;
                } else {
                    sub_sc[j] = (bx->scales[ksc + 4] & 0xF) | ((bx->scales[ksc - 4] >> 6) << 4);
                    sub_m [j] = (bx->scales[ksc + 4] >>  4) | ((bx->scales[ksc - 0] >> 6) << 4);
                }
            }
        }

        float sumf_d[TILE_N] = {0.0f};
        float sumf_m[TILE_N] = {0.0f};

        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;
            int v[8] = {0};
            if (row_valid) {
                const int * ql_ptr = (const int *)(bx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_ptr[j];
                    v[j] = (half == 0)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
                }
            }
            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m [sub];

            const int big_y      = 2 * ib + (sub >> 2);
            const int within_big = sub & 3;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int local_j = jt * TILE_N + c;
                if (local_j >= col_diff) continue;
                const int bucket_pos = col_low + local_j;

                const uint8_t * y_bytes = (const uint8_t *) vy
                    + ((size_t) big_y * stride_col_y + bucket_pos)
                      * Q8_1_MMQ_BYTES;
                const half2 * ds = (const half2 *) y_bytes;
                const int   * qs = (const int   *)(y_bytes + 16);

                const half2   ds8   = ds[within_big];
                const int   * qs_sub = qs + within_big * 8;
                const float   d8     = __half2float(__low2half(ds8));

                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = qs_sub[j];
                    sumi_d = dp4a_sdot4(v[j], y_j, sumi_d);
                    sumi_y = dp4a_sdot4(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * sc_f;
                sumf_m[c] += d8 * ((float) sumi_y) * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
        }
    }

    if (!row_valid) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int local_j = jt * TILE_N + c;
        if (local_j >= col_diff) continue;
        const int dst_row = ids_dst_shared[c];
        if (dst_row < 0) continue;
        if (need_check && row >= nrows_dst) continue;
        dst[(size_t) dst_row * nrows_dst + row] = sums[c];
    }
}

#define MMQ_TURBO_MOE_Q4K_EXPORT(MMQ_X, CHECK, SUFFIX)                      \
    extern "C" __global__                                                   \
    __launch_bounds__(WARP_SIZE, 1)                                         \
    void mul_mat_q4_K_turbo_moe_x##MMQ_X##_##SUFFIX(                        \
        const void * __restrict__ vx,                                       \
        const void * __restrict__ vy,                                       \
        const int  * __restrict__ ids_dst,                                  \
        const int  * __restrict__ expert_bounds,                            \
        float      * __restrict__ dst,                                      \
        const int ncols_x, const int nrows_x,                               \
        const int stride_col_y, const int stride_row_x,                     \
        const int stride_expert_blocks, const int nrows_dst) {              \
        mul_mat_q4_K_turbo_moe_impl<MMQ_X, CHECK>(                          \
            vx, vy, ids_dst, expert_bounds, dst,                            \
            ncols_x, nrows_x, stride_col_y, stride_row_x,                   \
            stride_expert_blocks, nrows_dst);                               \
    }

MMQ_TURBO_MOE_Q4K_EXPORT( 8, false, unchecked)
MMQ_TURBO_MOE_Q4K_EXPORT( 8, true,  checked)
MMQ_TURBO_MOE_Q4K_EXPORT(16, false, unchecked)
MMQ_TURBO_MOE_Q4K_EXPORT(16, true,  checked)
MMQ_TURBO_MOE_Q4K_EXPORT(32, false, unchecked)
MMQ_TURBO_MOE_Q4K_EXPORT(32, true,  checked)
MMQ_TURBO_MOE_Q4K_EXPORT(64, false, unchecked)
MMQ_TURBO_MOE_Q4K_EXPORT(64, true,  checked)

#undef MMQ_TURBO_MOE_Q4K_EXPORT

// ================================================================
// B3: Q6_K turbo_moe — dedicated per-row single-warp MMQ kernel.
//
// Same pattern as B2 Q4_K: 64 threads, each owns one output row across
// TILE_N activation columns; reads Y directly from gather buffer (no LDS
// tile). Q6_K has no dmin/bias correction — signed 6-bit quants
// centered at -32. 16 int8 scales per superblock, one per 16 quants;
// within a 32-quant sub-block, first 16 quants use scales[2s], next 16
// use scales[2s+1].
// ================================================================

#define QK_K_Q6K_MOE  256
#define QR6_K_Q6K_MOE 2
#define QI6_K_Q6K_MOE (QK_K_Q6K_MOE / (4 * QR6_K_Q6K_MOE))   // = 32

typedef struct {
    uint8_t ql[QK_K_Q6K_MOE / 2];     // 128 bytes (256 × 4-bit low)
    uint8_t qh[QK_K_Q6K_MOE / 4];     // 64 bytes  (256 × 2-bit high)
    int8_t  scales[QK_K_Q6K_MOE / 16]; // 16 int8 scales
    half    d;                         // super-block scale
} block_q6_K_moe;
static_assert(sizeof(block_q6_K_moe) == 128 + 64 + 16 + 2,
              "block_q6_K_moe size mismatch (expected 210)");

template <int TILE_N, bool need_check>
static __device__ void mul_mat_q6_K_turbo_moe_impl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    const int  * __restrict__ ids_dst,
    const int  * __restrict__ expert_bounds,
    float      * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_col_y,
    const int stride_row_x,             // = ncols_x / 256
    const int stride_expert_blocks,
    const int nrows_dst) {

    const int tid     = threadIdx.x;      // 0..63
    const int it      = blockIdx.x;
    const int jt      = blockIdx.y;
    const int expert  = blockIdx.z;

    const int col_low  = expert_bounds[expert];
    const int col_high = expert_bounds[expert + 1];
    const int col_diff = col_high - col_low;
    if (col_diff <= 0)             return;
    if (jt * TILE_N >= col_diff)   return;

    __shared__ int ids_dst_shared[64];
    if (tid < TILE_N) {
        const int local_j = jt * TILE_N + tid;
        ids_dst_shared[tid] =
            (local_j < col_diff) ? ids_dst[col_low + local_j] : -1;
    }
    __syncthreads();

    const int row        = it * WARP_SIZE + tid;
    const bool row_valid = (row < nrows_x);

    const block_q6_K_moe * x_exp = (const block_q6_K_moe *)
        ((const char *) vx + (size_t) expert * stride_expert_blocks
                               * (size_t) sizeof(block_q6_K_moe));

    const int blocks_per_row = stride_row_x;

    float sums[TILE_N] = {0.0f};

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        float super_d = 0.0f;
        int8_t scales8[16] = {0};
        const block_q6_K_moe * bx = nullptr;
        if (row_valid) {
            bx = &x_exp[row * blocks_per_row + ib];
            super_d = __half2float(bx->d);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                scales8[j] = bx->scales[j];
            }
        }

        float sumf_d[TILE_N] = {0.0f};

        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            const int ip            = sub >> 2;           // 0 or 1
            const int is_hi         = (sub >> 1) & 1;     // low vs high nibble
            const int ql_off_extra  = (sub & 1) * 32;
            const int qh_shift      = 2 * (sub & 3);

            int v[8] = {0};
            if (row_valid) {
                const int * ql_words = (const int *)(bx->ql + 64 * ip + ql_off_extra);
                const int * qh_words = (const int *)(bx->qh + 32 * ip);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_words[j];
                    const int qh_word = qh_words[j];
                    const int ql_nib = is_hi
                        ? ((ql_word >> 4) & 0x0F0F0F0F)
                        : ( ql_word       & 0x0F0F0F0F);
                    const int qh_bits = ((qh_word >> qh_shift) & 0x03030303) << 4;
                    v[j] = vsubss4_moe(ql_nib | qh_bits, 0x20202020);
                }
            }
            const int sc_lo = row_valid ? (int) scales8[2 * sub    ] : 0;
            const int sc_hi = row_valid ? (int) scales8[2 * sub + 1] : 0;

            const int big_y      = 2 * ib + (sub >> 2);
            const int within_big = sub & 3;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int local_j = jt * TILE_N + c;
                if (local_j >= col_diff) continue;
                const int bucket_pos = col_low + local_j;

                const uint8_t * y_bytes = (const uint8_t *) vy
                    + ((size_t) big_y * stride_col_y + bucket_pos)
                      * Q8_1_MMQ_BYTES;
                const half2 * ds = (const half2 *) y_bytes;
                const int   * qs = (const int   *)(y_bytes + 16);

                const half2   ds8    = ds[within_big];
                const int   * qs_sub = qs + within_big * 8;
                const float   d8     = __half2float(__low2half(ds8));

                int sumi_lo = 0, sumi_hi = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi_lo = dp4a_sdot4(v[j], qs_sub[j], sumi_lo);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    sumi_hi = dp4a_sdot4(v[j], qs_sub[j], sumi_hi);
                }
                sumf_d[c] += d8 * (float)(sc_lo * sumi_lo + sc_hi * sumi_hi);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c];
        }
    }

    if (!row_valid) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int local_j = jt * TILE_N + c;
        if (local_j >= col_diff) continue;
        const int dst_row = ids_dst_shared[c];
        if (dst_row < 0) continue;
        if (need_check && row >= nrows_dst) continue;
        dst[(size_t) dst_row * nrows_dst + row] = sums[c];
    }
}

#define MMQ_TURBO_MOE_Q6K_EXPORT(MMQ_X, CHECK, SUFFIX)                      \
    extern "C" __global__                                                   \
    __launch_bounds__(WARP_SIZE, 1)                                         \
    void mul_mat_q6_K_turbo_moe_x##MMQ_X##_##SUFFIX(                        \
        const void * __restrict__ vx,                                       \
        const void * __restrict__ vy,                                       \
        const int  * __restrict__ ids_dst,                                  \
        const int  * __restrict__ expert_bounds,                            \
        float      * __restrict__ dst,                                      \
        const int ncols_x, const int nrows_x,                               \
        const int stride_col_y, const int stride_row_x,                     \
        const int stride_expert_blocks, const int nrows_dst) {              \
        mul_mat_q6_K_turbo_moe_impl<MMQ_X, CHECK>(                          \
            vx, vy, ids_dst, expert_bounds, dst,                            \
            ncols_x, nrows_x, stride_col_y, stride_row_x,                   \
            stride_expert_blocks, nrows_dst);                               \
    }

MMQ_TURBO_MOE_Q6K_EXPORT( 8, false, unchecked)
MMQ_TURBO_MOE_Q6K_EXPORT( 8, true,  checked)
MMQ_TURBO_MOE_Q6K_EXPORT(16, false, unchecked)
MMQ_TURBO_MOE_Q6K_EXPORT(16, true,  checked)
MMQ_TURBO_MOE_Q6K_EXPORT(32, false, unchecked)
MMQ_TURBO_MOE_Q6K_EXPORT(32, true,  checked)
MMQ_TURBO_MOE_Q6K_EXPORT(64, false, unchecked)
MMQ_TURBO_MOE_Q6K_EXPORT(64, true,  checked)

#undef MMQ_TURBO_MOE_Q6K_EXPORT

// ================================================================
// B4: Q5_K turbo_moe — dedicated per-row single-warp MMQ kernel.
//
// Q5_K = Q4_K body + 32-byte qh array of 1-bit-per-quant high bits.
// Same Q4_K-style bias correction (dm = (d, dmin) half2 + 8 sc/m
// 6-bit-packed scales). Dequant: y = super_d * sc * q5 - super_dmin * m
// where q5 = ql_nibble | (qh_bit << 4) ∈ [0, 31].
// Within a 32-quant sub-block, the qh_bit index is just `sub` (0..7),
// because qh[32] stores 8 bits per byte × 32 bytes = 256 bits, one per
// quant, with layout such that bit `sub` of qh[j*4..j*4+4] gives the
// high-bits for the 4 quants in int j of sub s.
// ================================================================

#define QK_K_Q5K_MOE  256
#define K_SCALE_SIZE_Q5K_MOE 12

typedef struct {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE_Q5K_MOE]; // 12 bytes (same format as Q4_K)
    uint8_t qh[QK_K_Q5K_MOE / 8];         // 32 bytes (256 × 1-bit high)
    uint8_t qs[QK_K_Q5K_MOE / 2];         // 128 bytes (256 × 4-bit low)
} block_q5_K_moe;
static_assert(sizeof(block_q5_K_moe) == 4 + 12 + 32 + 128,
              "block_q5_K_moe size mismatch (expected 176)");

template <int TILE_N, bool need_check>
static __device__ void mul_mat_q5_K_turbo_moe_impl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    const int  * __restrict__ ids_dst,
    const int  * __restrict__ expert_bounds,
    float      * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_col_y,
    const int stride_row_x,             // = ncols_x / 256
    const int stride_expert_blocks,
    const int nrows_dst) {

    const int tid     = threadIdx.x;      // 0..63
    const int it      = blockIdx.x;
    const int jt      = blockIdx.y;
    const int expert  = blockIdx.z;

    const int col_low  = expert_bounds[expert];
    const int col_high = expert_bounds[expert + 1];
    const int col_diff = col_high - col_low;
    if (col_diff <= 0)             return;
    if (jt * TILE_N >= col_diff)   return;

    __shared__ int ids_dst_shared[64];
    if (tid < TILE_N) {
        const int local_j = jt * TILE_N + tid;
        ids_dst_shared[tid] =
            (local_j < col_diff) ? ids_dst[col_low + local_j] : -1;
    }
    __syncthreads();

    const int row        = it * WARP_SIZE + tid;
    const bool row_valid = (row < nrows_x);

    const block_q5_K_moe * x_exp = (const block_q5_K_moe *)
        ((const char *) vx + (size_t) expert * stride_expert_blocks
                               * (size_t) sizeof(block_q5_K_moe));

    const int blocks_per_row = stride_row_x;

    float sums[TILE_N] = {0.0f};

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};
        const block_q5_K_moe * bx = nullptr;
        if (row_valid) {
            bx = &x_exp[row * blocks_per_row + ib];
            const float2 dmf = __half22float2(bx->dm);
            super_d    = dmf.x;
            super_dmin = dmf.y;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int ksc = j;
                if (ksc < 4) {
                    sub_sc[j] = bx->scales[ksc]     & 63;
                    sub_m [j] = bx->scales[ksc + 4] & 63;
                } else {
                    sub_sc[j] = (bx->scales[ksc + 4] & 0xF) | ((bx->scales[ksc - 4] >> 6) << 4);
                    sub_m [j] = (bx->scales[ksc + 4] >>  4) | ((bx->scales[ksc - 0] >> 6) << 4);
                }
            }
        }

        float sumf_d[TILE_N] = {0.0f};
        float sumf_m[TILE_N] = {0.0f};

        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            const int il       = sub >> 1;
            const int half     = sub & 1;
            const int qh_shift = sub;          // bit index in qh byte
            int v[8] = {0};
            if (row_valid) {
                const int * ql_words = (const int *)(bx->qs + 32 * il);
                const int * qh_words = (const int *)(bx->qh);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_words[j];
                    const int qh_word = qh_words[j];
                    const int ql_nib = (half == 0)
                        ? ( ql_word       & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int qh_bits = ((qh_word >> qh_shift) & 0x01010101) << 4;
                    v[j] = ql_nib | qh_bits;
                }
            }
            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m [sub];

            const int big_y      = 2 * ib + (sub >> 2);
            const int within_big = sub & 3;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int local_j = jt * TILE_N + c;
                if (local_j >= col_diff) continue;
                const int bucket_pos = col_low + local_j;

                const uint8_t * y_bytes = (const uint8_t *) vy
                    + ((size_t) big_y * stride_col_y + bucket_pos)
                      * Q8_1_MMQ_BYTES;
                const half2 * ds = (const half2 *) y_bytes;
                const int   * qs = (const int   *)(y_bytes + 16);

                const half2   ds8   = ds[within_big];
                const int   * qs_sub = qs + within_big * 8;
                const float   d8     = __half2float(__low2half(ds8));

                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = qs_sub[j];
                    sumi_d = dp4a_sdot4(v[j], y_j, sumi_d);
                    sumi_y = dp4a_sdot4(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * sc_f;
                sumf_m[c] += d8 * ((float) sumi_y) * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
        }
    }

    if (!row_valid) return;
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int local_j = jt * TILE_N + c;
        if (local_j >= col_diff) continue;
        const int dst_row = ids_dst_shared[c];
        if (dst_row < 0) continue;
        if (need_check && row >= nrows_dst) continue;
        dst[(size_t) dst_row * nrows_dst + row] = sums[c];
    }
}

#define MMQ_TURBO_MOE_Q5K_EXPORT(MMQ_X, CHECK, SUFFIX)                      \
    extern "C" __global__                                                   \
    __launch_bounds__(WARP_SIZE, 1)                                         \
    void mul_mat_q5_K_turbo_moe_x##MMQ_X##_##SUFFIX(                        \
        const void * __restrict__ vx,                                       \
        const void * __restrict__ vy,                                       \
        const int  * __restrict__ ids_dst,                                  \
        const int  * __restrict__ expert_bounds,                            \
        float      * __restrict__ dst,                                      \
        const int ncols_x, const int nrows_x,                               \
        const int stride_col_y, const int stride_row_x,                     \
        const int stride_expert_blocks, const int nrows_dst) {              \
        mul_mat_q5_K_turbo_moe_impl<MMQ_X, CHECK>(                          \
            vx, vy, ids_dst, expert_bounds, dst,                            \
            ncols_x, nrows_x, stride_col_y, stride_row_x,                   \
            stride_expert_blocks, nrows_dst);                               \
    }

MMQ_TURBO_MOE_Q5K_EXPORT( 8, false, unchecked)
MMQ_TURBO_MOE_Q5K_EXPORT( 8, true,  checked)
MMQ_TURBO_MOE_Q5K_EXPORT(16, false, unchecked)
MMQ_TURBO_MOE_Q5K_EXPORT(16, true,  checked)
MMQ_TURBO_MOE_Q5K_EXPORT(32, false, unchecked)
MMQ_TURBO_MOE_Q5K_EXPORT(32, true,  checked)
MMQ_TURBO_MOE_Q5K_EXPORT(64, false, unchecked)
MMQ_TURBO_MOE_Q5K_EXPORT(64, true,  checked)

#undef MMQ_TURBO_MOE_Q5K_EXPORT
