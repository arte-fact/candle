#include "hip_utils.cuh"
#include <cmath>
#include <stdint.h>
#include <limits>

// hip_bfloat16 lacks a native atomicAdd on gfx906. Emulate via CAS on uint16.
__device__ __forceinline__ hip_bfloat16 atomicAdd(hip_bfloat16* addr, hip_bfloat16 val) {
    unsigned short int* addr_as_ushort = reinterpret_cast<unsigned short int*>(addr);
    unsigned short int old = *addr_as_ushort;
    unsigned short int assumed;
    do {
        assumed = old;
        // Reinterpret bits as bf16, add in f32, convert back.
        union { unsigned short int u; hip_bfloat16 b; } conv_old, conv_new;
        conv_old.u = assumed;
        conv_new.b = hip_bfloat16(float(conv_old.b) + float(val));
        old = atomicCAS(addr_as_ushort, assumed, conv_new.u);
    } while (assumed != old);
    union { unsigned short int u; hip_bfloat16 b; } conv_ret;
    conv_ret.u = old;
    return conv_ret.b;
}

// __half also lacks a native atomicAdd on gfx906. Same CAS emulation.
__device__ __forceinline__ __half atomicAdd(__half* addr, __half val) {
    unsigned short int* addr_as_ushort = reinterpret_cast<unsigned short int*>(addr);
    unsigned short int old = *addr_as_ushort;
    unsigned short int assumed;
    do {
        assumed = old;
        __half old_val = __ushort_as_half(assumed);
        __half new_val = __float2half(__half2float(old_val) + __half2float(val));
        old = atomicCAS(addr_as_ushort, assumed, __half_as_ushort(new_val));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// WARP_SIZE is defined externally via -DWARP_SIZE=64 for gfx906 (Wave64).
// Do NOT hardcode 32 here.
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

const int BLOCK_SIZE = 1024;

// Helpers to initialize reduction identities for both floating-point and
// integer types. For floats we keep using +/-INFINITY, while for integers
// we use well-defined numeric_limits values instead of relying on casting
// +/-INFINITY to an integer type (which is undefined behaviour).
template <typename T>
__device__ __forceinline__ T reduce_init_lowest() {
  // Default implementation is used for floating-point types (__half,
  // hip_bfloat16, float, double). The conversion from -INFINITY (double)
  // to these types is well-defined and produces -inf.
  return -INFINITY;
}

template <typename T>
__device__ __forceinline__ T reduce_init_highest() {
  // Default implementation is used for floating-point types (__half,
  // hip_bfloat16, float, double). The conversion from INFINITY (double)
  // to these types is well-defined and produces +inf.
  return INFINITY;
}

// Integer specializations -- use numeric_limits instead of +/-INFINITY.
template <>
__device__ __forceinline__ int64_t reduce_init_lowest<int64_t>() {
  return INT64_MIN;
}

template <>
__device__ __forceinline__ uint32_t reduce_init_lowest<uint32_t>() {
  return 0;
}

template <>
__device__ __forceinline__ uint8_t reduce_init_lowest<uint8_t>() {
  return 0;
}

template <>
__device__ __forceinline__ int64_t reduce_init_highest<int64_t>() {
  return INT64_MAX;
}

template <>
__device__ __forceinline__ uint32_t reduce_init_highest<uint32_t>() {
  return UINT32_MAX;
}

template <>
__device__ __forceinline__ uint8_t reduce_init_highest<uint8_t>() {
  return 0xFF;
}

// hip_bfloat16 has explicit constructors — must specialize.
template <>
__device__ __forceinline__ hip_bfloat16 reduce_init_lowest<hip_bfloat16>() {
  return hip_bfloat16(-INFINITY);
}
template <>
__device__ __forceinline__ hip_bfloat16 reduce_init_highest<hip_bfloat16>() {
  return hip_bfloat16(INFINITY);
}

// TODO: Maybe add some fast_sum_f16_f32 variant that not only accumulate in f32
// but also expect a f32 output so that this can be used for normalization e.g.
// in softmax.

// Fast reduce sum kernel, this assumes that the dimensions to loop over are at
// the end, each block is responsible for populating one value in the output
// array. There are at most 1024 threads per block.
template <typename T>
__device__ void
fast_sum(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  shr[tid] = 0;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] += src[strided_i];
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] += shr[tid + s];
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

// Warp-level reduction using __shfl_xor for Wave64.
// On HIP/AMD, __shfl_xor does not take a mask or width argument.
// For WARP_SIZE=64, we need 6 stages (32,16,8,4,2,1).
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x += __shfl_xor(a.x, mask);
        a.y += __shfl_xor(a.y, mask);
    }
    return a;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}

// LayerNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L477
template <typename T>
__device__ void layernorm(const T * x, T * dst, const T * alpha, const T * beta, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        // With WARP_SIZE=64 and BLOCK_SIZE=1024, we have at most 16 warps.
        // Shared memory array sized to hold one entry per warp.
        __shared__ float2 s_sum[BLOCK_SIZE / WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        // Only first warp participates in the final reduction.
        // Load from shared memory only for valid warp slots.
        int num_warps = block_size / WARP_SIZE;
        mean_var = (lane_id < num_warps) ? s_sum[lane_id] : make_float2(0.f, 0.f);
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    if (alpha == nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a + b);
      }
    }
}

// RmsNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L523
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = static_cast<float>(x[row*ncols + col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        // With WARP_SIZE=64 and BLOCK_SIZE=1024, we have at most 16 warps.
        __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        int num_warps = block_size / WARP_SIZE;
        tmp = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if (alpha == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]));
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]) * a);
      }
    }
}

// Fused: out_normed = rmsnorm(h + delta, alpha) ; out_h_new = h + delta
//
// Eliminates the separate `h = h + delta` launch + intermediate buffer in
// the transformer pre-norm path. The two outputs are written to a single
// packed buffer of shape (2, n_rows, n_cols):
//   dst[0      .. n_rows*n_cols] = normed (input to next attention/ffn)
//   dst[n_rows*n_cols .. 2*..]   = h_new   (residual stream for next add)
//
// Both halves are contiguous standalone (n_rows, n_cols) tensors so the
// caller can recover them via `narrow(0, 0, 1).squeeze(0)` and
// `narrow(0, 1, 1).squeeze(0)` at zero kernel cost.
//
// Two-pass design (matches the standalone `rmsnorm` kernel above):
//   Pass 1: read h[i] + delta[i], accumulate (h+delta)^2 into a per-thread
//           partial sum. No writes — keeps the loop cache-resident.
//   Pass 2: re-read h[i] + delta[i], compute the same s = h+delta, then
//           write *both* h_new[i] = s and normed[i] = s * scale * alpha[i].
//
// Re-reading h and delta in pass 2 (instead of writing s in pass 1 and
// reading it back) avoids the need for `__syncthreads()` between the two
// halves of the kernel. Total memory traffic is the same as the unfused
// (residual_add + rmsnorm) sequence — the saving is one launch + one
// intermediate allocation per call.
template <typename T>
__device__ void rmsnorm_add(
    const T * __restrict__ h,
    const T * __restrict__ delta,
    T * __restrict__ dst_normed,   // first half of the packed output
    T * __restrict__ dst_h_new,    // second half of the packed output
    const T * __restrict__ alpha,
    const int ncols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f;

    // Pass 1: accumulate sum of squares of (h + delta)
    for (int col = tid; col < ncols; col += block_size) {
        const int i = row * ncols + col;
        const float s = static_cast<float>(h[i]) + static_cast<float>(delta[i]);
        tmp += s * s;
    }

    // Block-level reduction (matches the standalone `rmsnorm` kernel).
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        int num_warps = block_size / WARP_SIZE;
        tmp = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    // Pass 2: write both outputs.
    for (int col = tid; col < ncols; col += block_size) {
        const int i = row * ncols + col;
        const float s = static_cast<float>(h[i]) + static_cast<float>(delta[i]);
        dst_h_new[i] = static_cast<T>(s);
        const float a = static_cast<float>(alpha[col]);
        dst_normed[i] = static_cast<T>(s * scale * a);
    }
}

// Softmax implementation adapted from ggml.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L4159
template <typename T, typename ACC>
__device__ void softmax(const T * x, T * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    T max_val = reduce_init_lowest<T>();

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = maxg(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        max_val = maxg(max_val, __shfl_xor(max_val, mask));
    }

    ACC tmp = static_cast<ACC>(0);

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const T val = expg(x[i] - max_val);
        tmp += static_cast<ACC>(val);
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += __shfl_xor(tmp, mask);
    }

    const ACC inv_tmp = static_cast<ACC>(1) / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] = static_cast<T>(static_cast<ACC>(dst[i]) * inv_tmp);
    }
}

// Fused masked_softmax_scale: replaces the
//   `attn_weights = attn_weights * scale`
//   `attn_weights = attn_weights + mask`   (additive f32 mask)
//   `attn_weights = softmax_last_dim(attn_weights)`
// chain that sits between QK^T and SV in every attention block.
// Eliminates 2 of the 3 kernel launches per layer (affine + broadcast_add)
// and 2 full memory sweeps over the (B, H, L_q, L_k) score tensor.
//
// Input `x` shape: (B, H, L_q, L_k), contiguous, row-major. One row of
// length L_k per (b, h, q).
//
// Mask layout: additive f32, broadcast via caller-provided strides:
//   `mask_b_stride`  = elements to skip from one batch to the next
//                      (0 for broadcast over B, L_q*L_k otherwise)
//   `mask_lq_stride` = elements to skip from row q to row q+1
//                      (0 for broadcast over L_q, L_k otherwise)
// Pass nullptr for `mask` when there is no mask.
//
// Launch:
//   grid  = (n_rows, 1, 1)              where n_rows = B * H * L_q
//   block = (1, WARP_SIZE, 1)           one Wave64 warp per row
//   block_dim.y = 64 so threadIdx.y handles cols {tid, tid+64, ...}
template <typename T, typename ACC>
__device__ void masked_softmax_scale(
        const T * __restrict__ x,
        const T * __restrict__ mask,
        T * __restrict__ dst,
        const int n_cols,
        const int b_dim,
        const int h_dim,
        const int lq_dim,
        const float scale,
        const int mask_b_stride,
        const int mask_lq_stride) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    // Decompose row index back to (b, q) so we can index the broadcast mask.
    // Layout: row = ((b * H) + h) * L_q + q.
    const int hq = h_dim * lq_dim;
    const int b  = row / hq;
    const int q  = row % lq_dim;

    const T * mask_row = nullptr;
    if (mask != nullptr) {
        mask_row = mask
                 + (long)b * (long)mask_b_stride
                 + (long)q * (long)mask_lq_stride;
    }

    // --- Pass 1: compute max(scale * x + mask) across the row. ---
    T max_val = reduce_init_lowest<T>();
    for (int col = tid; col < n_cols; col += block_size) {
        const int i = row*n_cols + col;
        T s = x[i] * (T)scale;
        if (mask_row != nullptr) {
            s = s + mask_row[col];
        }
        max_val = maxg(max_val, s);
    }
    #pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1) {
        max_val = maxg(max_val, __shfl_xor(max_val, m));
    }

    // --- Pass 2: compute exp(s - max), sum, write to dst. ---
    ACC tmp = static_cast<ACC>(0);
    for (int col = tid; col < n_cols; col += block_size) {
        const int i = row*n_cols + col;
        T s = x[i] * (T)scale;
        if (mask_row != nullptr) {
            s = s + mask_row[col];
        }
        const T val = expg(s - max_val);
        tmp += static_cast<ACC>(val);
        dst[i] = val;
    }
    #pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1) {
        tmp += __shfl_xor(tmp, m);
    }

    // --- Pass 3: normalise. ---
    const ACC inv_tmp = static_cast<ACC>(1) / tmp;
    for (int col = tid; col < n_cols; col += block_size) {
        const int i = row*n_cols + col;
        dst[i] = static_cast<T>(static_cast<ACC>(dst[i]) * inv_tmp);
    }
}

template <typename T>
__device__ void ropei(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t rope_idx = idx % (td / 2);
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];

    dst[2 * idx] = src[2 * idx] * c - src[2 * idx + 1] * s;
    dst[2 * idx + 1] = src[2 * idx] * s + src[2 * idx + 1] * c;
}

template <typename T>
__device__ void rope(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bh = idx / (td / 2);
    uint32_t i_td = idx - (td / 2) * i_bh;
    uint32_t i_t = i_td / (d / 2);
    uint32_t i_d = i_td - (d / 2) * i_t;
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void rope_thd(
    const T * src,
    const T * cos,
    const T * sin,
    T * dst,
    const uint32_t b,
    const uint32_t t,
    const uint32_t h,
    const uint32_t d,
    const uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= b * t * h * d) return;

    uint32_t i_bth = idx / (d / 2);
    uint32_t i_d = idx - (d / 2) * i_bth;
    uint32_t i_t = (i_bth / h) % t;
    uint32_t i1 = i_bth * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void
fast_max(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the lowest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_lowest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = maxg(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = maxg(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_min(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the highest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_highest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = ming(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = ming(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_argmin(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses +inf; for integer types we use the largest
  // representable value instead of casting INFINITY to an integer.
  shr[tid] = reduce_init_highest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] < shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] < shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

template <typename T>
__device__ void
fast_argmax(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses -inf; for integer types we use the lowest
  // representable value instead of casting -INFINITY to an integer.
  shr[tid] = reduce_init_lowest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] > shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] > shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

#define FAST_OP(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME, SUM_NAME) \
  extern "C" __global__ void ARGMIN_NAME(                                      \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void ARGMAX_NAME(                                     \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void MIN_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void MAX_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void SUM_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }

#define SUM_OP(TYPENAME, FN_NAME)                                              \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t num_sum_dims,    \
      const size_t *info, const TYPENAME *inp, TYPENAME *out) {                \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    const size_t *sum_dims_l = info + 2 * num_dims;                            \
    const size_t *sum_dims_s = info + 2 * num_dims + num_sum_dims;             \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[i]);                                    \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i]);                            \
      }                                                                        \
    }                                                                          \
  }

#define SOFTMAX_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst,                                      \
      const int n_cols) {                                                      \
    softmax<TYPENAME, ACC_TYPENAME>(src, dst, n_cols);                         \
  }                                                                            \

#define MASKED_SOFTMAX_SCALE_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, const TYPENAME *mask, TYPENAME *dst,                \
      const int n_cols, const int b_dim, const int h_dim, const int lq_dim,   \
      const float scale,                                                       \
      const int mask_b_stride, const int mask_lq_stride) {                     \
    masked_softmax_scale<TYPENAME, ACC_TYPENAME>(                              \
        src, mask, dst, n_cols, b_dim, h_dim, lq_dim,                          \
        scale, mask_b_stride, mask_lq_stride);                                 \
  }                                                                            \

#define RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const int n_cols, const int block_size, const float eps) {               \
    rmsnorm<TYPENAME>(src, dst, alpha, n_cols, block_size, eps);               \
  }                                                                            \

#define RMSNORM_ADD_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *h, const TYPENAME *delta,                                \
      TYPENAME *dst_normed, TYPENAME *dst_h_new,                               \
      const TYPENAME *alpha,                                                   \
      const int n_cols, const int block_size, const float eps) {               \
    rmsnorm_add<TYPENAME>(h, delta, dst_normed, dst_h_new, alpha,              \
                          n_cols, block_size, eps);                            \
  }                                                                            \

#define LAYERNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const TYPENAME *beta, const int n_cols, const int block_size, const float eps) { \
    layernorm<TYPENAME>(src, dst, alpha, beta, n_cols, block_size, eps);       \
  }                                                                            \

#define ROPE_OP(TYPENAME, FN_NAME, FN_NAME_I, FN_NAME_THD) \
  extern "C" __global__ void FN_NAME_I( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t stride_b) { \
    ropei<TYPENAME>(src, cos, sin, dst, bh, td, stride_b); \
  } \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope<TYPENAME>(src, cos, sin, dst, bh, td, d, stride_b); \
  } \
  extern "C" __global__ void FN_NAME_THD( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t b, \
      const uint32_t t, \
      const uint32_t h, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope_thd<TYPENAME>(src, cos, sin, dst, b, t, h, d, stride_b); \
  } \

// bf16 -- always available on HIP via hip_bfloat16
SOFTMAX_OP(hip_bfloat16, float, softmax_bf16)
RMSNORM_OP(hip_bfloat16, rmsnorm_bf16)
RMSNORM_ADD_OP(hip_bfloat16, rmsnorm_add_bf16)
LAYERNORM_OP(hip_bfloat16, layernorm_bf16)
ROPE_OP(hip_bfloat16, rope_bf16, rope_i_bf16, rope_thd_bf16)
SUM_OP(hip_bfloat16, sum_bf16)
FAST_OP(hip_bfloat16, fast_min_bf16, fast_max_bf16, fast_argmin_bf16, fast_argmax_bf16, fast_sum_bf16)

// NOTE: No reduce ops for fp8 (not supported on AMD gfx906)

// fp16 -- always available on HIP (gfx906+)
SOFTMAX_OP(__half, float, softmax_f16)
RMSNORM_OP(__half, rmsnorm_f16)
RMSNORM_ADD_OP(__half, rmsnorm_add_f16)
LAYERNORM_OP(__half, layernorm_f16)
ROPE_OP(__half, rope_f16, rope_i_f16, rope_thd_f16)
SUM_OP(__half, sum_f16)
FAST_OP(__half, fast_min_f16, fast_max_f16, fast_argmin_f16, fast_argmax_f16, fast_sum_f16)

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)
SOFTMAX_OP(float, float, softmax_f32)
SOFTMAX_OP(double, double, softmax_f64)
MASKED_SOFTMAX_SCALE_OP(float, float, masked_softmax_scale_f32)
RMSNORM_OP(float, rmsnorm_f32)
RMSNORM_OP(double, rmsnorm_f64)
RMSNORM_ADD_OP(float, rmsnorm_add_f32)
RMSNORM_ADD_OP(double, rmsnorm_add_f64)
LAYERNORM_OP(float, layernorm_f32)
LAYERNORM_OP(double, layernorm_f64)
ROPE_OP(float, rope_f32, rope_i_f32, rope_thd_f32)
ROPE_OP(double, rope_f64, rope_i_f64, rope_thd_f64)

FAST_OP(float, fast_min_f32, fast_max_f32, fast_argmin_f32, fast_argmax_f32, fast_sum_f32)
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)
