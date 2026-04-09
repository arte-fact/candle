#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#ifndef __HIP_PLATFORM_AMD__
#include "cuda_fp8.h"
#endif

// HIP compatibility layer for candle kernels.
// On AMD GPUs (gfx906+), fp16 and atomics are always available.
// bf16 is emulated via f32 conversions through hip_bfloat16.
// fp8 (e4m3) is not supported on HIP and is guarded out.

// __hmax_nan / __hmin_nan may not be provided by all HIP versions.
// Provide our own versions unconditionally with unique names to avoid conflicts.
__device__ __forceinline__ __half candle_hmax_nan(__half a, __half b) {
    float fa = __half2float(a), fb = __half2float(b);
    if (isnan(fa)) return a;
    if (isnan(fb)) return b;
    return __float2half(fmaxf(fa, fb));
}
__device__ __forceinline__ __half candle_hmin_nan(__half a, __half b) {
    float fa = __half2float(a), fb = __half2float(b);
    if (isnan(fa)) return a;
    if (isnan(fb)) return b;
    return __float2half(fminf(fa, fb));
}

// double atomicAdd is natively supported on gfx906+ (HIP), no polyfill needed.

// atomicMaxf for __half
__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
    // On HIP/AMD, atomicCAS for unsigned short is available on gfx906+.
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(candle_hmax_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMaxf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int *)addr, __double_as_longlong(value)));
    }
}


// atomicMinf for __half
__device__ __forceinline__ __half atomicMinf(__half* address, __half val) {
    // On HIP/AMD, atomicCAS for unsigned short is available on gfx906+.
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(candle_hmin_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMin is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMinf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int *)addr, __double_as_longlong(value)));
    }
}
