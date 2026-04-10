#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#ifndef __HIP_PLATFORM_AMD__
#include "cuda_fp8.h"
#endif

// HIP compatibility layer for candle kernels.

// ROCm 6.4 removed atomicCAS(unsigned short*, ...) but 7.1+ re-added it.
// Provide a polyfill only when the HIP header doesn't define it.
// Detection: HIP_VERSION >= 7.1 has it; HIP_VERSION 6.x does not.
#include <hip/hip_version.h>
#if !defined(__HIPCC_ATOMICS_USHORT_POLYFILL__) && defined(HIP_VERSION) && HIP_VERSION < 70100000
#define __HIPCC_ATOMICS_USHORT_POLYFILL__
__device__ __forceinline__ unsigned short atomicCAS(
    unsigned short* address, unsigned short compare, unsigned short val)
{
    unsigned int* base = (unsigned int*)((size_t)address & ~(size_t)2);
    unsigned int shift = ((size_t)address & 2) ? 16 : 0;
    unsigned int mask = 0xFFFFu << shift;
    unsigned int old_int = *base, assumed;
    do {
        assumed = old_int;
        unsigned short old_short = (unsigned short)(old_int >> shift);
        if (old_short != compare) return old_short;
        unsigned int new_int = (old_int & ~mask) | ((unsigned int)val << shift);
        old_int = atomicCAS(base, assumed, new_int);
    } while (assumed != old_int);
    return (unsigned short)(old_int >> shift);
}
#endif
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
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(candle_hmax_nan(val, __ushort_as_half(assumed))));
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
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(candle_hmin_nan(val, __ushort_as_half(assumed))));
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
