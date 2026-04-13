#pragma once
// gfx906_primitives.cuh — ISA-native primitives for AMD gfx906 (MI50/MI60).
//
// Provides DPP-fused warp reductions, hardware SFU math, and scalar broadcast.
// These replace the generic __shfl_xor loop reductions and software math
// functions with native GCN 5.1 instructions for ~1.7x faster reductions
// and ~8x faster transcendentals.
//
// Usage: #include "gfx906_primitives.cuh" in any .cu file, then call
//   gfx906_warp_reduce_sum(x)   instead of the __shfl_xor loop
//   gfx906_warp_reduce_max(x)   instead of the __shfl_xor fmaxf loop
//   gfx906_fast_exp(x)          instead of __expf(x) or expf(x)
//   etc.

#include <hip/hip_runtime.h>

// ============================================================================
// Section 1: DPP Warp Reductions
// ============================================================================
//
// On gfx906 (GCN 5.1), Data-Parallel Primitives (DPP) fuse lane permutations
// into VALU operations. A `v_add_f32_dpp ... quad_perm:[1,0,3,2]` does the
// xor-1 reduction in ONE cycle with zero data-movement overhead — the
// permutation happens in the register file's read ports.
//
// Full 64-wide reduction stages:
//   xor 1  → DPP quad_perm:[1,0,3,2]        (1 cycle, fused)
//   xor 2  → DPP quad_perm:[2,3,0,1]        (1 cycle, fused)
//   xor 4  → __shfl_xor(x, 4)               (2 cycles, readlane/writelane)
//   xor 8  → DPP row_ror:8                   (1 cycle, fused)
//   xor 16 → ds_swizzle_b32 xor:16           (1 cycle, LDS crossbar)
//   xor 32 → __shfl_xor(x, 32, 64)          (2 cycles, cross-half)
//
// Total: ~8 cycles vs __shfl_xor loop's ~12 cycles for 64-wide.
// Half-warp (32-wide): ~5 cycles vs ~10 cycles.

#ifdef __HIP_PLATFORM_AMD__

// --- DPP fused add/max primitives (2026-04-13 re-port of turbo pattern) ---
//
// Vega ISA Table 8 (§4.5) wait-state rules for DPP:
//   VALU writes VGPR  → VALU DPP reads that VGPR  → 2 wait states
//   VALU writes EXEC  → VALU DPP op               → 5 wait states
//
// The EXEC rule is the usual trap: any `if`/compare above a reduction call
// writes EXEC (via v_cmpx_*). The FIRST DPP after a reduction entry point
// therefore needs `s_nop 4` (= 5 cycles) to be robust. Subsequent DPPs in
// the same reduction chain follow other DPP ops (which don't write EXEC)
// and can use `s_nop 1` (= 2 cycles) per the VGPR-write rule.
//
// Key port points from turbo (`ggml-cuda/gfx906/gfx906-common.cuh:73-93`):
//   - SINGLE-arg inline asm: "%1, %1" ensures compiler uses same VGPR for
//     src0 and src1, no intermediate MOV that could break timing.
//   - NO `bound_ctrl` specified (default 0 = don't write on OOB). Our
//     earlier `bound_ctrl:1` was probably wrong for these patterns; the
//     quad_perm and row_ror:8 patterns never have OOB anyway but setting
//     bound_ctrl:1 might poke a different codegen path.
//   - "memory" clobber to prevent the compiler reordering instructions
//     across the DPP (critical for the wait-state to hold).
//   - `s_nop 4` on first DPP of each reduction chain (xor1) — 5 cycles to
//     cover EXEC hazard; `s_nop 1` on subsequent (xor2, xor8) since the
//     prior op is itself DPP.

#define CANDLE_FUSED_DPP_F32(name, barrier, dpp_ctrl, vop_instr)        \
    static __device__ __forceinline__ float name(float x) {             \
        float r;                                                        \
        asm volatile(                                                   \
            barrier                                                     \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory"                               \
        );                                                              \
        return r;                                                       \
    }

CANDLE_FUSED_DPP_F32(gfx906_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
CANDLE_FUSED_DPP_F32(gfx906_dpp_max_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")
CANDLE_FUSED_DPP_F32(gfx906_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
CANDLE_FUSED_DPP_F32(gfx906_dpp_max_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")
CANDLE_FUSED_DPP_F32(gfx906_dpp_add_ror8, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
CANDLE_FUSED_DPP_F32(gfx906_dpp_max_ror8, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef CANDLE_FUSED_DPP_F32

// --- xor-4 via v_mov_b32_dpp with split bank_mask (turbo pattern) ---
// row_shl:4 with bank_mask:0x5 (banks 0,2) + row_shr:4 with bank_mask:0xa (banks 1,3)
// writes both directions in-place; the result is a xor-4 permutation.

static __device__ __forceinline__ float gfx906_shuffle_xor4(float x) {
    int src = __float_as_int(x);
    int dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(dst) : "v"(src) : "memory"
    );
    return __int_as_float(dst);
}

// --- xor-16 via ds_swizzle_b32 (LDS crossbar, no actual LDS) ---
// Symbolic offset form: swizzle(SWAP,16) — more readable than hex constant.

static __device__ __forceinline__ float gfx906_swizzle_xor16(float x) {
    int src = __float_as_int(x);
    int dst;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(dst) : "v"(src) : "memory"
    );
    return __int_as_float(dst);
}

// --- Full 64-wide warp reductions (DPP chain) ---
//
// Stages match turbo's warp_reduce_amd_f32 template:
//   xor 1  → DPP quad_perm:[1,0,3,2]    (1 VALU cycle fused)
//   xor 2  → DPP quad_perm:[2,3,0,1]    (1 VALU cycle fused)
//   xor 4  → v_mov_b32_dpp split-bank   (2 VALU cycles)
//   xor 8  → DPP row_ror:8              (1 VALU cycle fused)
//   xor 16 → ds_swizzle_b32 SWAP,16     (1 LDS-xbar cycle)
//   xor 32 → __shfl_xor(x, 32, 64)      (cross-half, 2 cycles)
// Total: ~8 cycles vs __shfl_xor loop's ~12.
//
// Opt-out via CANDLE_DPP_OFF=1 at kernel-compile-time by passing
// -DCANDLE_DPP_OFF=1. Defaults to DPP-enabled after the 2026-04-13 port.

#ifndef CANDLE_DPP_OFF

static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    x = gfx906_dpp_add_xor1(x);
    x = gfx906_dpp_add_xor2(x);
    x = x + gfx906_shuffle_xor4(x);
    x = gfx906_dpp_add_ror8(x);
    x = x + gfx906_swizzle_xor16(x);
    x = x + __shfl_xor(x, 32, 64);
    return x;
}

static __device__ __forceinline__ float gfx906_warp_reduce_max(float x) {
    x = gfx906_dpp_max_xor1(x);
    x = gfx906_dpp_max_xor2(x);
    x = fmaxf(x, gfx906_shuffle_xor4(x));
    x = gfx906_dpp_max_ror8(x);
    x = fmaxf(x, gfx906_swizzle_xor16(x));
    x = fmaxf(x, __shfl_xor(x, 32, 64));
    return x;
}

// Half-warp (32-wide) — stops at xor-16, skips cross-half.
static __device__ __forceinline__ float gfx906_half_warp_reduce_sum_dpp(float x) {
    x = gfx906_dpp_add_xor1(x);
    x = gfx906_dpp_add_xor2(x);
    x = x + gfx906_shuffle_xor4(x);
    x = gfx906_dpp_add_ror8(x);
    x = x + gfx906_swizzle_xor16(x);
    return x;
}

static __device__ __forceinline__ float gfx906_half_warp_reduce_max_dpp(float x) {
    x = gfx906_dpp_max_xor1(x);
    x = gfx906_dpp_max_xor2(x);
    x = fmaxf(x, gfx906_shuffle_xor4(x));
    x = gfx906_dpp_max_ror8(x);
    x = fmaxf(x, gfx906_swizzle_xor16(x));
    return x;
}

#else // CANDLE_DPP_OFF — portable fallback path

static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}

static __device__ __forceinline__ float gfx906_warp_reduce_max(float x) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor(x, mask));
    }
    return x;
}

static __device__ __forceinline__ float gfx906_half_warp_reduce_sum_dpp(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

static __device__ __forceinline__ float gfx906_half_warp_reduce_max_dpp(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, 32));
    }
    return x;
}

#endif // CANDLE_DPP_OFF

// --- float2 warp reduction (used in layernorm for mean+variance) ---

static __device__ __forceinline__ float2 gfx906_warp_reduce_sum(float2 a) {
    a.x = gfx906_warp_reduce_sum(a.x);
    a.y = gfx906_warp_reduce_sum(a.y);
    return a;
}

// ============================================================================
// Section 2: Hardware SFU Intrinsics
// ============================================================================
//
// gfx906 has a Special Function Unit (SFU) that computes transcendentals in a
// single cycle with ~23 bits of precision. The compiler's __expf/__logf emit
// multi-instruction software sequences (~8 VALU ops for exp, ~6 for log).
//
// v_exp_f32: computes 2^x in 1 cycle (SFU)
// v_log_f32: computes log2(x) in 1 cycle (SFU)
// v_rcp_f32: computes 1/x in 1 cycle (SFU, ~23-bit precision)
//
// For natural exp/log, we convert bases:
//   exp(x) = 2^(x * log2(e))
//   log(x) = log2(x) * ln(2)

// log2(e) = 1.4426950408889634
#define GFX906_LOG2E  1.4426950408889634f
// ln(2) = 0.6931471805599453
#define GFX906_LN2    0.6931471805599453f

static __device__ __forceinline__ float gfx906_exp2(float x) {
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

static __device__ __forceinline__ float gfx906_log2(float x) {
    float r;
    asm volatile("v_log_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

static __device__ __forceinline__ float gfx906_rcp(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// exp(x) = 2^(x * log2(e))  — 1 FMA + 1 SFU cycle
static __device__ __forceinline__ float gfx906_fast_exp(float x) {
    return gfx906_exp2(x * GFX906_LOG2E);
}

// log(x) = log2(x) * ln(2)  — 1 SFU + 1 MUL cycle
static __device__ __forceinline__ float gfx906_fast_log(float x) {
    return gfx906_log2(x) * GFX906_LN2;
}

// tanh(x) via hardware SFU:
//   tanh(x) = 1 - 2 / (exp(2x) + 1)
// For |x| > 10, saturate to ±1 to avoid overflow.
// Total: 1 FMA + 1 v_exp_f32 + 1 v_add + 1 v_rcp_f32 + 1 FMA = ~5 cycles
// vs software tanhf() which is ~20 instructions.
static __device__ __forceinline__ float gfx906_fast_tanh(float x) {
    // Saturation for large |x| — avoids exp overflow
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = gfx906_exp2(2.0f * x * GFX906_LOG2E);
    return 1.0f - 2.0f * gfx906_rcp(e2x + 1.0f);
}

// ============================================================================
// Section 3: Scalar Broadcast (readfirstlane)
// ============================================================================
//
// When a value is uniform across all lanes (e.g., quantization scale `d` loaded
// by lane 0), broadcasting it to an SGPR saves a VGPR and avoids redundant
// per-lane global loads. On gfx906, `v_readfirstlane_b32` moves a VGPR value
// from lane 0 to an SGPR, accessible by all lanes.

static __device__ __forceinline__ float gfx906_sgpr_broadcast_f32(float x) {
    // Reinterpret float as int for readfirstlane, then back to float.
    int i = __float_as_int(x);
    i = __builtin_amdgcn_readfirstlane(i);
    return __int_as_float(i);
}

static __device__ __forceinline__ int gfx906_sgpr_broadcast_i32(int x) {
    return __builtin_amdgcn_readfirstlane(x);
}

#else // !__HIP_PLATFORM_AMD__

// ============================================================================
// Fallback: non-AMD platforms use generic __shfl_xor implementations
// ============================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}

static __device__ __forceinline__ float gfx906_warp_reduce_max(float x) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor(x, mask));
    }
    return x;
}

static __device__ __forceinline__ float2 gfx906_warp_reduce_sum(float2 a) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x += __shfl_xor(a.x, mask);
        a.y += __shfl_xor(a.y, mask);
    }
    return a;
}

static __device__ __forceinline__ float gfx906_half_warp_reduce_sum_dpp(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

static __device__ __forceinline__ float gfx906_half_warp_reduce_max_dpp(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, 32));
    }
    return x;
}

static __device__ __forceinline__ float gfx906_fast_exp(float x) { return expf(x); }
static __device__ __forceinline__ float gfx906_fast_log(float x) { return logf(x); }
static __device__ __forceinline__ float gfx906_fast_tanh(float x) { return tanhf(x); }
static __device__ __forceinline__ float gfx906_exp2(float x) { return exp2f(x); }
static __device__ __forceinline__ float gfx906_log2(float x) { return log2f(x); }
static __device__ __forceinline__ float gfx906_rcp(float x) { return 1.0f / x; }

static __device__ __forceinline__ float gfx906_sgpr_broadcast_f32(float x) { return x; }
static __device__ __forceinline__ int gfx906_sgpr_broadcast_i32(int x) { return x; }

#endif // __HIP_PLATFORM_AMD__
