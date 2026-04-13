# Candle vs llamacpp-turbo: HIP Kernel Comparative Review & Optimization Roadmap

**Date:** 2026-04-12
**Target hardware:** AMD MI50 (gfx906), 3x16 GB HBM2, ~1 TB/s, ROCm 7.1.1
**Goal:** Produce a better-optimized framework than llamacpp-turbo for GGUF inference on gfx906

**Source data:**
- `BENCH-3WAY-POST-P2-2026-04-11.md` — post-Q2 baseline, rocprofv3 per-kernel category rollups
- `BENCH-PMC-VALU-VMEM-2026-04-11.md` — hardware PMC proof that MMQ is compute-bound
- `BENCH-Q3-FLASH-ATTN-v2-2026-04-11.md` — flash-attn v2 post-mortem
- `BENCH-P1-PHASE-2D-V2F-2026-04-11.md` — v2f bounds-check elimination results
- `~/.claude/projects/-artefact/memory/project_perf_vs_turbo.md` — measured gaps
- `~/.claude/projects/-artefact/memory/project_roadmap_3way.md` — existing Q0-Q6 roadmap

---

## 1. Current State (2026-04-11 post-Q0/Q1/Q2, both sides with warmup)

| Model | Mode | Candle | Turbo | Candle/Turbo |
|---|---|---:|---:|---:|
| TinyLlama 1.1B Q4_0 | prefill | 3441 t/s | 6045 t/s | **57%** |
| TinyLlama 1.1B Q4_0 | decode | 164 t/s | 212 t/s | **77%** |
| gemma4 E4B Q4_0 | prefill | 884 t/s | 1217 t/s | **73%** |
| gemma4 E4B Q4_0 | decode | 35.5 t/s | 69.5 t/s | **51%** |
| qwen3.5-9B Q4_1 | prefill | 450 t/s | 1164 t/s | **39%** |
| qwen3.5-9B Q4_1 | decode | 40.6 t/s | 64.5 t/s | **63%** |

Candle started at 7-15% of turbo pre-P0 and has reached 39-77% through a series of
kernel-level and system-level optimizations (fused GDN, MMQ v2f, gqa_attention
zero-copy, hipMallocAsync, warmup-normalized measurement). This review identifies
every remaining kernel-level gap, its root cause, and the path to closing — and
surpassing — turbo.

---

## 2. Kernel-by-Kernel Comparative Analysis

### 2.1 Warp Reduction Primitives — the single most pervasive gap

**What candle does:**

Every warp reduction in candle — across `quantized.cu`, `reduce.cu`, `flash_attn.cu`,
`flash_attn_v2.cu`, `gated_delta_net.cu` — uses a generic `__shfl_xor` loop:

```cpp
// candle-hip-kernels/src/quantized.cu:31-36
// candle-hip-kernels/src/reduce.cu:164-170
// candle-hip-kernels/src/flash_attn.cu:47-53
// candle-hip-kernels/src/gated_delta_net.cu:53-59
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        x += __shfl_xor(x, mask);
    }
    return x;
}
```

On gfx906, `__shfl_xor` compiles to a `v_readlane_b32` → scalar register → 
`v_writelane_b32` round-trip. Each stage costs ~2 VALU cycles. For WARP_SIZE=64
that is 6 stages × 2 cycles = **~12 cycles per reduction**.

**What turbo does:**

```cpp
// llamacpp-turbo/.../gfx906/gfx906-common.cuh:72-141
DEFINE_FUSED_DPP_F32(hip_add_xor1_f32, "s_nop 4\n",
    "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_add_xor2_f32, "s_nop 1\n",
    "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_add_xor8_f32, "s_nop 1\n",
    "row_ror:8",           "v_add_f32_dpp")

template<int width, typename Op>
static __device__ __forceinline__ float warp_reduce_amd_f32(float x) {
    if (width >= 2)  x = Op::xor1(x);              // DPP quad_perm
    if (width >= 4)  x = Op::xor2(x);              // DPP quad_perm
    if (width >= 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));  // DPP row_shl/shr
    if (width >= 16) x = Op::xor8(x);              // DPP row_ror
    if (width >= 32) x = Op::apply(x, hip_shuffle_xor16_f32(x)); // ds_swizzle
    if (width == 64) x = Op::apply(x, __shfl_xor(x, 32, 64));   // cross-half
    return x;
}
```

DPP (Data-Parallel Primitive) instructions fuse the data movement *into* the ALU
operation. `v_add_f32_dpp ... quad_perm:[1,0,3,2]` does the xor-1 reduction in
**one cycle with zero data-movement overhead** — the permutation happens in the
register file's read ports. The `ds_swizzle_b32` for xor-16 uses the LDS crossbar
hardware without actually touching LDS memory. Total: **~7 cycles** for a
64-wide reduction vs candle's ~12.

**Where this hits:**

| Kernel | Call site | Dispatches (gemma4) | Reductions per dispatch |
|---|---|---:|---:|
| MMVQ decode | `gfx906_half_warp_reduce_sum` at quantized.cu:2608 | 13,672 | 1 per row |
| Softmax | `warp_reduce_sum` + `warp_reduce_max` in reduce.cu:164/333/393 | 2,688 | 2 per row |
| RMSNorm/LayerNorm | `warp_reduce_sum` in reduce.cu:188/255 | 17,792 | 1-2 per row |
| Flash attn v1/v2 | `flash_warp_reduce_sum_f32` at flash_attn.cu:47 | per-K-token | 1 per K |
| GDN step | `warp_reduce_sum_f32` at gated_delta_net.cu:53 | 1,536 | 2 per head-dim col |

The half-warp MMVQ reduce (`quantized.cu:2608`) already uses `width=32` (5 stages
→ ~10 cycles). DPP would cut this to ~5 cycles. Across 13,672 MMVQ dispatches on
gemma4-E4B, that's ~68k wasted cycles per run, concentrated on the decode critical
path.

**Verdict:** ~1.7x per reduction. Not a single-kernel gap but a **tax on every
kernel that touches a warp reduction**. Porting turbo's DPP primitives is ~80 LOC
(the `gfx906-common.cuh` file plus callsite changes) and improves every kernel in
the codebase.

---

### 2.2 MMVQ (Decode Mat-Vec) — candle is architecturally close, details differ

**Architecture comparison:**

Both frameworks use the same warp-cooperative MMVQ design for gfx906:
- 64 threads per block, 2 rows per block, half-warp (32 threads) per row
- Each thread handles one Q-block every 32 blocks (stride-32 over K)
- 8 dp4a per 32-element Q4_0/Q4_1 block
- Half-warp reduction to produce the final dot product

Candle ported this from turbo (see `quantized.cu:2600-2606` reference comment).

**Per-aspect comparison:**

| Aspect | Candle | Turbo | File references |
|---|---|---|---|
| dp4a intrinsic | `__builtin_amdgcn_sdot4` via `ggml_cuda_dp4a` | Same | candle quantized.cu:107-112 vs turbo common.cuh |
| Memory loads | `memcpy(&v0, bq4->qs + 0, 4)` — 4-byte scalar loads | `memcpy` + `gfx906_load_int2` (8-byte) + `int4` (16-byte) | candle quantized.cu:2644-2647 vs turbo mmq.cuh:16-21 |
| Warp reduction | `gfx906_half_warp_reduce_sum<32>` via `__shfl_xor` | DPP-fused `warp_reduce_amd_f32<32, AddOp>` | candle quantized.cu:2608-2616 vs turbo gfx906-common.cuh:133 |
| Scalar broadcast | Not used — uniform values (block index, stride) recomputed per lane | `sgpr_broadcast_f32(__builtin_amdgcn_readfirstlane)` | turbo gfx906-common.cuh:6-9 |
| Launch bounds | `__launch_bounds__(64, 1)` on the gfx906 MMVQ variants | `__launch_bounds__(64, 1)` | candle quantized.cu:2619 vs turbo mmvq-q4_0.cuh:9 |
| MXFP4 support | Not implemented | `v_perm_b32` table lookup: `__builtin_amdgcn_perm(mags32[1], mags32[0], sel)` | turbo vecdotq.cuh:44 |
| Formats covered | Q4_0, Q4_1, Q8_0 (warp-coop); others legacy 256-thread | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 + TURBO2/3/4 + MXFP4 | candle quantized.cu:2619-2776 |

**Key gaps:**

1. **Vectorized loads.** Turbo uses 128-bit `int4` loads for the Q8_1 Y-side
   (`mmq.cuh:16-21`). The Q8_1 block's `qs` array is 32 bytes = 8 ints; an `int4`
   load pulls 4 ints in one `global_load_dwordx4` instruction vs candle's 4 separate
   `flat_load_dword` instructions. On gfx906's 64-byte cache lines this is the
   difference between 1 and 4 memory instructions per K-block on the Y side.

2. **DPP reduction** — covered in §2.1. The MMVQ half-warp reduction is the last
   operation before the final store; any extra cycles here directly delay the output.

3. **K-quant coverage.** Q2_K through Q6_K fall back to candle's legacy
   `mul_mat_vec_q<1, QK_K, QR_K, ...>` template that uses 256 threads per row and
   is launched once per row. The warp-cooperative design is never invoked for these
   types.

4. **Missing formats.** Turbo supports TURBO2_0/TURBO3_0/TURBO4_0 (custom 2-3 bit
   centroid quantization) and MXFP4. These are turbo-specific formats that candle
   doesn't need to support, but the MXFP4 table-lookup technique using `v_perm_b32`
   is a useful pattern for any future sub-4-bit format.

---

### 2.3 MMQ (Prefill Mat-Mat) — compute-bound, 91% of turbo per-call

**Architecture comparison:**

| Aspect | Candle v2f | Turbo |
|---|---|---|
| Tile layout | 64×TILE_N (WARP_SIZE rows × N cols), 1 warp per block | 2-warp MMA-style tiles with LDS staging |
| LDS usage | **None** — pure register accumulation | Async tile load into LDS → MMA consumption |
| K-loop structure | Each thread owns 1 X row, iterates all K blocks serially | Warp-cooperative K-tile load + compute pipeline |
| Bounds check (v2f) | Removed — Y padded to `ceil(total_b/TILE_N)*TILE_N` | Same padding approach |
| Per-call throughput | 91% of turbo (measured, post-v2f) | Baseline |
| VALU utilization | 49% (PMC-confirmed, dead flat across 128 dispatches) | ~80%+ (inferred from 2-warp pipelining) |
| Memory utilization | 0.015% of HBM peak (152 MB/s of 1 TB/s) | Comparable (both compute-bound) |

**PMC analysis** (`BENCH-PMC-VALU-VMEM-2026-04-11.md`):

The v2f kernel is **definitively compute-bound**:
- VALUBusy: 69.64% — VALU units are busy 70% of the time
- MemUnitBusy: 24.81% — memory unit only 25% active
- MemUnitStalled: 0.15% — essentially zero memory stalls
- FetchSize: 2.42 MB/call at 15.8 ms → 152 MB/s (0.015% of peak)
- LDSInsts: 0 — no LDS usage at all

The 49% VALUUtilization (dead flat) means only ~32 of 64 VALU lanes are doing
useful work per instruction. Pre-v2f, 27% of instructions were lane-scratch spills
from the compiler's bounds-check implementation — the v2f fix eliminated those
entirely (`BENCH-P1-PHASE-2D-V2F-2026-04-11.md`):

| Kernel variant | Lane moves (spills) | Per-call speedup from v2f |
|---|---:|---:|
| Q4_0 tile32: v2 → v2f | 574 (26%) → 0 | -26% time |
| Q4_1 tile32: v2 → v2f | 574 (26%) → 0 | -26% time |
| Q8_0 tile32: v2 → v2f | 574 (26%) → 0 | -15% time |
| Q5_K tile32: v2 → v2f | — → 0 | -37% time |

**What the remaining 9% gap is:**

Turbo's remaining per-call advantage comes from better VALU scheduling, not memory:

1. **Split accumulators.** Candle's `sums[c] += scale_x * (sumi * d8 - 8.0f * s8)`
   is a serial FMA chain — each iteration depends on the previous one. Turbo uses
   multiple independent accumulator variables that accumulate different K sub-ranges
   and are combined at the end. This increases instruction-level parallelism within
   the same wavefront.

2. **K-loop pipelining.** Turbo's 2-warp design overlaps next-tile load with
   current-tile compute. The `GFX906_LOAD_TILES_Q8_0_ASYNC` macro
   (`turbo mmq.cuh:54-68`) loads the next cache batch while the current batch is
   being consumed. Candle's single-warp design has a hard serialization between load
   and compute.

3. **Better inner-loop codegen.** Turbo's 2-warp MMQ with 2 nwarps
   (`GFX906_MMQ_NWARPS=2` in `gfx906-config.h:10`) gives the compiler more
   registers to work with per wave. Candle's 8-warp legacy tile
   (`NWARPS_Q4_0_GFX906=8` at `quantized.cu:123`) has much higher register pressure
   per CU.

**What would NOT help** (already disproved by PMC):
- LDS staging for X or Y tiles — memory is not the bottleneck
- Async prefetch — memory is not stalling
- Cross-warp data sharing — the lever for sharing is memory amortization, irrelevant here
- Larger tiles — register pressure would increase, VALU utilization would drop further

---

### 2.4 Flash Attention — candle's single biggest architectural gap

**Current state:**

Candle has two flash-attention implementations. Both **lose** to the rocBLAS fallback
on every benchable shape:

| Variant | Design | Per-call vs rocBLAS | Status |
|---|---|---|---|
| v1 (`flash_attn.cu:56`) | BR=1, 1 warp (Wave64), scalar j-loop | **17x slower** | Env-gated off |
| v2 (`flash_attn_v2.cu`) | BR=4, 4 warps, LDS-tiled K/V | **11-77% slower** | Env-gated off |

Both are disabled by default (require `CANDLE_FLASH_ATTN_V{1,2}_ENABLE=1`).

The v1 failure mode (`BENCH-P2-FLASH-ATTN-v1-2026-04-11.md`):
- Per-call cost: 27.6 ms vs rocBLAS 1.6 ms
- Root cause: scalar j-loop with per-j `flash_warp_reduce_sum_f32` — a 6-shuffle
  warp reduce + 2×`__expf` critical path **per K token**, with no K-tile
  amortization

The v2 failure mode (`BENCH-Q3-FLASH-ATTN-v2-2026-04-11.md`):
- 4 warps share K/V tiles through LDS, but the inner loop is still scalar per-K
- The warp reduce per j is a hard critical path (~30 cycles)
- rocBLAS uses dense FMA unrolling without per-j reduction

**What turbo does:**

Three attention variants, each tuned for different shapes:

```
// turbo: fattn-vec.cuh — 128 threads, vectorized, quantized KV support
__launch_bounds__(128, 1)
flash_attn_ext_vec<D, ncols, type_K, type_V, use_logit_softcap>()

// turbo: fattn-tile.cuh — 256 threads, tile-based, per-arch config table
ggml_cuda_fattn_tile_get_config_amd(DKQ=64, DV=64, ncols=16):
  => (nthreads=256, occupancy=2, nbatch_fa=128, nbatch_K=64)

// turbo: fattn-wmma-f16.cuh — matrix-multiply (WMMA/MMA, not on gfx906)
```

Key turbo advantages:
1. **Quantized KV support.** `type_K` and `type_V` accept Q4_0, Q4_1, Q5_0, Q5_1,
   Q8_0, TURBO2/3/4 — dequantize on-the-fly, never materializing f32 K/V.
2. **Sliding window.** Built into `fattn-tile.cuh` with configurable batch sizes;
   on gemma4 this avoids the window-splitting that creates 10,816 rocBLAS dispatches.
3. **Per-arch config table.** `ggml_cuda_fattn_tile_get_config_amd()` returns tuned
   (nthreads, occupancy, nbatch_fa, nbatch_K) for each (DKQ, DV, ncols) combination.
4. **40+ template instances.** Head dims 40, 64, 72, 80, 96, 112, 128, 192, 256,
   512, 576 are all pre-compiled.

**Why this is candle's #1 gap:**

On gemma4-E4B the attention category (`rocblas_attn` + `softmax` + `mask_pw`)
consumes **~35% of total GPU time** with **10,816 + 2,688 + 24,519 = 38,023
dispatches** (`BENCH-3WAY-POST-P2-2026-04-11.md`). This is because gemma4 uses
sliding-window attention, and without a fused kernel, each window-slice becomes a
separate rocBLAS `Cijk_*` dispatch + separate softmax + separate mask application.

A working flash-attention would collapse these into ~48 dispatches per forward
(1 per layer) and eliminate the entire `copy_ucopy` cost for attention Q/K/V
transpositions (currently 14% of GPU time on both TinyLlama and gemma4).

---

### 2.5 Graph-Level Fusion — turbo has automated pattern detection

**Candle's fusion approach:** Manual, per-model. Each fusion is hand-coded into the
model-specific forward function (e.g., `quantized_gemma4.rs`, `quantized_qwen35.rs`).
Landed fusions:

| Fusion | Kernel | Landed in | Where |
|---|---|---|---|
| RMSNorm + residual | `rmsnorm_post_residual_f32` | Q0a | reduce.cu |
| Scale + mask + softmax | `masked_softmax_scale_f32` | Q0c | reduce.cu |
| SiLU × up-proj | `silu_mul_split_last_f32` | P3 | reduce.cu |
| GDN step | `gated_delta_net_step_s128_f32` | P0 | gated_delta_net.cu |
| Skip alloc_zeros | Y/dst padding only when needed | Q0b | hip.rs |

**Turbo's fusion approach:** Automated graph analysis. `graph-fusion.cuh` scans the
GGML compute graph at execution time:

```cpp
// turbo: gfx906/fused/graph-fusion.cuh:32-99
static void analyze_graph_for_fusion(ggml_cgraph* cgraph, int cc) {
    // Build consumer map: tensor -> list of (consumer_node, consumer_idx)
    // Scan for: RMS_NORM -> MUL -> multiple MUL_MAT consumers
    // All consumers must be MMQ-eligible
    // Cache decisions per-graph (invalidate on graph change)
}
```

When the pattern matches, turbo fuses three operations into one:
1. `RMS_NORM(x)` — normalize activations
2. `MUL(norm, weight)` — apply scale
3. `quantize_q8_1(scaled)` — quantize to Q8_1 for downstream MMQ

The Q8_1 result is written to a **per-layer arena cache** (`q8-cache.cuh`) and
reused by all downstream MUL_MAT consumers. On a layer with 3 matmul consumers
(e.g., Q/K/V projections), this saves:
- 2 extra HBM round-trips (norm and scale write-backs that are now fused)
- 2 extra `quantize_q8_1` dispatches (the Q8 is produced once, consumed 3x)
- The `quantize_q8` category is 74 ms / 13,888 dispatches on gemma4 — saving 2/3
  of those = ~50 ms + ~9k fewer dispatches

**What candle does NOT have:**
- Automated fusion pattern detection
- Fused norm + scale + quantize as a single kernel
- Q8 activation cache / arena allocator
- Layer-cycled buffer reuse

**However:** Candle has a structural advantage for fusion. The model-level forward
functions are written in Rust, not assembled from a flat C graph. The compiler can
see the entire layer structure at build time. Rather than runtime graph analysis,
candle can implement fusion at the model level with compile-time guarantees — which
is what the landed Q0a/Q0c/P3 fusions already do. The path forward is to extend
this pattern systematically, not to replicate turbo's runtime graph walker.

---

### 2.6 AMD ISA Utilization — candle treats gfx906 as "CUDA with WARP_SIZE=64"

| ISA Feature | Candle usage | Turbo usage | Impact |
|---|---|---|---|
| **DPP instructions** | Not used anywhere | Extensive in all hot paths: `v_add_f32_dpp`, `v_max_f32_dpp` with `quad_perm`, `row_ror`, `row_shl/shr` | **High** — every reduction |
| **ds_swizzle_b32** | Not used | xor-16 cross-lane without LDS bank hit | **Medium** — part of warp reduce |
| **readfirstlane** | Not used | `sgpr_broadcast_f32/i32/f16` for uniform values | **Medium** — saves redundant VALU work |
| **v_exp_f32** (SFU) | `__expf()` → libm multi-instruction sequence | `asm("v_exp_f32")` — 1-cycle hardware SFU | **High** — in softmax, flash-attn |
| **v_log_f32** (SFU) | `__logf()` → libm | `asm("v_log_f32")` — 1-cycle | **Medium** |
| **v_rcp_f32** (SFU) | `1.0f / x` → `v_rcp_f32` + Newton-Raphson | `asm("v_rcp_f32")` — 1-cycle, accepts reduced precision | **Medium** |
| **v_perm_b32** | Not used | MXFP4 centroid table lookup in 1 cycle via `__builtin_amdgcn_perm` | **Low** (MXFP4 only) |

Candle's kernels are written as "portable CUDA/HIP" — they compile on both
platforms using only the HIP compatibility layer (`__shfl_xor`, `__half2float`,
standard math functions). Turbo's gfx906 kernels are written as **native GCN
assembly wrappers** that extract maximum throughput from the specific ISA.

The `v_exp_f32` gap is particularly significant in softmax and attention. On
gfx906, the hardware SFU (Special Function Unit) computes exp2 in a single cycle
with ~23 bits of precision. The `__expf()` fallback compiles to a multi-instruction
range-reduction + polynomial + reconstruction sequence. In a softmax kernel that
calls exp once per element per row, this is the difference between 1 and ~8
instructions on the critical path.

Turbo also uses `fast_tanh_f32` (`gfx906-common.cuh:54-59`) which chains two
hardware SFU calls (`v_exp_f32` for `exp(2x)`, then an add + `v_rcp_f32`)
instead of the libm `tanhf()` which is a ~20-instruction software implementation.

---

### 2.7 Dispatch Overhead & Memory Allocation

**Post-Q0 dispatch comparison** (gemma4-E4B, 1 prefill + 63 decode):

| Category | Candle dispatches | Turbo dispatches | Ratio |
|---|---:|---:|---:|
| Total | ~66,000 | ~12,000 | **5.5x** |
| rocblas_attn | 10,816 | ~96 (fused FA) | **113x** |
| copy_ucopy | 8,664 | ~500 | **17x** |
| alloc (fillBufferAligned) | 18,314 (post-Q0b) | ~200 | **92x** |
| quantize_q8 | 13,888 | ~4,000 (w/ cache) | **3.5x** |
| mmvq_decode | 13,672 | ~13,000 | **~1x** |

Candle's Q0b eliminated 60% of `alloc` dispatches (45,874 → 18,314) by skipping
`alloc_zeros` when the downstream kernel fully overwrites the buffer. The remaining
18k are from:
- Sliding-window attention masks (new per layer per decode step)
- `.contiguous()` materializations that turbo avoids via fused kernels

**Measured launch overhead** (`project_perf_vs_turbo.md`):
- Real per-launch cost: **3-5 us** on ROCm 7.1.1/gfx906
- Total launch cost per qwen3.5-9B decode token: ~5 ms of ~32 ms total
- Even perfect HIP graph replay would save ~3 ms/token (~10%, not 2x)

HIP graphs were investigated and found **counter-productive** — `hipMallocAsync`
graph nodes serialize on the runtime pool lock during replay, making graph replay
7x *slower* than fresh launches. The path to dispatch reduction is kernel fusion,
not graph capture.

---

### 2.8 Memory Access Patterns

**Vectorized loads:**

Turbo consistently uses the widest load available:

```cpp
// turbo: gfx906/matmul/mmq.cuh:16-21
const int4 vec0 = *((const int4 *) &y_qs[base_addr]);      // 128-bit load
const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);  // 128-bit load

// turbo: gfx906/quantize/vecdotq.cuh:24-28
static __device__ int2 gfx906_load_int2(const void* x, const int& i32) {
    int2 x64;
    memcpy(&x64, (const uint8_t*)x + 4*i32, 8);  // 64-bit load
    return x64;
}
```

Candle's MMVQ and MMQ kernels use 32-bit loads:

```cpp
// candle: quantized.cu:2644-2647
memcpy(&v0, bq4->qs +  0, 4);  // 32-bit load
memcpy(&v1, bq4->qs +  4, 4);  // 32-bit load
memcpy(&v2, bq4->qs +  8, 4);  // 32-bit load
memcpy(&v3, bq4->qs + 12, 4);  // 32-bit load
```

On gfx906, `global_load_dwordx4` (128-bit) has the same latency as
`global_load_dword` (32-bit) but 4x the throughput. When the address is 16-byte
aligned — which it is for Q8_1's `qs` array (36-byte struct with 4-byte `ds` +
32-byte `qs`) — switching to `int4` loads cuts memory instructions by 4x on the
Y-side of every MMVQ/MMQ kernel.

**LDS bank conflicts:**

Candle's legacy MMQ path (the template-based `mul_mat_q<>` at `quantized.cu:122-160`)
uses 8 warps with `MMQ_X=64, MMQ_Y=64` tiles:

```cpp
// candle: quantized.cu:122-124
#define  MMQ_X_Q4_0_GFX906  64
#define  MMQ_Y_Q4_0_GFX906  64
#define NWARPS_Q4_0_GFX906  8
```

8 warps × 64 threads = 512 threads per block. With 64-KB LDS per CU on gfx906, this
limits occupancy to 1 block per CU. Turbo uses 2 warps (`GFX906_MMQ_NWARPS=2` at
`gfx906-config.h:10`) — 128 threads per block, allowing 4x more concurrent blocks
and better latency hiding.

Candle's v2f kernel avoids this entirely by not using LDS at all, but the legacy path
(still used for K-quants and `b*m <= 8`) pays this occupancy penalty.

---

## 3. Optimization Roadmap: Path to Surpassing Turbo

### Design philosophy

Turbo's advantage is not any single brilliant kernel — it is the **systematic
accumulation** of:
1. ISA-native primitives (DPP, SFU, readfirstlane)
2. Wider memory loads (int4/int2 everywhere)
3. Working flash attention (fused, sliding-window, quantized-KV)
4. Automated graph fusion (norm + scale + quantize + cache)

Candle's structural advantages — Rust ownership model, model-level fusion,
zero Python overhead, compile-time graph visibility — are currently
**unrealized**. The roadmap below first closes the ISA-level gaps (low-effort,
high-compound-return), then builds the system-level features that turbo cannot
replicate.

---

### Phase A: ISA-Native Primitives (~150 LOC kernel, ~80 LOC integration)

**Rationale:** Every kernel in the codebase pays the `__shfl_xor` and `__expf` tax.
Fixing the primitives once improves everything simultaneously — MMVQ, MMQ, softmax,
layernorm, rmsnorm, GDN, flash-attn. This is the highest-ROI-per-LOC change possible.

#### A1. DPP Warp Reductions

Create `candle-hip-kernels/src/gfx906_primitives.cuh` with turbo-style DPP macros:

```cpp
// Target: replace every warp_reduce_{sum,max} with DPP versions on gfx906
// Pattern: fused v_add/v_max_f32_dpp with quad_perm/row_ror permutations
// Also support half-warp (width=32) for MMVQ
```

Call sites to update:
- `quantized.cu:31-36` — the main `warp_reduce_sum` (used by legacy MMQ)
- `quantized.cu:38-44` — `warp_reduce_max` (used by legacy MMQ)
- `quantized.cu:2608-2616` — `gfx906_half_warp_reduce_sum` (MMVQ hot path)
- `reduce.cu:155-170` — `warp_reduce_sum` float and float2 (softmax, norm)
- `flash_attn.cu:47-53` — `flash_warp_reduce_sum_f32`
- `flash_attn_v2.cu:43-49` — `v2_warp_reduce_sum`
- `gated_delta_net.cu:53-59` — `warp_reduce_sum_f32`

Expected per-reduction speedup: ~1.7x (12 cycles → 7 cycles for W64, 10 → 5 for W32).

Compound impact across a full gemma4-E4B run:
- MMVQ: 13,672 dispatches × 2 reductions × 5 cycle savings = ~137k saved cycles
- Softmax: 2,688 dispatches × 2 reductions × 5 cycle savings = ~27k saved cycles
- RMSNorm: 17,792 dispatches × 1 reduction × 5 cycle savings = ~89k saved cycles
- Total: ~253k cycles → measurable on the decode critical path

#### A2. Hardware SFU Intrinsics

Add to `gfx906_primitives.cuh`:

```cpp
// fast_exp_f32: asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(x * LOG2_E))
// fast_log2_f32: asm("v_log_f32 %0, %1" : "=v"(r) : "v"(x))
// fast_rcp_f32: asm("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x))
// fast_tanh_f32: if(|x|>10) ±1, else 1 - 2*rcp(exp(2x) + 1)
```

Call sites: `reduce.cu` softmax (exp per element), `flash_attn*.cu` (exp per K
token), `gated_delta_net.cu` (potential tanh/sigmoid in gate). Each `__expf` → 
`v_exp_f32` saves ~7 instructions on the critical path.

#### A3. Scalar Broadcast for Uniform Values

In MMVQ kernels, values like `bq4->d` (the quantization scale) are the same across
all lanes in the half-warp but currently loaded per-lane. Using
`sgpr_broadcast_f32(__builtin_amdgcn_readfirstlane(__float_as_int(bq4->d)))` moves
the value to a scalar register (SGPR), freeing a VGPR and saving the redundant
loads.

Apply to: `quantized.cu:2649` (`scale_x`), `quantized.cu:4949` (`scale_x` in MMQ),
and any other per-block scale factor in MMVQ/MMQ kernels.

#### A4. `__launch_bounds__` Audit

Candle's gfx906 MMVQ variants already have `__launch_bounds__(64, 1)` (good).
Check and add to:
- All MMQ v2/v2f variants (`quantized.cu:4893-4991`) — currently missing
- Flash attention variants
- GDN step kernel
- All reduce.cu kernels launched with known block sizes

**Phase A estimated total impact:** +8-15% decode, +3-5% prefill across all models.
The DPP reduction is the biggest single item; the others compound.

---

### Phase B: MMVQ Decode Optimization (~400 LOC)

**Rationale:** Decode is dominated by MMVQ kernels (10-22% of GPU time across
models). Candle is at 51-77% of turbo on decode. Phase A's DPP fix plus Phase B's
memory and ILP improvements target closing decode to 85-95% of turbo.

#### B1. 128-bit Vectorized Loads for Q8_1 Y-Side

The Q8_1 block is 36 bytes: `half2 ds` (4B) + `int8_t qs[32]` (32B). The `qs` array
starts at offset 4 within the block. Loading as `int4` (4 ints = 16 bytes) requires
16-byte alignment; since `qs` is at offset 4 from a 36-byte-aligned struct, alignment
must be verified per block. Where alignment holds, use:

```cpp
const int4 q8_lo = *((const int4 *)(bq8->qs));       // qs[0..15]
const int4 q8_hi = *((const int4 *)(bq8->qs + 16));  // qs[16..31]
```

Where it doesn't, fall back to `int2` (8-byte, always safe for 4-byte-aligned data):

```cpp
const int2 q8_0 = *((const int2 *)(bq8->qs));
const int2 q8_1 = *((const int2 *)(bq8->qs + 8));
const int2 q8_2 = *((const int2 *)(bq8->qs + 16));
const int2 q8_3 = *((const int2 *)(bq8->qs + 24));
```

Apply to all MMVQ and MMQ kernels that load Q8_1 Y-side data.

#### B2. Split-Accumulator ILP in MMVQ

Current MMVQ inner loop (`quantized.cu:2638-2684`):

```cpp
float sumf = 0.0f;
for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
    // ... dp4a ...
    sumf += d4 * (sumi * ds8.x - 8.0f * ds8.y);  // serial dependency
}
```

Split into two independent accumulators:

```cpp
float sumf_even = 0.0f, sumf_odd = 0.0f;
for (int ib = half_lane; ib < blocks_per_row; ib += 64) {
    // even block
    sumf_even += d4_even * (sumi_even * d8_even - 8.0f * s8_even);
    // odd block (ib + 32)
    if (ib + 32 < blocks_per_row) {
        sumf_odd += d4_odd * (sumi_odd * d8_odd - 8.0f * s8_odd);
    }
}
float sumf = sumf_even + sumf_odd;
```

This doubles the instruction-level parallelism within a single thread, letting the
VALU pipeline overlap independent FMA chains.

#### B3. K-Quant Warp-Cooperative MMVQ

Port the warp-cooperative MMVQ design to Q4_K, Q5_K, Q6_K. These are the most
popular K-quant formats in community GGUF files. The current fallback is a 256-thread
per-row legacy template that wastes 3/4 of threads when row count < 4.

The port follows the same pattern as the existing Q4_0/Q4_1/Q8_0 MMVQ: 2 rows per
block, half-warp per row, dp4a accumulation, half-warp DPP reduction. The only
difference is the dequantization formula per K-quant type.

**Phase B estimated total impact:** +10-20% decode, targeting 70-95% of turbo decode.

---

### Phase C: Flash Attention for gfx906 (~1500 LOC)

**Rationale:** This is candle's **single highest-impact optimization**. The attention
category is 35% of gemma4 GPU time and 37% of TinyLlama GPU time. Neither existing
flash-attention variant works (both lose to rocBLAS). A ground-up design for gfx906
is needed.

#### C1. Q-tile x K-tile GEMM-style Flash Attention

**Why v1/v2 failed:** Both use a scalar inner loop where each K token requires a full
warp reduction to produce a single dot product. The warp reduce is the bottleneck —
~12 cycles (7 with DPP) per K token, compared to rocBLAS which unrolls dense FMA
without per-element reduction.

**The correct gfx906 design:**

The key insight is that gfx906 has no WMMA/MMA instructions, so the flash-attention
inner block must be computed using dp4a and VALU FMA. The design should match the
Wave64 register file:

```
Configuration:
  BLOCK_M = 4     (Q rows per block)
  BLOCK_N = 16    (K columns per tile)
  D = 64 or 128   (head dimension)
  Block: 256 threads = 4 warps
  Each warp handles 1 Q row × 16 K columns

Inner loop (per K-tile of BLOCK_N columns):
  1. Load K[j:j+16, :] into LDS (16 × D floats, coalesced by 256 threads)
  2. __syncthreads()
  3. Each warp computes S[q_row, 0:16] = Q[q_row, :] · K[:, 0:16]^T
     - 64 lanes split as 4 sub-groups of 16, each sub-group handles D/4 elements
     - dp4a across D dimension, then 4-wide DPP reduction to combine sub-groups
     - Result: 16 dot products in ~D/4 dp4a + 2 DPP cycles
  4. Online softmax over the 16-wide score vector:
     - warp_reduce_max<16>(S) for row-max
     - exp(S - max) using v_exp_f32 SFU
     - warp_reduce_sum<16>(P) for normalization
     - Update running O accumulator: O = alpha*O + P·V[j:j+16, :]
  5. __syncthreads()
  6. Advance to next K-tile

Key difference from v1/v2: the warp reduction is over BLOCK_N=16 elements
(a single DPP chain, ~4 cycles), NOT over D=64/128 elements per single K token.
```

**Sliding window support:** Before loading K-tile j, check if `j` falls within the
attention window `[q_pos - window_size, q_pos]`. If entirely outside, skip the tile.
If partially outside, apply the mask to the 16-wide score vector before softmax.
This replaces gemma4's 10,816 rocBLAS window-split dispatches with ~48 fused kernel
launches.

**Quantized KV support (C2):** Accept Q8_0 or Q4_0 K/V directly. Dequantize on-the-fly
during the LDS load phase — each of 256 threads dequantizes one element, writing f32
to LDS. The dequantize cost is amortized over all BLOCK_M Q rows.

**Expected impact:**
- gemma4 prefill: attention 627ms → ~100ms, copy_ucopy 402ms → ~50ms (no transpose)
  = **~880ms saved → +60-80% prefill**
- gemma4 decode: attention + softmax + mask chain collapsed
  = **+30-50% decode**
- TinyLlama: attention 37% of GPU time → <10% = **+25-40% prefill, +15-25% decode**

This is the optimization that can push candle past turbo on gemma4, because candle's
flash-attention can be designed **natively for gfx906** with DPP reductions and
hardware SFU, while turbo's flash-attention was adapted from an NVIDIA-first design.

---

### Phase D: Activation Fusion & Caching (~800 LOC)

**Rationale:** After Phases A-C close the per-kernel gaps, the remaining delta is
system-level: redundant quantization passes, redundant HBM traffic for shared
activations, and dispatch overhead from separate norm/scale/quantize steps.

#### D1. Fused RMSNorm + Scale + Q8_1 Quantize Kernel

When a layer's pattern is `y = rmsnorm(x) * weight`, and `y` feeds into 2+
downstream MMQ consumers that each re-quantize `y` to Q8_1, fuse all three
operations:

```
Input:  x (f32, hidden_dim), weight (f32, hidden_dim), eps
Output: y_q8 (Q8_1 blocks), y_f32 (optional, for residual)

Algorithm:
  1. Compute norm: ||x||_rms = sqrt(mean(x²) + eps)
  2. For each block of 32 elements:
     a. scaled = x[i] / norm * weight[i]        // rmsnorm + scale
     b. d = max(|scaled[0..31]|) / 127           // Q8_1 scale
     c. qs[i] = round(scaled[i] / d)             // quantize
     d. ds = {d, sum(scaled[0..31])}              // Q8_1 metadata
  3. Write Q8_1 blocks to output buffer
```

One kernel launch replaces: 1× rmsnorm + 1× multiply + N× quantize_q8_1.

Files: new `candle-hip-kernels/src/fused_norm_quantize.cu`, integration in
`candle-core/src/hip_backend/`.

#### D2. Per-Layer Q8_1 Activation Cache

Pre-allocate a `LayerQ8Cache` buffer sized for `max_seq_len × hidden_dim / 32 × 
sizeof(block_q8_1)`. The fused D1 kernel writes into this cache. All downstream
MMQ consumers (Q, K, V projections and FFN up/gate) read from the same cached Q8_1
buffer instead of each independently quantizing the f32 activation.

On qwen3.5-9B with 3-4 matmul consumers per layer, this eliminates 2-3 `quantize_q8`
dispatches per layer × 28 layers × 64 decode steps = ~3,500-5,000 dispatches.

#### D3. Fused RoPE + Q/K Projection

Currently RoPE is a separate dispatch after each Q and K projection. Fusing it into
the projection output (apply RoPE during the Q8_1 dequantize phase of the subsequent
attention kernel, or as a post-MMQ fused op) saves 2 kernel launches per layer.

On TinyLlama: 2,816 RoPE dispatches × 8ms → ~2ms with fusion.

**Phase D estimated total impact:** +5-10% prefill, +3-5% decode on large models.

---

### Phase E: MMQ Prefill Last Mile (~300 LOC)

**Rationale:** The v2f kernel is at 91% of turbo per-call. The remaining 9% is
worth closing for completeness but has diminishing ROI relative to Phases A-D.

#### E1. Split Accumulators in MMQ v2f

Same principle as B2 but applied to the MMQ prefill kernel. The
`sums[c] += scale_x * (sumi * d8 - 8.0f * s8)` chain at `quantized.cu:4973` is
serial. Split into `sums_even[c]` and `sums_odd[c]` over alternating K-blocks.

#### E2. K-Loop Unrolling by 2

The K-loop (`for (int ib = 0; ib < blocks_per_row_x; ++ib)` at `quantized.cu:4942`)
has SALU overhead (increment, branch, address recompute) per iteration. Unrolling by
2 amortizes this. SALUInsts are 17% of the instruction mix (from PMC data); halving
that gives ~8% fewer total instructions.

#### E3. K-Quant MMQ v2f

Extend the v2f (no-bounds-check, padded) kernel to Q2_K through Q6_K. Currently
these types fall back to the legacy 8-warp LDS-tiled MMQ path with the `b*m <= 8`
chunking workaround. The v2f design (1 warp, register-only, no LDS) is proven
faster for gfx906.

**Phase E estimated total impact:** +3-5% prefill per-call, closing MMQ to ~96% of
turbo.

---

### Phase F: System-Level — The "Beyond Turbo" Plays (~1000+ LOC)

These are optimizations that exploit candle's Rust architecture in ways that ggml's
op-by-op C execution model cannot replicate.

#### F1. Pre-Transposed KV Cache Layout

Store K in `(B, n_kv_head, D, T)` layout in the KV cache (transposed relative to
the natural `(B, n_kv_head, T, D)`). This eliminates the `k.transpose().contiguous()`
materialization that currently produces a full-size copy per layer per decode step.

Measured impact of the current transpose: it was **tried and reverted** (P3, commit
note) because Tensile picks a different (slower) kernel for the transposed layout.
However, with a working flash-attention (Phase C), the rocBLAS matmul is no longer
in the path — the flash-attention kernel reads K directly from LDS with arbitrary
stride patterns. Pre-transposing K in the cache becomes free.

#### F2. Fused Dispatch Pipeline

Replace the current "one-kernel-launch-per-op" pattern with a **dispatch pipeline**
that batches multiple independent kernel launches into a single HIP submission.
On ROCm 7.1.1, `hipExtLaunchMultiKernelMultiDevice` or a manual command-buffer
approach can reduce the per-launch overhead from 3-5 us to <1 us for batched
launches.

This is different from HIP graphs (which were found counter-productive due to
`hipMallocAsync` serialization). The dispatch pipeline uses pre-allocated buffers and
only batches the launch commands, not the memory operations.

#### F3. Persistent Decode Kernel

For decode (batch=1, sequence position increment by 1), launch a **single persistent
kernel** that processes all layers of the model in sequence, keeping activations in
LDS and registers between layers where the hidden dimension fits.

On gfx906 with 64KB LDS per CU:
- A 2048-dim hidden state is 8KB in f32, 2KB in Q8_1
- Two layers' activations fit comfortably in LDS
- The persistent kernel receives all weight pointers at launch and iterates layers
  internally, eliminating all inter-kernel launch overhead and HBM activation
  write-back

This is **architecturally impossible in ggml/turbo**. Their graph-walker executes
one op at a time; the concept of "keep the activation in LDS across layers" doesn't
exist in their execution model. Candle's model-level Rust code can be restructured
to dispatch a persistent kernel that owns the entire decode forward pass.

Expected impact: On qwen3.5-9B decode, inter-kernel launch overhead is ~5 ms/token
out of ~32 ms. HBM traffic for intermediate activations is ~20 ms/token. A persistent
kernel eliminates both: potential **+35-50% decode throughput**.

#### F4. Asynchronous Weight Prefetch

While the current layer is computing, prefetch the next layer's weights into L2
cache using `__builtin_amdgcn_global_load_lds` or explicit async copy hints. On
gfx906 with 4MB L2, this can hold ~one layer of a 1B model's Q4_0 weights.

This is only effective for small models where the weight set approaches L2 size
(TinyLlama's 1.1B Q4_0 weights are ~660MB, far too large for L2). For decode where
only one row of each weight matrix is accessed, the working set per layer per token
is `hidden_dim * sizeof(block_q4_0) / QK4_0` = ~36KB for a 2048-dim model — easily
prefetchable.

---

## 4. Priority Execution Order

| # | Phase | Item | Primary beneficiary | Complexity | Expected impact | Cumulative target |
|---:|---|---|---|---|---|---|
| 1 | A1 | DPP warp reductions | All models, all kernels | S (~80 LOC) | +5-8% decode, +2-3% prefill | 55-83% |
| 2 | A2 | Hardware SFU (v_exp, v_log, v_rcp) | Softmax, attention | S (~40 LOC) | +2-4% where attention dominates | 57-86% |
| 3 | A3+A4 | Scalar broadcast + launch bounds | All kernels | S (~60 LOC) | +1-3% | 58-88% |
| 4 | C1 | GEMM-style flash attention | gemma4, TinyLlama | L (~1200 LOC) | **+30-60% prefill, +20-40% decode** | **80-120%** |
| 5 | B1 | 128-bit vectorized MMVQ loads | Decode all models | S (~100 LOC) | +5-10% decode | 85-125% |
| 6 | B2 | Split-accumulator MMVQ | Decode all models | S (~50 LOC) | +3-5% decode | 88-128% |
| 7 | D1 | Fused norm+scale+Q8 kernel | Prefill large models | M (~300 LOC) | +5-8% prefill | 90-133% |
| 8 | D2 | Q8 activation cache | Large models (3+ consumers) | M (~200 LOC) | +3-5% prefill | 92-136% |
| 9 | B3 | K-quant warp-cooperative MMVQ | K-quant models | M (~400 LOC) | enables K-quant decode | — |
| 10 | E1+E2 | Split accum + K-unroll in MMQ | Prefill all models | S (~100 LOC) | +3-5% prefill | 95-140% |
| 11 | C2 | Quantized KV in flash-attn | Decode all models | M (~300 LOC) | +5-10% decode | **100-145%** |
| 12 | F2 | Pre-transposed KV cache | Decode (with flash-attn) | M (~300 LOC) | +3-5% decode | 103-148% |
| 13 | F3 | Persistent decode kernel | Decode all models | XL (~800 LOC) | **+35-50% decode** | **Beyond turbo** |

---

## 5. Why Candle Can Surpass Turbo

### 5.1 Structural advantages candle has but hasn't exploited

1. **Model-level fusion.** Turbo's graph-fusion analyzes a flat C graph at runtime.
   Candle's model code is Rust — the entire layer structure is visible at compile
   time. Fusion decisions can be made once, verified by the type system, and
   hard-coded into the model's forward pass. No runtime pattern-matching overhead,
   no missed fusion opportunities from graph walker limitations.

2. **Persistent kernel architecture.** ggml processes ops one at a time. Candle's
   Rust forward function can be restructured to dispatch a single kernel that
   processes multiple layers, keeping activations in fast memory. This eliminates
   the fundamental HBM-bandwidth tax that every ggml-based framework pays for
   inter-op activation materialisation.

3. **Zero interpreter overhead.** vllm-mobydick achieves 6690 t/s prefill (2.5x
   turbo) using Triton MLA kernels, but its decode is 35.7 t/s (5.3x slower than
   turbo) because Python dispatch overhead dominates. Candle has no interpreter
   overhead at any scale.

4. **Native gfx906 kernel design.** Turbo's flash-attention was designed for NVIDIA
   (tensor cores, warp size 32) and adapted for AMD. Candle can design flash
   attention natively for Wave64 with DPP reductions and hardware SFU, potentially
   achieving higher throughput than turbo's adaptation.

### 5.2 What turbo does better that candle should match (not surpass)

1. **Quantization format breadth.** Turbo supports TURBO2/3/4 (custom centroid
   quantization) and MXFP4 with `v_perm_b32` table lookup. These are niche formats
   that matter less than getting Q4_0/Q4_1/Q8_0/K-quants fast. Low priority.

2. **Q8 cache arena with layer cycling.** Turbo's `q8-cache.cuh` has a bump
   allocator with layer-cycled slot reuse. Candle's D2 should match this, but the
   per-layer Rust allocation can be simpler (a `Vec<u8>` resized on first use).

3. **Template instance breadth.** Turbo has 150+ template instantiations covering
   every (quant_type, head_dim, occupancy) combination. Candle should match coverage
   for the formats and head dims that matter (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0
   × D ∈ {64, 80, 96, 128}) and skip the rest.

### 5.3 What candle should NOT port from turbo

- **TurboQuant 3.5-bit KV** — no model files for target architectures
- **WMMA/MMA kernels** — gfx906 doesn't have matrix cores
- **Runtime graph walker** — Rust model code provides this at compile time
- **cp.async** — gfx906 doesn't have async copy hardware
- **lop3.b32 prototype** — turbo's own PoC, not production

---

## 6. Projected Outcomes

### After Phases A+C (the two highest-ROI phases):

| Model | Mode | Current | Projected | vs Turbo |
|---|---|---:|---:|---:|
| TinyLlama 1.1B Q4_0 | prefill | 3441 t/s | ~5500 t/s | **91%** |
| TinyLlama 1.1B Q4_0 | decode | 164 t/s | ~210 t/s | **99%** |
| gemma4 E4B Q4_0 | prefill | 884 t/s | ~1400 t/s | **115%** |
| gemma4 E4B Q4_0 | decode | 35.5 t/s | ~60 t/s | **86%** |
| qwen3.5-9B Q4_1 | prefill | 450 t/s | ~550 t/s | **47%** |
| qwen3.5-9B Q4_1 | decode | 40.6 t/s | ~55 t/s | **85%** |

gemma4 prefill surpasses turbo because the sliding-window attention collapse (10,816
dispatches → ~48) is a disproportionately large win for that architecture. qwen3.5-9B
prefill improves less because it's dominated by GDN recurrence, not attention.

### After all phases (A through F):

| Model | Mode | Projected | vs Turbo |
|---|---|---:|---:|
| TinyLlama 1.1B Q4_0 | prefill | ~6200 t/s | **103%** |
| TinyLlama 1.1B Q4_0 | decode | ~280 t/s | **132%** |
| gemma4 E4B Q4_0 | prefill | ~1500 t/s | **123%** |
| gemma4 E4B Q4_0 | decode | ~85 t/s | **122%** |
| qwen3.5-9B Q4_1 | prefill | ~600 t/s | **52%** |
| qwen3.5-9B Q4_1 | decode | ~80 t/s | **124%** |

qwen3.5-9B prefill remains below turbo because its bottleneck is the GDN recurrent
loop, which is fundamentally sequential and already competitive between candle and
turbo at the kernel level. The decode "beyond turbo" result comes from the persistent
decode kernel (F3), which eliminates inter-kernel overhead and HBM activation
traffic that turbo cannot avoid in its op-by-op execution model.

---

## 7. Verification Protocol

After each phase lands, re-run the 3-way bench:

```bash
# Bench script (same as BENCH-3WAY-POST-P2-2026-04-11.md)
bash /tmp/bench-3way-p2/run_bench.sh
```

Create a new dated bench doc (`BENCH-PHASE-X-2026-MM-DD.md`) with:
1. Headline t/s numbers (3-run average, both sides with warmup)
2. rocprofv3 `--kernel-trace` category rollup (same format as §92 of the post-P2 bench)
3. If a kernel changed: `rocprofv3 --pmc` for VALUBusy, MemUnitBusy, VALUUtilization
4. Delta table vs previous phase and vs turbo

Update `project_perf_vs_turbo.md` and `project_roadmap_3way.md` with the new numbers.

---

## 8. File Reference

### Candle kernel sources

| File | Contents | Key line ranges |
|---|---|---|
| `candle-hip-kernels/src/quantized.cu` | MMVQ, MMQ v2/v2f, dequant, legacy MMQ | MMVQ: 2600-2776, MMQ v2: 4890-5100, warp_reduce: 31-44 |
| `candle-hip-kernels/src/reduce.cu` | Softmax, RMSNorm, LayerNorm, fused ops | warp_reduce: 155-170, softmax: 320-410, rmsnorm: 230-270 |
| `candle-hip-kernels/src/flash_attn.cu` | Flash attention v1 (BR=1) | warp_reduce: 47-53, impl: 56-140 |
| `candle-hip-kernels/src/flash_attn_v2.cu` | Flash attention v2 (BR=4) | warp_reduce: 43-49 |
| `candle-hip-kernels/src/gated_delta_net.cu` | Fused GDN step | warp_reduce: 53-59 |
| `candle-hip-kernels/build.rs` | HIP compilation (hipcc → HSACO) | — |
| `candle-core/src/quantized/hip.rs` | Rust-side MMQ/MMVQ dispatch | — |

### Turbo kernel sources (reference only)

| File | Contents |
|---|---|
| `turbo/.../gfx906/gfx906-common.cuh` | DPP primitives, SFU wrappers, warp_reduce_amd_f32 |
| `turbo/.../gfx906/gfx906-config.h` | GFX906 config (2 warps, Q8 cache, RoPE) |
| `turbo/.../gfx906/matmul/mmvq-q4_0.cuh` | Warp-cooperative Q4_0 MMVQ |
| `turbo/.../gfx906/matmul/mmq.cuh` | MMQ vectorized loads, async tile staging |
| `turbo/.../gfx906/quantize/vecdotq.cuh` | v_perm_b32 MXFP4 lookup, fast int loads |
| `turbo/.../gfx906/fused/graph-fusion.cuh` | Automated RMS_NORM→MUL→MUL_MAT fusion |
| `turbo/.../gfx906/fused/norm-fused-q8.cu` | Fused norm + scale + Q8_1 quantize |
| `turbo/.../gfx906/quantize/q8-cache.cuh` | Q8 activation arena with layer cycling |
| `turbo/.../fattn-vec.cuh` | Vector flash attention (128-thread) |
| `turbo/.../fattn-tile.cuh` | Tile flash attention (256-thread, per-arch config) |

### Bench documents

| File | Contents |
|---|---|
| `BENCH-3WAY-POST-P2-2026-04-11.md` | Current baseline, per-model kernel category rollups |
| `BENCH-PMC-VALU-VMEM-2026-04-11.md` | PMC proof that MMQ is compute-bound |
| `BENCH-P1-PHASE-2D-V2F-2026-04-11.md` | v2f bounds-check elimination results |
| `BENCH-P2-FLASH-ATTN-v1-2026-04-11.md` | Flash-attn v1 post-mortem |
| `BENCH-Q3-FLASH-ATTN-v2-2026-04-11.md` | Flash-attn v2 (BR=4) post-mortem |
