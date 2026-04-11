# Post-P1 Phase 2d (v2f) bench — no-bounds-check MMQ kernel

**Date**: 2026-04-11
**Model**: `Qwen3.5-9B-Q4_1.gguf`, 1 GPU, 1149-tok prompt + 128 decode
**Commits**: `59544766` (Q4_1 v2f) + `c308d94b` (extend to Q4_0 / Q8_0 / Q5_K)

## The story, end to end

| Stage | Prefill | Decode | Total GPU | Gap vs turbo |
|---|---:|---:|---:|---:|
| Pre-P0 baseline | 78.60 t/s | 35.78 t/s | 17.4 s | 5.7× |
| Post-P0 (GDN fused) | ~145 t/s | ~35.4 t/s | 11.4 s | 4.2× |
| Post-P1 Phase 2c (TILE_N=32) | ~195 t/s | ~36 t/s | 9.4 s | 3.5× |
| Post-P5 (Q5_K MMQ) | ~328 t/s | ~36 t/s | 7.1 s | 2.6× |
| Post-P4 (softplus + l2_norm fused) | ~329 t/s | ~36 t/s | 6.7 s | 2.5× |
| **Post-P1 Phase 2d (v2f)** | **~411 t/s** | **~36.5 t/s** | **5.95 s** | **2.20×** |
| Turbo reference | 450 t/s | 58.76 t/s | 2.70 s | 1× |

**Prefill is now at 91.3 % of turbo.** Cumulative vs pre-P0: **5.22×**.

## What v2f did

The PMC analysis (`BENCH-PMC-VALU-VMEM-2026-04-11.md`) showed the MMQ kernels were compute-bound with a dead-flat 49.1 % VALU lane utilization. Disassembly with `llvm-objdump --mcpu=gfx906` found the smoking gun: 574 `v_readlane_b32 / v_writelane_b32` instructions per kernel call — **27 % of all instructions** were lane-scratch spill/fill emitted by the compiler as its implementation of `if (col >= ncols_y) break;` inside the fully-unrolled col loop.

The v2f variant eliminates the per-col bounds check entirely and pads the Y quant buffer to `ceil(total_b / tile_n) * tile_n` so OOB col reads land on zero-initialised memory and contribute nothing to the dp4a sums. Writebacks are still gated on `col < ncols_y`.

### Disassembly deltas

| Kernel | v2 instructions | v2 lane moves | v2f instructions | v2f lane moves |
|---|---:|---:|---:|---:|
| Q4_0 tile32 | 2163 | 574 (26 %) | **1394** | **0** |
| Q4_1 tile32 | 2130 | 574 (26 %) | **1425** | **0** |
| Q8_0 tile32 | 2170 | 574 (26 %) | **1337** | **0** |
| Q5_K tile32 | 19 415 | 3 608 (18 %) | 17 180 | 4 414 (25 %)† |

† Q5_K's lane-move count went UP at the instruction level, but the per-call runtime **dropped 37 %** — the compiler shifted to a different (cheaper) spill pattern, and removing the 1000+ instructions of per-col predicate more than paid for the new spills. The instruction-count metric is misleading for Q5_K.

## rocprofv3 category deltas (post-P4 → post-v2f)

| Category | Post-P4 | **Post-v2f** | Δ |
|---|---:|---:|---:|
| quantized_matmul_v2 (prefill MMQ) | 2877 ms | **2039 ms** | **−29 %** |
| quantized_matvec (decode MMVQ) | 1648 ms | 1676 ms | flat |
| pointwise | 913 ms | 802 ms | −12 % |
| gated_delta_net (P0 kernel) | 495 ms | 510 ms | flat |
| memcpy / fill | 472 ms | 431 ms | −9 % |
| q8_1 quantize | 140 ms | 153 ms | flat |
| fused_ops | 115 ms | 124 ms | flat |
| rocblas_attention | 102 ms | 107 ms | flat |
| norm | 63 ms | 64 ms | flat |
| **Total GPU kernel time** | **6665 ms** | **5951 ms** | **−714 ms (−11 %)** |

Only the MMQ category moved substantially, as expected. The "−12 % on pointwise" is noise from the reduced memory pressure hiding other ops in the VALU shadow.

### Per-kernel top-line deltas

| Kernel | Post-P4 avg | **Post-v2f avg** | Δ per call |
|---|---:|---:|---:|
| `mul_mat_q4_1_gfx906_v2_tile32` | 15 532 µs | — | — |
| `mul_mat_q4_1_gfx906_v2f_tile32` | — | **11 454 µs** | **−26 %** |
| `mul_mat_q5_K_gfx906_v2_tile32` | 34 573 µs | — | — |
| `mul_mat_q5_K_gfx906_v2f_tile32` | — | **21 799 µs** | **−37 %** |
| `mul_mat_q8_0_gfx906_v2_tile32` | 1 224 µs | — | — |
| `mul_mat_q8_0_gfx906_v2f_tile32` | — | **1 038 µs** | **−15 %** |

## Remaining gap to turbo (2.20×)

| Category | Post-v2f | Turbo | Ratio | Lever |
|---|---:|---:|---:|---|
| **quantized_matmul (prefill MMQ)** | **2039 ms** | **513 ms** | **4.0×** | Full turbo-style LDS + 2-D per-thread sub-tile rewrite. ~1000 LOC, uncertain ROI now that v2f already captured most of the cheap wins. |
| quantized_matvec (decode MMVQ) | 1676 ms | 1396 ms | 1.2× | **Nearly at parity.** No obvious lever. |
| Pointwise | 802 ms | 239 ms | 3.4× | P3 copy audit (ucopy_f32 = 435 ms, #5 kernel). |
| gated_delta_net | 510 ms | 240 ms | 2.1× | Minor tuning — low priority. |
| memcpy / fill | 431 ms | 168 ms | 2.6× | P3 audit. |

**Projected after P3 copy audit**: −200–300 ms → **~5.7 s total → ~435 t/s prefill**. That would put us within 3 % of turbo.

**Projected after full Phase 2d-rewrite** (if we commit to the turbo-style MMQ port): −1500 ms on the MMQ path → **~4.2 s total → matches turbo prefill**. But we're already so close that the ROI on a 1000+ LOC rewrite is marginal. Better to tighten up P3 first.

## Lessons

1. **PMC counters before big rewrites.** The original Phase 2d plan targeted LDS memory pipelining based on guesswork; measurement showed the kernel was compute-bound with essentially zero memory stalls. A 1000 LOC rewrite would have shipped the wrong optimization.

2. **Disassemble when PMC metrics are flat.** `VALUUtilization = 49.1 %` with a tiny σ across 128 calls was an architectural signal that the compiled code had a specific, repeatable waste pattern — not a workload-dependent thing. `llvm-objdump` showed the pattern in 30 seconds of looking.

3. **The compiler's unrolled-loop bounds-check implementation is expensive on AMD.** Any `#pragma unroll` loop with a runtime bounds check inside on gfx906 will emit a writelane-per-iteration predicate mask. Hoisting the check out of the loop (or padding the operand and dropping it) fixes it.

4. **Instruction count isn't pipeline performance.** Q5_K v2f has more instructions and more lane moves than v2, but runs 37 % faster. The removed per-col predicate was on the critical path; the new lane moves the compiler added aren't.

## Recommended next step

**P3 — copyBuffer / ucopy audit.** `ucopy_f32` is now the #5 kernel line at 435 ms, and the broader memcpy/fill category is 431 ms. Audit the call sites (`.contiguous()` materializations, reshape copies, state tensor clones) and eliminate the redundant ones.

Expected win: ~200–300 ms total GPU time → **~435 t/s prefill**, within 3 % of turbo.
