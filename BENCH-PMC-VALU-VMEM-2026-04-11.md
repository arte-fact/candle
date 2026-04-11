# PMC analysis — `mul_mat_q4_1_gfx906_v2_tile32` is **compute-bound**

**Date**: 2026-04-11
**Target**: `mul_mat_q4_1_gfx906_v2_tile32` (Q4_1 MMQ prefill kernel)
**Tool**: `rocprofv3 --pmc`
**Sample**: 128 dispatches from `quantized-qwen35` on qwen35-9B / 1149-tok prompt

## Headline numbers

| Counter | Mean | Min | Max | Interpretation |
|---|---:|---:|---:|---|
| **VALUBusy** | **69.64 %** | 61.58 | 78.91 | **VALU units are busy ~70 % of GPU time** |
| **MemUnitBusy** | **24.81 %** | 17.15 | 33.38 | Memory unit active only ~25 % of GPU time |
| **MemUnitStalled** | **0.15 %** | 0.04 | 0.25 | **Essentially zero memory stalls** |
| **VALUUtilization** | **49.11 %** | 49.06 | 49.12 | **Dead flat — only ~32 of 64 lanes active per VALU op** |
| **L2CacheHit** | 65.39 % | 48.73 | 90.00 | Decent, not the bottleneck |

Instruction mix (pass 2a):
| Counter | Mean per wave | Ratio |
|---|---:|---:|
| VALUInsts | 187,440 | 1× |
| SALUInsts | 37,920 | 0.20× (~17 % of total) |
| LDSInsts | 0 | — |
| VFetchInsts | 0 | (all loads are flat, see below) |

Memory volume (pass 3a):
| Counter | Mean per call | Note |
|---|---:|---|
| FetchSize | 2.42 MB | At 15.8 ms/call that's **152 MB/s** — **0.015 % of gfx906's 1 TB/s HBM peak** |

## Conclusion — memory is not the bottleneck

- **MemUnitStalled = 0.15 %** — the memory unit is almost never stalled waiting for data. Whatever memory latency exists is entirely hidden by VALU work.
- **152 MB/s effective DRAM fetch rate** vs **~1 TB/s peak**. We're at **six-thousandths of a percent of memory bandwidth**. There is essentially infinite memory headroom.
- **L2CacheHit = 65 %** — decent but not high. Doesn't matter because we're not memory-bound anyway; L1/L2 misses hide in the VALU pipeline.
- **LDSInsts = 0** — the current v2 kernel uses no LDS at all.

**LDS staging was the wrong optimization.** It can't help a kernel whose memory unit is 99.85 % idle. This empirically confirms the Phase 2d null result from
`BENCH-P1-PHASE-2D-NULL-2026-04-11.md` — we can now explain WHY it failed.

## The real bottleneck — VALU lane utilization

- **VALUBusy = 70 %**: VALU pipelines are running ~70 % of the time.
- **VALUUtilization = 49 %** (dead flat): when a VALU instruction executes, only ~32 of 64 lanes are doing useful work.
- Effective VALU throughput: `0.70 × 0.49 = 34 %` of peak.

**Reading a dead-flat 49.1 % utilization across 128 calls of varying shapes strongly suggests an architectural property of the compiled code**, not a workload-dependent thing like tile padding. Either:
- The compiler is emitting VALU instructions with predicated-off lanes in a consistent pattern (maybe around the `if (row_valid)` guard or the Q4_1 nibble-unpack loop).
- Some of the VALU instructions are 2-cycle (half-rate) on gfx906, so the "lane-cycles" counter reports half what the instruction count would suggest.
- Loop-control / bookkeeping VALU ops are counted in the denominator but run with a subset of lanes active.

Either way, the lever is **raising VALU lane utilization**, which is achieved by:
1. **Reducing wasted VALU cycles**: hoist any scalar-broadcast work out of the VALU path; ensure the Q4_1 unpack uses full-width int-vector ops; check the compiler isn't emitting any 2-cycle VALU patterns.
2. **Unrolling the K-block loop**: currently 128 iterations with SALU loop-control between them (increment `ib`, recompute `bx = &x[row * blocks_per_row_x + ib]`). Unrolling by 2-4× would amortise the SALU and may give the compiler room for better VALU scheduling.
3. **Double or triple accumulator variables**: break the serial `sums[c] += dp4a * d4 * d8 + m4 * s8` dependency chain into 2-3 independent accumulators, then combine at the end. That increases instruction-level parallelism within the same wave.
4. **Inspect the HSACO** with `llvm-objdump --disassemble` to see exactly what the compiled inner loop looks like. Sometimes hipcc emits surprising sub-optimal code that one targeted `__builtin_assume` or `#pragma unroll` fixes.

**What would NOT help** (already disproved by the PMC data):
- LDS staging for X or Y tiles — memory is not the bottleneck.
- Async prefetch of the next K tile — memory is not stalling.
- Cross-warp X/Y sharing — the lever for cross-warp sharing is memory-amortisation, which isn't needed here.
- A 2-D per-thread sub-tile rewrite — only helpful IF it raises lane utilization, which isn't guaranteed and is unrelated to the "LDS-driven speedup" story the Phase 2d plan assumed.

## Implication for Phase 2d rewrite plan

The current `ROADMAP-OPTIMIZATIONS-FROM-3WAY.md` P1 Phase 2d section
describes a 2-D per-thread sub-tile rewrite with collaborative LDS
staging as the path to outperform turbo. **That plan was based on an
incorrect assumption about the bottleneck.** PMC data says the kernel
is already memory-idle 75 % of the time; staging data through LDS can
only add LDS bank conflicts and `__syncthreads()` overhead.

Turbo's 6× per-call advantage MUST therefore come from one or more of:
1. Higher VALU lane utilization (better-scheduled inner loop)
2. More dp4a ops per SALU overhead (aggressive K-loop unrolling)
3. Better FMA pipelining (multiple independent accumulator chains)
4. Use of specialised v_dot* variants that do more work per instruction

None of these require LDS. They require **inner-loop code-gen
work**, which is a very different effort than the "port turbo's 2-D
tile structure" rewrite the roadmap currently prescribes.

## Recommended next steps

1. **Disassemble the HSACO** for `mul_mat_q4_1_gfx906_v2_tile32` and
   compare the inner loop against turbo's compiled `mul_mat_q<Q4_1>`.
   That will tell us exactly where the 2× lane-count gap comes from.
2. **Try split accumulators**: change `sums[c]` (one accumulator per
   column) to two independent chains that accumulate different K
   sub-ranges and are combined at the end. This is a small change
   that increases ILP per thread.
3. **Try K-loop unrolling by 2 or 4**: one line change, potentially
   reduces SALU overhead from 17 % to ~5 %.

If those three targeted changes close the per-call gap to < 2× of
turbo, Phase 2d rewrite is deferred indefinitely. If not, the real
rewrite path is "port turbo's inner loop verbatim" rather than
"port turbo's LDS staging structure".

## Raw PMC data

- `/tmp/bench-pmc/pmc1_counter_collection.csv` — VALUBusy, MemUnitBusy, MemUnitStalled, VALUUtilization, L2CacheHit
- `/tmp/bench-pmc/pmc2a_counter_collection.csv` — VALUInsts, SALUInsts, VFetchInsts, LDSInsts
- `/tmp/bench-pmc/pmc3a_counter_collection.csv` — FetchSize
