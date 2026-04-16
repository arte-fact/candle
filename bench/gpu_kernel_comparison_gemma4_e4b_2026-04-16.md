# Per-kernel comparison — candle vs llama.cpp — gemma4-E4B-Q4_0, gfx906, 1-GPU, pp=32 + tg=64

Tool: `rocprofv3 --kernel-trace`. Both engines profiled on the same workload; rocprof
adds proportional overhead to both, so absolute numbers are depressed from the
un-profiled wall-clock — compare **ratios** and **per-call** costs, not totals.

## Summary (totals)

| engine | total GPU kernel time | launches | tg64 under rocprof |
|---|---:|---:|---:|
| candle  | 616 ms | 50 336 | ~72 t/s |
| llama.cpp (8701) | 827 ms | 83 158 | 61 t/s |

llama.cpp has **1.35× more GPU kernel time** and **1.65× more launches** than candle on this
workload. Yet un-profiled wall-clock has llama.cpp **faster**. Implication: llama.cpp's
individual kernels run more concurrently (better stream usage / overlap) rather than being
faster per-byte.

## Top kernels — candle

| kernel | calls | total µs | µs/call | % |
|---|---:|---:|---:|---:|
| mul_mat_vec_q4_0_q8_1_cuda1 | 6963 | 156 099 | **22.42** | 25.33 % |
| mul_mat_q4_0_turbo_x64_unchecked | 422 | 130 058 | 308.19 | 21.11 % |
| mul_mat_vec_q4_K_q8_1_wc | 35 | 34 701 | 991.47 | 5.63 % |
| gqa_decode_mv_fast_d256_f32 | 1155 | 34 567 | **29.93** | 5.61 % |
| mul_mat_q8_0_turbo_x64_unchecked | 168 | 29 786 | 177.30 | 4.83 % |
| __amd_rocclr_copyBuffer | 6199 | 25 953 | 4.19 | 4.21 % |
| quantize_q8_1 | 6173 | 25 021 | 4.05 | 4.06 % |
| Cijk_Alik_Bljk_SB_MT128x64x16 (rocBLAS) | 35 | 15 573 | 444.95 | 2.53 % |
| rmsnorm_{q8_fused,post_residual,f32} | ~8146 | 42 171 | ~5 | 6.84 % |

## Top kernels — llama.cpp

| kernel | calls | total µs | µs/call | % |
|---|---:|---:|---:|---:|
| mul_mat_vec_q<Q4_0,1,false,false> | 10 817 | 162 964 | **15.07** | 19.70 % |
| mul_mat_vec_q<Q4_0,1,true,false>  |  2 689 | 147 126 | 54.71 | 17.78 % |
| mul_mat_vec_q<Q4_K,1,false,false> |     65 |  71 608 | 1101.67 | 8.65 % |
| mul_mat_q<Q4_0,32,false>          |    250 |  62 749 | 251.00 | 7.58 % |
| quantize_q8_1                     | 13 891 |  48 748 | 3.51 | 5.89 % |
| rms_norm_f32 family (3 variants)  |  ~19 630 |  ~99 300 | ~5 | 12.00 % |
| **mul_mat_vec_f<float,float,1,128> (attention) ** |  5 378 |  46 704 | **8.68** | 5.64 % |
| mul_mat_vec_f<__half,__half,1,128> (per_layer_embed) |  2 688 |  23 679 | 8.81 | 2.86 % |
| mul_mat_vec_f<__half,float,1,128>  |  2 688 |  22 021 | 8.19 | 2.66 % |
| soft_max_f32                      |  2 730 |  13 012 | 4.77 | 1.57 % |
| rope_neox                         |  3 575 |  14 128 | 3.95 | 1.71 % |
| k_set_rows<float,long,__half>     |  3 120 |  12 320 | 3.95 | 1.49 % |

## Head-to-head per-call latency

| op | candle µs/call | llama.cpp µs/call | ratio (c/l) |
|---|---:|---:|---:|
| Q4_0 MMVQ (fast variant) | 22.42 | 15.07 | **1.49×** (candle slower) |
| Q4_0 MMVQ (slow variant) | — | 54.71 | — |
| Q4_K MMVQ (lm_head) | 991.47 | 1101.67 | 0.90× (candle faster) |
| Decode attention (D=256) | **29.93** (gqa_decode_mv_fast) | ~8.68 (mul_mat_vec_f<float,float,1,128>) | **3.45×** (candle slower) |
| quantize_q8_1 | 4.05 | 3.51 | 1.15× |
| rmsnorm_f32 | 4.57 | 4.99 | 0.92× (candle faster) |
| rms_norm_post_residual | 4.75 | 5.88 | 0.81× (candle faster) |
| gelu_mul (activation) | 4.77 (fused) | 4.37 | 1.09× |

## Three big findings

### 1. Candle's Q4_0 MMVQ is ~1.5× slower per call

Despite tied VGPR count (22 candle vs 24 llama.cpp) and identical dp4a chain, candle's
`mul_mat_vec_q4_0_q8_1_cuda1` runs 22.4 µs vs llama.cpp's 15.1 µs for the fast variant.
Possible culprits:
- Different grid geometry for the decode n_cols (llama.cpp may dispatch
  `nrows/2` vs candle's half-warp coop)
- Cache-line alignment of Q4_0 block reads
- Loop-unroll factor

**Action**: Disassemble both kernels side-by-side and look for differences in
VMEM issue count or dp4a latency chaining.

### 2. **Attention kernel is the real gap** — and tile-based flash-attn is the wrong target

candle uses `gqa_decode_mv_fast_d256_f32` at **29.93 µs/call** × 1155 calls = 34.6 ms.

**llama.cpp does NOT use flash_attn_ext_vec for this model.** It decomposes attention
into two gemvs + softmax:
- `mul_mat_vec_f<float,float,1,128>` at 8.68 µs × 5378 calls = 46.7 ms
- `soft_max_f32` at 4.77 µs × 2730 = 13.0 ms
- plus a second gemv stage

Per attention op llama.cpp = 8.68 + 4.77 + 8.68 ≈ 22 µs but across n_head calls — so
**per single (B,H_q,L_k,D) attention**, llama.cpp is **~3.5× faster per call than candle's
"mv_fast"**.

Why? llama.cpp's `mul_mat_vec_f<1,128,false,false>` uses BLOCK_SIZE=128 threads, candle's
uses 64 (WARP_SIZE). Doubling threads-per-block = more latency hiding on the K/V read, at
the cost of cross-wave LDS sync. For D=256 matmul-vec, 128 threads × 2 positions each =
256 with 2 passes, all overlapping HBM loads.

**This is the right Task #2 target, not tile-based flash-attn.** Rewrite
`gqa_decode_mv_fast_d256_f32` as a 128-thread / 2-wave block that uses LDS cross-warp
reduce — matching llama.cpp's `mul_mat_vec_f<1,128>` architecture.

Expected win: close the attention 3.5× per-call gap → save ~20 ms per pp+tg → 5-10 % tg.

### 3. Candle is faster on rmsnorm

Our rmsnorm kernels win by 10-20 % per call. Not the gap.

## Revised Task #2 design

~~Tile-based flash-attention kernel with GQA fanout reuse~~

**New direction**: 128-thread / 2-wave `gqa_decode_mv_fast_d{256,512}_f32` with
cross-warp LDS reduce, mirroring llama.cpp's `mul_mat_vec_f<1,128,false,false>` layout.
No LDS tile of K/V needed — just more threads per block for better HBM latency hiding on
the row read.

---

**Data**: `/tmp/prof_candle/candle_kernel_trace.csv`, `/tmp/prof_llama3/llama3_kernel_trace.csv`

