# P0 fact-check — measured results (2026-04-19)

## P0.1 Baseline perf snapshot

| Model | GPUs | Env | pp512 (t/s) | tg64 (t/s) | Note |
|---|---|---|---|---|---|
| Qwen3.5-9B-Q4_1 | 1 (GPU1) | default | 700.4 | 54.83 | qwen35 hybrid (FullAttn+Recurrent) |
| Qwen3.5-9B-Q4_1 | 1 (GPU1) | CANDLE_KV_Q8=1 | 696.78 | 54.69 | ≈ no change (KV small, few FullAttn layers) |
| gemma-4-26B-A4B-Q8_0 | 4 | default | 679.85 | 48.52 | |
| gemma-4-26B-A4B-Q8_0 | 4 | CANDLE_KV_Q8=1 | — | — | **BUG**: `QKvCache::append: src dtype must be F32 (got F16)` |
| gemma-4-E4B-Q4_0 | 1 | default | — | — | **BUG**: lm_head missing (no token_embd / output.weight loadable via try_qmatmul) |

## P0.2 rocprofv3 trace — Q8-KV dequant kernel attribution

Qwen3.5-9B + CANDLE_KV_Q8=1, prompt=40-token explain, sample-len=40:
- **`dequantize_q8_0_kv_f32` fires 672 times** (42 decode steps × 8 FullAttn layers × 2 K+V) ✓ orphan confirmed.
- Total 4.4 µs avg × 672 = **0.21% of decode kernel time**.
- `gqa_decode_gemv_{qk,v}_q8_d128_f32`: **NOT PRESENT** — direct wrapper orphaned ✓.
- `gqa_decode_gemv_{qk,v}_d128_f32` (F32 path): also NOT present → attention runs via rocBLAS GEMM, not the direct candle gemv kernels.

Qwen3.5-9B + CANDLE_KV_Q8=1, prompt-len=2048, tg-len=64:
- `dequantize_q8_0_kv_f32`: 1056 calls, 5.2 ms, **0.36% of kernel time**.
- Dequant overhead is small on this hybrid model; payoff scales with FullAttn layer count × D × L_k. Memory records qwen3-coder-next 80B hit +71% tg with Q8 KV at long context → payoff is model-dependent.

## P0.3 Direct Q8 GEMV correctness on current binary — PASS ALL

| Test | max_abs_err | Tolerance | Result |
|---|---|---|---|
| hip_q8_attn_pipeline_test (D=128, L_k=256) | 1.863e-8 | 5e-5 × r_max | **PASS** |
| hip_q8_qk_attn_test | 4.768e-6 | n/a | **PASS** |
| hip_q8_v_attn_test | 2.421e-8 | n/a | **PASS** |
| hip_q8_decode_loop_test (16 decode steps) | 9.656e-5 | loose | **PASS** (worst at step 13) |

**Per-kernel perf (key data):**
- V·attn D=128 L_k=256: **F32 dequant+GEMV = 67 µs → Q8 direct = 44 µs, -34.3%**.
- Q·K D=128 L_k=256: Q8 direct median = 17 µs.

## P0.4 MMQ L2 prefetch gap — CONFIRMED at source

- Candle `mmq_turbo.cu:21` comment: `// Drop L2-prefetch: add back in stage 3 if useful.` Never done.
- Turbo `ggml-cuda/gfx906/matmul/mmq-prefetch.cuh`: 90-line file, `gfx906_prefetch_y_tile_v4` + `gfx906_prefetch_y_tile_second` issue `global_load_dword` for kb0+1 Y-tile using spare lanes. Per-iter L2 warmup.

## P0.5 MMVQ occupancy PMC — Q4_1 MMVQ is MEMORY-STARVED, Q5_K MMQ is VGPR-LOCKED

VGPR counts (from rocprof Kernel_Dispatch trace):

| Kernel | VGPR | Max waves/SIMD | SQ_WAVES | VALUBusy | VALUUtil | MemBusy | Diagnosis |
|---|---|---|---|---|---|---|---|
| mul_mat_vec_q4_1_q8_1_cuda1 | 24 | 8 | 5184 | **11.9%** | 96.6% | 65.8% | **Mem-latency bound**. VALU idle most of the time; adding waves would hide HBM latency. |
| mul_mat_vec_q5_K_q8_1_wc | 44 | 5 | 2048 | 20.9% | 50.7% | 75.3% | Low VALUUtil → execution hazards (DPP or LDS banks). |
| mul_mat_vec_q6_K_q8_1_cuda1 | 24 | 8 | 993k | 85.8% | 98.9% | 87.1% | Good — near-saturated. |
| mul_mat_q4_1_gfx906_v2f_tile32 | 92 | 2 | 162 | 8.6% | 100% | 17.6% | Prefill: compiler fits 2 waves. Room for turbo-style prefetch to hide mem. |
| **mul_mat_q5_K_gfx906_v2f_tile32** | **192** | **1** | 64 | 6.5% | 67.1% | 4.5% | **VGPR-locked to 1 wave/SIMD**. This is the 3.5× Q4_K/Q5_K MMQ gap vs turbo. |

**Key takeaways from PMC:**
1. Q4_1 MMVQ at 24 VGPRs could run 8 waves/SIMD but `__launch_bounds__(64, 1)` sets min=1, compiler picks low occupancy → CU busy only ~12% of time → **adding `__attribute__((amdgpu_waves_per_eu(4,8)))` is a measurable win, not speculation**.
2. Q5_K MMQ at 192 VGPRs is occupancy-locked. No annotation fixes that — must reduce register pressure (split accumulators, VMEM scheduling). Priority target for closing the turbo gap.
3. "Memory-bound" claim (from MEMORY line 54) needs nuance: the *kernel* is memory-latency-starved, but the *system* has plenty of aggregate HBM bandwidth. This is the textbook "more waves, not more bandwidth" fix.

## P0.6 fused_ffn_decode_q4_0 — DEAD CODE CONFIRMED

Grep `fused_ffn` in `candle-transformers/`: 0 hits. Only callers are in `candle-core/src/hip_backend/fused_pointwise.rs` (defines the dispatcher) and the kernel source. **Not on any model's forward**. Dropped from the priority list.

## Re-ranked priority after P0 data

1. **Wire `gqa_attention_decode_q8` into GatedAttention (L_q=1 only)** — kernels proven correct, 34% per-kernel win measured. Payoff scales with FullAttn layer count × head_dim × context length; biggest on qwen3-coder-next long-context. Low risk (decode-only branch, env rollback).
2. **`amdgpu_waves_per_eu(4,8)` on Q4/Q8/K-quant MMVQ kernels** — cheap annotation. Q4_1 MMVQ at 11.9% VALUBusy → ~3-4× headroom. Measure per-kernel before declaring a win.
3. **Reduce Q5_K MMQ VGPR count below 128 to unlock 2 waves/SIMD** — real fix for the 3.5× turbo gap. Multi-session work, needs correctness harness (C3-redo lesson).
4. **Port `mmq-prefetch.cuh` Y-tile prefetch into candle `mmq_turbo.cu`** — the TODO from line 21. Depends on (3) for best effect.
5. **Fix `CANDLE_KV_Q8=1` F16 dtype bug in `QKvCache::append`** — blocks Q8 KV on gemma4-26B-A4B. Small but unblocks a model family.

## Go decision for P2.1 (implement)

Target (1) meets all go criteria:
- ✓ P0.3: correctness established with margin.
- ✓ P0.2: orphan confirmed (direct GEMV not firing); dequant kernel fires on every Q8 decode step.
- ✓ P0.1: baselines captured.
- ✓ Per-kernel speedup measured (34% on V·attn at mid-scale).

**Go** on target (1). Proceed to P1.2 (implementation plan), then P2.1.
