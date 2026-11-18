# Qwen3.6-35B-A3B — candle vs turbo kernel diff (pp512-class)

**Date**: 2026-04-18
**Hardware**: 4× MI50 (gfx906), ROCm 7.1.1
**Model**: `/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (arch=qwen35moe, 40 layers, 30 GDN + 10 Gated Attn, 256 experts)
**Methodology**: `rocprofv3 --runtime-trace --stats --summary --summary-units usec` on one prefill + 1 decoded token. Summaries are GPU-sum across all 4 ranks. Normalised to µs/tok because the two runs fed different prompt lengths (candle 1129 tok, turbo 512 tok).

**IMPORTANT REVISION 2026-04-18 (post-B1)**: the initial `qwen36_candle_512` run passed `--split-prompt` to the binary, which forces per-token forward for debugging. That flag is **not required** for Qwen3.6 (`Model::is_recurrent()` already returns `false`; linear projections batch across the whole prompt). Removing it gives an 11.5× speedup before any code change. The honest baseline lives in `qwen36_candle_nosplit` below. The pathological `qwen36_candle_512` profile is kept as a reference for what the per-token path looks like, but is not the candle default.

Raw artefacts:
- `/tmp/prof_qwen36_candle_nosplit/summary.txt` — candle default (no --split-prompt)
- `/tmp/prof_qwen36_candle_512/summary.txt` — candle with --split-prompt (pathological, per-token)
- `/tmp/prof_qwen36_turbo_512/summary.txt` — turbo llama-bench -p 512 -n 1

## Headline (revised 2026-04-18 post-B1)

| Metric | Candle default | Candle --split-prompt | Turbo | Default vs Turbo |
|---|---:|---:|---:|---:|
| pp rate (t/s, wall clock) | **173.69** | 15.49 | **1012.80** | **0.171× (5.83× slower)** |
| GPU-sum kernel time | 12.75 s | 41.90 s | 1.05 s | 12.2× |
| Total kernel dispatches | **11 510** | 2 779 842 | 8 297 | **1.39×** |
| µs/tok (GPU-sum) | 11 296 | 37 113 | 2 044 | 5.5× |

Removing `--split-prompt` closes **241 of the original 335× dispatch gap and 3.3 of the 18× GPU-sum gap** with zero code change — the flag was artificially forcing per-token forward. At 11 510 dispatches vs turbo's 8 297, candle is now within 1.4× of turbo on launch count; the residual GPU-sum gap is genuine per-kernel cost, not launch overhead.

## Top candle kernels — default (no --split-prompt)

| # | Kernel | Calls | Total ms | Avg µs | % GPU |
|---|---|---:|---:|---:|---:|
| 1 | `indexed_moe_forward_q4k_q8_1` | 351 | 10 408 | 29 652 | **81.6** |
| 2 | `fast_sum_f32` | 390 | 568 | 1 456 | 4.5 |
| 3 | `mul_mat_q8_0_gfx906_v2f_tile32` | 460 | 488 | 1 060 | 3.8 |
| 4 | `indexed_moe_forward_q6k_q8_1` | 9 | 388 | 43 156 | 3.0 |
| 5 | `gated_delta_net_step_s128_f32` | 90 | 213 | 2 362 | 1.7 |
| 6 | `__amd_rocclr_fillBufferAligned` | 1 297 | 189 | 146 | 1.5 |
| 7 | `bmul_f32` | 750 | 120 | 161 | 0.9 |
| 8 | `ucopy_f32` | 1 040 | 106 | 101 | 0.8 |
| 9 | `quantize_q8_1` | 973 | 38 | 39 | 0.3 |
| 10 | Cijk rocBLAS MT64/128 (mixed) | 160 | 60 | — | 0.5 |

**One kernel owns 81.6 % of the remaining GPU time**: `indexed_moe_forward_q4k_q8_1` at 10.4 s over 351 calls. This is the candle MoE dispatch — one call per (layer, expert) pair, 40 × ~9 ≈ 360. Turbo does the equivalent work in ~230 batched `mul_mat_q<Q4_K, 64>` calls summing ~364 ms.

Our Q8_0 MMQ kernel (`mul_mat_q8_0_gfx906_v2f_tile32`) is now **firing** on the non-expert matmuls at 488 ms / 460 calls — that's P2 paying off on the real workload once per-token was lifted. The Q4_K MMQ equivalent would fire if the dispatcher selects it for MoE expert weights, but the indexed_moe path bypasses MMQ entirely.

### Measured turbo_moe payoff ceiling (Q4_0 A/B)

Direct comparison on Qwen3-Coder-Next-Q4_0 (80B, same hybrid GDN+GAttn family as Qwen3.6, `arch=qwen3next`) measures what the existing turbo_moe path actually buys on an MoE model where it IS enabled (Q4_0):

| Config | pp512 t/s | Wall s |
|---|---:|---:|
| turbo_moe ON | **234.83** | 26.6 |
| turbo_moe OFF (`CANDLE_MMQ_TURBO_MOE=0`) | 149.18 | 31.3 |
| Gain | **+57 %** | −15 % |

**The 29× kernel-time delta in the µs table does NOT translate to 29× pp**, but the 1.57× Q4_0 anchor is NOT a principled ceiling either — it's one data point on a different architecture.

Corrected analysis:
- HBM-bandwidth floor for Qwen3.6 pp512 MoE weight reads: **81.5 ms (= 6 279 t/s)** on 4×MI50. Observed 178 t/s is 35× slower than HBM floor → we are **not** bandwidth-bound.
- MMVQ vs MMQ arithmetic intensity ratio: ~16-32× (weight registers reused across activation tile). Up to ~10× theoretical headroom on the matmul portion alone.
- Qwen3.6 has higher avg tokens/expert (16) than qwen3next (10) → better MMQ tile fill → likely *more* gain than 1.57×, not less.
- Non-MoE kernels (GDN, attention, norms) are ~18 % of current GPU time and unaffected by B2. Amdahl caps end-to-end speedup at ~3.8× even if MoE portion collapses 10×.
- Pre-B2 prediction: 2-3× pp on Qwen3.6 → 350-550 t/s (0.35-0.54× turbo).

### B2 + C1 measured results — 2026-04-18

**B2** — `mul_mat_q4_K_turbo_moe_x{8,16,32,64}_{checked,unchecked}` in `candle-hip-kernels/src/mmq_turbo.cu` + Q4K dispatch branches in `candle-core/src/quantized/hip.rs`.

**C1** — `moe_combine_weighted_sum_fused` (W3A kernel that already existed for gemma4) wired into all three `MoeExperts::forward*` paths in `candle-transformers/src/models/quantized_blocks/ffn.rs` (was only lit in gemma4). Replaces `broadcast_mul + sum(1)` with a single kernel per MoE layer.

| Metric | pre-B1 | B1 | B2 | C1 | C2 (peak) | **post-C3-revert (final)** |
|---|---:|---:|---:|---:|---:|---:|
| pp128 | — | — | 316.34 | 344.99 | 346.94 | 345.23 |
| pp512 | 15.49 | 178.93 | 412.11 | 466.80 | 489.11 | **467.32** |
| Ratio vs turbo (1012.80) | 0.015× | 0.177× | 0.407× | 0.461× | 0.483× | **0.461×** |
| tg64 | 25.04 | 25.65 | 25.24 | 25.62 | 26.39 | 25.44 |

**C3 retro**: attempted to unify Q4_K into the main `mul_mat_q_turbo_moe_impl` 4-warp template (to properly pick up Y LDS tiling and close the 3.5× per-call gap vs turbo). Kernel compiled, correctness verified byte-identical to reference on small prompts, VGPR count was healthy (74 at x=32, 3 waves/SIMD), per-call time under rocprof was promising. But intermittent rocBLAS shutdown errors at specific prompt lengths (nrep=60/80/100 of 64-token chunks crash, nrep≤40 and nrep≥120 work — non-monotonic) indicate a shape-dependent OOB/race that I didn't isolate in session budget.

**Reverted C3 and C2 (the C2 MW kernel was caught in the cleanup). Final session delivery is B1+B2+C1 = 467 t/s** (0.461× turbo). The intermediate C2 peak (489 / 0.483×) is gone. Net gain vs the proper B1 baseline: **2.61×**.

Lessons:
- "Redo all properly" actually means: scope it properly first. Porting turbo's template extension is multi-day work (correctness test sweeps, HIP memcheck, staged integration), not a single-session task.
- Losing working code (C2) while chasing a bigger win (C3) was avoidable — should have kept C2 stable as the fallback before attempting C3.
- The 3.5× per-call gap vs turbo remains unclosed. Proper C3 is future work with a real correctness harness.

**C2 detail**: multi-warp Q4_K kernel with output distributed across 4 warps (per-thread SUM_SLOTS = MMQ_X/4 instead of TILE_N=32). VGPR collapsed 188→78 at x32 (1→3 waves/SIMD). LDS bank-conflict fix (row stride 32→33 ints) required. Per-kernel-call speedup **1.54×**, wall-clock **+4.4 %**.

### C2 retro — the "Amdahl-limited" framing was wrong

The initial C2 writeup blamed Amdahl for the modest wall-clock win. That was incorrect. Direct comparison against turbo's Q4_K MMQ kernel on the same workload:

| Kernel | per-call (ms) | vs turbo |
|---|---:|---:|
| turbo `mul_mat_q<Q4_K, 64>` | 1.58 | 1.00× |
| candle B2 single-warp (pre-C2) | 8.57 | 5.42× slower |
| **candle C2 multi-warp (current)** | **5.54** | **3.51× slower** |

Every input in turbo's `vec_dot_q4_K_q8_1_dp4a` comes from LDS (`x_qs`, `x_dm`, `x_sc`, `y_qs`, `y_ds`). My MW kernel does Y reads from HBM/L1 in the innermost loop — 8 cols × 8 sub-blocks = 64 times per thread per K-iter. Turbo amortises Y via a shared LDS tile loaded cooperatively once per K-iter.

**Projected C3 (port turbo's Y LDS tiling)**: if my kernel hits turbo per-call (1.58 ms × 234 calls = 370 ms), I save ~927 ms of GPU time, which translates to roughly **pp512 489 → ~680 t/s (0.67× turbo)**. There's another 1.4× end-to-end win available kernel-side, not gated by Amdahl.

### Honest accounting of this session's kernel work

- **B2 (single-warp)**: Shipped as "+2.3× pp" win. True, but the baseline `indexed_moe_forward_q4k_q8_1` is a bad kernel too. I replaced bad with less-bad; framing it as "physical gather-by-expert unlocked" overstated it.
- **B2 correctness**: Still no numerical test against reference. "Paris" + coherent prose ≠ validation. Task #74 description flagged this; I didn't fix it before moving on.
- **C2 LDS bank conflict**: My first MW draft hit 64-way bank conflicts (stride-32 access pattern) and got -1% vs single-warp. "Clean fix to stride 33" was 20 minutes of debugging that a proper implementation plan would have avoided entirely. Turbo's code has the `+1` pad built in by design (MMQ_TILE_Y_K_LDS etc.).
- **C2 MW pp128 regression**: Extra setup overhead under MW dominates at low avg_tokens/expert. Signals my kernel has more constant overhead than turbo's. Hiding behind a heuristic gate rather than fixing the root cause.
- **C2 projection**: I said "1.5-2× end-to-end". Measured 1.044×. I should have been honest that without Y LDS tiling, the MW rewrite was incomplete and the forecast was unfounded.

**What C2 is really worth**: a +4.4% end-to-end win that proves the register-pressure diagnosis was right but leaves the main implementation gap (Y LDS) unclosed. The framing that "next lever is attacking other kernels" is false — the biggest remaining win is still in the B2 kernel family.

Cumulative **2.61×** pp gain from B1→B2→C1. Decode unchanged throughout (expected — batching wins don't apply at tg where batch=1).

Post-C1 kernel profile (rocprof, pp=517 t/s under profiling overhead):
| Rank | Kernel | ms | % |
|---:|---|---:|---:|
| 1 | `mul_mat_q4_K_turbo_moe_x32_unchecked` (B2 kernel) | 2 004 | **48.3** |
| 2 | `mul_mat_q8_0_turbo_x64_unchecked` (attn QKV/out MMQ) | 898 | 21.6 |
| 3 | `indexed_moe_forward_q6k_q8_1` (lm_head Q6_K, per-token) | 389 | 9.4 |
| 4 | `gated_delta_net_step_s128_f32` (GDN recurrent) | 214 | 5.1 |
| 5 | `fillBufferAligned` | 110 | 2.6 |

`fast_sum_f32` is now absent from top-15 (was 569 ms / 11.8 % pre-C1). C1 delivered the predicted 400-500 ms save.

**Next single biggest lever**: the B2 Q4_K kernel itself is 48 % of remaining GPU time at 8.57 ms/call, vs turbo's equivalent `mul_mat_q<Q4_K, 64>` at ~1.58 ms/call — **5.4× per-call gap**. Root cause diagnosed: 188 VGPRs at x32 → **1 wavefront/SIMD occupancy** (gfx906 needs ≤128 VGPRs for 2 waves/SIMD, ≤80 for 3). The current kernel design (single-warp, each thread holds `sums[TILE_N]` + `sumf_d[TILE_N]` + `sumf_m[TILE_N]` = ~96 float accumulators per thread) can't fit in the register budget.

To close that gap: rewrite B2 as a 4-warp block matching turbo's template (distribute output tile across warps so per-thread accumulator count is `mmq_x * mmq_y / (4 * 64) = 32` floats, not 96). Estimated ~1-2 days, expected another 1.5-2× on that kernel → +~70 % pp end-to-end → 800 t/s (0.8× turbo).

## Top per-token deltas (µs/tok, candle --split-prompt pathological vs turbo) — historical reference

| Kernel family | Candle calls | Candle µs/tok | Turbo calls | Turbo µs/tok | Δ µs/tok | Commentary |
|---|---:|---:|---:|---:|---:|---|
| **rocBLAS Cijk gemm** | 80 080 | **10 828** | 196 | 88 | **+10 740** | 3 state matmuls × 30 GDN layers × 1129 tokens on candle, small tile `MT32x64x16` = per-token signature. Turbo avoids rocBLAS entirely for GDN. |
| **MMVQ (mul_mat_vec_q*)** | 260 799 | 6 376 | 916 | 44 | +6 332 | Candle uses MMVQ for attention proj + shared expert + router per-token. Turbo uses batched MMQ instead. |
| **MoE indexed (candle)** | 135 480 | 5 872 | 0 | 0 | +5 872 | `indexed_moe_forward_q4k_q8_1` fires 135 k× on candle; turbo batches MoE dispatches into `mul_mat_q<12,64>` (Q4_K MMQ). |
| **GDN step/kernel** | 33 870 | 3 044 | 120 | 226 | +2 818 | Candle `gated_delta_net_step_s128_f32` is per-token. Turbo `gated_delta_net_cuda<128>` + `ssm_conv_long_token_f32` are batched (120 = 30 layers × 4 ranks, 1 call per layer). |
| **copyBuffer / memcpy** | 746 943 | 2 847 | 447 | 5 | +2 842 | ~846 copies/tok on candle — cross-GPU activation moves + scatter/gather per token. |
| **sum / reduce** | 169 350 | 1 507 | 0 | 0 | +1 507 | `fast_sum_f32` — 267 per token. |
| **binary ops** (add/mul/div) | 343 216 | 1 306 | 1 040 | 47 | +1 259 | candle does these as separate kernels, turbo fuses into `k_bin_bcast` launched once per layer. |
| **quantize_q8_1** | 305 959 | 1 142 | 1 320 | 53 | +1 089 | candle re-quantizes per-token per-matmul; turbo quantises once per batched matmul. |
| **flash attn** | 5 120 | 1 005 | 0 | 0 | +1 005 | candle routes 10 full-attn layers through flash-attn-v2; turbo uses rocBLAS batched (in Cijk bucket). |

Families where **turbo spends more** (candle wins per-call or skips entirely):

| Family | Candle µs/tok | Turbo µs/tok | Δ |
|---|---:|---:|---:|
| MMQ batched (`mul_mat_q<ggml_type, 64>`) | 0 | 1 339 | −1 339 |
| concat | 0 | 71 | −71 |
| sgemm | 0 | 30 | −30 |
| mm_ids helper | 0 | 22 | −22 |
| ssm_conv | 0 | 18 | −18 |

Turbo spends **1 339 µs/tok on batched Q4_K/Q8_0 MMQ** — the single biggest cost on its side, processed 728 times across the whole 512-token prefill. Candle replaces that with ~395 000 MMVQ + indexed-MoE launches totaling ~12 250 µs/tok.

## Root cause (confirmed by this profile)

1. **Prefill is not batched on candle.** `quantized-qwen35` forces per-token forwards via `model.is_recurrent()` true. Every matmul in every layer is a `(1 × D) × (D × N)` gemv instead of a `(T × D) × (D × N)` gemm. The per-kernel overheads (dispatch + tile startup + quantize+dequant for Q8_1 activations) are paid 512× when they should be paid once.

2. **GDN state matmuls dominate (28.36 % / 10 828 µs/tok on candle).** The `Cijk_Alik...MT32x64x16...ISA906...WG8_16_1` fingerprint is a small-tile rocBLAS call — consistent with per-token 3-state matmuls (k@state, kᵀ@d, q@state) inside `delta_net_step`. Turbo's `gated_delta_net_cuda<128>` fuses these inside a single per-layer batched kernel.

3. **MoE dispatch is per-token on candle.** 135 k `indexed_moe_forward_q4k_q8_1` calls vs turbo's 230+494 `mul_mat_q<...>` calls. Turbo reorders tokens by routed-expert and runs a batched MMQ per expert group; candle iterates tokens and fires one indexed-MoE kernel per token per active expert.

## What this says about the roadmap (revised post-B1)

- **B1 turned out to be a phantom lever.** The original claim ("chunk≥2 OOMs") was wrong. The real fix was removing `--split-prompt` from the test harness — a one-line change in how benches are launched, no code edit. Default candle pp jumped 15.49 → 173.69 t/s (**11.2× free**) and GPU-sum fell from 41.9 s to 12.75 s. Dispatch count collapsed from 2.78 M to 11 510 (a 241× drop). The chunk env var still exists in the binary as a debug escape hatch but the default is full-prompt-in-one-forward.
- **New dominant lever: physical gather-by-expert MoE (former D1, promoted).** `indexed_moe_forward_q4k_q8_1` is now 81.6 % of GPU time (10.4 s). Turbo's equivalent (`mul_mat_q<Q4_K, 64>` after token reorder) is ~0.36 s. A 29× per-kernel gap concentrated in one kernel family — the clearest remaining target.
- **B2 (batch GDN state matmuls)** is validated: the 10 828 µs/tok in rocBLAS `MT32x64x16` is the GDN state matmul workload. Turbo's `gated_delta_net_cuda<128>` kernel proves it can be done in one batched kernel call per layer per rank. Port target: one fused batched GDN kernel over `chunk` tokens.
- **D1 (physical gather-by-expert MoE)** is validated. Turbo's approach (reorder-then-batched-MMQ) is ~2× faster per-token than candle's indexed-MoE even on per-token dispatch; batched, it would be much more.
- **C1 (attention kernel parity)** remains likely but is second-order here: flash-attn-v2 on candle is 1 005 µs/tok (10 full-attn layers × 1129 tok × ~89 µs) while turbo folds those into rocBLAS Cijk at similar cost. Only matters after B1/B2/D1 close the big gaps.
- **P2 (Q4_K MMQ kernel we landed)** still unused — turbo's top kernel is `mul_mat_q<12, 64>` = Q4_K MMQ batched over 512 tokens. That's the batched form our kernel was supposed to provide. Our dispatcher doesn't select it because the per-token workload never hits the batch threshold. B1 would give it a workload to fire on; D2 is a separate dispatcher-audit task.
- **P1 (fillBufferAligned)** is now **776 calls / 1 055 ms HIP-API time** on candle vs turbo's few dozen — but only 41.5 µs/tok in the GPU-kernel bucket. The HIP-API fillBufferAligned is host-side and mostly overlaps GPU. Low priority; confirms P1 was not the hotspot it looked like on the 17-tok profile.

## Turbo's top-20 (reference, for porting)

| # | Kernel | Calls | Total ms | Avg µs | % |
|---|---|---:|---:|---:|---:|
| 1 | `mul_mat_q<12, 64>` (Q4_K MMQ) | 230 | 364.4 | 1 584 | 34.83 |
| 2 | `mul_mat_q<8, 64>` (Q8_0 MMQ) | 494 | 308.1 | 624 | 29.45 |
| 3 | `gated_delta_net_cuda<128, false>` | 120 | 115.5 | 963 | 11.04 |
| 4 | `concat_f32_non_cont<0>` | 60 | 34.1 | 569 | 3.26 |
| 5 | `Cijk_Alik...MT128x64x16` (FP attn) | 78 | 19.8 | 254 | 1.89 |
| 6 | `gfx906_sgemm_tiled_fast` | 120 | 15.5 | 129 | 1.48 |
| 7 | `quantize_mmq_q8_1<1>` | 230 | 13.3 | 58 | 1.27 |
| 8 | `Cijk_Alik...MT32x128x32_SN` | 78 | 13.3 | 170 | 1.27 |
| 9 | `mul_mat_q<14, 64>` (Q6_K MMQ) | 4 | 13.0 | 3 250 | 1.24 |
| 10 | `k_bin_bcast<op_mul>` | 380 | 13.0 | 34 | 1.24 |
| 11 | `mm_ids_helper<8>` | 234 | 11.2 | 48 | 1.07 |
| 12 | `quantize_mmq_q8_1<0>` | 498 | 10.9 | 22 | 1.04 |
| 13 | `Cijk_Alik...HSS_BH_GB_MT64x32x8` | 20 | 9.1 | 456 | 0.87 |
| 14 | `ssm_conv_long_token_f32<true, 128, 4, 32>` | 60 | 9.0 | 150 | 0.86 |
| 15 | `mul_mat_vec_q<8, 1>` | 242 | 6.2 | 25 | 0.59 |
| 16 | `unary_gated_op_kernel<silu>` | 276 | 6.1 | 22 | 0.58 |
| 17 | `cpy_scalar` | 180 | 5.3 | 30 | 0.51 |
| 18 | `rms_norm_f32<256>` | 200 | 5.3 | 27 | 0.51 |
| 19 | `k_bin_bcast<op_add>` | 160 | 5.0 | 31 | 0.48 |
| 20 | `rms_norm_f32<1024>` | 324 | 4.6 | 14 | 0.44 |

Noteworthy ports candle is missing:
- `mul_mat_q<ggml_type, 64>` batched MMQ for Q4_K and Q8_0 (turbo spends 64 % of pp512 GPU time here). Our `mul_mat_q4_K_gfx906_v2f_*` kernel corresponds to (1), but only for the batched path — not reached on current qwen35moe per-token prefill.
- `gated_delta_net_cuda<128>` (batched GDN) and `ssm_conv_long_token_f32` (batched SSM conv).
- `mm_ids_helper<8>` = the physical "reorder-tokens-by-expert" step that enables batched-MMQ MoE.

## Gates unlocked / follow-up

- **B1/B2/D1 are all architecture changes**, not kernel tuning. None can be hit by patching existing per-token kernels.
- **A2 (fillBufferAligned attribution)**: deprioritised. 41.5 µs/tok in the kernel bucket; host-side 1055 ms mostly overlaps.
- **C1 (attention parity)**: defer until B1+B2+D1 land — attention is ~1 ms/tok on both sides and not the current bottleneck.
- Next concrete task: design B1 (MoE-aware chunked prefill). Touches `quantized_blocks/moe_experts.rs` (per-token fan-out API), `quantized_blocks/ffn.rs`, `delta_net.rs` (accept `chunk > 1` in forward), and the `quantized-qwen35` binary (drive chunk loop + honour `CANDLE_GDN_PREFILL_CHUNK` beyond the current 1).
