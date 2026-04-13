# Survey: gfx906 Inference Forks — Candle vs Turbo vs vLLM-Mobydick

**Date:** 2026-04-13
**Hardware target:** AMD MI50 / MI60 / Radeon VII (gfx906), ~1 TB/s HBM2, no MFMA tensor cores, Wave64 SIMD
**Why this doc exists:** Before committing to Phase Q (G2/P integration + mat-vec kernel tuning) or the ROCm 7.2.1 migration, compile one reference of what the three active gfx906 forks do differently — so downstream decisions can reference this doc instead of re-deriving the landscape.

The three active players (as of April 2026):
| fork | language | purpose | status |
|---|---|---|---|
| **candle + HIP kernels** (this repo) | Rust + `.cu` | GGUF inference, single-batch | ours — actively developed |
| **llamacpp-turbo** `moriyasujapan/llamacpp-gfx-906-turbo` | C++ + `.cu` | GGUF inference, single-batch | decode leader (75 t/s E4B Q4_0) |
| **vllm-gfx906-mobydick** `ai-infos/vllm-gfx906-mobydick` | Python + Triton + `.cu` | serving (paged, batched) | continuation of archived `nlzy/vllm-gfx906` |

Historical / adjacent:
| fork | role |
|---|---|
| `nlzy/vllm-gfx906` | archived Feb 2026 — ancestor of mobydick |
| `mixa3607/ML-gfx906` | ROCm 7.2.1 + Tensile gfx906 kernel re-add — driver distro |
| `iacopPBK/llama.cpp-gfx906` | older gfx906 llama.cpp variant |
| `PowerfulGhost/vllm-mi50` | another vllm-gfx906 sibling |

Source inspection: all three active forks are on-disk here. Turbo at `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/`, vllm-mobydick at `/artefact/vllm-gfx906-mobydick/`, candle at `/artefact/candle/`.

---

## 1. Driver & toolchain stack

| fork | ROCm target | rocBLAS soname | libamdhip64 | build system |
|---|---|---|---|---|
| candle | 7.1.1 (current), 7.2.1+Tensile planned (`ROADMAP-ROCM-722-MIGRATION`) | `.so.5` | `.so.7` | `cargo` + `hipcc` via `build.rs` |
| turbo | 7.1.1 per `bench-results/bench_20260408_105127.md:4` | `.so.5` | `.so.7` | `CMake` + `hipcc` |
| vllm-mobydick | **6.3.4** pinned, `README.md:4-12` | `.so.4` | `.so.6` | `CMake` + `hipcc` + Triton JIT |

Why the split: vllm-mobydick pinned ROCm 6.3.4 because it's the last version where AMD shipped complete Tensile gfx906 kernels pre-compiled. Under 7.1.1 the kernel library is incomplete — rocBLAS falls back to comgr runtime JIT for a few ops (notably `sgemv_strided_batched`), which segfaults on launch with `SI_KERNEL` / `si_addr=NULL`. We verified this on candle Phase O: the same binding that crashes under 7.1.1 runs clean under 6.3.4 (commit `2123e649` postmortem). Turbo hasn't hit this because its decode path doesn't use sgemv — it rolls its own `mul_mat_vec_f` HIP kernel. Community consensus per [ROCm GPU wishlist discussion](https://github.com/ROCm/ROCm/discussions/3893) is that gfx906 entered "maintenance" in ROCm 6.0 and 6.3.x is the last blessed version; 6.4+ works with minor modifications; 7.x is where the kernel library gaps show up.

mixa3607/ML-gfx906 solves the problem at the driver layer by shipping a `rocm-tensile` package that backports the gfx906 Tensile kernel files onto ROCm 7.2.1 — gives you the newest rocBLAS API AND a complete kernel library in one stack. Our `ROADMAP-ROCM-722-MIGRATION` proposes migrating here.

## 2. KV cache layout

| fork | K storage | V storage | note |
|---|---|---|---|
| candle (default) | `(B, H_kv, D, max_T)` D-major, T contiguous | `(B, H_kv, max_T, D)` | `KvCache::new_k_transposed`, seq on last dim for K |
| candle (Phase P opt-in, `CANDLE_KV_TMAJOR=1`) | `(B, H_kv, max_T, D)` T-major, D contiguous | same | `KvCache::new_k_canonical_stable`, matches llama.cpp |
| turbo | `(B, H_kv, T, D)` T-major (ggml standard) with TURBO2/3/4 sub-byte compression | `(B, H_kv, T, D)`, optional TURBO compression | F16 "shadow cache" materialized lazily — `fattn.cu:117-200` [turbo src] |
| vllm-mobydick | **paged**: `(num_blocks, block_size, head_size)` with block-indirection table | same | `mla_attention.py:1193` — DeepSeek-V3.2-style sparse MLA supported |

Candle's default D-major layout was a Phase G1 optimization: pre-transposing K at cache-append time skipped the per-call `K^T.contiguous()` inside flash-attn-v2. Paid off massively on TinyLlama (which uses rocBLAS attention reachable via D-major K) but hurts the mat-vec attention path we now want (which needs D contiguous). Phase P flips to T-major at the cost of re-introducing a per-prefill-call K transpose on the slower path — net +3.9% on gemma4-E4B decode. See `ROADMAP-MULTI-MODEL-2026-04-13.md` Phase P section.

Turbo's T-major-with-sub-byte-compression is the most advanced KV storage in any of these forks. TURBO3_0 (ggml-common.h:247-251) stores 32 values in 16 bytes using a 3-bit codebook (2 low bits + 1 sign bit, Lloyd-Max centroids at `{-0.190685, -0.117832, -0.065717, -0.021460, +…}`). TURBO2_0 (10 bytes / 32 vals, 4-value codebook) is harsher; TURBO4_0 (68 bytes / 128 vals, PolarQuant + QJL signs) is near-lossless. All three produce "byte-identical output at 2-3 bit precision on Gemma 3 4B" per the [llama.cpp TurboQuant discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) community reports (gfx1100 port discussion at [#21526](https://github.com/ggml-org/llama.cpp/discussions/21526)).

vLLM's paged KV with block_tables is the standard PagedAttention contract — `[num_blocks, block_size, num_kv_heads, head_size]` logical shape, physical storage fragmented. mobydick keeps this as-is from upstream; they don't optimize the layout for gfx906, they optimize the kernels that read it.

## 3. Attention dispatch

| fork | prefill L_q≥4 | decode L_q=1 | fusion |
|---|---|---|---|
| candle | `flash_attn_v2_fwd_ktvs_d{64,128,256,512}_f32` (LDS-tiled, K_TRANS=true) | `flash_attn_v2_fwd_ktvs_dyn_*` (D-major) OR `gqa_decode_mv_d{64..512}_f32` (T-major Phase P) | single fused kernel (scale+mask+softmax+GQA) |
| turbo | `fattn-tile.cuh`: D ∈ {40,64,72,80,96,112,128,256,512,576} `[fattn-tile.cuh:142-176]` | `fattn-vec.cuh`: D ∈ {128,256} L_q=1 with 128-thread blocks `[fattn-vec.cuh:46,78]` | separate `mul_mat_vec_f` + `soft_max_f32` + `mul_mat_vec_f` (unfused but each tuned) |
| vllm-mobydick | Triton-codegen page'd attention | Triton-codegen with gfx906 register tuning `[triton_decode_attention.py:37-38]` | fused in Triton; sparse MLA variant for DeepSeek-V3.2 |

Turbo's key dispatch insight: **on MI50 / gfx906 (no MFMA, no WMMA RDNA3), the `BEST_FATTN_KERNEL_TILE` path wins for non-trivial Q batches — but plain `mul_mat_vec_f` + `soft_max_f32` wins for L_q=1 decode** (`fattn.cu:524-760` picker). Without `-fa` flag it disables flash-attn entirely and goes full gemv+softmax split — we measured this mode gets 0.98 ms/tok attention (0.76 Q·K^T + 0.22 softmax + V·attn) vs candle's `flash_attn_v2_fwd_ktvs_d256+d512` at 3.98 ms/tok. Phase P's `gqa_decode_mv_*` closes this to 3.05 ms/tok (~75% of the gap); further closing needs Phase Q2.

vLLM-mobydick's attention tuning is entirely at Triton level — they don't write CUDA kernels, they tune block sizes and pipeline stages per arch: `BLOCK = 16 if on_gfx906() else 8` `[triton_decode_attention.py:37]`, `num_stages=1` forced on gfx906 to avoid LDS conflicts `[triton_decode_attention.py:438]`, `BLOCK_D=32` for MLA sparse `[rocm_aiter_mla_sparse.py:403]`. They report "2.3× faster than torch reference" `[rocm_aiter_mla_sparse.py:979]`. The whole MQA logits kernel is explicitly labelled "optimized for MI50 gfx906" `[rocm_aiter_mla_sparse.py:728]`.

## 4. Quantization paths

| fork | Q4_0 MMVQ | Q4_K MMVQ | K-quant warp-coop | sub-byte KV | FP8 / sparse |
|---|---|---|---|---|---|
| candle | ✓ v2 / v2f split-accumulator, 128-bit vec loads `[quantized.cu]` | ✓ warp-coop Q4_K/Q5_K/Q6_K (64 threads, 2 rows/block, DPP half-warp — opt-in via `CANDLE_KQUANT_WARP_COOP=1` because broken on MI50+ROCm 7.1.1) | ✓ | ✗ | ✗ |
| turbo | ✓ `mmvq-q4_0.cuh` + `mmvq-q4_1.cuh` + `mmvq-q8_0.cuh`, dp4a (`v_dot4`) | ✓ standard | ✓ | **✓ TURBO2/3/4 KV with lazy F16 shadow cache**, `convert.cu:765+`, `cpy.cu:549`, `fattn.cu:182-500` | ✗ |
| vllm-mobydick | via Triton generic matmul | via Triton | — | ✗ (uses paged f16/fp8 KV, not sub-byte) | ✓ FP8 via aiter `[_aiter_ops.py]`, GPTQ/AWQ/Quark via `amd-quark>=0.8.99` |

Turbo's TurboQuant KV is the single biggest memory-efficiency advantage they have. On gfx906 though, [community reports](https://github.com/ggml-org/llama.cpp/discussions/20969) note the prefill overhead is noticeable (vs ~1% on RDNA3) — an MI50/MI60 is **bandwidth-starved per compute** (10% HBM utilization at batch=1 per those threads), so the dequant-to-f16-shadow step bites harder than on RDNA3. Gains are in memory footprint (up to 5.2× KV cache reduction) at ~5-10% decode throughput cost on gfx906.

Candle's K-quant warp-coop implementation is in-tree but gated off (`CANDLE_KQUANT_WARP_COOP` default off per Phase J bug postmortem — DPP-fused half-warp reductions crash on MI50 + ROCm 7.1.1, root cause likely identical to the rocBLAS comgr-JIT bug affecting sgemv). Turbo avoids this by not relying on DPP for the K-quant path — they use `dp4a` (`v_dot4`) for the inner product which is hardware-intrinsic on gfx906 and independent of the broken reduction path. Our B3 implementation mirrors turbo's Q4_K/Q5_K/Q6_K design; the DPP reduction is our last-mile bug.

vLLM's quant story is orthogonal — they get FP8 KV, GPTQ/AWQ weights via aiter/Quark, and sparse MLA via the MoE path. None of those land on dense Q4_0 single-batch decode where our workload sits.

## 5. Host-side / system machinery

| fork | launch-capture replay | CUDA/HIP graphs | paged mem | batching |
|---|---|---|---|---|
| candle | ✓ **G2 decode op cache** — stable-addr alloc pool + per-launch arg recording + replay with counter/external patches; novel among these three (`ROADMAP-MULTI-MODEL:K13`) | ✓ G3 HIP graph capture on top of G2 | ✗ | single-batch only |
| turbo | ✓ standard ggml graph capture | — | ✗ | single-batch |
| vllm-mobydick | — | ✓ CUDA graphs per seq-length bucket | ✓ PagedAttention block_tables | ✓ continuous batching, speculative decoding, prefix cache, chunked prefill |

Candle's G2 is genuinely novel in this space — it records every `hipModuleLaunchKernel` argument tuple in a plan, then on replay patches three classes of args:
- **Counter** args (index_pos, current l_k) advance +1 per token via baked delta,
- **External** args (per-token input pointers, masks from prelude) get patched to fresh addresses,
- **Fixed** args (weight pointers, strides) stay as-recorded.

That lets us skip the entire CPU-side tensor graph rebuild between tokens. +29% decode on TinyLlama Q4_0 (190 → 245 t/s, beats turbo) is attributable to G2+G3. The mechanism fails on gemma4 (regresses instead) because flash-attn-v2-ktvs is more expensive than the launch-overhead savings G2 buys — which is exactly the Phase P motivation.

vLLM's machinery is orthogonal: they optimize for multi-request serving (paged KV + continuous batching), which we don't do and turbo doesn't do. For single-batch decode none of vLLM's infrastructure helps directly; for serving, candle and turbo are effectively not in the running.

## 6. Hot-path primitives

| fork | warp reduce (sum, max) | exp/rcp | LDS padding | dp4a | RoPE |
|---|---|---|---|---|---|
| candle | `__shfl_xor` portable loop, ~12 cycles/64-wide, DPP **disabled** per Phase J postmortem (broken on MI50+ROCm 7.1.1) | SFU (`v_exp_f32`, `v_rcp_f32`) ✓ | **untuned** (no `+1` padding observed) | ✓ in MMVQ inner products | generic `rope_f32` kernel |
| turbo | **DPP** `v_add_f32_dpp` / `v_max_f32_dpp` with `quad_perm`, `row_ror`, `ds_swizzle` crossbar — ~7 cycles/64-wide; `gfx906-common.cuh:73-93` | hand-coded | `cpy.cu:59`: `__shared__ float tile[2][TS][TS+1]` +1 col padding on transpose — confirmed only one site we saw | ✓ `v_dot4` in `vecdotq.cuh:64-67` | RoPE kernel in `gfx906/attention/rope.cuh`, purpose-built |
| vllm-mobydick | via Triton codegen (hand-tuned block size per gfx906) | Triton | Triton codegen handles it | via aiter | via Triton/aiter |

Turbo's DPP reductions are a ~40% speedup on every reduction hotpath vs our `__shfl_xor` loop. We hit the DPP intrinsic crash on MI50 + ROCm 7.1.1 and had to gate it off — same bug class as the K-quant warp-coop issue and the rocBLAS sgemv issue. All three reveal that ROCm 7.1.1's compiler output for certain gfx906-specific intrinsics is broken. ROCm 6.3.4 or 7.2.1+Tensile should fix all three in one migration.

Wiki source: [skyne98/wiki-gfx906 LDS layout study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-lds-layout-standard-llm.html) — the definitive gfx906 reference — measures 2× LDS bandwidth from `+1 vec4` padding on column-consumed tiles (3974 GB/s padded vs 1865 GB/s unpadded). Turbo applies this to exactly one kernel (cpy); our v2 flash-attn LDS tiles and the MMQ tiles would also benefit. Tracked implicitly under Phase Q2.

## 7. Per-model measured performance

Gemma4-E4B Q4_0, 52-tok prompt + 63 decoded, single MI50:

| fork | prefill | decode |
|---|---|---|
| candle default | 244 t/s | 52 t/s |
| candle + Phase P opt-in | 244 | **54** (+3.9%) |
| candle + G2 dyn_lk | 244 | 46 |
| turbo (no `-fa`) | ~250 | **75** |
| vllm-mobydick (untested on E4B Q4_0 GGUF) | — | — |

TinyLlama-1.1B Q4_0, 556-tok prompt + 256 decoded, single MI50 (memory, earlier runs):
| fork | decode |
|---|---|
| candle default | 203 t/s |
| candle + G2 + G3 | **261** |
| turbo | 212 |

Qwen3-32B-AWQ (not directly comparable — vllm's native format) on vllm-mobydick: 38-43 t/s decode, 50-92 t/s batched, per [community reports](https://github.com/nlzy/vllm-gfx906).

Pattern: **candle wins TinyLlama decode** (G2 machinery amortizes well on small model with cheap attention); **turbo wins gemma4 decode** (attention is the bottleneck; their gemv+softmax approach on T-major K beats our flash-attn-v2-ktvs on D-major K); **vllm wins serving** (paged + batched). No one fork dominates across the full workload matrix.

## 8. What turbo has that candle doesn't

1. **DPP warp reductions**, unbroken — our Phase J postmortem disabled DPP after the MI50+ROCm 7.1.1 miscompile. Gain: ~40% reduction perf everywhere in the codebase. Unlock: ROCm migration.
2. **TurboQuant KV compression** — 2-5× KV memory reduction at ≤5% decode perf cost. Useful for long-context scenarios (32K+).
3. **Full fattn-tile coverage** including d=40, 72, 80, 96, 112, 576 — we only hit d=64/128/256/512. Minor in practice (current models fit 256 or 512) but matters for future model families.
4. **Hand-coded LDS `+1` padding** on transpose kernels — we haven't audited our LDS layout for bank conflicts. Phase Q candidate.
5. **Graph fusion engine** — turbo detects RMS_NORM→MUL→MUL_MAT chains and fuses norm+scale into the prequantized Q8 matmul at graph-compile time (`gfx906/fused/graph-fusion.cuh:1-100`). We fuse ad-hoc (Phase D1, D2) but don't have a graph-level pattern detector.
6. **Persistent F16 shadow cache** for turbo-quant KV — lazy materialization with write-through from SET_ROWS `[fattn.cu:117-200]`. Enables sub-byte KV without paying dequant cost on every decode step.
7. **Published benchmark bundle** — `bench-results/` has documented runs across multiple configs + models; reproducible comparison points.

## 9. What vLLM-mobydick has that candle doesn't

1. **PagedAttention** — block-indirection KV cache, essential for serving arbitrary-length requests without over-allocation.
2. **Continuous batching** — dynamic request admission without restarting the graph; serving throughput scales with batch.
3. **Speculative decoding** — reuses vLLM's upstream speculator framework.
4. **Prefix cache** — reuses the common system-prompt prefix KV across requests `[config/vllm.py:1618]`.
5. **FP8 / GPTQ / AWQ / Quark** quant support via aiter — wider quant matrix than our Q4_0/Q4_1/Q5_K/Q6_K/Q8_0.
6. **Sparse MLA** for DeepSeek-V3.2 — sparse attention over an index table.
7. **Triton codegen** — portable kernel tuning; they get arch-specific perf without writing HIP. Easier to maintain cross-arch.

## 10. What candle has that others don't

1. **G2 decode op cache + G3 HIP graph capture** with stable-address KvCache + Counter/External arg-patching. Distinct from CUDA graphs because it captures at the `hipModuleLaunchKernel` arg-tuple level and patches per-replay without re-recording. Novel in this space. Delivers +29% decode on TinyLlama.
2. **Rust type-safety across the full inference stack** — kernel dispatch, tensor shapes, device placement, quantization dtype all compile-checked. Turbo is C++; vllm is Python+Triton. Different maintainability tradeoff.
3. **Single static binary** — `cargo build --release` yields one executable with embedded `.hsaco` kernels. Turbo needs the llama.cpp runtime scaffolding; vllm needs a full Python + Triton + Torch runtime.
4. **Pinned-memory fused GGUF load path** (recently merged: commit `39c3af3c feat(cuda): enable the pinned GGUF model collection on RTX 3090`) — `GgufBlob::read_to_vec` with `pread` + pinned staging + double-buffer H→D; `load_quantized_concat_from_blob` fuses QKV weight reads into one slab. Measured 17GB Qwen3.5-27B-Q4_1 in 32s on SATA SSD (713 MB/s disk cap), projects to 7s on NVMe. Not present in turbo or vllm.
5. **Dynamic-L_k flash-attn kernel variants** (`flash_attn_v2_fwd_ktvs_dyn_*`) — separates iterate bound from stride so G2 replay can advance l_k as a Counter arg. Makes the G2 padded-n_kv pattern work with correct attention semantics.

## 11. Recommendations — feeding Phase Q and driver migration

Ranked by payoff × certainty × effort:

1. **Migrate to ROCm 7.2.1+Tensile** via mixa3607's `rocm-tensile` package (existing `ROADMAP-ROCM-722-MIGRATION`). Unlocks:
   - DPP warp reductions (~40% on reduction hotpath, applied codebase-wide)
   - K-quant warp-coop (Phase B3 was gated off waiting for this)
   - rocBLAS `sgemv_strided_batched` — unblocks Phase O if we revisit it (probably obsolete given Phase P)
   Effort: ~1 day for clean rebuild + regression bench across all models.

2. **Port turbo's LDS `+1` padding pattern** to our flash-attn-v2 LDS tiles and MMQ v2f. Wiki measures 2× LDS bandwidth. Effort: ~2-4 hours per kernel.

3. **Tune `gqa_decode_mv` kernel (Phase Q2)** following turbo's `mul_mat_vec_f` template:
   - 128-thread block (2 warps) vs our 64.
   - `float2` vectorised loads (match `mmvf.cu:132`).
   - Split-K across CUs for small batch (currently 8-32 blocks = partial occupancy on 60 CUs).
   Target: 73 μs/call → ~30 μs/call for d=256. Expected +3-4 t/s on E4B decode.

4. **Consider TurboQuant KV port (Phase M or later)** — 5× KV memory reduction, enables 32K+ context on 16GB MI50. The gfx906 decode-throughput cost (~5-10%) is real but acceptable for the memory win. Blocked on Phase Q first because TurboQuant layers atop attention kernels.

5. **Graph fusion engine (optional)** — pattern-matching norm→mul→matmul chains at plan build time. Our per-layer fusion (Phase D1/D2) already covers the hot cases; a general engine would help future model families. Deprioritised vs above.

6. **Phase Q1 — G2 + Phase P integration**. Today Phase P is opt-in because G2 regresses under canonical K. Making G2 tolerate the new dispatch is bookkeeping-heavy but known-achievable. Priority: medium — G2 is an opt-in feature; users aren't forced to hit the incompatibility.

What NOT to pursue:
- **Don't port PagedAttention / continuous batching** unless the product shifts to serving. Candle's comparative advantage is single-binary + G2 for latency-sensitive single-batch; duplicating vLLM's serving stack dilutes that.
- **Don't port Triton codegen** — it would trade type-safety and the G2 capture machinery for portable kernel tuning we can approximate by hand on gfx906.

---

## Sources

### On-disk sources (authoritative)
- Turbo: `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/ggml/src/ggml-cuda/` — fattn.cu, fattn-tile.cuh, fattn-vec.cuh, mmvf.cu, gfx906/gfx906-common.cuh, gfx906/fused/graph-fusion.cuh, convert.cu, cpy.cu
- vllm-mobydick: `/artefact/vllm-gfx906-mobydick/` — README.md, vllm/platforms/rocm.py, vllm/v1/attention/ops/rocm_aiter_mla_sparse.py, vllm/v1/attention/ops/triton_decode_attention.py, _aiter_ops.py, csrc/
- Candle roadmaps: `/artefact/candle/ROADMAP-MULTI-MODEL-2026-04-13.md`, `/artefact/candle/ROADMAP-ROCM-722-MIGRATION-2026-04-13.md`, `/artefact/candle/REVIEW-CANDLE-VS-TURBO-HIP-KERNELS-2026-04-12.md`

### Community sources (web)
- [TurboQuant — Extreme KV Cache Quantization (ggml-org/llama.cpp #20969)](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [TurboQuant KV Cache Compression — HIP/ROCm Port (#21526)](https://github.com/ggml-org/llama.cpp/discussions/21526)
- [Performance of llama.cpp on AMD ROCm (HIP) (ggml-org/llama.cpp #15021)](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [MI50+ROCm7.2+Qwen3.5 segfault (ggml-org/llama.cpp #19975)](https://github.com/ggml-org/llama.cpp/issues/19975) — same ROCm 7.x instability class we hit
- [Support gfx906 GPUs for 8+ years (ROCm/ROCm #3893)](https://github.com/ROCm/ROCm/discussions/3893) — gfx906 entered ROCm 6.0 maintenance
- [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) — ROCm 7.2.1 + Tensile gfx906 kernels
- [nlzy/vllm-gfx906 RELEASE.md](https://github.com/nlzy/vllm-gfx906/blob/gfx906/main/RELEASE.md) — mobydick's ancestor
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) — alternate gfx906 llama.cpp fork
- [Pascal-SAPUI5/llama.cpp-turboquant](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant) — TurboQuant AMD ROCm port
- [skyne98/wiki-gfx906 LDS layout study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-lds-layout-standard-llm.html) — definitive gfx906 engineering reference (LDS bandwidth, KV layout)
- [skyne98/wiki-gfx906 KV-Cache study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-kv-cache-read-write-study.html) — confirms Phase P K layout rationale (4.75× read bw HSD vs HDS)
