# Candle CUDA — Make All 8 GGUFs Runnable on RTX 3090 (24 GiB)

## Starting point (bench 2026-04-13, `scripts/bench-models-cuda.sh`)

| # | Model | File size | Status | Prefill t/s | Decode t/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| 1 | gemma-4-E4B-it-Q8_0                      |  8.2 GB | **ok**            | 1156.9 | 75.2 | 12 652 |
| 2 | gemma-4-31B-it-Q4_K_M                    | 18.3 GB | OOM (workspace)   |    —   |  —   | 24 047 |
| 3 | gemma-4-26B-A4B-it-MXFP4_MOE             | 16.6 GB | skip (MXFP4→F32)  |    —   |  —   |   —    |
| 4 | Qwen3.5-9B-BF16                          | 17.9 GB | OOM (BF16→F32)    |    —   |  —   | 24 025 |
| 5 | Qwen3.5-27B-Q4_1                         | 17.2 GB | **ok**            |  138.1 | 29.3 | 22 151 |
| 6 | Qwen3.5-35B-A3B-MXFP4_MOE                | 21.6 GB | skip (MXFP4→F32)  |    —   |  —   |   —    |
| 7 | Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0     | 17.4 GB | fail (MoE kernel) |    —   |  —   | 18 896 |
| 8 | Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL | 14.5 GB | fail (dtype) |    —   |  —   |    842 |

**Runnable today: 2/8.** Root causes cluster into three classes:

1. **Force-dequant-to-F32 at load** — `QMatMul::from_arc` at `candle-core/src/quantized/mod.rs:882–886` sets `dequantize = true` for `F32 | F16 | BF16 | Mxfp4`. A 16.6 GiB MXFP4 weight tensor becomes ~133 GiB F32 on GPU. Blocks models 3, 4, 6.
2. **Quantized + runtime workspace barely over 24 GiB** — gemma-4-31B Q4_K_M (model 2) peaks at exactly 24 047 MiB: 18.3 GiB weights + dequant workspace + KV cache + activations.
3. **Codebase feature gaps** — Qwen3-MoE gemm refuses Q4_0/Q4_1 (model 7, runtime error `moe_gemm_gguf ISQ only accept q2k, q3k, q4k, q5k, q6k or q8_0`); GGUF loader rejects an Unsloth "UD" dtype (model 8, `unknown dtype for tensor 23`).

**Goal:** 8/8 runnable, with decode t/s within 30 % of llama.cpp-CUDA (`b6950` master) on the same 3090 as a correctness/perf sanity floor.

---

## Phase C — CUDA all-models unblocker sequence

Phases are ordered by (unblocked models) / (engineering hours). C1 and C2 together unblock 4/6 of the currently-failing models and both deliver real runtime perf wins beyond just fitting.

### C1 — Native BF16 matmul, skip dequant *(unblocks Qwen3.5-9B-BF16, model 4)*

**Target:** BF16 weights stay as BF16 on GPU; route through cuBLAS `CUBLAS_COMPUTE_32F_FAST_BF16` (Ampere+ has BF16 tensor cores). Halves VRAM vs the current F32-dequant path.

**Projected gain:** 17.9 GiB model loads cleanly in ~18 GiB of BF16 weights + KV; expected ~35–45 t/s decode (bandwidth-bound at 2 × 936 GB/s ÷ 18 GiB).

- **C1a** — Drop `BF16` from the force-dequant arm at `quantized/mod.rs:882`. Gate on `device.is_cuda() && device.cuda_compute_cap() >= (8, 0)` so pre-Ampere still force-dequants.
- **C1b** — Wire `QTensor::BF16` → keep as raw BF16 buffer in `CudaStorage`; reuse the existing F16 matmul path with `dtype=BF16` or add a BF16 cuBLAS arm in `candle-core/src/cuda_backend/gemm.rs`.
- **C1c** — Parity test: cosine-similarity vs CPU reference on Qwen3.5-9B first-token logits with temp=0, seed=42.

**Effort:** 1 day. `cudarc` already supports BF16. Candle's `DType::BF16` storage path exists — this is mostly rewiring `QMatMul::from_arc` and the matmul dispatch.

---

### C2 — MXFP4 MMVQ kernel (keep 4-bit on GPU) *(unblocks models 3 + 6)*

**Target:** Add `mul_mat_vec_mxfp4_q8_1_cuda` and `mul_mat_mxfp4_q8_1` kernels so MXFP4 matmul runs in-place on the GPU, dequantizing per-block inside the kernel. Drops the `GgmlDType::Mxfp4 => true` arm in `from_arc`, letting MXFP4 follow the standard MMVQ path.

**Projected gain:** gemma-4-26B-A4B-MXFP4 fits in ~17 GiB (3 B active); Qwen3.5-35B-A3B-MXFP4 fits in ~21–22 GiB (3 B active). Decode throughput should be 2–3× the dequant-to-F32 path because weight bandwidth drops 8×.

- **C2a** — Port `mul_mat_vec_mxfp4_q8_1` from llama.cpp `ggml-cuda/mmvq.cu` into `candle-kernels/src/quantized.cu`. Reuse the `KVALUES_MXFP4` table and E8M0 scale decode that C0 (the now-landed `dequantize_block_mxfp4`) already introduced.
- **C2b** — Add the `GgmlDType::Mxfp4 =>` arm in `dequantize_mul_mat_vec` dispatch at `quantized/cuda.rs:238–248`. Remove the `Mxfp4 => true` force-dequant at `quantized/mod.rs:886`.
- **C2c** — MMQ prefill variant (`mul_mat_mxfp4_q8_1`) for long-context prefill. Lower priority — MMVQ alone makes both MoE models runnable.
- **C2d** — HIP port for gfx906 parity (same kernel, 64-wide wavefront, `v_dot4` lanes). Parked behind a follow-up since it unblocks the same models on MI50 multi-GPU setups.
- **C2e** — Correctness test: extend `candle-core/examples/cuda_mxfp4_correctness.rs` with an MMVQ round-trip (MXFP4 × Q8_1 → f32) vs CPU reference.

**Effort:** 3–5 days. The hardest part is the Q8_1 activation quantization wiring (already exists for Q4_0); the kernel itself is structural copy of `mul_mat_vec_q4_0_q8_1`.

---

### C3 — gemma-4-31B workspace trim *(unblocks model 2)*

**Target:** Get the 18.3 GiB Q4_K_M model under 24 GiB at runtime. It peaks at 24 047 MiB — ~1 GiB of headroom loss would unblock it.

**Projected gain:** model 2 runs; decode ~15–20 t/s (Q4_K_M bandwidth at 18.3 GiB).

- **C3a** — Profile peak allocation with `CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck` + `nvidia-smi --query-accounted-apps`. Identify which allocation pushes us past 24 GiB (suspects: (1) full-tensor f32 dequant workspace during load, (2) KV cache pre-allocated at max context, (3) activation buffers for Gemma4's unusually wide MLP).
- **C3b** — Stream dequant per layer — allocate one layer's worth of f32 workspace, reuse across layers. Currently `from_arc` → `dequantize_as_f32` allocates per-tensor; if `dequantize = false` (default for Q4_K), this path shouldn't fire, so the OOM is almost certainly KV or activations.
- **C3c** — Shrink KV cache default context from the currently-allocated max to a CLI-configurable value (`--max-context 2048` instead of model max 8192). Gemma4 has GQA + SWA; wasted KV for unused positions is likely the killer.
- **C3d** — If (a) and (c) aren't enough, expose a `--cpu-offload-lm-head` flag — the LM head is ~1.2 GiB F16 for gemma-31B and only used once per token.

**Effort:** 1–2 days. Mostly measurement + one small config change.

---

### C4 — Qwen3-MoE gemm accepts Q4_0 / Q4_1 *(unblocks model 7)*

**Target:** Remove the ISQ allow-list check that rejects Q4_0 in `moe_gemm_gguf`, or add a Q4_0 code path that mirrors the existing Q4_K implementation.

**Projected gain:** Qwen3-Coder-30B-A3B runs; decode ~25–35 t/s (3 B active × MMVQ).

- **C4a** — Find the error site: `grep 'ISQ.*only accept' candle-transformers/src/models/quantized_qwen3moe_blocks`. Understand why it's restricted (likely the MoE fused gemm kernel was only templated for K-quant scale/mins layouts).
- **C4b** — Pick one:
  - (i) **Preferred:** add Q4_0 / Q4_1 arms to the MoE gemm dispatch, reusing the per-expert MMVQ kernels that already support these dtypes for dense layers.
  - (ii) **Fallback:** load-time requantize Q4_0 → Q4_K_M for MoE experts only. Cheap, loses Q4_0's 12.5 % size advantage but reuses working kernels.
- **C4c** — Regression test: run a non-MoE Q4_0 model (TinyLlama-1.1B-Q4_0 is in existing benches) through the same kernel change to confirm no dense-path breakage.

**Effort:** 1 day for (ii), 2–3 days for (i).

---

### C5 — Unsloth Dynamic quant loader support *(unblocks model 8)*

**Target:** Parse the "UD-Q4_K_XL" GGUF, which mixes per-tensor quant types (Q6_K, Q5_K, Q4_K, some Q3_K) with an imatrix header. Error surfaces at `ggml_file.rs` dtype matching.

**Projected gain:** Devstral-24B runs; decode ~25 t/s (14.5 GiB weights, Mistral architecture).

- **C5a** — Identify the rejected dtype. Read tensor 23 raw from the GGUF and print its `ggml_type` value — likely `GGML_TYPE_Q4_K_XXS` (41) or a newer type added after this candle fork diverged from upstream. Cross-reference against `llama.cpp` `ggml.h` enum.
- **C5b** — Add the missing GGML type(s) to `GgmlDType` (at `quantized/mod.rs:320+`), the dtype-from-u32 decoder at `quantized/mod.rs:347`, the `type_size`/`block_size` tables, and the matching `BlockX` struct + `GgmlType` impl in `k_quants.rs`.
- **C5c** — Verify existing K-quant CUDA kernels handle the new type (most UD extensions are re-labelings of existing K-quant layouts with different scale packing).
- **C5d** — If the type genuinely is new (e.g., imatrix-aware Q4_K_XL), port the llama.cpp CUDA kernel for it into `quantized.cu`.

**Effort:** 1 day for a repackaged K-quant; 3–5 days if it's genuinely new math.

---

### C6 — Bench harness hardening

**Target:** Make `scripts/bench-models-cuda.sh` a stable regression tool, not a one-shot measurement.

- **C6a** — CSV output (one row per (model, run)) in addition to the markdown table, so results feed into `scripts/plot-*` tooling that already exists for HIP benches.
- **C6b** — Record load time separately. Bench currently conflates load with first-run warmup, which hurts visibility for the 18+ GiB models where loading is 30–60 s.
- **C6c** — Std-dev / min / max over N runs, not just mean. 29.3 t/s ± ? is ambiguous.
- **C6d** — Optional llama.cpp-CUDA baseline row per model (mirrors `bench-candle-vs-turbo.sh` pattern). Needs a `LLAMACPP_CUDA_BIN=...` env var.
- **C6e** — `--context-len <N>` flag plumbed through to runners — useful for C3's KV-size exploration and for bench reproducibility across different default contexts.
- **C6f** — Per-run `nvidia-smi dmon -s pucvmet` capture, saved alongside results. Gives us clock/power/PCIe correlation when investigating outliers.

**Effort:** 1 day total.

---

## Dependency graph & exit criteria

```
          C1 (BF16)  ─────────┐
          C2 (MXFP4 MMVQ) ────┤
 C6a/b/c ─────► bench re-run ─┼─► 7/8 runnable
          C3 (31B workspace) ─┤
          C4 (MoE Q4_0) ──────┤
          C5 (UD-Q4_K_XL) ────┘
                              │
                     All green └─► 8/8 + perf table in repo as
                                  BENCH-CUDA-3090-FULL-SWEEP-*.md
```

**Definition of done:**
- All 8 GGUFs in `models/` produce non-zero prefill and decode t/s under `scripts/bench-models-cuda.sh` with peak VRAM ≤ 23 500 MiB (leaves 1 GiB safety margin on the 24 GiB card).
- Results within 30 % of `llama.cpp-CUDA` master on the same 3090, measured via C6d.
- No new failures in HIP benches (phases H–K stay green on MI50).

---

## Not in scope

- **New models.** Scope is the pinned 8 in `scripts/download-models.sh`.
- **Multi-GPU.** Single RTX 3090 only. Multi-GPU HIP work continues in the existing `ROADMAP-ROCM-GFX906.md` track.
- **Mixed CUDA/HIP builds.** Feature gates stay orthogonal.
- **New quantization types.** Adding e.g. Q3_K_XXS for its own sake is deferred unless it lands as a side-effect of C5.
