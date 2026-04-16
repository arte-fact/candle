# Roadmap: Beat llama.cpp-turbo on Gemma-4-26B-A4B Decode

**Date:** 2026-04-16 (rev. 2 — post kernel-family analysis)
**Model under test:** `gemma-4-26B-A4B-it-Q8_0.gguf` on 4× MI50 (`-ts 1/1/1/1`)
**Profile tooling:** `rocprofv3 --runtime-trace --stats` + `bash time` (sys/user)

## Starting point

| metric | candle | turbo | ratio |
|---|---:|---:|---:|
| pp512 t/s (uninstrumented, `candle-bench`) | 566.11 | 899.46 | 0.63× |
| **tg128 t/s (uninstrumented)** | **43.84** | **66.74** | **0.66×** |
| aggregate GPU kernel time (rocprofv3) | 10.37 s | 3.53 s | **2.94×** more work |
| wall s under rocprofv3 | 67.5 | 29.7 | 2.27× |
| user CPU s | 122.6 | 45.7 | 2.68× |
| **sys CPU s** | **282.4** | **9.4** | **30.0×** |

## Goal

**Beat turbo on tg128. Target ≥ 70 t/s (= 1.05× turbo), stretch 75 t/s (= 1.12×).**

Secondary:
- **zero** `hipMallocAsync` during decode (13,181 → 0)
- sys CPU < 15 s under rocprofv3 (30× → ≤ 1.6×)
- hold every kernel family where candle already wins (MMVQ-Q8, MMQ-Q8, flash-decode split-L_k, RoPE, Q8-quant)

## Why beating turbo is feasible

Per-kernel, **candle already leads or ties on 7 of 12 families**. The deficit is concentrated in three places with clear engineering fixes, and candle has one optimisation turbo does not have (G2/G3 replay) that can knock CPU overhead to essentially zero once we light it up on this model.

### Family-level ledger (aggregate µs across 4 GPUs)

| family | candle | turbo | **winner** |
|---|---:|---:|---|
| MoE FFN | 6,572 ms | 983 ms | **turbo 6.7×** |
| Bookkeeping/copies | 946 ms | 64 ms | **turbo 14.8×** |
| Elementwise arithmetic | 631 ms | 168 ms | **turbo 3.8×** |
| topk + router | 217 ms | 55 ms | **turbo 4.0×** |
| Decode MMVQ (Q/K/V/O + dense FFN) | 917 ms | 1,404 ms | **candle 1.5×** |
| Prefill MMQ Q8 | 533 ms | 925 ms | **candle 1.7×** |
| Q8_1 activation quantize | 135 ms | 176 ms | **candle 1.3×** |
| RoPE | 36 ms | 48 ms | **candle 1.3×** |
| RMSNorm | 244 ms | 278 ms (sans outlier) | **candle 1.1×** |
| Activation (GeLU) | 43 ms | 45 ms | tied |
| Flash-attn decode | 198 ms | ≈196 ms | tied |
| rocBLAS prefill-attn | 55 ms | 66 ms | tied |

Preserving the candle-wins and wiping the three turbo-wins gets us past 1.0× turbo on its own. Lighting up G2/G3 replay pushes us decisively ahead.

## Root-cause map (with per-fix expected decode lift)

| root cause | evidence | fix | expected tg lift |
|---|---|---|---:|
| MoE fused kernel **serializes experts** | `indexed_moe_forward_q8_0_q8_1` 818 µs/call vs turbo's parallel MMVQs at 34 µs | **W3** defuse into gather → per-expert MMVQ → scatter | **+10–15 pp** |
| 13,181 `hipMallocAsync` during decode | 25 s CPU + most of the 282 s sys time | **W1** decode alloc pool | +8–12 pp |
| `fast_sum_f32` alone burns 358 ms on router normalise | single-purpose kernel after `topk_moe_norm` | **W3a** fuse into `topk_moe_norm` | +2–3 pp |
| `ucopy_f32` + `__amd_rocclr_copyBuffer` = 880 ms GPU | defensive copies around in-place ops | **W5** kill aliasing-copies | +3–5 pp |
| Zero peer-to-peer, host-bounce cross-GPU | 0 `hipMemcpyPeerAsync` vs turbo's 393 | **W2** wire `MemcpyPeerAsync` | +4–6 pp |
| 8 elementwise kernels where turbo has 1 | candle 631 ms vs turbo 168 ms via `k_bin_bcast` parametric | **W6** parametric binary-broadcast kernel | +3–5 pp |
| 1,128 `hipMemsetAsync` × 8 ms = 9 s | residual `alloc_zeros` not caught by O3 | **W4** trace + remove | +2–3 pp |
| CPU dispatches 200k kernels per decode run | no graph replay on Gemma4-26B-MoE (Phase K was E4B only) | **W7** extend G2/G3 to 26B MoE | +5–10 pp on top of W1 |

Phases are not strictly additive (W1 and W7 both attack CPU overhead, W3 and W3a both attack MoE), but even a conservative sum clears the 1.0× turbo line.

---

## Phase W — order of attack

Reordered from rev. 1 by **impact** rather than effort. W3 and W1 are the two biggest levers and should land first; everything else compounds on top.

### W3 — Physical gather-by-expert MoE **[NOT A WIN on Gemma-4; retained as opt-in]**

**Status: landed as opt-in, default OFF (2026-04-16).** Performance measurement invalidated the original premise.

**What was implemented** (`CANDLE_W3_GATHER=1` to enable):
- `launch_mul_mat_vec_q8_1_chunk_raw` view-based MMVQ dispatcher (`candle-core/src/quantized/hip.rs`).
- Per-expert branch inside `launch_indexed_moe_forward_with_q8` that loops `topk` times calling the warp-coop MMVQ kernel on expert-sliced weights. Covers both `input_dim1 == 1` (gate/up) and `input_dim1 == topk` (down).
- Thread-local CPU-ids cache (`w3_cached_ids_set/take/clear`) in the same file, wired in `quantized_gemma4.rs` and `quantized_blocks/ffn.rs` — avoids per-projection D2H syncs.
- `hipdarc::Stream::clone_dtoh_view` helper.
- Extended `candle-core/examples/hip_moe_mmq_test.rs` with tokens=1 decode rows — rel-err ≤ 3e-6 vs fused kernel (essentially bit-exact).

**Why it doesn't win on Gemma-4-26B-A4B-Q8_0 decode:**

The roadmap assumed fused-kernel per-call = 818 µs based on `7800 calls × 818 µs = 6.38 s` from the 2026-04-16 rocprofv3 baseline. That average is **dominated by the 120 prefill calls at ~46 ms each (5.52 s)** — the 7,680 decode calls averaged only **~79–112 µs**. The per-decode-call cost was already competitive.

Per equivalent work on decode:
| path | per call | equivalent decode cost |
|---|---:|---:|
| fused `indexed_moe_forward_q8_0_q8_1` | ~112 µs | 112 µs |
| 8× warp-coop MMVQ | 17 µs | **136 µs (21 % slower)** |

Measured outcome (uninstrumented, 4-GPU, pp512+tg128, best-of-3):

| | W3 off (W3a baseline) | W3 on |
|---|---:|---:|
| pp512 | 655 t/s | 639 t/s (−2.4 %) |
| **tg128** | **44.81 t/s** | **42.83 t/s (−4.4 %)** |

Regression gate: `mmvq_q8 +69 %` (PRESERVE family FAIL).

**When it might help:** the opt-in path is kept because shape/tile economics differ per model. Qwen3-Coder-30B-A3B at Q4_0 has smaller `n * k` per expert; the 8×-MMVQ path might pay off there. To validate, run `CANDLE_W3_GATHER=1 candle-bench bench --backend candle --bin quantized-qwen35 --model <qwen-gguf> --gpus 0,1,2,3 --prompt-len 512 --tg-len 128` and compare.

**Lesson** (added to MEMORY feedback): aggregate-averaged per-call µs on profile summaries can conflate prefill and decode. Split per-call-µs by `input_dim1` / `b_size` before projecting wins.

**Remaining levers:** W3's planned +10–15 pp on tg is off the table. The roadmap's 1.05× turbo goal now rests on **W1 (alloc pool), W5 (ucopy), W6 (elementwise), W7 (G2/G3)** — all CPU-overhead fixes. W3 artefacts remain in tree as the retained escape-hatch for future models.

**Implementation:**
- **W3a — LANDED 2026-04-16.** Fused `moe_combine_weighted_sum_f32` replaces
  `broadcast_mul(topk_weights) + sum(1)` in `quantized_gemma4.rs:2021-2023`
  (kernel in `candle-hip-kernels/src/topk_moe.cu`, Rust dispatcher in
  `candle-core/src/hip_backend/fused_pointwise.rs`). Kill-switch
  `CANDLE_W3A_COMBINE=0`. **Measured:** `router_topk` -60.8%,
  `elementwise` -17.2%, aggregate GPU -4.7%, all PRESERVE unchanged.
  Uninstrumented: pp512 **+15.6%** (566→655 t/s, fast_sum was launching
  2560×T blocks on prefill), tg128 **+2.2%** (43.84→44.81 t/s, modest
  because decode overlapped it with MoE-fused on critical path). Slight
  numerical drift from reduction-order change is expected.
  *Originally scoped to "fuse fast_sum into topk_moe_norm" — pivoted to
  fusing the combine-sum (line 2023) instead after discovering
  `topk_moe_norm_f32_n128` already normalises in-kernel. The actual
  `fast_sum_f32` was the post-expert combine, not the router sum.*
- **W3b** — New `gather_by_expert_q8_0` kernel: input `(x_q8_1: (N,D), routing_idx: (N,topk))` → output `(x_by_expert: (E, M_e, D), offsets: (E+1,))`. One workgroup per `(expert, row_block)`. Reuse gfx906 primitives (`gfx906_primitives.cuh`).
- **W3c** — Rewrite `MoeExperts::forward_shared_q8_decode` (`candle-transformers/src/models/quantized_blocks/ffn.rs:258`): call `gather_by_expert_q8_0`, then loop over experts calling the existing `mul_mat_vec_q8_0_q8_1_cuda1` (one per expert), then `scatter_by_expert_q8_0` (inverse of gather, with routing-weighted sum).
- **W3d** — Scatter kernel with weighted sum: atomic adds in f32 are native on gfx906 (no need for 64-bit atomic compare-swap).
- **W3e** — Bit-exact correctness gate vs `indexed_moe_forward_q8_0_q8_1` on reference tensors (topk=2, E=64, D=2560). Extend `candle-core/examples/hip_moe_mmq_test.rs`.
- **W3f** — Keep old fused kernel behind `CANDLE_MOE_FUSED_GATHER=1` for emergency fallback. Default off after bench confirms win.

**Files:**
- `candle-hip-kernels/src/quantized.cu` (new kernels)
- `candle-hip-kernels/build.rs`
- `candle-core/src/quantized/hip.rs` (`QHipStorage::indexed_moe_forward_preq8` dispatch + new entry points)
- `candle-transformers/src/models/quantized_blocks/ffn.rs:258`

**Effort:** 4–6 days (W3a: 0.5 day, W3b+c+d+e: 3–5 days, W3f: 0.5 day).

---

### W1 — Decode-path allocation pool **[biggest CPU-side win]**

**Target:** Eliminate all `hipMallocAsync` inside the forward pass. Per-device pool, size-bucketed, returned via `Drop` rather than `hipFree`.

**Projected gain:** 25 s CPU reclaimed, wall-clock **+8–12 pp tg**. Also unlocks W7 (graph replay requires stable buffer addresses).

**Evidence:**
- `hipMallocAsync` 13,181 × 1.9 ms = 25.0 s (**31 %** of candle's HIP API time)
- `hipFree` 11,916 × 51 µs = 610 ms (pool eliminates this too)
- Turbo does **zero** mallocs during inference (64 total, all at load time)
- Each Async-malloc on gfx906 triggers `amdgpu_vm_bo_map` ioctl → explains most of the 282 s sys time

**Implementation:**
- **W1a** — `DecodeAllocPool` in `candle-core/src/quantized/hip.rs` (or a new `hip/pool.rs`). Buckets: 2^k from 4 KiB to 256 MiB, per-bucket `Mutex<Vec<HipSlice>>` free-list. API:
  ```rust
  impl HipDevice {
      fn alloc_pooled(&self, bytes: usize) -> Result<PoolSlice>;
  }
  // Drop for PoolSlice returns it to the pool.
  ```
- **W1b** — Mark decode-scope allocations by wrapping `ModelWeights::forward()` bodies with a `scoped_pool!(device)` macro that swaps `alloc_uninit` → `alloc_pooled` for the span.
- **W1c** — `CANDLE_DECODE_POOL=0` opt-out.
- **W1d** — `CANDLE_TRACE_ALLOC=1` diagnostic that prints a stack on every `hipMallocAsync` during decode when the pool is active. Target: **zero** traces on the hot path.
- **W1e** — Parity bench `candle-bench compare ... --a 'CANDLE_DECODE_POOL=0' --b ''` must show tg128 ratio ≥ 1.10.

**Files:**
- `candle-core/src/quantized/hip.rs`
- `candle-core/src/hip/` (new pool module)
- `candle-transformers/src/models/quantized_gemma4.rs` (forward span wrap)
- `candle-transformers/src/models/quantized_qwen3*.rs` (same pattern, regression-tested)

**Effort:** 3–5 days.

---

### W5 — `ucopy_f32` + `copyBuffer` elimination **[expanded from rev. 1]**

**Target:** 946 ms of candle's bookkeeping GPU time → ≤ 80 ms (turbo parity).

**Projected gain:** ~850 ms GPU time reclaimed. On aggregate this is ~5.4 % of the 10.37 s GPU budget, **+3–5 pp tg** wall-clock.

**Evidence:**
- `ucopy_f32`: 15,950 × 40 µs = **634 ms** GPU time (6.1 % of GPU)
- `__amd_rocclr_copyBuffer`: 46,263 × 5.3 µs = **247 ms**
- `__amd_rocclr_fillBufferAligned`: 1,188 × 36 µs = 43 ms
- `copy2d_f32`: 7,800 × 2.8 µs = 22 ms
- Turbo's entire bookkeeping budget is 64 ms
- The gap comes from two sources: (a) candle's tensor aliasing defence inserts `Tensor::copy()` before in-place mutations; (b) the HIP `to_device` / `slice_set` paths fall back to `copyBuffer` instead of in-kernel writes

**Implementation:**
- **W5a** — Add `CANDLE_TRACE_UCOPY=1` env: `ucopy_f32` dispatch prints the Rust-side `std::backtrace` of the caller for the first N invocations. Landing this first localises the hot callers — without this we're guessing.
- **W5b** — For each top-5 caller identified in W5a:
  1. If the copy is defensive (Rust aliasing rule), prove unaliased and remove.
  2. If it's a layout change, see if the downstream kernel can accept the source layout.
  3. If it's a gather/narrow, fold into the downstream kernel's indexing.
- **W5c** — Regression gate: add a per-run launch counter; `assert!(ucopy_count < 200 per decode step)` (currently ~125 per step on Gemma-4-26B).
- **W5d** — Related: reduce `__amd_rocclr_copyBuffer` calls (46k) by replacing device-to-device stream copies with direct `hipMemcpyAsync(D2D)` when size > 1 MiB — the runtime path has overhead the explicit call doesn't.

**Files:** varies by findings — likely `candle-core/src/tensor.rs`, `candle-core/src/cuda_backend/storage.rs`, `candle-nn/src/kv_cache.rs`.

**Effort:** 2–3 days, dominated by tracing/understanding; fix itself is surgical.

---

### W2 — Peer-to-peer GPU copies

**Target:** Route `Tensor::to_device` and pipeline-boundary transfers through `hipMemcpyPeerAsync` when both devices are sibling HIP devices.

**Projected gain:** 9.3 → ~3 s H2D time, **+4–6 pp tg**.

**Evidence:**
- Candle: **0** `hipMemcpyPeerAsync`; 1,420 H2D × 6.6 ms = 9.3 s
- Turbo: 393 `hipMemcpyPeerAsync` + 1,073 H2D × 2.9 ms = 3.1 s
- MI50 supports p2p over XGMI between pairs and over PCIe elsewhere; `hipDeviceCanAccessPeer` reports which pairs

**Implementation:**
- **W2a** — `HipDevice::copy_peer(src_slice, dst_dev, dst_slice, bytes)` → `hipMemcpyPeerAsync`. Log `hipDeviceCanAccessPeer` matrix at init.
- **W2b** — `Tensor::to_device` dispatch: if `src.device().is_hip() && dst.is_hip() && src.ord() != dst.ord()`, prefer `copy_peer` over the current `copy_via_cpu` fallback.
- **W2c** — Pipeline-parallel boundary in `quantized_gemma4.rs` model forward: direct `copy_peer` on layer output → next-device input.
- **W2d** — Parity: single-GPU bench unchanged; multi-GPU tg128 must improve ≥ 4 %.

**Files:** `candle-core/src/hip/`, `candle-core/src/cuda_backend/storage.rs` (HIP equivalent), `quantized_gemma4.rs` (pipeline boundary).

**Effort:** 2 days.

---

### W6 — Parametric binary-broadcast kernel **[NEW]**

**Target:** Replace candle's 8 specialised elementwise kernels (`bmul_f32`, `bdiv_f32`, `badd_f32`, `affine_f32`, `is_u32_f32`, `usqr_f32`, `usqrt_f32`, `fast_sum_f32`, plus `copy2d_f32`) with one or two templated `bin_bcast<Op>` kernels, matching turbo's `k_bin_bcast` approach.

**Projected gain:** 631 ms → ~170 ms on elementwise, **+3–5 pp tg**.

**Evidence:**
- Candle 631 ms in 8+ kernel families × ~54,830 launches
- Turbo 168 ms via `k_bin_bcast<op_mul>` (97 ms), `<op_add>` (27 ms), `<op_repeat>` (22 ms), `scale_f32` (20 ms)
- Per-call candle 12 µs avg vs turbo 6 µs — candle's extra time is per-kernel launch-overhead amplification from many families + cold I-cache on the CP

**Why this matters for beating turbo:** elementwise ops aren't bandwidth-bound on MI50; they're launch-bound. Reducing distinct kernel binaries reduces ROCclr's code-cache pressure and the CP's dispatch cost. This is the secondary "turbo strength" and closing it removes a structural disadvantage.

**Implementation:**
- **W6a** — `bin_bcast<Op, IsInPlace, Rank>` template in `candle-hip-kernels/src/binary.cu` (new file). One C++ template, specialised at compile time for `Op ∈ {add, sub, mul, div, affine, sqr, sqrt, sum_reduce}`. ~200 LOC total.
- **W6b** — Rust dispatch: `candle-core/src/hip/binary.rs` picks the right specialisation from `Op` + shape. Drop `bmul_f32`/`bdiv_f32`/etc. callers in favour of the parametric version.
- **W6c** — Ensure no regression on fast paths: microbench each replaced op vs old kernel (e.g. `hip_binary_bench.rs`). Require ≤ 5 % regression on any, ≥ 10 % improvement in aggregate decode GPU time.
- **W6d** — Retire the 8 old kernels from the HSACO build.

**Files:**
- `candle-hip-kernels/src/binary.cu` (new)
- `candle-hip-kernels/build.rs`
- `candle-core/src/hip/binary.rs` (new or existing)
- Various caller sites across `candle-core/src/tensor.rs` and model forwards

**Effort:** 3–4 days.

---

### W4 — `hipMemsetAsync` storm cleanup

**Target:** 1,128 calls × 8 ms = 9.0 s → ≤ 150 ms (turbo parity).

**Projected gain:** ~9 s CPU, **+2–3 pp tg**.

Implementation unchanged from rev. 1:
- **W4a** — `CANDLE_TRACE_ZEROS=1 CANDLE_TRACE_ZEROS_START=50` (already wired per MEMORY) — dump caller stacks.
- **W4b** — Convert each hot caller: `alloc_uninit` + targeted overwrite, or `alloc_uninit` + `hipMemsetAsync` only of the tail. Same pattern as Phase O3's fix to `load_quantized`.
- **W4c** — Regression guard: `CANDLE_TRACE_ZEROS_ASSERT=30` panics at >30 fills per forward.

**Effort:** 1 day.

---

### W7 — Activate G2/G3 replay on Gemma4-26B-A4B **[NEW — decisive over-turbo lever]**

**Target:** End-to-end Phase K-style capture of the decode forward pass, eliminating per-token CPU dispatch overhead on the replay path.

**Projected gain:** +5–10 pp tg *on top of W1*, and effectively caps the CPU-side gap at near-zero for sustained decode. This is the move that pushes us decisively past turbo.

**Why this is feasible:** MEMORY §"Phase K" documents the full correctness machinery working end-to-end on Gemma4-E4B at 34 t/s (G2+G3) — proving the architecture supports multi-device capture, per-layer weight quant-on-load, and external-input patching. The 26B-A4B version adds MoE, which is why it wasn't in Phase K scope at the time. Post-W3 the MoE path is a predictable gather + MMVQ + scatter — much easier to capture than a fused indexed gather kernel.

**Dependencies:** W1 (stable buffer addresses for captured args) and W3 (MoE kernels capturable — current `indexed_moe_forward` may hit the 432-Counter-patch problem from Phase T).

**Implementation:**
- **W7a** — Verify W3 gather/MMVQ/scatter kernels use `LaunchArgs::launch` (not rocBLAS) — otherwise they're invisible to the G2 recorder (Phase K11 bug). Likely OK by default since they'll be custom HIP kernels.
- **W7b** — Add MoE path to the anchor-input table: router probs, expert-id tensor. These change per token and need `ExternalInput` registration.
- **W7c** — Run `CANDLE_G2_REPLAY=1 CANDLE_G2_REPLAY_MAX=256` on gemma-4-26B-A4B-Q8_0. Expected issues: (1) multi-device MoE gather/scatter across GPU pipeline boundaries (Phase K7 / K13-style); (2) `hipErrorNotReady` from pool-slot interference during sampler (mitigated by W1's stable-address pool).
- **W7d** — Add `CANDLE_G3_GRAPH=1` on top. Target: 70+ t/s tg64 short prompt.
- **W7e** — Multi-device robustness: apply the `RecordedOp::Memcpy` design from MEMORY §V1b-blocker (deferred from Phase K) to make peer-to-peer copies capturable.

**Files:**
- `candle-core/src/hipdarc/` (recorder + graph)
- `candle-transformers/src/models/quantized_gemma4.rs` (decode state machine wrap around new MoE path)
- Possibly extend `candle-core/src/hipdarc/decode_plan.rs` to handle MoE expert-ID anchoring

**Effort:** 5–8 days.

---

### W8 — Lock in existing wins **[LANDED 2026-04-16]**

**Target:** Every phase above touches hot-path code. Add a kernel-family regression gate so we don't lose candle's current strengths while chasing the deltas.

**Landed artefacts:**
- `bench/kernel_family.py` — parser + classifier library (ordered regex rules, warmup-outlier trim, JSON emitter).
- `bench/kernel_family_baseline.py` — CLI that captures a rocprofv3 `summary.txt` → baseline JSON + markdown snapshot; fails if unmatched kernels contribute > 1 % of GPU time.
- `bench/kernel_family_regression.py` — CLI that diffs a new `summary.txt` against a baseline; exit 0 pass / 1 preserve-family or aggregate regressed / 2 setup error.
- `bench/profile_decode.sh` — reusable wrapper that runs `rocprofv3 --runtime-trace --stats` on either candle or turbo and flattens the output to `/tmp/prof_<tag>/{summary.txt,time.txt,stdout.log}`.
- `bench/results/kfr_gemma4-26B-A4B-Q8_0.json` + `.md` — **committed candle baseline** from the 2026-04-16 profile. Starting line: 14 families, 9,971 ms aggregate GPU time, tg128 43.84 t/s (uninstrumented).
- `bench/results/kfr_gemma4-26B-A4B-Q8_0_turbo.json` + `.md` — turbo reference (not a gate).

**Family definitions** (in `kernel_family.py`):
- `PRESERVE = {mmvq_q8, mmq_q8_prefill, flash_decode, softmax_attn, rmsnorm, activation, quant_q8_1, rope}` — must not regress > 10 %.
- `DEGRADE_TARGET = {moe_fused, bookkeeping, elementwise, router_topk}` — what W1-W7 attack; reductions expected.
- Aggregate GPU time must not regress > 5 %.

**Workflow for every W1–W7 PR:**
```bash
# 1. Run the profile under the change.
bench/profile_decode.sh candle_w3 candle

# 2. Gate the change.
python3 bench/kernel_family_regression.py \
  --baseline bench/results/kfr_gemma4-26B-A4B-Q8_0.json \
  --summary  /tmp/prof_candle_w3/summary.txt
# Exit 0 required to land.
```

**Smoke tests** (3 cases, all passing):
- Self-check (baseline vs itself) → exit 0, all deltas within noise.
- Perturb `mul_mat_vec_q8_0_q8_1_cuda1` ×2 → exit 1, `FAIL (preserve regressed)` on `mmvq_q8` and `FAIL (aggregate regressed)`.
- Inject an uncategorised kernel contributing 5 % → exit 2 with a request to extend `FAMILY_RULES`.

**Effort actual:** ~0.5 day.

---

## Target end state

Projections per phase (non-additive: dependencies and diminishing-returns noted). Numbers in tg128 t/s (uninstrumented).

| state | tg128 | vs turbo | sys CPU (rocprof) | hipMallocAsync |
|---|---:|---:|---:|---:|
| today | 43.84 | 0.66× | 282 s | 13,181 |
| + W3a (fused moe-combine — **landed**) | 44.81 | 0.67× | 282 s | 13,181 |
| + W3 (gather-by-expert — **net-negative, reverted to opt-in**) | — | — | — | — |
| + W1 (alloc pool) | ~52 | 0.78× | ~50 s | 0 |
| + W5 (ucopy clean) | ~54 | 0.81× | ~45 s | 0 |
| + W2 (peer-to-peer) | ~57 | 0.85× | ~40 s | 0 |
| + W6 (param bin_bcast) | ~60 | 0.90× | ~35 s | 0 |
| + W4 (memset clean) | ~62 | 0.93× | ~28 s | 0 |
| **+ W7 (G2/G3 active)** | **~68–72** | **1.02–1.08×** | **~12 s** | **0** |

Revised after W3 retro: without W3's planned 8–12 pp, the parity line now sits at **W1+W5+W2+W6+W4** (≈ 0.93× turbo) and the over-turbo push depends on **W7** (G2/G3 replay) delivering. W7 is the phase that decisively beats turbo via the architecture lever turbo doesn't have. W1 alone is the biggest remaining win (projected +8–12 pp from eliminating 13,181 `hipMallocAsync` calls per decode run).

**Fallback plan if W7 hits blockers**: W1–W6 lands ≈ 0.90–0.93× turbo. Additional push would come from revisiting the MoE path (W3b gather-by-expert for prefill + physical MoE for large b_size) or from unplanned wins in W1 itself (driver sys-time drop can exceed linear projections once ioctl amplification is removed).

## Validation harness

Every phase must pass before being marked done:

1. **Correctness gate** — `candle-bench bench --backend candle --bin quantized-gemma4 --gpus 0,1,2,3 --model /artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf --prompt-len 128 --tg-len 64 --reps 1` produces coherent output (no "I I I I"); recorded first-token-logits cosine similarity ≥ 0.9999 vs baseline.
2. **Per-phase perf gate** — the specific metric named in the phase (e.g. `hipMallocAsync` count for W1, MoE kernel per-call µs for W3) must hit the stated target.
3. **W8 regression gate** — no candle-winning family regresses > 10 %.
4. **Cross-model no-regression** — `candle-bench matrix --tag moe` and `--tag gemma4` over the built-in catalog; no cell regresses > 2 % on tg.

## Out of scope (stays out)

- Long-prompt prefill (pp > 2k) — MMQ is at gfx906 hardware ceiling per Phase I3. This roadmap moves pp modestly (~15 %) via W1/W5 but isn't the main axis.
- TurboQuant / MXFP4 / BF16 expansions — different roadmap (`ROADMAP-CUDA-ALL-MODELS` and follow-ups).
- Kernel-level tuning of already-winning families (MMVQ, MMQ, flash-decode) — W8 actively guards these from being touched.

## Execution order

Recommended sequence, optimised for *earliest* break above turbo parity:

1. **W8** (0.5–1 day) — baseline + regression script first, so every later phase is verified.
2. **W3a** (0.5 day) — fuse `fast_sum_f32` into `topk_moe_norm`. Quick win, builds muscle.
3. **W3** (4–6 days) — defuse MoE. Biggest GPU-side win. Stand-alone value.
4. **W1** (3–5 days) — alloc pool. Biggest CPU-side win. Unlocks W7.
5. **W5** (2–3 days) — ucopy trace + eliminate.
6. **W2** (2 days) — peer-to-peer copies.
7. **W4** (1 day) — memset cleanup.
8. **W6** (3–4 days) — parametric bin_bcast.
9. **W7** (5–8 days) — G2/G3 activation. **Decisive lever to clear 1.05× turbo.**

Total: ~22–34 days of eng. time. Expected outcome: tg128 ≥ 70 t/s, stretch 75 t/s, on Gemma-4-26B-A4B-Q8_0, 4× MI50.

## Links

- Profile evidence: `/tmp/prof_candle/summary.txt`, `/tmp/prof_turbo/summary.txt`, `/tmp/prof_{candle,turbo}/time.txt`
- Uninstrumented bench: `candle-bench bench ... --reps 1 --json` 2026-04-16 session
- Family comparison: see the ledger in the "Goal" section above
- Prior related: `ROADMAP-CPU-OVERHEAD-2026-04-12.md` (Phase G1/G2/G3 foundation), MEMORY §Phase K (G2/G3 on Gemma4-E4B, proof W7 is feasible), MEMORY §Phase O (alloc_zeros lineage that seeded W1/W4), MEMORY §Phase I3 (why we don't touch MMQ).
