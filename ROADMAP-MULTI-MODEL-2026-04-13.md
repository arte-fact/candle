# Candle Multi-Model Optimization Roadmap — 2026-04-13

## Starting point

**Decode throughput (best of 3, 48-tok prompt + 256-tok decode, MI50 gfx906):**

| Model | Candle | vanilla llama.cpp master | Δ vs llama.cpp |
|---|---|---|---|
| TinyLlama-1.1B Q4_0 | 267 t/s | 209 | **+28%** |
| Gemma4-E4B Q4_0 | 44 t/s | 67 | **-34%** |
| Qwen3.5-9B Q4_1 | 39 t/s (broken ⚠) | 61 | -36% (invalid) |

**Prefill:** Candle is 55-85% slower than llama.cpp across the board.

**Root cause of the gap on Gemma4/Qwen:** all llama-path optimisations from
this session (n_kv pad, G2 replay, G3 graph, fused FFN) live in
`quantized_llama.rs` and have not been ported to the Gemma4/Qwen code paths.

## Profile evidence (Gemma4-E4B, single decode run, total GPU time)

|  | Candle | llama.cpp | Delta |
|---|---|---|---|
| **rocBLAS Cijk** | **568 ms (32 %)** | **39 ms (3 %)** | **15× more** |
| MMVQ | 447 ms | 614 ms | fewer calls, slower per call |
| MMQ (prefill) | 296 ms | 119 ms | **2.9× slower per call** |
| Total GPU | 1782 ms | 1185 ms | +50 % |

## Priority update — post-survey (2026-04-13)

See `SURVEY-GFX906-FORKS-2026-04-13.md` for the full landscape comparison of candle vs turbo vs vllm-mobydick. The survey's ranked recommendations reshape the roadmap order-of-attack:

| # | action | unlocks / payoff | effort | survey §11 rank |
|---|---|---|---|---|
| 1 | **ROCm 7.2.1 + Tensile migration** (existing `ROADMAP-ROCM-722-MIGRATION`) | DPP warp reductions (Phase J disabled), K-quant warp-coop (Phase B3 gated), rocBLAS sgemv (Phase O) — THREE gated features in one migration | 1 day rebuild + regression | #1 |
| 2 | **Phase R — LDS `+1` padding** on flash-attn-v2 tiles + MMQ v2f | 2× LDS bandwidth per skyne98 wiki (3974 vs 1865 GB/s); direct port of turbo's `cpy.cu:59` pattern | 2-4 h per kernel | #2 |
| 3 | **Phase Q2 — `gqa_decode_mv` tuning** | 73 μs/call → ~30 μs matching turbo's `mul_mat_vec_f`; +3-4 t/s on E4B | 1-2 days | #3 |
| 4 | **Phase S — TurboQuant KV port** | 5× KV memory reduction; enables 32K+ context on 16 GB MI50 | 2-3 days | #4 |
| 5 | graph-fusion engine | marginal over existing Phase D1/D2 ad-hoc fusions | 1-2 days | #5 |
| 6 | **Phase Q1 — G2 + Phase P integration** | make Phase P default-on without regressing G2 users | 1-2 days | #6 |

**NOT pursuing** (survey §11): PagedAttention / continuous batching (vLLM's serving domain, not ours); Triton codegen (would trade type-safety + G2 capture for portability we don't need on the single-arch gfx906 target).

## Phases

### Phase H — Port attention optimisations to Gemma4 (highest single leverage)

**Target:** eliminate the 568 ms of rocBLAS Cijk calls in Gemma4 decode.
**Projected gain:** Gemma4-E4B decode 44 → 60+ t/s (+35 %).

- **H1** — Port `n_kv` padding (256 quantum) into `quantized_gemma4.rs` so
  kernel args stay stable across a 256-token decode window.
- **H2** — Ensure `gqa_attention_k_transposed` dispatch picks rocBLAS-free
  flash-attn v2 path for Gemma4 (same logic as llama).
- **H3** — Verify Gemma4 output still correct after padding (compare first
  10 sampled tokens against no-pad baseline, temp=0).

**Owner:** me, next.
**Effort:** ~2-4 hours. Pattern proven on llama path.

### Phase I — MMQ kernel rewrite (cross-model win)

**Target:** `mul_mat_q4_0_gfx906_v2f_tile32` kernel at 652 μs/call vs
llama.cpp's `mul_mat_q<type>` at 225 μs/call (2.9× slower). Impacts prefill
for **every** Q4_0 model.

**Projected gain:** prefill 2-3× across all models (TinyLlama prefill 801 →
~2 000 t/s; Gemma4 87 → ~250 t/s).

- **I1** — Read llama.cpp `ggml-cuda/mmq.cuh` template, understand tile
  structure, block layout, shared-memory staging.
- **I2** — Either port their template or rewrite our v2f kernel to match
  — likely bigger output tiles, better LDS reuse, matching warp layout.
- **I3** — Validate correctness (MMQ tolerance) against existing tests.

**Owner:** TBD.
**Effort:** 1-2 days (real kernel work).

### Phase J — Fix Qwen3.5 correctness bug

**Blocks** any meaningful Qwen perf comparison.

- **J1** — Reproduce: confirm `!!!!!` output is token 0 / <unk> being
  sampled because logits are NaN or identical.
- **J2** — Bisect which layer/module is producing bad output — likely
  attention or qkv projection. Check in sequence: embedding → layer 0 Q →
  K → V → attn → output.
- **J3** — Fix, regression-test against known-good checkpoint.

**Owner:** TBD (investigative).
**Effort:** 0.5-1 day.

### Phase K — Port G2 / G3 + fused FFN to Gemma4 path

**Prerequisites:** Phase H complete (rocBLAS out of attention hot path
first, otherwise G2 replay divergence returns).

**Target:** another +10-15 % on top of Phase H, like we saw on llama.

- **K1** — Add the `DecodeState` state machine to the Gemma4 `ModelWeights::forward`.
  Detect `is_decode` (seq_len=1, idx_pos>0), warm up → record → replay. — **DONE** (commit 10185833)
- **K2** — Wire G3 graph capture with `hipGraphExecKernelNodeSetParams` node
  patching (`DecodeGraph::patch_and_launch` already exists, just plumb it). — **DONE** (commit 10185833)
- **K3** — Port fused Q4_0 FFN path where applicable (Gemma4-E4B has dense
  Mlp in some layers — fits). Gemma4 MoE uses different FFN structure;
  won't apply there without a separate fused-MoE kernel.

**Status (2026-04-13):** Infrastructure complete, two known blockers:

  - **K1/K2** (commit 10185833): scaffolding + state machine.
  - **K5** (commit a8d22fa0): multi-anchor `decode_cache` API
    (`from_two_recordings_with_inputs` + `patch_external_input`) and a
    `with_recording_paused` helper that pauses both kernel recording
    and decode_alloc.
  - **K6** (commit a8d22fa0): E4B prelude (CPU embed + per_layer
    compute) runs paused; results anchored as inputs #0/#1.
  - **K7** (commit 5a645f77): documented multi-device blocker
    (`hipErrorInvalidImage`) for follow-up under K9.
  - **K8** (commit 4fb9db04): pad SWA mask via new
    `causal_mask_padded`; anchor the mask as a third external input
    (one per device + sliding-window pair). `with_recording_paused`
    no longer pauses `decode_alloc` — prelude tensors stay at
    sentinel-anchored pool slots, refreshed in-place by per-call
    CPU→GPU memcpys. `decode_alloc_resume()` runs at the top of
    forward when state is Replay/Graph so prelude allocs return the
    recorded slots.
  - **K9** (commit d53af394): per-device tracking in `RecordedOp`
    (`device_ordinal` captured via `hipGetDevice` at record time);
    `DecodePlan::replay` + `capture_graph_full` `hipSetDevice` once
    per device boundary. Multi-device 31B G2 plan now builds cleanly
    (2109 ops, 5 anchors) without the `hipErrorInvalidImage` crash.

**K11 (commit b38aab1b)** plumbed kernel names through the recorder
and added per-kernel arg breakdown to the multi-input debug. That
immediately localized the K10 bug — the captured plan had ZERO
`rocblas_*` ops because `rocblas_gemm_strided_batched_ex` bypasses
`LaunchArgs::launch` (the only recorder hook). Gemma4's attention
gemms were running but invisible to G2.

**K12a (commit 3f315625)** — when `CANDLE_G2_REPLAY=1`, force
`gqa_attention_k_transposed` to use flash-attn-v2 instead of rocBLAS.
Flash-attn IS recordable (it goes through `LaunchArgs::launch`).
Replay output starts evolving per call.

**K12b (commit d66df38e)** — added `flash_attn_v2_fwd_ktvs_d512_f32`
for Gemma4-E4B's global-layer head_dim=512 (BC=8, 32 KiB LDS).
The captured plan now contains 100% of attention compute (35×
d=256 SWA + 7× d=512 global, no masked_softmax fallback).

**K13:** cross-checked our gemma4 forward against
`llama.cpp/src/models/gemma4-iswa.cpp`. Structure matches —
`inpL = embed * sqrt(n_embd)` outside the loop, `inp_per_layer`
prelude (CPU embed + GPU project + norm + residual) outside the
loop, per-layer block (norm → wq/wk/wv → q/k norm → RoPE →
attention → post-norm → residual → ffn → post-ffn-norm → residual
→ per-layer-embed injection → layer scale) inside the loop. So the
model formulation isn't the bug — it has to be correct because
baseline+flash produces correct output.

**K13 ROOT CAUSE FOUND (commit 2d8c5ea0):** Gemma4-E4B's
per-layer-embedding projections (`*.inp_gate.weight` and
`*.proj.weight`) are stored as **F16** in the GGUF.
`QMatMul::from_arc` unconditionally dequantizes F16/F32/BF16 to a
plain `Tensor`, whose `forward` dispatches via `Tensor::matmul` →
rocBLAS — bypassing the G2 launch recorder. 84 missing matmuls
(2 per layer × 42 layers). Verified by per-op arg-pointer dump:
`op[1378] rmsnorm` reads from `0x...95bb00` while preceding
`op[1377] bmul` writes to `0x...95a600` — no captured op fills
the gap (that's the missing per_layer_proj's MMVQ output).

Fix: `QMatMul::requantize_to(Q8_0)` at load. Captured plan grew
1389 → 1557 ops, all op outputs now evolve per replay. G2 replay
output partially recovers ("Hello! How can I I I I can I help you
today?"). G3 graph capture works (1557 nodes, 240 dynamic ops).
Default decode unchanged AND faster — 47 → 62 t/s (Q8_0 MMVQ
beats rocBLAS gemv for the 256-wide per_layer projections).

**K13 SECOND ROOT CAUSE (commit f5fb8a78) — PHASE K CORRECTNESS COMPLETE.**
Even after every matmul was captured, replay still produced
"I I I I" garbage. Trace of `inp_per_layer.ptr` across replays
revealed the smoking gun: the prelude allocator returns DIFFERENT
addresses each replay (decode_alloc serves different slots), but
the captured plan only marked args matching the BASE pointer as
External. Each per-layer kernel reads via
`inp_per_layer.narrow(2, il, 1).ptr() = base + il*stride`, so
layers 1..41's args sat in the MIDDLE of the input buffer and
matched nothing in the patcher's pointer-equality check.

Fix: extend `ExternalInput` with `bytes`, scan args for values
inside `[input_ptr, input_ptr + bytes)`, capture per-loc
`external_offsets`, patch as `fresh_ptr + offset`. External
patches went 3 → 44.

**Final results on Gemma4-E4B Q4_0:**
- Default: 52-67 t/s decode, correct output.
- G2 (`CANDLE_G2_REPLAY=1 CANDLE_G2_REPLAY_MAX=256`): 27 t/s
  decode, VALID output — matches default exactly for short
  sequences, stays sensible (and informative) on longer ones.
  Subtle drift over many tokens is from Q8_0 vs F16
  quantization, NOT from G2 itself: re-running default with
  Q8_0 requant but no G2 also diverges from default-no-requant
  in the same way.
- G2+G3 (`CANDLE_G3_GRAPH=1`): 34 t/s decode, also valid output.

**Phase K correctness goal MET.**

Performance hit (G2 27 t/s vs default 52 t/s) is from per-call
prelude work (CPU embed + transfer + per_layer compute + SWA mask
rebuild — runs FRESH every replay since it's outside the captured
plan). NOTE: the prelude's `model_proj` must stay F16 (tested:
requantizing it adds an extra alloc per call that throws the
decode_alloc cursor out of sync, causing mid-replay
`hipErrorNotReady`). Optimizing the prelude is follow-up work.

---

(historical, K10) Detailed diagnostics from earlier:

  - `layer_in`, `inp_per_layer`, and SWA mask are all verified refreshed
    per call — prelude memcpys hit the sentinel-anchored pool slots
    that the captured kernels read from.
  - `advance_counters` mutates all 432 (E4B) / 720 (31B) Counter args.
    Delta histogram shows the strides we'd expect: 110× delta=512
    (RoPE cos/sin advance per token at head_dim/2 stride), 42× 1024
    and 4× 2048 (K-cache slice_set per-token pointer advances), 84×
    delta=1 (position counters).
  - Yet `lm_head` emits **byte-identical logits across all 4 replays**:

        replay#1 idx=14 head: [-23.92, -10.76, -21.68, …]
        replay#2 idx=15 head: [-23.92, -10.76, -21.68, …]
        replay#3 idx=16 head: [-23.92, -10.76, -21.68, …]
        replay#4 idx=17 head: [-23.92, -10.76, -21.68, …]

    Some critical kernel arg that the lm_head output transitively
    depends on isn't in the dynamic set, so the captured chain is a
    constant function of the current cache state.

  Two prime suspects:
    1. The attention K-pointer arg in the rocBLAS `Cijk_*` batched-gemm
       kernel may be captured as `cache_base + recorded_offset` rather
       than just `cache_base + 0`. The K narrow always spans
       `(0..pad_t)`, so the slice_set Counter advance updates the
       cache content at position N+1, but if the gemm reads
       `cache_base + N*stride` (recorded), it won't see the new K at
       position N+1. The right Counter delta would then need to apply
       to the gemm K-arg, not just to slice_set's dst.
    2. rocBLAS may cache solution-internal state across calls — the
       captured `Cijk_*` launches don't go through the rocBLAS handle
       on replay (we relaunch the kernel directly), but the trailing
       `rocBLAS error during freeing of allocated memory` suggests
       handle-side state corruption.

  Concrete debug step: dump per-op arg layout (name + size + role) at
  recording time, then cross-reference the static-vs-dynamic
  classification against expectations. Saved as a follow-up task.

Default path (G2 disabled) verified intact — Gemma4-E4B Q4_0 still
emits `"Hello! How can I help you today?"` at the baseline 47 t/s
decode.

**Owner:** TBD.
**Effort:** 1 day after Phase H. Each follow-up (SWA mask anchor /
multi-device device-ordinal tracking): ~half day each.

### Phase L — MMVQ per-call tuning

**Target:** Candle MMVQ 27 μs/call vs llama.cpp 18.5 μs/call (46 % slower).

- **L1** — Diff our `mul_mat_vec_q4_0_q8_1_cuda1` against llama.cpp's
  `mul_mat_vec_q<type=2, bsize=1, mmq=false, mxfp4=false>`. Measure what
  accounts for the 8.5 μs gap.
- **L2** — Port winning bits (block count, thread count, split-accumulator
  config, etc.).

**Effort:** 0.5-1 day.
**Note:** MMVQ is already at 19 % of HBM peak — there's a physical ceiling.
This is a fine-tuning task, not a breakthrough.

### Phase M — Fused MoE-FFN kernel for Gemma4-E4B

**Context:** Gemma4-E4B routes through MoE experts. Each expert is a
dense FFN that could use our `fused_q4_0_ffn_decode`. Gate + up + down per
expert. With ~4 experts × top-2 routing = 8 FFN paths per token.

- **M1** — Figure out Gemma4-E4B MoE structure: how many experts, top-k.
- **M2** — Fused MoE kernel or reuse `fused_q4_0_ffn_decode` per active
  expert (gating is scalar, cheap).

**Effort:** 1-2 days.
**Deferred** until Phases H-I-J-K land.

### Phase N — General prefill optimisation

**Target:** Candle is 55-85 % slower than llama.cpp at prefill across the
board. Mostly Phase I covers this (MMQ). Also:

- **N1** — Fused attention for prefill (multi-query, seq_len≥4).
  `flash_attn_v2` already does this for llama; port to Gemma4.
- **N2** — Batched quantize for Q8_1 activation cache (one kernel for the
  whole prefill row batch, not per-token).

**Effort:** 1 day after Phase H.

### Phase O — rocBLAS GEMV attention for Gemma4-E4B decode (2026-04-13)

**Target:** close the 3.4 ms/tok attention gap to llama.cpp-turbo on E4B
decode. Per-kernel comparison via rocprofv3:

| Op | turbo (E4B Q4_0 decode, ms/tok) | candle (post-Phase L dyn-lk) |
|---|---|---|
| Attention (gemv + softmax) | **0.6** | **3.98** (flash_attn_v2 d256 + d512) |

That's the single biggest remaining gap. Closing it would push candle
E4B decode from ~52 t/s toward turbo's 75 t/s.

**Why our flash-attn loses:** the v2 LDS-tiled kernel runs at 89 μs/call
for D=256. For L_q=1 decode with GQA n_rep=2, BR=4 warps cooperate to
load a BC=16 K-tile into LDS — but only 2 of those BR warps have a
real Q-row (n_head=2 per kv_head), so half the LDS cooperative-load
work is wasted. Plus the cross-warp `__syncthreads` after each tile
serialises against the inner softmax pass.

**Why a naive 1-warp-per-q_head GEMV ALSO loses** (tried 2026-04-13,
commit `4967ab3e`): KvCache K is `(B, H_kv, D, max_T)` D-major. The
GEMV pattern `K[lane * max_T + j]` (each lane owns one of D
elements, iterates over j) makes adjacent lanes read max_T=4096
floats apart — fully uncoalesced. Regression: 50.3 → 43.1 t/s.

**Plan — wrap rocBLAS gemv directly:**

- **O1** — Add `hipblasSgemv` wrapper to `hipdarc/src/rocblas.rs`. This
  bypasses rocBLAS's gemm-based attention dispatch (which the Tensile
  picker won't touch for our (1×D)·(D×T) shape — it falls back to a
  generic gemm internally). Direct gemv lets rocBLAS pick its hand-
  tuned MI50 gemv kernel, which IS what turbo's `mul_mat_vec_f<float,
  float, ...>` is calling under the hood (their llama.cpp hipBLAS
  build redirects `mul_mat_vec_f` to `hipblasSgemv` for f32×f32).

- **O2** — In `gqa_attention_k_transposed`, when `seq_len == 1` and
  D∈{128,256,512}, use the gemv path instead of flash_attn_v2:
  ```
  // For each kv_head h:
  //   attn = K[h, :, :].T @ Q_grouped[h, :, :]  → shape (n_rep, T)
  //   attn = scale * attn + mask
  //   attn = softmax(attn)
  //   out[h, :, :] = V[h, :, :] @ attn          → shape (n_rep, D)
  ```
  Each "matmul" is 2 gemv calls per kv_head (one for K, one for V).
  For E4B SWA d256: n_kv_head ~16, so ~32 gemv calls/layer ×
  42 layers = 1344 gemv calls/token. At ~0.4 μs each = 0.55 ms/tok —
  matches turbo's number.

- **O3** — Reuse the existing `masked_softmax_scale_fused` kernel for
  the softmax step (already optimal).

- **O4** — Verify rocBLAS gemv is captured by the G2 plan recorder
  (`hipblasSgemv` dispatches to a HIP launch under the hood — should
  be visible). If not, gate this path off when `CANDLE_G2_REPLAY=1`
  and keep the v2 dyn-lk kernel for replay.

**Effort:** 2-3 days. Most of it is the hipdarc gemv binding +
plumbing the per-kv-head dispatch loop.

**Risks:**
- rocBLAS `hipblasSgemv` may itself dispatch through Tensile gemm and
  not actually use the fast hand-tuned kernel — verify with rocprofv3
  that the kernel name matches turbo's `mul_mat_vec_f<float,float,...>`.
- Per-call CPU overhead from issuing many gemv launches (1344/tok)
  may eat the win. If so, batch via `hipblasSgemvBatched`.
- Won't help D=64/128 (those already use a fast `flash_attn_decode_*`
  path) — only Gemma4-E4B's wide-head SWA layers benefit.

**Expected outcome:** E4B decode 52 → ~70 t/s (matches turbo within
noise). No effect on llama / TinyLlama (those use D=64 / 128 with the
existing fast path).

**Status (2026-04-13)**: scaffolding committed but blocked on rocBLAS
sgemv runtime.
- O1 (FFI binding) DONE. Added `rocblas_sgemv_strided_batched` in
  `hipdarc/src/sys.rs` and `sgemv_strided_batched` safe wrapper in
  `hipdarc/src/rocblas.rs`. Also added `rocblas_set_pointer_mode` and
  set `pointer_mode_host` explicitly at `RocBlas::new` (defensive — no
  observable change but it's correct hygiene). Builds clean against
  ROCm 7.1.1 librocblas.so.5; symbol verified via `nm -D`.
- O2 (dispatch) DONE shape-wise but SEGFAULTS on first sgemv call.
  Even a TINY smoke-test sgemv (m=n=4, lda=4, batch=1, fresh tiny
  buffers) crashes the same way. So the bug is NOT shape/stride
  dependent.
- **strace pinpoints the cause**: rocBLAS sgemv triggers runtime JIT
  via comgr — the strace shows
  `comgr-XXXX/output/hipfatbin-...gfx906:sramecc+:xnack-.o` files
  being created as the call enters, and SIGSEGV (`SI_KERNEL`,
  `si_addr=NULL`) immediately after. Best read: the gfx906-patched
  rocBLAS install ships Tensile gemm kernels but its sgemv kernel
  selection falls back to a comgr JIT path that crashes on this
  ROCm 7.1.1 / gfx906 combo.
- **Worked-around attempts**:
  - Pointer mode = host (explicit): no effect.
  - Stream sync before call: no effect.
  - batch_count = 1 with per-head loop: same crash.
  - Smoke test (m=n=4): also crashes.
- **Possible future fixes**:
  1. Phrase the gemv as a tiny gemm via existing
     `rocblas_gemm_strided_batched_ex` (which works and uses Tensile
     gfx906 kernels), with `n=1`. May or may not pick a fast Tensile
     kernel — if Tensile lacks a `n=1` fast path it'll just be the
     same as our current matmul-based path.
  2. Switch to rocm-6.4 librocblas (also available on this system) —
     untested whether 6.4's sgemv kernel selection takes a different
     path.
  3. Build a custom mat-vec kernel in HIP. Bypasses rocBLAS entirely.
     This is what we tried with `flash_attn_decode_strided` D=256/512
     extension — regressed because of K-storage uncoalescing. Would
     need a kernel that loads K T-major (which our storage isn't).
  4. Change KvCache K storage from `(B, H, D, max_T)` to
     `(B, H, max_T, D)` — natural layout for GEMV. Big refactor,
     touches every flash-attn caller.
- **Dispatch is gated `#[cfg(any())]`** in `attention.rs` — path
  unreachable, default decode unaffected (verified 53.7 t/s post-revert
  on E4B Q4_0). Toggle via `CANDLE_DECODE_GEMV_ON=1` AND remove the
  cfg gate to reproduce the segfault.

**Closed (2026-04-13, superseded by Phase P).** Pursuing the alts
revealed Phase O's premise was wrong in the first place:

1. **O-alt-2 (rocm-6.3.4 rebuild) — DONE, partial success.** Rebuilt
   candle against `ROCM_PATH=/opt/rocm-6.3.4`. Binary links
   `librocblas.so.4` (6.3.4 soname), runs with
   `LD_LIBRARY_PATH=/opt/rocm-6.3.4/lib`. **rocBLAS sgemv does NOT
   crash on 6.3.4** — the comgr-JIT bug is specific to rocm-7.1.1's
   rocBLAS sgemv kernel selection. Measured on E4B Q4_0 (52-tok +
   63 decoded):
     rocm-6.3.4 baseline   : 36.68 t/s
     rocm-6.3.4 + GEMV     : 40.93 t/s   (+11.6 % vs same-rocm
                                          baseline — gemv IS faster
                                          when available)
     rocm-7.1.1 baseline   : 52.03 t/s
   So gemv itself is correct and wins on its native rocm, but
   rocm-6.3.4 loses 30 % on non-gemv kernels (different Tensile
   gfx906 kernels, rocm-7.1.1 is better for everything else). Net
   loss. Rolled back to rocm-7.1.1.

2. **Went back to the turbo rocprofv3 trace and looked harder** at
   what `mul_mat_vec_f<float,float,1,128,false,false>` actually IS:
   found it in `ggml-cuda/mmvf.cu`. **Not a rocBLAS sgemv call** —
   it's ggml-cuda's own hand-written HIP kernel. So the whole
   "wrap rocBLAS sgemv" premise was a misread of the trace.

3. **ggml-cuda's mmvf kernel semantics** (`mmvf.cu:8` onwards):
   - Grid: `blockIdx.x = row` (one block per output row).
   - Threads in block iterate the reduced axis coalesced — read
     `float2` pairs with stride = `block_size` floats. Adjacent
     threads read adjacent memory → full coalescing.
   - This assumes the matrix is row-major with the reduced axis
     **contiguous in memory**.

4. **Turbo's K layout IS T-major (`(B, H_kv, T, D)`)** — D contiguous,
   which makes the Q·K^T mat-vec coalesce naturally. Candle's K is
   **D-major (`(B, H_kv, D, max_T)`)** — T contiguous, because we
   went T-major-in-the-cache to skip a K^T transpose in flash-attn.
   That decision was right for flash-attn-v2 but it's EXACTLY what
   makes mul_mat_vec_f impossible for us. Every prior gfx906
   mat-vec-style attempt on candle hit the same uncoalescing wall:
   - O scaffolding (sgemv_strided_batched, d1307b92): segfaulted on
     rocm-7.1.1 AND would have been uncoalesced anyway.
   - K-stride dyn flash-attn (67a03ebf): regressed 46.6 → 42.6 t/s.
   - Decode-strided D=256/512 kernel (4967ab3e): 195 μs/call vs
     flash_attn_v2's 89 μs/call.

5. **Turbo's measured attention-on-gfx906:**
     `mul_mat_vec_f<float,float>` × 84 calls/tok = 0.76 ms/tok
     `soft_max_f32`              × 42 calls/tok = 0.22 ms/tok
     **total attention: 0.98 ms/tok**
   vs candle's `flash_attn_v2_fwd_ktvs_d256 + d512` = **3.98 ms/tok**.
   3 ms/tok of avoidable GPU work = **+9 t/s** on E4B decode if we
   match turbo's approach.

**→ All further work tracked under Phase P (below).**

### Phase P — K layout flip to T-major (2026-04-13)

**Goal:** close the 3 ms/tok attention gap to turbo on E4B decode by
switching KvCache K storage from `(B, H_kv, D, max_T)` to
`(B, H_kv, max_T, D)` so that mat-vec style attention (llama.cpp's
`mul_mat_vec_f` pattern) becomes coalesced on our layout.

**Design target:**
- `attn = K · q^T` becomes one mat-vec per q_head: blockIdx.x = t
  (output index in T), threads stride along D (coalesced since D is
  now contiguous).
- `o = attn^T · V` — V is already T-major `(B, H_kv, T, D)`, fits
  natively.
- Softmax stays as the existing fused kernel.

**Scope (ranked by blast radius):**
- **P1 — KvCache layout.** `candle-nn/src/kv_cache.rs`: flip what
  `k_is_transposed=true` means (current: D-major / stored `D,T`;
  new: T-major / stored `T,D`). Consider just removing the flag and
  standardising on T-major.
- **P2 — Flash-attn kernels.** `candle-hip-kernels/src/flash_attn.cu`,
  `flash_attn_v2.cu`: every kernel with the `K_TRANS` template arm
  reads `k_ptr[d * L_k + row]`. Under T-major K that becomes
  `k_ptr[row * D + d]`. The kernels need to be updated OR they need
  to start taking explicit k_stride args. Easiest: swap the
  semantics of `K_TRANS=true` to mean (T, D) row-major, update the
  indexing in the one template, recompile. Every other kernel with
  custom K-access (`flash_attn_decode_strided_*`, `ktvs_*`,
  `flash_attn_v2_fwd_kt_*`) needs the same treatment.
- **P3 — Callers.** `quantized_gemma4.rs`, `quantized_llama.rs`,
  `quantized_qwen35.rs` all narrow K from the cache before attention.
  Today: `k_full.narrow(3, 0, l_k_padded)`. Under T-major K the
  narrow axis changes to 2: `k_full.narrow(2, 0, l_k_padded)`.
  Grep + fix mechanically.
- **P4 — Mat-vec attention kernel.** New HIP kernel mirroring the
  `mul_mat_vec_f<float,float,1,128,false,false>` template but
  taking explicit K strides (so it works even during the transition
  while some callers still pass D-major K). Grid = (T, H_q, B),
  block_size=128 (2 warps), threads coalesced along D. The
  post-softmax `O = attn · V` is a second mat-vec; the V path works
  natively since V is already T-major.
- **P5 — Dispatch.** `gqa_attention_k_transposed` picks mat-vec for
  `seq_len==1 && head_dim ≥ 128` ahead of flash_attn_v2_ktvs.

**Effort estimate:** 2 days including full regression bench
(TinyLlama, Qwen3.5-9B, Gemma4-E4B / 31B / 26B-MoE) to make sure no
caller silently relies on the old D-major K.

**Risks:**
- Flash-attn-v2 prefill performance might regress if the new K
  indexing is less friendly to the LDS-tile load pattern. Prefill
  is already a separate bottleneck (Phase N); acceptable if decode
  wins dominate.
- G2/G3 captured plans will need re-recording across the layout
  change — straightforward but means re-verifying Phase K
  correctness afterwards.

**Expected result:** E4B decode 52 → ~60-65 t/s (close most of the
gap to turbo's 75 t/s; residual gap is MMVQ tuning per Phase L and
CPU overhead).

**Landed (2026-04-13, Stage 1 — opt-in):**

Scaffolding done, measured, OPT-IN via `CANDLE_KV_TMAJOR=1`:
  default (legacy D-major K):   51.82 t/s
  Phase P (CANDLE_KV_TMAJOR=1): **53.87 t/s**  (+3.9%, output byte-identical)

Implementation:
- New `KvCache::new_k_canonical_stable(dim, max_seq_len)` stores K
  in `(B, H_kv, max_T, D)` row-major (D contiguous), identical to V.
- New HIP kernel `gqa_decode_mv_d{64,128,256,512}_f32` in
  `candle-hip-kernels/src/flash_attn_v2.cu`. One block per (b, h_q),
  64-thread warp. Threads stride D coalesced. Online softmax, no LDS
  K/V tiling. Takes dynamic `L_k_iter` for G2 replay compatibility.
- Rust wrapper `gqa_attention_decode_mv` in
  `candle-core/src/hip_backend/flash_attn.rs`.
- Gemma4 opt-in dispatch in `quantized_gemma4.rs`:
  - KvCache constructor selection via `CANDLE_KV_TMAJOR`.
  - K narrow dim 2 (canonical) vs dim 3 (legacy).
  - Decode L_q=1: route to `gqa_attention_decode_mv` (both
    `shared_qkv=Some` and `shared_qkv=None` branches).
  - Prefill: transpose canonical K back to D-major and dispatch via
    legacy `forward_with_kv_transposed` (hits
    `flash_attn_v2_fwd_ktvs_d{256,512}`) — keeps d=512 prefill
    working since the canonical `flash_attn_v2_fused` lacks a d=512
    variant, and its rocBLAS fallback errors on the E4B shape.

Per-kernel attention under Phase P (all 42 layers routed through the
new kernel, 63 decode tokens):
  gqa_decode_mv_d256_f32:   73.0 us/call × 2205 = 2.55 ms/tok
  gqa_decode_mv_d512_f32:   71.3 us/call × 441  = 0.50 ms/tok
  total attention:          3.05 ms/tok
  (baseline flash_attn_v2:  3.98 ms/tok — saved 0.93 ms/tok)

Why Stage 1 is opt-in, not default:
- G2/G3 captured plans reference the legacy kernel names; enabling
  Phase P under G2 regresses E4B decode from 46 → 27 t/s because the
  captured plan rebuilds on every replay. Defaulting Phase P on
  would silently break the Phase K gemma4 G2/G3 users. Opt-in keeps
  the upgrade path safe.

**Why the +3.9 % gain is smaller than the +9 t/s target:**

The per-call mat-vec time is 73 μs for d=256 vs turbo's `mul_mat_vec_f`
at ~9 μs — still 8× slower per call. Turbo's kernel is narrower in
scope (separate Q·K^T, softmax, V·attn — three launches per layer vs
our single fused kernel), and uses vectorised float2 reads + a 128-
thread block. Our kernel uses 64 threads (one warp) and scalar reads.

Optimisation ideas tracked under Phase Q (below):

### Phase Q — Integrate Phase P with G2/G3, tune mat-vec kernel

- **Q1** — Record G2 plans against the canonical-K path. Currently
  `CANDLE_G2_REPLAY=1 + CANDLE_KV_TMAJOR=1` regresses to 37 t/s
  because the decode state machine keeps rebuilding when the
  captured kernel signature doesn't match live dispatch. Need the
  recorder to either (a) accept the new kernel names or (b) flip the
  entire plan's kernel-call table when KV layout changes.
- **Q2** — Kernel tuning for `gqa_decode_mv`. Candidates:
    * 128-thread block (2 warps), warp-cooperative K loading into
      registers, then parallel dot product reduction.
    * `float2`/`float4` vectorised loads (8/16 bytes per thread per
      iteration) — matches llama.cpp-turbo's `float2` pattern in
      `ggml-cuda/mmvf.cu:132`.
    * Split-K via multiple blocks per (b, h_q) and a merge kernel,
      filling gfx906's 60 CUs (we currently issue only n_head
      blocks = 8-32 per layer = partial occupancy).
- **Q3** — Drop the per-decode `.contiguous()` on K/V narrow views
  by extending the mat-vec kernel to take explicit head/seq/dim
  strides (like `flash_attn_v2_kt_strided_v`). Eliminates the
  per-layer ~5 μs copy.
- **Q4** — Dedicated `flash_attn_v2_fwd_d512_f32` (K_TRANS=false)
  kernel so prefill on E4B global layers doesn't need the transpose-
  to-D-major workaround.

**Effort:** Q2 alone should close most of the remaining gap to the
9 t/s expected improvement. Q1 is a separate track — important for
users who need both G2 replay and fast decode.

### Phase R — LDS `+1` padding (bank-conflict elimination)

Motivation from the survey (§6 + §8-1): skyne98 [wiki-gfx906 LDS
layout study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-lds-layout-standard-llm.html)
measures **2× LDS bandwidth** (3974 vs 1865 GB/s) when a column-
consumed shared-memory tile gets a `+1 vec4` padding row to serialise
bank conflicts. Turbo applies this in `cpy.cu:59`:

```cpp
__shared__ float tile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];
```

— only on the 2D transpose kernel. We have no `+1` padding anywhere.
Two direct targets in our codebase:

- **R1** — `flash_attn_v2.cu`: the `__shared__ float k_lds[BC * D];`
  and `v_lds[BC * D];` declarations in `flash_attn_fwd_v2_impl`
  (template at line 63). Pad the inner dim to `D + 1` (or `D + 4`
  for 16-byte alignment) and adjust indexing. Each warp-coop tile
  read currently hits bank conflicts since D=256 / 64 banks means
  4-way stride-aligned accesses.
- **R2** — `quantized.cu`: MMQ v2f tile32 LDS tiles. Less certain
  of the expected gain (the Tensile gemm kernels turbo uses for
  prefill are already optimised); profile before committing.

**Expected:** attention kernel per-call time drops 10-20 % on d=256
and d=512 paths (LDS read-bound for the K/V tile load). Compounds
with Phase Q2 — the `+1` padding is invisible to the outer
kernel-rewrite work.

**Effort:** 2-4 hours per kernel including regression + perf bench.
Low risk: correctness only depends on the indexing update matching
the declared shape.

### Phase S — TurboQuant KV port (long-context memory win)

Motivation (survey §4, §8-2): turbo ships sub-byte KV cache
quantization (TURBO2_0 at 2.5 bits/val, TURBO3_0 at 4 bits,
TURBO4_0 at 5.3 bits) via lazy F16 shadow cache. ICLR 2026 paper
([Zandieh et al. arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
shows near-lossless quality at 3-4 bit. Gives candle two things:

- **Decode-latency-neutral memory reduction** at batch=1 —
  dequant-to-f16 runs once per cache-write, amortised.
- **32K+ context on 16 GB MI50** — current E4B Q4_0 + KV fits in 8 GB
  at 256-tok context; at 32 K context the f16 KV dominates. Turbo
  quant gets us there.

Reference impls to port from:

- Turbo: `ggml/src/ggml-cuda/convert.cu:765+` (dequant kernels),
  `cpy.cu:549` (append path), `fattn.cu:117-200` (shadow cache).
- Pascal-SAPUI5's AMD-ROCm-specific port:
  [Pascal-SAPUI5/llama.cpp-turboquant](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant)
  — already gfx1100-validated, reuse where possible.

Community observations worth noting before porting:

- [llama.cpp TurboQuant discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
  reports "gfx906 at batch=1 is latency-bound (10 % bandwidth
  utilization)" — dequant overhead eats more of our cycle budget
  than on RDNA3. Expect 5-10 % decode throughput cost on our
  workload (turbo measures ~1 % on RX 7900 XTX).
- Trade-off: big memory win, small latency loss. Worth it for
  long-context; probably not worth defaulting on for the short-
  prompt bench workload.

**Scope:**

- **S1** — Add block_turbo2/3/4 dtypes to `candle-core/src/quantized/`.
  Mirror `k_quants.rs` structure.
- **S2** — Port dequant-to-f16 shadow kernels. Run paused under
  G2 / lazy on first access.
- **S3** — Extend KvCache with a shadow-cache indirection (or reuse
  the existing pad + narrow machinery with dtype=u8 storage +
  f16 materialised view).
- **S4** — Opt-in via `CANDLE_KV_QUANT={turbo2,turbo3,turbo4}`.
  Default stays f16/f32.

**Effort:** 2-3 days. Gated on Phase Q1 (KV layout changes during S
would collide with G2 plan capture if Q1 isn't done first).

## Order of attack

Updated 2026-04-13 after `SURVEY-GFX906-FORKS-2026-04-13.md`. Phases
H, J, K already landed (see memory + commits); listed here for
completeness only.

```
[DONE: H (n_kv pad + flash-attn-v2 on gemma4)]
[DONE: K (G2/G3 on gemma4, decode correctness)]
[DONE: P Stage 1 (T-major K + mat-vec decode, opt-in)]
       ↓
STEP 1: ROCm 7.2.1 + Tensile migration
        (see ROADMAP-ROCM-722-MIGRATION-2026-04-13.md)
        → unblocks Phase J DPP reductions (≈40% on reduction hotpath)
        → unblocks Phase B3 K-quant warp-coop (decode Q4_K faster)
        → unblocks Phase O rocBLAS sgemv (probably obsolete post-P)
       ↓
STEP 2: Phase R (LDS +1 padding) + Phase Q2 (mat-vec kernel tuning)
        (parallel; neither blocks the other)
        → Phase R:  flash-attn-v2 LDS tiles +1 pad → +10-20% per call
        → Phase Q2: 128-thread block, float2 loads, split-K, strided
                    K args → 73 μs → ~30 μs per gqa_decode_mv call
        Combined expected: Gemma4-E4B decode 54 → 62-65 t/s
       ↓
STEP 3: Phase Q1 (G2 + Phase P integration)
        → flip Phase P from opt-in to default without regressing G2
        → so everyone gets the decode win transparently
       ↓
STEP 4: Phase I (MMQ rewrite) + Phase S (TurboQuant KV) + Phase N (prefill)
        — tracks are independent, pick by schedule / model priority.
        → I: prefill 2-3× across every Q4_0 model (TinyLlama 801 → 2k)
        → S: 5× KV memory compression, long-context unlock
        → N: fused attention for prefill (also needs Phase Q4 — d=512
             K_TRANS=false kernel)
```

## Expected final numbers

Target evolves per step (numbers track Gemma4-E4B Q4_0 single-MI50 decode):

| Step | Gemma4-E4B decode | TinyLlama decode | Qwen3.5 decode | Gemma4 prefill |
|---|---|---|---|---|
| Today (post-P Stage 1 opt-in) | 54 | 267 | broken | 244 |
| +STEP 1 (ROCm 7.2.1) | 55 | 275 | broken | 250 |
| +STEP 2 (Phase R+Q2) | **62-65** | 275 | — | 260 |
| +STEP 3 (Phase Q1 → P default on) | 62-65 (+G2 working) | 275 | — | 260 |
| +STEP 4 (Phase I prefill rewrite) | 65 | 275 | 55 (after J) | **~500** |

Turbo's 75 t/s on E4B decode remains the ceiling at batch=1 (`SURVEY
§7`); closing the last 10 t/s requires either TurboQuant KV (memory-
bound path) or would need MFMA hardware we don't have. 2× turbo
(420 t/s TinyLlama decode) is still unreachable without matrix cores.

## Cross-reference

- `SURVEY-GFX906-FORKS-2026-04-13.md` — landscape + technical-choice
  comparison vs turbo / vllm-mobydick. All phase priorities above are
  traceable to survey §11 recommendations.
- `ROADMAP-ROCM-722-MIGRATION-2026-04-13.md` — STEP 1 detail.
- `REVIEW-CANDLE-VS-TURBO-HIP-KERNELS-2026-04-12.md` — original
  per-kernel comparison (supersedes parts of survey §3-6 that pre-
  date Phase P; survey reflects post-P state).
