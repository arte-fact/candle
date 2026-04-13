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

**Remaining:** replay output still incorrect (" I" instead of " help"
at index_pos=14) despite full kernel coverage and per-call output
evolution. Detailed diagnostics from earlier:

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

## Order of attack

```
H (Gemma4 attention) → J (Qwen correctness) → I (MMQ rewrite, cross-cutting)
       ↓                       ↓
K (Gemma4 G2/G3)          (Qwen bench valid)
       ↓
L (MMVQ tune) + M (MoE FFN) + N (prefill) in parallel
```

## Expected final numbers after H+K+I

| Model | Current | After H+K | After H+K+I |
|---|---|---|---|
| TinyLlama Q4_0 decode | 267 | 267 (no change) | 280 (prefill faster doesn't help decode) |
| Gemma4-E4B Q4_0 decode | 44 | 62 | 65 |
| Qwen3.5-9B Q4_1 decode | broken | 55 (after J) | 58 |
| TinyLlama Q4_0 prefill | 801 | 801 | ~2 000 |
| Gemma4-E4B Q4_0 prefill | 87 | 87 | ~250 |

Targets remain honest: 2× turbo (420 t/s TinyLlama decode) is still
unreachable without matrix-core hardware. But Gemma4/Qwen gap to vanilla
llama.cpp should close and we get faster prefill everywhere.
