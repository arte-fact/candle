# Qwen3.6 integration into candle — plan & design

**Date**: 2026-04-18
**Author**: candle-fork work-in-progress notes
**Target model**: `Qwen/Qwen3.6-35B-A3B` (released 2026-04-16, Apache-2.0)

## 1. Architecture (from the official HF model card)

### 1.1 Top-level shape

| Field | Value | Notes |
|---|---|---|
| Total params | 35 B | |
| Activated params | 3 B | Sparse MoE — only 8 routed + 1 shared expert fire per token |
| Layers | 40 | |
| Hidden dim | 2048 | |
| Vocab size (padded) | 248 320 | |
| Native context | 262 144 | Extensible to 1 010 000 via YaRN |
| MTP | yes | Multi-token-prediction head trained in, for speculative decoding |
| Modality | vision + text | Multimodal input; vision encoder is a separate ViT module |

### 1.2 Layer pattern — *this is the key thing that makes Qwen3.6 new*

The 40 transformer blocks follow a **repeated 3:1 hybrid pattern**:

```
10 × [ 3 × (Gated DeltaNet  → MoE)   +   1 × (Gated Attention → MoE) ]
```

i.e. 30 linear-attention (DeltaNet) blocks + 10 full-attention (Gated Attention) blocks, each followed by a shared-expert MoE. This is the same family as **Qwen3-Next-80B-A3B** but with a different total size and mix.

### 1.3 Per-block hyper-parameters

**Gated DeltaNet (30 blocks)**:
- V heads: 32
- QK heads: 16
- Head dim: 128

**Gated Attention (10 blocks)**:
- Q heads: 16
- KV heads: 2 (GQA ratio 8:1)
- Head dim: 256
- Rotary-embedding dim: 64 (partial RoPE, 25 % of head_dim — matches gemma4-global convention)

**MoE (after every block)**:
- Experts: 256
- Activated per token: 8 routed + 1 shared
- Expert intermediate dim: 512

### 1.4 Comparison vs existing candle archs

| Feature | Qwen3-Next-80B (`quantized_qwen3next`) | **Qwen3.6-35B-A3B** | Gemma4-26B-A4B |
|---|---|---|---|
| Layer pattern | 3 GDN + 1 Gated Attn, repeated | Same | Dense only |
| GDN V / QK heads | 16 / 8 | **32 / 16** | — |
| Full-attn GQA | 16 Q / 2 KV | **16 Q / 2 KV** | 16 Q / 8 KV |
| Head dim | 256 | 256 | 256 (global) / 256 (SWA) |
| Partial RoPE | yes (64 dim) | **yes (64 dim)** | yes |
| MoE experts | 512, 10 routed + 1 shared | **256, 8 routed + 1 shared** | 128, 8 routed (no shared) |
| Shared expert | yes | **yes** | no |
| Expert intermediate | 512 | **512** | 704 |
| Vision encoder | no | **yes** (ViT) | no |
| Native context | 262 144 | **262 144** | 131 072 |
| MTP | yes | **yes** | no |

**The text path is 99 % compatible with `quantized_qwen3next`** — same block types, same hybrid pattern, same partial RoPE, same shared-expert MoE. Only the per-head counts and expert counts change.

The vision encoder is *new territory* for candle's quantized model line; none of the currently-shipped quantized architectures carry a vision tower.

## Bench results (2026-04-18, 4× MI50)

Model: `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (20.9 GB, 40 layers, 256 experts, arch=qwen35moe, identical spec to Qwen3.5-35B-A3B).

| Backend | Config | pp512 t/s | tg64 t/s | ratio pp | ratio tg |
|---|---|---:|---:|---:|---:|
| **Turbo** (llama.cpp gfx906-turbo) | PP-4 | **1012.8** | **61.25** | 1.00× | 1.00× |
| **Candle + B2 + C1** (2026-04-18) | PP-4 | **466.80** | 25.62 | **0.461×** | **0.418×** |
| Candle + B2 only (Q4_K turbo_moe) | PP-4 | 412.11 | 25.24 | 0.407× | 0.412× |
| Candle default (pre-B2) | PP-4, no flag | 178.93 | 25.65 | 0.177× | 0.419× |
| Candle + `--split-prompt` (debug) | PP-4, per-token | 15.49 | 25.04 | 0.015× | 0.409× |

**Important correction (2026-04-18)**: earlier versions of this doc cited `21.76` / `0.41×` for candle pp512 — that measurement was taken with `--split-prompt`, which forces per-token forward and is **not the candle default**. The actual default-path candle is 178.93 t/s, 0.177× turbo on pp, with tg unchanged. See `bench/qwen36_3way_findings.md` for the full reprofile. The --split-prompt numbers are kept here for historical context only.

Candle loads and runs Qwen3.6 **unchanged** via the existing `quantized_qwen35_moe` path — zero integration code needed for the text-only path. Turbo likewise works unchanged because llama.cpp already has `qwen35moe` arch support (merged months ago for Qwen3.5).

**Gap analysis**: after the B1 correction (removing --split-prompt), candle pp512 is 0.177× turbo and tg is 0.42× turbo. The remaining pp gap is concentrated in a single kernel (`indexed_moe_forward_q4k_q8_1`, 81 % of candle GPU time, 29× slower than turbo's batched `mul_mat_q<Q4_K,64>`). Physical gather-by-expert MoE is the next concrete lever.

### Perf investigation 2026-04-18 — honest summary

Three optimisation prongs (P1/P2/P3) were chased in a single session. All three landed in some form; **none produced a measured end-to-end improvement on this workload**. What was actually learned:

| Prong | Intent | What landed | Measured impact |
|---|---|---|---|
| **P1** | Eliminate `alloc_zeros` leak attributed to 8.5% of GPU time (`fillBufferAligned`) | Audit of explicit call sites only — all 100 are load-time, not forward-time. No code change. | None. The 8.5% source is **NOT conclusively identified**; the leading candidate (rocBLAS workspace init inside the 1020 `Cijk_*` gemm calls) is an unverified hypothesis, not a measured fact. |
| **P2** | Q4_K MMQ kernel to undercut rocBLAS on Qwen3.6's Q4_K_S weights | `mul_mat_q4_K_gfx906_v2f_tile{8,16,32}` kernel + dispatcher entry. Compiles. | None. Kernel **never dispatched** on the Qwen3.6 bench (absent from rocprofv3 top-20). Correctness **not validated** against a reference path. No before/after win. |
| **P3** | Batch GDN step across prefill tokens (510 → 60 kernel calls) | `CANDLE_GDN_PREFILL_CHUNK` env var in `quantized-qwen35` binary. Batched GDN path runs correctly at small chunks on short prompts. | None at pp512. Chunk≥2 **OOMs** — MoE activation buffers scale with chunk × 256 experts × 512 expert_ffn × hidden. Win (~3× pp) was extrapolated from call-count reduction, not measured. |

The 24.5% rocBLAS attribution to 30-layer DeltaNet state matmuls × 17-tokens-per-chunk math was also partially circumstantial (predicted ~1500 calls vs ~1020 observed; 500-call gap unreconciled).

**What this means for a real Qwen3.6 perf follow-up**:
1. P3's real fix is MoE-aware chunked prefill (batch GDN across `chunk`, but fan MoE expert matmul per-token so expert activation doesn't scale with chunk). Multi-day — equivalent to vLLM's expert-batch dispatch. This is the highest-confidence lever for the 47× pp512 gap.
2. P2's Q4_K MMQ kernel is scaffolding that needs (a) a correctness test on a dense Q4_K model and (b) a dispatcher audit to understand why Qwen3.6's Q4_K tensors don't route through it.
3. P1's 8.5% needs direct instrumentation (`hipMemsetAsync` callsite annotation or rocBLAS LD_PRELOAD shim) before assuming rocBLAS workspace is the source.

Primary lesson: the pp512 0.41× ratio reflects a **full GDN+MoE+Attention kernel-family gap**, not any single hotspot. One-session kernel patches won't close it.

## 2. Integration plan

### 2.0 Audit result (2026-04-18)

Confirmed by comparing GGUF metadata of **Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf** (already on disk) vs the Qwen3.6 spec: **identical architecture**, identical `qwen35moe` arch string. Candle's existing `quantized_qwen35_moe.rs` + `quantized-qwen35` example binary auto-detect arch from GGUF metadata and handle this architecture unchanged — so the text path is a zero-code integration.

Consequence:
- **No new `quantized_qwen36.rs` file needed** — arch is qwen35moe, candle already ships the loader.
- **No new example binary needed** — `quantized-qwen35` already multiplexes qwen35 / qwen35moe / qwen3next / qwen3moe at runtime.
- Remaining work is **benchmarking** (candle stock + candle TP+EP vs turbo) and optionally **TP+EP port** for the qwen35moe family (GatedAttention + DeltaNetLayer load_tp hooks; MoeExperts EP already works).

### 2.1 Text-only path — deprecated (merged into 2.0)

Create `candle-transformers/src/models/quantized_qwen36.rs` **cloned from `quantized_qwen3next.rs`** with:

1. **Arch string**: `qwen3_5_moe` (yes, the HF config reuses the Qwen3.5 MoE arch string — confirmed by the `other=qwen3_5_moe` tag on the HF page).
2. **Layer pattern resolver**: read `model.layers[*].mlp.experts` count + `layer_types` metadata and build the per-layer dispatch table. Qwen3-Next's current code already does this for its specific 3:1 pattern — should drop in unchanged.
3. **Per-block wiring**:
   - 30 `DeltaNetLayer` (V=32, QK=16, d=128) — need to verify candle's `DeltaNetLayer` parameterises V/QK head counts independently; patch if not.
   - 10 `GatedAttention` (q_heads=16, kv_heads=2, head_dim=256, rope_dim=64) — this is identical to qwen3next's full-attn layer; should just work.
4. **MoE**: 256 experts / 8 routed / 1 shared / 512 intermediate — use the existing `MoeExperts` block which already supports shared expert + shared_gate. Only change is router output dim (256 instead of 512). No code change expected.
5. **Final norm + lm_head + tied-weight check** — same as qwen3next.
6. **TP+EP compatibility** (our X8 session):
   - With W=2: n_head=16/2=8, n_kv_head=2/2=1, experts=256/2=128 — all clean divisions, no uneven-shard code path needed.
   - With W=4: n_head=16/4=4 ✓, **n_kv_head=2/4** → fall back to `kv_replicated` (the code path we added today for gemma4-26B-A4B SWA layers), experts=256/4=64 ✓.
   - Dense FFN isn't present (MoE replaces it), so the `intermediate=2112` divisibility issue from gemma4 doesn't apply.

### 2.2 Multi-modal (vision) — phase 2

Vision encoder is a Qwen-ViT variant. We defer this phase until text-only is benchmark-validated:
- New `candle-vision` module or reuse of existing `blip` / `clip` encoders.
- Patch embedding, 2D-RoPE, bfloat16 path (vision weights are BF16, not quantized).
- Multi-modal input collator: cross-attention lane that reads `inputs_embeds` from the vision tower.
- Video sampling / frame batching — can start at 2 fps, same default as vLLM.

Gated by `--multimodal` flag. Text-only builds skip vision tower instantiation.

### 2.3 MTP (Multi-Token Prediction) head — phase 3

Qwen3.6 ships trained MTP weights for speculative decoding. vLLM exposes this via `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`. For candle:
- Add `MTPHead` block (small prediction head, usually a linear + output_proj sharing lm_head).
- Extend the decode loop to propose N future tokens, verify them against a draft-model or EAGLE-style rewind.
- Out of scope for initial integration; test gain on gfx906 first (MTP is CPU-launch-heavy on a throughput-optimised runtime).

## 3. GGUF expectations

- Unsloth has released Qwen3.6 GGUFs at `unsloth/Qwen3.6-*-GGUF`; **Unsloth Q4_K_S for -35B is 20.9 GB** (cf. Simon Willison's pelican write-up).
- Text-only GGUFs land under `Qwen3.6-35B-A3B-GGUF`; multimodal variants ship separate `mmproj` files (note from Unsloth docs: "currently no Qwen3.6 GGUF works in Ollama due to separate mmproj vision files — use llama.cpp compatible backends").
- For our bench flow: download via `candle-bench fetch 'https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_S.gguf' --expected-bytes $N` once URL is known.

## 4. VRAM budget on 4× MI50 (16 GB each)

### Q4_K_S (~20 GB model file)

| Config | Per-rank VRAM (est.) | Fits? |
|---|---|---|
| PP-1 (single GPU) | 20 GB + KV | **no** (OOM; 16 GB card) |
| PP-2 | ~10 GB + half KV | yes |
| PP-4 | ~5 GB + quarter KV | yes, generous headroom |
| **TP+EP W=2** | ~10 GB + half KV | yes |
| **TP+EP W=4** | ~5 GB + quarter KV | yes |

### Q8_0 (~38 GB model file, extrapolated)

| Config | Per-rank VRAM (est.) | Fits? |
|---|---|---|
| PP-1 | 38 GB | **no** |
| PP-2 | ~19 GB + KV | **tight** — may OOM with full 262k context KV |
| PP-4 | ~9.5 GB + KV | yes |
| **TP+EP W=2** | ~19 GB — OOM | **no** |
| **TP+EP W=4** | ~9.5 GB | yes |

**Recommendation**: benchmark Q4_K_S at TP+EP W=2 first (best speed + comfortable VRAM), then Q8_0 at TP+EP W=4 if the quality gap demands it.

## 5. Estimated perf ceiling on 4× MI50

Extrapolating from our gemma4-26B-A4B results (Apr 2026 session):
- Turbo (baseline PP-4, 68 t/s decode, 911 pp512) is the target.
- Candle stock PP-4 hit 0.73× turbo on gemma4-26B-A4B. Expect similar on Qwen3.6 if the qwen3next path's hybrid GDN+Gated Attn already has kernel parity (per memory notes, qwen3next got pp512=29.98 → 31.20 and tg=20.64 → 35.34 after Q8 KV).
- TP+EP W=2 hit 0.55× turbo on gemma4-26B-A4B. Same expectation for Qwen3.6 as-is; the 0.75× roadmap target needs further all-reduce work (section 6).

## 6. Dependencies / blockers

| # | Item | Blocker? | Action |
|---|---|---|---|
| 1 | `DeltaNetLayer` V/QK head-count parameters | Unknown — check | Read `candle-transformers/src/models/quantized_blocks/delta_net.rs` |
| 2 | Qwen3.6 GGUF availability at release quality | no | unsloth already ships variants |
| 3 | Vision encoder | **yes** for multi-modal, no for text-only | Defer to phase 2 |
| 4 | MTP head | no for bring-up | Defer to phase 3 |
| 5 | TP+EP correctness on 3:1 hybrid | no — qwen3next uses the same pattern; our TP hooks on StandardAttention/GatedAttention should carry over | Verify once loader compiles |
| 6 | n_kv_head=2 at W=4 | no — we added `kv_replicated` fallback in today's session | — |
| 7 | `Content::tensor_row_range` / `column_range` on qwen3_5_moe tensor names | no — names are different strings but same shape semantics | Verify via `candle-bench meta` once a GGUF is downloaded |

## 7. Milestones

**M1 — text-only load**: clone qwen3next → qwen36, patch architectural knobs, load the unsloth Q4_K_S GGUF without error on 1 GPU. *(~1 day)*

**M2 — text-only forward parity**: one forward step, logit argmax matches vLLM / turbo for a fixed prompt. *(~1 day)*

**M3 — text-only TP+EP W=2 bench**: run our hip_gemma4_tp_ep_bench analog, report decode t/s and VRAM vs turbo's llama-bench. *(~1 day)*

**M4 — vision encoder**: load ViT, run a small image+text prompt, compare against vLLM. *(~1 week)*

**M5 — MTP**: implement the prediction head, wire speculative decoding. *(~1 week)*

Total bring-up for text-only: **~3 days**. Full feature parity incl. vision + MTP: **~3 weeks**.

## 8. References

- GitHub repo: https://github.com/QwenLM/Qwen3.6
- HF model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Unsloth GGUF collection: https://huggingface.co/collections/unsloth/qwen3 (Qwen3.6 variants)
- Qwen Team release blog: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- Simon Willison's LM Studio note (Q4_K_S file size = 20.9 GB): https://simonwillison.net/2026/Apr/16/qwen-beats-opus/
- Prior art in candle: `candle-transformers/src/models/quantized_qwen3next.rs` (3:1 GDN+GAttn hybrid, same family), `quantized_gemma4.rs` (TP+EP reference port), `DOCS-QWEN35-DISPATCH-2026-04-14.md` (qwen3.5 attention dispatch).
