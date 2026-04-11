# Optimization roadmap: porting the best of llamacpp-turbo and vllm-mobydick into candle

**Date:** 2026-04-11
**Source data:**
- `BENCH-3WAY-2026-04-11.md` — TinyLlama 3-way + Qwen3.5-9B 2-way bench results
- Earlier rocprofv3 kernel breakdown (`/tmp/bench/REPORT.md` — 1.35M dispatches in candle vs 127k in turbo)
- `~/.claude/projects/-artefact/memory/project_perf_vs_turbo.md` — measured per-model gaps after Phase 1+2b/2d

This complements `ROADMAP-ROCM-GFX906.md` (the macro phase plan). Where the macro plan says "port the gfx906 fused kernels", this document says **which kernels to port first, in what order, against what measured gap, with what expected payoff.**

---

## TL;DR — the gap by category

From rocprofv3 on Qwen3.5-9B Q4_1, single-GPU, 1149-token prompt + 128 decode:

| Category                          | Turbo (ms) | Candle (ms) | Ratio | Where the gap lives                                                                                |
|-----------------------------------|-----------:|------------:|------:|----------------------------------------------------------------------------------------------------|
| GDN / pointwise micro-kernels     |        239 |        4948 | **20.7×** | candle launches 842k tiny `bmul/badd/usqr/usilu/silu_mul/rmsnorm/...` ops vs turbo's single fused `gated_delta_net_cuda` kernel |
| Quantized matmul (prefill)        |        513 |        4118 |  **8.0×** | Phase 2b/2d kernel works but is 11× slower per call than turbo's stream-K mmq                      |
| Quantized matvec (decode)         |       1396 |        4793 |  **3.4×** | Q5_K still on chunked-vector fallback; Q4_1 MMVQ is fine                                            |
| rocBLAS attention `Cijk_*`        |          8 |        1829 |  **224×** | candle's Phase 1 GQA reshape feeds rocBLAS Tensile (small-tile, decode-time inefficient); turbo has its own custom attention kernel |
| memcpy / pipeline                 |        168 |        1474 |   **8.8×** | candle issues 289k `__amd_rocclr_copyBuffer` calls — Tensor temporary alloc + 4-GPU pipeline transfers |
| Q8_1 input quantize               |        137 |         147 |   1.07× | parity ✓ (already fast)                                                                            |
| Total GPU kernel time             |       2704 |       17448 |  **6.5×** | combined effect                                                                                     |
| Wall-clock first-to-last dispatch |       5.05 |       26.77 |   5.3× | reflects launch-overhead tax                                                                       |
| Kernel dispatches                 |     127k   |       1.35M |  **10.7×** | each launch costs 3-5 µs of HIP runtime overhead → ~5 s of pure launch overhead                   |

From the 3-way TinyLlama bench, vllm-mobydick added a third axis: **its triton-compiled prefill kernels are 2.5× faster than turbo's prefill** (6690 vs 2687 t/s on TinyLlama). vLLM's decode is slow because of Python dispatch latency, not GPU work — that's a serving-mode tradeoff that doesn't apply to candle's Rust runtime.

## What candle should steal, in priority order

Each item lists **source repo**, **target candle file**, **which dispatch category it shrinks**, **measured candle ms today**, **expected ms after fix**, and **est. SLOC**.

---

### P0 — Fused gated delta net kernel ⚡ biggest single win

**Current pain**: candle's `delta_net_step_vectorized` runs the per-token GDN forward as a sequence of small Tensor ops. Each one is a separate HIP kernel launch. For qwen35-9B prefill that's **842,728 launches consuming 4948 ms of GPU time and ~5 s of launch overhead** — by far the biggest line in the rocprofv3 breakdown.

**Source to port**: `llamacpp-turbo/ggml/src/ggml-cuda/gated-delta-net.cu` (or wherever the fused `gated_delta_net_cuda<128,false>` kernel lives — it appeared in rocprofv3 as `_Z20gated_delta_net_cudaILi128ELb0EE...` with **903 calls totaling 30 ms**, head-dim parameterized).

**Target**: new file `candle-hip-kernels/src/gated_delta_net.cu` + a thin Rust launcher in `candle-core/src/hip_backend/` invoked from the module that today calls `delta_net_step_vectorized` (look at `candle-transformers/src/models/quantized_blocks/delta_net.rs:delta_net_step_vectorized` and its callers in `quantized_qwen35.rs`, `quantized_qwen35_moe.rs`).

**Implementation**: one CUDA/HIP kernel that takes the per-layer state, q/k/v/gate/beta inputs for the current token (or chunk of tokens), and the head-dim constant as a template parameter. Single launch per layer per (chunk of) tokens, fully fused norm/silu/multiply/state-update.

**Expected payoff**:
- 842k → ~3k dispatches = **278× fewer launches** for the GDN path alone.
- 4948 ms → ~30 ms (matching turbo) = **165× speedup** on the GDN category.
- Decode launch-overhead drops from ~5 s to ~0.3 s.
- **Net qwen35-9B decode**: roughly 2.5–3× from this alone (35 t/s → 90+ t/s), without touching anything else.

**Estimated SLOC**: ~600 lines of HIP + 100 lines of Rust glue. Largest risk is correctness — we have a vectorized scalar reference (`test_delta_net_vectorized_matches_scalar_reference`) we can reuse to validate.

**Touches**: `quantized_blocks/delta_net.rs`, `candle-hip-kernels/src/quantized.cu` (or new file), `candle-core/src/hip_backend/`, and the GDN tests.

---

### P1 — Stream-K MMQ port (finish Phase 2c/2e from `synchronous-drifting-allen.md`)

**Current pain**: Phase 2b/2d landed correct Q4_0/Q4_1/Q8_0 single-launch MMQ kernels, but each call averages 23–32 ms vs turbo's ~2.5 ms — **11–13× slower per call**. For prefill on qwen35-9B that's 4118 ms vs turbo's 513 ms.

**Why our kernel is slow**: one Wave64 warp per 64×8 output tile, sequential K reduction in registers, no LDS staging, no async prefetch, no stream-K K-dimension parallelism. Correctness-first, perf-second by design.

**Source to port**: `llamacpp-turbo/ggml/src/ggml-cuda/mmq.cuh` (4273 lines, the `mul_mat_q` template) + `mmq.cu` (489 lines, the `launch_mul_mat_q` host dispatch with `use_stream_k`) + `gfx906/matmul/mmq.cuh` (vectorized int4 loads) + `gfx906/matmul/mmq-prefetch.cuh` (async prefetch).

**Target**: extend `candle-hip-kernels/src/quantized.cu` (or split into a new `mmq.cuh` header) and add the stream-K decomposition + fixup pass. Wire into `candle-core/src/quantized/hip.rs::dequantize_matmul` as the dispatch for `b*m >= 9`.

**Expected payoff**:
- Per-call: 23-32 ms → 2-3 ms (match turbo).
- Category total: 4118 ms → ~500 ms = **8× speedup** on prefill matmuls.
- **qwen35-9B pp512**: ~78 t/s → 250-300 t/s (closer to turbo's 450 t/s; remaining gap is GDN, addressed by P0).

**Estimated SLOC**: ~2000 lines of HIP (this is mostly verbatim port from upstream, with template instantiation list narrowed to `{Q4_0, Q4_1, Q8_0} × mmq_x ∈ {8,16,32,64}`).

**Already planned**: Phase 2c/2e of `/home/sandbox/.claude/plans/synchronous-drifting-allen.md`. This is the next chunk of that plan after the working Phase 2b/2d landed.

---

### P2 — Custom flash-attention-style kernel (replace rocBLAS attention path)

**Current pain**: candle's Phase 1 GQA reshape converts attention into a batched rocBLAS GEMM. rocBLAS picks the smallest `Cijk_*_MT16x16x4_*` Tensile kernel for the small per-batch shape at decode time. Result: **93,920 rocBLAS calls totaling 1829 ms** vs turbo's **16 calls totaling 8 ms** (turbo uses its own non-rocBLAS attention kernel).

**Why turbo wins**: turbo's attention kernel is a hand-written tile-based Q·K^T·V kernel that fuses softmax inline and uses GQA-aware indexing (no Q reshape, no rocBLAS, no separate softmax kernel). Mobydick does the same via a triton-compiled MLA kernel (BLOCK_M=64, BLOCK_N=16) — that's where its 6690 t/s tinyllama prefill comes from.

**Source to port**: `llamacpp-turbo/ggml/src/ggml-cuda/fattn-tile.cu` (or whichever `fattn-*.cu` is the tile-based, GQA-aware FA-1 kernel) + `fattn-common.cuh:1027` (`gqa_ratio = nh_q / nh_kv` indexing). For inspiration on the prefill path, mobydick's `vllm/attention/backends/triton_attn.py` shows the BLOCK_M/BLOCK_N tuning for gfx906.

**Target**: new crate `candle-flash-attn-hip` (already in the macro roadmap as Phase 4.1) **or** a `flash_attn_hip.rs` module in `candle-nn` that calls a new `candle-hip-kernels/src/attention/fattn_tile.cu`. The hook point in the Rust side is `gqa_attention()` in `candle-transformers/src/models/quantized_blocks/attention.rs` — replace its Q-reshape + matmul + softmax + matmul chain with a single HIP-kernel call.

**Expected payoff**:
- rocBLAS attention: 1829 ms → ~8 ms (match turbo) = **228× speedup** on this category.
- 93,920 dispatches → ~150 = **626× fewer launches**, freeing ~370 ms of launch overhead.
- **qwen35-9B decode** (where attention dominates): an additional ~1.5× on top of the GDN fusion win.

**Estimated SLOC**: ~1500 lines of HIP for the kernel + ~300 Rust. Larger risk: correctness across all the variants candle's attention supports (causal mask, GQA, optional Q/K norms, sigmoid gate from `GatedAttention`). Recommend porting **only the inference (prefill+decode) variant**, keeping rocBLAS as a fallback for any shape the kernel doesn't handle.

**Reusable test surface**: the 8 GQA reshape unit tests in `quantized_blocks/attention.rs::tests` already give us an oracle.

---

### P3 — Pipeline buffer pre-allocation (kill the memcpy storm)

**Current pain**: 289,002 `__amd_rocclr_copyBuffer` calls totaling 1474 ms. Each call is ~5 µs of launch overhead and ~5 KB of data. They come from two sources:
1. **Per-op Tensor temporary alloc** in `candle-core/src/hip_backend/mod.rs` — every op creates a new `HipSlice` via `hipMallocAsync`, then frees it via `hipFreeAsync` (or holds the reference until the next forward step). The `dropped Tensor → freed buffer → next op alloc` cycle re-issues a malloc.
2. **4-GPU layer-split pipeline** activations crossing GPU boundaries via `to_device(Cpu).to_device(next_gpu)` — that's 2 host roundtrips per inter-GPU transfer.

**Source to port**: turbo's pipeline buffer management is just `cudaMemcpyAsync` with pre-allocated dst buffers, no big lesson there. Mobydick uses paged KV cache + a memory pool. The right fix in candle is **a per-stream workspace pool** in `HipDevice` keyed by buffer size buckets.

**Target**: extend `candle-core/src/hip_backend/device.rs::HipDevice` with a `WorkspaceArena` that caches up-to-N freed buffers per size bucket. Wire it into `BackendStorage::alloc_uninit` so per-op temporaries hit the cache instead of `hipMallocAsync`. For pipeline transfers, pre-allocate a per-layer activation buffer on each GPU at model load time and reuse it across forward steps.

**Expected payoff**:
- memcpy/fill: 1474 ms → ~150 ms = **10× speedup** on this category.
- Frees ~1.4 GB/s of bandwidth previously spent on per-op alloc traffic.
- Smaller wall-clock impact than P0/P1/P2 but **dramatically reduces sys-CPU time** (the 94 s of sys-time on candle's qwen35-9B run is mostly HIP runtime spent in malloc/free).

**Estimated SLOC**: ~400 lines of Rust. No HIP code. Easy to TDD via `cargo test`.

**Side benefit**: this is a prerequisite for HIP graphs (Phase 3.10 in the macro roadmap). With pre-allocated buffers, capturing a candle Tensor forward as a HIP graph stops being counter-productive.

---

### P4 — Fused norm + silu + quantize kernels (small but free win)

**Current pain**: candle issues separate kernels for `rmsnorm`, `silu_mul`, `bdiv` (residual), `quantize_q8_1`. Each pair could be one fused kernel.

**rocprofv3 numbers**:
- `rmsnorm_f32`: 32928 calls × 4.4 µs = 145 ms
- `silu_mul_f32`: 34720 calls × 4.2 µs = 146 ms
- `quantize_q8_1`: 25728 calls × 5.7 µs = 147 ms
- `rmsnorm_add_f32`: 8064 calls × 9.2 µs = 75 ms
- Total: ~513 ms across ~100k launches

**Source to port**: turbo's `gfx906/fused/norm-fused-q8.cu` (mentioned in macro Phase 5.1).

**Target**: `candle-hip-kernels/src/gfx906/fused/` (new dir).

**Expected payoff**: 513 ms → ~150 ms = **3.4× speedup** on the fused-eligible categories, plus 100k fewer launches (~400 ms of launch overhead).

**Estimated SLOC**: ~300 lines of HIP. Independent of the bigger items above so it's a good "small wins" task.

---

### P5 — K-quant MMQ (Q2K..Q6K)

**Current pain**: `mul_mat_vec_q5_K_q8_1_cuda8` takes 18.7 % of candle's GPU time on qwen35-9B because the model's embed/output head tensor is Q5_K (not Q4_1). This is the chunked-vector fallback path — slower than the (now-correct) Phase 2 MMQ but candle has no MMQ for K-quants yet.

**Source to port**: `llamacpp-turbo/ggml/src/ggml-cuda/mmq.cuh` template instantiations for `GGML_TYPE_Q{2,3,4,5,6}_K`.

**Target**: extend the Phase 2 MMQ port to instantiate the K-quant variants. Once the stream-K Q4_0/Q4_1/Q8_0 path lands (P1), this is mostly per-dtype copy-paste of the load_tiles + vec_dot logic.

**Expected payoff**:
- For models with K-quant tensors (qwen35-9B output head, qwen35-35B-Q8_K_XL, gemma4-31B-Q8_0): closes the Q5_K decode path which is currently 18.7 % of total GPU time on qwen35-9B.
- 3193 ms → ~400 ms = **8× speedup** for the Q5_K category.

**Estimated SLOC**: ~1000 lines of HIP (mostly mechanical).

**Already planned**: Phase 2f of `synchronous-drifting-allen.md`.

---

### P6 — Triton-compiled attention prefill kernel (the mobydick borrow)

**Current pain**: mobydick's prefill on TinyLlama hits **6690 t/s** vs turbo's 2687 and candle's 384. The triton-compiled MLA prefill kernel (`fwd_prefill BLOCK_M=64, BLOCK_N=16, waves_per_eu=1, num_warps=4`) is genuinely faster than turbo's hand-written CUDA kernel.

**The catch**: this is a triton-JIT-compiled kernel. Candle has no triton runtime in its build pipeline today — adding one means linking `libtriton.so` into `candle-hip-kernels` or invoking a triton compiler at build time to produce HSACO. Both are non-trivial.

**Source to port**: `triton-gfx906/python/triton_kernels/...` is the kernel set; mobydick wraps them in `vllm/attention/backends/triton_attn.py`.

**Two paths**:
1. **Build-time compile**: invoke triton at `candle-hip-kernels/build.rs` time to produce HSACO blobs that get embedded the same way the existing `.cu` files are. Requires triton CLI in the build env.
2. **Runtime JIT**: link `libtriton.so` and call `triton.jit` from Rust at first-call time. Adds a ~80 MB runtime dependency and cold-start latency.

**Expected payoff**: prefill on llama-class models reaches mobydick numbers (~6700 t/s on TinyLlama, scale up for larger models). For qwen35-9B that's ~600 t/s prefill (matching or exceeding turbo's 450 t/s).

**Recommendation**: **defer** unless P2 (custom flash-attention HIP kernel) underdelivers. Adding a triton dep contradicts candle's "pure HIP kernels via hipcc" architecture. If P2 lands a custom kernel that hits ~3000 t/s on TinyLlama prefill, that's already turbo-parity and we don't need triton.

**Estimated SLOC**: ~200 Rust + 1500 build infrastructure if going the build-time route. Out-of-tree dependency (`libtriton.so`) if going the runtime route.

---

## Priority matrix

| # | Item | Source | Expected speedup | Complexity | Blocks/depends on |
|---|---|---|---|---|---|
| **P0** | Fused gated_delta_net | turbo | 2.5–3× decode on qwen35-family | M (~700 LOC) | nothing — independent |
| **P1** | Stream-K MMQ | turbo | 1.5× prefill, 11× per-call | M-L (~2000 LOC) | extends Phase 2b/2d (already in tree) |
| **P2** | Custom flash-attention HIP | turbo + mobydick | ~1.5× decode + 200× attention | L (~1800 LOC) | nothing — independent |
| **P3** | Pre-allocated workspace pool | own design | sys-CPU 94 s → ~10 s, frees ~400 ms | S-M (~400 LOC) | nothing |
| **P4** | Fused norm+silu+quantize | turbo | ~3.4× on those categories | S (~300 LOC) | nothing |
| **P5** | K-quant MMQ (Q2K..Q6K) | turbo | 8× on Q5_K decode (where used) | M (~1000 LOC) | P1 (extends the same template) |
| **P6** | Triton attention prefill | mobydick | matches mobydick (6700 t/s TinyLlama prefill) | L (build infra) | should be **deferred** until P2 measured |

## Suggested implementation order

1. **P3 first** — pre-allocated workspace pool. Smallest, lowest-risk, frees up GPU sys-CPU time before any other measurement. Also a prerequisite for HIP graphs later.
2. **P0** — fused GDN kernel. Biggest single win, independent, doesn't block anything.
3. **P1** — finish stream-K MMQ. Already in the active plan; this is the natural continuation of Phase 2.
4. **P2** — custom attention kernel. After P0+P1 land, P2's gain becomes the dominant remaining bottleneck.
5. **P4** — fused norm/silu/quant. Quick win, can be done in parallel with P2.
6. **P5** — K-quant MMQ. Mechanical extension of P1.
7. **P6** — defer or skip. Re-measure after P2 and decide.

## Projected end-state

If P0+P1+P2+P3+P4 all land at expected payoffs:

| Metric | Today (Phase 2b/2d) | After P0..P4 | Turbo today |
|---|---:|---:|---:|
| qwen35-9B pp512 | 78 t/s | ~350 t/s | 450 t/s |
| qwen35-9B tg128 | 35 t/s | ~75 t/s | 58 t/s ← we'd exceed turbo |
| TinyLlama pp512 | 384 t/s | ~2500 t/s | 2687 t/s |
| TinyLlama tg128 | 88 t/s | ~250 t/s | 190 t/s ← we'd exceed turbo |
| GPU kernel time (qwen35-9B run) | 17.4 s | ~3.5 s | 2.7 s |
| Kernel dispatches (qwen35-9B run) | 1.35M | ~150k | 127k |
| sys-CPU time (qwen35-9B run) | 94 s | ~12 s | 12 s |

The decode-side projected wins reflect the fact that **turbo doesn't have anything fundamentally faster than what we're proposing** — it has a fused GDN kernel, fused attention, fused norm+quant, and a stream-K MMQ. If candle ports all four, we hit feature parity with turbo on the kernel set. The remaining gap (~30 % on prefill) is because turbo also has gfx906-tuned tile sizes, async prefetch, and DPP-based reductions that we'd need to add via Phase 3 of the macro roadmap.

## What to NOT port

- **vLLM's Python EngineCore loop**: it's the source of vllm-mobydick's slow decode (35 t/s on tinyllama vs turbo's 190). Candle's Rust runtime is already strictly better than this for single-stream inference. Don't replicate it.
- **vLLM's paged KV cache**: useful for multi-tenant serving with disparate sequence lengths, irrelevant for candle's single-stream use case. Pre-allocated KV cache (already in candle's `KvCache`) is the right trade.
- **Turbo's `-mmp 0` workaround**: turbo needs `--mmap 0` because mmap+SVM page migration on gfx906 stalls for large GGUFs. Candle uses direct host→device copy at load time and doesn't have this problem.
- **Turbo's TurboQuant 3.5-bit KV (Phase 5.3 in macro roadmap)**: niche, complex (FWHT butterfly, `-O1` build flag bug), and the model files don't exist for our targets. Skip until someone actually has a turbo3 model.
- **Custom LLVM/clang fork**: mobydick needs `triton-gfx906` because of an LLIR pass bug in upstream clang for AMD targets. Candle compiles HIP via system hipcc and doesn't hit triton, so this is a non-issue.

## Critical files referenced

| Need | File path |
|---|---|
| GDN sequential code today | `candle-transformers/src/models/quantized_blocks/delta_net.rs::delta_net_step_vectorized` |
| GDN test oracle | `quantized_blocks/delta_net.rs::tests::test_delta_net_vectorized_matches_scalar_reference` |
| Phase 2 MMQ kernel | `candle-hip-kernels/src/quantized.cu` (search for `mul_mat_q4_0_gfx906_v2`, `mul_mat_q4_1_gfx906_v2`, `mul_mat_q8_0_gfx906_v2`) |
| MMQ Rust dispatch | `candle-core/src/quantized/hip.rs::dequantize_matmul` (line ~1160) and `mul_mat_q_v2` |
| Phase 1 GQA reshape attention | `candle-transformers/src/models/quantized_blocks/attention.rs::gqa_attention` |
| HIP backend per-op alloc | `candle-core/src/hip_backend/mod.rs` (search for `alloc_uninit`, `HipSlice::drop`) |
| HIP device | `candle-core/src/hip_backend/device.rs::HipDevice` |
| Pipeline activation transfers | `candle-transformers/src/models/quantized_blocks/` model assemblers (search for `to_device(Cpu)`) |
| Turbo source: GDN kernel | `llamacpp-turbo/ggml/src/ggml-cuda/` (find `gated_delta_net*`) |
| Turbo source: stream-K MMQ | `llamacpp-turbo/ggml/src/ggml-cuda/mmq.cuh` (4273 lines) + `mmq.cu` (489 lines) |
| Turbo source: attention | `llamacpp-turbo/ggml/src/ggml-cuda/fattn-*.{cu,cuh}` |
| Turbo source: gfx906 helpers | `llamacpp-turbo/ggml/src/ggml-cuda/gfx906/{matmul,attention,fused}/` |
| Mobydick source: triton MLA | `vllm-gfx906-mobydick/vllm/attention/backends/triton_attn.py` (and the fwd_prefill config it logs) |
| Mobydick install ref | `BENCH-3WAY-2026-04-11.md` § "Install path that worked" |

## Verification scripts to add

After each P-item lands, add a regression entry to `scripts/test-suite.sh`:

| Item | Test command |
|---|---|
| P0 GDN fused | `cargo test -p candle-transformers --lib quantized_blocks::delta_net::tests::` (covers correctness vs scalar reference) + bench: `./target/release/examples/quantized-qwen35 --model Qwen3.5-9B-Q4_1.gguf --prompt "$LONG" -n 128` (decode rate before/after) |
| P1 stream-K MMQ | extend `hip_mmq_v2_matches_chunked` to cover stream-K threshold; bench: gemma4-E4B Q4_0 long-prompt prefill |
| P2 custom attention | new `hip_flash_attn_matches_gqa_oracle` test using the existing Phase 1 oracle from `attention.rs::tests` |
| P3 workspace pool | new `hip_workspace_pool_reuses_buffers` test that asserts free→alloc cycle hits the cache (no `hipMallocAsync` system call) |
| P4 fused norm+quant | unit test: fused output matches separate `rmsnorm_f32` + `quantize_q8_1` outputs to within Q8_1 quantization precision |
| P5 K-quant MMQ | extend `hip_mmq_v2_matches_chunked` to cover Q2K..Q6K |

Each P-item should also update `~/.claude/projects/-artefact/memory/project_perf_vs_turbo.md` with the new measured pp/tg numbers so future sessions see the gap close.
