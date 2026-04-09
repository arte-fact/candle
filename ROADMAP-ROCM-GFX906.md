# Candle ROCm GFX906 Support — Macro Roadmap

> Bringing first-class AMD GFX906 (MI50/MI60) support to Candle, informed by
> the kernel-level optimizations proven in the llamacpp-turbo fork.

---

## Phase 0 — Foundation: HIP Backend Scaffolding

**Goal**: Candle compiles and runs basic tensor ops on any ROCm GPU.

| # | Task | Key files / crates |
|---|------|--------------------|
| 0.1 | Add `hip` feature flag to workspace + `candle-core` Cargo.toml. Gate behind `#[cfg(feature = "hip")]` throughout. | `Cargo.toml`, `candle-core/Cargo.toml` |
| 0.2 | Evaluate HIP bindings strategy: wrap `cudarc` with HIP shim (HIP's CUDA compat layer) **or** create a dedicated `hipdarc`-style safe wrapper. Decide based on cudarc's `cuda-version-from-build-system` feature and HIP's `hipify` path. | — |
| 0.3 | Implement `HipDevice` (BackendDevice) and `HipStorage` (BackendStorage) structs mirroring `CudaDevice`/`CudaStorage`. Start with memory alloc, D2H/H2D copies, synchronize. | `candle-core/src/hip_backend/` |
| 0.4 | Add `Device::Hip` and `Storage::Hip` variants. Wire dispatch in `storage.rs` and `device.rs`. Add `dummy_hip_backend.rs` fallback. | `storage.rs`, `device.rs` |
| 0.5 | Create `candle-hip-kernels` crate. Set up `build.rs` to compile `.cu` → HIP via `hipcc` (LLVM/Clang), producing GCN ISA or HSACO. Target `--offload-arch=gfx906`. | `candle-hip-kernels/` |
| 0.6 | Port the 11 kernel families from `candle-kernels` (affine, binary, cast, conv, fill, indexing, quantized, reduce, sort, ternary, unary) via `hipify-perl` as baseline. Validate correctness against CPU backend with existing test suite. | `candle-hip-kernels/src/*.cu` |
| 0.7 | Integrate `rocblas` for GEMM/BLAS operations (equivalent of cuBLAS). Wire into `HipDevice`. | `candle-core/src/hip_backend/device.rs` |

**Exit criteria**: `cargo test --features hip` passes on ROCm with gfx906, all dtypes, basic ops + matmul.

---

## Phase 1 — GFX906 Wave64 Kernel Optimizations

**Goal**: Replace hipified baseline kernels with tuned GFX906 variants, porting
proven optimizations from llamacpp-turbo.

| # | Task | Source (llamacpp-turbo) |
|---|------|------------------------|
| 1.1 | **DPP warp primitives library**: Port `gfx906-common.cuh` — DPP-based shuffles (`hip_add_xor{1,2,8}_f32`, `hip_shuffle_xor{4,16}_f32`), `__builtin_amdgcn_readfirstlane`, inline ASM for `v_exp_f32`, `v_log_f32`, `v_rcp_f32`. Build as shared header for all gfx906 kernels. | `gfx906-common.cuh` |
| 1.2 | **Reduction kernels**: Replace generic warp shuffles with 6-stage Wave64 DPP unrolled reduction (xor1→xor2→xor8→xor16→xor32→shift). Apply to `reduce.cu` (sum, max, min, argmax, argmin, softmax). | `gfx906/quantize/vecdotq.cuh` |
| 1.3 | **Unary fast-math**: Use GCN ISA intrinsics (`v_exp_f32`, `v_log_f32`, `v_rcp_f32`, `v_rsq_f32`) for exp, log, recip, rsqrt paths when precision allows. Keep f64 on generic path. | `gfx906-common.cuh` |
| 1.4 | **Matmul tuning**: Tune rocBLAS GEMM tile sizes. For small/medium batches (M<2048), implement custom SGEMM with 32×64×64 tiles and 2-warp register pressure sweet spot. Add gfx906-specific `ROCBLAS_GEMM_FLAGS` env override. | `gfx906/matmul/mmf-sgemm.cuh` |
| 1.5 | **Memory access patterns**: Add shared-memory padding (+4 elements) to avoid LDS bank conflicts. Enforce 64KB LDS limit (not 160KB). Use vectorized 32-bit loads where applicable. | `gfx906-config.h` |
| 1.6 | **Launch config tuning**: Create `gfx906-config.h` with tuned launch bounds — 64-thread blocks for MMVQ, 256 threads for bandwidth-bound ops, `num_warps=2` default for register pressure. | `gfx906-config.h` |

**Exit criteria**: Benchmark suite shows measurable throughput improvement over Phase 0 hipified kernels on MI50.

---

## Phase 2 — Quantization Backend

**Goal**: Full quantized inference support — Q4_0, Q4_1, Q8_0, Q8_1 — with
GFX906-optimized dequant and matmul-vec kernels.

| # | Task | Source |
|---|------|--------|
| 2.1 | Port GGML quantization type definitions and dequant routines to `candle-hip-kernels`. Candle's `quantized.cu` already has `CC_OFFSET_AMD` / `CC_RDNA*` defines — extend with `CC_GCN906`. | `candle-kernels/src/quantized.cu` |
| 2.2 | **Warp-cooperative MMVQ kernels**: Port `gfx906_mul_mat_vec_q4_0_warp_coop` — half-warp (32 threads) cooperative per-row MVMs for Q4_0, Q4_1, Q8_0. Use vectorized 32-bit loads for nibble unpacking. | `gfx906/matmul/mmvq-q4_0.cuh` |
| 2.3 | **VecDotQ with DPP**: Port dot-product accumulation using DPP reductions instead of `__shfl_xor_sync`. | `gfx906/quantize/vecdotq.cuh` |
| 2.4 | **Q8 activation cache**: Implement fused RMS-norm + Q8_1 quantize kernel for KV cache compression. Shared-memory tile caching with adaptive loads. | `gfx906/fused/norm-fused-q8.cu` |
| 2.5 | Wire quantized types into Candle's `GgmlDType` enum and `QStorage` trait. Ensure `candle-transformers` quantized model loaders work with HIP backend. | `candle-core/src/quantized/` |

**Exit criteria**: Quantized Llama/Mistral inference runs end-to-end on MI50 with Q4_0 GGUF weights.

---

## Phase 3 — Flash Attention for GCN

**Goal**: Performant attention for long-context inference on GFX906.

| # | Task | Source |
|---|------|--------|
| 3.1 | Create `candle-flash-attn-hip` crate. Cannot use CUTLASS (NVIDIA-only) — implement tile-based attention from scratch or adapt Composable Kernel (CK) library from AMD. | — |
| 3.2 | **Q8 tile attention kernels**: Port tile-based FA kernels for head sizes 64, 96, 128, 256 from llamacpp-turbo. GCN-specific tile dimensions, 64KB LDS limit, Wave64 reductions. | `gfx906/attention/` |
| 3.3 | **RoPE kernel**: Port custom `__sincosf()`-based RoPE with YaRN extended context and multi-RoPE section support. 256 threads/block for bandwidth. | `gfx906/attention/rope.cu` |
| 3.4 | **Attention dispatch**: Integrate into Candle's `ScaledDotProductAttention` custom op. GQA support (gqa2, gqa4, gqa8). | `candle-nn/src/ops.rs` |
| 3.5 | **Causal mask + ALiBi**: Support causal masking and ALiBi position bias within the fused kernel. | — |

**Exit criteria**: Flash attention benchmark within 2× of CUDA FA on equivalent compute, supports 32K+ context on 16GB MI50.

---

## Phase 4 — Multi-GPU & Tensor Parallelism

**Goal**: Scale across multiple MI50s (PCIe, no P2P).

| # | Task | Notes |
|---|------|-------|
| 4.1 | **RCCL integration**: Add `rccl` feature flag. Implement AllReduce, AllGather, ReduceScatter over PCIe (no P2P on MI50). | Analogous to `nccl` feature |
| 4.2 | **Pipeline parallelism**: Layer-wise distribution across GPUs. No row-split (requires P2P). | llamacpp-turbo uses `--split-mode row` over pipeline |
| 4.3 | **KV cache sharding**: Distribute KV cache across GPU memories for extended context. | — |
| 4.4 | **HIP graphs**: Implement graph capture for repeated kernel sequences (+8-10% throughput from llamacpp-turbo measurements). | llamacpp-turbo `GGML_HIP_GRAPHS` |

**Exit criteria**: 4×MI50 tensor-parallel inference with linear-ish scaling on memory-bound models.

---

## Phase 5 — Fused Operations & Advanced Optimizations

**Goal**: Close remaining perf gap with production-grade fused kernels.

| # | Task | Source |
|---|------|--------|
| 5.1 | **Fused RMS-norm + quantize**: Single kernel for norm→quantize path, eliminating intermediate global memory round-trip. | `gfx906/fused/norm-fused-q8.cu` |
| 5.2 | **Fused MoE dispatch**: Reduce kernel launch count for MoE models — port `moe_gguf.cu` patterns with GFX906 tuning. | `candle-kernels/src/moe/` |
| 5.3 | **TurboQuant (3.5-bit KV cache)**: Port WHT rotation + turbo3 dequant for extreme KV compression (3.3× more context than f16). | llamacpp-turbo turbo3 system |
| 5.4 | **Operator fusion pass**: Identify and fuse common subgraphs (layernorm→linear, attention→output projection) using HIP graph capture. | — |
| 5.5 | **Software pipelining**: Overlap memory loads with compute in MMQ kernels. Double-buffered shared memory. | `gfx906/matmul/` |

**Exit criteria**: Token generation throughput matches or exceeds llamacpp-turbo fork numbers on equivalent models.

---

## Phase 6 — CI, Testing & Ecosystem

**Goal**: Production-ready, maintainable ROCm support.

| # | Task |
|---|------|
| 6.1 | CI pipeline with ROCm container (rocm/dev-ubuntu-22.04). Test matrix: gfx906, gfx908, gfx90a. |
| 6.2 | Benchmark suite: pp512/tg64 throughput, memory bandwidth utilization, kernel roofline analysis. |
| 6.3 | `candle-examples` ROCm variants — quantized LLM inference, vision models, MoE. |
| 6.4 | Documentation: ROCm setup guide, GFX906 tuning knobs, multi-GPU configuration. |
| 6.5 | Feature parity checklist vs CUDA backend. Track gaps in issue tracker. |

---

## Architecture Diagram

```
candle (workspace)
├── candle-core
│   ├── src/hip_backend/          ← NEW: HipDevice, HipStorage, BackendStorage impl
│   │   ├── mod.rs                   (mirrors cuda_backend/mod.rs)
│   │   ├── device.rs                (HipDevice + rocBLAS handle)
│   │   └── utils.rs                 (Map1/Map2/Map3 for HIP dispatch)
│   ├── src/dummy_hip_backend.rs  ← NEW: stubs when hip feature disabled
│   ├── src/storage.rs               (add Storage::Hip variant)
│   └── src/device.rs                (add Device::Hip variant)
│
├── candle-hip-kernels/           ← NEW CRATE
│   ├── build.rs                     (hipcc compilation → HSACO/GCN ISA)
│   ├── src/lib.rs                   (kernel module registry)
│   ├── src/ptx.rs                   (embedded kernel binary loader)
│   ├── src/ffi.rs                   (FFI bindings for compiled kernels)
│   ├── src/*.cu                     (hipified baseline kernels)
│   └── src/gfx906/                  (Wave64-optimized kernels)
│       ├── common.cuh               (DPP primitives, fast math)
│       ├── config.h                 (launch bounds, LDS limits)
│       ├── matmul/                  (MMVQ, SGEMM)
│       ├── attention/               (tile FA, RoPE)
│       ├── fused/                   (norm+quant, MoE)
│       └── quantize/                (Q4/Q8 dequant, vecdotq)
│
├── candle-flash-attn-hip/        ← NEW CRATE
│   ├── build.rs                     (CK or custom build)
│   └── src/                         (GCN flash attention)
│
└── candle-transformers              (model code — mostly unchanged,
                                      device-agnostic via trait dispatch)
```

---

## Key Technical Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| HIP runtime binding | `cudarc` + HIP compat layer first, `hipdarc` later | Fastest path to working backend; HIP's CUDA compat covers ~95% of cudarc API surface |
| Kernel compilation | `hipcc` → HSACO (not hipify→PTX) | Native GCN codegen required for DPP intrinsics and ISA-level tuning |
| BLAS | rocBLAS (not hipBLAS) | rocBLAS is the actual implementation; hipBLAS is a thin wrapper |
| Flash attention | Custom tile kernels, not CK | CK is heavy dependency; llamacpp-turbo kernels are proven and tuned for gfx906 specifically |
| Multi-GPU | RCCL over PCIe | MI50 lacks P2P; RCCL handles topology-aware routing |
| Build flag | `-O1` for turbo-quant paths | HIP compiler FWHT butterfly misoptimization bug at -O3 (proven in llamacpp-turbo) |

---

## Estimated Complexity & Dependencies

```
Phase 0  ██████████████████████████  [Large]   — ROCm 7.1+, hipcc, rocBLAS
Phase 1  ████████████████            [Medium]  — Phase 0
Phase 2  ████████████████            [Medium]  — Phase 0 + 1
Phase 3  ██████████████████████████  [Large]   — Phase 0 + 1
Phase 4  ████████████████            [Medium]  — Phase 0 + Phase 3
Phase 5  ████████████████████        [Med-Lrg] — Phase 1-3
Phase 6  ████████████                [Small]   — All phases (incremental)
```

Phases 1, 2, and 3 can be developed in parallel after Phase 0 completes.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Q4_0 Llama-7B tg64 on 1×MI50 | ≥ 25 t/s |
| Q4_0 Qwen-27B tg64 on 4×MI50 | ≥ 20 t/s (matching llamacpp-turbo) |
| Flash attention context length | 32K+ on 16GB |
| Op coverage vs CUDA backend | ≥ 95% |
| CI green on gfx906 | Every PR |
