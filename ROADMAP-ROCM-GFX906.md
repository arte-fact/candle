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

## Phase 1 — Quantized Inference (GGUF on GPU)

**Goal**: Run quantized GGUF models end-to-end on MI50. This is the fastest
path to real-world usability — the quantized kernels from Phase 0 (`quantized.cu`)
are already compiled, they just need to be wired into Candle's `QStorage` system.

| # | Task | Key files |
|---|------|-----------|
| 1.1 | Create `candle-core/src/quantized/hip.rs` mirroring `quantized/cuda.rs` (~1000 lines). Implement `QHipStorage` with `dequantize`, `matmul`, `fwd`, `dtype`, `device`, `to_device` methods. | `quantized/cuda.rs` → `quantized/hip.rs` |
| 1.2 | Add `QStorage::Hip(QHipStorage)` variant. Wire `Device::Hip` arms in `qzeros()` and `from_data()`. | `quantized/mod.rs` |
| 1.3 | Implement `load_quantized<T>()` for HIP — upload raw quantized blocks to GPU via `clone_htod`. | `quantized/hip.rs` |
| 1.4 | Implement quantized matmul dispatch — call `dequantize_mul_mat_vec` and `mul_mat_vec_q` kernels from `quantized.cu` HSACO. | `quantized/hip.rs` |
| 1.5 | Test with `quantized-qwen3` example on Qwen3.5-9B-Q4_1.gguf. | `scripts/test-hip-quantized.sh` |

**Exit criteria**: `quantized-qwen3` generates tokens from a GGUF model on MI50.

---

## Phase 2 — Multi-GPU & Layer-Split Parallelism

**Goal**: Run large models across 4×MI50 (64GB total). Use **layer-split**
parallelism (not tensor-parallel) following llamacpp-turbo's default approach.
Each GPU gets N/4 layers, activations flow GPU→CPU→GPU between stages.

> **RCCL status**: RCCL 2.27.7 AllReduce segfaults on gfx906 PCIe — confirmed
> in pure C. RCCL bindings are implemented in hipdarc but unusable until a
> working RCCL version is found or ROCm fixes the issue. Layer-split avoids
> RCCL entirely.

| # | Task | Reference |
|---|------|-----------|
| 2.1 | **DONE** — RCCL FFI bindings in hipdarc (group API, CommInitRank, AllReduce). Blocked by RCCL segfault. | `hipdarc/src/rccl.rs` |
| 2.2 | **DONE** — Multi-device support: create tensors, matmul, transfer via CPU on all 4 GPUs. | `hip_multi_gpu_test` |
| 2.3 | **Layer-split model runner**: Assign layers to GPUs by index range. GPU 0 gets layers 0..N/4, GPU 1 gets N/4..N/2, etc. Forward pass moves activations between stages via CPU. | llamacpp-turbo `LLAMA_SPLIT_MODE_LAYER` |
| 2.4 | **Sharded GGUF loader**: Each GPU loads only its layer range from the GGUF file. Parse metadata once, seek to tensor offsets per device. Avoid OOM from loading full model. | — |
| 2.5 | **Pipeline driver**: Orchestrate forward pass across devices. Token embeddings on GPU 0, final norm + output on last GPU, intermediate activations transferred via `to_device(Cpu).to_device(next_gpu)`. | — |
| 2.6 | **Test with Devstral 24B**: 14GB model split across 2-3 MI50s. Verify correct output. | — |

**Exit criteria**: Devstral-24B-Q4 runs across 2+ MI50s with correct output.

> **Future**: When RCCL is fixed (newer ROCm or different RCCL build), tensor
> parallelism with AllReduce can be enabled using the existing hipdarc::rccl
> bindings. This would give better throughput than layer-split for latency-bound
> generation.

---

## Phase 3 — GFX906 Kernel Optimizations & Tradeoff Resolution

**Goal**: Maximum performance. Replace baseline kernels with tuned GFX906
variants from llamacpp-turbo. Resolve all Phase 0/1 tradeoffs.

| # | Task | Source |
|---|------|--------|
| 3.1 | **DPP warp primitives**: Port `gfx906-common.cuh` — DPP shuffles, `__builtin_amdgcn_readfirstlane`, GCN ISA intrinsics. | `gfx906-common.cuh` |
| 3.2 | **Wave64 tuned reductions**: 6-stage DPP unrolled reduction for reduce.cu. | `gfx906/quantize/vecdotq.cuh` |
| 3.3 | **Fast-math unary**: GCN ISA intrinsics (`v_exp_f32`, `v_log_f32`, `v_rcp_f32`). | `gfx906-common.cuh` |
| 3.4 | **MMVQ warp-cooperative kernels**: Port half-warp MVMs for Q4_0, Q4_1, Q8_0. | `gfx906/matmul/mmvq-q4_0.cuh` |
| 3.5 | **VecDotQ with DPP**: DPP-based dot-product accumulation. | `gfx906/quantize/vecdotq.cuh` |
| 3.6 | **MMQ tile tuning**: Profile per-quant-type tile sizes for gfx906 64KB LDS. | `gfx906-config.h` |
| 3.7 | **Resolve CPU RNG**: Fix hiprand/rocrand segfault or implement GPU-side RNG. | Phase 0 tradeoff |
| 3.8 | **BF16 strategy**: Profile bf16 emulation overhead, decide f16 vs bf16 for activations. | Phase 0 tradeoff |
| 3.9 | **rocBLAS GEMM tuning**: Custom SGEMM for small batches, `ROCBLAS_GEMM_FLAGS` override. | `gfx906/matmul/mmf-sgemm.cuh` |
| 3.10 | **HIP graphs**: Capture repeated kernel sequences for 8-10% throughput gain. | llamacpp-turbo `GGML_HIP_GRAPHS` |

**Exit criteria**: Measurable throughput improvement over Phase 0/1. Quantized tg64 within 80% of llamacpp-turbo on same model.

---

## Phase 4 — Flash Attention for GCN

**Goal**: Performant attention for long-context inference on GFX906.

| # | Task | Source |
|---|------|--------|
| 4.1 | Create `candle-flash-attn-hip` crate. Custom tile-based attention (no CUTLASS). | — |
| 4.2 | **Q8 tile attention kernels**: Port from llamacpp-turbo for head sizes 64, 96, 128, 256. | `gfx906/attention/` |
| 4.3 | **RoPE kernel**: Custom `__sincosf()`-based RoPE with YaRN support. | `gfx906/attention/rope.cu` |
| 4.4 | **GQA + causal mask**: Grouped query attention dispatch. | `candle-nn/src/ops.rs` |

**Exit criteria**: 32K+ context on 16GB MI50, within 2× of CUDA FA perf.

---

## Phase 5 — Fused Operations & Advanced Optimizations

**Goal**: Close remaining perf gap with production-grade fused kernels.

| # | Task | Source |
|---|------|--------|
| 5.1 | **Fused RMS-norm + quantize**: Eliminate intermediate global memory round-trip. | `gfx906/fused/norm-fused-q8.cu` |
| 5.2 | **Fused MoE dispatch**: Reduce kernel launch count for MoE models. | `candle-kernels/src/moe/` |
| 5.3 | **TurboQuant (3.5-bit KV)**: WHT rotation + turbo3 dequant. | llamacpp-turbo turbo3 system |
| 5.4 | **Operator fusion**: Fuse layernorm→linear, attention→output via HIP graphs. | — |
| 5.5 | **Software pipelining**: Double-buffered shared memory in MMQ kernels. | `gfx906/matmul/` |

**Exit criteria**: Token generation throughput matches or exceeds llamacpp-turbo.

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
| HIP runtime binding | `hipdarc` (dedicated safe wrapper) | cudarc assumes CUDA types too deeply; hipdarc gives full control |
| Kernel compilation | `hipcc` → HSACO (not hipify→PTX) | Native GCN codegen required for DPP intrinsics and ISA-level tuning |
| BLAS | rocBLAS (not hipBLAS) | rocBLAS is the actual implementation; hipBLAS is a thin wrapper |
| Flash attention | Custom tile kernels, not CK | CK is heavy dependency; llamacpp-turbo kernels are proven and tuned for gfx906 specifically |
| Multi-GPU | RCCL over PCIe | MI50 lacks P2P; RCCL handles topology-aware routing |
| Build flag | `-O1` for turbo-quant paths | HIP compiler FWHT butterfly misoptimization bug at -O3 (proven in llamacpp-turbo) |

---

## Estimated Complexity & Dependencies

```
Phase 0  ██████████████████████████  [Large]   — DONE ✓  HIP backend + dense ops
Phase 1  ████████████                [Medium]  — DONE ✓  Quantized inference on GPU
Phase 2  ████████████                [Medium]  — Phase 1 (multi-GPU, RCCL)
Phase 3  ████████████████            [Medium]  — Phase 1 (kernel optimizations + tradeoffs)
Phase 4  ██████████████████████████  [Large]   — Phase 1 (flash attention)
Phase 5  ████████████████████        [Med-Lrg] — Phase 3-4
Phase 6  ████████████                [Small]   — All phases (incremental)
```

Phase 2 (multi-GPU) unlocks large model inference on 4×MI50 (64GB).
Phases 3 and 4 can proceed in parallel after Phase 2.

---

## Known Tradeoffs & Technical Debt

| Item | Current state | Impact | Revisit when |
|------|--------------|--------|-------------|
| **RNG on CPU** | `rand` crate generates on host, uploads to device. hiprand/rocrand segfaults on ROCm 7.1.1. | Slower for large random tensors (extra H2D copy). Doesn't affect inference perf. | ROCm update or rocrand fix. Phase 2+ if profiling shows bottleneck. |
| **No FP8 support** | F8E4M3 gated out (`UnsupportedDtype`). GFX906 has no FP8 hardware. | No impact — FP8 is CDNA2+ only. | Never for gfx906; add for gfx90a+ if targeting MI250X. |
| **BF16 emulated** | All bf16 math goes through f32 promotion. Correct but ~3× slower than native. | Acceptable for Phase 0-1. | Phase 2 — consider keeping activations in f16 instead of bf16. |
| **Raw ELF HSACO** | `--no-gpu-bundle-output` to avoid Clang offload bundle. | Works but limits to single arch per build. | Multi-arch support: remove flag, let runtime unbundle. |
| **RCCL broken on gfx906** | RCCL 2.27.7 (ROCm 7.1.1) segfaults in ncclAllReduce on MI50 PCIe, even in pure C with iommu=pt. 2-GPU and 4-GPU both crash. | No tensor parallelism via AllReduce. Layer-split used instead. | Try RCCL from ROCm 6.x, or build RCCL from source with gfx906 fixes. |
| **PASCAL-era MMQ tiles** | All quantized MMQ kernels use 64×64 tiles (renamed to GFX906). | Correctness OK, not perf-optimal. | Phase 2 — tune tile sizes per quant type for gfx906 LDS. |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Q4_0 Llama-7B tg64 on 1×MI50 | ≥ 25 t/s |
| Q4_0 Qwen-27B tg64 on 4×MI50 | ≥ 20 t/s (matching llamacpp-turbo) |
| Flash attention context length | 32K+ on 16GB |
| Op coverage vs CUDA backend | ≥ 95% |
| CI green on gfx906 | Every PR |
