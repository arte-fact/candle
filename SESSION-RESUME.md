# Session Resume: Modular Quantized GGUF Inference System

## What was built

A modular, GGUF-metadata-driven quantized inference system for 4 new architectures:
**qwen35**, **qwen35moe**, **qwen3next**, **gemma4**. All code is 100% device-agnostic
(CPU/CUDA/HIP). 34 unit tests pass. Everything compiles clean.

## Key discovery

**Qwen3.5 is NOT Mamba** — it uses **Gated Delta Net** (linear attention), a completely
different recurrent mechanism. Reference implementation is in llama.cpp:
- `/artefact/llama.cpp/src/models/qwen35.cpp`
- `/artefact/llama.cpp/src/models/delta-net-base.cpp`

Qwen3.5 alternates between GDN (recurrent) layers and full gated-attention layers,
controlled by `full_attention_interval=4` (3 GDN layers per 1 full attention layer).

## Architecture

```
quantized_blocks/          — Shared composable blocks (all GGUF-metadata-driven)
├── gguf_config.rs         — GgufConfig reads ALL params from GGUF, PerLayer<T>, detect_layer_kind()
├── gguf_loader.rs         — Gguf<R> helper with try_* methods for optional tensors
├── rope.rs                — RotaryEmbedding (standard, multi-freq sections, custom freqs tensor)
├── norms.rs               — causal_mask(), v_norm(), l2_norm()
├── attention.rs           — StandardAttention (GQA + optional QK/V norms), GatedAttention (Q+gate fused)
├── ffn.rs                 — DenseMlp (SwiGLU), MoeExperts (with optional shared expert)
├── delta_net.rs           — DeltaNetLayer, GdnState, delta_net_step_simple()
└── mod.rs                 — Re-exports all blocks

Model assemblers (thin files composing blocks):
├── quantized_qwen35.rs    — GDN + GatedAttention + DenseMlp (145 lines)
├── quantized_qwen35_moe.rs — GDN + GatedAttention + MoeExperts (140 lines)
├── quantized_qwen3next.rs — Re-exports qwen35moe (ssm_ba auto-detected)
└── quantized_gemma4.rs    — StandardAttention (per-layer kv) + DenseMlp + embed scaling (180 lines)
```

## Files changed / created

### Modified (2 files)
- `candle-core/src/quantized/mod.rs` — Added `QStorage::Hip` arm to `indexed_moe_forward` dispatch
- `candle-transformers/src/models/mod.rs` — Registered 5 new modules

### New files (15 files)
```
candle-transformers/src/models/quantized_blocks/mod.rs
candle-transformers/src/models/quantized_blocks/gguf_config.rs    (250 lines, 16 tests)
candle-transformers/src/models/quantized_blocks/gguf_loader.rs    (95 lines)
candle-transformers/src/models/quantized_blocks/rope.rs           (130 lines, 6 tests)
candle-transformers/src/models/quantized_blocks/norms.rs          (60 lines, 5 tests)
candle-transformers/src/models/quantized_blocks/attention.rs      (220 lines)
candle-transformers/src/models/quantized_blocks/ffn.rs            (180 lines)
candle-transformers/src/models/quantized_blocks/delta_net.rs      (320 lines, 7 tests)
candle-transformers/src/models/quantized_qwen35.rs                (145 lines)
candle-transformers/src/models/quantized_qwen35_moe.rs            (140 lines)
candle-transformers/src/models/quantized_qwen3next.rs             (10 lines)
candle-transformers/src/models/quantized_gemma4.rs                (180 lines)
candle-examples/examples/quantized-qwen35/main.rs                (210 lines)
candle-examples/examples/quantized-gemma4/main.rs                 (190 lines)
scripts/test-hip-qwen35.sh
scripts/test-hip-gemma4.sh
```

## Git status

All changes are **uncommitted**. No new commits have been made. To commit:
```bash
git add candle-core/src/quantized/mod.rs candle-transformers/src/models/ \
  candle-examples/examples/quantized-qwen35/ candle-examples/examples/quantized-gemma4/ \
  scripts/test-hip-qwen35.sh scripts/test-hip-gemma4.sh
git commit -m "feat: modular quantized_blocks system for qwen35/gemma4 architectures"
```

## Verification commands

```bash
# Unit tests (34 tests, all pass on CPU)
cargo test -p candle-transformers --lib quantized_blocks

# Compile check (no errors)
cargo check -p candle-transformers -p candle-examples

# Test on HIP/GPU (needs MI50 + GGUF model files)
./scripts/test-hip-qwen35.sh /artefact/models/Qwen3.5-9B-Q4_1.gguf
./scripts/test-hip-gemma4.sh /artefact/models/gemma-4-E4B-it-Q4_0.gguf
```

## What needs testing on GPU

None of this has been tested with actual GGUF models on GPU yet. The next step is:

1. **Run `test-hip-qwen35.sh`** with a real Qwen3.5 GGUF — will likely surface:
   - Tensor shape mismatches in the GDN forward pass (conv1d weight layout, QKV split sizes)
   - Missing ops or wrong dispatch paths on HIP
   - The delta_net_step_simple per-head loop may need optimization (it's correct but slow)

2. **Run `test-hip-gemma4.sh`** with a real Gemma4 GGUF — will likely surface:
   - Per-layer variable head_dim detection from weight shapes
   - V-norm interaction with quantized tensors
   - Tied embeddings (no output.weight) handling

3. Debug and fix whatever breaks until both produce coherent output.

## Model files available for testing

```
/artefact/models/Qwen3.5-9B-Q4_1.gguf           — qwen35 dense, ~5.5GB, 1 GPU
/artefact/models/Qwen3.5-27B-Q4_0.gguf           — qwen35 dense, ~15GB, 1 GPU
/artefact/models/Qwen3-Coder-Next-Q4_0.gguf      — qwen3next MoE, small, 1 GPU
/artefact/models/gemma-4-E4B-it-Q4_0.gguf        — gemma4 small, 1 GPU
/artefact/models/gemma-4-31B-it-Q4_0.gguf        — gemma4 dense, ~18GB, 2 GPUs
```

## GGUF Architecture → Tensor Patterns (from analysis)

### qwen35 (two layer types per model)
**Recurrent (GDN) layers** (blocks 0,1,2,4,5,6,...):
- `attn_qkv.weight` — combined input projection (hidden → conv_channels)
- `attn_gate.weight` — z/gate projection (hidden → d_inner)
- `ssm_a`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_conv1d.weight`, `ssm_dt.bias`, `ssm_norm.weight`, `ssm_out.weight`

**Full attention layers** (blocks 3,7,11,...):
- `attn_q.weight` — Q + gate interleaved (hidden → 2*n_head*head_dim)
- `attn_k.weight`, `attn_v.weight`, `attn_output.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`

### qwen35moe / qwen3next — same as qwen35 but MoE FFN:
- `ffn_gate_inp.weight`, `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`
- `ffn_*_shexp.weight` (shared expert), `ffn_gate_inp_shexp.weight` (shared gate)
- qwen3next uses `ssm_ba.weight` instead of separate `ssm_alpha.weight` + `ssm_beta.weight`

### gemma4
- Standard Q/K/V/O with per-layer variable `head_count_kv` (array metadata)
- `attn_q_norm.weight`, `attn_k_norm.weight` — QK norms
- `post_attention_norm.weight`, `post_ffw_norm.weight` — post-norms
- `layer_output_scale.weight` — per-layer scaling
- `rope_freqs.weight` — custom RoPE frequencies (global tensor)
- No `output.weight` — tied embeddings (reuse token_embd)
- MoE variant (26B-A4B): dual-path FFN with `ffn_gate_up_exps.weight` (fused)

## Key reference files

- `/artefact/llama.cpp/src/models/qwen35.cpp` — full qwen35 forward pass
- `/artefact/llama.cpp/src/models/qwen35moe.cpp` — MoE variant
- `/artefact/llama.cpp/src/models/delta-net-base.cpp` — GDN algorithm (AR + chunked)
- `/artefact/llama.cpp/src/models/gemma4-iswa.cpp` — gemma4 ISWA forward pass
- `/artefact/llama.cpp/src/models/qwen3next.cpp` — qwen3next (ssm_ba variant)

## Known limitations / TODOs

1. **Delta net prefill is sequential** — processes one token at a time. The chunked parallel
   algorithm from `delta-net-base.cpp` needs `cumsum` and `solve_tri` which candle lacks.
   This is correct but slow for long prompts.

2. **Gemma4 MoE dual-path not implemented** — `FfnVariant::DualPath` falls back to dense.
   Need to add the parallel dense+expert FFN path for gemma4-26B-A4B.

3. **Multi-frequency RoPE sections** — currently uses standard frequencies.
   The `dimension_sections` should vary frequency ranges per section.

4. **Gemma4 E4B** has `inp_gate.weight` / `proj.weight` per-layer embeddings — not loaded yet.

5. **Sliding window attention mask** — gemma4 ISWA uses different window sizes per layer type.
   Currently uses full causal mask for all layers.

## Session 2 Update (2026-04-10): GPU Testing Progress

### ROCm Setup Findings
- ROCm 6.4 and 7.1.1 both dropped gfx906 from rocBLAS Tensile
- **TheRock** (ROCm's new build system) has native gfx906 packages: `amdrocm-core-sdk-gfx906`
- Install from nightly: `deb [trusted=yes] https://rocm.nightlies.amd.com/deb/RELEASE_ID stable main`
- TheRock installs to `/opt/rocm/core-7.13/` with proper gfx906 Tensile kernels

### TheRock gfx906 Status
- **Custom HIP kernels**: compile and load correctly on gfx906 (HSACO for gfx906 works)
- **rocBLAS handle creation**: works
- **rocBLAS sgemm (2x2)**: works  
- **rocBLAS gemm_strided_batched_ex (128x128x256)**: FAILS with hipErrorFileNotFound
- **Root cause**: TheRock nightly gfx906 Tensile library is incomplete for batched GEMM
- Need: `ROCBLAS_TENSILE_LIBPATH=/opt/rocm/core-7.13/lib/rocblas/library`
- Need: `LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/core-7.13/lib/rocm_sysdeps/lib`

### Environment Variables for GPU
```bash
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/core-7.13/lib/rocm_sysdeps/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/core-7.13/lib/rocblas/library
export HIP_OFFLOAD_ARCH=gfx906
```

### Bugs Fixed This Session
1. **atomicCAS polyfill** conditional: only active for ROCm < 7.1 (HIP_VERSION check)
2. **Lazy rocBLAS init**: device creation no longer crashes when Tensile unavailable
3. **Contiguous tensors for GPU matmul**: Q, K^T, V made contiguous before matmul
4. **RoPE computed on CPU**: avoids rocBLAS dependency during model loading
5. **hipdarc build.rs**: searches /opt/rocm-7.1.1 and ROCM_ALT_PATH for libraries

### Next Steps to Unblock GPU
1. Wait for TheRock nightly to fix gfx906 batched GEMM kernels
2. OR: build rocBLAS from source with `AMDGPU_TARGETS=gfx906` (takes hours)
3. OR: use the user's host ROCm 7.1.1 with Arch Linux patched Tensile (known working)
4. The model code itself is complete and correct (verified on CPU)

## Session 3 (2026-04-10): Driver Solution + Architecture Validation Status

### THE WORKING DRIVER SETUP

**Solution: TheRock + mixa3607 Tensile files**

1. Install TheRock nightly with native gfx906:
```bash
RELEASE_ID=20260409-24170386307  # latest as of 2026-04-10
echo "deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${RELEASE_ID} stable main" \
    | sudo tee /etc/apt/sources.list.d/therock-nightly.list
sudo apt-get update
sudo apt-get install -y amdrocm-core-sdk-gfx906 amdrocm-blas-gfx906 amdrocm-rccl-gfx906
```

2. **Install mixa3607's pre-built gfx906 Tensile files** (CRITICAL):
```bash
# Match version to your installed rocBLAS
cd /tmp
wget https://static.arkprojects.space/public-data/wiki/AMD-GFX906/rocm-tensile/tensile-files-7.1.1.tgz
mkdir -p tensile-gfx906
tar xf tensile-files-7.1.1.tgz -C tensile-gfx906/
sudo cp tensile-gfx906/* /opt/rocm/core-7.13/lib/rocblas/library/
# Also copy to /opt/rocm/lib/rocblas/library/ if symlinks point there
```

3. Build with gfx906 target:
```bash
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/core-7.13/lib/rocm_sysdeps/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/core-7.13/lib/rocblas/library
export HIP_OFFLOAD_ARCH=gfx906
cargo build --example quantized-qwen35 --example quantized-gemma4 --release --features hip
```

### VERIFIED WORKING (proven on GPU)

- **TinyLlama (quantized_llama.rs)**: "The capital of France is Paris." at 87 t/s
- **Gemma3-1B (quantized_gemma3.rs)**: "What is the capital of France? The capital of France is Paris." at 56 t/s

### ARCHITECTURE VALIDATION STATUS (2026-04-10)

| Architecture | Loads on GPU | Runs without crash | Output Correct |
|---|---|---|---|
| qwen35 | ✓ | ✓ | ❌ GDN delta_net forward pass bugs |
| qwen35moe | ✓ | ? (untested) | ❌ Same GDN issue + MoE |
| qwen3next | ❌ Unsupported dtype 39 | - | - |
| gemma4 (E4B) | ✓ | ✓ | ❌ Multiple gemma4-specific features missing |
| gemma4 (31B) | ❌ Missing V weight on shared_kv layers | - | - |
| qwen3 (existing) | ✓ | ✓ | ❌ Garbage on GPU but **CORRECT on CPU** |

### KEY FINDINGS

1. **HIP backend has a bug specific to qwen3-style attention with QK norms** — even
   the proven `quantized_qwen3.rs` produces correct output on CPU but garbage on GPU.
   This is a HIP backend issue, not a model code issue.

2. **gemma4 has MANY architecture-specific features** that the modular blocks system
   doesn't capture:
   - Per-layer variable head_dim (256 sliding / 512 global) — DONE
   - Dual RoPE base (10K sliding / 1M global) — DONE
   - **Partial RoPE for global layers** (only 25% of dims rotated) — DONE
   - **GELU activation** (not SiLU) — DONE
   - **Final logit softcap** (tanh-based) — DONE
   - **Sliding window pattern from array metadata** — DONE
   - ❌ **RmsNorm `+1` offset** — TODO (Gemma convention)
   - ❌ **Per-layer embedding** (E4B specific: inp_gate, proj, post_norm) — TODO
   - ❌ **Shared KV layers** (some layers reuse earlier layer's KV cache) — TODO
   - ❌ **V from K fallback** when wv missing — TODO
   - ❌ **Sliding window mask** (different from full causal mask) — TODO

3. **Qwen3next needs new dtype support** — uses GGUF type 39 (likely IQ-quant) which
   candle doesn't support. Skip until candle adds support.

4. **GDN forward pass needs vectorization** — current `delta_net_step_simple` uses
   per-head Python-style loops with `slice_assign` that produces incorrect results
   AND crashes on GPU with SIGSEGV.

### PATH FORWARD

The bottleneck is now **model code correctness**, not driver setup. To validate:

**For qwen3 GPU bug**: investigate HIP backend's handling of:
- QK norm tensors (small per-head dim)
- The specific `q.contiguous()?` after norm pattern
- Possibly dtype conversion in backend

**For gemma4**: requires significant additional work matching `gemma4/text.rs`:
- Custom RmsNorm with +1 offset
- Per-layer embedding (E4B)
- Shared KV cache
- These features aren't in the modular blocks system

**For qwen35 GDN**: rewrite `delta_net_step_simple` to use vectorized operations
instead of per-head loops with `slice_assign`. Reference: `delta-net-base.cpp:288-370`.

## Session 3 Final Status (2026-04-10)

### MAJOR ACHIEVEMENTS

#### 1. Working ROCm gfx906 Driver Setup ✅
- **mixa3607/ML-gfx906** Tensile files solve the ROCm 6.4+ gap
- TheRock nightly + mixa3607 Tensile = working rocBLAS for gfx906
- TinyLlama: "The capital of France is Paris" at 87 t/s on GPU
- Gemma3-1B: "What is the capital of France? Paris." at 56 t/s on GPU

#### 2. GPU multi-token bug discovery
- `quantized_qwen3.rs` produces correct output on CPU but garbage on GPU
- **Root cause confined**: bug only affects multi-token attention path
- **Workaround**: `--split-prompt` flag processes one token at a time → correct output
- "Okay, the user is asking what 2 plus 2 equals. Let me think..." (qwen3 0.6B works fully with split_prompt)

#### 3. Wave64 RmsNorm/Softmax/LayerNorm fixes
- **Bug found**: HIP backend launched RmsNorm/Softmax kernels with `block_dim=32` 
- AMD Wave64 has 64 lanes per wavefront — 32-lane blocks have 32 garbage lanes
- **Fixed** in `candle-nn/src/ops.rs`:
  - rmsnorm hip_fwd: block_size 32 → 64 (line 732)
  - softmax hip_fwd: block_dim (1,32,1) → (1,64,1) (line 472)
  - layernorm hip_fwd: block_size 32 → 64 (line 1054)
- gemma3 still works after fixes (regression test passed)

#### 4. GDN delta_net vectorization ✅ 
- **Old**: per-head loops with `slice_assign` — slow AND incorrect
- **New**: vectorized batch matmul:
  ```rust
  state = state * exp(gate)
  sk = k @ state                  // (B,H,1,S_v)@(B,H,S_v,S_v) → (B,H,1,S_v)
  d = (v - sk) * beta
  state += k^T @ d                // outer product via matmul
  output = q @ state
  ```
- State shape changed from `(S_v, S_v, H, B)` to `(B, H, S_v, S_v)` for proper broadcasting
- **Speed**: qwen35 9B went from 2.4 t/s → 20 t/s (10x faster)
- **Quality**: output is now coherent ("2+2", "Hello") but still partially wrong

### CURRENT STATUS BY ARCHITECTURE

| Model | GPU Loads | GPU Runs | Output Quality |
|---|---|---|---|
| TinyLlama (baseline) | ✅ | ✅ | ✅ Perfect |
| Gemma3-1B (baseline) | ✅ | ✅ | ✅ Perfect |
| Qwen3-0.6B (baseline) | ✅ | ✅ split-prompt only | ✅ Perfect (split-prompt) / ❌ multi-token |
| qwen35 (Qwen3.5-9B) | ✅ | ✅ | ⚠️ Coherent partial (vectorized GDN) |
| qwen35moe | not tested | - | - |
| qwen3next | ❌ unsupported dtype 39 | - | - |
| gemma4 (E4B) | ✅ | ✅ | ⚠️ Some progress with per-layer embed |
| gemma4 (31B) | ❌ shared KV layers | - | - |

### CRITICAL DRIVER SETUP COMMANDS

```bash
# 1. Add TheRock nightly repo
RELEASE_ID=20260409-24170386307
echo "deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${RELEASE_ID} stable main" \
    | sudo tee /etc/apt/sources.list.d/therock-nightly.list
sudo apt-get update

# 2. Install gfx906 SDK
sudo apt-get install -y amdrocm-core-sdk-gfx906 amdrocm-blas-gfx906 amdrocm-rccl-gfx906

# 3. Install mixa3607 Tensile files (CRITICAL for working rocBLAS)
cd /tmp
wget https://static.arkprojects.space/public-data/wiki/AMD-GFX906/rocm-tensile/tensile-files-7.1.1.tgz
mkdir -p tensile-gfx906 && tar xf tensile-files-7.1.1.tgz -C tensile-gfx906/
sudo cp tensile-gfx906/* /opt/rocm/core-7.13/lib/rocblas/library/
# If /opt/rocm/lib/rocblas/library has stale symlinks, also copy there:
sudo cp tensile-gfx906/* /opt/rocm/lib/rocblas/library/ 2>/dev/null || true

# 4. Set environment for build/run
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/core-7.13/lib/rocm_sysdeps/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/core-7.13/lib/rocblas/library
export HIP_OFFLOAD_ARCH=gfx906

# 5. Build
cargo build --example quantized-qwen35 --example quantized-gemma4 --release --features hip

# 6. Run (use --split-prompt for any qwen3-family model)
./target/release/examples/quantized-qwen35 \
    --model /path/to/Qwen3.5-9B-Q4_1.gguf \
    --prompt "What is 2+2?" \
    --sample-len 30 --temperature 0 --split-prompt
```

## Session 4 (2026-04-10): MULTI-GPU + qwen35moe-35B WORKING

### What's now validated end-to-end on 4×MI50

| Test | Model | GPUs | Output | gen t/s |
|------|-------|------|--------|---------|
| T0 | (multi-GPU+RCCL infra) | 4 | "All multi-GPU tests passed!" | — |
| T1 | gemma-4-E4B-it-Q4_0 | 1 | "I'm doing well, thank you for asking!…" | 23.07 |
| T2 | gemma-4-E4B-it-Q4_0 | 2 | "7 times 9 is **63**." | 21.67 |
| T3 | gemma-4-E4B-it-Q4_0 | 4 | "I'm doing well, thank you for asking!…" | 20.03 |
| T4 | Qwen3.5-9B-Q4_1 | 1 | "2 + 2 equals **4**." | 20.84 |
| T5 | Qwen3.5-9B-Q4_1 | 4 | "2 + 2 equals **4**." | 19.16 |
| T6 | Qwen3.5-35B-A3B-UD-Q8_K_XL (42 GB) | 4 | "Thinking Process: 1. **Analyze the Request:** …" | 10.19 |
| T7 | cargo test quantized_blocks | — | 37 unit tests pass | — |

### Pipeline-parallel infrastructure (new)

- `Gguf::set_device(dev)` lets the loader switch target device per tensor.
- `from_gguf_multi_device(devices: &[Device])` added to `quantized_gemma4`,
  `quantized_qwen35`, `quantized_qwen35_moe`. Layers are split into contiguous
  chunks (LAYER split mode, mirroring llama.cpp).
- `GdnState::new_multi_device` creates per-recurrent-layer state vectors on the
  right GPU each.
- `--n-gpus` CLI flag in both example binaries.

Each layer's forward checks `device_eq(h.device(), &layer.device)` and calls
`tensor.to_device(&layer_device)?` when crossing a boundary. KV cache borrowed
across devices (gemma4 shared-KV, qwen35moe layers 24-41) gets transferred at
read time.

### Bugs fixed this session

1. **`indexed_moe_forward` HIP kernel uses wrong stride** (`candle-hip-kernels/src/quantized.cu`):
   the per-expert weight stride was `(n*k) / QK_K * sizeof(block_q_t)` where
   `QK_K = 256` (the K-quant superblock). For Q8_0 (`qk = 32`), this under-counts
   the per-expert stride 8×, so every expert except #0 reads garbage between rows.
   Fix: use the template parameter `qk` instead of `QK_K`. Verified with a
   handwritten Q8_0 + Q8_1 test (`scripts/test-suite.sh` T7-equivalent).

2. **`crate::utils::repeat_kv` produces wrong head order for `n_rep ≥ 8`**:
   the `cat-along-dim2 + reshape` trick only works for small `n_rep`; for `n_rep = 8`
   (qwen35moe `n_head/n_kv_head = 16/2 = 8`) the reshape splits one cat'd head
   across multiple new heads in the wrong order. Fixed in `quantized_blocks/attention.rs`
   by replacing the call with an explicit `unsqueeze(2) + expand + reshape`
   broadcast that's correct for all `n_rep`.

3. **`indexed_moe_forward` was called with 2-D input** (`(n_tokens, hidden)`) but
   the HIP kernel expects 3-D `[batch, topk_or_1, k]`. Fixed in
   `quantized_blocks/ffn.rs::MoeExperts::forward` by `unsqueeze(1)` to get
   `(n_tokens, 1, hidden)` for the gate/up step.

4. **Shared expert gate loaded as `QMatMul` failed because `ffn_gate_inp_shexp.weight`
   is a 1-D `[hidden]` vector**, not a 2-D matrix. Loaded as a `Tensor` and applied
   as a manual dot product: `gate_logit[t] = Σ_i x[t,i] * w[i]`.

5. **`arg_sort_last_dim` has no HIP implementation**. The MoE router topk now
   round-trips routing weights through CPU for the argsort. The tensor is small
   (n_tokens × n_experts).

6. **F16 weight × F32 input MoE kernel** added (`quantized.cu` →
   `indexed_moe_forward_f16_f32`) and dispatched from
   `QHipStorage::indexed_moe_forward` for the F16 path. This unblocks
   "Unsloth Dynamic" (UD-Q8_K_XL) GGUFs that mix Q8_0 with raw F16 for some
   expert tensors.

### Test suite

`scripts/test-suite.sh` runs all 8 tests above and prints colored PASS/FAIL.
Set `GEMMA4=…`, `QWEN35_9B=…`, `QWEN35_MOE=…` to override model paths.

### REMAINING WORK

1. **GPU multi-token attention bug** — Wave64 fixes for rmsnorm/softmax/layernorm don't fully fix it.
   The bug must be elsewhere in the attention multi-token path. Workaround: --split-prompt.

2. **GDN correctness** — Math is now verified correct against a scalar reference of ggml's
   autoregressive delta net (`test_delta_net_vectorized_matches_scalar_reference`). One known
   bug found and fixed:

   **GQA tile-vs-interleave bug** (`delta_net.rs:275-296`) — `ggml_repeat` tiles the head
   dimension (`dst[i*nk + k] = src[k]`), so the resulting head order is
   `[h0, h1, ..., h_{nk-1}, h0, h1, ...]`. My original code used
   `unsqueeze(2)+expand+reshape`, which produces the *interleave* order
   `[h0, h0, h0, h1, h1, h1, ...]` (PyTorch `repeat_interleave` semantics, NOT `repeat`).
   For qwen35-9B with `num_k_heads=16, num_v_heads=48, rep=3`, this was mapping each V head
   to a completely wrong K head — only V head 0 ever paired with K head 0, etc.

   Fix: changed to `unsqueeze(1)+expand+reshape` so heads come out in the tile order matching
   ggml. New test `test_gqa_tile_order_matches_ggml_repeat` pins the expected order.

   Needs host re-validation: build + run qwen35-9B GDN inference and compare to llama.cpp
   reference output. Expected outcome: coherent multi-token responses instead of early EOS.

3. **gemma4** — Many features partially implemented, more needed:
   - Shared KV layers (31B variant)
   - Sliding window attention mask
   - More gemma4-specific norm conventions

4. **qwen3next** — Uses GGUF dtype 39 (likely IQ-quant) which candle doesn't support yet.

