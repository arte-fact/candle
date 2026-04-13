# AMD Library Reference for gfx906 Work

Compiled reference of AMD-official documentation for the ISA instructions,
library APIs, and hardware rules relevant to our candle gfx906 work. Every
claim cites a source (PDF section, URL, or on-disk header path). Pair with
`SURVEY-GFX906-FORKS-2026-04-13.md`.

**Raw PDFs locally mirrored:**
- `docs/vega-isa-26nov2019.txt` — "Vega 7nm Shader ISA" (290 pages). Original at
  [gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf](https://gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf).
- `docs/vega-isa-27jan2020.txt` — "Vega ISA Reference Guide" (288 pages), later
  revision. Content is near-identical for our sections.
- Original PDF: `vega-shader-instruction-set-architecture.pdf` in repo root.

---

## 1. DPP (Data-Parallel Primitive) — Vega ISA §13.3.9 + Table 81

**Our relevance:** we disabled DPP in Phase J because warp reductions miscompiled on
gfx906+ROCm-7.1.1. Turbo uses DPP successfully on the exact same hardware/ROCm —
so the bug is in our code or ASM, not the platform.

### 1.1 Mandatory wait states (Table 8, §4.5)

| First instruction | Second instruction | Wait states | Notes |
|---|---|---|---|
| VALU writes VGPR | VALU DPP reads that VGPR | **2** | |
| VALU writes EXEC | VALU DPP op | **5** | ALU does not forward EXEC to DPP |
| VALU sets VCC/EXEC | VALU reads EXECZ/VCCZ | 5 | |
| VALU writes SGPR | VMEM reads that SGPR | 5 | |
| VALU writes SGPR | V_{READ,WRITE}LANE | 4 | |

**The hardware does not check these dependencies; they must be resolved by inserting NOPs or independent instructions.**

### 1.2 DPP format (Table 80, §13.3.9)

DPP is a 32-bit modifier word that can follow VOP1/VOP2/VOPC in place of a literal constant:

| Field | Bits | Meaning |
|---|---|---|
| SRC0 | [39:32] | Real SRC0 operand (VGPR) |
| DPP_CTRL | [48:40] | 9-bit enum — see Table 81 below |
| BC (bound_ctrl) | [51] | 0 = don't write when source OOB; 1 = write (use `bound_ctrl:0` in asm to set) |
| SRC0_NEG | [52] | 1 = negate src0 |
| SRC0_ABS | [53] | 1 = abs(src0) |
| SRC1_NEG | [54] | 1 = negate src1 |
| SRC1_ABS | [55] | 1 = abs(src1) |
| BANK_MASK | [59:56] | 4-bit mask disabling destination lanes per 4-bank group |
| ROW_MASK | [63:60] | 4-bit mask disabling dest lanes per 16-lane row |

### 1.3 DPP_CTRL enumeration (Table 81)

| name | hex | semantics |
|---|---|---|
| `DPP_QUAD_PERM` | `0x000-0x0FF` | Permute of four threads: `pix[n] = pix[(n&0x3c) + cntl[n%4*2+1:n%4*2]]`. Control bits are 8 bits picking 4× (0-3) mappings. |
| `DPP_ROW_SL*` | `0x101-0x10F` | Row shift LEFT by 1-15 lanes within each 16-lane row |
| `DPP_ROW_SR*` | `0x111-0x11F` | Row shift RIGHT by 1-15 lanes |
| `DPP_ROW_RR*` | `0x121-0x12F` | Row ROTATE right by 1-15 lanes (wraps within row) |
| `DPP_WF_SL1` | `0x130` | Wavefront shift left by 1 lane |
| `DPP_WF_RL1` | `0x134` | Wavefront rotate left by 1 lane |
| `DPP_WF_SR1` | `0x138` | Wavefront shift right by 1 lane |
| `DPP_WF_RR1` | `0x13C` | Wavefront rotate right by 1 lane |
| `DPP_ROW_MIRROR` | `0x140` | Mirror within 16-lane row: `pix[n] = pix[15-(n&0xF)]` |
| `DPP_ROW_HALF_MIRROR` | `0x141` | Mirror within 8-lane halfrow: `pix[n] = pix[7-(n&7)]` |
| `DPP_ROW_BCAST15` | `0x142` | Broadcast lane 15 of each row to the next row |
| `DPP_ROW_BCAST31` | `0x143` | Broadcast lane 31 to rows 2 and 3 |

### 1.4 DPP cannot be used with (§12.19.1)

Specifically excluded from DPP:
- `v_madmk_{f32,f16}`, `v_madak_{f32,f16}`
- `v_readfirstlane_b32`
- All F64 conversions: `v_cvt_{i32_f64, f64_i32, f32_f64, f64_f32, u32_f64, f64_u32}`
- All F64 arithmetic: `v_trunc_f64`, `v_ceil_f64`, `v_rndne_f64`, `v_floor_f64`, `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64`, `v_frexp_exp_i32_f64`, `v_frexp_mant_f64`, `v_fract_f64`
- `v_clrexcp`
- `v_swap_b32`
- All 64-bit compares: `v_cmp(x)_*_{f64,i64,u64}`, `v_cmp(x)_class_f64`

**These restrictions mean DPP IS valid for**: `v_add_f32`, `v_sub_f32`, `v_mul_f32`, `v_max_f32`, `v_min_f32`, `v_mov_b32`, `v_and_b32`, `v_or_b32`, `v_xor_b32`, all 16-bit FP ops, all 32-bit integer compares — the full set of single-lane ops turbo uses.

### 1.5 LLVM/clang intrinsics (not DPP but related)

From `/opt/rocm-7.1.1/core-7.13/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc`:
- `__builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)` — gfx906+
- `__builtin_amdgcn_update_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)` — gfx906+, preferred for preserving disabled-lane values
- `__builtin_amdgcn_mov_dpp8(src, sel)` — gfx10+, N/A for us

The `update_dpp` variant is what turbo uses (`gfx906-common.cuh:73-93`) for the fused DPP add/max — the `old` operand keeps disabled lanes as-is, which matches the "keep zero on row boundary" semantics the warp reduction needs.

---

## 2. v_dot* packed-dot-product instructions — gfx906

**Our relevance:** Phase B3 K-quant warp-coop uses `dp4a` (v_dot4_i32_i8) for the Q4_K/Q5_K/Q6_K inner products. Turbo uses this successfully in `ggml/src/ggml-cuda/gfx906/quantize/vecdotq.cuh:64-67`; our gated-off version has a bug.

From [LLVM gfx906 asm ref](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html) + `BuiltinsAMDGPU.inc`:

### 2.1 Instructions (VOP3P format, 64-bit encoding)

| instruction | src0 / src1 | src2 / dst | clang builtin |
|---|---|---|---|
| `v_dot2_f32_f16` | f16×2 | f32 accumulator | `__builtin_amdgcn_fdot2(a, b, c, clamp)` |
| `v_dot2_i32_i16` | i16×2 | i32 | `__builtin_amdgcn_sdot2` |
| `v_dot2_u32_u16` | u16×2 | u32 | `__builtin_amdgcn_udot2` |
| **`v_dot4_i32_i8`** | **i8×4** | **i32** | **`__builtin_amdgcn_sdot4(a, b, c, clamp)`** |
| `v_dot4_u32_u8` | u8×4 | u32 | `__builtin_amdgcn_udot4` |
| `v_dot8_i32_i4` | i4×8 | i32 | `__builtin_amdgcn_sdot8` |
| `v_dot8_u32_u4` | u4×8 | u32 | `__builtin_amdgcn_udot8` |

All four-input: `dst = src0 · src1 + src2` (fused multiply-accumulate). All support the `[clamp]` modifier.

### 2.2 Target features

gfx906 ships `+dot1-insts` (`dot1`) and `+dot2-insts` (`dot2`) features — these enable the above instructions. `+dot3-insts` and higher are gfx908+/gfx942+ (MFMA family).

### 2.3 Why candle's gated-off path likely fails

We use `__builtin_amdgcn_sdot4(a, b, c, false)` — same call as turbo. If turbo works and ours doesn't, suspects:
- Wrong operand packing: `v_dot4_i32_i8` expects i8×4 packed into a uint32 with a specific lane order. Check if we pack LSB-first consistently.
- Wrong signed/unsigned flag: Q4_0 values are nominally u4 but post-offset become i4/i8. Use `udot4` if your operand is already unsigned; `sdot4` if signed.
- `clamp` set when shouldn't be — clamping to i32 range isn't free.

Turbo's use at `vecdotq.cuh:64-67` should be the reference: operand pack order, signed/unsigned choice, clamp setting. Direct port gives a working baseline.

---

## 3. Wait states beyond DPP (Table 8 entirety)

Important rules candle kernels must follow (our kernels compile to correct code because clang handles this; relevant if we drop to inline asm):

- **VALU writes EXEC → VALU uses EXECZ/VCCZ: 5 wait states.** If our kernel does `v_cmpx_*` (writes EXEC) then branches/reads EXECZ, we need `s_nop 4` or 5 independent instructions between them.
- **VALU writes SGPR → VMEM reads that SGPR: 5 wait states.** Rarely applicable to our compute kernels.
- **VALU writes SGPR/VCC → V_READLANE/V_WRITELANE using that SGPR as lane select: 4 wait states.** Turbo's `hip_shuffle_xor4_f32` uses `ds_swizzle_b32` (LDS path) specifically to avoid this.

---

## 4. LDS bank model — §3.6 "GPRs and LDS"

**Our relevance:** Phase R (LDS `+1` padding) — port turbo's cpy.cu:59 pattern to our flash-attn-v2 k_lds/v_lds tiles. Reference: [AMD ROCm blog on LDS bank conflicts](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html).

### 4.1 LDS physical layout

- **32 banks** per CU, each 4 bytes wide → **128 bytes per LDS clock cycle max throughput**.
- Wave64: 64 threads issue 2 cycles per LDS op (32 threads issue per cycle on Vega/CDNA1 — different from CDNA2+ that issues all 64 in 1 cycle).
- Bank index = `(byte_addr / 4) % 32`.

### 4.2 N-way bank conflicts

A conflict occurs when N threads in a wave/half-wave target the same bank. Worst case on wave64:
- Stride-32 access: all 32 issued threads hit bank 0 → 32-way conflict → 32× slowdown.
- Stride-1 access: no conflict (each thread owns its bank).

### 4.3 Padding rule — the `+1` trick

For a 2D tile `float tile[ROWS][COLS]` where threads consume COLUMNS (each thread reads `tile[row_tid][row_col]` for fixed `row_col` varying `row_tid`):
- Base layout: `tile[ROWS][COLS]` — column stride = COLS floats = 4*COLS bytes. If `COLS % 32 == 0`, bank conflict every column access.
- Padded layout: `tile[ROWS][COLS+1]` — column stride shifted by 1 float. Adjacent threads hit different banks.

Turbo applies this exactly once in `ggml/src/ggml-cuda/cpy.cu:59`:
```c
__shared__ float tile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];
```
(2D transpose kernel, TS=32 → 32×33 layout.)

### 4.4 XOR-swizzle alternative

For certain access patterns, transforming the column index via XOR can also break conflicts without spending LDS:
```
index = row * COLS + (col ^ (row & 0x3))  // or similar XOR pattern
```
From AMD's CK-Tile blog — used for sub-tiled GEMM. Harder to reason about; padding is generally the safer first step.

### 4.5 Our relevant tile declarations

- `flash_attn_v2.cu` `flash_attn_fwd_v2_impl` template: `__shared__ float k_lds[BC * D];` and `v_lds[BC * D];`
  - For D=256, BC=16 (d256 variant): 16×256 bytes, stride-256 column access — D % 32 == 0, potential conflicts.
  - Apply `+1 vec4` padding by declaring `k_lds[BC * (D + 4)]` and indexing `k_lds[j * (D+4) + d]`.
- `quantized.cu` MMQ v2f tile32: similar pattern.

---

## 5. rocBLAS API — sgemv_strided_batched

**Our relevance:** Phase O attempted this kernel; crashes on gfx906+ROCm-7.1.1 comgr-JIT path (Phase P rendered it obsolete, but if we ever resume):

### 5.1 Signature (from [rocm.docs.amd.com/projects/rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-2.html))

```c
rocblas_status rocblas_sgemv_strided_batched(
    rocblas_handle handle,
    rocblas_operation transA,   // NoTrans / Trans / ConjTrans
    rocblas_int m, rocblas_int n,
    const float *alpha,
    const float *A, rocblas_int lda, rocblas_stride strideA,
    const float *x, rocblas_int incx, rocblas_stride stridex,
    const float *beta,
    float *y, rocblas_int incy, rocblas_stride stridey,
    rocblas_int batch_count);
```

### 5.2 Semantics

For i = 0 to batch_count − 1:
```
y_i := alpha · op(A_i) · x_i + beta · y_i
A_i = A + i·strideA
x_i = x + i·stridex
y_i = y + i·stridey
```

### 5.3 Constraints

- **lda**: `>= max(1, m)` (A shape in col-major view is (m, n), lda is first-dim stride).
- **strideA**: no fixed minimum; typically `>= lda * n`.
- **stridex**: typically `>= n * incx` (NoTrans) or `>= m * incx` (Trans).
- **stridey**: typically `>= m * incy` (NoTrans) or `>= n * incy` (Trans); **must be non-zero**.
- **alpha/beta**: accept host OR device pointers, switched via `rocblas_set_pointer_mode(handle, rocblas_pointer_mode_{host,device})`. Default host.

### 5.4 Known gfx906 issue

> "Performance of non-batched and batched rocblas_sgemv has been improved for gfx906 when m <= 6000 and n <= 6000." — [rocBLAS changelog](https://github.com/ROCm/rocBLAS/blob/develop/CHANGELOG.md)

On ROCm 7.1.1, the sgemv dispatch bottom falls through to comgr JIT for certain shapes (observed via strace: `/tmp/comgr-XXXX/output/hipfatbin-*.o` creation right before SIGSEGV). This is NOT a missing Tensile kernel — direct diff of
`/opt/rocm-7.1.1/lib/rocblas/library/` vs mixa3607's 7.2.1 tensile bundle shows we have a strict superset (255 vs 156 files, our 255 includes all 156). See `ROADMAP-ROCM-722-MIGRATION-2026-04-13.md` §1 withdrawn note.

Workaround: don't use rocBLAS sgemv; write the mat-vec as a custom kernel (which Phase P did — `gqa_decode_mv_d{64..512}_f32`) or call it as a gemm with n=1 (Phase Q2 option).

---

## 6. HIP warp primitives — what `__shfl_xor` lowers to

**Our relevance:** understand why our `__shfl_xor`-based warp reduce is ~12 cycles vs turbo's DPP ~7.

### 6.1 `__shfl_xor` (HIP)

HIP's `__shfl_xor(value, laneMask, width=WARP_SIZE)` is lowered by clang to:
- `v_readlane_b32 s_tmp, value, (tid ^ laneMask)` — scalar register
- `v_writelane_b32 result, s_tmp, tid` — back to VGPR

Each lane takes 2 VALU cycles per shuffle. For a wave64 reduction over 6 stages (xor_1, 2, 4, 8, 16, 32): **~12 VALU cycles + SGPR hazards**.

### 6.2 `__builtin_amdgcn_ds_bpermute(addr, data)`

Uses LDS crossbar hardware without reading LDS itself — a single cycle for arbitrary permutation. gfx906-specific gotcha: `addr` operand is in BYTES, so multiply lane index by 4 before passing.

Turbo uses this for xor-16 reduction: `hip_shuffle_xor16_f32(x)` at `gfx906-common.cuh:108-141`.

### 6.3 `ds_swizzle_b32`

LDS-crossbar swizzle WITHOUT actually reading LDS storage. Pattern encoded in 16-bit operand:
- Bits [15]: `1` = constant swizzle (via FFT pattern), `0` = constant broadcast
- Bits [14:0]: pattern bits, format depends on [15]

Turbo uses this for specific fixed permutations. Single cycle, no SGPR traffic.

### 6.4 DPP reduction — how turbo does it in one cycle

Instead of `v_readlane` round-trip, turbo uses `v_add_f32_dpp` which performs:
```
dst_lane = src0_lane + src1_permuted_lane    (in 1 VALU cycle)
```
where `src1_permuted_lane` comes from an adjacent lane determined by `dpp_ctrl`. Zero extra register traffic.

For a 64-wide sum reduction:
```
x = v_add_f32_dpp x, x, quad_perm:[1,0,3,2]   // pairs swap, 1 cycle
x = v_add_f32_dpp x, x, quad_perm:[2,3,0,1]   // 4-groups swap
x = x + __shfl_xor(x, 4)                     // DPP only supports row_shl/shr:N<=15; 4 fits via row_ror:4
x = v_add_f32_dpp x, x, row_ror:8
x = x + __ds_bpermute(x, ((tid^16)<<2))       // xor-16, LDS crossbar
x = x + __shfl_xor(x, 32)                     // cross-half, fall back to standard shfl
```

~7 VALU cycles + 1 LDS xbar cycle + 1 standard shfl. Turbo's `gfx906-common.cuh:119-142` implements this in the `warp_reduce_amd_f32` template.

---

## 7. Candle quick-reference: bugs to re-examine in light of this

| feature | status | likely real cause | fix path |
|---|---|---|---|
| Phase J DPP reductions | disabled (was "broken on MI50+ROCm 7.1.1") | our DPP impl missing the 2-wait-state between VALU write and DPP read; OR wrong `dpp_ctrl` encoding; OR wrong operand order. | Port turbo's `DEFINE_FUSED_DPP_F32` macro verbatim (includes explicit `s_nop N`). `gfx906-common.cuh:73-93`. |
| Phase B3 K-quant warp-coop | gated behind `CANDLE_KQUANT_WARP_COOP=1` | `v_dot4_i32_i8` operand packing or signed/unsigned wrong. | Diff our `dp4a` calls vs turbo's `vecdotq.cuh:64-67`; match packing. |
| Phase O rocBLAS sgemv | abandoned | comgr JIT segfault on specific shape. Not a missing kernel. | Don't use rocBLAS sgemv; wrote own kernel (Phase P `gqa_decode_mv`). Done. |

---

## 8. Reference URLs

- [Vega ISA Reference (7nm, 26-Nov-2019)](https://gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf)
- [Vega ISA Reference (27-Jan-2020 update)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-shader-instruction-set-architecture.pdf) — also saved locally at `vega-shader-instruction-set-architecture.pdf`
- [AMD GCN Assembly Cross-Lane Operations (GPUOpen)](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/) — DPP tutorial
- [LLVM AMDGPU gfx906 asm reference](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html) — v_dot* instructions
- [LLVM AMDGPU User Guide](https://llvm.org/docs/AMDGPUUsage.html) — target features (+dot1-insts, +dot2-insts), general codegen
- [rocBLAS API Reference Level 2](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-2.html) — sgemv_strided_batched signature
- [rocBLAS changelog](https://github.com/ROCm/rocBLAS/blob/develop/CHANGELOG.md) — gfx906 perf notes
- [AMD ROCm blog: LDS bank conflicts + CK-Tile](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html) — padding + XOR swizzle
- [skyne98/wiki-gfx906 LDS layout study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-lds-layout-standard-llm.html) — 2× LDS bw from +1 padding (measured)
- [skyne98/wiki-gfx906 KV-Cache study](https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-kv-cache-read-write-study.html) — validates Phase P T-major K layout
- [arkprojects.space AMD gfx906 wiki](https://arkprojects.space/wiki/AMD_GFX906) — community references

## 9. Local header paths for gfx906 codegen

- `/opt/rocm-7.1.1/core-7.13/lib/llvm/lib/clang/23/include/amdgpuintrin.h` — HIP intrinsics header
- `/opt/rocm-7.1.1/core-7.13/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc` — clang AMDGCN builtin declarations (full list of `__builtin_amdgcn_*`)
- `/opt/rocm-7.1.1/include/hip/` — HIP runtime headers
- `/opt/rocm-7.1.1/include/rocblas/` — rocBLAS public headers (use `rocblas-functions.h` for signatures)
- `/opt/rocm-7.1.1/lib/rocblas/library/` — Tensile pre-compiled kernels (255 gfx906 files)
