# Qwen3.5-9B architecture + matmul-dispatch comparison: candle vs llama.cpp

**Date:** 2026-04-14
**Model:** `/artefact/models/Qwen3.5-9B-Q4_1.gguf` (Unsloth quant of Qwen/Qwen3.5-9B)
**Hardware:** AMD MI50 (gfx906), ROCm 7.1.1
**Why this exists:** the M3a/M3b Q4_1 turbo port is bit-exact and faster
per-call on aligned shapes, but yields **~0% wall improvement** on
Qwen3.5-9B prefill. This doc nails down why — by laying out Qwen3.5's
real architecture, every matmul candle issues, and the dispatcher
each backend chooses for it.

---

## 1. Qwen3.5 architecture (from the GGUF metadata)

```
qwen35.block_count                 = 32
qwen35.context_length              = 262 144
qwen35.embedding_length            = 4 096    ← hidden
qwen35.feed_forward_length         = 12 288   ← FFN intermediate
qwen35.attention.head_count        = 16       ← Q heads
qwen35.attention.head_count_kv     = 4        ← KV heads (GQA, ratio 4:1)
qwen35.attention.key_length        = 256      ← head_dim
qwen35.attention.value_length      = 256
qwen35.full_attention_interval     = 4        ← 1 of every 4 layers is full attention
qwen35.rope.dimension_count        = 64       ← partial RoPE on first 64 dims of head
qwen35.ssm.state_size              = 128      ← GDN recurrent state
qwen35.ssm.inner_size              = 4 096
qwen35.ssm.group_count             = 16
qwen35.ssm.conv_kernel             = 4
qwen35.ssm.time_step_rank          = 32
```

**Layer schedule (32 layers total):**
- `full_attention_interval = 4` ⇒ every 4th layer is **full attention**, the
  rest are **Gated-Delta-Net** (GDN, linear-attention).
- 32/4 = **8 full attention layers** + **24 GDN layers**.
- All 32 layers also have a dense FFN (gate-up fused + down).

**Per-layer matmul inventory:**

| layer kind | #/model | matmuls per layer (Rust struct) | shapes (input → output) |
|---|---:|---|---|
| **Full attention** (`GatedAttention`) | 8 | `wqkv` (Q+gate +K+V fused) | 4096 → 8192 + 1024 + 1024 = **10240** |
|  |  | `wo` | 4096 → 4096 |
|  |  | `gate_up` (fused) | 4096 → **24576** (= 2 × 12288) |
|  |  | `down` | **12288** → 4096 |
| **GDN** (`DeltaNetLayer`) | 24 | `wqkv` (attn_qkv) | 4096 → conv_channels |
|  |  | `wqkv_gate` (attn_gate) | 4096 → 4096 |
|  |  | `ba` (ssm_ba, fused α/β) OR `alpha` + `beta` | 4096 → 2 × num_v_heads |
|  |  | `ssm_out` | 4096 → 4096 |
|  |  | `gate_up` (FFN) | 4096 → 24576 |
|  |  | `down` (FFN) | 12288 → 4096 |
| `lm_head` | 1 | output projection | 4096 → vocab (~152 064) |

**Q+gate fusion in `GatedAttention`** (qwen3.5-specific): `wqkv` produces
Q (16 × 256 = 4096) **interleaved with a gate** (16 × 256), then K (4 × 256
= 1024) and V (1024) — total output 8192 + 1024 + 1024 = **10240**. The
attention output is multiplied by `sigmoid(gate)` before `wo`. This is
why you see N=10240 not N=6144 (= Q+K+V) in the candle profile.

---

## 2. Per-tensor quantization mix in this GGUF

Unsloth's `Qwen3.5-9B-Q4_1.gguf` is **not uniformly Q4_1**. From the
candle prefill profile (rocprofv3, pp512):

| dtype | candle MMQ kernel calls | µs/call (avg) |
|---|---:|---:|
| **Q4_1** | **256** | 7 631 (largest matmuls — body weights) |
| **Q5_K** | **48** | 10 645 (lm_head + a few large projections) |
| **Q8_0** | **96** | 981 (smaller projections) |

So the inference touches **three** distinct quantization formats per
forward pass. Any "MMQ port" that only covers Q4_1 leaves Q5_K (511 ms,
**18 % of total MMQ time**) and Q8_0 (94 ms, 3 %) on the baseline path.

---

## 3. Candle's dispatch chain (this codebase)

### 3.1 The decision tree

For each `qweight × x` matmul, candle's path depends on:

```
QMatMul::forward(x)                                                      [transformers/with_tracing.rs]
└─> Module::forward → inner.forward                                      [core/quantized/mod.rs:1115]
    ├─> Self::QTensor(t)        → x.apply_op1_no_bwd(t)
    │   └─> CustomOp1::hip_fwd  → QHipStorage::fwd                       [core/quantized/hip.rs:1483]
    │       ├─ if b*m ≤ 8       → dequantize_matmul_vec ───────────────────►  MMVQ chunked path
    │       └─ else (prefill)   → dequantize_matmul                      [hip.rs:1670]
    │                                  └─> mul_mat_q_v2                  [hip.rs:524]
    │                                       ├─ M2/M3 turbo gate (Q4_0/Q4_1, K%32==0)
    │                                       │       └─> mul_mat_q40q41_turbo (NEW)
    │                                       └─ else → mul_mat_q_v2_with_tile_n
    │                                                   └─> mul_mat_<type>_gfx906_v2f_tile32
    ├─> Self::Tensor(w)         → xs.matmul(&w)         (rocBLAS path)
    └─> Self::TensorF16(w)      → similar via rocBLAS
```

The last two arms are taken when `from_arc` decided to dequantize at
load time (F16/F32/BF16/MXFP4/IQ4_XS, or `CANDLE_DEQUANTIZE_ALL=1`).

### 3.2 Specialized "preq8" entry for decode

For **decode (b ≤ 8)**, qwen35's attention/FFN bypass `QMatMul::forward`
and call `forward_preq8` directly with a pre-quantized Q8_1 buffer:

```
GatedAttention::compute_qkv_shared_q8 (attention.rs:748)                 // b ≤ 8 only
└─> wq.forward_preq8(y_q8_1, b_size, rhs_shape)
    └─> QHipStorage::fwd_with_preq8                                      [hip.rs:1544]
        └─> launch_mul_mat_vec_q8_1_chunk                                [hip.rs:342]
            └─> mul_mat_vec_<type>_q8_1_cuda{N}                          // MMVQ family
```

This skips the per-call `quantize_q8_1` dispatch by reusing the same
Q8_1 buffer across all 3 (Q/K/V) projections of the layer.

### 3.3 Per-dtype kernel families (current state)

For each quant type candle has its own `_v2{,_tile16,_tile32,_tile64}` set:

| dtype  | MMVQ (decode) | MMQ baseline (prefill) | M2/M3 turbo port |
|---|---|---|---|
| Q4_0   | `mul_mat_vec_q4_0_q8_1_cudaN` | `mul_mat_q4_0_gfx906_v2f_tile32` | **`mul_mat_q4_0_turbo_x{8,16,32,64}_{checked,unchecked}`** |
| Q4_1   | `mul_mat_vec_q4_1_q8_1_cudaN` | `mul_mat_q4_1_gfx906_v2f_tile32` | **`mul_mat_q4_1_turbo_x{8,16,32,64}_{checked,unchecked}`** (M3a) |
| Q5_0   | … `q5_0_q8_1_cudaN` | `mul_mat_q5_0_gfx906_v2f_tile32` | — |
| Q5_1   | … `q5_1_q8_1_cudaN` | `mul_mat_q5_1_gfx906_v2f_tile32` | — |
| Q8_0   | … `q8_0_q8_1_cudaN` | `mul_mat_q8_0_gfx906_v2f_tile32` | — |
| Q4_K   | … (warp-coop variant) | `mul_mat_q4_K_gfx906_v2f_tile32` | — |
| **Q5_K** | … (warp-coop variant) | `mul_mat_q5_K_gfx906_v2f_tile32` | — |
| Q6_K   | … (warp-coop variant) | `mul_mat_q6_K_gfx906_v2f_tile32` | — |

### 3.4 Tile geometry of the v2f baseline (per `quantized.cu:5530`)

Single kernel: `__launch_bounds__(64, 1)` — wg = **1 wave (64 threads)**,
TILE_M = WARP_SIZE = 64 (one row per thread), TILE_N = 32 (one tile is
64 rows × 32 cols). 88 VGPRs, 0 LDS, scalar X loads. Optimised for
single-wave occupancy and simple K-loop unroll-by-2 with split
accumulators (Phase 2d v2f line of work).

### 3.5 Tile geometry of the M3a/M3b turbo port (`mmq_turbo.cu`)

Mirrors turbo: wg = **64 × 4 = 256 threads** (4 waves), `mmq_y = 128`
output rows per WG, `mmq_x ∈ {8,16,32,64}` output cols per WG (adaptive
to batch), `__launch_bounds__(256, 2)` ⇒ 128 VGPRs, 2 waves/SIMD
occupancy. 20-40 KB LDS for the X qs/df + Y qs tiles. K-OOB handled
in-kernel via `kb_remaining` clamp; N-OOB handled by `_checked` variant.

### 3.6 What the qwen35 Rust model actually calls

In `quantized_qwen35.rs::forward`, per layer:
- For full-attention: `attn.forward(x, mask, offset)` →
  `GatedAttention::forward` (attention.rs:1107) → `wqkv.forward(x)` and
  `wo.forward(attn_out)` — **standard `QMatMul::forward` path**, hits
  `QHipStorage::fwd` then `mul_mat_q_v2`.
- For GDN: `gdn.forward_step(x, &mut state)` →
  `DeltaNetLayer::forward_prefill` (delta_net.rs:452) → `wqkv.forward`,
  `wqkv_gate.forward`, `ba.forward`, `ssm_out.forward` — **same path**.
- FFN every layer: `ffn.forward(x)` → `gate_up.forward(x)` then
  `down.forward(activated)` (ffn.rs:183) — **same path**.

So **all** prefill matmuls go through `QHipStorage::fwd`. There's no
hidden bypass. (My earlier suspicion that they bypassed was a
build-cache artifact — `cargo build -p candle-examples --features hip`
doesn't rebuild the `quantized-qwen35` example, which lives in
`candle-core/examples/`. The eprintln'd binary was stale.)

---

## 4. llama.cpp's dispatch (vanilla, b8703)

llama.cpp uses **a single templated kernel family** for all DP4A-eligible
quants:

```c++
template <ggml_type type, int mmq_x, bool need_check>
__launch_bounds__(64*4, 2)
__global__ void mul_mat_q(...);   // mmq.cuh:3638
```

with type-specific `load_tiles_<type>` and `vec_dot_<type>_q8_1_dp4a`
(or `_mma`) selected via `mmq_type_traits<type>`. On gfx906:
- `nwarps = 256/64 = 4` (mmq.cuh:299)
- `mmq_y = 128` (`get_mmq_y_device`, line 153)
- `mmq_x` adaptive: `{8, 16, 24, 32, ..., 64}` (host-side switch at
  line 4183)
- `MMQ_ITER_K = 256` K-elements per outer K-iter
- DP4A path; no MFMA on gfx906

Dispatch in `launch_mul_mat_q` (mmq.cuh:4060):

```c++
const int nty  = ceil(nrows_x / mmq_y);     // M tiles
const int ntx  = ceil(ncols_max / mmq_x);   // N tiles
dim3 block_dims(64, 4, 1);
dim3 block_nums(nty, ntx, ntzw);
mul_mat_q<type, mmq_x, need_check><<<block_nums, block_dims, ...>>>
```

with `need_check = (nrows_x % mmq_y != 0)` chosen at the launch site.

**Stream-K alternative** (mmq.cuh:4106) is on for CDNA/non-AMD; on gfx906
the regular xy-tiling branch wins, and that's what we measured (`156 ms
/ 302 calls = 517 µs/call` for TinyLlama Q4_0 pp512).

---

## 5. Side-by-side: per-matmul dispatch on Qwen3.5-9B prefill

Considering the ~177 MMQ calls per inference (32 layers × ~5-6 matmuls +
lm_head), grouped by shape and dtype, here is what each backend chooses:

| matmul group | shape (M=512 batch) | dtype | candle baseline | candle M3a/b | llama.cpp |
|---|---|---|---|---|---|
| Full-attn `wqkv` (×8) | 4096 → 10240 (K=4096, N=10240) | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| Full-attn `wo` (×8) | 4096 → 4096 | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| FFN `gate_up` (×32) | 4096 → 24576 | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| FFN `down` (×32) | 12288 → 4096 | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| GDN `wqkv` (×24) | 4096 → conv_channels | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| GDN `wqkv_gate` (×24) | 4096 → 4096 | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| GDN `ba` (×24) | 4096 → 2·num_v_heads | Q4_1 (likely) | v2f_tile32 | **turbo_x64_unchecked** ✓ if N%128=0 | mul_mat_q<x?> |
| GDN `ssm_out` (×24) | inner=4096 → 4096 | Q4_1 | v2f_tile32 | **turbo_x64_unchecked** ✓ | mul_mat_q<x64> |
| `lm_head` (×1) | 4096 → 152064 | **Q5_K** | mul_mat_q5_K_v2f_tile32 | **NOT YET PORTED** | mul_mat_q<Q5_K> |
| (small projections, ×96) | various | **Q8_0** | mul_mat_q8_0_v2f_tile32 | **NOT YET PORTED** | mul_mat_q<Q8_0> |

**All Q4_1 K-dims are 4096 or 12288 — both multiples of 256.** So my
M3b K-OOB handling (`kb_remaining` clamp) is *not the lever* here. The
real lever is "Q4_1 turbo coverage", which M3a already provides.

**The reason the Qwen3.5 wall didn't move** in my last run: I was
benching against a **stale `quantized-qwen35` example binary** that
didn't have the M3a Q4_1 wiring. `cargo build -p candle-examples` only
rebuilds the `quantized` (llama-style) example. To exercise M3a on
qwen35 I need to rebuild that example explicitly:

```bash
cargo build --release --example quantized-qwen35 -p candle-core --features hip
```

Then re-run with `CANDLE_MMQ_TURBO_PORT=1` and the wall should jump.

---

## 6. What llama.cpp does that candle (still) doesn't

Even after M2 (Q4_0) and M3a/b (Q4_1) close the per-call MMQ gap to
~1.27× of llama.cpp on Q4_0, three structural differences remain:

| diff | candle | llama.cpp | Qwen3.5 impact |
|---|---|---|---|
| **Per-dtype kernel families** | 8 hand-tuned families (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K) | one `mul_mat_q<type>` template, type-specific traits | Adding a new dtype to candle = full new kernel; in llama.cpp = a `load_tiles_<type>` + `vec_dot_<type>` pair. We need separate Q5_K/Q8_0 ports; llama.cpp gets them for free. |
| **Fused FFN/attention pointwise** | separate `silu_mul_split_last_fused`, separate `.contiguous()` on q/k transposes (~13 ms `copy2d_f32` per pp512 inference) | fully fused inside `mul_mat_q` epilogue + flash-attn | ~50 ms gap on Qwen3.5 prefill that's not in MMQ. |
| **MMVQ vs MMQ split at b≤8** | hard split: `forward_preq8` for ≤8, `mul_mat_q_v2` for >8 | unified — `mul_mat_q` template handles both via `mmq_x ∈ {8..64}` adaptive | candle decode goes through MMVQ-vec-chunked (4096 cols × 1 batch); llama.cpp uses `mmq_x=8` of the same `mul_mat_q` family. |

---

## 7. Concrete asks for the next M3 steps

1. **M3-rebench**: rebuild `quantized-qwen35` and re-measure with the
   M3a Q4_1 turbo on. Expected (per-call already verified bit-exact):
   pp512 candle 352 → ~600-700 t/s, closing 0.47× → ~0.85× of llama.cpp.

2. **M3c: port Q8_0** (the cheap one). Same pattern as Q4_0 (single
   delta, no min); only the qs unpacking changes (already int8, no
   nibble split). 96 calls × ~1 ms × 1.7× = ~40 ms saved on Qwen3.5.

3. **M3d: port Q5_K** (the expensive one). K-quants have a different
   block layout (super-block of 8 sub-blocks, scales split across `qs`
   + `sc`/`dm` headers). Requires a new tile_x layout. ~250 ms saved
   on Qwen3.5 lm_head + projections.

4. **M4: fused-FFN-prefill** kernel — rmsnorm + gate_up + silu*gate +
   down, single launch. Removes the per-FFN copy/silu/binary calls
   (~13 ms × 32 layers per Qwen3.5 inference).

5. **M5: unify MMVQ + MMQ** — rewire the decode path through the same
   `mul_mat_q4_*_turbo_x8_unchecked` so we have a single MMQ family
   covering b=1..512+, like llama.cpp. Decode parity is already there;
   the win is consolidation + future maintainability.

---

## 8. References

- Qwen3.5 GGUF metadata: read with `gguf-py` from
  `/artefact/models/Qwen3.5-9B-Q4_1.gguf`.
- candle dispatch: `candle-core/src/quantized/hip.rs:524-1700`.
- candle qwen35 model: `candle-transformers/src/models/quantized_qwen35.rs`.
- candle attention/FFN blocks: `candle-transformers/src/models/quantized_blocks/{attention,ffn,delta_net}.rs`.
- llama.cpp dispatch: `/artefact/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:309-3730`.
- M2/M3 port: `candle-hip-kernels/src/mmq_turbo.cu`,
  `candle-core/src/quantized/hip.rs:567-720` (`mul_mat_q40q41_turbo`).
- prior 3-way bench: `BENCH-3WAY-2026-04-11.md` and
  `BENCH-3WAY-POST-P2-2026-04-11.md`.
