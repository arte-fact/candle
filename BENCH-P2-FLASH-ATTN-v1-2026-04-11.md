P2 — Custom flash-attention HIP kernel, v1 (2026-04-11)
======================================================

Goal
----
Replace the `repeat_kv + matmul + mask + softmax + matmul` chain in
`quantized_llama::LayerWeights::forward_attn` with a single-pass
flash-attention kernel on HIP. Scope: f32 Q/K/V, head_dim ∈ {64, 128},
llama-style GQA via `h_kv = h_idx / n_rep`.

Rationale: pre-P2 profiling on TinyLlama showed rocBLAS attention
(`Cijk_*`) at ~20 % of GPU time, plus softmax + where + copy2d support
kernels adding another ~30 %. A fused kernel should collapse all of
these into one launch.

What landed
-----------
New files:
- `candle-hip-kernels/src/flash_attn.cu` — templated D=64 / D=128 kernel.
  BR=1 (one Q row per Wave64), direct global loads (no LDS tile —
  the one-reuse pattern made the LDS write pure overhead and capped
  occupancy at 2 waves/CU). Grid = (L_q, n_head, B), block = 1 Wave64.
  Online softmax (`m_i, l_i, o_reg`) via `__expf` + `__shfl_xor`
  warp-reduce. Supports additive mask with arbitrary broadcast over
  the B and L_q dims (mask_b_stride / mask_l_q_stride parameters).
- `candle-core/src/hip_backend/flash_attn.rs` — Rust launcher with
  full precondition validation (device / dtype / contiguity / head
  dim) and mask-shape-to-stride derivation.
- `candle-hip-kernels/src/lib.rs` — `FlashAttn` module id (index 5,
  existing ids shifted).
- `candle-core/src/hip_backend/mod.rs` — `pub use flash_attn_fused`.
- `candle-transformers/src/models/quantized_blocks/attention.rs` —
  oracle test `hip_flash_attn_matches_cpu_oracle_d64` with 7 cases
  (no-GQA, GQA n_rep ∈ {2, 4, 8}, batch=2, long decode L_k=1149,
  with/without mask). All within `max_abs < 1e-4` of the CPU oracle.
- `candle-transformers/src/models/quantized_llama.rs` — opt-in fast
  path gated on `CANDLE_FLASH_ATTN_ENABLE`, `seq_len >= 4` (decode
  stays on rocBLAS), HIP-only, f32 only, contiguous only.

Why opt-in (not default)
------------------------
Measured on TinyLlama 1.1B Q4_0 (n_head=32, n_kv_head=4, head_dim=64,
MI50 gfx906, ROCm 7.1.1), 1065-token prompt, 63-token sample, 3 runs:

| Mode               | Prefill t/s       | Decode t/s      |
|--------------------|-------------------|-----------------|
| baseline (OFF)     | 1116 / 1144 / 1159| 98 / 99 / 102   |
| flash-attn ON      | 1209 / 982 /1205  | 65 / 64 / 64    |

Prefill: ~4 % median win, but noisy (one run dropped to 982).
Decode:  ~35 % regression despite the fact that the fast path is
gated out at `seq_len >= 4`, so flash-attn doesn't fire at all
during the decode phase. Root cause still unexplained — hypotheses
include allocator fragmentation from the (1, 1, L_q, L_k) f32 mask
materialisation, or numerical divergence in the prefill output
pushing the sampled token stream through an unluckier KV-cache
growth pattern. Needs follow-up.

Kernel trace (rocprofv3 --kernel-trace) on the same workload:

| Kernel                  | OFF (ms) / calls | ON (ms) / calls |
|-------------------------|------------------|-----------------|
| flash_attn_fwd_d64_f32  | —                | 607 / 22        |
| Cijk_* (rocBLAS attn)   | 159 / 2773       | 107 / 2377      |
| softmax_f32             |  74 / 1449       |  41 / 1243      |
| where_u8_f32            |  77 /   22       |   — /    0      |
| mul_mat_q4_0_v2f_tile32 | 208 /  154       | 204 /  154      |
| copy2d_f32              | 130 / 27632      | 112 / 23760     |
| **TOTAL**               | **943**          | **1328**        |

Flash-attn fires 22× for prefill (one call per layer) and eats 607 ms
— ~17× worse than the estimated rocBLAS prefill attention cost
(~36 ms, back-calculated from the per-call decode cost × prefill
call count). The support kernels `softmax / where / affine` go away
as expected, but the kernel itself is much bigger than the sum of
what it replaced.

Per-call cost: 607 ms / 22 layers = 27.6 ms/layer prefill vs
rocBLAS ~1.6 ms/layer prefill. Theoretical lower bound at 10 %
memory BW is ~12 ms/layer, so even unoptimised we're 2.3× off
ideal — and rocBLAS is ~17× better.

Why BR=1 loses
--------------
The BR=1 design has each Wave64 own one Q row, with a scalar `j = 0..L_k`
inner loop that does:
  1. `partial = q_reg * k[j*D + lane]`  (1 VMUL)
  2. warp-reduce-sum (6 × `__shfl_xor` + adds)
  3. `s_j = scale * sum + mask[j]`
  4. online softmax (2 × `__expf`)
  5. `o_reg = α·o_reg + p·v[j*D + lane]`
  6. `l_i = α·l_i + p`

Dependency chain is serialized on the warp reduce → step 3 can't
start until step 2 completes → step 6 feeds back into step 5 for j+1.
No ILP across j iterations. Peak throughput is capped at ~1 wave
instruction every ~30 cycles (vs ideal 4 cycles on gfx906).

More importantly: *K/V loads are not amortised*. Each K/V row is
loaded by `n_head/n_rep × L_q` = 4×1065 blocks for TinyLlama,
~270 KB working set per layer — won't fit in L1 (16 KB). Redundant
loads are the real tax.

rocBLAS GEMM tiles Q and K together (typically 64×64 or 128×64)
and reads each K slab once per output tile, amortising 64× better.

Path to competitive flash-attn
------------------------------
To beat rocBLAS on these shapes we need:

1. **BR > 1**: multiple Q rows per block, sharing the K/V load
   through LDS or L1. BR=4 gives 4× amortisation, BR=8 gives 8×.
   Ballpark: 12 ms/layer → ~3 ms/layer with BR=4, matching rocBLAS.

2. **Split-K for decode** (L_q = 1): grid = (NSPLIT, n_head, B)
   with a merge-kernel reduction. Decode currently has grid size
   `n_head` (= 32) on a 60-CU GPU — parallelism wall. Split-K
   lifts this ceiling.

3. **Causal early-exit** for prefill: for a causal mask, query row q
   doesn't need K rows > q. Saves ~50 % of j iterations on average.

Both (1) and (2) are significant rewrites; parking until after the
3-way bench sweep across gemma4 / qwen3.5 / qwen3.5-moe shows which
models actually spend enough time in attention to justify the
engineering cost. TinyLlama spends ~20 % in attention; qwen-moe
class and long-context workloads are the likely winners.

Correctness
-----------
`cargo test --release --features hip -p candle-transformers
hip_flash_attn_matches_cpu_oracle_d64` — 7/7 cases pass within
`max_abs < 1e-4` of the CPU oracle, covering:
- (1,8,8,4,4,64,false)   — no-GQA prefill
- (1,8,8,4,4,64,true)    — small prefill + mask
- (1,32,8,16,16,64,true) — GQA n_rep=4
- (1,32,4,12,12,64,true) — GQA n_rep=8
- (1,32,32,1,16,64,false)— decode-like L_q=1 × L_k=16
- (2,8,4,8,8,64,true)    — batch=2, n_rep=2
- (1,8,8,1,1149,64,false)— long decode L_k=1149

How to enable
-------------
```
export CANDLE_FLASH_ATTN_ENABLE=1
./target/release/examples/quantized ...
```

When re-enabling by default, remove the `std::env::var` gate in
`candle-transformers/src/models/quantized_llama.rs::forward_attn`
(the `seq_len >= 4` check can stay — L_q=1 decode always loses
with BR=1).
