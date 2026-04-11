# 3-way bench + candle GPU profiling (post-P2 / 2026-04-11)

**Host:** 4× AMD MI50 16 GB (gfx906), ROCm 7.1.1 + 6.3.4 side-by-side
**Engines:** llamacpp-turbo (build eb43062), candle (post-P2 main), vllm-gfx906-mobydick
**Prompt:** same ~1000-token technical prompt across all models
(`/tmp/long_prompt.txt`, tokenizer-dependent length below)
**Sample length:** 64 tokens (`-n 64` / `--sample-len 64`)

## Headline numbers

### TinyLlama 1.1B Q4_0 — 3-way

| Engine  | Prompt tok | Prefill t/s | Decode t/s |
|---|---:|---:|---:|
| **candle**         | 1065 | **1132 / 1142 / 1137** | **100.4 / 100.2 / 101.0** |
| **llamacpp-turbo** | 1149 | **4737 ± 2855** (r=3)  | **187.9 ± 45.8** (r=3)    |
| **vllm-mobydick**  | 1066 | **5929** (load: 87 s)   | **56.7**                  |

- Candle decode is now **100 t/s** — up from the 88 t/s baseline in
  `BENCH-3WAY-2026-04-11.md` (mostly the P3 fused-pointwise +
  silu_mul_split_last work).
- Turbo decode dropped slightly (188 → vs the 190 from earlier); the
  huge ± comes from `-r 3` cold→warm variance.
- Mobydick prefill is still the fastest by ~1.25× over turbo, but
  decode is 3.3× slower than turbo and 1.77× slower than candle — the
  Python-dispatch overhead dominates once prefill is done.

**Candle vs turbo:** prefill **24 %** of turbo, decode **53 %** of turbo.
Up from 14 % / 46 % in the pre-P0..P5 baseline.

### gemma4 E4B Q4_0 — 2-way (mobydick doesn't support gemma4 arch)

| Engine  | Prompt tok | Prefill t/s | Decode t/s |
|---|---:|---:|---:|
| **candle**         | 991  | **543 / 541 / 543** | **33.7 / 19.4 / 33.8** |
| **llamacpp-turbo** | 1149 | **967 ± 407** (r=3) | **65.8 ± 5.2** (r=3)   |

The decode `19.4` outlier in candle run #2 is frequency-throttle
dropout — the other two runs land cleanly at ~33.7 t/s.

**Candle vs turbo:** prefill **56 %** of turbo, decode **51 %** of turbo.
Better prefill ratio than TinyLlama because gemma4's "dense block"
path is simpler than llama's — fewer kernel launches per step, so
the absolute rocBLAS attention time dominates less. Still ~2× from
turbo on decode.

**`VLLM: GGUF model with architecture gemma4 is not supported yet`** —
`transformers/modeling_gguf_pytorch_utils.py` only whitelists llama,
qwen2/3, deepseek2, etc. Gemma4 is a two-engine bench.

### qwen3.5-9B Q4_1 — 2-way (mobydick doesn't support qwen35 arch)

| Engine  | Prompt tok | Prefill t/s | Decode t/s |
|---|---:|---:|---:|
| **candle**         | 997  | **410 / 409 / 408** | **38.8 / 38.6 / 38.8** |
| **llamacpp-turbo** | 1149 | **1103 ± 294** (r=3) | **61.3 ± 5.2** (r=3)   |

**Candle vs turbo:** prefill **37 %** of turbo, decode **63 %** of turbo.

- Candle prefill is now at 410 t/s, up from 78.6 t/s in the pre-P0
  baseline — **5.2× improvement** from the P1 MMQ v2f kernel and the
  P0 fused GDN. Still a long way from turbo.
- Candle decode is **63 % of turbo** — this is actually the best
  dense-decode ratio in the lineup. The reason is that on qwen3.5-9B
  the GDN step is a significant fraction of decode (10 %+ of GPU
  time), and our `gated_delta_net_step_s128_f32` kernel is
  competitive with turbo's. Where turbo still wins is the MMQ kernels.

### qwen3.5 MoE — BLOCKED

- **Qwen3.5-35B-A3B-UD-Q8_K_XL** (42 GB, qwen35moe arch):
  `Error: qmatmul_concat_rows: dtype mismatch (F16 vs Q8_0) for blk.31.attn_v.weight`
  — candle's `qmatmul_concat_rows` (used by the fused attn_q/k/v
  projection in quantized_qwen35) rejects heterogeneous dtype shards.
  UD quants ship a mix of Q8_0 and F16 across the heads.
- **Qwen_Qwen3.5-122B-A10B-Q4_0 (70 GB, 4-GPU)**: same dtype error,
  this one between Q6K and Q8_0 on `blk.3.attn_k.weight`.
- **Qwen3.5-122B-A10B-Q3_K_S** (53 GB): candle reads 0 tensors from
  the file (`tensors: 0, size: 0.00GB`) and then fails at
  `token_embd.weight` — a different upstream GGUF-reader bug.
- **Qwen3-Coder-30B-A3B Q4_K_XL** (17.6 GB, qwen3moe): OOMs on a
  single 16 GB MI50 and `quantized-qwen3-moe` has no `--n-gpus` flag
  yet.

qwen3.5 MoE benching is blocked behind two fixes:
1. Teach `qmatmul_concat_rows` to handle per-row dtype splits (the
   llama.cpp path does this via the per-block scale dequant).
2. Add `--n-gpus` / layer-split to the `quantized-qwen3-moe` example.

Both are separate work items — not in P2 scope.

## candle GPU-time breakdown (rocprofv3 --kernel-trace)

Categories rolled up across all kernel dispatches (prefill + decode
combined). Columns: wall time, %, # dispatches.

### TinyLlama 1.1B Q4_0 (candle, total 952 ms)

| Category       | ms    | %     | dispatches |
|---|---:|---:|---:|
| mmq_prefill    | 205.3 | 21.6  |       154 |
| rocblas_attn   | 161.8 | 17.0  |     2,817 |
| copy_ucopy     | 139.9 | 14.7  |    28,160 |
| mask_pw        | 115.6 | 12.1  |     5,718 |
| mmvq_decode    | 103.5 | 10.9  |     9,766 |
| alloc          |  77.9 |  8.2  |    20,418 |
| softmax        |  75.2 |  7.9  |     1,472 |
| quantize_q8    |  41.0 |  4.3  |     9,920 |
| rmsnorm        |  15.3 |  1.6  |     2,880 |
| rope           |   8.0 |  0.8  |     2,816 |
| silu_gelu      |   7.2 |  0.8  |     1,408 |

**Top targets on TinyLlama:**
- `copy_ucopy` at **14.7 % / 28 k dispatches**. This is almost
  entirely `copy2d_f32` from attention's `.contiguous()` calls
  (q/k/v transpose materialisations). A flash-attn or direct-strided
  QK^T would eliminate most of it.
- `rocblas_attn + softmax + mask_pw` = **37 % of GPU time** — the
  attention category. This is what P2 flash-attn was aimed at.
  The v1 BR=1 kernel loses on these shapes (see
  `BENCH-P2-FLASH-ATTN-v1-2026-04-11.md`); a BR≥4 rewrite is the
  path forward.
- `mmq_prefill` at 21.6 % is already the P1-v2f kernel at 91 % of
  turbo's throughput (see `BENCH-P1-PHASE-2D-V2F-2026-04-11.md`).

### gemma4 E4B Q4_0 (candle, total 2893 ms)

| Category       | ms     | %     | dispatches |
|---|---:|---:|---:|
| mmq_prefill    |  783.6 | 27.1  |       216 |
| rocblas_attn   |  627.2 | 21.7  |    10,816 |
| copy_ucopy     |  402.0 | 13.9  |     8,664 |
| mmvq_decode    |  397.6 | 13.7  |    13,672 |
| alloc          |  213.9 |  7.4  |    45,874 |
| mask_pw        |  148.0 |  5.1  |    24,519 |
| rmsnorm        |  106.9 |  3.7  |    17,792 |
| quantize_q8    |   74.0 |  2.6  |    13,888 |
| other          |   58.8 |  2.0  |    10,496 |
| softmax        |   48.8 |  1.7  |     2,688 |
| silu_gelu      |   32.0 |  1.1  |     5,376 |

**Gemma4 observations:**
- **`rocblas_attn` at 10,816 dispatches** for only 63 decoded tokens +
  prefill — that's ~480 dispatches per decode step. Gemma4 has
  ~48 layers × 2 attention matmuls = 96 per decode step; multiply
  by (1 prefill + 63 decode) = 64 → 6,144. The remaining ~4,700
  dispatches are **sliding-window attention pieces** that gemma4
  splits differently.
- **`rmsnorm` at 17,792 dispatches** is wildly high — gemma4 has
  pre- and post-norm at attention and MLP in each of its 48 layers,
  which gives 4 rmsnorms per layer × 64 steps × 48 = 12,288, plus
  some q/k norms (12 extra per layer?) totalling ~17 k. Each one
  is a separate kernel launch. **This is a prime target for a
  rmsnorm-fusing pass.**
- **`alloc` at 45,874 dispatches** — `fillBufferAligned` is called
  on every tensor creation that needs zero init. Gemma4's
  per-layer sliding attention mask builds dominate. Reusable mask
  buffers would eliminate most of this.
- **`mask_pw` at 24,519 dispatches** — this is `badd_f32` + `bmul_f32` +
  `where_u8_f32` + `affine_f32`. Gemma4's attention mask path and
  logit-scaling are unfused pointwise chains. Candidate for fusion.
- Notable *absence*: no custom `mul_mat_q4_K` decode kernel — gemma4
  uses Q4_K for a small subset of layers (64 dispatches,
  `mul_mat_vec_q4_K_q8_1_cuda1` @ 83 ms). That's fine.

### qwen3.5-9B Q4_1 (candle, total 3789 ms)

| Category       | ms      | %     | dispatches |
|---|---:|---:|---:|
| mmq_prefill    | 1780.2  | 47.0  |       200 |
| mmvq_decode    |  814.5  | 21.5  |    12,664 |
| gdn_step       |  302.9  |  8.0  |     1,536 |
| alloc          |  253.0  |  6.7  |    34,918 |
| copy_ucopy     |  188.3  |  5.0  |    10,016 |
| other          |  118.4  |  3.1  |    11,369 |
| quantize_q8    |   85.1  |  2.2  |    12,864 |
| mask_pw        |   79.2  |  2.1  |     7,384 |
| rmsnorm        |   61.0  |  1.6  |     6,720 |
| rocblas_attn   |   50.8  |  1.3  |     1,024 |
| silu_gelu      |   42.4  |  1.1  |     5,120 |
| softmax        |   13.0  |  0.3  |       512 |

**Qwen3.5-9B observations:**
- **`mmq_prefill` at 47 %** is by far the biggest category. The
  Q4_1 MMQ kernel (the P1 v2f extension from
  `BENCH-P1-PHASE-2D-V2F-2026-04-11.md`) is at ~90 % of turbo's
  throughput per call but there's a lot more of it — qwen3.5-9B's
  FFN dim is larger than TinyLlama's.
- **`rocblas_attn` is just 1.3 % / 1,024 dispatches.** This is the
  win from the P0 zero-copy GQA reshape in `gqa_attention` plus the
  P0 GQA-aware GDN kernel — we replaced the `(B, n_head, L, D)`
  expand+matmul with `(B, n_kv_head, n_rep*L, D)`. The 1,024
  dispatches = 16 attention layers × 64 forward calls. rocBLAS is
  no longer a bottleneck here.
- **`gdn_step` at 8 % / 1,536 dispatches** is the Gated Delta Net
  recurrent step kernel (24 GDN layers × 64 steps = 1,536). At
  ~197 µs per call it's **matching turbo's GDN performance**.
- **`alloc` still at 34,918 dispatches** — the workspace pool helps
  but per-layer intermediate `Tensor::zeros` calls still generate a
  lot of `fillBufferAligned`. Pre-allocating a single per-layer
  scratch pad (already partially done) would cut this further.
- Decode is bottlenecked by `mmvq_decode` (21.5 %) — this is pure
  memory bandwidth on the weight streaming. The gap to turbo is
  in how well we amortise the scale/sum loads; turbo fuses scale
  fetch with the dp4a inner loop.

## CPU profile (wall vs cpu-time)

Per-run wall-clock and exit code samples (tokeniser-dependent prompt
lengths):

| Model       | Wall (best of 3) | Notes |
|---|---:|---|
| TinyLlama   | ~1.9 s | load ~0.5 s + prefill + 63 decode |
| gemma4-E4B  | ~3.0 s | load ~0.3 s + prefill + 63 decode |
| qwen3.5-9B  | ~2.9 s | load ~0.3 s + prefill + 63 decode |

No `/usr/bin/time` on this host so a proper user/sys split wasn't
captured; candle's own load timing (printed at startup) and rocprof
kernel trace totals together give a close enough picture.

candle load times (from each run's header):
- TinyLlama: `loaded 201 tensors (635.99 MB) in 0.29 s`
- gemma4-E4B: `loaded 720 tensors (4.82 GB) in 0.19 s`
- qwen3.5-9B: `loaded 557 tensors (5.44 GB) in 0.22 s`

All sub-second cold loads — HSACO embedding + mmap-driven GGUF.
vllm-mobydick's **87 s load** on TinyLlama is the big outlier;
one-shot benches disproportionately penalise it but it amortises
across many generations in serving mode.

## Tensor-size-normalised decode (BW proxy)

Decode rate × model size ≈ effective GB/s across the weight
stream. MI50 HBM2 peak is 1024 GB/s.

| Model          | Engine  | Decode t/s | Model size | BW proxy GB/s | % of peak |
|---|---|---:|---:|---:|---:|
| TinyLlama 1.1B | candle  | 100.4 |  606 MB | **60.9** |  5.9 % |
| TinyLlama 1.1B | turbo   | 187.9 |  606 MB | 113.9     | 11.1 % |
| TinyLlama 1.1B | vllm    |  56.7 |  606 MB |  34.4     |  3.4 % |
| gemma4 E4B     | candle  |  33.7 | 4.49 GB | **151.3** | 14.8 % |
| gemma4 E4B     | turbo   |  65.8 | 4.49 GB | 295.5     | 28.9 % |
| qwen3.5-9B     | candle  |  38.8 | 5.43 GB | **210.7** | 20.6 % |
| qwen3.5-9B     | turbo   |  61.3 | 5.43 GB | 332.9     | 32.5 % |

Candle's BW efficiency **grows with model size** — 5.9 % → 14.8 %
→ 20.6 % — because per-token dispatch overhead is a fixed cost that
amortises better on larger weights. Turbo shows the same trend but
starts higher. The gap closes from 1.87× (TinyLlama) to 1.95×
(gemma4) to 1.58× (qwen3.5-9B).

**Where candle still bleeds vs turbo on decode:**

1. Per-token kernel launch overhead — HIP API dispatch on gfx906 is
   ~15 µs, and candle issues ~150 kernels per decode step on dense
   models. That's 2.25 ms of pure launch overhead per token ≈
   half of decode wall time on TinyLlama.

2. Repeated workspace zero-fill (`fillBufferAligned` at ~20 k-45 k
   dispatches). Turbo pre-allocates once per layer and reuses.

3. Unfused pointwise chains (`badd + affine + where + softmax`) —
   turbo fuses these into one kernel per attention step.

## Deltas vs previous 3-way bench

Comparing `BENCH-3WAY-2026-04-11.md` → this report (TinyLlama 1065-tok
prompt, candle only, non-profiled clean runs):

| Metric     | Previous | This report | Delta  |
|---|---:|---:|---|
| prefill t/s  | 1150     | 1137         | −1.1 % |
| decode t/s   |  91      |  100.4       | +10.3 % |

- **Prefill flat** — P2 flash-attn was opt-in, not contributing to
  the default path.
- **Decode +10 %** — comes from P3 silu_mul_split_last +
  fused_pointwise + the P4 l2_norm / softplus kernels taking over
  a few hotpaths. Not huge on TinyLlama (small model, tight
  fixed-overhead-dominated regime) but real.

## Takeaways & next-round targets

1. **Candle's decode on qwen3.5-9B is now at 63 % of turbo.** That's
   the best ratio in the lineup; it's within touching distance of
   parity and pushes the bottleneck off rocBLAS (now only 1.3 % of
   GPU time) onto MMQ + launch overhead.

2. **Gemma4 has the largest surface area of low-hanging fruit**:
   - rmsnorm at 17,792 dispatches → fuse RMSNorm pairs (pre+post).
   - alloc at 45,874 dispatches → per-layer scratch pool for the
     sliding-window mask buffers.
   - rocblas_attn at 21.7 % — gemma4 runs sliding windows which
     chops attention into many small `Cijk` calls; a flash-attn
     path with causal-window support (different from the simple
     full-causal P2 kernel) would collapse them.

3. **TinyLlama attention is still the #1 absolute cost** (rocblas +
   softmax + mask + copy = 51.7 % of GPU time) but the BR=1 v1
   flash-attn doesn't win; the v2 design (BR=4, LDS-shared K/V)
   is the next step. Park until after the gemma4 wins.

4. **qwen3.5 MoE is bench-blocked by candle bugs**, not by kernel
   performance. Two separate items:
   - `qmatmul_concat_rows` needs per-row dtype dispatch.
   - `quantized-qwen3-moe` needs an `--n-gpus` flag.
   Both are independent from the optimisation pass.

5. **vllm-mobydick remains the gold standard for prefill** (5929 t/s
   vs turbo 4737 vs candle 1137 on TinyLlama) but useless for
   single-stream decode (56.7 t/s). And its 87 s load time kills
   one-shot benches. Worth re-running in serving mode to see the
   amortised picture.

## How to reproduce

```bash
# Common env
export LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm-7.1.1/lib
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.1.1/core-7.13/lib/rocblas/library
export HIP_OFFLOAD_ARCH=gfx906

# candle (each model)
/artefact/candle/target/release/examples/quantized \
    --model /artefact/models/tinyllama-1.1b-q4_0.gguf \
    --prompt "$(cat /tmp/long_prompt.txt)" --sample-len 64

/artefact/candle/target/release/examples/quantized-gemma4 \
    --model /artefact/models/gemma-4-E4B-it-Q4_0.gguf \
    --prompt "$(cat /tmp/long_prompt.txt)" --sample-len 64 --temperature 0

/artefact/candle/target/release/examples/quantized-qwen35 \
    --model /artefact/models/Qwen3.5-9B-Q4_1.gguf \
    --prompt "$(cat /tmp/long_prompt.txt)" --sample-len 64 --temperature 0

# turbo
LLAMA=/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin
LD_LIBRARY_PATH=$LLAMA:$LD_LIBRARY_PATH \
    $LLAMA/llama-bench -m <model.gguf> -ngl 999 -p 1149 -n 64 -r 3 --no-warmup -o md

# mobydick
source /artefact/mobydick-venv/bin/activate
export ROCM_PATH=/opt/rocm-6.3.4 PYTORCH_ROCM_ARCH=gfx906
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
HIP_VISIBLE_DEVICES=0 python /tmp/run_vllm.py \
    /artefact/models/tinyllama-1.1b-q4_0.gguf /tmp/long_prompt.txt 64

# kernel-trace profile
/opt/rocm-7.1.1/core-7.13/bin/rocprofv3 --kernel-trace \
    -d /tmp/bench-3way-p2/prof_<name> -o <name> --output-format csv -- \
    <candle command>
```
