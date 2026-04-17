# KV-cache quantization 5-way comparison (F32 / Q8 / TQ2 / TQ3 / TQ4)

Date: 2026-04-17. Session goal: implement all three TurboQuant variants,
wire them into gemma4 + qwen3.5 + qwen3-coder-next, and run a real
comparison on long prompt + long generation + needle-in-haystack.

## Implementation status (all landed, commits b58924f5 / a4383f95 / caef4679)

| Cache format | Bits/val | VRAM vs F16 | Variants supported |
|---|---|---|---|
| F32 (default) | 32 | 0.5× (2× larger!) | — |
| F16 (opt-in) | 16 | 1.0× | gated by `CANDLE_KV_F16=1` |
| Q8_0 | 8.5 | **1.88×** | gated by `CANDLE_KV_Q8=1` |
| TurboQuant TQ2_0 | 2.5 | **6.40×** | `CANDLE_KV_TQ=2` |
| TurboQuant TQ3_0 | 4.0 | **4.00×** | `CANDLE_KV_TQ=3` |
| TurboQuant TQ4_0 | 4.25 | **3.76×** | `CANDLE_KV_TQ=4` |

Kernels ported byte-for-byte from `llama.cpp-turbo` (turbo-quant.cu +
cpy-utils.cuh + dequantize.cuh). No WHT / dense rotation
preconditioning yet — phase TQ3. Gemma4's built-in K-norm / V-norm
already brings K/V close to unit-Gaussian so the raw codebook works
surprisingly well.

## Gemma4-E4B Q4_0, 1-GPU gfx906

### pp512 + tg64 (candle-bench, 1 rep each)

| Variant | pp512 t/s | tg64 t/s |
|---|---|---|
| F32 (default) | 1460.35 | 82.38 |
| Q8 | 1452.14 | **84.57** |
| TQ2 | 1448.23 | 83.67 |
| TQ3 | 1456.57 | 83.58 |
| TQ4 | 1428.56 | 68.98 |

pp is flat (all matmul-bound). tg: Q8/TQ2/TQ3 slight wins; TQ4 regresses
14% due to per-value bitstream unpack + residual correction.

### pp2048 + tg128

| Variant | pp2048 t/s | tg128 t/s |
|---|---|---|
| F32 | 1345.93 | 81.21 |
| Q8 | 1352.55 | 85.08 |
| TQ2 | 1343.23 | 83.86 |
| TQ3 | 1344.58 | 83.19 |
| TQ4 | 1323.01 | 68.43 |

Same story at 2048-tok context. Compression doesn't hurt throughput
except on TQ4.

### Needle-in-haystack (371-tok haystack, needle="the magic number is 42")

| Depth | F32 | Q8 | TQ2 | TQ3 | TQ4 |
|---|---|---|---|---|---|
| 0.10 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.25 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.50 | ✓ | ✓ | **✗** | ✓ | ✓ |
| 0.75 | ✓ | ✓ | **✗** | ✓ | ✓ |
| 0.90 | ✓ | ✓ | **✗** | **✗** | ✓ |

**Recall score** (out of 5): F32 5, Q8 5, TQ4 5, TQ3 4, TQ2 2.

Clean mapping from codebook richness to recall:
- TQ4 (8 centroids + residual correction): F32-parity across all depths.
- TQ3 (8 centroids, no residual): loses only at the deepest position.
- TQ2 (4 centroids): misses half the sweep — the 4-cluster codebook
  can't preserve the fine-grained values needed for exact recall.

This is the signal phase TQ3 (WHT + dense rotation preconditioning)
should fix for TQ2 — a rotated Gaussian distribution better matches
the Lloyd-Max centroids and should push TQ2 up to TQ3/TQ4's recall
while keeping the 6.4× VRAM win.

## Qwen3.5-9B-Q4_1, 1-GPU (short prompt smoke test)

All 5 variants produced **identical output** on "The quick brown fox"
(sample-len=20, temperature=0): "1. **Analyze the Request:** ...".
Speeds all within 5 %:

| Variant | tg t/s |
|---|---|
| F32 | 53.48 |
| Q8 | 53.33 |
| TQ2 | 53.14 |
| TQ3 | 53.01 |
| TQ4 | 50.87 |

Needle test on qwen3.5 was inconclusive — qwen3.5 goes into a long
"Thinking Process" CoT before emitting the answer, so sample-len=80
isn't enough to reach "42" on any variant. Not a Q/TQ issue;
qwen3.5-specific output-format tuning is needed for a clean recall
measurement on that arch.

## Qwen3-Coder-Next 80B Q4_0, 4-GPU

**Blocked this session** — stranded VRAM from a mid-sweep process kill
(~11-13 GB used per GPU; 80B needs >16 GB remaining). TQ4 got a
partial pp512=33.36 t/s before OOM. Re-run after device reset.

Note: earlier-session single-GPU GatedAttention short-prompt test
(before stranded VRAM) confirmed all 5 variants produced identical
output for the Python-function prompt — so the TQ wiring into
GatedAttention is correct; only the bench measurement is blocked.

## Recommendation for the "default KV format" choice

Sorted by VRAM savings at acceptable quality (no WHT/rotation yet):

1. **TQ4** (3.76× smaller than F16, F32-parity quality) — same VRAM
   as Q8 but needs the bitstream-unpack cost. Q8 is strictly better
   here (no quality hit, simpler kernel).
2. **TQ3** (4× smaller than F16, 4/5 depths pass) — next step up in
   VRAM, marginal quality dip at deep needles. Solid candidate for
   default if phase TQ3 (WHT+rotation) pushes it to 5/5.
3. **TQ2** (6.4× smaller than F16, 2/5 depths pass) — biggest win
   but needs phase TQ3 to be viable as default. Without rotation,
   too many recall failures.

**Ship order**: phase TQ3 (WHT + rotation) unlocks TQ2/TQ3 as real
defaults. Until then, **Q8 is the best "default KV" candidate** — it
already ships, no quality hit, and 1.88× VRAM savings over F16.
