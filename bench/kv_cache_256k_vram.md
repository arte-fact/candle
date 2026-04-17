# 256K-context KV cache VRAM analysis

Target: Qwen3-Coder-Next 80B (262144-token context, the design goal
for "use full 256K context efficiently"). Arch: block_count=48 layers,
hybrid of GDN + full-attention blocks. Full-attention layers carry the
quadratic KV cost; GDN layers keep a small fixed recurrent state.

## Assumptions

- Full-attention layer count: **≈ 24** (conservative; the hybrid
  repeats a 3-GDN / 3-full pattern through all 48 blocks).
- Per full-attn layer: `H_kv=2`, `head_dim=256`, `T=262144`.
- K and V each: `B × H_kv × T × D = 1 × 2 × 262144 × 256 = 134M elements`.
- K+V per layer: 268M elements.

## Per-variant total KV footprint at 256K

| Format | Bytes/val | MB/layer (K+V) | GB for 24 layers |
|---|---|---|---|
| F32 | 4.0 | 1073 | **25.2** |
| F16 | 2.0 | 536 | **12.6** |
| Q8_0 | 1.0625 | 284 | **6.7** |
| TQ2_0 | 0.3125 | **84** | **2.0** |
| TQ3_0 | 0.50 | 134 | 3.2 |
| TQ4_0 | 0.53125 | 143 | 3.4 |

## Fit check on 4× MI50 16GB (64GB total) with 45GB model weights

Free VRAM: 64 − 45 = **19 GB**.

| Format | 256K KV | Fits? | Remaining for activations |
|---|---|---|---|
| F32 | 25.2 GB | **NO** | — |
| F16 | 12.6 GB | yes | 6.4 GB (tight) |
| Q8_0 | 6.7 GB | yes | 12.3 GB (comfortable) |
| TQ2_0 | **2.0 GB** | yes | **17.0 GB** (plenty) |
| TQ3_0 | 3.2 GB | yes | 15.8 GB |
| TQ4_0 | 3.4 GB | yes | 15.6 GB |

**F32 KV can't fit at 256K on this hardware.** F16 fits but leaves
little room for activations. Q8 is comfortable. TQ2 is the only
format with serious headroom for batching / multi-session use.

## Decode-time bandwidth per step at 256K

Each decode step reads the full KV cache for attention (once). On
gfx906 MI50 each card has ~900 GB/s HBM2:

| Format | Bytes/step per card | ms/step @ 900 GB/s |
|---|---|---|
| F16 | 3.15 GB | 3.50 ms |
| Q8_0 | 1.67 GB | 1.86 ms |
| TQ2_0 | **0.50 GB** | **0.55 ms** |

So at 256K context, **TQ2 attention reads are ~6.3× faster than F16**.
With dequant + WHT adding ~30 ops/val, the compute side is tiny
relative to the bandwidth win — the attention step should indeed
scale with the compressed size.

## Practical note on testing to 256K

Full 256K prefill at current speeds (~30 t/s on 4× MI50 with
`--split-prompt`): `262144 / 30 ≈ 2h30m per run`. Real-world usage
would feed large prompts once and decode for a long while, so the
prefill cost amortises — but for this bench, a 256K recall test
would take ~30 min per variant × 5 variants = 2-3 h. Scaled the
sweep to 16K max this session; the **VRAM and bandwidth math above
is the efficiency story** that doesn't need a 256K run to be valid.
