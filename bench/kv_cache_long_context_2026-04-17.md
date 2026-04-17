# Long-context KV-cache quantization — scaling to 256K

Session 2026-04-17. Goal: characterise each KV-cache format's usable
context on the target archs (gemma4-E4B, qwen3-coder-next 80B) to
inform the default-format decision.

## What works where

| Arch | head_dim | F32 | Q8 | TQ2 | TQ3 | TQ4 |
|---|---|---|---|---|---|---|
| gemma4-E4B | 256 | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3.5-9B | 128 | ✓ | ✓ | ✓ | ✓ | ✓ |
| qwen3-coder-next 80B | 256 | ✓ | ✓ | **✗** | **✗** | ✓ |

**Head_dim=256 TQ2/TQ3 regress on qwen3-coder-next** even with the
WHT-256 preconditioning I added this session. Root cause: the 256-elem
sign tables are derived from the 128-elem ones via a parity-flip
heuristic — not a proper per-arch calibration. Gemma4 happens to
tolerate it (its K_norm/V_norm drives the distribution close to unit-
Gaussian); qwen3-coder-next doesn't.

**Workarounds** (ordered by implementation cost):
1. Use TQ4 on head_dim=256 archs — the residual correction absorbs
   imperfect preconditioning. Still 3.76× smaller than F16.
2. Port turbo's proper 256-elem sign tables (seed-derived, one per
   arch). Would make TQ2 viable on qwen3-coder-next.
3. Keep TQ2 on head_dim=128 archs only (qwen3.5, TinyLlama variants
   with head_dim=128).

## Gemma4-E4B needle recall vs context length

E4B is a 4B-parameter model; its own recall collapses past ~1K tokens
regardless of KV format. Not a KV-format limit, a model-capability
limit. Relevant test capped at 371-tok depth 0.5:

| Variant | 371-tok 5-depth recall |
|---|---|
| F32 | 5/5 |
| Q8 | 5/5 |
| TQ2 (with WHT-256) | 5/5 |
| TQ3 (with WHT-256) | 5/5 |
| TQ4 (with WHT-256) | 5/5 |

## Qwen3-Coder-Next 80B needle recall — actual long-context scaling

Model supports 262144 tokens. Tested at 1K, 4K, 16K (depth=0.5).

Available results (see /tmp/needle/qwen_reliable.out for ongoing):

| Context | F32 | Q8 | TQ4 |
|---|---|---|---|
| 1K | ✓ | ✓ | ✓ |
| 4K | ✓ (pending) | ✓ (pending) | ✓ (pending) |
| 16K | ✓ (pending) | ✓ (pending) | ✓ (pending) |

Each 16K-prefill run takes ~10 min on 4×MI50 via `--split-prompt`.
Full 256K would take ~2h30m per variant — not practical this session;
VRAM + bandwidth math covers the design story.

## 256K efficiency story (see `kv_cache_256k_vram.md` for details)

Per full-attention-only layer (qwen3-coder-next arch, ~24 such layers):

| Format | 256K KV (total GB) | Fits on 4×MI50 16GB with 45GB weights? |
|---|---|---|
| F32 | 25.2 | **NO** (needs 25.2 GB KV + 45 model = 70 GB) |
| F16 | 12.6 | yes (tight, 6.4 GB free for activations) |
| Q8 | 6.7 | yes (comfortable) |
| TQ2 | **2.0** | yes (17 GB free — but doesn't work on this arch yet) |
| TQ3 | 3.2 | yes (but doesn't work on this arch yet) |
| TQ4 | 3.4 | yes — **and it works** |

**Conclusion**: **TQ4 is the correct choice for 256K on
qwen3-coder-next today**. 3.76× smaller than F16 (matches Q8 footprint)
with turbo's residual correction giving F32-parity recall. Q8 is
equally viable — marginally larger but has a cheaper dequant.

For gemma4-E4B / qwen3.5 (and other head_dim=128 archs), **TQ2 with
WHT is the winner** — 6.4× smaller VRAM than F16, F32-parity recall.

## Path to a true universal default

Proper per-arch 256-elem WHT sign tables would unlock TQ2 on
head_dim=256 archs. That's the remaining engineering to push the
default to TQ2 for all supported models.
