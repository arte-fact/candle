//! Phase A4 — central attention-backend selection table.
//!
//! Today candle's attention dispatch is spread across several sites with
//! their own env-var switches:
//!
//!   * `CANDLE_FLASH_V2_MAX_LK`   — `t_k ≤ v2_threshold` picks flash-attn
//!     v2 over rocBLAS GEMM; default `512`, bumped to `4096` under
//!     `CANDLE_G2_REPLAY=1` so the captured plan includes attention.
//!   * `CANDLE_FLASH_SPLIT_LK`    — Phase-R1 split-L_k two-kernel design
//!     (chunk + combine).  Default on; `=0` falls back to the monolithic
//!     kernel.
//!   * `CANDLE_FLASH_ATTN_V2_ENABLE` — legacy v1 vs v2 switch.
//!   * `FLASH_L_K_ITER_OVERRIDE`  — thread-local L_k override for G2
//!     dyn-lk (Phase L).
//!
//! This module does NOT yet replace those sites — that's a multi-day
//! migration.  It surfaces a single `select()` function that encodes the
//! current implicit policy, so new call sites and benches can reason
//! about "what backend WOULD fire here?" without replicating the scattered
//! logic.  Existing sites can migrate incrementally.
//!
//! Reference: Aphrodite-engine's `v1/attention/backends/` selects between
//! {flash_attn, triton_attn, paged_attention, rocm_aiter_*} via a similar
//! `(head_dim, kv_len, uniform_decode, dtype)` dispatch.

/// Attention backend candidates candle currently dispatches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnBackend {
    /// rocBLAS GEMM-based attention (fallback for long context / non-
    /// matching head_dim).
    RocBlas,
    /// Flash-attention v2 monolithic kernel — `flash_attn_v2_fwd_*`.
    FlashV2,
    /// Flash-attention v2 K-transposed variant — `flash_attn_v2_fwd_ktvs_*`.
    FlashV2Kt,
    /// Phase-R1 split-L_k two-kernel flash-attn (chunk + combine).
    FlashSplitLk,
    /// GQA mat-vec decode kernel — `gqa_decode_mv_fast_d{256,512}_f32`.
    GqaDecodeMvFast,
    /// GQA mat-vec decode with device-resident counter slot (T2) —
    /// `gqa_decode_mv_fast_d{256,512}_f32_ctr`.
    GqaDecodeMvFastCtr,
}

/// Inputs the selector keys on.  All the fields the scattered policy
/// reads today are collected here.  Add to this as migration proceeds.
#[derive(Debug, Clone, Copy)]
pub struct AttnCtx {
    /// Per-head dimension (64 / 128 / 256 / 512 for supported kernels).
    pub head_dim: usize,
    /// KV length at this call (post-pad when G2 n_kv padding is in effect).
    pub n_kv: usize,
    /// Effective L_k iteration count (may be shorter than `n_kv` under
    /// G2 dyn-lk).  Usually `index_pos + 1` during decode.
    pub l_k_iter: usize,
    /// Query batch × sequence (1 for decode, prompt length for prefill).
    pub batch_size: usize,
    /// Whether this is the sliding-window attention variant.
    pub is_swa: bool,
    /// Whether K is T-major (K_TRANS=true) vs canonical.
    pub is_kt: bool,
    /// Is this being called inside a G2-captured plan?
    pub g2_replay: bool,
}

impl AttnCtx {
    /// Decode calls have `batch_size == 1`.
    pub fn is_decode(&self) -> bool {
        self.batch_size == 1
    }
}

/// Pick a backend for the given context.  Encodes the current implicit
/// policy — call sites that don't yet use this can drift, but the
/// selector documents the intended table.
///
/// Today's default chain:
///   1. Decode + head_dim ∈ {256, 512} + T-major K  → GqaDecodeMvFast(Ctr).
///      Fires under G2_REPLAY for the `_ctr` variant.
///   2. Prefill (batch > 1) with `t_k ≤ v2_threshold` (from
///      `CANDLE_FLASH_V2_MAX_LK`, default 512, G2:4096) and head_dim in
///      {64, 128, 256, 512} → Flash (Split-L_k when enabled).
///   3. Otherwise → RocBlas.
pub fn select(ctx: AttnCtx) -> AttnBackend {
    let v2_threshold: usize = std::env::var("CANDLE_FLASH_V2_MAX_LK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(if ctx.g2_replay { 4096 } else { 512 });
    let split_lk_on = std::env::var("CANDLE_FLASH_SPLIT_LK")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(true);

    // 1. decode mat-vec fast path (head_dim 256/512 only).
    if ctx.is_decode()
        && matches!(ctx.head_dim, 256 | 512)
        && ctx.is_kt
    {
        return if ctx.g2_replay {
            AttnBackend::GqaDecodeMvFastCtr
        } else {
            AttnBackend::GqaDecodeMvFast
        };
    }

    // 2. flash-attn v2 family.
    if ctx.n_kv <= v2_threshold
        && matches!(ctx.head_dim, 64 | 128 | 256 | 512)
    {
        if split_lk_on && !ctx.is_kt && ctx.n_kv >= 64 {
            return AttnBackend::FlashSplitLk;
        }
        return if ctx.is_kt {
            AttnBackend::FlashV2Kt
        } else {
            AttnBackend::FlashV2
        };
    }

    // 3. fallback.
    AttnBackend::RocBlas
}

/// Optional trace — `CANDLE_ATTN_BACKEND_TRACE=1` logs each selection.
/// Call at the dispatch site after `select()` to check that the new
/// central policy matches the ad-hoc behaviour at that site.
pub fn trace(ctx: AttnCtx, picked: AttnBackend) {
    if std::env::var("CANDLE_ATTN_BACKEND_TRACE").is_ok() {
        eprintln!(
            "[attn-backend] picked {:?} for head_dim={} n_kv={} l_k_iter={} b={} swa={} kt={} g2={}",
            picked, ctx.head_dim, ctx.n_kv, ctx.l_k_iter,
            ctx.batch_size, ctx.is_swa, ctx.is_kt, ctx.g2_replay,
        );
    }
}
