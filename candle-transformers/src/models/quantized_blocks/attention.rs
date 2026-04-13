//! Attention block variants for quantized GGUF models.
//!
//! - [`StandardAttention`]: GQA with separate Q/K/V, optional QK norms, optional V norm
//! - [`GatedAttention`]: Q+gate fused projection, sigmoid-gated output (qwen35 full-attn)

use super::gguf_config::GgufConfig;
use super::gguf_loader::Gguf;
use super::norms::GemmaRmsNorm;
use super::rope::RotaryEmbedding;
use super::super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::{Module, Result, Tensor, D};
use candle_nn::kv_cache::{ConcatKvCache, KvCache};
use std::sync::Arc;

/// Initial KV cache capacity in tokens. The cache grows by this many
/// tokens at a time once full — for typical chat workloads (≤4K
/// context) it never grows past the initial allocation, eliminating
/// the per-step `Tensor::cat` from `ConcatKvCache` which would
/// otherwise allocate a new buffer of size `cache_len + 1` every
/// decode step (turning into O(N²) memory traffic on long contexts
/// and defeating the workspace pool — each new size hits its own
/// bucket exactly once).
const KV_CACHE_INITIAL: usize = 4096;

/// Run scaled-dot-product attention with GQA sharing done via zero-copy
/// reshape of Q, not a physical broadcast of K/V.
///
/// For interleaved GQA (the convention candle uses everywhere in this module)
/// Q head `h` attends to KV head `h / n_rep`, so reshaping Q from
/// `(B, n_head, L, D)` to `(B, n_kv_head, n_rep * L, D)` groups together all
/// the Q sub-heads that share a given KV head. A plain batched matmul against
/// K shape `(B, n_kv_head, T, D)` then gives exactly the right result — each
/// rocBLAS batch matrix handles one KV head at a time and the `n_rep` Q
/// sub-heads for that KV head live contiguously in the row dimension.
///
/// No K/V copy is performed. This replaces the earlier `broadcast_kv` helper
/// which materialised `(B, n_head, T, D)` via `unsqueeze → expand → reshape →
/// contiguous`, followed by a `k.t().contiguous()` that made a second copy of
/// the same size. Both are gone; only one `k.t().contiguous()` on the
/// `(B, n_kv_head, D, T)` view remains, which is `n_rep×` smaller.
///
/// Cross-reference: llama.cpp divides `zt / channel_ratio` in its MMQ offset
/// calc (`ggml/src/ggml-cuda/mmq.cuh:3640`); vLLM launches a grid per query
/// head and computes `kv_head_idx = head_idx / num_queries_per_kv`
/// (`csrc/attention/attention_kernels.cuh:145-148`). Our approach is closest
/// to vLLM's — we get the sharing "for free" by reshaping Q rather than
/// teaching the kernel about GQA.
///
/// The reshape round-trips are zero-copy when the inputs are contiguous:
/// `Tensor::reshape` returns a view in that case (see
/// `candle-core/src/tensor.rs:2547-2557`).
///
/// Shapes:
/// - `q`: `(B, n_head, L, D)`
/// - `k`: `(B, n_kv_head, T, D)`
/// - `v`: `(B, n_kv_head, T, D)`
/// - `mask`: optional broadcast-compatible with `(B, n_head, L, T)`
///
/// Returns attention output with shape `(B, n_head, L, D)`.
pub(crate) fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attn_scale: f64,
) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, head_dim) = q.dims4()?;
    let (_, n_kv_head, t_k, _) = k.dims4()?;
    debug_assert_eq!(n_head % n_kv_head, 0, "n_head must be divisible by n_kv_head");
    let n_rep = n_head / n_kv_head;

    // Flash-attention v2 with DPP warp reductions + SFU exp/rcp.
    // Opt-in via CANDLE_FLASH_ATTN_V2_ENABLE=1. The fused kernel is
    // correct but currently ~15-50% slower per-call than rocBLAS on
    // gfx906 due to the per-j warp reduction bottleneck. The dispatch
    // reduction benefit (10k+ → 48 on gemma4 sliding window) can
    // outweigh the per-call penalty at very long sequences.
    #[cfg(feature = "hip")]
    {
        if std::env::var("CANDLE_FLASH_ATTN_V2_ENABLE").is_ok()
            && seq_len >= 4
            && matches!(head_dim, 64 | 128 | 256)
            && matches!(q.device(), candle::Device::Hip(_))
            && q.dtype() == candle::DType::F32
            && mask
                .map(|m| m.dtype() == candle::DType::F32)
                .unwrap_or(true)
        {
            // Force all four tensors contiguous — KvCache narrow
            // views are strided when the cache buffer is over-
            // allocated.
            let q_c = if q.is_contiguous() { q.clone() } else { q.contiguous()? };
            let k_c = if k.is_contiguous() { k.clone() } else { k.contiguous()? };
            let v_c = if v.is_contiguous() { v.clone() } else { v.contiguous()? };
            let mask_c = match mask {
                None => None,
                Some(m) => Some(if m.is_contiguous() {
                    m.clone()
                } else {
                    m.contiguous()?
                }),
            };
            if let Ok(o) = candle::hip_backend::flash_attn_v2_fused(
                &q_c,
                &k_c,
                &v_c,
                mask_c.as_ref(),
                attn_scale,
            ) {
                return Ok(o);
            }
        }
    }

    // Materialise k as (B, n_kv_head, D, T) for the Q·K^T matmul. This
    // is the big "hot" transpose that used to show up per call; Attack C
    // (`gqa_attention_k_transposed` below) skips this entirely when the
    // caller already has K in the transposed layout (e.g. via a
    // pre-transposed KvCache).
    //
    // Empirically on gfx906: dropping the `.contiguous()` and relying
    // on `gemm_config`'s stride-detected `GemmOp::Trans` path (which
    // IS supported) regresses prefill by 7 % because rocBLAS/Tensile
    // picks a different (slower) kernel for the transposed case on
    // our prefill shapes. Keep the materialisation on the fallback
    // path.
    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    gqa_attention_inner(q, &k_t, v, mask, attn_scale, b_sz, n_head, n_kv_head, n_rep, seq_len, t_k, head_dim)
}

/// Attack C entry point: `gqa_attention` with K **already transposed**
/// to `(B, n_kv_head, D, T)`. Skips the internal
/// `k.transpose().contiguous()` materialisation, which dominates decode
/// when the cache grows (O(T) copy per step → O(T²) per generation).
///
/// Caller invariant: `k` is `(B, n_kv_head, D, T)` and contiguous. V is
/// the usual `(B, n_kv_head, T, D)` layout. `mask` same rules as in
/// [`gqa_attention`].
pub(crate) fn gqa_attention_k_transposed(
    q: &Tensor,
    k_t: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attn_scale: f64,
) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, head_dim) = q.dims4()?;
    let (kb, kkv, kd, t_k) = k_t.dims4()?;
    if (kb, kkv, kd) != (b_sz, q.dim(1)? / (q.dim(1)? / kkv), head_dim) {
        candle::bail!(
            "gqa_attention_k_transposed: k_t shape {:?} incompatible with q {:?}",
            k_t.dims(),
            q.dims()
        );
    }
    let n_kv_head = kkv;
    debug_assert_eq!(n_head % n_kv_head, 0, "n_head must be divisible by n_kv_head");
    let n_rep = n_head / n_kv_head;

    // G1: Strided decode/prefill attention — fused kernel that reads K/V
    // via explicit strides from KvCache pre-allocated buffers.
    // Uses the v2 LDS-tiled kernel with V stride support for both
    // decode (L_q=1) and prefill (L_q>=4).
    // G1: Strided flash attention dispatch.
    // - Prefill (seq_len >= 4): v2 LDS-tiled kernel with strided V (default).
    // - Decode (L_q=1): rocBLAS via the inner fallback (default). Strided
    //   flash-attn decode kernels lose to rocBLAS on gfx906 for K/V in
    //   transposed layout — the per-thread access pattern across D is
    //   stride-maxT (uncoalesced). The split-K kernel is kept as opt-in
    //   via CANDLE_SPLITK=1 for experimentation.
    #[cfg(feature = "hip")]
    {
        let strided_off = std::env::var("CANDLE_FLASH_STRIDED_OFF").is_ok();
        // v2's LDS-tiled kernel beats rocBLAS for narrow T (<=256) but
        // loses badly at wide T (the BC=64 outer loop isn't unrolled and
        // occupancy is limited). Fall through to rocBLAS when L_k is
        // large regardless of V contiguity.
        // v2's LDS-tiled kernel is typically faster than rocBLAS for
        // narrow L_k (<=128 observed), but rocBLAS wins for wider T —
        // v2's BC=64 outer loop isn't unrolled and occupancy is limited.
        // When G2 replay is active, we pad L_k to 256 so rocBLAS becomes
        // the decisive winner; turn v2 off entirely in that mode.
        let v2_threshold: usize = if std::env::var("CANDLE_G2_REPLAY").is_ok() {
            0 // skip v2 → always rocBLAS
        } else {
            std::env::var("CANDLE_FLASH_V2_MAX_LK")
                .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(512)
        };
        if !strided_off
            && !v.is_contiguous()
            && t_k <= v2_threshold
            && matches!(head_dim, 64 | 128)
            && matches!(q.device(), candle::Device::Hip(_))
            && q.dtype() == candle::DType::F32
            && mask.map(|m| m.dtype() == candle::DType::F32).unwrap_or(true)
        {
            let q_c = if q.is_contiguous() { q.clone() } else { q.contiguous()? };
            let mask_c = match mask {
                None => None,
                Some(m) => Some(if m.is_contiguous() { m.clone() } else { m.contiguous()? }),
            };

            // v2 LDS-tiled kernel for both prefill (seq_len >= 4) and decode
            // (L_q=1 handled via q_in_range guard). This replaces rocBLAS on
            // the attention hot path — critical for G2/G3 replay which needs
            // a stable kernel set (rocBLAS Tensile picks different kernels
            // as GEMM dims grow past thresholds, breaking graph replay).
            let kt_c = if k_t.is_contiguous() { k_t.clone() } else { k_t.contiguous()? };
            if let Ok(o) = candle::hip_backend::flash_attn_v2_kt_strided_v(
                &q_c, &kt_c, v, mask_c.as_ref(), attn_scale, t_k,
            ) {
                return Ok(o);
            }
            if std::env::var("CANDLE_SPLITK").is_ok() {
                if let Ok(o) = candle::hip_backend::flash_attn_decode_strided_split_k(
                    &q_c, k_t, v, mask_c.as_ref(), attn_scale, t_k,
                ) {
                    return Ok(o);
                }
            }
        }
    }

    // Flash-attention v2 with native K-transposed support.
    // Opt-in via CANDLE_FLASH_ATTN_V2_ENABLE=1.
    #[cfg(feature = "hip")]
    {
        if std::env::var("CANDLE_FLASH_ATTN_V2_ENABLE").is_ok()
            && seq_len >= 4
            && matches!(head_dim, 64 | 128 | 256)
            && matches!(q.device(), candle::Device::Hip(_))
            && q.dtype() == candle::DType::F32
            && mask
                .map(|m| m.dtype() == candle::DType::F32)
                .unwrap_or(true)
        {
            let q_c = if q.is_contiguous() { q.clone() } else { q.contiguous()? };
            let kt_c = if k_t.is_contiguous() { k_t.clone() } else { k_t.contiguous()? };
            // Flash attn kernel requires contiguous V; rocBLAS handles strided.
            let v_c = if v.is_contiguous() { v.clone() } else { v.contiguous()? };
            let mask_c = match mask {
                None => None,
                Some(m) => Some(if m.is_contiguous() { m.clone() } else { m.contiguous()? }),
            };
            if let Ok(o) = candle::hip_backend::flash_attn_v2_kt_fused(
                &q_c, &kt_c, &v_c, mask_c.as_ref(), attn_scale,
            ) {
                return Ok(o);
            }
        }
    }

    // k_t is assumed already contiguous; if not, materialise (this is a
    // one-time transition cost that rocBLAS batched gemm needs).
    let k_t_owned;
    let k_t_ref: &Tensor = if k_t.is_contiguous() {
        k_t
    } else {
        k_t_owned = k_t.contiguous()?;
        &k_t_owned
    };
    gqa_attention_inner(
        q, k_t_ref, v, mask, attn_scale, b_sz, n_head, n_kv_head, n_rep, seq_len, t_k, head_dim,
    )
}

/// Shared inner computation used by both `gqa_attention` and
/// `gqa_attention_k_transposed`. Assumes `k_t` is the already-transposed
/// K tensor of shape `(B, n_kv_head, D, T)`.
#[allow(clippy::too_many_arguments)]
fn gqa_attention_inner(
    q: &Tensor,
    k_t: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attn_scale: f64,
    b_sz: usize,
    n_head: usize,
    n_kv_head: usize,
    n_rep: usize,
    seq_len: usize,
    t_k: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // Zero-copy reshape: Q (B, n_head, L, D) → (B, n_kv_head, n_rep*L, D).
    // When `n_rep == 1` this is a no-op rename; otherwise it groups the
    // n_rep Q sub-heads that share each KV head into one batch matrix row.
    let q_grouped = q.reshape((b_sz, n_kv_head, n_rep * seq_len, head_dim))?;

    // Batched matmul: batch = B * n_kv_head, each is (n_rep*L, D) × (D, T).
    let attn_weights = q_grouped.matmul(k_t)?;

    // Un-flatten for scale + mask + softmax: (B, n_kv_head, n_rep*L, T) →
    // (B, n_head, L, T). Zero-copy because the matmul output is contiguous.
    let attn_weights = attn_weights.reshape((b_sz, n_head, seq_len, t_k))?;

    // Q0c: fuse `scale + mask_add + softmax_last_dim` into a single HIP
    // launch when preconditions allow. The fused kernel only supports
    // f32 + contiguous + broadcast-compatible (B|1, 1, L_q|1, L_k) masks.
    // Everything else falls through to the unfused tensor-op chain.
    let attn_weights = {
        #[cfg(feature = "hip")]
        {
            let mask_ok = mask
                .map(|m| {
                    m.is_contiguous()
                        && m.dtype() == candle::DType::F32
                        && matches!(m.device(), candle::Device::Hip(_))
                })
                .unwrap_or(true);
            if matches!(attn_weights.device(), candle::Device::Hip(_))
                && attn_weights.dtype() == candle::DType::F32
                && attn_weights.is_contiguous()
                && mask_ok
            {
                candle::hip_backend::masked_softmax_scale_fused(&attn_weights, mask, attn_scale)?
            } else {
                let s = (attn_weights * attn_scale)?;
                let m = match mask {
                    Some(m) => s.broadcast_add(m)?,
                    None => s,
                };
                candle_nn::ops::softmax_last_dim(&m)?
            }
        }
        #[cfg(not(feature = "hip"))]
        {
            let s = (attn_weights * attn_scale)?;
            let m = match mask {
                Some(m) => s.broadcast_add(m)?,
                None => s,
            };
            candle_nn::ops::softmax_last_dim(&m)?
        }
    };

    // Re-group for V matmul: (B, n_head, L, T) → (B, n_kv_head, n_rep*L, T).
    let attn_weights_grouped =
        attn_weights.reshape((b_sz, n_kv_head, n_rep * seq_len, t_k))?;

    let attn_output = if v.is_contiguous() {
        attn_weights_grouped.matmul(v)?
    } else {
        // V is non-contiguous (KvCache narrow+transpose). Make contiguous.
        let v_c = v.contiguous()?;
        attn_weights_grouped.matmul(&v_c)?
    };

    // Un-flatten back to (B, n_head, L, D) for the caller.
    attn_output.reshape((b_sz, n_head, seq_len, head_dim))
}

/// Variants of RmsNorm used by different model families for QK norms.
#[derive(Debug, Clone)]
pub enum AttnNorm {
    /// Standard RmsNorm (Llama, Qwen)
    Standard(RmsNorm),
    /// Gemma-style RmsNorm with +1 offset on the weight
    Gemma(GemmaRmsNorm),
}

impl AttnNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            AttnNorm::Standard(n) => n.forward(x),
            AttnNorm::Gemma(n) => n.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// StandardAttention
// ---------------------------------------------------------------------------

/// Standard grouped-query attention with separate Q/K/V projections.
///
/// Supports:
/// - Optional QK norms (gemma4, qwen3)
/// - Parameter-free V norm (gemma4)
/// - Per-layer rotary embedding (per-layer head_dim, freq_base, freq_factors)
/// - Configurable attention scale (gemma4 uses 1.0)
/// - Shared-KV layers (qwen35moe layers 24..41, gemma4-31B): `wk`/`wv` may be `None`
///   and the caller is expected to pass borrowed `(K, V)` to [`Self::forward_with_kv`].
///
/// **No internal KV cache** — the cache lives at the model level so cross-layer
/// borrows for shared-KV layers can read another layer's cache without lifetime
/// gymnastics. Each forward pass receives the cumulative `(K, V)` to attend over.
pub struct StandardAttention {
    wq: QMatMul,
    wk: Option<QMatMul>,
    wv: Option<QMatMul>,
    wo: QMatMul,
    q_norm: Option<AttnNorm>,
    k_norm: Option<AttnNorm>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub attn_scale: f64,
    v_norm_eps: Option<f64>,
    /// Per-layer RoPE — owned, not Arc'd, so each layer can have its own
    /// `freq_base`/`freq_factors`/`dim` (gemma4 sliding vs global, etc.).
    rotary: Arc<RotaryEmbedding>,
}

/// Configuration for loading a StandardAttention layer.
#[derive(Debug, Clone, Copy)]
pub struct StandardAttentionOpts {
    /// Use parameter-free V norm (gemma4)
    pub use_v_norm: bool,
    /// Use Gemma-style RmsNorm with +1 offset for q_norm/k_norm (gemma4)
    pub use_gemma_norms: bool,
    /// Attention scale; if None, defaults to 1/sqrt(head_dim).
    /// gemma4 sets this to 1.0 (no pre-attn scaling).
    pub attention_scale: Option<f64>,
}

impl Default for StandardAttentionOpts {
    fn default() -> Self {
        Self {
            use_v_norm: false,
            use_gemma_norms: false,
            attention_scale: None,
        }
    }
}

impl StandardAttention {
    /// Load from GGUF with default options (Llama/Qwen style).
    pub fn load(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
        layer_idx: usize,
        rotary: Arc<RotaryEmbedding>,
        use_v_norm: bool,
    ) -> Result<Self> {
        Self::load_with_opts(
            gg,
            prefix,
            cfg,
            layer_idx,
            rotary,
            StandardAttentionOpts {
                use_v_norm,
                use_gemma_norms: false,
                attention_scale: None,
            },
        )
    }

    /// Load from GGUF with explicit options. Reads whatever norms/projections exist.
    ///
    /// Reads `wk`/`wv` only if their tensors are present in the GGUF — shared-KV
    /// layers (e.g. gemma4-31B layers without `attn_v.weight`) will have those
    /// fields set to `None`, and [`Self::compute_kv`] will return `None`.
    pub fn load_with_opts(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
        layer_idx: usize,
        rotary: Arc<RotaryEmbedding>,
        opts: StandardAttentionOpts,
    ) -> Result<Self> {
        let n_kv_head = *cfg.head_count_kv.get(layer_idx);

        let wq = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        // K and V may be missing in shared-KV layers
        let wk = gg.try_qmatmul(&format!("{prefix}.attn_k.weight"));
        let wv = gg.try_qmatmul(&format!("{prefix}.attn_v.weight"));
        let wo = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        // Infer head_dim from K weight shape rather than relying on metadata alone.
        // K weight maps hidden → n_kv_head * head_dim, so we can derive head_dim.
        // This handles per-layer variable head_dim (e.g. gemma4 sliding vs global layers).
        let head_dim = if let Ok(k_tensor) = gg.tensor(&format!("{prefix}.attn_k.weight")) {
            let k_out_dim = k_tensor.shape().dims()[0];
            if n_kv_head > 0 { k_out_dim / n_kv_head } else { cfg.head_dim }
        } else {
            cfg.head_dim
        };
        let n_head = {
            let q_tensor = gg.tensor(&format!("{prefix}.attn_q.weight"))?;
            let q_out_dim = q_tensor.shape().dims()[0];
            if head_dim > 0 { q_out_dim / head_dim } else { cfg.head_count }
        };

        // Q/K norms with optional Gemma +1 offset
        let q_norm = if opts.use_gemma_norms {
            GemmaRmsNorm::try_load(gg, &format!("{prefix}.attn_q_norm.weight"), cfg.rms_norm_eps)
                .map(AttnNorm::Gemma)
        } else {
            gg.try_rms_norm(&format!("{prefix}.attn_q_norm.weight"), cfg.rms_norm_eps)
                .map(AttnNorm::Standard)
        };
        let k_norm = if opts.use_gemma_norms {
            GemmaRmsNorm::try_load(gg, &format!("{prefix}.attn_k_norm.weight"), cfg.rms_norm_eps)
                .map(AttnNorm::Gemma)
        } else {
            gg.try_rms_norm(&format!("{prefix}.attn_k_norm.weight"), cfg.rms_norm_eps)
                .map(AttnNorm::Standard)
        };

        let attn_scale = opts
            .attention_scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_head,
            n_kv_head,
            head_dim,
            attn_scale,
            v_norm_eps: if opts.use_v_norm { Some(cfg.rms_norm_eps) } else { None },
            rotary,
        })
    }

    /// True if this layer has its own K/V projections. False for shared-KV layers
    /// where the caller must source `(K, V)` from another layer's cache.
    pub fn has_kv(&self) -> bool {
        self.wk.is_some() && self.wv.is_some()
    }

    /// Compute fresh `(K, V)` from this layer's `wk`/`wv` for the current step.
    /// The returned tensors have RoPE already applied to K and V-norm already
    /// applied to V (when configured).
    ///
    /// Returns `None` if this layer has neither `wk` nor `wv` (a fully shared-KV
    /// layer that must borrow `(K, V)` from another layer's cache).
    ///
    /// **Gemma4 quirk:** when `wk` is present but `wv` is `None`, the layer
    /// derives V from K (`V = K`) per `gemma4-iswa.cpp:74`. This is *not* the
    /// same as borrowing from another layer.
    ///
    /// Output shapes: `(B, n_kv_head, seq_len, head_dim)`.
    pub fn compute_kv(&self, x: &Tensor, offset: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let wk = match &self.wk {
            Some(wk) => wk,
            None => return Ok(None),
        };
        let k_raw = wk.forward(x)?;
        let k_3d = k_raw
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Compute V from wv if present, else fall back to V = K (pre-norm, pre-rope).
        // gemma4-iswa.cpp:73-74 binds Vcur = Kcur BEFORE the reshape, k_norm, and RoPE,
        // so we clone k_3d here before applying any of those to K.
        let v_pre = if let Some(ref wv) = self.wv {
            let v = wv.forward(x)?;
            v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            k_3d.clone()
        };

        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k_3d)?
        } else {
            k_3d
        };
        let v = if let Some(eps) = self.v_norm_eps {
            super::norms::v_norm(&v_pre, eps)?
        } else {
            v_pre
        };
        // K gets RoPE'd here so the value stored in the cache is already rotated.
        // Q is RoPE'd inside `forward_with_kv` since the offset can move per call.
        let k = self.rotary.apply_one(&k, offset)?;
        Ok(Some((k, v)))
    }

    /// Compute K, V, and Q projections with **shared Q8_1 quantization**.
    ///
    /// When all three projections consume the same normalized activation
    /// `x_norm`, the standard path quantizes `x_norm` to Q8_1 three times
    /// (once per matmul). This method quantizes once and reuses the buffer
    /// for all three projections, saving 2 `quantize_q8_1` kernel launches
    /// per layer per forward step.
    ///
    /// Returns `(q, k, v)` with Q already RoPE'd and K already RoPE'd + normed.
    /// Falls back to separate compute_kv + prepare_q when conditions aren't met.
    #[cfg(feature = "hip")]
    pub fn compute_qkv_shared_q8(
        &self,
        x: &Tensor,
        offset: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        use candle::quantized::hip::{quantize_q8_1, pad, MATRIX_ROW_PADDING};
        use candle::quantized::GgmlDType;

        let (b_sz, seq_len, hidden) = x.dims3()?;
        let b_size = b_sz * seq_len;

        // Only use shared Q8_1 for small batch (decode path, b*m <= 8)
        // and when all three projections exist and input is HIP f32 contiguous
        if b_size > 8
            || !matches!(x.device(), candle::Device::Hip(_))
            || x.dtype() != candle::DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }

        let wk = match &self.wk {
            Some(wk) => wk,
            None => return Ok(None),
        };
        let wv = match &self.wv {
            Some(wv) => wv,
            None => return Ok(None),
        };

        // Only use shared Q8_1 when all three are quantized (not dequantized f32/f16)
        if !self.wq.is_qtensor() || !wk.is_qtensor() || !wv.is_qtensor() {
            return Ok(None);
        }

        // Pre-quantize x to Q8_1 once
        let dev = match x.device() {
            candle::Device::Hip(d) => d.clone(),
            _ => return Ok(None),
        };
        let (x_st, x_l) = x.storage_and_layout();
        let x_hip = match &*x_st {
            candle::Storage::Hip(s) => s,
            _ => return Ok(None),
        };
        let x_slice = x_hip.as_hip_slice::<f32>()?;
        let x_view = match x_l.contiguous_offsets() {
            Some((lo, hi)) => x_slice.slice(lo..hi),
            None => return Ok(None),
        };

        let ncols_padded = pad(hidden, MATRIX_ROW_PADDING);
        let q8_bytes = b_size * ncols_padded
            * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(q8_bytes)? };
        quantize_q8_1(&x_view, &mut y_q8_1, hidden, b_size, &dev)?;
        let q8_view = y_q8_1.slice(0..y_q8_1.len());

        let rhs_shape = x.dims();
        drop(x_st);

        // Project Q using pre-quantized Q8_1
        let q_raw = self.wq.forward_preq8(&q8_view, b_size, rhs_shape)?;
        let mut q = q_raw
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        let q = self.rotary.apply_one(&q, offset)?;

        // Project K using pre-quantized Q8_1
        let k_raw = wk.forward_preq8(&q8_view, b_size, rhs_shape)?;
        let k_3d = k_raw
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Project V using pre-quantized Q8_1
        let v_pre = if let Some(ref wv_mat) = self.wv {
            let v_raw = wv_mat.forward_preq8(&q8_view, b_size, rhs_shape)?;
            v_raw.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            k_3d.clone()
        };

        // Apply K norm + RoPE
        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k_3d)?
        } else {
            k_3d
        };
        let v = if let Some(eps) = self.v_norm_eps {
            super::norms::v_norm(&v_pre, eps)?
        } else {
            v_pre
        };
        let k = self.rotary.apply_one(&k, offset)?;

        Ok(Some((q, k, v)))
    }

    /// Run the attention block with externally-provided (K, V) tensors.
    ///
    /// `k` and `v` are the **full cumulative cache** for this layer (or its
    /// shared-source layer). They must already have RoPE applied to K and any
    /// V-norm applied — exactly what [`Self::compute_kv`] returns.
    ///
    /// `offset` is the position offset for the *new* tokens in `x`, used by
    /// the Q rotary embedding.
    pub fn forward_with_kv(
        &self,
        x: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let q = self.prepare_q(x, offset)?;
        let (b_sz, seq_len, _) = x.dims3()?;
        // GQA via zero-copy Q reshape — see `gqa_attention` doc. K and V stay
        // at `(B, n_kv_head, T, D)`; there is no physical broadcast copy.
        let attn_output = gqa_attention(&q, k, v, mask, self.attn_scale)?;
        self.finish_attn(attn_output, b_sz, seq_len)
    }

    /// Attack C variant: run the attention block with a **pre-transposed K**
    /// (`(B, n_kv_head, D, T)`) so the internal `k.transpose().contiguous()`
    /// materialisation is skipped. Caller must supply K in that layout —
    /// typically from a `KvCache::new_k_transposed` cache.
    pub fn forward_with_kv_transposed(
        &self,
        x: &Tensor,
        k_t: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let q = self.prepare_q(x, offset)?;
        let (b_sz, seq_len, _) = x.dims3()?;
        let attn_output = gqa_attention_k_transposed(&q, k_t, v, mask, self.attn_scale)?;
        self.finish_attn(attn_output, b_sz, seq_len)
    }

    /// Shared prefix for `forward_with_kv*`: project + reshape + Q-norm + RoPE.
    fn prepare_q(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let mut q = self.wq.forward(x)?;
        q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        // Q gets RoPE'd here. K was already rotated inside compute_kv.
        self.rotary.apply_one(&q, offset)
    }

    /// Shared suffix for `forward_with_kv*`: reshape back + wo.
    pub fn finish_attn(&self, attn_output: Tensor, b_sz: usize, seq_len: usize) -> Result<Tensor> {
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.wo.forward(&attn_output)
    }

    /// Convenience wrapper that allocates a per-instance KV cache and runs the
    /// full attention. Used by simple models (qwen35-9B dense, llama-style)
    /// that don't need cross-layer sharing.
    pub fn forward(
        &mut self,
        x: &Tensor,
        kv_cache: &mut ConcatKvCache,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (k_new, v_new) = self
            .compute_kv(x, offset)?
            .ok_or_else(|| candle::Error::Msg(
                "StandardAttention::forward called on a shared-KV layer; \
                 use forward_with_kv to pass borrowed (K, V)".into()
            ))?;
        let (k_full, v_full) = kv_cache.append(&k_new, &v_new)?;
        self.forward_with_kv(x, &k_full, &v_full, mask, offset)
    }
}

// ---------------------------------------------------------------------------
// GatedAttention
// ---------------------------------------------------------------------------

/// Gated attention: Q projection outputs Q + gate interleaved.
/// Attention output is element-wise multiplied by sigmoid(gate) before output projection.
///
/// Used by qwen35/qwen3next full-attention layers.
/// Reference: llama.cpp qwen35.cpp `build_layer_attn`
pub struct GatedAttention {
    /// Fused Q+gate, K, V weight: rows
    /// `[0 .. 2*n_head*head_dim)` are the Q+gate projection (which is
    /// itself the qwen35 fused Q + per-head gate),
    /// `[2*n_head*head_dim .. + n_kv_head*head_dim)` are K, the next
    /// `n_kv_head*head_dim` rows are V. One matmul launch instead of
    /// three on the forward path.
    wqkv: QMatMul,
    /// Output dim of the Q+gate slice: `2 * n_head * head_dim`.
    qg_out: usize,
    /// Output dim of the K (and V) slice: `n_kv_head * head_dim`.
    kv_out: usize,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    attn_scale: f64,
    rotary: Arc<RotaryEmbedding>,
    /// Pre-allocated KV cache (slice-set into a fixed buffer).
    /// Replaces `ConcatKvCache` which allocated a new growing-size
    /// buffer every step.
    kv_cache: KvCache,
}

impl GatedAttention {
    /// Load from GGUF for a full-attention layer in qwen35-family models.
    /// Concatenates wq + wk + wv into a single fused weight matrix at
    /// load time so the forward path issues one matmul launch instead
    /// of three. The narrows on the fused output recover the
    /// individual Q+gate, K, and V slices.
    pub fn load(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
        layer_idx: usize,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let n_kv_head = *cfg.head_count_kv.get(layer_idx);
        let n_head = cfg.head_count;
        let head_dim = cfg.head_dim;
        // qwen35 wq carries Q AND a per-head gate, so its row count is
        // 2 * n_head * head_dim. K and V are vanilla GQA projections.
        let qg_out = 2 * n_head * head_dim;
        let kv_out = n_kv_head * head_dim;

        let wq_name = format!("{prefix}.attn_q.weight");
        let wk_name = format!("{prefix}.attn_k.weight");
        let wv_name = format!("{prefix}.attn_v.weight");
        let wqkv = gg.qmatmul_concat_rows(&[&wq_name, &wk_name, &wv_name])?;

        let wo = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), cfg.rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), cfg.rms_norm_eps)?;

        // Attack C: K is stored pre-transposed (B, n_kv_head, D, T) so
        // attention can skip the per-call `k.t().contiguous()`. V stays
        // in the canonical `(B, n_kv_head, T, head_dim)` layout (dim=2).
        let kv_cache = KvCache::new_k_transposed(2, KV_CACHE_INITIAL);

        Ok(Self {
            wqkv,
            qg_out,
            kv_out,
            wo,
            q_norm,
            k_norm,
            n_head,
            n_kv_head,
            head_dim,
            attn_scale: cfg.default_attention_scale(),
            rotary,
            kv_cache,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // Single fused QKV matmul. Output shape:
        // (B, L, qg_out + kv_out + kv_out)
        let qkv = self.wqkv.forward(x)?;

        // Slice the fused output back into Q+gate, K, and V along
        // the last dim. The narrows are non-contiguous views; the
        // downstream norms will materialize via .contiguous().
        let q_full = qkv.narrow(D::Minus1, 0, self.qg_out)?;
        let k = qkv.narrow(D::Minus1, self.qg_out, self.kv_out)?;
        let v = qkv.narrow(D::Minus1, self.qg_out + self.kv_out, self.kv_out)?;

        // Q+gate path: reshape to (B, L, n_head, 2*head_dim), split.
        let q_full = q_full.reshape((b_sz, seq_len, self.n_head, 2 * self.head_dim))?;
        let q = q_full.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_full.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        // Transpose to (B, H, L, D)
        let q = q.transpose(1, 2)?;
        let gate = gate.transpose(1, 2)?;

        // Q norm (applied per-head)
        let q = self.q_norm.forward(&q.contiguous()?)?;

        // K, V: reshape to (B, L, n_kv_head, head_dim), transpose to
        // (B, n_kv_head, L, head_dim).
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        // K norm
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // RoPE (multi-frequency sections applied by the RotaryEmbedding)
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // KV cache. The pre-allocated `KvCache::append` uses
        // `slice_set` under the hood, which requires contiguous
        // sources. `Tensor::contiguous` is a no-op when the tensor
        // is already contiguous, so we can call it unconditionally
        // without wasting work on the common case.
        //
        // With Attack C, `kv_cache` is `new_k_transposed` so the
        // returned K is already `(B, n_kv_head, D, T)` and we pass it
        // straight into `gqa_attention_k_transposed`.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (k_t, v) = self.kv_cache.append(&k, &v)?;
        let attn_output = gqa_attention_k_transposed(&q, &k_t, &v, mask, self.attn_scale)?;

        // Gate: attn_output * sigmoid(gate). Both are `(B, n_head, L, D)`.
        let gate_sigmoid = candle_nn::ops::sigmoid(&gate)?;
        let attn_output = (attn_output * gate_sigmoid)?;

        // Reshape: (B, H, L, D) -> (B, L, H*D)
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        self.wo.forward(&attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    /// Oracle implementation using the old `broadcast_kv` physical
    /// expansion. Used only for regression testing — the production code
    /// path is `gqa_attention`.
    fn broadcast_kv_oracle(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, n_kv, t, d) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, n_kv, n_rep, t, d))?
            .reshape((b, n_kv * n_rep, t, d))?
            .contiguous()
    }

    /// Oracle attention: physically broadcasts K/V, then does the matmul
    /// chain. Mirrors the pre-refactor `StandardAttention::forward_with_kv`
    /// attention block exactly.
    fn gqa_attention_oracle(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        attn_scale: f64,
    ) -> Result<Tensor> {
        let (_, n_head, _, _) = q.dims4()?;
        let (_, n_kv_head, _, _) = k.dims4()?;
        let n_rep = n_head / n_kv_head;

        let k_full = broadcast_kv_oracle(k, n_rep)?;
        let v_full = broadcast_kv_oracle(v, n_rep)?;

        let attn_weights = (q.matmul(&k_full.t()?.contiguous()?)? * attn_scale)?;
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights.matmul(&v_full.contiguous()?)
    }

    /// Maximum absolute difference between two tensors, reduced to a f32
    /// scalar. Used as the correctness metric in the oracle tests.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let d = (a - b).unwrap().abs().unwrap();
        d.flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    fn run_gqa_case(
        b_sz: usize,
        n_head: usize,
        n_kv_head: usize,
        seq_len: usize,
        t_cache: usize,
        head_dim: usize,
        use_mask: bool,
    ) {
        let dev = &Device::Cpu;
        // Fixed seed via deterministic values — `Tensor::randn` uses the
        // global rand state which the test harness doesn't seed, but the
        // oracle vs. new path are both pure functions of the same inputs
        // so any non-zero random draw is fine.
        let q = Tensor::randn(0f32, 0.1, (b_sz, n_head, seq_len, head_dim), dev).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b_sz, n_kv_head, t_cache, head_dim), dev).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b_sz, n_kv_head, t_cache, head_dim), dev).unwrap();

        // Triangular mask shape `(L, T)` — broadcasts against `(B, n_head, L, T)`.
        let mask = if use_mask {
            let raw: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..t_cache).map(move |j| {
                        if j > i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(raw, (seq_len, t_cache), dev).unwrap())
        } else {
            None
        };

        let scale = 1.0 / (head_dim as f64).sqrt();

        let got = gqa_attention(&q, &k, &v, mask.as_ref(), scale).unwrap();
        let want = gqa_attention_oracle(&q, &k, &v, mask.as_ref(), scale).unwrap();

        assert_eq!(got.dims(), want.dims(), "output shape differs");
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < 1e-5,
            "gqa_attention diverged from oracle: max_abs_diff = {diff} \
             (b={b_sz}, n_head={n_head}, n_kv={n_kv_head}, L={seq_len}, \
              T={t_cache}, D={head_dim}, mask={use_mask})"
        );
    }

    /// Attack C regression: `gqa_attention_k_transposed(q, k_t, v, ...)`
    /// must produce bit-identical output to `gqa_attention(q, k, v, ...)`
    /// when `k_t = k.transpose(-2, -1).contiguous()`. Covers several GQA
    /// ratios and both masked / unmasked paths.
    fn run_gqa_k_transposed_case(
        b_sz: usize,
        n_head: usize,
        n_kv_head: usize,
        seq_len: usize,
        t_cache: usize,
        head_dim: usize,
        use_mask: bool,
    ) {
        let dev = &Device::Cpu;
        let q = Tensor::randn(0f32, 0.1, (b_sz, n_head, seq_len, head_dim), dev).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b_sz, n_kv_head, t_cache, head_dim), dev).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b_sz, n_kv_head, t_cache, head_dim), dev).unwrap();

        let mask = if use_mask {
            let raw: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..t_cache).map(move |j| {
                        if j > i + (t_cache - seq_len) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(raw, (seq_len, t_cache), dev).unwrap())
        } else {
            None
        };

        let scale = 1.0 / (head_dim as f64).sqrt();

        // Reference: normal gqa_attention with canonical K layout.
        let reference = gqa_attention(&q, &k, &v, mask.as_ref(), scale).unwrap();

        // Attack C path: pre-transpose K to (B, n_kv_head, D, T) and use
        // the `*_k_transposed` entry point, which skips the internal
        // k.t().contiguous() materialisation.
        let k_t = k.transpose(2, 3).unwrap().contiguous().unwrap();
        let got = gqa_attention_k_transposed(&q, &k_t, &v, mask.as_ref(), scale).unwrap();

        assert_eq!(got.dims(), reference.dims());
        let diff = max_abs_diff(&got, &reference);
        assert!(
            diff < 1e-5,
            "gqa_attention_k_transposed diverged from gqa_attention: \
             max_abs_diff = {diff} (b={b_sz}, n_head={n_head}, n_kv={n_kv_head}, \
             L={seq_len}, T={t_cache}, D={head_dim}, mask={use_mask})"
        );
    }

    #[test]
    fn test_gqa_attention_k_transposed_matches_reference() {
        // n_rep = 1 / 2 / 4 / 8, with and without mask, various shapes.
        run_gqa_k_transposed_case(1, 8, 8, 4, 4, 16, false);
        run_gqa_k_transposed_case(1, 8, 8, 4, 4, 16, true);
        run_gqa_k_transposed_case(1, 8, 4, 5, 5, 16, true);
        run_gqa_k_transposed_case(1, 16, 4, 6, 6, 16, true);
        run_gqa_k_transposed_case(1, 16, 2, 4, 7, 16, true);
        run_gqa_k_transposed_case(1, 32, 2, 3, 9, 16, true);
        // Decode-like rectangular mask: L_q=1, T > 1 with KV prefix.
        run_gqa_k_transposed_case(1, 8, 4, 1, 16, 16, true);
    }

    #[test]
    fn test_gqa_attention_n_rep_1_identity() {
        // n_rep = 1 (plain MHA): the reshapes are no-ops, must match.
        run_gqa_case(1, 8, 8, 4, 4, 16, false);
        run_gqa_case(1, 8, 8, 4, 4, 16, true);
    }

    #[test]
    fn test_gqa_attention_n_rep_2() {
        run_gqa_case(1, 8, 4, 5, 5, 16, false);
        run_gqa_case(1, 8, 4, 5, 5, 16, true);
    }

    #[test]
    fn test_gqa_attention_n_rep_4() {
        run_gqa_case(1, 16, 4, 6, 6, 16, false);
        run_gqa_case(1, 16, 4, 6, 6, 16, true);
    }

    #[test]
    fn test_gqa_attention_n_rep_8() {
        // n_rep = 8 is the boundary where the old `utils::repeat_kv`
        // (cat-based) produced wrong head ordering. Verify the new
        // reshape path agrees with the `broadcast_kv` oracle here.
        run_gqa_case(1, 16, 2, 4, 7, 16, false);
        run_gqa_case(1, 16, 2, 4, 7, 16, true);
    }

    #[test]
    fn test_gqa_attention_n_rep_16() {
        // qwen35 dense: 16 Q heads per KV head.
        run_gqa_case(1, 32, 2, 3, 9, 16, false);
        run_gqa_case(1, 32, 2, 3, 9, 16, true);
    }

    #[test]
    fn test_gqa_attention_decode_step() {
        // seq_len = 1 (decode): n_rep * 1 = n_rep, still correct.
        run_gqa_case(1, 16, 2, 1, 12, 32, false);
        run_gqa_case(1, 16, 2, 1, 12, 32, true);
    }

    #[test]
    fn test_gqa_attention_batch_size_2() {
        // Non-unit batch — the reshape math has to hold across the batch.
        run_gqa_case(2, 12, 3, 5, 5, 16, false);
        run_gqa_case(2, 12, 3, 5, 5, 16, true);
    }

    #[test]
    fn test_gqa_attention_reshapes_are_zero_copy() {
        // Prove the reshape round-trips don't materialise: start from a
        // contiguous Q and manually replay the reshapes, asserting
        // `is_contiguous()` at each step. If candle ever decides to
        // insert a copy for the `(B, n_kv_head, n_rep*L, D)` ↔
        // `(B, n_head, L, D)` transition, this test will catch it.
        let dev = &Device::Cpu;
        let (b, n_head, l, d) = (1usize, 16usize, 5usize, 16usize);
        let n_kv_head = 2usize;
        let n_rep = n_head / n_kv_head;
        let q = Tensor::randn(0f32, 0.1, (b, n_head, l, d), dev).unwrap();
        assert!(q.is_contiguous());

        let q_grouped = q.reshape((b, n_kv_head, n_rep * l, d)).unwrap();
        assert!(
            q_grouped.is_contiguous(),
            "Q reshape to grouped form should be zero-copy"
        );

        // Simulate a matmul-output shape and verify the inverse reshape
        // from grouped back to (B, n_head, L, D) is also zero-copy when
        // the source is contiguous.
        let fake_out = Tensor::zeros((b, n_kv_head, n_rep * l, d), DType::F32, dev).unwrap();
        assert!(fake_out.is_contiguous());
        let unflat = fake_out.reshape((b, n_head, l, d)).unwrap();
        assert!(
            unflat.is_contiguous(),
            "matmul-output reshape back to (B, n_head, L, D) should be zero-copy"
        );
    }

    /// P2 regression: the new HIP flash-attention kernel must match
    /// the CPU `gqa_attention_oracle` output within numerical
    /// tolerance across a spread of shapes (prefill-like large L,
    /// decode-like L=1, with and without mask, n_rep in {1, 4, 8}).
    #[cfg(feature = "hip")]
    #[test]
    fn hip_flash_attn_matches_cpu_oracle_d64() {
        let dev_hip = match Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = Device::Cpu;
        // (B, n_head, n_kv_head, L_q, L_k, D, use_mask)
        let cases: &[(usize, usize, usize, usize, usize, usize, bool)] = &[
            (1, 8, 8, 4, 4, 64, false),    // small no-GQA
            (1, 8, 8, 4, 4, 64, true),     // small with mask
            (1, 32, 8, 16, 16, 64, true),  // GQA n_rep=4
            (1, 32, 4, 12, 12, 64, true),  // GQA n_rep=8
            (1, 32, 32, 1, 16, 64, false), // decode-like L_q=1 over T=16
            (2, 8, 4, 8, 8, 64, true),     // batch=2, n_rep=2
            (1, 8, 8, 1, 1149, 64, false), // long decode
        ];

        for &(b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask) in cases {
            let qkv_len_q = b_sz * n_head * l_q * d;
            let qkv_len_kv = b_sz * n_kv_head * l_k * d;
            let q_vals: Vec<f32> = (0..qkv_len_q)
                .map(|i| ((i as f32) * 0.00037).sin() * 0.1)
                .collect();
            let k_vals: Vec<f32> = (0..qkv_len_kv)
                .map(|i| ((i as f32) * 0.00053).cos() * 0.1)
                .collect();
            let v_vals: Vec<f32> = (0..qkv_len_kv)
                .map(|i| ((i as f32) * 0.00041).sin() * 0.1)
                .collect();

            let q_cpu = Tensor::from_slice(&q_vals, (b_sz, n_head, l_q, d), &dev_cpu).unwrap();
            let k_cpu = Tensor::from_slice(&k_vals, (b_sz, n_kv_head, l_k, d), &dev_cpu).unwrap();
            let v_cpu = Tensor::from_slice(&v_vals, (b_sz, n_kv_head, l_k, d), &dev_cpu).unwrap();

            let q_hip = Tensor::from_slice(&q_vals, (b_sz, n_head, l_q, d), &dev_hip).unwrap();
            let k_hip = Tensor::from_slice(&k_vals, (b_sz, n_kv_head, l_k, d), &dev_hip).unwrap();
            let v_hip = Tensor::from_slice(&v_vals, (b_sz, n_kv_head, l_k, d), &dev_hip).unwrap();

            let mask_cpu_hip = if use_mask {
                // Simple causal mask on last-L_q rows of a L_q × L_k
                // matrix. Equivalent shape: (1, 1, L_q, L_k).
                let mut mv = vec![0.0f32; l_q * l_k];
                for qi in 0..l_q {
                    for ki in 0..l_k {
                        // Causal: allow positions [0, qi + (L_k - L_q)]
                        let past = l_k - l_q;
                        if ki > qi + past {
                            mv[qi * l_k + ki] = f32::NEG_INFINITY;
                        }
                    }
                }
                Some((
                    Tensor::from_slice(&mv, (1, 1, l_q, l_k), &dev_cpu).unwrap(),
                    Tensor::from_slice(&mv, (1, 1, l_q, l_k), &dev_hip).unwrap(),
                ))
            } else {
                None
            };

            let scale = 1.0 / (d as f64).sqrt();

            // CPU oracle
            let mask_cpu = mask_cpu_hip.as_ref().map(|(c, _)| c);
            let out_cpu = gqa_attention_oracle(&q_cpu, &k_cpu, &v_cpu, mask_cpu, scale).unwrap();

            // HIP flash-attention
            let mask_hip = mask_cpu_hip.as_ref().map(|(_, h)| h);
            let out_hip = candle::hip_backend::flash_attn_fused(
                &q_hip, &k_hip, &v_hip, mask_hip, scale,
            )
            .unwrap();

            let out_cpu_vals: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
            let out_hip_vals: Vec<f32> = out_hip
                .to_device(&dev_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                out_cpu_vals.len(),
                out_hip_vals.len(),
                "length mismatch for case {:?}",
                (b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask)
            );

            let max_abs = out_cpu_vals
                .iter()
                .zip(out_hip_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Online softmax accumulates in a different order than the
            // full `softmax(Q@K^T) @ V` oracle. Small FMA drift is
            // expected, especially for long L_k. 1e-4 absolute is
            // defensive; TinyLlama outputs tend to match much tighter.
            assert!(
                max_abs < 1e-4,
                "flash_attn HIP vs CPU oracle drift on {:?}: max_abs={max_abs}",
                (b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask)
            );
        }
    }

    /// Q0c regression: fused masked_softmax_scale on HIP must match the
    /// chained `att * scale + mask → softmax_last_dim(att)` reference.
    /// Covers the three common broadcast shapes for `mask` and the
    /// "no mask" decode case.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_masked_softmax_scale_matches_chain() {
        let dev_hip = match Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        // (B, H, L_q, L_k, mask_shape_kind)
        // mask_shape_kind: 0=None, 1=(1,1,1,Lk), 2=(1,1,Lq,Lk),
        //                  3=(B,1,1,Lk), 4=(B,1,Lq,Lk)
        let cases: &[(usize, usize, usize, usize, u8)] = &[
            (1, 4, 4, 4, 0),
            (1, 4, 4, 4, 2),
            (1, 8, 16, 16, 2),
            (2, 8, 8, 8, 4),
            (1, 32, 1, 1024, 0),
            (1, 32, 1, 1024, 1),
            (1, 32, 1, 1024, 3),
            (2, 8, 4, 32, 4),
        ];

        for &(b_sz, n_head, l_q, l_k, mask_kind) in cases {
            let numel = b_sz * n_head * l_q * l_k;
            let att_vals: Vec<f32> = (0..numel)
                .map(|i| ((i as f32) * 0.00073).sin() * 0.5)
                .collect();
            let att_hip = Tensor::from_slice(
                &att_vals,
                (b_sz, n_head, l_q, l_k),
                &dev_hip,
            )
            .unwrap();

            let mask_hip: Option<Tensor> = match mask_kind {
                0 => None,
                1 => {
                    let mv: Vec<f32> = (0..l_k)
                        .map(|i| if i > l_k / 2 { f32::NEG_INFINITY } else { 0.0 })
                        .collect();
                    Some(Tensor::from_slice(&mv, (1, 1, 1, l_k), &dev_hip).unwrap())
                }
                2 => {
                    let mut mv = vec![0.0f32; l_q * l_k];
                    for qi in 0..l_q {
                        for ki in 0..l_k {
                            if ki > qi + (l_k - l_q) {
                                mv[qi * l_k + ki] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    Some(Tensor::from_slice(&mv, (1, 1, l_q, l_k), &dev_hip).unwrap())
                }
                3 => {
                    let mut mv = vec![0.0f32; b_sz * l_k];
                    for bi in 0..b_sz {
                        for ki in 0..l_k {
                            if ki >= l_k - bi {
                                mv[bi * l_k + ki] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    Some(Tensor::from_slice(&mv, (b_sz, 1, 1, l_k), &dev_hip).unwrap())
                }
                4 => {
                    let mut mv = vec![0.0f32; b_sz * l_q * l_k];
                    for bi in 0..b_sz {
                        for qi in 0..l_q {
                            for ki in 0..l_k {
                                if ki > qi + (l_k - l_q) {
                                    mv[(bi * l_q + qi) * l_k + ki] = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }
                    Some(Tensor::from_slice(&mv, (b_sz, 1, l_q, l_k), &dev_hip).unwrap())
                }
                _ => None,
            };

            let scale = 1.0 / (l_k as f64).sqrt();

            // Reference: the unfused chain.
            let ref_out = {
                let scaled = (&att_hip * scale).unwrap();
                let masked = match mask_hip.as_ref() {
                    Some(m) => scaled.broadcast_add(m).unwrap(),
                    None => scaled,
                };
                candle_nn::ops::softmax_last_dim(&masked).unwrap()
            };

            // Fused.
            let fused_out = candle::hip_backend::masked_softmax_scale_fused(
                &att_hip,
                mask_hip.as_ref(),
                scale,
            )
            .unwrap();

            let ref_vals: Vec<f32> = ref_out
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let fused_vals: Vec<f32> = fused_out
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ref_vals.len(), fused_vals.len());

            let max_abs = ref_vals
                .iter()
                .zip(fused_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "masked_softmax_scale drift on (B={b_sz}, H={n_head}, Lq={l_q}, Lk={l_k}, kind={mask_kind}): max_abs={max_abs}"
            );
        }
    }

    /// Q0a regression: fused post-norm residual on HIP must match the
    /// chained `rms_norm(x, weight) + residual` reference across a
    /// spread of shapes / hidden dims.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_rms_norm_post_residual_matches_chain() {
        let dev_hip = match Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        // (batch, seq_len, hidden)
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 64),      // decode-like tiny
            (1, 1, 2048),    // decode, block_size=1024 path
            (1, 1, 1536),    // decode, block_size=1024 path
            (2, 16, 256),    // small batch prefill
            (1, 1024, 512),  // long prefill, hidden<1024 path
            (1, 1024, 2048), // long prefill, hidden>=1024 path
        ];

        for &(b, s, h) in cases {
            let el = b * s * h;
            let x_vals: Vec<f32> = (0..el)
                .map(|i| ((i as f32) * 0.00091).sin() * 0.3)
                .collect();
            let r_vals: Vec<f32> = (0..el)
                .map(|i| ((i as f32) * 0.00057).cos() * 0.4)
                .collect();
            let w_vals: Vec<f32> = (0..h)
                .map(|i| 1.0 + ((i as f32) * 0.01).sin() * 0.1)
                .collect();
            let eps: f32 = 1e-6;

            let x_hip = Tensor::from_slice(&x_vals, (b, s, h), &dev_hip).unwrap();
            let r_hip = Tensor::from_slice(&r_vals, (b, s, h), &dev_hip).unwrap();
            let w_hip = Tensor::from_slice(&w_vals, h, &dev_hip).unwrap();

            // Reference.
            let ref_out = {
                let normed = candle_nn::ops::rms_norm(&x_hip, &w_hip, eps).unwrap();
                (&normed + &r_hip).unwrap()
            };

            // Fused.
            let fused_out = candle::hip_backend::rms_norm_post_residual_fused(
                &x_hip, &r_hip, &w_hip, eps,
            )
            .unwrap();

            let ref_vals: Vec<f32> = ref_out
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let fused_vals: Vec<f32> = fused_out
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ref_vals.len(), fused_vals.len());

            let max_abs = ref_vals
                .iter()
                .zip(fused_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "rms_norm_post_residual drift on (b={b}, s={s}, h={h}): max_abs={max_abs}"
            );
        }
    }

    /// Q3 regression: flash-attention v2 (BR=4, LDS-tiled) must match
    /// the CPU `gqa_attention_oracle` output across shapes that cover
    /// every head dim instantiation (64, 128, 256) and every mask
    /// broadcast kind. Also covers L_q just above the FLASH_V2_MIN_L_Q
    /// gate to exercise the block-boundary `q_in_range` path.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_flash_attn_v2_matches_cpu_oracle() {
        let dev_hip = match Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = Device::Cpu;
        // (B, n_head, n_kv_head, L_q, L_k, D, use_mask)
        let cases: &[(usize, usize, usize, usize, usize, usize, bool)] = &[
            // D=64 — TinyLlama-like
            (1, 8, 8, 4, 4, 64, false),
            (1, 32, 4, 8, 8, 64, true),
            (1, 8, 8, 16, 32, 64, true),
            // BR boundary: L_q=5, 6, 7, 8 (not all multiples of BR=4)
            (1, 8, 8, 5, 16, 64, true),
            (1, 8, 8, 7, 32, 64, true),
            // D=128 — qwen-class
            (1, 16, 4, 8, 16, 128, true),
            (1, 32, 8, 16, 64, 128, true),
            // D=256 — gemma4
            (1, 8, 4, 8, 16, 256, true),
            (1, 8, 4, 16, 32, 256, true),
            (1, 8, 4, 32, 64, 256, false),
            // batch=2
            (2, 8, 4, 8, 8, 64, true),
        ];

        for &(b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask) in cases {
            let qkv_len_q = b_sz * n_head * l_q * d;
            let qkv_len_kv = b_sz * n_kv_head * l_k * d;
            let q_vals: Vec<f32> = (0..qkv_len_q)
                .map(|i| ((i as f32) * 0.00037).sin() * 0.1)
                .collect();
            let k_vals: Vec<f32> = (0..qkv_len_kv)
                .map(|i| ((i as f32) * 0.00053).cos() * 0.1)
                .collect();
            let v_vals: Vec<f32> = (0..qkv_len_kv)
                .map(|i| ((i as f32) * 0.00041).sin() * 0.1)
                .collect();

            let q_cpu = Tensor::from_slice(&q_vals, (b_sz, n_head, l_q, d), &dev_cpu).unwrap();
            let k_cpu = Tensor::from_slice(&k_vals, (b_sz, n_kv_head, l_k, d), &dev_cpu).unwrap();
            let v_cpu = Tensor::from_slice(&v_vals, (b_sz, n_kv_head, l_k, d), &dev_cpu).unwrap();

            let q_hip = Tensor::from_slice(&q_vals, (b_sz, n_head, l_q, d), &dev_hip).unwrap();
            let k_hip = Tensor::from_slice(&k_vals, (b_sz, n_kv_head, l_k, d), &dev_hip).unwrap();
            let v_hip = Tensor::from_slice(&v_vals, (b_sz, n_kv_head, l_k, d), &dev_hip).unwrap();

            let mask_cpu_hip = if use_mask {
                let mut mv = vec![0.0f32; l_q * l_k];
                for qi in 0..l_q {
                    for ki in 0..l_k {
                        let past = l_k - l_q;
                        if ki > qi + past {
                            mv[qi * l_k + ki] = f32::NEG_INFINITY;
                        }
                    }
                }
                Some((
                    Tensor::from_slice(&mv, (1, 1, l_q, l_k), &dev_cpu).unwrap(),
                    Tensor::from_slice(&mv, (1, 1, l_q, l_k), &dev_hip).unwrap(),
                ))
            } else {
                None
            };

            let scale = 1.0 / (d as f64).sqrt();

            let mask_cpu = mask_cpu_hip.as_ref().map(|(c, _)| c);
            let out_cpu = gqa_attention_oracle(&q_cpu, &k_cpu, &v_cpu, mask_cpu, scale).unwrap();

            let mask_hip = mask_cpu_hip.as_ref().map(|(_, h)| h);
            let out_hip = candle::hip_backend::flash_attn_v2_fused(
                &q_hip, &k_hip, &v_hip, mask_hip, scale,
            )
            .unwrap();

            let out_cpu_vals: Vec<f32> =
                out_cpu.flatten_all().unwrap().to_vec1().unwrap();
            let out_hip_vals: Vec<f32> = out_hip
                .to_device(&dev_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                out_cpu_vals.len(),
                out_hip_vals.len(),
                "length mismatch for case {:?}",
                (b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask)
            );

            let max_abs = out_cpu_vals
                .iter()
                .zip(out_hip_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 2e-4,
                "flash_attn_v2 vs CPU oracle drift on {:?}: max_abs={max_abs}",
                (b_sz, n_head, n_kv_head, l_q, l_k, d, use_mask)
            );
        }
    }
}
