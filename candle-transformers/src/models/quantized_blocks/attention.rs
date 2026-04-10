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

/// GQA broadcast: expand `k` (or `v`) from `n_kv_head` to `n_head` by duplicating
/// each KV head `n_rep` times in place.
///
/// Replaces `crate::utils::repeat_kv` which uses `cat-along-dim2 + reshape`.
/// That trick produces wrong head ordering when `n_rep ≥ 8` (the cat puts
/// repeated tokens consecutively, then the reshape splits one head's data
/// across multiple new heads).
///
/// Input  shape: `(B, n_kv_head, T, D)`
/// Output shape: `(B, n_kv_head * n_rep, T, D)` where new heads `[k*n_rep .. (k+1)*n_rep)`
/// all hold KV head `k`'s values.
fn broadcast_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, t, d))?
        .reshape((b, n_kv * n_rep, t, d))?
        .contiguous()
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
    attn_scale: f64,
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
        let q = self.rotary.apply_one(&q, offset)?;

        // GQA broadcast K/V
        let n_rep = self.n_head / self.n_kv_head;
        let k_full = broadcast_kv(k, n_rep)?;
        let v_full = broadcast_kv(v, n_rep)?;

        // Q @ K^T scaled
        let attn_weights = (q.matmul(&k_full.t()?.contiguous()?)? * self.attn_scale)?;
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_full.contiguous()?)?;

        // Reshape back: (B, H, L, D) → (B, L, H*D) → wo
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

        // dim=2 because the cached K/V are stored as
        // (B, n_kv_head, seq, head_dim).
        let kv_cache = KvCache::new(2, KV_CACHE_INITIAL);

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
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA broadcast — see `broadcast_kv` doc for why we don't use repeat_kv.
        let n_rep = self.n_head / self.n_kv_head;
        let k = broadcast_kv(&k, n_rep)?;
        let v = broadcast_kv(&v, n_rep)?;

        // Scaled dot-product attention. `q` is the rotary output,
        // which is already contiguous, so we drop the redundant
        // `.contiguous()` here. `k.t()` flips strides so we still
        // need to materialize it for matmul.
        let attn_weights = (q.matmul(&k.t()?.contiguous()?)? * self.attn_scale)?;
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        // Gate: attn_output * sigmoid(gate)
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
