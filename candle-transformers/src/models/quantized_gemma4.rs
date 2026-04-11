//! Gemma 4 model implementation with quantization support.
//!
//! Built on the modular [`crate::models::quantized_blocks`] system. The
//! per-architecture file is now a thin assembler that wires shared blocks
//! (`StandardAttention`, `DenseMlp`, `RotaryEmbedding`, …) with gemma4-specific
//! features:
//!
//! - **Per-layer head_dim** (sliding=`key_length_swa`, global=`key_length`)
//! - **Per-layer RoPE** with `freq_base` (global) or `freq_base_swa` (sliding)
//! - **`rope_freqs.weight`** proportional rope for non-SWA layers
//! - **Partial RoPE** when `rope.dimension_count` < head_dim
//! - **Parameter-free V-norm** (tri-norm pattern)
//! - **`f_attention_scale = 1.0`** (no pre-attention scaling)
//! - **GeGLU** in the FFN (not SiLU like Llama/Qwen)
//! - **Final logit softcapping** (tanh-based)
//! - **Layer output scale** (per-layer learned scalar)
//! - **Per-layer embedding** for E4B variant
//! - **Shared KV layers** for the 31B variant (`n_layer_kv_from_start` < `n_layer`)
//! - **Pipeline-parallel layer split** (`from_gguf_multi_device`)
//!
//! GGUF arch string: `gemma4`.

use crate::models::quantized_blocks::*;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, GgufBlob};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use rayon::prelude::*;
use std::sync::Arc;

pub const MAX_SEQ_LEN: usize = 131072;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

// ---------------------------------------------------------------------------
// Per-layer embedding (E4B-specific)
// ---------------------------------------------------------------------------

/// Per-layer embedding components (E4B-specific).
///
/// Each layer reads a slice from a global per-layer embedding tensor and
/// gates it through a learned `inp_gate → gelu → proj → post_norm` path.
struct PerLayerEmbed {
    inp_gate: super::with_tracing::QMatMul,
    proj: super::with_tracing::QMatMul,
    post_norm: RmsNorm,
}

/// Global per-layer embedding components (E4B only).
///
/// `token_embd` is the huge `per_layer_token_embd.weight` tensor (~11GB
/// dequantized for E4B). It's kept on CPU and only the looked-up rows are
/// moved to the model device per forward pass.
struct PerLayerEmbeddings {
    token_embd: Tensor,
    model_proj: super::with_tracing::QMatMul,
    proj_norm: RmsNorm,
    n_embd_per_layer: usize,
}

// ---------------------------------------------------------------------------
// LayerWeights
// ---------------------------------------------------------------------------

struct LayerWeights {
    attn: StandardAttention,
    attn_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: DenseMlp,
    /// Optional sliding-window mask radius (None = global attention).
    sliding_window_size: Option<usize>,
    /// True if this layer computes its own K/V (the first
    /// `n_layer_kv_from_start` layers). False for shared-KV layers
    /// that reuse an earlier layer's cache.
    has_kv: bool,
    /// Index of the source layer to borrow K/V cache from when `has_kv == false`.
    kv_source_idx: usize,
    /// Per-layer learned output scale (gemma4 only). When present, the residual
    /// stream after the FFN block is multiplied by this scalar.
    layer_output_scale: Option<Tensor>,
    /// Per-layer embedding components (E4B only).
    per_layer_embed: Option<PerLayerEmbed>,
    /// Device this layer's weights live on.
    device: Device,
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: super::with_tracing::QMatMul,
    final_logit_softcap: Option<f64>,
    /// Per-layer embedding components (E4B only).
    per_layer_embeddings: Option<PerLayerEmbeddings>,
    /// Per-layer KV cache. For shared-KV layers (`has_kv=false`) the entry is
    /// always `None` and the layer reads from `kv_caches[kv_source_idx]`.
    ///
    /// Uses the pre-allocated `KvCache` (slice-set into a fixed buffer)
    /// instead of `Tensor::cat`, which would otherwise allocate a new
    /// `cache_len + 1` buffer every decode step and copy the entire
    /// cache contents into it — O(N²) memory traffic on long contexts
    /// and the dominant decode cost on gemma4 (which slowed 31% on
    /// candle vs 3% on turbo at long-cache decode before this fix).
    kv_caches: Vec<Option<candle_nn::kv_cache::KvCache>>,
}

impl ModelWeights {
    pub fn from_gguf(
        ct: gguf_file::Content,
        blob: Arc<GgufBlob>,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_multi_device(ct, blob, &[device.clone()])
    }

    /// Load with pipeline-parallel layer split across multiple devices.
    /// Token embedding lives on `devices[0]`; output norm + lm_head live on
    /// the device of the last layer. Per-layer weights are loaded in
    /// parallel via rayon (mmap-backed `Gguf`).
    pub fn from_gguf_multi_device(
        ct: gguf_file::Content,
        blob: Arc<GgufBlob>,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("from_gguf_multi_device requires at least one device");
        }
        let dev0 = &devices[0];

        // ----- read all metadata up front via the shared GgufConfig --------
        let gg = Gguf::new(ct, blob, dev0.clone());
        let cfg = GgufConfig::from_metadata(gg.metadata())?;
        if cfg.arch != "gemma4" {
            candle::bail!("quantized_gemma4 expects arch=gemma4, got {}", cfg.arch);
        }
        let block_count = cfg.block_count;
        let head_count = cfg.head_count;
        let head_count_kv_default = *cfg.head_count_kv.get(0);
        let key_length = cfg.head_dim;
        let rms_norm_eps = cfg.rms_norm_eps;
        let embedding_length = cfg.hidden_size;

        // ----- gemma4-specific metadata fields not on GgufConfig -----------
        let metadata = gg.metadata();
        let md_get = |k: &str| metadata.get(&format!("gemma4.{k}"));
        let key_length_swa = md_get("attention.key_length_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(key_length);
        let sliding_window_size = md_get("attention.sliding_window")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(4096);
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_swa = md_get("rope.freq_base_swa")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);
        let rope_dim_count = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let rope_dim_count_swa = md_get("rope.dimension_count_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let final_logit_softcap = md_get("final_logit_softcapping")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let n_embd_per_layer = md_get("embedding_length_per_layer_input")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(0);

        // Per-layer kv heads (gemma4 may have an array)
        let head_count_kv_per_layer: Vec<usize> = (0..block_count)
            .map(|i| *cfg.head_count_kv.get(i))
            .collect();
        let _ = head_count_kv_default;

        // Sliding-window pattern: prefer the per-layer bool array; fall back
        // to the standard "every Nth layer is global" rule.
        let sliding_window_pattern: Option<Vec<bool>> = metadata
            .get("gemma4.attention.sliding_window_pattern")
            .and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    let mut bools = Vec::with_capacity(arr.len());
                    for item in arr {
                        match item {
                            gguf_file::Value::Bool(b) => bools.push(*b),
                            gguf_file::Value::U8(x) => bools.push(*x != 0),
                            _ => return None,
                        }
                    }
                    Some(bools)
                }
                _ => None,
            });
        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(6);

        // Shared-KV layers (31B variant): the LAST `shared_kv_layers` layers
        // reuse K/V from earlier layers via cache aliasing.
        let shared_kv_layers = md_get("attention.shared_kv_layers")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(0);
        let n_layer_kv_from_start = block_count.saturating_sub(shared_kv_layers);

        // ----- pipeline-parallel layer-to-device assignment ----------------
        let layer_to_device = split_layers_across_devices(block_count, devices.len());

        // Resolve sliding/global per layer.
        let is_sliding_per_layer: Vec<bool> = (0..block_count)
            .map(|i| {
                if let Some(ref pat) = sliding_window_pattern {
                    pat.get(i).copied().unwrap_or(false)
                } else {
                    (i + 1) % sliding_window_type > 0
                }
            })
            .collect();

        // For shared-KV layers, find the most recent same-pattern layer in
        // [0..n_layer_kv_from_start). Mirrors llama.cpp's iswa cache aliasing
        // (logged as `reuse layer N, is_swa = X`).
        let kv_source_per_layer: Vec<usize> = (0..block_count)
            .map(|il| {
                if il < n_layer_kv_from_start {
                    il
                } else {
                    let want = is_sliding_per_layer[il];
                    (0..n_layer_kv_from_start)
                        .rev()
                        .find(|&j| is_sliding_per_layer[j] == want)
                        .unwrap_or(0)
                }
            })
            .collect();

        // ----- shared rope_freqs (proportional rope for global layers) -----
        // Read from the dev0-targeted gg (no need for set_device since gg is
        // already on dev0).
        let rope_freqs: Option<Vec<f32>> = gg
            .try_dequantize("rope_freqs.weight")
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.flatten_all().ok())
            .and_then(|t| t.to_vec1::<f32>().ok());

        // ----- token embedding -------------------------------------------
        // For larger gemma4 variants (31B has hidden=5376, vocab=262144) the
        // dequantized F32 embedding is 5.6 GB and pushes dev0 over the 16 GB
        // MI50 limit. Keep the embedding on CPU and let the forward pass move
        // the looked-up rows to dev0 (a few KB per token).
        let cpu = candle::Device::Cpu;
        let cpu_gg = gg.with_device(cpu.clone());
        let tok_tensor = cpu_gg.tensor("token_embd.weight")?;
        let tok_embeddings = Embedding::new(tok_tensor.dequantize(&cpu)?, embedding_length);

        // ----- per-layer embedding global components (E4B only) ------------
        let per_layer_embeddings = if n_embd_per_layer > 0 {
            // Keep the huge per_layer_token_embd on CPU; the proj/norm live on dev0.
            let pl_embd = cpu_gg.try_dequantize("per_layer_token_embd.weight");
            let pl_proj = gg.try_qmatmul("per_layer_model_proj.weight");
            let pl_pn = gg.try_rms_norm("per_layer_proj_norm.weight", rms_norm_eps);
            match (pl_embd, pl_proj, pl_pn) {
                (Some(embd), Some(proj), Some(pn)) => Some(PerLayerEmbeddings {
                    token_embd: embd,
                    model_proj: proj,
                    proj_norm: pn,
                    n_embd_per_layer,
                }),
                _ => None,
            }
        } else {
            None
        };

        // ----- build layers in parallel via rayon --------------------------
        // Each rayon worker pulls a cheap Gguf clone for its layer's device,
        // then loads attn / norms / FFN / per-layer-embed from the shared
        // mmap'd blob in one shot. With 4 MI50s and gemma4 31B (~17 GB Q4_0)
        // this drops model build time from ~52 s (sequential) to ~6 s.
        let layers: Vec<LayerWeights> = (0..block_count)
            .into_par_iter()
            .map(|il| -> Result<LayerWeights> {
                let block_prefix = format!("blk.{il}");
                let layer_dev_idx = layer_to_device[il];
                let layer_device = devices[layer_dev_idx].clone();
                let lgg = gg.with_device(layer_device.clone());

                let is_sliding = is_sliding_per_layer[il];
                let has_kv = il < n_layer_kv_from_start;
                let kv_source_idx = kv_source_per_layer[il];

                // Per-layer head_dim (sliding=key_length_swa, global=key_length).
                let layer_head_dim = if is_sliding { key_length_swa } else { key_length };

                // Per-layer rotated dimension count.
                let rotated_dim = if is_sliding {
                    rope_dim_count_swa.unwrap_or(layer_head_dim)
                } else {
                    rope_dim_count.unwrap_or_else(|| {
                        let partial = (layer_head_dim as f64 * 0.25) as usize;
                        (partial & !1).max(2)
                    })
                };
                let layer_rope_freq = if is_sliding {
                    rope_freq_base_swa
                } else {
                    rope_freq_base
                };
                let layer_freq_factors: Option<&[f32]> =
                    if !is_sliding { rope_freqs.as_deref() } else { None };

                let rotary = Arc::new(RotaryEmbedding::new_with_freq_factors(
                    layer_rope_freq as f64,
                    rotated_dim,
                    MAX_SEQ_LEN.min(cfg.max_seq_len()),
                    layer_freq_factors,
                    DType::F32,
                    &layer_device,
                )?);

                // Per-layer GgufConfig clone with the right head_dim/n_kv_head.
                let mut layer_cfg = cfg.clone();
                layer_cfg.head_dim = layer_head_dim;
                layer_cfg.head_count_kv = PerLayer::Uniform(head_count_kv_per_layer[il]);

                let attn_opts = StandardAttentionOpts {
                    use_v_norm: true,
                    use_gemma_norms: false,
                    attention_scale: Some(1.0),
                };
                let attn = StandardAttention::load_with_opts(
                    &lgg,
                    &block_prefix,
                    &layer_cfg,
                    il,
                    rotary,
                    attn_opts,
                )?;

                let attn_norm =
                    lgg.rms_norm(&format!("{block_prefix}.attn_norm.weight"), rms_norm_eps)?;
                let post_attention_norm = lgg.rms_norm(
                    &format!("{block_prefix}.post_attention_norm.weight"),
                    rms_norm_eps,
                )?;
                let ffn_norm =
                    lgg.rms_norm(&format!("{block_prefix}.ffn_norm.weight"), rms_norm_eps)?;
                let post_ffn_norm = lgg
                    .rms_norm(&format!("{block_prefix}.post_ffw_norm.weight"), rms_norm_eps)?;

                // Gemma4 FFN uses GeGLU.
                let mlp = DenseMlp::load_with_activation(
                    &lgg,
                    &block_prefix,
                    MlpActivation::Gelu,
                )?;

                let layer_output_scale =
                    lgg.try_dequantize(&format!("{block_prefix}.layer_output_scale.weight"));

                let per_layer_embed = if n_embd_per_layer > 0 {
                    let inp_gate = lgg.try_qmatmul(&format!("{block_prefix}.inp_gate.weight"));
                    let proj = lgg.try_qmatmul(&format!("{block_prefix}.proj.weight"));
                    let pn = lgg.try_rms_norm(
                        &format!("{block_prefix}.post_norm.weight"),
                        rms_norm_eps,
                    );
                    match (inp_gate, proj, pn) {
                        (Some(inp_gate), Some(proj), Some(post_norm)) => Some(PerLayerEmbed {
                            inp_gate,
                            proj,
                            post_norm,
                        }),
                        _ => None,
                    }
                } else {
                    None
                };

                Ok(LayerWeights {
                    attn,
                    attn_norm,
                    post_attention_norm,
                    ffn_norm,
                    post_ffn_norm,
                    mlp,
                    sliding_window_size: if is_sliding { Some(sliding_window_size) } else { None },
                    has_kv,
                    kv_source_idx,
                    layer_output_scale,
                    per_layer_embed,
                    device: layer_device,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let _ = head_count; // silence "unused" if metadata reads more than we use

        // ----- output norm + lm_head live on the device of the last layer --
        let last_dev = devices[layer_to_device[block_count - 1]].clone();
        let last_gg = gg.with_device(last_dev.clone());
        let norm = last_gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = last_gg
            .try_qmatmul("output.weight")
            .or_else(|| last_gg.try_qmatmul("token_embd.weight"))
            .ok_or_else(|| candle::Error::Msg("missing output.weight and token_embd.weight".into()))?;
        let output = lm_head_tensor;

        // Pre-allocated KV cache slot per layer. Slot is `None` for
        // shared-KV layers (which borrow from another layer's slot).
        // Initialized lazily on first append in the forward.
        let kv_caches: Vec<Option<candle_nn::kv_cache::KvCache>> =
            (0..block_count).map(|_| None).collect();

        Ok(Self {
            tok_embeddings: Embedding::new_unused(),
            embedding_length,
            layers,
            norm,
            output,
            final_logit_softcap,
            per_layer_embeddings,
            kv_caches,
        }
        .with_tok_embeddings(tok_embeddings))
    }

    fn with_tok_embeddings(mut self, e: Embedding) -> Self {
        self.tok_embeddings = e;
        self
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        // Token embedding lookup (lives on CPU to avoid VRAM blow-up on the
        // larger 31B variants). Move the looked-up rows to the first layer's
        // device — that's where the actual transformer work begins.
        let cpu = candle::Device::Cpu;
        let x_cpu = if device_eq(x.device(), &cpu) {
            x.clone()
        } else {
            x.to_device(&cpu)?
        };
        let layer_in_cpu = self.tok_embeddings.forward(&x_cpu)?;
        let first_layer_dev = self.layers[0].device.clone();
        let mut layer_in = layer_in_cpu.to_device(&first_layer_dev)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        // ----- per-layer embedding (E4B) — compute once on dev0 ------------
        let inp_per_layer: Option<Tensor> = if let Some(ref ple) = self.per_layer_embeddings {
            let n_layer = self.layers.len();
            let n_embd_per_layer = ple.n_embd_per_layer;
            let model_device = layer_in.device().clone();

            // Token embedding lookup on CPU (per_layer_token_embd is huge).
            let cpu = candle::Device::Cpu;
            let x_cpu = x.to_device(&cpu)?;
            let pl_tok_embd = Embedding::new(ple.token_embd.clone(), n_embd_per_layer * n_layer);
            let inp_pe_cpu = pl_tok_embd.forward(&x_cpu)?;
            let inp_pe_cpu = inp_pe_cpu.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
            let inp_pe = (inp_pe_cpu * (n_embd_per_layer as f64).sqrt())?
                .to_device(&model_device)?;

            // Project main hidden through per_layer_model_proj on dev0.
            let proj_out = ple.model_proj.forward(&layer_in)?;
            let proj_out = (proj_out * (1.0 / (self.embedding_length as f64).sqrt()))?;
            let proj_out = proj_out.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
            let proj_out = ple.proj_norm.forward(&proj_out)?;

            // Combine and scale by 1/sqrt(2).
            let combined = ((proj_out + inp_pe)? * (1.0 / 2f64.sqrt()))?;
            Some(combined)
        } else {
            None
        };

        // ----- transformer block loop --------------------------------------
        for il in 0..self.layers.len() {
            let (has_kv, kv_source_idx, layer_device, sliding_window_size) = {
                let l = &self.layers[il];
                (l.has_kv, l.kv_source_idx, l.device.clone(), l.sliding_window_size)
            };

            // Pipeline-parallel: move residual stream to this layer's device.
            if !device_eq(layer_in.device(), &layer_device) {
                layer_in = layer_in.to_device(&layer_device)?;
            }

            // Build the per-device causal mask (sliding-aware).
            //
            // Skip the mask for the trivial decode-on-global-layer case
            // (single token, no SWA): a 1-row causal mask is all zeros and the
            // attention is correct without it. For sliding layers we MUST build
            // a mask even at decode, otherwise the query attends to keys older
            // than the window — see llama.cpp `gemma4-iswa.cpp` SWA mask path.
            let attention_mask = if seq_len == 1 && sliding_window_size.is_none() {
                None
            } else {
                Some(causal_mask(
                    b_sz,
                    seq_len,
                    index_pos,
                    sliding_window_size,
                    layer_in.dtype(),
                    &layer_device,
                )?)
            };

            // -------- attention block --------
            let residual = layer_in.clone();
            let x_norm = self.layers[il].attn_norm.forward(&layer_in)?;

            // Compute fresh K/V if this layer owns its cache and append
            // them to the pre-allocated `KvCache` (which uses
            // `slice_set` into a fixed buffer instead of `Tensor::cat`,
            // so the buffer pointer stays stable across decode steps).
            if has_kv {
                if let Some((k_new, v_new)) =
                    self.layers[il].attn.compute_kv(&x_norm, index_pos)?
                {
                    // First-touch initialization at index_pos == 0:
                    // wipe any state from a previous generation. The
                    // `KvCache::reset` is `O(1)` and doesn't free the
                    // backing buffer.
                    if index_pos == 0 {
                        if let Some(ref mut c) = self.kv_caches[il] {
                            c.reset();
                        }
                    }
                    let cache = self.kv_caches[il]
                        .get_or_insert_with(|| {
                            // dim=2 = sequence dim of (B, n_kv_head, T, head_dim).
                            // 4096 covers the typical chat context;
                            // KvCache grows automatically beyond.
                            candle_nn::kv_cache::KvCache::new(2, 4096)
                        });
                    // KvCache::append needs contiguous sources.
                    let k_new = if k_new.is_contiguous() { k_new } else { k_new.contiguous()? };
                    let v_new = if v_new.is_contiguous() { v_new } else { v_new.contiguous()? };
                    let _ = cache.append(&k_new, &v_new)?;
                }
            }

            // Read K/V from this layer's slot or borrow from the source slot
            // (transferring across devices if needed).
            let (k_use, v_use) = {
                let src = if has_kv { il } else { kv_source_idx };
                let cache = self.kv_caches[src]
                    .as_ref()
                    .expect("KV cache must exist for has_kv source layer");
                let k = cache
                    .k()?
                    .ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                let v = cache
                    .v()?
                    .ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                let k = if device_eq(k.device(), &layer_device) {
                    k
                } else {
                    k.to_device(&layer_device)?
                };
                let v = if device_eq(v.device(), &layer_device) {
                    v
                } else {
                    v.to_device(&layer_device)?
                };
                (k, v)
            };

            let attn = self.layers[il].attn.forward_with_kv(
                &x_norm,
                &k_use,
                &v_use,
                attention_mask.as_ref(),
                index_pos,
            )?;
            // Fused post-attn norm + residual add (Q0a). On HIP this is
            // one launch; falls back to rms_norm + add otherwise.
            let x = self.layers[il]
                .post_attention_norm
                .forward_post_residual(&attn, &residual)?;

            // -------- FFN block --------
            let residual = x.clone();
            let x = self.layers[il].ffn_norm.forward(&x)?;
            let x = self.layers[il].mlp.forward(&x)?;
            // Fused post-ffn norm + residual add (Q0a).
            let mut x = self.layers[il]
                .post_ffn_norm
                .forward_post_residual(&x, &residual)?;

            // -------- per-layer embedding injection (E4B) --------
            if let (Some(ref ple), Some(ref ipl)) =
                (&self.layers[il].per_layer_embed, &inp_per_layer)
            {
                let pe_in = x.clone();
                let pe_cur = ple.inp_gate.forward(&x)?;
                let pe_cur = pe_cur.gelu()?;
                let inp_this_layer = ipl.narrow(2, il, 1)?.squeeze(2)?;
                let inp_this_layer = if device_eq(inp_this_layer.device(), &layer_device) {
                    inp_this_layer
                } else {
                    inp_this_layer.to_device(&layer_device)?
                };
                let pe_cur = (pe_cur * inp_this_layer)?;
                let pe_cur = ple.proj.forward(&pe_cur)?;
                let pe_cur = ple.post_norm.forward(&pe_cur)?;
                x = (pe_in + pe_cur)?;
            }

            // -------- layer output scale (gemma4) --------
            if let Some(ref scale) = self.layers[il].layer_output_scale {
                x = x.broadcast_mul(scale)?;
            }

            layer_in = x;
        }

        // ----- final norm + lm_head + softcap ------------------------------
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let logits = self.output.forward(&x)?;

        if let Some(soft_cap) = self.final_logit_softcap {
            (&logits / soft_cap)?.tanh()? * soft_cap
        } else {
            Ok(logits)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for slot in self.kv_caches.iter_mut() {
            if let Some(c) = slot.as_mut() {
                c.reset();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `Embedding::new_unused` – placeholder used during construction
// ---------------------------------------------------------------------------
//
// `ModelWeights::from_gguf_multi_device` builds the struct in two stages so we
// can move the dequantized embedding tensor in *after* the rest of the fields
// are populated. This avoids cloning the (potentially large) embedding twice.

trait EmbeddingUnused {
    fn new_unused() -> Self;
}

impl EmbeddingUnused for Embedding {
    fn new_unused() -> Self {
        let dev = candle::Device::Cpu;
        let zero = Tensor::zeros((1, 1), DType::F32, &dev).unwrap();
        Embedding::new(zero, 1)
    }
}
