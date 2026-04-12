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
use crate::models::with_tracing::QMatMul;
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

/// Optional MoE branch present in gemma4-A4B (26B) but absent in
/// gemma4-E4B (4B dense). When present, the FFN block is dual:
///   out = post_norm_1(dense_mlp(norm_1(attn_out)))
///       + post_norm_2(moe_ffn(norm_2(attn_out), router(attn_out)))
/// Both branches read from `attn_out` (attention residual), NOT from
/// each other. The router uses a custom logit computation:
///   logits = gate_inp @ (rms_norm(attn_out) / sqrt(n_embd) * gate_inp_s)
struct MoeBranch {
    /// Router weight matrix [n_experts, hidden_dim].
    gate_inp: QMatMul,
    /// Router input scale: learned per-element scale applied to the
    /// rms-normed input before the router matmul.
    gate_inp_s: Tensor,
    /// Fused expert gate+up [n_experts, 2*expert_ffn_dim, hidden_dim].
    gate_up_exps: std::sync::Arc<candle::quantized::QTensor>,
    /// Expert down [n_experts, hidden_dim, expert_ffn_dim].
    down_exps: std::sync::Arc<candle::quantized::QTensor>,
    /// Optional per-expert down projection scale (ffn_down_exps.scale).
    /// Applied element-wise to the down projection output before
    /// combining across topk experts.
    down_exps_scale: Option<Tensor>,
    /// Expert FFN intermediate size (per expert).
    expert_intermediate: usize,
    /// Number of experts used per token.
    num_experts_used: usize,
    /// Pre-MoE norm (pre_ffw_norm_2).
    pre_norm: RmsNorm,
    /// Post-MoE norm (post_ffw_norm_2).
    post_norm: RmsNorm,
    /// Embedding length for the 1/sqrt(n_embd) router scale.
    n_embd: usize,
    /// RMS norm epsilon (reused for the inline router rms_norm).
    eps: f32,
}

struct LayerWeights {
    attn: StandardAttention,
    attn_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: DenseMlp,
    /// Optional MoE expert branch for A4B-class models.
    moe: Option<MoeBranch>,
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

                // Detect MoE layer: present if ffn_gate_inp.weight exists.
                let moe = if lgg.has_tensor(&format!("{block_prefix}.ffn_gate_inp.weight")) {
                    let gate_inp = lgg.qmatmul(&format!("{block_prefix}.ffn_gate_inp.weight"))?;
                    let gate_inp_s = lgg
                        .try_dequantize(&format!("{block_prefix}.ffn_gate_inp.scale"))
                        .ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "MoE layer {block_prefix} has gate_inp but no gate_inp.scale"
                            ))
                        })?;
                    let gate_up_exps = std::sync::Arc::new(
                        lgg.tensor(&format!("{block_prefix}.ffn_gate_up_exps.weight"))?,
                    );
                    let down_exps = std::sync::Arc::new(
                        lgg.tensor(&format!("{block_prefix}.ffn_down_exps.weight"))?,
                    );
                    // Expert FFN intermediate: infer from gate_up_exps shape
                    // [n_experts, 2*intermediate, hidden] → intermediate = dim[1]/2
                    let expert_intermediate = gate_up_exps.shape().dims()[1] / 2;
                    let num_experts_used = cfg.expert_used_count.unwrap_or(8);
                    let pre_norm = lgg.rms_norm(
                        &format!("{block_prefix}.pre_ffw_norm_2.weight"),
                        rms_norm_eps,
                    )?;
                    let post_norm = lgg.rms_norm(
                        &format!("{block_prefix}.post_ffw_norm_2.weight"),
                        rms_norm_eps,
                    )?;
                    let down_exps_scale =
                        lgg.try_dequantize(&format!("{block_prefix}.ffn_down_exps.scale"));
                    Some(MoeBranch {
                        gate_inp,
                        gate_inp_s,
                        gate_up_exps,
                        down_exps,
                        down_exps_scale,
                        expert_intermediate,
                        num_experts_used,
                        pre_norm,
                        post_norm,
                        n_embd: embedding_length,
                        eps: rms_norm_eps as f32,
                    })
                } else {
                    None
                };

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
                    moe,
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
                            // Attack C: K is stored pre-transposed
                            // (B, n_kv_head, D, T) so attention can skip
                            // the per-call `k.t().contiguous()` materialise.
                            // `dim_v=2` = sequence dim of V's
                            // `(B, n_kv_head, T, head_dim)`. K lives at
                            // dim_v+1 = 3. 4096 covers the typical chat
                            // context; KvCache grows automatically beyond.
                            candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096)
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

            // Attack C: K is pre-transposed in the cache, so use the
            // `*_transposed` variant which skips the internal
            // `k.t().contiguous()` materialisation inside attention.
            let attn = self.layers[il].attn.forward_with_kv_transposed(
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
            let skip_moe = std::env::var("CANDLE_GEMMA4_SKIP_MOE").is_ok();
            let mut x = if !skip_moe && self.layers[il].moe.is_some() {
                let moe_branch = self.layers[il].moe.as_ref().unwrap();
                // Dual dense+MoE: both branches read from `x` (the
                // attention residual output, a.k.a. attn_out in turbo).
                let attn_out = x;

                // Branch 1: Dense MLP
                let dense_in = self.layers[il].ffn_norm.forward(&attn_out)?;
                let dense_out = self.layers[il].mlp.forward(&dense_in)?;
                let dense_normed = self.layers[il].post_ffn_norm.forward(&dense_out)?;

                // Branch 2: MoE experts
                let moe_in = moe_branch.pre_norm.forward(&attn_out)?;

                // Custom router logits:
                //   tmp = rms_norm(attn_out) / sqrt(n_embd) * gate_inp_s
                //   logits = gate_inp @ tmp
                let router_input = {
                    let normed = crate::models::quantized_blocks::norms::v_norm(
                        &attn_out,
                        moe_branch.eps as f64,
                    )?;
                    let scaled = (normed * (1.0 / (moe_branch.n_embd as f64).sqrt()))?;
                    scaled.broadcast_mul(&moe_branch.gate_inp_s)?
                };
                let router_logits = moe_branch.gate_inp.forward(&router_input)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // TopK selection (route through CPU for argsort).
                let device = routing_weights.device().clone();
                let (b_sz_ff, seq_len_ff, _hidden) = moe_in.dims3()?;
                let flat_tokens = b_sz_ff * seq_len_ff;
                let n_experts = routing_weights.dim(candle::D::Minus1)?;
                let rw_flat = routing_weights.reshape((flat_tokens, n_experts))?;
                let topk_ids = {
                    let rw_cpu = rw_flat.to_device(&candle::Device::Cpu)?;
                    let ids = rw_cpu
                        .arg_sort_last_dim(false)?
                        .narrow(candle::D::Minus1, 0, moe_branch.num_experts_used)?
                        .contiguous()?;
                    ids.to_device(&device)?
                };
                let topk_weights = rw_flat.gather(&topk_ids, candle::D::Minus1)?;
                let topk_weights = (&topk_weights
                    / topk_weights
                        .sum_keepdim(candle::D::Minus1)?
                        .broadcast_as(topk_weights.shape())?)?;

                // Expert computation via fused gate_up_exps
                let hidden = moe_in.dim(candle::D::Minus1)?;
                let moe_in_flat = moe_in.reshape((flat_tokens, hidden))?;
                let x_3d = moe_in_flat.unsqueeze(1)?.contiguous()?;
                if il == 0 && std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
                    eprintln!("[MoE L{il}] attn_out={:?} moe_in={:?} x_3d={:?} topk_ids={:?} gate_up_exps={:?} down_exps={:?} expert_intermediate={}",
                        attn_out.shape(), moe_in.shape(), x_3d.shape(), topk_ids.shape(),
                        moe_branch.gate_up_exps.shape(), moe_branch.down_exps.shape(),
                        moe_branch.expert_intermediate);
                    // Print first token's routing
                    let ids_cpu: Vec<u32> = topk_ids.to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    let wts_cpu: Vec<f32> = topk_weights.to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    eprintln!("[MoE L{il}] token0 expert_ids={:?} weights={:.4?}", &ids_cpu[..8.min(ids_cpu.len())], &wts_cpu[..8.min(wts_cpu.len())]);
                    eprintln!("[MoE L{il}] gate_inp_s={:?}", moe_branch.gate_inp_s.shape());
                    let rl_cpu: Vec<f32> = router_logits.narrow(0, 0, 1).unwrap().to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    let top5: Vec<(usize, f32)> = {
                        let mut indexed: Vec<_> = rl_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        indexed[..5].to_vec()
                    };
                    eprintln!("[MoE L{il}] router_logits top5={:?}", top5);
                }
                let gate_up = moe_branch
                    .gate_up_exps
                    .indexed_moe_forward(&x_3d, &topk_ids)?;
                let gate = gate_up
                    .narrow(candle::D::Minus1, 0, moe_branch.expert_intermediate)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(
                        candle::D::Minus1,
                        moe_branch.expert_intermediate,
                        moe_branch.expert_intermediate,
                    )?
                    .contiguous()?;
                let activated = gate.gelu()?.broadcast_mul(&up)?;
                let mut moe_out = moe_branch
                    .down_exps
                    .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?;
                // Apply per-expert down scale if present (ffn_down_exps.scale).
                // Shape: [n_experts]; gather the scales for the selected
                // topk experts and broadcast over the hidden dim.
                if let Some(ref scale) = moe_branch.down_exps_scale {
                    // scale: [n_experts=128], topk_ids: [tokens, topk=8]
                    // Gather: [tokens, topk] per-expert scales
                    let expert_scales = scale
                        .index_select(&topk_ids.flatten_all()?, 0)?
                        .reshape(topk_ids.shape())?;
                    // Broadcast to [tokens, topk, 1] for hidden-dim multiply
                    let expert_scales = expert_scales.unsqueeze(candle::D::Minus1)?;
                    moe_out = moe_out.broadcast_mul(&expert_scales)?;
                }

                // Weight + sum across topk
                let topk_weights = topk_weights.unsqueeze(candle::D::Minus1)?;
                let weighted = moe_out.broadcast_mul(&topk_weights)?;
                let moe_summed = weighted.sum(1)?; // sum across topk dim
                let moe_summed = moe_summed.reshape((b_sz_ff, seq_len_ff, hidden))?;
                let moe_normed = moe_branch.post_norm.forward(&moe_summed)?;

                // Combine branches + residual
                if std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
                    let d_abs = dense_normed.abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_scalar::<f32>().unwrap();
                    let m_abs = moe_normed.abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_scalar::<f32>().unwrap();
                    let a_abs = attn_out.abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_scalar::<f32>().unwrap();
                    let combined_abs = ((&attn_out + &dense_normed).unwrap() + &moe_normed).unwrap().abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_scalar::<f32>().unwrap();
                    eprintln!("[L{il}] attn={a_abs:.3} dense={d_abs:.3} moe={m_abs:.3} combined={combined_abs:.3}");
                }
                ((&attn_out + &dense_normed)? + &moe_normed)?
            } else {
                // Dense-only path (E4B and similar).
                let residual = x.clone();
                let ffn_in = self.layers[il].ffn_norm.forward(&x)?;
                let ffn_out = self.layers[il].mlp.forward(&ffn_in)?;
                // Fused post-ffn norm + residual add (Q0a).
                self.layers[il]
                    .post_ffn_norm
                    .forward_post_residual(&ffn_out, &residual)?
            };

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
