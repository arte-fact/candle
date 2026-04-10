//! Quantized Qwen3.5 (dense) model — hybrid GDN + full attention architecture.
//!
//! Qwen3.5 alternates between Gated Delta Net (linear attention) layers and
//! standard gated full-attention layers, controlled by `full_attention_interval`.
//! All layers use dense SwiGLU FFN.
//!
//! Architecture (from llama.cpp qwen35.cpp):
//! - Recurrent layers: GDN with conv1d → delta_net → gated_norm
//! - Full attention layers: Q+gate fused → sigmoid gating → output projection
//! - All layers: pre_norm → attn → residual → post_attn_norm → dense_ffn → residual
//!
//! GGUF arch string: "qwen35"

use super::quantized_blocks::*;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, GgufBlob};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use rayon::prelude::*;
use std::sync::Arc;

enum AttnBlock {
    FullAttn(GatedAttention),
    Recurrent(DeltaNetLayer),
}

struct Layer {
    attn_norm: RmsNorm,
    attn: AttnBlock,
    post_attn_norm: RmsNorm,
    ffn: DenseMlp,
    /// Device this layer's weights live on (pipeline-parallel split).
    device: Device,
}

pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: super::with_tracing::QMatMul,
    gdn_state: GdnState,
    #[allow(dead_code)]
    config: GgufConfig,
    #[allow(dead_code)]
    device: Device,
}

/// Compare two devices for equality without allocating.
fn device_eq(a: &Device, b: &Device) -> bool {
    format!("{:?}", a.location()) == format!("{:?}", b.location())
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
    /// Layer weights are loaded in parallel via rayon (mmap-backed `Gguf`).
    pub fn from_gguf_multi_device(
        ct: gguf_file::Content,
        blob: Arc<GgufBlob>,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("from_gguf_multi_device requires at least one device");
        }
        let dev0 = &devices[0];
        let gg = Gguf::new(ct, blob, dev0.clone());
        let cfg = GgufConfig::from_metadata(gg.metadata())?;

        let layer_to_device = split_layers_across_devices(cfg.block_count, devices.len());
        let block_count = cfg.block_count;

        // Build a RotaryEmbedding per device for full-attention layers.
        let rope_dim = cfg.rope_dimension_count.unwrap_or(cfg.head_dim);
        let rope_base = cfg.rope_freq_base.unwrap_or(10000.0);
        let max_seq = cfg.max_seq_len();
        let rotary_per_device: Vec<Arc<RotaryEmbedding>> = devices
            .iter()
            .map(|dev| -> Result<Arc<RotaryEmbedding>> {
                Ok(Arc::new(if let Some(ref sections) = cfg.rope_sections {
                    RotaryEmbedding::new_multi_freq(
                        rope_base, sections, rope_dim, max_seq, cfg.dtype, dev,
                    )?
                } else {
                    RotaryEmbedding::new(rope_base, rope_dim, max_seq, cfg.dtype, dev)?
                }))
            })
            .collect::<Result<Vec<_>>>()?;

        // Embedding lives on dev0.
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(dev0)?, cfg.hidden_size);

        // Pre-compute the recurrent index for each layer so the parallel
        // load doesn't need a shared counter. The mapping is: walk layers
        // 0..block_count and assign a fresh index to each `is_recurrent`.
        let mut recurrent_idx_for_layer: Vec<Option<usize>> = vec![None; block_count];
        let mut recurrent_devices: Vec<Device> = Vec::new();
        {
            let mut next_recurrent_idx = 0usize;
            for il in 0..block_count {
                if cfg.is_recurrent(il) {
                    recurrent_idx_for_layer[il] = Some(next_recurrent_idx);
                    recurrent_devices.push(devices[layer_to_device[il]].clone());
                    next_recurrent_idx += 1;
                }
            }
        }

        // Parallel per-layer load. Each rayon worker gets its own cheap
        // Gguf clone retargeted to that layer's device.
        let layers: Vec<Layer> = (0..block_count)
            .into_par_iter()
            .map(|il| -> Result<Layer> {
                let prefix = format!("blk.{il}");
                let layer_dev_idx = layer_to_device[il];
                let layer_device = devices[layer_dev_idx].clone();
                let lgg = gg.with_device(layer_device.clone());

                let attn = if let Some(rec_idx) = recurrent_idx_for_layer[il] {
                    AttnBlock::Recurrent(DeltaNetLayer::load(&lgg, &prefix, &cfg, rec_idx)?)
                } else {
                    AttnBlock::FullAttn(GatedAttention::load(
                        &lgg,
                        &prefix,
                        &cfg,
                        il,
                        rotary_per_device[layer_dev_idx].clone(),
                    )?)
                };

                Ok(Layer {
                    attn_norm: lgg
                        .rms_norm(&format!("{prefix}.attn_norm.weight"), cfg.rms_norm_eps)?,
                    attn,
                    post_attn_norm: lgg.rms_norm(
                        &format!("{prefix}.post_attention_norm.weight"),
                        cfg.rms_norm_eps,
                    )?,
                    ffn: DenseMlp::load(&lgg, &prefix)?,
                    device: layer_device,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Output norm + lm_head on the device of the last layer.
        let last_dev = devices[layer_to_device[block_count - 1]].clone();
        let last_gg = gg.with_device(last_dev.clone());
        let norm = last_gg.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head_tensor = match last_gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => last_gg.tensor("token_embd.weight")?,
        };
        let lm_head = super::with_tracing::QMatMul::from_weights(lm_head_tensor.into())?;

        let gdn_state = GdnState::new_multi_device(&cfg, 1, &recurrent_devices)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            gdn_state,
            config: cfg,
            device: dev0.clone(),
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        // One mask per device, built lazily and only when seq_len > 1.
        let mut mask_cache: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for layer in &mut self.layers {
            // Pipeline-parallel: transfer the residual stream onto this layer's device.
            if !device_eq(h.device(), &layer.device) {
                h = h.to_device(&layer.device)?;
            }

            let mask_for_layer = if l == 1 {
                None
            } else {
                let key = format!("{:?}", layer.device.location());
                if !mask_cache.contains_key(&key) {
                    let m = causal_mask(b, l, offset, None, DType::F32, &layer.device)?;
                    mask_cache.insert(key.clone(), m);
                }
                mask_cache.get(&key).cloned()
            };

            let normed = layer.attn_norm.forward(&h)?;
            let attn_out = match &mut layer.attn {
                AttnBlock::FullAttn(a) => a.forward(&normed, mask_for_layer.as_ref(), offset)?,
                AttnBlock::Recurrent(gdn) => gdn.forward_step(&normed, &mut self.gdn_state)?,
            };
            // Fused: post_attn_norm(h + attn_out) + (h + attn_out) in one
            // HIP launch. Saves the separate residual-add launch +
            // intermediate alloc per layer, per token.
            let (normed, h_after_attn) = layer.post_attn_norm.forward_residual(&h, &attn_out)?;
            h = h_after_attn;

            let ffn_out = layer.ffn.forward(&normed)?;
            h = (&h + ffn_out)?;
        }

        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.attn {
                AttnBlock::FullAttn(a) => a.clear_kv_cache(),
                AttnBlock::Recurrent(_) => {}
            }
        }
        self.gdn_state.reset().ok();
    }
}
