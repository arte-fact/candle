//! Quantized Qwen3 MoE — full attention every layer + MoE FFN, no GDN.
//!
//! Architecture variant used by `Qwen3-Coder-30B-A3B-Instruct` and similar
//! `qwen3moe` GGUFs. Differs from `quantized_qwen35_moe` (which is hybrid
//! GDN + GatedAttention) and from the legacy `quantized_qwen3_moe` (which
//! depends on the CUDA-only `FusedMoeGGUF`).
//!
//! Per-layer structure:
//! ```text
//!     attn_norm        → StandardAttention (Q/K/V with QK norms) + residual
//!     ffn_norm         → MoeExperts                              + residual
//! ```
//!
//! GGUF arch string: `"qwen3moe"`. Multi-GPU pipeline-parallel via per-layer
//! device assignment. Uses the modular `quantized_blocks` system end-to-end.

use super::quantized_blocks::{
    causal_mask, split_layers_across_devices, Gguf, GgufConfig, MoeExperts, RotaryEmbedding,
    StandardAttention,
};
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, GgufBlob};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::Embedding;
use rayon::prelude::*;
use std::sync::Arc;

struct Layer {
    attn_norm: RmsNorm,
    attn: StandardAttention,
    kv_cache: ConcatKvCache,
    ffn_norm: RmsNorm,
    ffn: MoeExperts,
    /// Device this layer's weights live on (pipeline-parallel split).
    device: Device,
}

pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: super::with_tracing::QMatMul,
    #[allow(dead_code)]
    config: GgufConfig,
    #[allow(dead_code)]
    device: Device,
}

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
    ///
    /// Layer weights are loaded in parallel via rayon: each block in the
    /// transformer fans out to a worker thread that pulls every tensor for
    /// that block from the shared mmap'd [`GgufBlob`] in one shot. With 4
    /// MI50s and a 17 GB Q4_K_XL GGUF this drops model build time from
    /// ~17 s (sequential) to ~3 s.
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

        // Build a RotaryEmbedding per device — full-attention layers on each
        // device share that device's instance.
        let rope_dim = cfg.rope_dimension_count.unwrap_or(cfg.head_dim);
        let rope_base = cfg.rope_freq_base.unwrap_or(10000.0);
        let max_seq = cfg.max_seq_len();
        let rotary_per_device: Vec<Arc<RotaryEmbedding>> = devices
            .iter()
            .map(|dev| {
                Ok(Arc::new(RotaryEmbedding::new(
                    rope_base, rope_dim, max_seq, cfg.dtype, dev,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;

        // Embedding lives on dev0.
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(dev0)?, cfg.hidden_size);

        // Parallel per-layer load. Each rayon worker gets its own cheap Gguf
        // clone (Arc bumps only) retargeted to that layer's device.
        let layers: Vec<Layer> = (0..cfg.block_count)
            .into_par_iter()
            .map(|il| -> Result<Layer> {
                let prefix = format!("blk.{il}");
                let layer_dev_idx = layer_to_device[il];
                let layer_device = devices[layer_dev_idx].clone();
                let lgg = gg.with_device(layer_device.clone());

                let attn = StandardAttention::load(
                    &lgg,
                    &prefix,
                    &cfg,
                    il,
                    rotary_per_device[layer_dev_idx].clone(),
                    /* use_v_norm */ false,
                )?;
                let attn_norm =
                    lgg.rms_norm(&format!("{prefix}.attn_norm.weight"), cfg.rms_norm_eps)?;
                let ffn_norm =
                    lgg.rms_norm(&format!("{prefix}.ffn_norm.weight"), cfg.rms_norm_eps)?;
                let ffn = MoeExperts::load(&lgg, &prefix, &cfg)?;

                Ok(Layer {
                    attn_norm,
                    attn,
                    kv_cache: ConcatKvCache::new(2),
                    ffn_norm,
                    ffn,
                    device: layer_device,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Output norm + lm_head on the device of the last layer.
        let last_dev = devices[layer_to_device[cfg.block_count - 1]].clone();
        let last_gg = gg.with_device(last_dev.clone());
        let norm = last_gg.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head_tensor = match last_gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => last_gg.tensor("token_embd.weight")?,
        };
        let lm_head = super::with_tracing::QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: cfg,
            device: dev0.clone(),
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        // Cache one mask per device (only built when seq_len > 1).
        let mut mask_cache: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Across-layer fusion: hoist the first attn_norm out of the loop
        // and produce next_normed_for_attn as part of the post-FFN fused
        // op. See quantized_qwen35.rs for the structural reasoning.
        let layers_len = self.layers.len();
        let layer0_dev = self.layers[0].device.clone();
        if !device_eq(h.device(), &layer0_dev) {
            h = h.to_device(&layer0_dev)?;
        }
        let mut normed_for_attn = self.layers[0].attn_norm.forward(&h)?;

        for il in 0..layers_len {
            let layer_dev = self.layers[il].device.clone();

            let mask_for_layer = if l == 1 {
                None
            } else {
                let key = format!("{:?}", layer_dev.location());
                if !mask_cache.contains_key(&key) {
                    let m = causal_mask(b, l, offset, None, DType::F32, &layer_dev)?;
                    mask_cache.insert(key.clone(), m);
                }
                mask_cache.get(&key).cloned()
            };

            let attn_out = {
                let layer = &mut self.layers[il];
                layer.attn.forward(
                    &normed_for_attn,
                    &mut layer.kv_cache,
                    mask_for_layer.as_ref(),
                    offset,
                )?
            };
            // Within-layer fusion: ffn_norm(h + attn_out) + (h + attn_out)
            let (normed_for_ffn, h_after_attn) =
                self.layers[il].ffn_norm.forward_residual(&h, &attn_out)?;
            h = h_after_attn;

            let ffn_out = self.layers[il].ffn.forward(&normed_for_ffn)?;

            let is_last = il + 1 == layers_len;
            if is_last {
                h = (&h + ffn_out)?;
            } else {
                let next_dev = self.layers[il + 1].device.clone();
                if device_eq(&next_dev, &layer_dev) {
                    let (next_normed, h_after_ffn) = self.layers[il + 1]
                        .attn_norm
                        .forward_residual(&h, &ffn_out)?;
                    h = h_after_ffn;
                    normed_for_attn = next_normed;
                } else {
                    h = (&h + ffn_out)?;
                    h = h.to_device(&next_dev)?;
                    normed_for_attn = self.layers[il + 1].attn_norm.forward(&h)?;
                }
            }
        }

        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache.reset();
        }
    }
}
