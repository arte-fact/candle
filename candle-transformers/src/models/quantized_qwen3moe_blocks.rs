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
use candle::quantized::gguf_file;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::Embedding;
use std::io::{Read, Seek};
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
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_multi_device(ct, reader, &[device.clone()])
    }

    /// Load with pipeline-parallel layer split across multiple devices.
    pub fn from_gguf_multi_device<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("from_gguf_multi_device requires at least one device");
        }
        let dev0 = &devices[0];
        let mut gg = Gguf::new(ct, reader, dev0.clone());
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
        gg.set_device(dev0.clone());
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(dev0)?, cfg.hidden_size);

        let mut layers = Vec::with_capacity(cfg.block_count);
        for il in 0..cfg.block_count {
            let prefix = format!("blk.{il}");
            let layer_dev_idx = layer_to_device[il];
            let layer_device = devices[layer_dev_idx].clone();
            gg.set_device(layer_device.clone());

            let attn = StandardAttention::load(
                &mut gg,
                &prefix,
                &cfg,
                il,
                rotary_per_device[layer_dev_idx].clone(),
                /* use_v_norm */ false,
            )?;

            let attn_norm =
                gg.rms_norm(&format!("{prefix}.attn_norm.weight"), cfg.rms_norm_eps)?;
            let ffn_norm =
                gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), cfg.rms_norm_eps)?;
            let ffn = MoeExperts::load(&mut gg, &prefix, &cfg)?;

            layers.push(Layer {
                attn_norm,
                attn,
                kv_cache: ConcatKvCache::new(2),
                ffn_norm,
                ffn,
                device: layer_device,
            });
        }

        // Output norm + lm_head on the device of the last layer.
        let last_dev = devices[layer_to_device[cfg.block_count - 1]].clone();
        gg.set_device(last_dev.clone());
        let norm = gg.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => gg.tensor("token_embd.weight")?,
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

        for layer in &mut self.layers {
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
            let attn_out = layer.attn.forward(
                &normed,
                &mut layer.kv_cache,
                mask_for_layer.as_ref(),
                offset,
            )?;
            h = (&h + attn_out)?;

            let normed = layer.ffn_norm.forward(&h)?;
            let ffn_out = layer.ffn.forward(&normed)?;
            h = (&h + ffn_out)?;
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
