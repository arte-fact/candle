//! Quantized Qwen3.5 MoE model — hybrid GDN + full attention with MoE FFN.
//!
//! Same hybrid attention structure as qwen35 (GDN + gated full-attention),
//! but uses MoE with shared experts instead of dense FFN.
//!
//! GGUF arch strings: "qwen35moe"

use super::quantized_blocks::*;
use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use std::io::{Read, Seek};
use std::sync::Arc;

enum AttnBlock {
    FullAttn(GatedAttention),
    Recurrent(DeltaNetLayer),
}

struct Layer {
    attn_norm: RmsNorm,
    attn: AttnBlock,
    post_attn_norm: RmsNorm,
    ffn: MoeExperts,
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
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_multi_device(ct, reader, &[device.clone()])
    }

    /// Load with pipeline-parallel layer split across multiple devices.
    /// Layers are assigned in contiguous chunks (LAYER split mode).
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

        // Compute layer-to-device mapping (chunked, first `extra` devices get +1).
        let n_dev = devices.len();
        let block_count = cfg.block_count;
        let base = block_count / n_dev;
        let extra = block_count % n_dev;
        let layer_to_device: Vec<usize> = {
            let mut v = Vec::with_capacity(block_count);
            for d in 0..n_dev {
                let count = base + if d < extra { 1 } else { 0 };
                for _ in 0..count {
                    v.push(d);
                }
            }
            v
        };

        // Build a RotaryEmbedding per device — full-attention layers on each
        // device share that device's instance.
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
        gg.set_device(dev0.clone());
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(dev0)?, cfg.hidden_size);

        // Track device per recurrent layer for GdnState init.
        let mut recurrent_devices: Vec<Device> = Vec::new();

        let mut layers = Vec::with_capacity(block_count);
        let mut recurrent_idx = 0;

        for il in 0..block_count {
            let prefix = format!("blk.{il}");
            let layer_dev_idx = layer_to_device[il];
            let layer_device = devices[layer_dev_idx].clone();
            gg.set_device(layer_device.clone());

            let attn = if cfg.is_recurrent(il) {
                let gdn = DeltaNetLayer::load(&mut gg, &prefix, &cfg, recurrent_idx)?;
                recurrent_idx += 1;
                recurrent_devices.push(layer_device.clone());
                AttnBlock::Recurrent(gdn)
            } else {
                AttnBlock::FullAttn(GatedAttention::load(
                    &mut gg,
                    &prefix,
                    &cfg,
                    il,
                    rotary_per_device[layer_dev_idx].clone(),
                )?)
            };

            layers.push(Layer {
                attn_norm: gg.rms_norm(&format!("{prefix}.attn_norm.weight"), cfg.rms_norm_eps)?,
                attn,
                post_attn_norm: gg.rms_norm(
                    &format!("{prefix}.post_attention_norm.weight"),
                    cfg.rms_norm_eps,
                )?,
                ffn: MoeExperts::load(&mut gg, &prefix, &cfg)?,
                device: layer_device,
            });
        }

        // Output norm + lm_head on the device of the last layer.
        let last_dev = devices[layer_to_device[block_count - 1]].clone();
        gg.set_device(last_dev.clone());
        let norm = gg.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => gg.tensor("token_embd.weight")?,
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

        // Cache one mask per device (only built when seq_len > 1).
        // Single-GPU stays in this map under one key.
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
            h = (&h + attn_out)?;

            let normed = layer.post_attn_norm.forward(&h)?;
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
