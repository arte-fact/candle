//! Quantized Qwen3.5 MoE model — hybrid GDN + full attention with MoE FFN.
//!
//! Same hybrid attention structure as qwen35 (GDN + gated full-attention),
//! but uses MoE with shared experts instead of dense FFN.
//!
//! GGUF arch strings: "qwen35moe"

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
    pub fn from_gguf(
        ct: gguf_file::Content,
        blob: Arc<GgufBlob>,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_multi_device(ct, blob, &[device.clone()])
    }

    /// Load with pipeline-parallel layer split across multiple devices.
    /// Layer weights are loaded in parallel via rayon.
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

        let block_count = cfg.block_count;
        let layer_to_device = split_layers_across_devices(block_count, devices.len());

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
        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(dev0)?, cfg.hidden_size);

        // Pre-compute the recurrent index for each layer so the parallel
        // load doesn't need a shared counter.
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
                    ffn: MoeExperts::load(&lgg, &prefix, &cfg)?,
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

        // Cache one mask per device (only built when seq_len > 1).
        // Single-GPU stays in this map under one key.
        let mut mask_cache: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Across-layer fusion: see quantized_qwen35.rs for the structural
        // explanation. The first layer's attn_norm is hoisted out of the
        // loop, and each iteration produces the next iteration's
        // `normed_for_attn` via the fused (residual + next attn_norm)
        // path when the next layer is on the same device.
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

            let attn_out = match &mut self.layers[il].attn {
                AttnBlock::FullAttn(a) => {
                    a.forward(&normed_for_attn, mask_for_layer.as_ref(), offset)?
                }
                AttnBlock::Recurrent(gdn) => {
                    gdn.forward_step(&normed_for_attn, &mut self.gdn_state)?
                }
            };
            // Within-layer fusion.
            let (normed_for_ffn, h_after_attn) =
                self.layers[il].post_attn_norm.forward_residual(&h, &attn_out)?;
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
            match &mut layer.attn {
                AttnBlock::FullAttn(a) => a.clear_kv_cache(),
                AttnBlock::Recurrent(_) => {}
            }
        }
        self.gdn_state.reset().ok();
    }
}
