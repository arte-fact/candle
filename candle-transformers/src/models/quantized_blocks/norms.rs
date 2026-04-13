//! Shared norm and mask utilities for quantized models.

use super::gguf_loader::Gguf;
use candle::quantized::QTensor;
use candle::{DType, Device, Module, Result, Tensor, D};

/// Gemma-style RmsNorm wrapper.
///
/// In safetensors form, Gemma stores the RmsNorm weight centered around 0 and
/// adds 1.0 at runtime. **However**, llama.cpp's GGUF converter for Gemma
/// already bakes the +1 offset into the stored weights, so direct multiplication
/// is correct for GGUF. We keep this as a wrapper for clarity and to allow
/// future converters that don't bake the offset in.
#[derive(Debug, Clone)]
pub struct GemmaRmsNorm {
    weight_plus_one: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    /// Build from a quantized weight tensor.
    /// The weight is dequantized as-is — the +1 offset is assumed to be baked
    /// into the GGUF weights by llama.cpp's converter.
    pub fn from_qtensor(qweight: QTensor, eps: f64) -> Result<Self> {
        let device = qweight.device();
        let weight = qweight.dequantize(&device)?;
        Ok(Self { weight_plus_one: weight, eps })
    }

    /// Load from a GGUF tensor name.
    pub fn load(gg: &Gguf, name: &str, eps: f64) -> Result<Self> {
        let qt = gg.tensor(name)?;
        Self::from_qtensor(qt, eps)
    }

    /// Try to load from GGUF; return None if the tensor doesn't exist.
    pub fn try_load(gg: &Gguf, name: &str, eps: f64) -> Option<Self> {
        if gg.has_tensor(name) {
            Self::load(gg, name, eps).ok()
        } else {
            None
        }
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight_plus_one, self.eps as f32)
    }
}

/// Build a causal attention mask with optional sliding window.
///
/// Returns a mask of shape (batch, 1, seq_len, seq_len + offset) where
/// allowed positions are 0.0 and masked positions are -inf.
pub fn causal_mask(
    batch: usize,
    seq_len: usize,
    offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    causal_mask_padded(batch, seq_len, offset, sliding_window, dtype, device, 0)
}

/// Like [`causal_mask`] but pads the result's last dim out to at least
/// `pad_to` columns. Positions beyond `seq_len + offset` are filled with
/// -inf so the masked softmax ignores them.
///
/// Returns a mask of shape (batch, 1, seq_len, max(seq_len + offset, pad_to)).
///
/// Used by the G2 decode replay path (gemma4): the captured plan needs
/// the mask `last_dim` to be stable across replays even though
/// `seq_len + offset` grows by 1 per token until it exceeds the pad
/// threshold.
pub fn causal_mask_padded(
    batch: usize,
    seq_len: usize,
    offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
    pad_to: usize,
) -> Result<Tensor> {
    let live = seq_len + offset;
    let total = live.max(pad_to);
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total).map(move |j| {
                if j >= live {
                    return f32::NEG_INFINITY;
                }
                let causal_ok = j <= i + offset;
                let sw_ok = match sliding_window {
                    Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                    None => true,
                };
                if causal_ok && sw_ok {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (1, 1, seq_len, total), device)?;
    let mask = mask.to_dtype(dtype)?;
    if batch > 1 {
        mask.broadcast_as((batch, 1, seq_len, total))?.contiguous()
    } else {
        Ok(mask)
    }
}

/// Parameter-free RMS normalization on a tensor.
/// Used for V-norm in Gemma4 (`ggml_rms_norm` without learned weights).
pub fn v_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let norm = (mean_sq + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// L2 normalization along the last dimension.
/// Used by delta net for Q/K normalization.
///
/// On HIP with contiguous f32 inputs, lowers to a single fused kernel
/// launch via `candle::hip_backend::l2_norm_fused`. Otherwise falls
/// back to the 5-op tensor chain (sqr + sum_keepdim + add_eps + sqrt
/// + broadcast_div). Both paths are mathematically identical.
pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "hip")]
    {
        if matches!(x.device(), candle::Device::Hip(_))
            && x.dtype() == DType::F32
            && x.is_contiguous()
            && x.rank() >= 1
        {
            return candle::hip_backend::l2_norm_fused(x, eps);
        }
    }
    // Fallback: the pre-Phase-4 tensor-op chain.
    let x_sq = x.sqr()?;
    let sum_sq = x_sq.sum_keepdim(D::Minus1)?;
    let norm = (sum_sq + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_causal_mask_no_sliding_window() {
        let mask = causal_mask(1, 3, 0, None, DType::F32, &Device::Cpu).unwrap();
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // 3x3 lower-triangular: (0,0)=0, (0,1)=-inf, (0,2)=-inf, (1,0)=0, (1,1)=0, ...
        assert_eq!(vals[0], 0.0);        // (0,0) allowed
        assert!(vals[1].is_infinite());   // (0,1) masked
        assert!(vals[2].is_infinite());   // (0,2) masked
        assert_eq!(vals[3], 0.0);        // (1,0) allowed
        assert_eq!(vals[4], 0.0);        // (1,1) allowed
        assert!(vals[5].is_infinite());   // (1,2) masked
        assert_eq!(vals[6], 0.0);        // (2,0) allowed
        assert_eq!(vals[7], 0.0);        // (2,1) allowed
        assert_eq!(vals[8], 0.0);        // (2,2) allowed
    }

    #[test]
    fn test_causal_mask_with_offset() {
        let mask = causal_mask(1, 2, 3, None, DType::F32, &Device::Cpu).unwrap();
        // Shape: (1, 1, 2, 5) — 2 query positions, 5 key positions (3 past + 2 current)
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0 (query pos 0+offset=3): can attend to keys 0..3 (positions 0,1,2,3)
        assert_eq!(vals[0], 0.0);  // key 0
        assert_eq!(vals[1], 0.0);  // key 1
        assert_eq!(vals[2], 0.0);  // key 2
        assert_eq!(vals[3], 0.0);  // key 3 (self)
        assert!(vals[4].is_infinite()); // key 4 (future)
    }

    #[test]
    fn test_causal_mask_with_sliding_window() {
        let mask = causal_mask(1, 4, 0, Some(2), DType::F32, &Device::Cpu).unwrap();
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 3 (query pos 3): can attend to keys within window of 2 → keys 1,2,3
        assert!(vals[12].is_infinite()); // key 0: outside window (3-0=3 > 2)
        assert_eq!(vals[13], 0.0);       // key 1: within window
        assert_eq!(vals[14], 0.0);       // key 2: within window
        assert_eq!(vals[15], 0.0);       // key 3: self
    }

    #[test]
    fn test_causal_mask_decode_with_sliding_window() {
        // Single-token decode (seq_len=1) with offset=5 — query is at position 5,
        // KV cache has 6 keys [0..5]. Sliding window=2 means the query attends to
        // its own position and the 2 nearest past keys (3, 4, 5) only.
        let mask = causal_mask(1, 1, 5, Some(2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 6]);
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals[0].is_infinite(), "key 0 should be outside window");
        assert!(vals[1].is_infinite(), "key 1 should be outside window");
        assert!(vals[2].is_infinite(), "key 2 should be outside window");
        assert_eq!(vals[3], 0.0, "key 3 within window");
        assert_eq!(vals[4], 0.0, "key 4 within window");
        assert_eq!(vals[5], 0.0, "key 5 (self) allowed");
    }

    #[test]
    fn test_v_norm() {
        let x = Tensor::new(&[3.0f32, 4.0], &Device::Cpu).unwrap();
        let result = v_norm(&x, 1e-6).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32).sqrt();
        assert!((vals[0] - 3.0 / rms).abs() < 1e-4);
        assert!((vals[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_l2_norm() {
        let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu).unwrap();
        let result = l2_norm(&x, 1e-6).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // L2 norm = sqrt(9+16) = 5
        assert!((vals[0] - 0.6).abs() < 1e-4);
        assert!((vals[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_l2_norm_preserves_direction() {
        let x = Tensor::new(&[[1.0f32, 0.0, 0.0]], &Device::Cpu).unwrap();
        let result = l2_norm(&x, 1e-6).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-4);
        assert!(vals[1].abs() < 1e-4);
        assert!(vals[2].abs() < 1e-4);
    }

    /// P4 regression: HIP fused l2_norm kernel must match the CPU
    /// tensor-op chain `x / sqrt(Σ x² + eps)` within FMA rounding on
    /// every element. Covers multiple shapes: small (D=32), the
    /// qwen35 head_k_dim (D=128), and an odd non-multiple of 64
    /// (D=96) to exercise the kernel's strided-loop fallback.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_l2_norm_matches_cpu() {
        let dev_hip = match Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = Device::Cpu;

        for (n_rows, d) in [(8usize, 32usize), (16, 128), (32, 96), (64, 256)] {
            // Deterministic small-magnitude rows.
            let vals: Vec<f32> = (0..n_rows * d)
                .map(|i| ((i as f32) * 0.0001).sin() * 0.3)
                .collect();
            let x_cpu = Tensor::from_slice(&vals, (n_rows, d), &dev_cpu).unwrap();
            let x_hip = Tensor::from_slice(&vals, (n_rows, d), &dev_hip).unwrap();
            let eps = 1e-6;
            let out_cpu: Vec<f32> =
                l2_norm(&x_cpu, eps).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let out_hip: Vec<f32> = l2_norm(&x_hip, eps)
                .unwrap()
                .to_device(&dev_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(out_cpu.len(), out_hip.len(), "shape ({n_rows},{d})");
            let max_abs = out_cpu
                .iter()
                .zip(out_hip.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Reduction order differs between CPU (sequential) and HIP
            // (warp-shuffle butterfly), so allow a little FMA slack.
            assert!(
                max_abs < 1e-5,
                "l2_norm HIP vs CPU drift on ({n_rows},{d}): max_abs={max_abs}"
            );
        }
    }
}
