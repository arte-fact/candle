//! Rotary Position Embedding variants.
//!
//! Supports standard RoPE, multi-frequency RoPE (dimension_sections),
//! and custom-frequency RoPE (from a stored tensor). All driven by GGUF metadata.

use candle::{DType, Device, Result, Tensor};

/// Rotary position embedding. Precomputes sin/cos tables for a given max sequence length.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// How many dimensions RoPE applies to (may be < head_dim for partial RoPE).
    pub dim: usize,
}

impl RotaryEmbedding {
    /// Standard RoPE: compute inverse frequencies from `freq_base` and `dim`.
    pub fn new(
        freq_base: f64,
        dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_freq_factors(freq_base, dim, max_seq_len, None, dtype, device)
    }

    /// Standard RoPE with optional per-frequency scaling factors (a.k.a. "rope_freqs"
    /// or "freq_factors" in llama.cpp).
    ///
    /// `freq_factors` must have length `dim/2`. When provided, each inverse
    /// frequency is divided by the corresponding factor: `theta_i = inv_freq[i] / ff[i]`.
    /// This implements proportional / NTK-aware rope, used by gemma4 global
    /// (non-SWA) layers via the `rope_freqs.weight` tensor.
    ///
    /// Reference: `ggml/src/ggml-cpu/ops.cpp:5628` `rope_yarn(theta/ff, …)`
    pub fn new_with_freq_factors(
        freq_base: f64,
        dim: usize,
        max_seq_len: usize,
        freq_factors: Option<&[f32]>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (freq_base as f32).powf(i as f32 / dim as f32))
            .collect();
        if let Some(ff) = freq_factors {
            if ff.len() != inv_freq.len() {
                candle::bail!(
                    "rope freq_factors length {} != dim/2 = {}",
                    ff.len(),
                    inv_freq.len()
                );
            }
            for (t, f) in inv_freq.iter_mut().zip(ff.iter()) {
                *t /= *f;
            }
        }
        Self::from_inv_freq(&inv_freq, dim, max_seq_len, dtype, device)
    }

    /// Multi-frequency RoPE with dimension sections.
    ///
    /// `sections` splits the frequency bins into groups, each potentially
    /// with different frequency ranges. Used by qwen35 full-attention layers.
    /// For now, computes standard frequencies — section-aware computation
    /// can be added as an optimization.
    pub fn new_multi_freq(
        freq_base: f64,
        _sections: &[usize],
        dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // TODO: implement section-aware frequency distribution
        // For initial correctness, use standard frequencies
        Self::new(freq_base, dim, max_seq_len, dtype, device)
    }

    /// RoPE from a stored inverse-frequency tensor (e.g. gemma4 `rope_freqs.weight`).
    pub fn from_freqs_tensor(
        freqs: &Tensor,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let freqs_vec: Vec<f32> = freqs.flatten_all()?.to_vec1()?;
        let dim = freqs_vec.len() * 2;
        Self::from_inv_freq(&freqs_vec, dim, max_seq_len, dtype, device)
    }

    /// Build sin/cos tables from an inverse-frequency vector.
    fn from_inv_freq(
        inv_freq: &[f32],
        dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Compute sin/cos tables on CPU to avoid GPU BLAS dependency during init
        let inv_freq_len = inv_freq.len();
        let cpu = &candle::Device::Cpu;
        let inv_freq_t = Tensor::from_slice(inv_freq, (1, inv_freq_len), cpu)?
            .to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, cpu)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq_t)?;
        let sin = freqs.sin()?.to_device(device)?;
        let cos = freqs.cos()?.to_device(device)?;
        Ok(Self { sin, cos, dim })
    }

    /// Apply RoPE to a single tensor (Q or K).
    ///
    /// Expected shape: `(batch, n_heads, seq_len, head_dim)`.
    /// If `head_dim > self.dim`, only the first `dim` dimensions are rotated
    /// (partial RoPE) and the remainder is passed through unchanged.
    pub fn apply_one(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(x.dtype())?;
        if head_dim == self.dim {
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        } else {
            let x_rot = x.narrow(candle::D::Minus1, 0, self.dim)?;
            let x_pass = x.narrow(candle::D::Minus1, self.dim, head_dim - self.dim)?;
            let x_rot = candle_nn::rotary_emb::rope(&x_rot.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[x_rot, x_pass], candle::D::Minus1)?.contiguous()
        }
    }

    /// Apply RoPE to Q and K tensors with the same offset.
    /// Convenience wrapper around [`Self::apply_one`].
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_e = self.apply_one(q, offset)?;
        let k_e = self.apply_one(k, offset)?;
        Ok((q_e, k_e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_new() {
        let rope = RotaryEmbedding::new(10000.0, 64, 128, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.dim, 64);
        assert_eq!(rope.sin.dims(), &[128, 32]); // max_seq x dim/2
        assert_eq!(rope.cos.dims(), &[128, 32]);
    }

    #[test]
    fn test_rope_apply_preserves_shape() {
        let rope = RotaryEmbedding::new(10000.0, 8, 64, DType::F32, &Device::Cpu).unwrap();
        // (batch=1, heads=2, seq=4, head_dim=8)
        let q = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let (q_out, k_out) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_out.dims(), &[1, 2, 4, 8]);
        assert_eq!(k_out.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_rope_apply_with_offset() {
        let rope = RotaryEmbedding::new(10000.0, 8, 64, DType::F32, &Device::Cpu).unwrap();
        let q = Tensor::randn(0f32, 1.0, (1, 2, 1, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 1, 8), &Device::Cpu).unwrap();
        // Should work with offset
        let (q_out, _) = rope.apply(&q, &k, 10).unwrap();
        assert_eq!(q_out.dims(), &[1, 2, 1, 8]);
    }

    #[test]
    fn test_rope_partial_preserves_unrotated_dims() {
        // RoPE with dim=4 applied to head_dim=8
        let rope = RotaryEmbedding::new(10000.0, 4, 64, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.dim, 4);

        let q = Tensor::ones((1, 1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 1, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let (q_out, _) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_out.dims(), &[1, 1, 1, 8]);

        // Last 4 dims should be unmodified (still 1.0)
        let last4: Vec<f32> = q_out.narrow(3, 4, 4).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for v in &last4 {
            assert!((v - 1.0).abs() < 1e-6, "pass-through dims should be unchanged");
        }
    }

    #[test]
    fn test_rope_from_freqs_tensor() {
        // 4 inverse frequencies → dim=8
        let freqs = Tensor::new(&[1.0f32, 0.5, 0.25, 0.125], &Device::Cpu).unwrap();
        let rope = RotaryEmbedding::from_freqs_tensor(&freqs, 32, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.dim, 8);
        assert_eq!(rope.sin.dims(), &[32, 4]);
    }

    #[test]
    fn test_rope_multi_freq_same_as_standard() {
        // With trivial sections, multi-freq should produce same result as standard
        let standard = RotaryEmbedding::new(10000.0, 8, 32, DType::F32, &Device::Cpu).unwrap();
        let multi = RotaryEmbedding::new_multi_freq(10000.0, &[4, 0, 0, 0], 8, 32, DType::F32, &Device::Cpu).unwrap();

        let q = Tensor::randn(0f32, 1.0, (1, 1, 4, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 1, 4, 8), &Device::Cpu).unwrap();

        let (qs, _) = standard.apply(&q, &k, 0).unwrap();
        let (qm, _) = multi.apply(&q, &k, 0).unwrap();

        let diff: f32 = (qs - qm).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(diff < 1e-5, "multi-freq with trivial sections should match standard");
    }
}
