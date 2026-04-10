//! Gated Delta Net (GDN) linear attention layer.
//!
//! A recurrent linear attention mechanism used as an alternative to standard attention
//! in hybrid architectures like Qwen3.5 and Qwen3-Next. Most layers are GDN (recurrent),
//! with every N-th layer being full quadratic attention.
//!
//! Reference: llama.cpp `delta-net-base.cpp` and `qwen35.cpp`
//!
//! The GDN layer performs:
//! 1. Input projection (wqkv) → conv1d → silu → split into Q/K/V
//! 2. L2-normalize Q and K
//! 3. Compute decay gate from alpha/beta/A parameters
//! 4. Delta net recurrence: state update + output
//! 5. Gated normalization: rms_norm(output) * silu(z)
//! 6. Output projection

use super::gguf_config::GgufConfig;
use super::gguf_loader::Gguf;
use super::norms::l2_norm;
use super::super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::{Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// GdnDimensions: all derived from GgufConfig, no hardcoded values
// ---------------------------------------------------------------------------

/// All GDN dimensions derived from GGUF metadata.
#[derive(Debug, Clone)]
pub struct GdnDimensions {
    /// SSM inner dimension (d_inner)
    pub d_inner: usize,
    /// Key/query head dimension (ssm_d_state)
    pub head_k_dim: usize,
    /// Number of key/query heads (ssm_n_group)
    pub num_k_heads: usize,
    /// Number of value heads (ssm_dt_rank)
    pub num_v_heads: usize,
    /// Value head dimension (d_inner / num_v_heads)
    pub head_v_dim: usize,
    /// Conv channels (d_inner + 2 * num_k_heads * head_k_dim)
    pub conv_channels: usize,
    /// Conv kernel size
    pub conv_kernel: usize,
}

impl GdnDimensions {
    pub fn from_config(cfg: &GgufConfig) -> Result<Self> {
        let d_inner = cfg.ssm_d_inner
            .ok_or_else(|| candle::Error::Msg("GDN requires ssm.inner_size".into()))?;
        let head_k_dim = cfg.ssm_d_state
            .ok_or_else(|| candle::Error::Msg("GDN requires ssm.state_size".into()))?;
        let num_k_heads = cfg.ssm_n_group
            .ok_or_else(|| candle::Error::Msg("GDN requires ssm.group_count".into()))?;
        let num_v_heads = cfg.ssm_dt_rank
            .ok_or_else(|| candle::Error::Msg("GDN requires ssm.time_step_rank".into()))?;
        let conv_kernel = cfg.ssm_conv_kernel.unwrap_or(4);

        let head_v_dim = d_inner / num_v_heads;
        let conv_channels = d_inner + 2 * num_k_heads * head_k_dim;

        Ok(Self {
            d_inner,
            head_k_dim,
            num_k_heads,
            num_v_heads,
            head_v_dim,
            conv_channels,
            conv_kernel,
        })
    }
}

// ---------------------------------------------------------------------------
// GdnState: recurrent state for all GDN layers
// ---------------------------------------------------------------------------

/// Recurrent state for all GDN layers in the model.
/// Each recurrent layer has a conv state and a delta net state.
pub struct GdnState {
    /// Conv state per recurrent layer: (batch, conv_kernel-1, conv_channels)
    pub conv_states: Vec<Tensor>,
    /// Delta net state per recurrent layer: (batch, num_v_heads, head_v_dim, head_v_dim)
    pub net_states: Vec<Tensor>,
}

impl GdnState {
    /// Initialize zero states for all recurrent layers on a single device.
    pub fn new(cfg: &GgufConfig, batch: usize, device: &candle::Device) -> Result<Self> {
        let num_recurrent = cfg.num_recurrent_layers();
        let devices: Vec<candle::Device> = (0..num_recurrent).map(|_| device.clone()).collect();
        Self::new_multi_device(cfg, batch, &devices)
    }

    /// Initialize zero states with one device per recurrent layer.
    /// `devices_per_recurrent_layer.len()` must equal `cfg.num_recurrent_layers()`.
    pub fn new_multi_device(
        cfg: &GgufConfig,
        batch: usize,
        devices_per_recurrent_layer: &[candle::Device],
    ) -> Result<Self> {
        let dims = GdnDimensions::from_config(cfg)?;
        let num_recurrent = cfg.num_recurrent_layers();
        if devices_per_recurrent_layer.len() != num_recurrent {
            candle::bail!(
                "GdnState::new_multi_device: expected {} device entries, got {}",
                num_recurrent,
                devices_per_recurrent_layer.len()
            );
        }

        let mut conv_states = Vec::with_capacity(num_recurrent);
        let mut net_states = Vec::with_capacity(num_recurrent);

        for dev in devices_per_recurrent_layer {
            conv_states.push(Tensor::zeros(
                (batch, dims.conv_kernel - 1, dims.conv_channels),
                candle::DType::F32,
                dev,
            )?);
            net_states.push(Tensor::zeros(
                (batch, dims.num_v_heads, dims.head_v_dim, dims.head_v_dim),
                candle::DType::F32,
                dev,
            )?);
        }

        Ok(Self {
            conv_states,
            net_states,
        })
    }

    /// Reset all states to zero (for new sequence).
    pub fn reset(&mut self) -> Result<()> {
        for s in &mut self.conv_states {
            *s = s.zeros_like()?;
        }
        for s in &mut self.net_states {
            *s = s.zeros_like()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DeltaNetLayer: single GDN layer weights + forward
// ---------------------------------------------------------------------------

/// Either two separate alpha/beta projections (qwen35) or a single fused
/// ssm_ba projection (qwen3next) that outputs `2 * num_v_heads` and is split
/// per llama.cpp `qwen3next.cpp:287-315` into interleaved-per-k-head beta/alpha.
enum BaProj {
    /// Separate projections: alpha outputs `num_v_heads`, beta outputs `num_v_heads`.
    Split { alpha: QMatMul, beta: QMatMul },
    /// Fused projection: output `2 * num_v_heads`, reshaped to
    /// `(num_k_heads, 2 * num_v_heads / num_k_heads)` and split on the last dim
    /// into `[beta_per_k_head, alpha_per_k_head]`.
    Fused { ba: QMatMul },
}

/// Single Gated Delta Net layer.
pub struct DeltaNetLayer {
    /// Combined input projection: hidden → conv_channels
    wqkv: QMatMul,
    /// Gate/z projection: hidden → d_inner
    wqkv_gate: QMatMul,
    /// Alpha/beta parameter projection(s).
    ba_proj: BaProj,
    /// Decay parameter A (f32, shape: [num_v_heads])
    ssm_a: Tensor,
    /// Time step bias (f32, shape: [num_v_heads])
    ssm_dt: Tensor,
    /// Conv1d weight (f32, shape: [conv_kernel, conv_channels])
    ssm_conv1d: Tensor,
    /// Output norm (RMS norm, shape: [head_v_dim])
    ssm_norm: RmsNorm,
    /// Output projection: d_inner → hidden
    ssm_out: QMatMul,
    /// Dimensions
    dims: GdnDimensions,
    /// Index into GdnState vectors
    recurrent_idx: usize,
    /// RMS norm epsilon
    rms_norm_eps: f64,
}

impl DeltaNetLayer {
    /// Load from GGUF. Auto-detects alpha/beta vs ba tensor format.
    pub fn load(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
        recurrent_idx: usize,
    ) -> Result<Self> {
        let dims = GdnDimensions::from_config(cfg)?;

        let wqkv = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let wqkv_gate = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;

        // Auto-detect alpha/beta vs ba (qwen3next uses ssm_ba instead of ssm_alpha+ssm_beta)
        let ba_proj = if gg.has_tensor(&format!("{prefix}.ssm_alpha.weight")) {
            BaProj::Split {
                alpha: gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?,
                beta: gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?,
            }
        } else if gg.has_tensor(&format!("{prefix}.ssm_ba.weight")) {
            // ssm_ba: fused projection outputting 2*num_v_heads (see qwen3next.cpp:287-315).
            // The layout is reshape(..., num_k_heads, 2*num_v_heads/num_k_heads) with the
            // first half on the last dim = beta, second half = alpha. The forward pass
            // performs the split each step.
            BaProj::Fused {
                ba: gg.qmatmul(&format!("{prefix}.ssm_ba.weight"))?,
            }
        } else {
            candle::bail!("GDN layer {prefix} has neither ssm_alpha nor ssm_ba");
        };

        let ssm_a = gg.dequantize(&format!("{prefix}.ssm_a"))?;
        let ssm_dt = gg.dequantize(&format!("{prefix}.ssm_dt.bias"))?;
        // GGUF stores conv1d as (channels, kernel) but we need (kernel, channels) for broadcast_mul
        let ssm_conv1d = gg.dequantize(&format!("{prefix}.ssm_conv1d.weight"))?.t()?.contiguous()?;
        let ssm_norm = gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), cfg.rms_norm_eps)?;
        let ssm_out = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;

        Ok(Self {
            wqkv,
            wqkv_gate,
            ba_proj,
            ssm_a,
            ssm_dt,
            ssm_conv1d,
            ssm_norm,
            ssm_out,
            dims,
            recurrent_idx,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Forward pass for a single token (autoregressive generation).
    ///
    /// Reference: llama.cpp delta-net-base.cpp `build_delta_net_autoregressive` (lines 288-370)
    /// and qwen35.cpp `build_layer_attn_linear` (lines 198-369)
    pub fn forward_step(&self, x: &Tensor, state: &mut GdnState) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        if seq_len > 1 {
            return self.forward_prefill(x, state);
        }

        let x_2d = x.squeeze(1)?; // (B, hidden)

        // 1. Input projections
        let qkv_mixed = self.wqkv.forward(&x_2d)?;  // (B, conv_channels)
        let z = self.wqkv_gate.forward(&x_2d)?;      // (B, d_inner)

        // 2. Alpha, beta projections — either from two separate weights (qwen35)
        // or from a single fused ssm_ba that outputs 2*num_v_heads and is split.
        let (alpha, beta) = match &self.ba_proj {
            BaProj::Split { alpha, beta } => {
                let a = alpha.forward(&x_2d)?;
                let b = candle_nn::ops::sigmoid(&beta.forward(&x_2d)?)?;
                (a, b)
            }
            BaProj::Fused { ba } => {
                // Follow qwen3next.cpp:287-315: reshape to (B, num_k_heads, 2*v_per_k)
                // and split last dim in half.
                let mixed = ba.forward(&x_2d)?; // (B, 2*num_v_heads)
                let v_per_k = self.dims.num_v_heads / self.dims.num_k_heads;
                let ba_new_dim = 2 * v_per_k;
                let mixed = mixed.reshape((b_sz, self.dims.num_k_heads, ba_new_dim))?;
                // split_sizes = [v_per_k (beta), v_per_k (alpha)]
                let b = mixed.narrow(D::Minus1, 0, v_per_k)?.contiguous()?;
                let a = mixed.narrow(D::Minus1, v_per_k, v_per_k)?.contiguous()?;
                // Flatten to (B, num_v_heads)
                let alpha = a.reshape((b_sz, self.dims.num_v_heads))?;
                let beta_flat = b.reshape((b_sz, self.dims.num_v_heads))?;
                let beta = candle_nn::ops::sigmoid(&beta_flat)?;
                (alpha, beta)
            }
        };

        // 3. Decay gate: softplus(alpha + dt_bias) * A
        let alpha_biased = alpha.broadcast_add(&self.ssm_dt)?;
        let alpha_sp = softplus(&alpha_biased)?;
        let gate = alpha_sp.broadcast_mul(&self.ssm_a)?; // (B, num_v_heads)

        // 4. Conv1d step: shift conv_state, apply depthwise conv, silu
        let conv_state = &mut state.conv_states[self.recurrent_idx];
        let qkv_expanded = qkv_mixed.unsqueeze(1)?; // (B, 1, conv_channels)
        let conv_input = Tensor::cat(&[conv_state.clone(), qkv_expanded], 1)?; // (B, conv_kernel, conv_channels)

        // Update conv state: keep last (conv_kernel-1) entries
        *conv_state = conv_input.narrow(1, 1, self.dims.conv_kernel - 1)?.contiguous()?;

        // Depthwise conv: sum along kernel dimension with weights
        // conv1d_weight shape: (conv_kernel, conv_channels) — apply as dot product
        let conv_out = conv_input
            .broadcast_mul(&self.ssm_conv1d.unsqueeze(0)?)?  // (B, conv_kernel, conv_channels)
            .sum(1)?; // (B, conv_channels)
        let conv_out = candle_nn::ops::silu(&conv_out)?;

        // 5. Split conv output into Q, K, V
        // Layout: Q = [head_k_dim * num_k_heads], K = [head_k_dim * num_k_heads], V = [head_v_dim * num_v_heads]
        let qk_size = self.dims.head_k_dim * self.dims.num_k_heads;
        let v_size = self.dims.head_v_dim * self.dims.num_v_heads;

        let q = conv_out.narrow(D::Minus1, 0, qk_size)?;
        let k = conv_out.narrow(D::Minus1, qk_size, qk_size)?;
        let v = conv_out.narrow(D::Minus1, 2 * qk_size, v_size)?;

        // Reshape to head form
        let q = q.reshape((b_sz, self.dims.num_k_heads, self.dims.head_k_dim))?;
        let k = k.reshape((b_sz, self.dims.num_k_heads, self.dims.head_k_dim))?;
        let v = v.reshape((b_sz, self.dims.num_v_heads, self.dims.head_v_dim))?;

        // 6. L2 normalize Q and K
        // Reshape to (B, H, S_k) for l2_norm on last dim
        let q = l2_norm(&q, self.rms_norm_eps)?;
        let k = l2_norm(&k, self.rms_norm_eps)?;

        // Repeat Q/K heads to match V heads if needed (GQA-style for delta net).
        //
        // ggml_repeat_4d TILES the head dimension: dst[i*nk + k] = src[k].
        // So output head 0 = source head 0, output head 1 = source head 1,
        // ..., output head nk = source head 0 (wrap around).
        // This is "tile" (PyTorch .repeat()), NOT "interleave"
        // (PyTorch .repeat_interleave()).
        //
        // To match in candle, we unsqueeze along axis 1 (NOT 2) and expand,
        // which produces head order [h0, h1, ..., h_{nk-1}, h0, h1, ...].
        let (q, k) = if self.dims.num_k_heads != self.dims.num_v_heads {
            let rep = self.dims.num_v_heads / self.dims.num_k_heads;
            let q = q.unsqueeze(1)?
                .expand((b_sz, rep, self.dims.num_k_heads, self.dims.head_k_dim))?
                .reshape((b_sz, self.dims.num_v_heads, self.dims.head_k_dim))?;
            let k = k.unsqueeze(1)?
                .expand((b_sz, rep, self.dims.num_k_heads, self.dims.head_k_dim))?
                .reshape((b_sz, self.dims.num_v_heads, self.dims.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. Delta net autoregressive step (vectorized)
        // Reshape to (B, H, 1, S_v) for delta_net_step_vectorized
        // Note: head_k_dim == head_v_dim for delta net (qwen35 uses S_k=S_v=128)
        let s_v = self.dims.head_v_dim;
        let h_v = self.dims.num_v_heads;
        let scale = 1.0 / (self.dims.head_k_dim as f64).sqrt();
        let q_scaled = (q * scale)?
            .reshape((b_sz, h_v, 1, s_v))?
            .contiguous()?;
        let k_4d = k.reshape((b_sz, h_v, 1, s_v))?.contiguous()?;
        let v_4d = v.reshape((b_sz, h_v, 1, s_v))?.contiguous()?;

        // gate, beta: (B, H) → (B, H, 1, 1) for broadcast
        let gate_4d = gate.reshape((b_sz, h_v, 1, 1))?.contiguous()?;
        let beta_4d = beta.reshape((b_sz, h_v, 1, 1))?.contiguous()?;

        let net_state = &mut state.net_states[self.recurrent_idx];

        let output_4d = delta_net_step_vectorized(
            &q_scaled,
            &k_4d,
            &v_4d,
            &gate_4d,
            &beta_4d,
            net_state,
        )?;
        // output_4d: (B, H, 1, S_v) → (B, H, S_v)
        let output = output_4d.squeeze(2)?;

        // 8. Gated normalization: rms_norm(output) * silu(z)
        // output shape: (B, num_v_heads, head_v_dim) → (B, d_inner)
        let output = output.reshape((b_sz, self.dims.d_inner))?;

        // Apply per-head RMS norm then silu gate
        // ssm_norm weight is [head_v_dim], applied per head
        let output_normed = self.ssm_norm.forward(
            &output.reshape((b_sz * self.dims.num_v_heads, self.dims.head_v_dim))?
        )?;
        let output_normed = output_normed.reshape((b_sz, self.dims.d_inner))?;

        // Fused: gated = silu(z) * output_normed in one launch instead
        // of two, plus one fewer intermediate buffer per recurrent step.
        // (Multiplication commutes, so this matches the original
        // `output_normed * silu(z)`.)
        let gated = candle_nn::ops::silu_mul(&z, &output_normed)?;

        // 9. Output projection
        let out = self.ssm_out.forward(&gated)?;
        Ok(out.unsqueeze(1)?) // (B, 1, hidden)
    }

    /// Multi-token forward (prefill).
    ///
    /// Batches the position-independent linear projections (`wqkv`,
    /// `wqkv_gate`, `ssm_alpha`/`ssm_beta` or `ssm_ba`, `ssm_out`) across
    /// **all** prompt tokens in one shot, then runs only the recurrent
    /// pieces (conv1d state shift and delta-net state update) inside a
    /// per-token loop. The recurrent state at step `t` depends on the
    /// state at step `t-1`, so those steps remain sequential — but each
    /// of them is now small because the heavy matmuls happen outside
    /// the loop.
    ///
    /// On 4×MI50 with qwen3next 45 GB the GDN-layer prefill speed roughly
    /// doubled vs the previous "call `forward_step` once per token" path.
    fn forward_prefill(&self, x: &Tensor, state: &mut GdnState) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        if seq_len == 1 {
            // Shouldn't reach here (forward_step handles seq_len=1) but be safe.
            return self.forward_step(x, state);
        }
        // Diagnostic toggle: env CANDLE_GDN_PER_TOKEN=1 forces the legacy
        // per-token forward_step loop so we can A/B compare against the
        // new batched path on real models.
        if std::env::var("CANDLE_GDN_PER_TOKEN")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false)
        {
            let mut outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = x.narrow(1, t, 1)?;
                let out_t = self.forward_step(&x_t, state)?;
                outputs.push(out_t.squeeze(1)?);
            }
            return Tensor::stack(&outputs, 1);
        }
        // Per-token loop assumes B=1 so the per-step `narrow(0, t, 1)`
        // produces a contiguous slice without a copy. For B>1 fall back
        // to the legacy per-token forward_step path.
        if b_sz != 1 {
            let mut outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = x.narrow(1, t, 1)?;
                let out_t = self.forward_step(&x_t, state)?;
                outputs.push(out_t.squeeze(1)?);
            }
            return Tensor::stack(&outputs, 1);
        }

        // ----- Stage 1: batched linear projections for all tokens ---------
        // Keep everything in (B*L, X) layout from here on. With B=1 the
        // per-step row at index t is contiguous (just one row of a flat
        // (L, X) tensor) so the recurrent loop can `narrow(0, t, 1)`
        // without triggering an internal copy.
        let bl = b_sz * seq_len;
        let x_flat = x.reshape((bl, hidden))?;
        let qkv_mixed = self.wqkv.forward(&x_flat)?; // (B*L, conv_channels)
        let z = self.wqkv_gate.forward(&x_flat)?; // (B*L, d_inner)

        // Alpha / beta — same auto-detect logic as forward_step but batched.
        let (alpha, beta) = match &self.ba_proj {
            BaProj::Split { alpha, beta } => {
                let a = alpha.forward(&x_flat)?; // (B*L, num_v_heads)
                let b = candle_nn::ops::sigmoid(&beta.forward(&x_flat)?)?;
                (a, b)
            }
            BaProj::Fused { ba } => {
                let mixed = ba.forward(&x_flat)?; // (B*L, 2*num_v_heads)
                let v_per_k = self.dims.num_v_heads / self.dims.num_k_heads;
                let ba_new_dim = 2 * v_per_k;
                let mixed = mixed.reshape((bl, self.dims.num_k_heads, ba_new_dim))?;
                let b = mixed.narrow(D::Minus1, 0, v_per_k)?.contiguous()?;
                let a = mixed.narrow(D::Minus1, v_per_k, v_per_k)?.contiguous()?;
                let alpha = a.reshape((bl, self.dims.num_v_heads))?;
                let beta_flat = b.reshape((bl, self.dims.num_v_heads))?;
                let beta = candle_nn::ops::sigmoid(&beta_flat)?;
                (alpha, beta)
            }
        };

        // Decay gate: softplus(alpha + dt_bias) * A. Single elementwise
        // pass over (B*L, num_v_heads).
        let alpha_biased = alpha.broadcast_add(&self.ssm_dt)?;
        let alpha_sp = softplus(&alpha_biased)?;
        let gate = alpha_sp.broadcast_mul(&self.ssm_a)?; // (B*L, num_v_heads)

        // ----- Stage 2: per-token recurrent loop --------------------------
        // Each iteration only touches the conv1d shift, the delta-net state
        // update, and the per-head ssm_norm. All the heavy matmuls already
        // happened above. Per-step access is `narrow(0, t, 1)` on the
        // contiguous (B*L, X) tensors — for B=1 this is a single contiguous
        // row, so no copies happen.
        let qk_size = self.dims.head_k_dim * self.dims.num_k_heads;
        let v_size = self.dims.head_v_dim * self.dims.num_v_heads;
        let s_v = self.dims.head_v_dim;
        let h_v = self.dims.num_v_heads;
        let scale = 1.0 / (self.dims.head_k_dim as f64).sqrt();
        let conv_kernel = self.dims.conv_kernel;
        let n_k = self.dims.num_k_heads;
        let d_inner = self.dims.d_inner;

        let mut gated_outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // narrow(0, t, 1) on a contiguous (L, X) tensor returns a
            // contiguous (1, X) view — zero copy.
            let qkv_mixed_t = qkv_mixed.narrow(0, t, 1)?; // (1, conv_channels)
            let z_t = z.narrow(0, t, 1)?; // (1, d_inner)
            let gate_t = gate.narrow(0, t, 1)?; // (1, num_v_heads)
            let beta_t = beta.narrow(0, t, 1)?; // (1, num_v_heads)

            // Conv1d step: shift state, append new token, apply weights, silu.
            // qkv_mixed_t is already (1, conv_channels), unsqueeze to
            // (1, 1, conv_channels) to cat with the (1, K-1, conv_channels)
            // conv_state along dim 1.
            let conv_state = &mut state.conv_states[self.recurrent_idx];
            let qkv_expanded = qkv_mixed_t.unsqueeze(1)?; // (1, 1, conv_channels)
            let conv_input = Tensor::cat(&[conv_state.clone(), qkv_expanded], 1)?;
            *conv_state = conv_input.narrow(1, 1, conv_kernel - 1)?.contiguous()?;
            let conv_out = conv_input
                .broadcast_mul(&self.ssm_conv1d.unsqueeze(0)?)?
                .sum(1)?; // (1, conv_channels)
            let conv_out = candle_nn::ops::silu(&conv_out)?;

            // Split conv_out into Q, K, V along the channel dim.
            let q = conv_out.narrow(D::Minus1, 0, qk_size)?;
            let k = conv_out.narrow(D::Minus1, qk_size, qk_size)?;
            let v = conv_out.narrow(D::Minus1, 2 * qk_size, v_size)?;

            let q = q.reshape((1, n_k, self.dims.head_k_dim))?;
            let k = k.reshape((1, n_k, self.dims.head_k_dim))?;
            let v = v.reshape((1, h_v, self.dims.head_v_dim))?;

            let q = l2_norm(&q, self.rms_norm_eps)?;
            let k = l2_norm(&k, self.rms_norm_eps)?;

            // GQA-style head repeat (tile order, matches ggml_repeat).
            let (q, k) = if n_k != h_v {
                let rep = h_v / n_k;
                let q = q
                    .unsqueeze(1)?
                    .expand((1, rep, n_k, self.dims.head_k_dim))?
                    .reshape((1, h_v, self.dims.head_k_dim))?;
                let k = k
                    .unsqueeze(1)?
                    .expand((1, rep, n_k, self.dims.head_k_dim))?
                    .reshape((1, h_v, self.dims.head_k_dim))?;
                (q, k)
            } else {
                (q, k)
            };

            // Reshape for the (B, H, 1, S_v) delta-net step.
            let q_scaled = (q * scale)?.reshape((1, h_v, 1, s_v))?.contiguous()?;
            let k_4d = k.reshape((1, h_v, 1, s_v))?.contiguous()?;
            let v_4d = v.reshape((1, h_v, 1, s_v))?.contiguous()?;
            let gate_4d = gate_t.reshape((1, h_v, 1, 1))?.contiguous()?;
            let beta_4d = beta_t.reshape((1, h_v, 1, 1))?.contiguous()?;

            let net_state = &mut state.net_states[self.recurrent_idx];
            let output_4d = delta_net_step_vectorized(
                &q_scaled, &k_4d, &v_4d, &gate_4d, &beta_4d, net_state,
            )?;

            // Per-head RMS norm + silu(z) gate.
            let output = output_4d.squeeze(2)?; // (1, h_v, s_v)
            let output = output.reshape((1, d_inner))?;
            let output_normed = self.ssm_norm.forward(
                &output.reshape((h_v, self.dims.head_v_dim))?,
            )?;
            let output_normed = output_normed.reshape((1, d_inner))?;

            // Fused silu(z_t) * output_normed in one launch.
            let gated = candle_nn::ops::silu_mul(&z_t, &output_normed)?;
            gated_outputs.push(gated);
        }

        // ----- Stage 3: batched output projection -------------------------
        // Concat the per-token gated outputs along dim 0 → (B*L=L, d_inner)
        // and run the d_inner → hidden projection in a single matmul.
        let gated_flat = Tensor::cat(&gated_outputs, 0)?; // (L, d_inner)
        let out_flat = self.ssm_out.forward(&gated_flat)?; // (L, hidden)
        out_flat.reshape((b_sz, seq_len, hidden))
    }
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    // For numerical stability: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    let abs_x = x.abs()?;
    let max_x = x.maximum(&x.zeros_like()?)?;
    let stable = (abs_x.neg()?.exp()? + 1.0)?.log()?;
    max_x + stable
}

/// Vectorized delta net autoregressive step (single token).
///
/// Reference: llama.cpp delta-net-base.cpp lines 288-370
///
/// State update (per (b, h)):
///   state = state * exp(gate)
///   sk = k @ state            // (1, S_v) @ (S_v, S_v) → (1, S_v)
///   d = (v - sk) * beta       // (1, S_v)
///   state += k^T @ d          // (S_v, 1) @ (1, S_v) → (S_v, S_v)
///   output = q @ state        // (1, S_v) @ (S_v, S_v) → (1, S_v)
///
/// All shapes use candle's row-major (B, H, ..., ...) layout. State is
/// (B, H, S_v, S_v). q/k/v are (B, H, 1, S_v). gate/beta are (B, H, 1, 1).
fn delta_net_step_vectorized(
    q: &Tensor,         // (B, H, 1, S_v) — must be pre-scaled by 1/sqrt(S_k)
    k: &Tensor,         // (B, H, 1, S_v)
    v: &Tensor,         // (B, H, 1, S_v)
    gate: &Tensor,      // (B, H, 1, 1)
    beta: &Tensor,      // (B, H, 1, 1)
    state: &mut Tensor, // (B, H, S_v, S_v)
) -> Result<Tensor> {
    // 1. Apply decay: state = state * exp(gate)
    let g_exp = gate.exp()?;
    *state = state.broadcast_mul(&g_exp)?;

    // 2. sk = k @ state → (B, H, 1, S_v)
    let sk = k.matmul(state)?;

    // 3. d = (v - sk) * beta → (B, H, 1, S_v)
    let d = (v - sk)?.broadcast_mul(beta)?;

    // 4. outer(k^T, d) = k^T @ d → (B, H, S_v, S_v)
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let outer = k_t.matmul(&d)?;

    // 5. state += outer
    *state = (&*state + outer)?;

    // 6. output = q @ state → (B, H, 1, S_v)
    q.matmul(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_dimensions_from_config() {
        let mut m = std::collections::HashMap::new();
        m.insert("general.architecture".into(), candle::quantized::gguf_file::Value::String("qwen35".into()));
        m.insert("qwen35.embedding_length".into(), candle::quantized::gguf_file::Value::U32(5120));
        m.insert("qwen35.block_count".into(), candle::quantized::gguf_file::Value::U32(64));
        m.insert("qwen35.attention.head_count".into(), candle::quantized::gguf_file::Value::U32(24));
        m.insert("qwen35.attention.head_count_kv".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("qwen35.attention.layer_norm_rms_epsilon".into(), candle::quantized::gguf_file::Value::F32(1e-6));
        m.insert("qwen35.ssm.inner_size".into(), candle::quantized::gguf_file::Value::U32(6144));
        m.insert("qwen35.ssm.state_size".into(), candle::quantized::gguf_file::Value::U32(128));
        m.insert("qwen35.ssm.group_count".into(), candle::quantized::gguf_file::Value::U32(16));
        m.insert("qwen35.ssm.time_step_rank".into(), candle::quantized::gguf_file::Value::U32(48));
        m.insert("qwen35.ssm.conv_kernel".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("qwen35.full_attention_interval".into(), candle::quantized::gguf_file::Value::U32(4));

        let cfg = super::super::gguf_config::GgufConfig::from_metadata(&m).unwrap();
        let dims = GdnDimensions::from_config(&cfg).unwrap();

        assert_eq!(dims.d_inner, 6144);
        assert_eq!(dims.head_k_dim, 128);
        assert_eq!(dims.num_k_heads, 16);
        assert_eq!(dims.num_v_heads, 48);
        assert_eq!(dims.head_v_dim, 128); // 6144/48
        assert_eq!(dims.conv_channels, 10240); // 6144 + 2*16*128
        assert_eq!(dims.conv_kernel, 4);
    }

    #[test]
    fn test_softplus() {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 10.0, -10.0], &candle::Device::Cpu).unwrap();
        let result = softplus(&x).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // softplus(0) = ln(2) ≈ 0.6931
        assert!((vals[0] - 0.6931).abs() < 1e-3);
        // softplus(1) = ln(1+e) ≈ 1.3133
        assert!((vals[1] - 1.3133).abs() < 1e-3);
        // softplus(-1) = ln(1+e^-1) ≈ 0.3133
        assert!((vals[2] - 0.3133).abs() < 1e-3);
        // softplus(10) ≈ 10 (for large x)
        assert!((vals[3] - 10.0).abs() < 1e-3);
        // softplus(-10) ≈ 0 (for large negative x)
        assert!(vals[4] < 1e-3);
    }

    #[test]
    fn test_delta_net_step_vectorized_zero_state() {
        // With zero state, after one step state should be non-zero.
        // Inputs: (B=1, H=2, 1, S_v=2). State: (B, H, S_v, S_v).
        let dev = &candle::Device::Cpu;

        let q = Tensor::randn(0f32, 0.1, (1, 2, 1, 2), dev).unwrap();
        let k = Tensor::randn(0f32, 0.1, (1, 2, 1, 2), dev).unwrap();
        let v = Tensor::randn(0f32, 0.1, (1, 2, 1, 2), dev).unwrap();
        let gate = Tensor::new(&[[[[-0.1f32]], [[-0.1]]]], dev).unwrap(); // (1,2,1,1)
        let beta = Tensor::new(&[[[[0.5f32]], [[0.5]]]], dev).unwrap();

        let mut state = Tensor::zeros((1, 2, 2, 2), candle::DType::F32, dev).unwrap();

        let output =
            delta_net_step_vectorized(&q, &k, &v, &gate, &beta, &mut state).unwrap();
        assert_eq!(output.dims(), &[1, 2, 1, 2]); // (B, H, 1, S_v)

        // After one step, state should be non-zero (rank-1 outer update)
        let state_sum: f32 = state.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(state_sum > 0.0, "state should be updated after one step");
    }

    #[test]
    fn test_delta_net_step_vectorized_state_accumulates() {
        // Single head, S_v=2. With no decay and repeated input,
        // state should be updated each step (delta rule converges to v / k.k).
        let dev = &candle::Device::Cpu;

        let q = Tensor::new(&[[[[0.5f32, 0.5]]]], dev).unwrap();   // (1,1,1,2)
        let k = Tensor::new(&[[[[1.0f32, 0.0]]]], dev).unwrap();
        let v = Tensor::new(&[[[[1.0f32, 0.0]]]], dev).unwrap();
        let gate = Tensor::new(&[[[[0.0f32]]]], dev).unwrap();     // exp(0)=1, no decay
        let beta = Tensor::new(&[[[[1.0f32]]]], dev).unwrap();     // full update

        let mut state = Tensor::zeros((1, 1, 2, 2), candle::DType::F32, dev).unwrap();

        // Step 1: state = k^T @ ((v - 0) * 1) = outer(k, v) = [[1,0],[0,0]]
        let _o1 =
            delta_net_step_vectorized(&q, &k, &v, &gate, &beta, &mut state).unwrap();
        let s1: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        // Expected: [[1, 0], [0, 0]] flattened
        assert!((s1[0] - 1.0).abs() < 1e-5);
        assert!(s1[1].abs() < 1e-5);
        assert!(s1[2].abs() < 1e-5);
        assert!(s1[3].abs() < 1e-5);

        // Step 2: sk = k @ state = [1,0] @ [[1,0],[0,0]] = [1,0]
        // d = (v - sk) * beta = ([1,0] - [1,0]) * 1 = [0,0]
        // state += k^T @ d = 0 → state unchanged
        let _o2 =
            delta_net_step_vectorized(&q, &k, &v, &gate, &beta, &mut state).unwrap();
        let s2: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        assert!((s2[0] - 1.0).abs() < 1e-5);
        assert!(s2[1].abs() < 1e-5);
        assert!(s2[2].abs() < 1e-5);
        assert!(s2[3].abs() < 1e-5);
    }

    /// Reference scalar implementation that mirrors ggml's
    /// `build_delta_net_autoregressive` exactly. Single batch, single head.
    /// Returns (output, new_state) where state is the (S_v, S_v) matrix.
    fn scalar_gdn_step(
        q: &[f32], k: &[f32], v: &[f32], gate: f32, beta: f32,
        state: &mut Vec<f32>, s_v: usize,
    ) {
        // 1. state *= exp(gate)  (scalar multiply)
        let g = gate.exp();
        for s in state.iter_mut() { *s *= g; }
        // 2. sk[j] = sum_i state[i*s_v + j] * k[i]   (M^T @ k)
        let mut sk = vec![0f32; s_v];
        for j in 0..s_v {
            for i in 0..s_v {
                sk[j] += state[i * s_v + j] * k[i];
            }
        }
        // 3. d[j] = (v[j] - sk[j]) * beta
        let d: Vec<f32> = (0..s_v).map(|j| (v[j] - sk[j]) * beta).collect();
        // 4. state[i, j] += k[i] * d[j]   (rank-1 outer product)
        for i in 0..s_v {
            for j in 0..s_v {
                state[i * s_v + j] += k[i] * d[j];
            }
        }
    }

    #[test]
    fn test_delta_net_vectorized_matches_scalar_reference() {
        // Compare vectorized candle implementation against a hand-rolled scalar
        // version that exactly mirrors ggml's autoregressive delta net.
        // Multiple steps, single batch, single head, S_v=4.
        let dev = &candle::Device::Cpu;
        let s_v = 4;

        // Random-ish but deterministic inputs across 3 steps
        let inputs: Vec<([f32; 4], [f32; 4], [f32; 4], f32, f32)> = vec![
            ([0.5, -0.2, 0.3, 0.1], [0.7, 0.1, -0.4, 0.2], [1.0, 0.5, -0.3, 0.6], -0.1, 0.8),
            ([0.2,  0.4, -0.1, 0.3], [0.3, -0.5, 0.6, 0.0], [0.4, 0.2, 0.1, -0.5], -0.05, 0.6),
            ([-0.3, 0.1, 0.5, -0.2], [0.0, 0.4, 0.3, 0.5], [-0.6, 0.7, 0.2, 0.1], -0.2, 0.9),
        ];

        // Scalar reference state (row-major (S_v, S_v))
        let mut scalar_state = vec![0f32; s_v * s_v];

        // Vectorized state: (1, 1, S_v, S_v)
        let mut vec_state = Tensor::zeros((1, 1, s_v, s_v), candle::DType::F32, dev).unwrap();

        for (q_arr, k_arr, v_arr, gate_v, beta_v) in &inputs {
            // Run scalar reference
            scalar_gdn_step(q_arr, k_arr, v_arr, *gate_v, *beta_v, &mut scalar_state, s_v);

            // Run vectorized candle version
            let q = Tensor::from_slice(q_arr, (1, 1, 1, s_v), dev).unwrap();
            let k = Tensor::from_slice(k_arr, (1, 1, 1, s_v), dev).unwrap();
            let v = Tensor::from_slice(v_arr, (1, 1, 1, s_v), dev).unwrap();
            let gate = Tensor::new(&[[[[*gate_v]]]], dev).unwrap();
            let beta = Tensor::new(&[[[[*beta_v]]]], dev).unwrap();
            let _ = delta_net_step_vectorized(&q, &k, &v, &gate, &beta, &mut vec_state).unwrap();

            // Compare states
            let vec_vals: Vec<f32> = vec_state.flatten_all().unwrap().to_vec1().unwrap();
            for (i, (sv, vv)) in scalar_state.iter().zip(vec_vals.iter()).enumerate() {
                assert!(
                    (sv - vv).abs() < 1e-5,
                    "state mismatch at idx {i}: scalar={sv}, vectorized={vv}"
                );
            }
        }
    }

    #[test]
    fn test_gqa_tile_order_matches_ggml_repeat() {
        // ggml_repeat tiles: dst[i*nk + k] = src[k]
        // So if num_k_heads=2 and num_v_heads=6, the output head order is
        // [src_h0, src_h1, src_h0, src_h1, src_h0, src_h1].
        //
        // Verify that unsqueeze(1)+expand+reshape produces this tile order
        // (NOT the interleave order [h0,h0,h0,h1,h1,h1] from unsqueeze(2)).
        let dev = &candle::Device::Cpu;
        let num_k_heads = 2;
        let num_v_heads = 6;
        let head_dim = 3;
        let rep = num_v_heads / num_k_heads; // 3
        let b_sz = 1;

        // Source: head 0 = [1,1,1], head 1 = [2,2,2]
        let src = Tensor::new(
            &[[[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]],
            dev,
        ).unwrap(); // (B=1, num_k_heads=2, head_dim=3)

        // Tile (correct, matches ggml_repeat)
        let tiled = src
            .unsqueeze(1).unwrap()
            .expand((b_sz, rep, num_k_heads, head_dim)).unwrap()
            .reshape((b_sz, num_v_heads, head_dim)).unwrap();
        let vals: Vec<f32> = tiled.flatten_all().unwrap().to_vec1().unwrap();
        // Expected order: [h0, h1, h0, h1, h0, h1] = [1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,2,2,2]
        let expected: Vec<f32> = vec![
            1.0, 1.0, 1.0,  // h0
            2.0, 2.0, 2.0,  // h1
            1.0, 1.0, 1.0,  // h0 (tile)
            2.0, 2.0, 2.0,  // h1
            1.0, 1.0, 1.0,  // h0
            2.0, 2.0, 2.0,  // h1
        ];
        assert_eq!(vals, expected, "tile order does not match ggml_repeat");

        // Sanity check: interleave (incorrect) order would be different
        let interleaved = src
            .unsqueeze(2).unwrap()
            .expand((b_sz, num_k_heads, rep, head_dim)).unwrap()
            .reshape((b_sz, num_v_heads, head_dim)).unwrap();
        let bad_vals: Vec<f32> = interleaved.flatten_all().unwrap().to_vec1().unwrap();
        // Interleave order: [h0, h0, h0, h1, h1, h1]
        let interleave_expected: Vec<f32> = vec![
            1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,  2.0, 2.0, 2.0,  2.0, 2.0, 2.0,
        ];
        assert_eq!(bad_vals, interleave_expected);
        // Confirm the two orders are genuinely different
        assert_ne!(vals, bad_vals);
    }

    #[test]
    fn test_delta_net_step_vectorized_decay() {
        // With strongly negative gate (large decay), state should shrink.
        let dev = &candle::Device::Cpu;
        let q = Tensor::new(&[[[[0.0f32, 0.0]]]], dev).unwrap();
        let k = Tensor::new(&[[[[0.0f32, 0.0]]]], dev).unwrap(); // no update
        let v = Tensor::new(&[[[[0.0f32, 0.0]]]], dev).unwrap();
        // exp(-2) ≈ 0.135 → big decay
        let gate = Tensor::new(&[[[[-2.0f32]]]], dev).unwrap();
        let beta = Tensor::new(&[[[[1.0f32]]]], dev).unwrap();

        // Pre-fill state with 1s
        let mut state = Tensor::ones((1, 1, 2, 2), candle::DType::F32, dev).unwrap();
        let pre: f32 = state.sum_all().unwrap().to_scalar().unwrap();

        let _o = delta_net_step_vectorized(&q, &k, &v, &gate, &beta, &mut state).unwrap();
        let post: f32 = state.sum_all().unwrap().to_scalar().unwrap();
        // Decay factor is exp(-2) ≈ 0.135
        assert!(post < pre * 0.2, "state should decay (pre={pre}, post={post})");
    }

    #[test]
    fn test_gdn_state_new_and_reset() {
        let mut m = std::collections::HashMap::new();
        m.insert("general.architecture".into(), candle::quantized::gguf_file::Value::String("test".into()));
        m.insert("test.embedding_length".into(), candle::quantized::gguf_file::Value::U32(64));
        m.insert("test.block_count".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("test.attention.head_count".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("test.attention.head_count_kv".into(), candle::quantized::gguf_file::Value::U32(2));
        m.insert("test.attention.layer_norm_rms_epsilon".into(), candle::quantized::gguf_file::Value::F32(1e-6));
        m.insert("test.ssm.inner_size".into(), candle::quantized::gguf_file::Value::U32(8));
        m.insert("test.ssm.state_size".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("test.ssm.group_count".into(), candle::quantized::gguf_file::Value::U32(2));
        m.insert("test.ssm.time_step_rank".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("test.ssm.conv_kernel".into(), candle::quantized::gguf_file::Value::U32(4));
        m.insert("test.full_attention_interval".into(), candle::quantized::gguf_file::Value::U32(4));

        let cfg = super::super::gguf_config::GgufConfig::from_metadata(&m).unwrap();
        // 4 blocks, interval=4 → 3 recurrent, 1 full attn
        assert_eq!(cfg.num_recurrent_layers(), 3);

        let mut state = GdnState::new(&cfg, 1, &candle::Device::Cpu).unwrap();
        assert_eq!(state.conv_states.len(), 3);
        assert_eq!(state.net_states.len(), 3);

        // Verify shapes
        // conv: (batch=1, conv_kernel-1=3, conv_channels=8+2*2*4=24)
        assert_eq!(state.conv_states[0].dims(), &[1, 3, 24]);
        // net: (batch=1, num_v_heads=4, head_v_dim=2, head_v_dim=2)
        assert_eq!(state.net_states[0].dims(), &[1, 4, 2, 2]);

        // Reset should zero everything
        state.reset().unwrap();
        let sum: f32 = state.conv_states[0].abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }
}
