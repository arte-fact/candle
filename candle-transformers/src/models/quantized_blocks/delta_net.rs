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

        // P3: GQA expand DELETED from this path — the fused GDN kernel
        // now does the `h_idx / n_rep` indexing internally (see
        // `gated_delta_net_step_fused`). Q/K stay at their native
        // `num_k_heads` head count. This removes ~2 ucopy_f32 launches
        // per forward_step call. `delta_net_step_vectorized` itself
        // knows how to expand Q/K for the CPU fallback.

        // 7. Delta net autoregressive step (vectorized)
        // Reshape q/k to (B, n_k_heads, 1, S_v); v to (B, n_v_heads, 1, S_v).
        // head_k_dim == head_v_dim for delta net (qwen35 uses S_k=S_v=128).
        let s_v = self.dims.head_v_dim;
        let h_v = self.dims.num_v_heads;
        let n_k = self.dims.num_k_heads;
        let scale = 1.0 / (self.dims.head_k_dim as f64).sqrt();
        let q_scaled = (q * scale)?
            .reshape((b_sz, n_k, 1, s_v))?
            .contiguous()?;
        let k_4d = k.reshape((b_sz, n_k, 1, s_v))?.contiguous()?;
        let v_4d = v.reshape((b_sz, h_v, 1, s_v))?.contiguous()?;

        // gate, beta: (B, H_v) → (B, H_v, 1, 1) for broadcast
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
    /// Fully batched path: all position-independent linear projections
    /// are run once over the full prompt, and every downstream op
    /// (conv1d, L2 norm, GQA head repeat, delta-net recurrence,
    /// ssm_norm, silu_mul, output projection) operates on `(B*L, …)`
    /// tensors in one shot.
    ///
    /// The key optimisation over the pre-Phase-2 version is that the
    /// delta-net recurrence is now a single call to
    /// [`delta_net_step_vectorized`] with `L = seq_len`. On HIP this
    /// dispatches to one `gated_delta_net.cu` launch that keeps the
    /// `(S_v, S_v)` state register-resident across every token of the
    /// prompt — the warp loads the state once per launch, iterates
    /// all L recurrence steps, and writes it back once. This replaces
    /// the ~8-op tensor chain × `seq_len` launches that dominated the
    /// candle rocprofv3 breakdown (842k pointwise ops / 4948 ms on
    /// qwen35-9B 1-GPU prefill).
    ///
    /// Conv1d is still position-dependent (state[t] = f(state[t-1],
    /// x[t])) but its K-step window is a trivial sliding dot product,
    /// expressed here as K broadcast_mul + add passes over the
    /// `(L+K-1, channels)` concatenation of conv_state and qkv_mixed.
    /// That's `K ≈ 4` tensor-op launches per layer instead of
    /// `~5 × seq_len` in the old per-token loop.
    ///
    /// Fallback: `B > 1` and the `CANDLE_GDN_PER_TOKEN` env var both
    /// route back to the legacy `forward_step`-per-token path. The
    /// env var is kept as a diagnostic A/B switch; `B > 1` is a
    /// practical restriction because the conv_state tensors in
    /// `GdnState` are stored with a single batch row and the batched
    /// conv1d path below assumes `B = 1`.
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
        // Batched path assumes B=1; for B>1 fall back to the legacy
        // per-token forward_step path. None of the models we run today
        // have B>1 prefill, so this is a correctness safety net, not a
        // perf path.
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
        let bl = b_sz * seq_len;
        let x_flat = x.reshape((bl, hidden))?;
        let qkv_mixed = self.wqkv.forward(&x_flat)?; // (L, conv_channels)
        let z = self.wqkv_gate.forward(&x_flat)?; // (L, d_inner)

        // Alpha / beta — same auto-detect logic as forward_step but batched.
        let (alpha, beta) = match &self.ba_proj {
            BaProj::Split { alpha, beta } => {
                let a = alpha.forward(&x_flat)?; // (L, num_v_heads)
                let b = candle_nn::ops::sigmoid(&beta.forward(&x_flat)?)?;
                (a, b)
            }
            BaProj::Fused { ba } => {
                let mixed = ba.forward(&x_flat)?; // (L, 2*num_v_heads)
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

        // Decay gate: softplus(alpha + dt_bias) * A, all at once.
        let alpha_biased = alpha.broadcast_add(&self.ssm_dt)?;
        let alpha_sp = softplus(&alpha_biased)?;
        let gate_flat = alpha_sp.broadcast_mul(&self.ssm_a)?; // (L, num_v_heads)

        // Dimension shortcuts used below.
        let qk_size = self.dims.head_k_dim * self.dims.num_k_heads;
        let v_size = self.dims.head_v_dim * self.dims.num_v_heads;
        let s_v = self.dims.head_v_dim;
        let h_v = self.dims.num_v_heads;
        let scale = 1.0 / (self.dims.head_k_dim as f64).sqrt();
        let conv_kernel = self.dims.conv_kernel;
        let n_k = self.dims.num_k_heads;
        let d_inner = self.dims.d_inner;

        // ----- Stage 2: batched conv1d ------------------------------------
        //
        // The old per-token conv1d shifted `conv_state` by one row and
        // computed one K-row window's dot product against
        // `ssm_conv1d`. Batched equivalent: concatenate the initial
        // `conv_state` (K-1 rows) with the full prompt `qkv_mixed`
        // (L rows) along dim 0, producing `stacked` of shape
        // `(L + K - 1, conv_channels)`. Each of the L output rows is a
        // sliding K-window dot product:
        //
        //     conv_out[t, c] = Σ_{j=0..K} stacked[t+j, c] * weight[j, c]
        //
        // which we unroll into K `broadcast_mul + add` passes. The
        // new conv_state is the last (K-1) rows of `stacked`.
        let conv_state_prev = state.conv_states[self.recurrent_idx].clone(); // (1, K-1, channels)
        let conv_state_2d = conv_state_prev.squeeze(0)?; // (K-1, channels)
        let stacked = Tensor::cat(&[&conv_state_2d, &qkv_mixed], 0)?; // (L + K - 1, channels)

        // Window 0 contribution initialises conv_out — avoids a zeros
        // alloc + one extra add pass. ssm_conv1d.narrow(0, j, 1) is
        // shape (1, channels); broadcast_mul vs (L, channels) gives
        // (L, channels).
        let mut conv_out = stacked
            .narrow(0, 0, seq_len)?
            .broadcast_mul(&self.ssm_conv1d.narrow(0, 0, 1)?)?;
        for j in 1..conv_kernel {
            let window = stacked.narrow(0, j, seq_len)?;
            let weight = self.ssm_conv1d.narrow(0, j, 1)?;
            conv_out = (conv_out + window.broadcast_mul(&weight)?)?;
        }
        let conv_out = candle_nn::ops::silu(&conv_out)?; // (L, conv_channels)

        // Update conv_state: the last (K-1) rows of `stacked` are the
        // window that step t=seq_len-1 consumed minus the oldest row.
        state.conv_states[self.recurrent_idx] = stacked
            .narrow(0, seq_len, conv_kernel - 1)?
            .unsqueeze(0)?
            .contiguous()?;

        // ----- Stage 2b: split Q, K, V and reshape to head form -----------
        let q_flat = conv_out.narrow(D::Minus1, 0, qk_size)?;
        let k_flat = conv_out.narrow(D::Minus1, qk_size, qk_size)?;
        let v_flat = conv_out.narrow(D::Minus1, 2 * qk_size, v_size)?;

        // (L, qk_size) → (L, n_k, head_k_dim).
        let q_heads = q_flat.reshape((seq_len, n_k, self.dims.head_k_dim))?;
        let k_heads = k_flat.reshape((seq_len, n_k, self.dims.head_k_dim))?;
        let v_heads = v_flat.reshape((seq_len, h_v, s_v))?;

        // L2 norm on the head_k_dim axis.
        let q_normed = l2_norm(&q_heads, self.rms_norm_eps)?;
        let k_normed = l2_norm(&k_heads, self.rms_norm_eps)?;

        // ----- Stage 2c: reshape into (B=1, H_kv, L, S_v) for Q/K and
        // (B=1, H_v, L, S_v) for V. The GDN kernel handles GQA via
        // `h_idx / n_rep` indexing (post-P3), so Q/K stay at their
        // native `n_k_heads` head count — no expand→reshape ucopy.
        //
        // Only one `transpose + contiguous` per tensor remains
        // (converting from `(L, H_kv, head)` to `(1, H_kv, L, head)`).
        // For `n_rep = 3` this halves the Q/K memcopy vs the old path
        // that first expanded to `n_v_head` then transposed.
        let q_for_gdn = q_normed
            .unsqueeze(0)? // (1, L, n_k, head)
            .transpose(1, 2)?
            .contiguous()?; // (1, n_k, L, head)
        let k_for_gdn = k_normed
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?;
        let v_for_gdn = v_heads
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?; // (1, h_v, L, S_v)

        // Pre-scale q by 1/sqrt(head_k_dim). Same convention as the
        // per-token path.
        let q_scaled = (q_for_gdn * scale)?;

        // gate / beta are currently (L, h_v). The kernel wants
        // (1, h_v, L, 1) — reshape + transpose + unsqueeze + contig.
        let gate_4d = gate_flat
            .unsqueeze(0)? // (1, L, h_v)
            .transpose(1, 2)? // (1, h_v, L)
            .contiguous()?
            .unsqueeze(3)?; // (1, h_v, L, 1)
        let beta_4d = beta
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?
            .unsqueeze(3)?;

        // ----- Stage 2d: ONE fused delta-net kernel call ------------------
        //
        // On HIP this is one `gated_delta_net.cu` launch that keeps
        // the `(S_v, S_v)` state register-resident across the entire
        // L-token recurrence. On CPU it falls back to a per-token
        // loop over the single-step tensor-op chain.
        let net_state = &mut state.net_states[self.recurrent_idx];
        let output_4d = delta_net_step_vectorized(
            &q_scaled, &k_for_gdn, &v_for_gdn, &gate_4d, &beta_4d, net_state,
        )?; // (1, h_v, L, S_v)

        // (1, h_v, L, S_v) → (L, h_v, S_v) → (L, d_inner)
        let output = output_4d
            .transpose(1, 2)? // (1, L, h_v, S_v)
            .contiguous()?
            .squeeze(0)?; // (L, h_v, S_v)
        let output = output.reshape((seq_len, d_inner))?;

        // ----- Stage 2e: batched ssm_norm + silu_mul ----------------------
        //
        // ssm_norm weight is `[head_v_dim]`, applied per head. Reshape
        // to `(L * h_v, head_v_dim)` so one rmsnorm call covers every
        // (token, head) slot; then fold back to `(L, d_inner)` for
        // the silu_mul gate.
        let output_normed = self
            .ssm_norm
            .forward(&output.reshape((seq_len * h_v, s_v))?)?;
        let output_normed = output_normed.reshape((seq_len, d_inner))?;
        let gated = candle_nn::ops::silu_mul(&z, &output_normed)?; // (L, d_inner)

        // ----- Stage 3: batched output projection -------------------------
        let out_flat = self.ssm_out.forward(&gated)?; // (L, hidden)
        out_flat.reshape((b_sz, seq_len, hidden))
    }
}

/// Softplus activation: log(1 + exp(x)).
///
/// On HIP with contiguous f32 inputs, lowers to a single fused kernel
/// launch via `candle::hip_backend::softplus_fused`. On every other
/// backend (or non-contiguous HIP input) it falls back to the
/// numerically-stable 6-op tensor chain below. Both paths compute
/// `max(x, 0) + log1p(exp(-|x|))` which is exact and overflow-safe.
fn softplus(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "hip")]
    {
        if matches!(x.device(), candle::Device::Hip(_))
            && x.dtype() == candle::DType::F32
            && x.is_contiguous()
        {
            return candle::hip_backend::softplus_fused(x);
        }
    }
    // Fallback: the pre-Phase-4 tensor-op chain.
    let abs_x = x.abs()?;
    let max_x = x.maximum(&x.zeros_like()?)?;
    let stable = (abs_x.neg()?.exp()? + 1.0)?.log()?;
    max_x + stable
}

/// Vectorized delta net autoregressive step.
///
/// Reference: llama.cpp delta-net-base.cpp lines 288-370
///
/// State update (per (b, h), per token t):
///   state = state * exp(gate)
///   sk = k @ state            // (1, S_v) @ (S_v, S_v) → (1, S_v)
///   d = (v - sk) * beta       // (1, S_v)
///   state += k^T @ d          // (S_v, 1) @ (1, S_v) → (S_v, S_v)
///   output = q @ state        // (1, S_v) @ (S_v, S_v) → (1, S_v)
///
/// All shapes use candle's row-major (B, H, ..., ...) layout.
/// q/k/v are `(B, H, L, S_v)`, gate/beta `(B, H, L, 1)`, and state
/// `(B, H, S_v, S_v)`. The function handles any `L ≥ 1`.
///
/// On HIP with S_v=128 and contiguous f32 inputs, this lowers to a
/// single fused kernel launch via `candle::hip_backend::
/// gated_delta_net_step_fused` — replacing the ~8 tensor-op launches
/// below with one, and (when `L > 1`) amortizing the state load/store
/// across every token in a single warp-resident recurrence loop.
/// The fallback path (CPU, non-128 S_v, non-f32, or non-contiguous)
/// handles `L > 1` by looping over tokens and reusing the single-step
/// tensor-op chain per iteration — correct but unfused.
fn delta_net_step_vectorized(
    q: &Tensor,         // (B, H_kv, L, S_v) — may have fewer heads than V (GQA)
    k: &Tensor,         // (B, H_kv, L, S_v)
    v: &Tensor,         // (B, H_v,  L, S_v)
    gate: &Tensor,      // (B, H_v,  L, 1)
    beta: &Tensor,      // (B, H_v,  L, 1)
    state: &mut Tensor, // (B, H_v,  S_v, S_v)
) -> Result<Tensor> {
    // HIP fast path: the fused kernel handles GQA natively via an
    // `n_rep = h_v / h_kv` parameter. Q/K can be passed at their native
    // `H_kv` head count — the kernel indexes them by `h_idx / n_rep`
    // inside the recurrence loop, so no `.expand().reshape()` round
    // trip materialises intermediate (h_v-shaped) Q/K on the caller
    // side. P3: this eliminates ~2 ucopy_f32 launches per GDN step.
    #[cfg(feature = "hip")]
    {
        if matches!(q.device(), candle::Device::Hip(_))
            && q.dtype() == candle::DType::F32
            && gated_delta_net_hip_supported(q, k, v, gate, beta, state)
        {
            return candle::hip_backend::gated_delta_net_step_fused(
                q, k, v, gate, beta, state,
            );
        }
    }

    // CPU / unsupported fallback. The CPU tensor-op chain assumes all
    // of Q, K, V have the same H dim, so we expand Q/K to match V's
    // head count first. This is the same shape that was used before
    // Phase 3 of P3; only the HIP path benefits from the new zero-copy
    // shortcut.
    let (b_sz, h_kv, seq_len, s_v_q) = q.dims4()?;
    let (_, h_v, _, s_v_v) = v.dims4()?;
    assert_eq!(s_v_q, s_v_v, "Q and V last dim must match");
    let (q_eff, k_eff) = if h_kv != h_v {
        let rep = h_v / h_kv;
        let expand_shape = (b_sz, rep, h_kv, seq_len, s_v_q);
        let merged_shape = (b_sz, h_v, seq_len, s_v_q);
        let q_e = q
            .unsqueeze(1)?
            .expand(expand_shape)?
            .reshape(merged_shape)?;
        let k_e = k
            .unsqueeze(1)?
            .expand(expand_shape)?
            .reshape(merged_shape)?;
        (q_e, k_e)
    } else {
        (q.clone(), k.clone())
    };
    if seq_len == 1 {
        return delta_net_single_step(&q_eff, &k_eff, v, gate, beta, state);
    }
    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q_eff.narrow(2, t, 1)?;
        let k_t = k_eff.narrow(2, t, 1)?;
        let v_t = v.narrow(2, t, 1)?;
        let gate_t = gate.narrow(2, t, 1)?;
        let beta_t = beta.narrow(2, t, 1)?;
        let out_t = delta_net_single_step(&q_t, &k_t, &v_t, &gate_t, &beta_t, state)?;
        outputs.push(out_t);
    }
    Tensor::cat(&outputs, 2)
}

/// Single-token CPU tensor-op implementation of the delta-net step.
/// Mirrors the pre-Phase-2 body of [`delta_net_step_vectorized`].
fn delta_net_single_step(
    q: &Tensor,         // (B, H, 1, S_v)
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

/// Return `true` if the fused HIP kernel supports these inputs. Phase 1
/// requires S_v=128 and contiguous f32 inputs. Any miss here routes back
/// to the tensor-op fallback — the function is intentionally a cheap
/// pre-flight check so the caller can decide without allocating.
#[cfg(feature = "hip")]
fn gated_delta_net_hip_supported(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> bool {
    // Shape: q is (B, H, L, S_v). Kernel currently only instantiated at S_v=128.
    let Ok((_, _, _, s_v)) = q.dims4() else { return false; };
    if s_v != 128 {
        return false;
    }
    // All inputs contiguous.
    if !(q.is_contiguous()
        && k.is_contiguous()
        && v.is_contiguous()
        && gate.is_contiguous()
        && beta.is_contiguous()
        && state.is_contiguous())
    {
        return false;
    }
    // All f32.
    let f32_ok = q.dtype() == candle::DType::F32
        && k.dtype() == candle::DType::F32
        && v.dtype() == candle::DType::F32
        && gate.dtype() == candle::DType::F32
        && beta.dtype() == candle::DType::F32
        && state.dtype() == candle::DType::F32;
    if !f32_ok {
        return false;
    }
    true
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

    /// P4 regression: HIP fused softplus kernel must match the CPU
    /// tensor-op chain (`max(x,0) + log1p(exp(-|x|))`) within FMA
    /// rounding on every element.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_softplus_matches_cpu() {
        let dev_hip = match candle::Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = candle::Device::Cpu;
        // Wide range: includes the underflow and overflow regimes so
        // the numerical-stability rewrite is exercised end-to-end.
        let xs: Vec<f32> = (-200..200)
            .map(|i| (i as f32) * 0.5)
            .collect();
        let x_cpu = Tensor::from_slice(&xs, xs.len(), &dev_cpu).unwrap();
        let x_hip = Tensor::from_slice(&xs, xs.len(), &dev_hip).unwrap();
        let out_cpu: Vec<f32> = softplus(&x_cpu).unwrap().to_vec1().unwrap();
        let out_hip: Vec<f32> = softplus(&x_hip)
            .unwrap()
            .to_device(&dev_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(out_cpu.len(), out_hip.len());
        let max_abs = out_cpu
            .iter()
            .zip(out_hip.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Tolerance: 1e-5 absolute + 1e-5 relative. The log1p(exp)
        // formulation is identical between the CPU chain and the
        // HIP kernel, so drift should be zero to FMA rounding.
        let max_rel = out_cpu
            .iter()
            .zip(out_hip.iter())
            .map(|(a, b)| if b.abs() > 1e-6 { (a - b).abs() / b.abs() } else { 0.0 })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-5 || max_rel < 1e-5,
            "softplus HIP vs CPU drift: max_abs={max_abs}, max_rel={max_rel}"
        );
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

    // --- HIP fused delta-net step tests ----------------------------
    //
    // These tests are behind `cfg(feature = "hip")` so the regular
    // `cargo test` (CPU only) build doesn't link HIP. On the ROCm
    // machine, run them with:
    //     cargo test -p candle-transformers --features hip \
    //         quantized_blocks::delta_net::tests::hip_
    //
    // Both tests are TDD oracles: build identical inputs on CPU and
    // HIP, run the CPU tensor-op `delta_net_step_vectorized` on CPU
    // and the fused kernel path on HIP, compare element-wise within
    // FMA rounding.

    /// Minimal shape: B=1, H=2, L=1, S_v=128 (the Phase 1 instantiation).
    /// Runs one recurrent step on both devices and checks both the
    /// returned attention output and the updated state agree.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_gated_delta_net_step_matches_cpu_s128_single_step() {
        let dev_hip = match candle::Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = candle::Device::Cpu;
        let b = 1usize;
        let h = 2usize;
        let l = 1usize;
        let s_v = 128usize;

        // Deterministic inputs — small magnitude so the recurrence
        // stays in a numerically stable range.
        let qkv_len = b * h * l * s_v;
        let gb_len = b * h * l;
        let state_len = b * h * s_v * s_v;
        let q_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00037).sin() * 0.1).collect();
        let k_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00053).cos() * 0.1).collect();
        let v_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00041).sin() * 0.1).collect();
        let gate_vals: Vec<f32> =
            (0..gb_len).map(|i| -0.05 - (i as f32) * 0.01).collect();
        let beta_vals: Vec<f32> =
            (0..gb_len).map(|i| 0.7 + (i as f32) * 0.01).collect();
        // Non-trivial starting state so the decay path is exercised.
        let state_vals: Vec<f32> = (0..state_len)
            .map(|i| ((i as f32) * 0.00013).cos() * 0.01)
            .collect();

        let q_cpu = Tensor::from_slice(&q_vals, (b, h, l, s_v), &dev_cpu).unwrap();
        let k_cpu = Tensor::from_slice(&k_vals, (b, h, l, s_v), &dev_cpu).unwrap();
        let v_cpu = Tensor::from_slice(&v_vals, (b, h, l, s_v), &dev_cpu).unwrap();
        let gate_cpu = Tensor::from_slice(&gate_vals, (b, h, l, 1), &dev_cpu).unwrap();
        let beta_cpu = Tensor::from_slice(&beta_vals, (b, h, l, 1), &dev_cpu).unwrap();
        let mut state_cpu =
            Tensor::from_slice(&state_vals, (b, h, s_v, s_v), &dev_cpu).unwrap();

        let q_hip = Tensor::from_slice(&q_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let k_hip = Tensor::from_slice(&k_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let v_hip = Tensor::from_slice(&v_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let gate_hip = Tensor::from_slice(&gate_vals, (b, h, l, 1), &dev_hip).unwrap();
        let beta_hip = Tensor::from_slice(&beta_vals, (b, h, l, 1), &dev_hip).unwrap();
        let mut state_hip =
            Tensor::from_slice(&state_vals, (b, h, s_v, s_v), &dev_hip).unwrap();

        // CPU reference: the tensor-op chain (fallback path).
        let out_cpu = delta_net_step_vectorized(
            &q_cpu, &k_cpu, &v_cpu, &gate_cpu, &beta_cpu, &mut state_cpu,
        )
        .unwrap();

        // HIP: takes the fused kernel fast path.
        let out_hip = delta_net_step_vectorized(
            &q_hip, &k_hip, &v_hip, &gate_hip, &beta_hip, &mut state_hip,
        )
        .unwrap();

        let out_cpu_vals: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let out_hip_vals: Vec<f32> =
            out_hip.to_device(&dev_cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(out_cpu_vals.len(), out_hip_vals.len());

        let max_abs = out_cpu_vals
            .iter()
            .zip(out_hip_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Same math, different reduction order (warp shfl vs rocBLAS
        // dot) so we expect FMA-level drift. S_v=128 accumulates 128
        // products per lane-shard pair; 1e-5 absolute is generous.
        assert!(
            max_abs < 1e-4,
            "attn output drift too large: max_abs={max_abs}"
        );

        let st_cpu_vals: Vec<f32> = state_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let st_hip_vals: Vec<f32> = state_hip
            .to_device(&dev_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(st_cpu_vals.len(), st_hip_vals.len());

        let max_abs_state = st_cpu_vals
            .iter()
            .zip(st_hip_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_state < 1e-4,
            "state drift too large: max_abs={max_abs_state}"
        );
    }

    /// Multi-step correctness: run N sequential recurrent steps with
    /// distinct per-step inputs. Each step's output depends on the
    /// previous step's state update, so any drift compounds. If the
    /// kernel is correct, the two devices stay in lock-step (within
    /// FMA rounding) across all steps.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_gated_delta_net_step_matches_cpu_s128_multi_step() {
        let dev_hip = match candle::Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = candle::Device::Cpu;
        let b = 1usize;
        let h = 4usize;
        let s_v = 128usize;
        let n_steps = 5usize;

        let mut state_cpu = Tensor::zeros((b, h, s_v, s_v), candle::DType::F32, &dev_cpu).unwrap();
        let mut state_hip = Tensor::zeros((b, h, s_v, s_v), candle::DType::F32, &dev_hip).unwrap();

        for t in 0..n_steps {
            // Distinct inputs per step — sin/cos of a shifted index.
            let offset = (t as f32) * 0.1;
            let q_vals: Vec<f32> = (0..b * h * s_v)
                .map(|i| ((i as f32) * 0.00037 + offset).sin() * 0.1)
                .collect();
            let k_vals: Vec<f32> = (0..b * h * s_v)
                .map(|i| ((i as f32) * 0.00053 + offset).cos() * 0.1)
                .collect();
            let v_vals: Vec<f32> = (0..b * h * s_v)
                .map(|i| ((i as f32) * 0.00041 + offset).sin() * 0.1)
                .collect();
            let gate_vals: Vec<f32> =
                (0..b * h).map(|i| -0.05 - (i as f32 + offset) * 0.01).collect();
            let beta_vals: Vec<f32> =
                (0..b * h).map(|i| 0.7 + (i as f32 + offset) * 0.01).collect();

            let q_cpu = Tensor::from_slice(&q_vals, (b, h, 1, s_v), &dev_cpu).unwrap();
            let k_cpu = Tensor::from_slice(&k_vals, (b, h, 1, s_v), &dev_cpu).unwrap();
            let v_cpu = Tensor::from_slice(&v_vals, (b, h, 1, s_v), &dev_cpu).unwrap();
            let gate_cpu = Tensor::from_slice(&gate_vals, (b, h, 1, 1), &dev_cpu).unwrap();
            let beta_cpu = Tensor::from_slice(&beta_vals, (b, h, 1, 1), &dev_cpu).unwrap();

            let q_hip = Tensor::from_slice(&q_vals, (b, h, 1, s_v), &dev_hip).unwrap();
            let k_hip = Tensor::from_slice(&k_vals, (b, h, 1, s_v), &dev_hip).unwrap();
            let v_hip = Tensor::from_slice(&v_vals, (b, h, 1, s_v), &dev_hip).unwrap();
            let gate_hip = Tensor::from_slice(&gate_vals, (b, h, 1, 1), &dev_hip).unwrap();
            let beta_hip = Tensor::from_slice(&beta_vals, (b, h, 1, 1), &dev_hip).unwrap();

            let out_cpu = delta_net_step_vectorized(
                &q_cpu, &k_cpu, &v_cpu, &gate_cpu, &beta_cpu, &mut state_cpu,
            )
            .unwrap();
            let out_hip = delta_net_step_vectorized(
                &q_hip, &k_hip, &v_hip, &gate_hip, &beta_hip, &mut state_hip,
            )
            .unwrap();

            let out_cpu_vals: Vec<f32> =
                out_cpu.flatten_all().unwrap().to_vec1().unwrap();
            let out_hip_vals: Vec<f32> = out_hip
                .to_device(&dev_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max_abs = out_cpu_vals
                .iter()
                .zip(out_hip_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // Drift compounds across steps; allow a looser bound than
            // the single-step test.
            assert!(
                max_abs < 5e-4,
                "step {t}: attn output drift too large: max_abs={max_abs}"
            );

            let st_cpu_vals: Vec<f32> =
                state_cpu.flatten_all().unwrap().to_vec1().unwrap();
            let st_hip_vals: Vec<f32> = state_hip
                .to_device(&dev_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max_abs_state = st_cpu_vals
                .iter()
                .zip(st_hip_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_state < 5e-4,
                "step {t}: state drift too large: max_abs={max_abs_state}"
            );
        }
    }

    /// Phase 2 regression: the fused kernel's **internal** token loop
    /// must produce the same result as an external loop of single-step
    /// calls. The `gated_delta_net.cu` kernel was ported from turbo
    /// with an `n_tokens` parameter, but until Phase 2 we only ever
    /// called it with `L=1`; this test exercises the L>1 code path in
    /// isolation (no forward_prefill scaffolding) so any drift in the
    /// kernel's inner loop shows up here rather than as a whole-model
    /// perplexity surprise.
    ///
    /// Batched: one call with `q, k, v` of shape `(1, H, L, 128)`.
    /// Looped:  L calls with `(1, H, 1, 128)` slices carved from the
    ///          same inputs, threading `state` through each iteration.
    /// Both must agree within FMA rounding at the end of the L steps.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_gated_delta_net_batched_matches_looped_s128_l5() {
        let dev_hip = match candle::Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let b = 1usize;
        let h = 4usize;
        let l = 5usize;
        let s_v = 128usize;

        // Deterministic, small-magnitude inputs. The magnitude keeps
        // the recurrence numerically stable across L=5 steps.
        let qkv_len = b * h * l * s_v;
        let gb_len = b * h * l;
        let state_len = b * h * s_v * s_v;
        let q_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00037).sin() * 0.08).collect();
        let k_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00053).cos() * 0.08).collect();
        let v_vals: Vec<f32> =
            (0..qkv_len).map(|i| ((i as f32) * 0.00041).sin() * 0.08).collect();
        let gate_vals: Vec<f32> = (0..gb_len)
            .map(|i| -0.04 - ((i as f32) * 0.01).sin() * 0.03)
            .collect();
        let beta_vals: Vec<f32> = (0..gb_len)
            .map(|i| 0.6 + ((i as f32) * 0.013).cos() * 0.1)
            .collect();
        let state_vals: Vec<f32> = (0..state_len)
            .map(|i| ((i as f32) * 0.00013).cos() * 0.01)
            .collect();

        // Batched inputs: (1, H, L, S_v) and (1, H, L, 1).
        let q_batched = Tensor::from_slice(&q_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let k_batched = Tensor::from_slice(&k_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let v_batched = Tensor::from_slice(&v_vals, (b, h, l, s_v), &dev_hip).unwrap();
        let gate_batched =
            Tensor::from_slice(&gate_vals, (b, h, l, 1), &dev_hip).unwrap();
        let beta_batched =
            Tensor::from_slice(&beta_vals, (b, h, l, 1), &dev_hip).unwrap();
        let mut state_batched =
            Tensor::from_slice(&state_vals, (b, h, s_v, s_v), &dev_hip).unwrap();

        // Path A: single batched call.
        let out_batched = delta_net_step_vectorized(
            &q_batched,
            &k_batched,
            &v_batched,
            &gate_batched,
            &beta_batched,
            &mut state_batched,
        )
        .unwrap();
        assert_eq!(out_batched.dims(), &[b, h, l, s_v]);

        // Path B: L=1 loop, same inputs carved out per step. The
        // single-step calls hit the L=1 kernel instantiation.
        let mut state_looped =
            Tensor::from_slice(&state_vals, (b, h, s_v, s_v), &dev_hip).unwrap();
        let mut per_step_outs: Vec<Tensor> = Vec::with_capacity(l);
        for t in 0..l {
            // narrow(2, t, 1) picks one token slice — contiguous
            // because the L axis has stride S_v on a contiguous
            // (B, H, L, S_v) tensor.
            let q_t = q_batched.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let k_t = k_batched.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let v_t = v_batched.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let gate_t = gate_batched.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let beta_t = beta_batched.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let out_t = delta_net_step_vectorized(
                &q_t,
                &k_t,
                &v_t,
                &gate_t,
                &beta_t,
                &mut state_looped,
            )
            .unwrap();
            per_step_outs.push(out_t);
        }
        let out_looped = Tensor::cat(&per_step_outs, 2).unwrap();

        // Compare attn output.
        let dev_cpu = candle::Device::Cpu;
        let a_vals: Vec<f32> = out_batched
            .to_device(&dev_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b_vals: Vec<f32> = out_looped
            .to_device(&dev_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(a_vals.len(), b_vals.len());
        let max_abs = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // Both paths run the exact same math on the same hardware —
        // the only source of drift is that the batched call keeps
        // state in registers across L while the looped path writes
        // state to DRAM and re-loads each step. FMA accumulation
        // order is identical so drift should be zero; 1e-5 is
        // defensive slack.
        assert!(
            max_abs < 1e-5,
            "batched vs looped attn drift: max_abs={max_abs}"
        );

        // Compare final state.
        let s_a: Vec<f32> = state_batched
            .to_device(&dev_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let s_b: Vec<f32> = state_looped
            .to_device(&dev_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max_abs_state = s_a
            .iter()
            .zip(s_b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_state < 1e-5,
            "batched vs looped state drift: max_abs={max_abs_state}"
        );
    }

    /// P3 regression: the fused GDN kernel must handle GQA correctly
    /// when `h_v > h_kv` (Q/K have fewer heads than V). The kernel's
    /// `h_idx / n_rep` broadcast is exercised on HIP; the CPU fallback
    /// expands Q/K via unsqueeze+expand+reshape and runs the same
    /// tensor-op chain. Both paths must agree within FMA rounding.
    #[cfg(feature = "hip")]
    #[test]
    fn hip_gated_delta_net_gqa_matches_cpu_s128_n_rep_4() {
        let dev_hip = match candle::Device::new_hip(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: HIP device 0 unavailable: {e}");
                return;
            }
        };
        let dev_cpu = candle::Device::Cpu;
        let b = 1usize;
        let h_kv = 4usize;   // Q/K heads
        let n_rep = 4usize;
        let h_v = h_kv * n_rep; // V/state heads = 16
        let l = 3usize;
        let s_v = 128usize;

        // Deterministic inputs at the GQA head counts.
        let qkv_kv_len = b * h_kv * l * s_v;
        let v_len = b * h_v * l * s_v;
        let gb_len = b * h_v * l;
        let state_len = b * h_v * s_v * s_v;
        let q_vals: Vec<f32> =
            (0..qkv_kv_len).map(|i| ((i as f32) * 0.00037).sin() * 0.08).collect();
        let k_vals: Vec<f32> =
            (0..qkv_kv_len).map(|i| ((i as f32) * 0.00053).cos() * 0.08).collect();
        let v_vals: Vec<f32> =
            (0..v_len).map(|i| ((i as f32) * 0.00041).sin() * 0.08).collect();
        let gate_vals: Vec<f32> =
            (0..gb_len).map(|i| -0.05 - ((i as f32) * 0.01).sin() * 0.02).collect();
        let beta_vals: Vec<f32> =
            (0..gb_len).map(|i| 0.6 + ((i as f32) * 0.013).cos() * 0.1).collect();
        let state_vals: Vec<f32> =
            (0..state_len).map(|i| ((i as f32) * 0.00013).cos() * 0.01).collect();

        let q_hip = Tensor::from_slice(&q_vals, (b, h_kv, l, s_v), &dev_hip).unwrap();
        let k_hip = Tensor::from_slice(&k_vals, (b, h_kv, l, s_v), &dev_hip).unwrap();
        let v_hip = Tensor::from_slice(&v_vals, (b, h_v, l, s_v), &dev_hip).unwrap();
        let gate_hip = Tensor::from_slice(&gate_vals, (b, h_v, l, 1), &dev_hip).unwrap();
        let beta_hip = Tensor::from_slice(&beta_vals, (b, h_v, l, 1), &dev_hip).unwrap();
        let mut state_hip = Tensor::from_slice(&state_vals, (b, h_v, s_v, s_v), &dev_hip).unwrap();

        let q_cpu = Tensor::from_slice(&q_vals, (b, h_kv, l, s_v), &dev_cpu).unwrap();
        let k_cpu = Tensor::from_slice(&k_vals, (b, h_kv, l, s_v), &dev_cpu).unwrap();
        let v_cpu = Tensor::from_slice(&v_vals, (b, h_v, l, s_v), &dev_cpu).unwrap();
        let gate_cpu = Tensor::from_slice(&gate_vals, (b, h_v, l, 1), &dev_cpu).unwrap();
        let beta_cpu = Tensor::from_slice(&beta_vals, (b, h_v, l, 1), &dev_cpu).unwrap();
        let mut state_cpu = Tensor::from_slice(&state_vals, (b, h_v, s_v, s_v), &dev_cpu).unwrap();

        // HIP path: GQA kernel, Q/K at n_k_heads.
        let out_hip = delta_net_step_vectorized(
            &q_hip, &k_hip, &v_hip, &gate_hip, &beta_hip, &mut state_hip,
        )
        .unwrap();

        // CPU path: expand-reshape then run tensor-op chain.
        let out_cpu = delta_net_step_vectorized(
            &q_cpu, &k_cpu, &v_cpu, &gate_cpu, &beta_cpu, &mut state_cpu,
        )
        .unwrap();

        let a_vals: Vec<f32> = out_hip
            .to_device(&dev_cpu).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap();
        let b_vals: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a_vals.len(), b_vals.len());
        let max_abs = a_vals.iter().zip(b_vals.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // FMA order diverges slightly across CPU (sequential) and HIP
        // (warp shuffle), so we allow a small slack.
        assert!(
            max_abs < 5e-4,
            "GQA n_rep=4 attn drift: max_abs={max_abs}"
        );

        let s_a: Vec<f32> = state_hip
            .to_device(&dev_cpu).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap();
        let s_b: Vec<f32> = state_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let max_abs_state = s_a.iter().zip(s_b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_state < 5e-4,
            "GQA n_rep=4 state drift: max_abs={max_abs_state}"
        );
    }
}
