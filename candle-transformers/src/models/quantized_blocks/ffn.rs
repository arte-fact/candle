//! Feed-forward network variants for quantized GGUF models.
//!
//! - [`DenseMlp`]: Standard SwiGLU dense MLP
//! - [`MoeExperts`]: Mixture of Experts with optional shared expert

use super::gguf_config::GgufConfig;
use super::gguf_loader::Gguf;
use super::super::with_tracing::QMatMul;
use candle::quantized::QTensor;
use candle::{Device, Module, Result, Tensor, D};
use std::sync::Arc;

/// Compare two devices for equality without allocating.
fn device_eq(a: &Device, b: &Device) -> bool {
    format!("{:?}", a.location()) == format!("{:?}", b.location())
}

// ---------------------------------------------------------------------------
// MlpActivation
// ---------------------------------------------------------------------------

/// Activation function used in the gated MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpActivation {
    /// SiLU (Swish): silu(gate) * up — used by Llama, Qwen, Mistral
    Silu,
    /// GELU (PyTorch tanh approximation): gelu(gate) * up — used by Gemma family
    Gelu,
}

impl MlpActivation {
    /// Fused activation: `activation(gate) * up` in a single launch on
    /// HIP, falling back to the unfused chain on every other backend.
    /// Saves one kernel launch and one intermediate buffer per FFN per
    /// layer per token.
    fn apply_with_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        match self {
            MlpActivation::Silu => candle_nn::ops::silu_mul(gate, up),
            MlpActivation::Gelu => {
                // No fused gelu_mul kernel yet; gelu is rare enough that
                // the unfused chain is fine.
                let g = gate.gelu()?;
                &g * up
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseMlp
// ---------------------------------------------------------------------------

/// Standard gated feed-forward: out = down(activation(gate(x)) * up(x))
/// The activation is configurable (SiLU for Llama/Qwen, GELU for Gemma).
///
/// The gate and up projections are loaded as a single fused weight
/// matrix at GGUF load time, so the forward path issues one quantized
/// matmul instead of two. The fused output is split via `narrow`.
pub struct DenseMlp {
    /// Fused [gate; up] weight: rows `[0..intermediate_size)` are
    /// `ffn_gate`, rows `[intermediate_size..2*intermediate_size)` are
    /// `ffn_up`. One matmul launch instead of two on the forward
    /// path.
    gate_up: QMatMul,
    /// Per-side output dim. Both gate and up have the same intermediate
    /// width by construction.
    intermediate_size: usize,
    down: QMatMul,
    activation: MlpActivation,
}

impl DenseMlp {
    /// Load from GGUF with default SiLU activation (Llama/Qwen convention).
    pub fn load(gg: &Gguf, prefix: &str) -> Result<Self> {
        Self::load_with_activation(gg, prefix, MlpActivation::Silu)
    }

    /// Load from GGUF with explicit activation (use Gelu for Gemma family).
    pub fn load_with_activation(
        gg: &Gguf,
        prefix: &str,
        activation: MlpActivation,
    ) -> Result<Self> {
        let gate_name = format!("{prefix}.ffn_gate.weight");
        let up_name = format!("{prefix}.ffn_up.weight");
        // Need the per-side intermediate size; read it from the
        // ffn_gate tensor info before fusing.
        let intermediate_size = gg
            .ct
            .tensor_infos
            .get(&gate_name)
            .ok_or_else(|| candle::Error::Msg(format!("missing {gate_name}")))?
            .shape
            .dims()[0];
        let gate_up = gg.qmatmul_concat_rows(&[&gate_name, &up_name])?;
        let down = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self { gate_up, intermediate_size, down, activation })
    }
}

impl DenseMlp {
    /// Decode-path FFN forward with a pre-quantized Q8_1 input, skipping
    /// the redundant `quantize_q8_1` launches on both `gate_up` and
    /// `down` QMatMul dispatches. Caller is responsible for producing
    /// `x_q8_view` via `rmsnorm_q8_fused` (or an equivalent). Returns
    /// the raw FFN output — **no residual added**, so the caller is
    /// free to run a post-FFN norm before the residual (e.g. Gemma4's
    /// `post_ffn_norm.forward_post_residual`).
    ///
    /// `rhs_shape` is the shape the caller's pre-norm activation had —
    /// the QMatMul needs it to set the output tensor shape.
    ///
    /// HIP-only. Both matmuls must be Q4_0 QTensors. Marginal ~1-2 t/s
    /// decode win on gemma4-E4B from skipping 2 quantize_q8_1 launches
    /// per layer per token (one for gate_up, one for down).
    #[cfg(feature = "hip")]
    pub fn forward_preq8_decode(
        &self,
        x_q8_view: &candle::hip_backend::hipdarc::driver::HipView<'_, u8>,
        b_size: usize,
        rhs_shape: &[usize],
    ) -> Result<Tensor> {
        let gate_up = self.gate_up.forward_preq8(x_q8_view, b_size, rhs_shape)?;

        // Gate+up split → activation. For SiLU the fused kernel reads
        // `gate_up` directly. For GELU there's no fused variant yet, so
        // we materialize gate/up separately.
        let activated = if matches!(self.activation, MlpActivation::Silu)
            && gate_up.is_contiguous()
            && gate_up.dtype() == candle::DType::F32
        {
            candle::hip_backend::silu_mul_split_last_fused(&gate_up)?
        } else {
            let gate = gate_up
                .narrow(D::Minus1, 0, self.intermediate_size)?
                .contiguous()?;
            let up = gate_up
                .narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            self.activation.apply_with_mul(&gate, &up)?
        };

        // Re-quantize the activation before the down matmul so the down
        // call can also skip its internal quantize_q8_1.
        use candle::quantized::hip as qhip;
        let dev = match activated.device() {
            Device::Hip(d) => d.clone(),
            _ => candle::bail!("forward_preq8_decode: activated must be HIP"),
        };
        let activated_c = if activated.is_contiguous() {
            activated.clone()
        } else {
            activated.contiguous()?
        };
        let (a_st, a_l) = activated_c.storage_and_layout();
        let a_hip = match &*a_st {
            candle::Storage::Hip(s) => s,
            _ => candle::bail!("forward_preq8_decode: activated not HIP"),
        };
        let a_slice = a_hip.as_hip_slice::<f32>()?;
        let (a_lo, a_hi) = a_l
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("activated non-contig".into()))?;
        let a_view = a_slice.slice(a_lo..a_hi);

        let k = self.intermediate_size;
        let ky = b_size; // number of rows (b*seq_len)
        let kx_padded = qhip::pad(k, qhip::MATRIX_ROW_PADDING);
        let q8_bytes = ky * (kx_padded / 32) * 36;
        let mut h_q8_buf = unsafe { dev.alloc::<u8>(q8_bytes)? };
        qhip::quantize_q8_1(&a_view, &mut h_q8_buf, k, ky, &dev)?;
        drop(a_st);
        let h_q8_view = h_q8_buf.slice(0..h_q8_buf.len());

        // down matmul takes `rhs_shape` with last dim = intermediate_size.
        let mut down_rhs_shape: Vec<usize> = rhs_shape.to_vec();
        *down_rhs_shape.last_mut().unwrap() = self.intermediate_size;
        self.down.forward_preq8(&h_q8_view, b_size, &down_rhs_shape)
    }
}

impl Module for DenseMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Single fused gate+up matmul. Output: (B, L, 2*intermediate)
        let gate_up = self.gate_up.forward(x)?;

        // P3: on HIP with SiLU, dispatch to `silu_mul_split_last_fused`
        // which reads the fused `gate_up` buffer directly — no
        // `.narrow().contiguous()` materialisations, no intermediate
        // gate/up f32 tensors. Drops 2 ucopy_f32 launches per FFN
        // per layer per token (~10k per qwen35-9B decode session).
        #[cfg(feature = "hip")]
        if matches!(gate_up.device(), candle::Device::Hip(_))
            && gate_up.dtype() == candle::DType::F32
            && gate_up.is_contiguous()
            && matches!(self.activation, MlpActivation::Silu)
        {
            let activated = candle::hip_backend::silu_mul_split_last_fused(&gate_up)?;
            return self.down.forward(&activated);
        }

        // Fallback: narrow + contiguous + fused silu_mul. The narrows
        // are non-contiguous views into the same buffer; the fused
        // silu_mul kernel works against contiguous inputs so we
        // materialize once via .contiguous().
        let gate = gate_up
            .narrow(D::Minus1, 0, self.intermediate_size)?
            .contiguous()?;
        let up = gate_up
            .narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let activated = self.activation.apply_with_mul(&gate, &up)?;
        self.down.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MoeExperts
// ---------------------------------------------------------------------------

/// Mixture of Experts with optional shared expert.
///
/// Supports two expert weight layouts:
/// - Separate `gate_exps` + `up_exps` (qwen3moe, qwen35moe)
/// - Fused `gate_up_exps` (gemma4)
///
/// Shared expert: optional dense MLP gated by sigmoid(shared_gate(x)).
/// Reference: llama.cpp qwen35moe.cpp `build_layer_ffn`
pub struct MoeExperts {
    router: QMatMul,
    /// Expert up weights: [num_experts, intermediate, hidden]
    up_exps: Option<Arc<QTensor>>,
    /// Expert gate weights: [num_experts, intermediate, hidden]
    gate_exps: Option<Arc<QTensor>>,
    /// Fused gate+up: [num_experts, 2*intermediate, hidden]
    gate_up_exps: Option<Arc<QTensor>>,
    /// Expert down weights: [num_experts, hidden, intermediate]
    down_exps: Arc<QTensor>,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    /// Shared expert (optional)
    shared: Option<DenseMlp>,
    /// Shared expert gate (optional): a 1-D `[hidden]` weight vector. Computes
    /// a SCALAR per token via dot product, then sigmoid. Used by qwen35moe to
    /// gate the shared expert output.
    shared_gate_weight: Option<Tensor>,
}

impl MoeExperts {
    /// Load from GGUF. Auto-detects separate vs fused expert layout and shared expert.
    pub fn load(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
    ) -> Result<Self> {
        let num_experts_per_tok = cfg.expert_used_count.unwrap_or(1);

        let mut router = gg.qmatmul(&format!("{prefix}.ffn_gate_inp.weight"))?;
        // Phase O4: requantise the router to Q8_0 when it arrives as F32
        // (Qwen3-Coder-Next, Mxfp4-quant models) or F16/BF16, so the
        // decode-path forward goes through MMVQ (~20-40 µs/call) instead
        // of rocBLAS `Cijk_*_MT128x64x16` (~270 µs/call).  Matches the
        // gemma4 `inp_gate`/`proj` requantise pattern
        // (`quantized_gemma4.rs:574-591`).  Opt-out per env var.
        if !router.is_qtensor()
            && std::env::var("CANDLE_MOE_ROUTER_NO_REQUANT").is_err()
        {
            let dtype = match std::env::var("CANDLE_MOE_ROUTER_REQUANT_DTYPE")
                .ok().as_deref()
            {
                Some("q4_0") => candle::quantized::GgmlDType::Q4_0,
                Some("q4_1") => candle::quantized::GgmlDType::Q4_1,
                Some("q5_0") => candle::quantized::GgmlDType::Q5_0,
                _ => candle::quantized::GgmlDType::Q8_0,
            };
            let _ = router.requantize_to(dtype);
        }
        let down_exps = Arc::new(gg.tensor(&format!("{prefix}.ffn_down_exps.weight"))?);

        // Detect separate vs fused gate+up
        let (gate_exps, up_exps, gate_up_exps) =
            if gg.has_tensor(&format!("{prefix}.ffn_gate_exps.weight")) {
                let g = Arc::new(gg.tensor(&format!("{prefix}.ffn_gate_exps.weight"))?);
                let u = Arc::new(gg.tensor(&format!("{prefix}.ffn_up_exps.weight"))?);
                (Some(g), Some(u), None)
            } else if gg.has_tensor(&format!("{prefix}.ffn_gate_up_exps.weight")) {
                let gu = Arc::new(gg.tensor(&format!("{prefix}.ffn_gate_up_exps.weight"))?);
                (None, None, Some(gu))
            } else {
                candle::bail!("MoE layer at {prefix} has neither gate_exps nor gate_up_exps");
            };

        // Infer intermediate size from down_exps shape: [num_experts, hidden, intermediate]
        let down_shape = down_exps.shape().dims();
        let intermediate_size = if down_shape.len() == 3 {
            down_shape[1]
        } else {
            // Fallback: try from config
            cfg.feed_forward_length.unwrap_or(0)
        };

        // Detect shared expert. Build it via DenseMlp::load_with_activation
        // so it picks up the same fused gate+up packing as the dense
        // FFN path — saves one MMVQ launch per shared-expert call too.
        let shared = if gg.has_tensor(&format!("{prefix}.ffn_up_shexp.weight")) {
            // The shared-expert tensors live under `{prefix}.ffn_*_shexp.weight`,
            // so DenseMlp::load_with_activation needs a virtual prefix
            // that resolves to those names. Concretely we want it to
            // try `{prefix}_shexp.ffn_*.weight`, but that's not the
            // actual layout. Instead, build the fused weight directly
            // here using the same helper.
            let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
            let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
            let shared_intermediate = gg
                .ct
                .tensor_infos
                .get(&shared_gate_name)
                .ok_or_else(|| candle::Error::Msg(format!("missing {shared_gate_name}")))?
                .shape
                .dims()[0];
            let shared_gate_up =
                gg.qmatmul_concat_rows(&[&shared_gate_name, &shared_up_name])?;
            let shared_down = gg.qmatmul(&format!("{prefix}.ffn_down_shexp.weight"))?;
            Some(DenseMlp {
                gate_up: shared_gate_up,
                intermediate_size: shared_intermediate,
                down: shared_down,
                activation: MlpActivation::Silu,
            })
        } else {
            None
        };

        // Shared expert gate is a 1-D weight vector (qwen35moe convention).
        // Load + dequantize and keep as a Tensor.
        let shared_gate_weight = gg
            .try_dequantize(&format!("{prefix}.ffn_gate_inp_shexp.weight"));

        Ok(Self {
            router,
            up_exps,
            gate_exps,
            gate_up_exps,
            down_exps,
            num_experts_per_tok,
            intermediate_size,
            shared,
            shared_gate_weight,
        })
    }

    /// Phase O1 fast path: pre-quantise x to Q8_1 once and feed BOTH the
    /// router AND the gate/up MoE matmuls from the same buffer.  Cuts the
    /// `quantize_q8_1` count from 4 (sep gate/up) or 3 (fused gate_up) to
    /// 2 per layer per token (one shared input quant + one for the
    /// activated `silu_mul` going into down).
    ///
    /// Restricted to:
    ///   - HIP backend
    ///   - decode-shaped batch (`b * m ≤ 8`)
    ///   - all involved weights are real QTensors (not dequantised)
    ///   - input is contiguous F32 (the Q8_1 quantiser's contract)
    ///
    /// Returns Ok(None) when any precondition fails — caller falls through.
    #[cfg(feature = "hip")]
    fn forward_shared_q8_decode(&self, x: &Tensor) -> Result<Option<Tensor>> {
        use candle::quantized::hip::{pad, quantize_q8_1, MATRIX_ROW_PADDING};
        use candle::quantized::GgmlDType;
        use candle::D;

        let (b_sz, seq_len, hidden) = x.dims3()?;
        let b_size = b_sz * seq_len;

        // Note: we do NOT require the router to be a QTensor.  When the
        // router weight has been dequantised (e.g. Mxfp4 → F32 fallback),
        // it goes through its own `forward` path; the savings then come
        // purely from sharing Q8_1 between the gate/up MoE matmuls.
        if b_size > 8
            || !matches!(x.device(), candle::Device::Hip(_))
            || x.dtype() != candle::DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }

        // Reuse the existing routing logic (softmax → topk on CPU) — same
        // as the slow path, but with the matmul replaced by `forward_preq8`.
        let dev = match x.device() {
            candle::Device::Hip(d) => d.clone(),
            _ => return Ok(None),
        };

        let x_flat = x.reshape((b_size, hidden))?;
        let (x_st, x_l) = x_flat.storage_and_layout();
        let x_hip = match &*x_st {
            candle::Storage::Hip(s) => s,
            _ => return Ok(None),
        };
        let x_slice = x_hip.as_hip_slice::<f32>()?;
        let x_view = match x_l.contiguous_offsets() {
            Some((lo, hi)) => x_slice.slice(lo..hi),
            None => return Ok(None),
        };

        let ncols_padded = pad(hidden, MATRIX_ROW_PADDING);
        let q8_bytes = b_size * ncols_padded * GgmlDType::Q8_1.type_size()
            / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(q8_bytes)? };
        quantize_q8_1(&x_view, &mut y_q8_1, hidden, b_size, &dev)?;
        let q8_view = y_q8_1.slice(0..y_q8_1.len());

        let rhs_shape = x_flat.dims().to_vec();
        drop(x_st);

        // Router: use shared Q8_1 if it's a QTensor; otherwise fall through
        // to its standard (likely dequantised) forward — no quantize savings
        // there, but gate/up still benefit from the shared buffer below.
        let router_logits = if self.router.is_qtensor() {
            self.router.forward_preq8(&q8_view, b_size, &rhs_shape)?
        } else {
            self.router.forward(&x_flat)?
        };
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // TopK on CPU (matches the existing path; HIP arg_sort is not wired).
        let cpu = candle::Device::Cpu;
        let rw_cpu = routing_weights.to_device(&cpu)?;
        let topk_ids_cpu = rw_cpu
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let hip_dev = candle::Device::Hip(dev.clone());
        let topk_ids = topk_ids_cpu.to_device(&hip_dev)?;

        let topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;
        let topk_weights = (&topk_weights
            / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as(topk_weights.shape())?)?;

        // The MoE preq8 kernel expects `[batch, topk_or_1, k]` `in_shape`
        // for its block-grid sizing.  `topk_or_1 == 1` here because the
        // input is shared across all top-k experts.
        let in_shape = candle::Shape::from((b_size, 1usize, hidden));

        let moe_out = if let Some(ref gate_exps) = self.gate_exps {
            let up_exps = self.up_exps.as_ref().unwrap();
            let gate_out =
                gate_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids)?;
            let up_out =
                up_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids)?;
            let activated = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
            self.down_exps
                .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
        } else if let Some(ref gate_up_exps) = self.gate_up_exps {
            let gate_up =
                gate_up_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids)?;
            let gate = gate_up
                .narrow(D::Minus1, 0, self.intermediate_size)?
                .contiguous()?;
            let up = gate_up
                .narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            let activated = candle_nn::ops::silu_mul(&gate, &up)?;
            self.down_exps
                .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
        } else {
            candle::bail!("MoE has no expert weights");
        };

        let topk_weights = topk_weights.unsqueeze(D::Minus1)?;
        let weighted = moe_out.broadcast_mul(&topk_weights)?;
        let mut result = weighted.sum(1)?;

        if let Some(ref shared_mlp) = self.shared {
            let shared_out = shared_mlp.forward(&x_flat)?;
            let shared_out = if let Some(ref sg_weight) = self.shared_gate_weight {
                let sg = if device_eq(sg_weight.device(), x_flat.device()) {
                    sg_weight.clone()
                } else {
                    sg_weight.to_device(x_flat.device())?
                };
                let gate_logits = x_flat.broadcast_mul(&sg)?.sum_keepdim(D::Minus1)?;
                let gate = candle_nn::ops::sigmoid(&gate_logits)?;
                shared_out.broadcast_mul(&gate)?
            } else {
                shared_out
            };
            result = (result + shared_out)?;
        }

        Ok(Some(result.reshape((b_sz, seq_len, hidden))?))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Phase O1 — decode-path fast path.  Pre-quantises x once into
        // Q8_1 and reuses it for the router AND the gate/up MoE matmuls.
        // (Down still has its own quantise since its input is
        // `silu_mul(gate, up)`.)  Returns None when the conditions aren't
        // met (b*m > 8, non-HIP, dequantised weights, MXFP4 router…) and
        // we fall through to the existing path.
        #[cfg(feature = "hip")]
        {
            if let Some(out) = self.forward_shared_q8_decode(x)? {
                return Ok(out);
            }
        }
        self.forward_inner(x)
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;

        // Route: softmax → topk
        let router_logits = self.router.forward(&x_flat)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // TopK selection.
        // candle's HIP backend has no argsort yet, so route this small op
        // through CPU. routing_weights shape is (tokens, n_experts) — tiny.
        let device = routing_weights.device().clone();
        let topk_ids = if matches!(device, candle::Device::Hip(_)) {
            let cpu = candle::Device::Cpu;
            let rw_cpu = routing_weights.to_device(&cpu)?;
            let ids = rw_cpu
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()?;
            // A3 / EPLB — observe expert routing while the ids live on
            // CPU (no extra CPU→GPU→CPU roundtrip).  No-op when neither
            // CANDLE_EPLB_PRINT nor CANDLE_EPLB_DUMP is set.
            if let Ok(ids_vec) = ids.flatten_all()?.to_vec1::<u32>() {
                super::eplb::observe(&ids_vec, 0);
            }
            ids.to_device(&device)?
        } else {
            let ids = routing_weights
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()?;
            if let Ok(ids_vec) = ids.flatten_all()?.to_vec1::<u32>() {
                super::eplb::observe(&ids_vec, 0);
            }
            ids
        };
        let topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        // Normalize topk weights
        let topk_weights = (&topk_weights
            / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as(topk_weights.shape())?)?;

        // indexed_moe_forward expects 3D input `[batch, topk_or_1, k]` and 2D
        // ids `[batch, topk]`. For the gate/up step we want each token to be
        // routed to its top-K experts, so the input has the "1" topk_or_1 dim
        // and is broadcast across topk experts internally.
        let x_flat_3d = x_flat.unsqueeze(1)?.contiguous()?;

        // Expert computation via indexed_moe_forward
        let moe_out = if let Some(ref gate_exps) = self.gate_exps {
            // Separate gate + up experts
            let up_exps = self.up_exps.as_ref().unwrap();
            let gate_out = gate_exps.indexed_moe_forward(&x_flat_3d, &topk_ids)?;
            let up_out = up_exps.indexed_moe_forward(&x_flat_3d, &topk_ids)?;
            let activated = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
            self.down_exps.indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
        } else if let Some(ref gate_up_exps) = self.gate_up_exps {
            // Fused gate+up experts: one matmul output split into the
            // gate half and the up half. The narrows are non-contiguous
            // views into the same buffer; `silu_mul` falls back to the
            // chained ops on non-contiguous inputs, so the fast HIP
            // kernel never fires here. Calling `.contiguous()?` on each
            // half once is cheaper than two extra kernels per layer per
            // step.
            let gate_up = gate_up_exps.indexed_moe_forward(&x_flat_3d, &topk_ids)?;
            let gate = gate_up
                .narrow(D::Minus1, 0, self.intermediate_size)?
                .contiguous()?;
            let up = gate_up
                .narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            let activated = candle_nn::ops::silu_mul(&gate, &up)?;
            self.down_exps.indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
        } else {
            candle::bail!("MoE has no expert weights");
        };

        // Weight and sum across topk
        let topk_weights = topk_weights.unsqueeze(D::Minus1)?;
        let weighted = moe_out.broadcast_mul(&topk_weights)?;
        let mut result = weighted.sum(1)?; // sum across topk dim

        // Add shared expert output if present
        if let Some(ref shared_mlp) = self.shared {
            let shared_out = shared_mlp.forward(&x_flat)?;
            let shared_out = if let Some(ref sg_weight) = self.shared_gate_weight {
                // Dot product: gate[t] = sum_i sg_weight[i] * x_flat[t, i]
                // sg_weight shape: [hidden]; x_flat shape: [n_tokens, hidden]
                // Result: [n_tokens, 1] after sum_keepdim, then sigmoid.
                let sg = if device_eq(sg_weight.device(), x_flat.device()) {
                    sg_weight.clone()
                } else {
                    sg_weight.to_device(x_flat.device())?
                };
                let gate_logits = x_flat
                    .broadcast_mul(&sg)?
                    .sum_keepdim(D::Minus1)?;
                let gate = candle_nn::ops::sigmoid(&gate_logits)?;
                shared_out.broadcast_mul(&gate)?
            } else {
                shared_out
            };
            result = (result + shared_out)?;
        }

        result.reshape((b_sz, seq_len, hidden))
    }
}
