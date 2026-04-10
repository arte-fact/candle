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
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            MlpActivation::Silu => candle_nn::ops::silu(x),
            MlpActivation::Gelu => x.gelu(),
        }
    }
}

// ---------------------------------------------------------------------------
// DenseMlp
// ---------------------------------------------------------------------------

/// Standard gated feed-forward: out = down(activation(gate(x)) * up(x))
/// The activation is configurable (SiLU for Llama/Qwen, GELU for Gemma).
pub struct DenseMlp {
    gate: QMatMul,
    up: QMatMul,
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
        let gate = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self { gate, up, down, activation })
    }
}

impl Module for DenseMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.activation.apply(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
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

        let router = gg.qmatmul(&format!("{prefix}.ffn_gate_inp.weight"))?;
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

        // Detect shared expert
        let shared = if gg.has_tensor(&format!("{prefix}.ffn_up_shexp.weight")) {
            let shared_gate_w = gg.qmatmul(&format!("{prefix}.ffn_gate_shexp.weight"))?;
            let shared_up = gg.qmatmul(&format!("{prefix}.ffn_up_shexp.weight"))?;
            let shared_down = gg.qmatmul(&format!("{prefix}.ffn_down_shexp.weight"))?;
            Some(DenseMlp {
                gate: shared_gate_w,
                up: shared_up,
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
            ids.to_device(&device)?
        } else {
            routing_weights
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()?
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
            let activated = (candle_nn::ops::silu(&gate_out)? * up_out)?;
            self.down_exps.indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
        } else if let Some(ref gate_up_exps) = self.gate_up_exps {
            // Fused gate+up experts
            let gate_up = gate_up_exps.indexed_moe_forward(&x_flat_3d, &topk_ids)?;
            let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            let activated = (candle_nn::ops::silu(&gate)? * up)?;
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
