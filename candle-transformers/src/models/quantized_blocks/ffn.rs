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

/// T1.2: try the fused softmax+topk+normalise GPU kernel.
/// Returns `(topk_weights, topk_ids)` both shape `(rows, n_used)` on
/// HIP, or `None` if pre-conditions don't hold (non-HIP, non-f32, non-
/// contiguous input, n_experts not in {64, 128, 512}, or env-disabled).
///
/// `routing_logits` is `(rows, n_experts)`. EPLB observation is skipped
/// on the fast path (would defeat the purpose); EPLB users should set
/// `CANDLE_GEMMA4_TOPK_FUSED_OFF=1` to fall back to the legacy CPU
/// arg_sort chain that observes ids while they're on CPU.
#[cfg(feature = "hip")]
fn try_topk_fused(
    routing_logits: &Tensor,
    n_experts_used: usize,
) -> Option<(Tensor, Tensor)> {
    if std::env::var("CANDLE_GEMMA4_TOPK_FUSED_OFF").is_ok() {
        return None;
    }
    if !matches!(routing_logits.device(), candle::Device::Hip(_))
        || routing_logits.dtype() != candle::DType::F32
        || !routing_logits.is_contiguous()
    {
        return None;
    }
    let n_experts_dim = routing_logits.dim(D::Minus1).ok()?;
    // Only 64 / 128 / 256 / 512 are compiled; reject other counts.
    if !matches!(n_experts_dim, 64 | 128 | 256 | 512) {
        return None;
    }
    let flat = routing_logits.reshape(((), n_experts_dim)).ok()?;
    candle::hip_backend::topk_moe_norm_fused(&flat, n_experts_used).ok()
}

#[cfg(not(feature = "hip"))]
fn try_topk_fused(_: &Tensor, _: usize) -> Option<(Tensor, Tensor)> {
    None
}

/// Compare two devices for equality without allocating.
/// Load the shared-expert MLP and shared-gate weight for a MoE layer.
/// Shared across `Self::load` and `Self::load_ep` — both replicate the
/// shared expert on every rank.
fn load_shared_expert(
    gg: &Gguf,
    prefix: &str,
) -> Result<(Option<DenseMlp>, Option<Tensor>)> {
    let shared = if gg.has_tensor(&format!("{prefix}.ffn_up_shexp.weight")) {
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
            #[cfg(feature = "hip")]
            tp_ctx: None,
        })
    } else {
        None
    };
    // Shared-gate weight (qwen35moe convention): 1-D `[hidden]` vector.
    let shared_gate_weight = gg
        .try_dequantize(&format!("{prefix}.ffn_gate_inp_shexp.weight"));
    Ok((shared, shared_gate_weight))
}

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
    /// When `Some`, the down matmul output is treated as a PARTIAL
    /// `[B, S, hidden]` and all-reduced across ranks via the barrier
    /// before returning from `forward`. Set via `set_tp_context` after
    /// `load_tp`.
    #[cfg(feature = "hip")]
    tp_ctx: Option<super::attention::TpContext>,
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
        Ok(Self {
            gate_up,
            intermediate_size,
            down,
            activation,
            #[cfg(feature = "hip")]
            tp_ctx: None,
        })
    }

    /// TP-aware load: range-reads the fused gate+up matmul by row-split
    /// and the down matmul by column-split, so each rank holds only
    /// `intermediate_size / world_size` slice of the dense FFN.
    ///
    /// Partition (contiguous-block, rank `r` of `world`):
    /// - `ffn_gate.weight` shape `[intermediate, hidden]` — rows
    ///   `[r * I/W, (r+1) * I/W)`.
    /// - `ffn_up.weight` same.
    /// - Fused gate+up QTensor shape per rank: `[2 * I/W, hidden]`.
    /// - `ffn_down.weight` shape `[hidden, intermediate]` — cols
    ///   `[r * I/W, (r+1) * I/W)`.
    /// - Forward output per rank is a partial `[B, S, hidden]` that
    ///   caller must all-reduce across ranks to recover the full
    ///   hidden-dim sum.
    ///
    /// Requires `intermediate_size % world_size == 0` and additionally
    /// `intermediate_size / world_size` divisible by the GGML block
    /// size of `ffn_down.weight` (so column-range stays block-aligned).
    ///
    /// Peak VRAM = (gate+up+down bytes) / world_size. Uses the X5
    /// range readers end-to-end.
    #[cfg(feature = "hip")]
    pub fn load_tp(
        gg: &Gguf,
        prefix: &str,
        activation: MlpActivation,
        cluster: &std::sync::Arc<candle::hip_backend::cluster::HipCluster>,
        rank: usize,
    ) -> Result<Self> {
        let world = cluster.world_size();
        let gate_name = format!("{prefix}.ffn_gate.weight");
        let up_name = format!("{prefix}.ffn_up.weight");
        let down_name = format!("{prefix}.ffn_down.weight");

        let gate_info = gg
            .ct
            .tensor_infos
            .get(&gate_name)
            .ok_or_else(|| candle::Error::Msg(format!("missing {gate_name}")))?;
        let gate_dims = gate_info.shape.dims();
        if gate_dims.len() != 2 {
            candle::bail!(
                "DenseMlp::load_tp({prefix}): {gate_name} expected 2D, got {:?}",
                gate_dims
            );
        }
        // Uneven-shard partitioning. `intermediate` may not be divisible
        // by `world * block_size` (e.g. gemma4-26B-A4B has
        // intermediate=2112 which fails at W=4, block=32: 66 blocks /
        // 4 ranks = 16 base + 2 extra). Distribute the extra blocks
        // across the first `extra` ranks so every rank's shard stays
        // block-aligned.
        let intermediate = gate_dims[0];
        let down_info = gg
            .ct
            .tensor_infos
            .get(&down_name)
            .ok_or_else(|| candle::Error::Msg(format!("missing {down_name}")))?;
        let down_bs = down_info.ggml_dtype.block_size();
        // Use down's block_size as the partition granularity — gate/up
        // are typically the same dtype and the column-split on down is
        // the tighter alignment constraint.
        if intermediate % down_bs != 0 {
            candle::bail!(
                "DenseMlp::load_tp({prefix}): intermediate={intermediate} not divisible by {down_name} block_size={down_bs}"
            );
        }
        let total_blocks = intermediate / down_bs;
        let base_blocks = total_blocks / world;
        let extra = total_blocks % world;
        // Ranks [0, extra) get base_blocks+1; ranks [extra, world) get base_blocks.
        let local_blocks = if rank < extra { base_blocks + 1 } else { base_blocks };
        let prev_extra = rank.min(extra);
        let start_blocks = prev_extra * (base_blocks + 1) + (rank - prev_extra) * base_blocks;
        let local_intermediate = local_blocks * down_bs;
        let row_start = start_blocks * down_bs;

        if local_intermediate == 0 {
            candle::bail!(
                "DenseMlp::load_tp({prefix}): rank {rank} has zero local_intermediate (world={world} > blocks={total_blocks})"
            );
        }

        // Gate+up fused: per-rank rows concatenated for gate then up.
        let gate_up = gg.qmatmul_concat_row_range(
            &[&gate_name, &up_name],
            row_start,
            local_intermediate,
        )?;

        // Down: column-split using the same per-rank ranges.
        let col_start = row_start;
        let down_qt = gg.tensor_column_range(&down_name, col_start, local_intermediate)?;
        let down = crate::models::with_tracing::QMatMul::from_weights(
            std::sync::Arc::new(down_qt).into(),
        )?;

        let _ = cluster;
        Ok(Self {
            gate_up,
            intermediate_size: local_intermediate,
            down,
            activation,
            tp_ctx: None,
        })
    }

    /// Attach TP context so `forward` all-reduces the down-proj output
    /// across ranks before returning. Call after `load_tp`.
    #[cfg(feature = "hip")]
    pub fn set_tp_context(&mut self, ctx: super::attention::TpContext) {
        self.tp_ctx = Some(ctx);
    }

    /// Diagnostic: packed-storage bytes for gate_up + down (sums).
    pub fn q_weight_bytes_total(&self) -> usize {
        self.gate_up.storage_size_in_bytes().unwrap_or(0)
            + self.down.storage_size_in_bytes().unwrap_or(0)
    }

    /// Accessor for the per-side intermediate width.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
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

        // Gate+up split → activation. Both SiLU and GELU have fused
        // kernels that read `gate_up` directly (W5 wired the GELU
        // variant for Gemma-family). This eliminates two
        // `.narrow().contiguous()` (ucopy_f32) launches plus the
        // activation and the bmul — one fused kernel replaces four.
        // Saves ~3 launches per FFN per layer per token.
        let activated = if gate_up.is_contiguous()
            && gate_up.dtype() == candle::DType::F32
        {
            match self.activation {
                MlpActivation::Silu => {
                    candle::hip_backend::silu_mul_split_last_fused(&gate_up)?
                }
                MlpActivation::Gelu => {
                    candle::hip_backend::gelu_mul_split_last_fused(&gate_up)?
                }
            }
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
        let partial = self.down.forward_preq8(&h_q8_view, b_size, &down_rhs_shape)?;
        // TP: under tp_ctx, `down` is column-split so `partial` is the
        // rank-local partial sum — fire the cross-rank all-reduce before
        // returning, mirroring the Module::forward path.
        #[cfg(feature = "hip")]
        if let Some(ref ctx) = self.tp_ctx {
            return super::attention::all_reduce_sum_tensor(&ctx.cluster, ctx.rank, &partial);
        }
        Ok(partial)
    }
}

impl Module for DenseMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Single fused gate+up matmul. Output: (B, L, 2*intermediate)
        let gate_up = self.gate_up.forward(x)?;

        // P3 / W5: on HIP, dispatch directly to a fused
        // `{silu,gelu}_mul_split_last_fused` kernel that reads the
        // fused `gate_up` buffer in place — no `.narrow().contiguous()`
        // materialisations, no intermediate gate/up f32 tensors.
        // Drops 2 ucopy_f32 launches per FFN per layer per token.
        // Silu covers Qwen3.5; Gelu (W5) covers Gemma-family dense
        // layers.
        let partial = {
            #[cfg(feature = "hip")]
            {
                if matches!(gate_up.device(), candle::Device::Hip(_))
                    && gate_up.dtype() == candle::DType::F32
                    && gate_up.is_contiguous()
                {
                    let activated = match self.activation {
                        MlpActivation::Silu => Some(
                            candle::hip_backend::silu_mul_split_last_fused(&gate_up)?,
                        ),
                        MlpActivation::Gelu => Some(
                            candle::hip_backend::gelu_mul_split_last_fused(&gate_up)?,
                        ),
                    };
                    if let Some(activated) = activated {
                        self.down.forward(&activated)?
                    } else {
                        let gate = gate_up
                            .narrow(D::Minus1, 0, self.intermediate_size)?
                            .contiguous()?;
                        let up = gate_up
                            .narrow(
                                D::Minus1,
                                self.intermediate_size,
                                self.intermediate_size,
                            )?
                            .contiguous()?;
                        let activated = self.activation.apply_with_mul(&gate, &up)?;
                        self.down.forward(&activated)?
                    }
                } else {
                    let gate = gate_up
                        .narrow(D::Minus1, 0, self.intermediate_size)?
                        .contiguous()?;
                    let up = gate_up
                        .narrow(
                            D::Minus1,
                            self.intermediate_size,
                            self.intermediate_size,
                        )?
                        .contiguous()?;
                    let activated = self.activation.apply_with_mul(&gate, &up)?;
                    self.down.forward(&activated)?
                }
            }
            #[cfg(not(feature = "hip"))]
            {
                let gate = gate_up
                    .narrow(D::Minus1, 0, self.intermediate_size)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(
                        D::Minus1,
                        self.intermediate_size,
                        self.intermediate_size,
                    )?
                    .contiguous()?;
                let activated = self.activation.apply_with_mul(&gate, &up)?;
                self.down.forward(&activated)?
            }
        };

        #[cfg(feature = "hip")]
        if let Some(ref ctx) = self.tp_ctx {
            return super::attention::all_reduce_sum_tensor(&ctx.cluster, ctx.rank, &partial);
        }
        Ok(partial)
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
    /// Expert-parallel partition (present when EP is active). Contains
    /// the rank mapping + local expert range on this rank. Forward
    /// passes with `ep_partition.is_some()` should go through
    /// `forward_ep` instead of the default path.
    ///
    /// State in this session: the field + load-time partition metadata
    /// are wired, but `forward_ep` is still a fallback to the full
    /// (replicated) forward — turning on EP reduces NOTHING yet. Filling
    /// in the real gather/scatter is the next phase (X3 core).
    #[cfg(feature = "hip")]
    pub(crate) ep_partition: Option<EpPartition>,
}

impl MoeExperts {
    /// Diagnostic accessor: the down-projection expert weight tensor.
    /// Exposed (post-X5) so external tooling — VRAM parity tests and
    /// benches — can inspect per-rank shard sizes without poking at
    /// private fields.
    pub fn down_exps(&self) -> &std::sync::Arc<QTensor> {
        &self.down_exps
    }

    /// EP partition metadata, if this MoE branch was constructed via
    /// `load_ep`. Returns `None` when EP is not attached.
    #[cfg(feature = "hip")]
    pub fn ep_partition(&self) -> Option<&EpPartition> {
        self.ep_partition.as_ref()
    }
}

/// Expert-parallel partition state for one rank.
#[cfg(feature = "hip")]
#[derive(Debug, Clone)]
pub struct EpPartition {
    /// Cluster this rank participates in.
    pub cluster: std::sync::Arc<candle::hip_backend::cluster::HipCluster>,
    /// This rank's index within the cluster.
    pub rank: usize,
    /// Contiguous range of experts owned by this rank:
    /// `[local_start, local_start + local_count)`.
    pub local_start: usize,
    pub local_count: usize,
    /// Total experts across the cluster.
    pub num_experts: usize,
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

        let (shared, shared_gate_weight) = load_shared_expert(gg, prefix)?;

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
            #[cfg(feature = "hip")]
            ep_partition: None,
        })
    }

    /// EP-aware load: reads ONLY experts `[local_start, local_start +
    /// local_count)` directly from the mmap'd GGUF blob (no full-tensor
    /// materialisation, no 2× VRAM spike). Router, shared expert, and
    /// shared gate are loaded in full on every rank (replicated).
    ///
    /// Partition policy v1: contiguous block. Rank r owns experts
    /// `[r * block, min((r+1) * block, num_experts))` where
    /// `block = ceil(num_experts / world_size)`.
    ///
    /// Peak VRAM = steady-state: each rank allocates exactly its shard's
    /// worth of expert bytes, nothing more. This is what unlocks
    /// OOM-class MoE models (DeepSeek-V3-Lite, Kimi-K2.5) that cannot
    /// fit the full expert tensor on any single MI50.
    #[cfg(feature = "hip")]
    pub fn load_ep(
        gg: &Gguf,
        prefix: &str,
        cfg: &GgufConfig,
        cluster: std::sync::Arc<candle::hip_backend::cluster::HipCluster>,
        rank: usize,
    ) -> Result<Self> {
        let num_experts_per_tok = cfg.expert_used_count.unwrap_or(1);

        // --- Router (replicated across ranks). Same path as Self::load. ---
        let mut router = gg.qmatmul(&format!("{prefix}.ffn_gate_inp.weight"))?;
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

        // --- Determine partition by peeking at the down_exps shape. ---
        let down_exps_name = format!("{prefix}.ffn_down_exps.weight");
        let down_info = gg
            .ct
            .tensor_infos
            .get(&down_exps_name)
            .ok_or_else(|| candle::Error::Msg(format!("missing {down_exps_name}")))?;
        let num_experts = down_info.shape.dims()[0];
        let world = cluster.world_size();
        let block = (num_experts + world - 1) / world;
        let local_start = rank * block;
        let local_count = num_experts.saturating_sub(local_start).min(block);

        // Intermediate size derived from the FULL tensor shape (invariant
        // across ranks); range-reading doesn't change dims[1].
        let down_shape_full = down_info.shape.dims().to_vec();
        let intermediate_size = if down_shape_full.len() == 3 {
            down_shape_full[1]
        } else {
            cfg.feed_forward_length.unwrap_or(0)
        };

        if local_count == 0 {
            // Rank past the end (W > num_experts). Build with empty experts:
            // forward_ep short-circuits on local_count==0.
            //
            // We still need SHAPE-valid expert tensors so broadcasting
            // doesn't panic; read an empty range (byte_len=0) → QTensor
            // with shape [0, ...].
            let down_exps = std::sync::Arc::new(gg.tensor_expert_range(&down_exps_name, num_experts, 0)?);
            let (gate_exps, up_exps, gate_up_exps) = if gg.has_tensor(&format!("{prefix}.ffn_gate_exps.weight")) {
                let g = gg.tensor_expert_range(&format!("{prefix}.ffn_gate_exps.weight"), num_experts, 0)?;
                let u = gg.tensor_expert_range(&format!("{prefix}.ffn_up_exps.weight"), num_experts, 0)?;
                (Some(std::sync::Arc::new(g)), Some(std::sync::Arc::new(u)), None)
            } else {
                let gu = gg.tensor_expert_range(&format!("{prefix}.ffn_gate_up_exps.weight"), num_experts, 0)?;
                (None, None, Some(std::sync::Arc::new(gu)))
            };

            let (shared, shared_gate_weight) = load_shared_expert(gg, prefix)?;

            return Ok(Self {
                router,
                up_exps, gate_exps, gate_up_exps, down_exps,
                num_experts_per_tok, intermediate_size,
                shared, shared_gate_weight,
                ep_partition: Some(EpPartition {
                    cluster, rank, local_start, local_count, num_experts,
                }),
            });
        }

        // --- Range-read expert tensors (this rank's shard only). ---
        let down_exps = std::sync::Arc::new(
            gg.tensor_expert_range(&down_exps_name, local_start, local_count)?,
        );
        let (gate_exps, up_exps, gate_up_exps) = if gg.has_tensor(&format!("{prefix}.ffn_gate_exps.weight")) {
            let g = gg.tensor_expert_range(
                &format!("{prefix}.ffn_gate_exps.weight"),
                local_start, local_count,
            )?;
            let u = gg.tensor_expert_range(
                &format!("{prefix}.ffn_up_exps.weight"),
                local_start, local_count,
            )?;
            (Some(std::sync::Arc::new(g)), Some(std::sync::Arc::new(u)), None)
        } else if gg.has_tensor(&format!("{prefix}.ffn_gate_up_exps.weight")) {
            let gu = gg.tensor_expert_range(
                &format!("{prefix}.ffn_gate_up_exps.weight"),
                local_start, local_count,
            )?;
            (None, None, Some(std::sync::Arc::new(gu)))
        } else {
            candle::bail!("MoE layer at {prefix} has neither gate_exps nor gate_up_exps");
        };

        // --- Shared expert + gate (replicated across ranks). ---
        let (shared, shared_gate_weight) = load_shared_expert(gg, prefix)?;

        Ok(Self {
            router,
            up_exps, gate_exps, gate_up_exps, down_exps,
            num_experts_per_tok, intermediate_size,
            shared, shared_gate_weight,
            ep_partition: Some(EpPartition {
                cluster, rank, local_start, local_count, num_experts,
            }),
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
        // T1.2 fast path: fused softmax+topk+norm on GPU.  Falls back
        // to the legacy CPU arg_sort chain when not applicable.
        //
        // W3 step 4: only sync topk_ids to CPU when W3 gather is on
        // (opt-in via CANDLE_W3_GATHER=1). The cache is reset at end
        // of the MoE layer so the next layer doesn't pick up stale
        // ids.
        #[cfg(feature = "hip")]
        let w3_gather_on = std::env::var("CANDLE_W3_GATHER")
            .map_or(false, |v| v != "0");

        let (topk_weights, topk_ids) = if let Some((w, ids)) =
            try_topk_fused(&router_logits, self.num_experts_per_tok)
        {
            #[cfg(feature = "hip")]
            {
                if w3_gather_on {
                    let ids_cpu: Vec<u32> =
                        ids.flatten_all()?.to_vec1::<u32>()?;
                    candle::quantized::hip::w3_cached_ids_set(ids_cpu);
                }
            }
            (w, ids)
        } else {
            let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
            let cpu = candle::Device::Cpu;
            let rw_cpu = routing_weights.to_device(&cpu)?;
            let topk_ids_cpu_tensor = rw_cpu
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()?;
            #[cfg(feature = "hip")]
            {
                if w3_gather_on {
                    let ids_cpu: Vec<u32> = topk_ids_cpu_tensor
                        .flatten_all()?
                        .to_vec1::<u32>()?;
                    candle::quantized::hip::w3_cached_ids_set(ids_cpu);
                }
            }
            let hip_dev = candle::Device::Hip(dev.clone());
            let topk_ids = topk_ids_cpu_tensor.to_device(&hip_dev)?;
            let topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;
            let topk_weights = (&topk_weights
                / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as(topk_weights.shape())?)?;
            (topk_weights, topk_ids)
        };

        // The MoE preq8 kernel expects `[batch, topk_or_1, k]` `in_shape`
        // for its block-grid sizing.  `topk_or_1 == 1` here because the
        // input is shared across all top-k experts.
        let in_shape = candle::Shape::from((b_size, 1usize, hidden));

        let moe_out = if let Some(ref gate_exps) = self.gate_exps {
            let up_exps = self.up_exps.as_ref().unwrap();
            // P21: try the fused gate+up kernel first (one launch for both
            // outputs instead of two). Falls through to sequential calls
            // when dtype/shape isn't supported.
            let (gate_out, up_out) = match gate_exps
                .indexed_moe_gate_up_forward_preq8(up_exps, &q8_view, &in_shape, &topk_ids)?
            {
                Some((g, u)) => (g, u),
                None => {
                    let g = gate_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids)?;
                    let u = up_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids)?;
                    (g, u)
                }
            };
            // P25: try fused silu_mul + quantize_q8_1 → down preq8
            // (single kernel instead of silu_mul + contiguous + internal
            // requantise inside indexed_moe_forward).  Falls back to the
            // legacy 3-kernel path if the gate/up shapes / dtypes don't
            // match what the kernel expects, or env opt-out.
            #[cfg(feature = "hip")]
            let try_p25 = matches!(gate_out.device(), candle::Device::Hip(_))
                && gate_out.dtype() == candle::DType::F32
                && up_out.dtype() == candle::DType::F32
                && gate_out.rank() == 3
                && self.intermediate_size % 32 == 0
                && std::env::var("CANDLE_MOE_DOWN_FUSED_SILU").as_deref() != Ok("0");
            #[cfg(not(feature = "hip"))]
            let try_p25 = false;
            let used_p25 = if try_p25 {
                #[cfg(feature = "hip")]
                {
                    let g_c = gate_out.contiguous()?;
                    let u_c = up_out.contiguous()?;
                    match candle::hip_backend::silu_mul_quantize_q8_1_separate_fused(&g_c, &u_c) {
                        Ok((q8_buf, q8_shape)) => {
                            let q8_view = q8_buf.slice(0..q8_buf.len());
                            Some(self.down_exps.indexed_moe_forward_preq8(
                                &q8_view, &q8_shape, &topk_ids,
                            )?)
                        }
                        Err(_) => None,
                    }
                }
                #[cfg(not(feature = "hip"))]
                None
            } else {
                None
            };
            match used_p25 {
                Some(out) => out,
                None => {
                    let activated = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
                    self.down_exps
                        .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
                }
            }
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

        // End-of-layer cleanup: drop the stashed ids so the next MoE
        // layer doesn't pick up stale values (its own ids will be set
        // on entry).
        #[cfg(feature = "hip")]
        candle::quantized::hip::w3_cached_ids_clear();

        // W3a (C1): prefer fused `moe_combine_weighted_sum_fused` on HIP —
        // replaces broadcast_mul + sum(1) with a single kernel launch.
        // Kill-switch CANDLE_W3A_COMBINE=0 restores the unfused path.
        let mut result = {
            #[cfg(feature = "hip")]
            let try_fused =
                matches!(moe_out.device(), candle::Device::Hip(_))
                && moe_out.dtype() == candle::DType::F32
                && topk_weights.dtype() == candle::DType::F32
                && moe_out.rank() == 3
                && topk_weights.rank() == 2
                && std::env::var("CANDLE_W3A_COMBINE").as_deref() != Ok("0");
            #[cfg(not(feature = "hip"))]
            let try_fused = false;
            if try_fused {
                #[cfg(feature = "hip")]
                {
                    let moe_out_c = moe_out.contiguous()?;
                    let topk_w_c = topk_weights.contiguous()?;
                    candle::hip_backend::moe_combine_weighted_sum_fused(&moe_out_c, &topk_w_c)?
                }
                #[cfg(not(feature = "hip"))]
                unreachable!()
            } else {
                let topk_weights_u = topk_weights.unsqueeze(D::Minus1)?;
                let weighted = moe_out.broadcast_mul(&topk_weights_u)?;
                weighted.sum(1)?
            }
        };

        if let Some(ref shared_mlp) = self.shared {
            let shared_out = shared_mlp.forward(&x_flat)?;
            let shared_out = if let Some(ref sg_weight) = self.shared_gate_weight {
                let sg = if device_eq(sg_weight.device(), x_flat.device()) {
                    sg_weight.clone()
                } else {
                    sg_weight.to_device(x_flat.device())?
                };
                // P28: fused dot replaces broadcast_mul + sum_keepdim (2
                // kernels → 1) when both are on HIP and f32.
                #[cfg(feature = "hip")]
                let gate_logits = if matches!(x_flat.device(), candle::Device::Hip(_))
                    && x_flat.dtype() == candle::DType::F32
                    && sg.dtype() == candle::DType::F32
                    && x_flat.is_contiguous()
                    && sg.is_contiguous()
                    && sg.rank() == 1
                    && std::env::var("CANDLE_DOT_LAST_DIM_FUSED").as_deref() == Ok("1")
                {
                    candle::hip_backend::dot_last_dim_fused(&x_flat, &sg)?
                } else {
                    x_flat.broadcast_mul(&sg)?.sum_keepdim(D::Minus1)?
                };
                #[cfg(not(feature = "hip"))]
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
        // EP entry-point: when a partition is attached, forward_ep owns
        // the compute (broadcast → local experts → all-reduce).
        // v1 stub: delegate to the normal forward so output is correct
        // while the real gather/scatter wiring lands in X3.5.
        #[cfg(feature = "hip")]
        if self.ep_partition.is_some() {
            return self.forward_ep(x);
        }

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

    /// Expert-parallel forward. Returns THIS RANK's partial MoE output:
    /// `sum_k(topk_weight_k * local_expert_k(x))` where the sum is
    /// restricted to expert IDs inside `[local_start, local_start + local_count)`.
    /// Out-of-range experts contribute zero by construction (their
    /// topk_weight is zeroed before the weighted sum).
    ///
    /// The caller is responsible for summing partial outputs across
    /// ranks (all-reduce / broadcast+add) to recover the full MoE
    /// output. Shared expert + shared_gate are applied on rank 0
    /// only — adding them to the partial sum gives the correct final
    /// output when all ranks' partials are summed.
    ///
    /// Implementation notes:
    /// - Every rank computes the router + topk independently (router
    ///   weights are replicated across ranks). This is cheap and
    ///   avoids a synchronisation point.
    /// - topk_ids are remapped to local indices: global_id - local_start
    ///   if in range, else 0 (safe dummy; corresponding weight = 0).
    /// - Uses the same Q8_1 shared-input fast path as `forward_shared_q8_decode`.
    ///
    /// `forward_ep` returns the PARTIAL MoE output (this rank's
    /// contribution). For multi-threaded TP+EP forward the caller wants
    /// the FULL (summed) output on every rank — see
    /// [`Self::forward_ep_reduced`] which wraps `forward_ep` with the
    /// barrier all-reduce.
    #[cfg(feature = "hip")]
    pub fn forward_ep_reduced(&self, x: &Tensor) -> Result<Tensor> {
        let partial = self.forward_ep(x)?;
        if let Some(ref ep) = self.ep_partition {
            super::attention::all_reduce_sum_tensor(&ep.cluster, ep.rank, &partial)
        } else {
            Ok(partial)
        }
    }

    #[cfg(feature = "hip")]
    pub fn forward_ep(&self, x: &Tensor) -> Result<Tensor> {
        use candle::quantized::hip::{pad, quantize_q8_1, MATRIX_ROW_PADDING};
        use candle::quantized::GgmlDType;
        use candle::D;

        let partition = self
            .ep_partition
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("forward_ep called without ep_partition".into()))?;
        if partition.local_count == 0 {
            // Rank holds no experts (world > num_experts). Contribute zero.
            return Tensor::zeros(x.shape(), x.dtype(), x.device());
        }

        let (b_sz, seq_len, hidden) = x.dims3()?;
        let b_size = b_sz * seq_len;

        // Fallback: if the input isn't in the decode-fast-path shape,
        // drop to the slow forward on the narrowed weights. The
        // narrowed weights hold only local experts, so
        // `indexed_moe_forward` with topk_ids containing global IDs
        // would go out-of-range — slow path needs the same remap.
        // For now, return an error when fast path doesn't apply.
        if b_size > 8
            || !matches!(x.device(), candle::Device::Hip(_))
            || x.dtype() != candle::DType::F32
            || !x.is_contiguous()
        {
            candle::bail!(
                "forward_ep: only decode-path (b*m <= 8, HIP, F32, contiguous) is supported in v1"
            );
        }

        let dev = match x.device() {
            candle::Device::Hip(d) => d.clone(),
            _ => candle::bail!("forward_ep: expected HIP device"),
        };

        // Shared Q8_1 input (same as forward_shared_q8_decode).
        let x_flat = x.reshape((b_size, hidden))?;
        let (x_st, x_l) = x_flat.storage_and_layout();
        let x_hip = match &*x_st {
            candle::Storage::Hip(s) => s,
            _ => candle::bail!("forward_ep: x must be HipStorage"),
        };
        let x_slice = x_hip.as_hip_slice::<f32>()?;
        let x_view = match x_l.contiguous_offsets() {
            Some((lo, hi)) => x_slice.slice(lo..hi),
            None => candle::bail!("forward_ep: x layout not contiguous"),
        };
        let ncols_padded = pad(hidden, MATRIX_ROW_PADDING);
        let q8_bytes = b_size * ncols_padded * GgmlDType::Q8_1.type_size()
            / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(q8_bytes)? };
        quantize_q8_1(&x_view, &mut y_q8_1, hidden, b_size, &dev)?;
        let q8_view = y_q8_1.slice(0..y_q8_1.len());
        let rhs_shape = x_flat.dims().to_vec();
        drop(x_st);

        // Router + topk (global ids).
        let router_logits = if self.router.is_qtensor() {
            self.router.forward_preq8(&q8_view, b_size, &rhs_shape)?
        } else {
            self.router.forward(&x_flat)?
        };
        let (topk_weights, topk_ids_global) = if let Some((w, ids)) =
            try_topk_fused(&router_logits, self.num_experts_per_tok)
        {
            (w, ids)
        } else {
            let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
            let cpu = candle::Device::Cpu;
            let rw_cpu = routing_weights.to_device(&cpu)?;
            let topk_ids_cpu_tensor = rw_cpu
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                .contiguous()?;
            let hip_dev = candle::Device::Hip(dev.clone());
            let topk_ids = topk_ids_cpu_tensor.to_device(&hip_dev)?;
            let topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;
            let topk_weights = (&topk_weights
                / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as(topk_weights.shape())?)?;
            (topk_weights, topk_ids)
        };

        // EP remap: for each topk slot, if the global id is in our
        // local range, map to local index; else map to 0 and zero its
        // weight. Done on CPU since topk_ids is tiny (b_size * k).
        let ids_cpu: Vec<u32> = topk_ids_global
            .to_device(&candle::Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let local_start = partition.local_start as u32;
        let local_count = partition.local_count as u32;
        let mut local_ids_cpu = Vec::with_capacity(ids_cpu.len());
        let mut weight_mask_cpu = Vec::with_capacity(ids_cpu.len());
        for id in &ids_cpu {
            if *id >= local_start && *id < local_start + local_count {
                local_ids_cpu.push(*id - local_start);
                weight_mask_cpu.push(1.0f32);
            } else {
                local_ids_cpu.push(0);
                weight_mask_cpu.push(0.0f32);
            }
        }
        let hip_dev = candle::Device::Hip(dev.clone());
        let topk_ids_local = Tensor::from_vec(local_ids_cpu, topk_ids_global.shape().dims(), &hip_dev)?;
        let weight_mask = Tensor::from_vec(
            weight_mask_cpu,
            topk_weights.shape().dims(),
            &hip_dev,
        )?;
        // Zero out-of-range weights element-wise.
        let topk_weights_masked = topk_weights.mul(&weight_mask)?;

        // Run indexed_moe_forward on narrowed expert tensors with local ids.
        let in_shape = candle::Shape::from((b_size, 1usize, hidden));
        let moe_out = if let Some(ref gate_exps) = self.gate_exps {
            let up_exps = self.up_exps.as_ref().unwrap();
            let gate_out =
                gate_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids_local)?;
            let up_out =
                up_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids_local)?;
            let activated = candle_nn::ops::silu_mul(&gate_out, &up_out)?;
            self.down_exps
                .indexed_moe_forward(&activated.contiguous()?, &topk_ids_local)?
        } else if let Some(ref gate_up_exps) = self.gate_up_exps {
            let gate_up =
                gate_up_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids_local)?;
            let gate = gate_up
                .narrow(D::Minus1, 0, self.intermediate_size)?
                .contiguous()?;
            let up = gate_up
                .narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?
                .contiguous()?;
            let activated = candle_nn::ops::silu_mul(&gate, &up)?;
            self.down_exps
                .indexed_moe_forward(&activated.contiguous()?, &topk_ids_local)?
        } else {
            candle::bail!("forward_ep: MoE has no expert weights");
        };

        // Weighted sum with masked weights — out-of-range slots
        // contribute zero.
        let topk_weights_masked = topk_weights_masked.unsqueeze(D::Minus1)?;
        let weighted = moe_out.broadcast_mul(&topk_weights_masked)?;
        let mut result = weighted.sum(1)?;

        // Shared expert: added on rank 0 ONLY. When all ranks' partial
        // outputs are summed, the shared expert appears exactly once.
        if partition.rank == 0 {
            if let Some(ref shared_mlp) = self.shared {
                let shared_out = shared_mlp.forward(&x_flat)?;
                let shared_out = if let Some(ref sg_weight) = self.shared_gate_weight {
                    let sg = if device_eq(sg_weight.device(), x_flat.device()) {
                        sg_weight.clone()
                    } else {
                        sg_weight.to_device(x_flat.device())?
                    };
                    #[cfg(feature = "hip")]
                    let gate_logits = if matches!(x_flat.device(), candle::Device::Hip(_))
                        && x_flat.dtype() == candle::DType::F32
                        && sg.dtype() == candle::DType::F32
                        && x_flat.is_contiguous()
                        && sg.is_contiguous()
                        && sg.rank() == 1
                        && std::env::var("CANDLE_DOT_LAST_DIM_FUSED").as_deref() == Ok("1")
                    {
                        candle::hip_backend::dot_last_dim_fused(&x_flat, &sg)?
                    } else {
                        x_flat.broadcast_mul(&sg)?.sum_keepdim(D::Minus1)?
                    };
                    #[cfg(not(feature = "hip"))]
                    let gate_logits = x_flat.broadcast_mul(&sg)?.sum_keepdim(D::Minus1)?;
                    let gate = candle_nn::ops::sigmoid(&gate_logits)?;
                    shared_out.broadcast_mul(&gate)?
                } else {
                    shared_out
                };
                result = (result + shared_out)?;
            }
        }

        Ok(result.reshape((b_sz, seq_len, hidden))?)
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden))?;

        // Route: softmax → topk
        let router_logits = self.router.forward(&x_flat)?;

        // T1.2 fast path: fused softmax+topk+norm on GPU.  Falls back
        // to the legacy softmax + CPU arg_sort + gather + norm chain
        // when not applicable (non-HIP, n_experts not in {64,128,512},
        // env-disabled).  EPLB observation is only available on the
        // legacy path; users wanting EPLB stats must set
        // `CANDLE_GEMMA4_TOPK_FUSED_OFF=1`.
        let device = router_logits.device().clone();
        let (topk_weights, topk_ids) = if let Some((w, ids)) =
            try_topk_fused(&router_logits, self.num_experts_per_tok)
        {
            (w, ids)
        } else {
            let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
            let topk_ids = if matches!(device, candle::Device::Hip(_)) {
                let cpu = candle::Device::Cpu;
                let rw_cpu = routing_weights.to_device(&cpu)?;
                let ids = rw_cpu
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, self.num_experts_per_tok)?
                    .contiguous()?;
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
            let topk_weights = (&topk_weights
                / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as(topk_weights.shape())?)?;
            (topk_weights, topk_ids)
        };

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

        // W3a (C1): fused combine on HIP — see line 926 for detail.
        let mut result = {
            #[cfg(feature = "hip")]
            let try_fused =
                matches!(moe_out.device(), candle::Device::Hip(_))
                && moe_out.dtype() == candle::DType::F32
                && topk_weights.dtype() == candle::DType::F32
                && moe_out.rank() == 3
                && topk_weights.rank() == 2
                && std::env::var("CANDLE_W3A_COMBINE").as_deref() != Ok("0");
            #[cfg(not(feature = "hip"))]
            let try_fused = false;
            if try_fused {
                #[cfg(feature = "hip")]
                {
                    let moe_out_c = moe_out.contiguous()?;
                    let topk_w_c = topk_weights.contiguous()?;
                    candle::hip_backend::moe_combine_weighted_sum_fused(&moe_out_c, &topk_w_c)?
                }
                #[cfg(not(feature = "hip"))]
                unreachable!()
            } else {
                let topk_weights_u = topk_weights.unsqueeze(D::Minus1)?;
                let weighted = moe_out.broadcast_mul(&topk_weights_u)?;
                weighted.sum(1)?
            }
        };

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
