//! Gemma 4 model implementation with quantization support.
//!
//! Built on the modular [`crate::models::quantized_blocks`] system. The
//! per-architecture file is now a thin assembler that wires shared blocks
//! (`StandardAttention`, `DenseMlp`, `RotaryEmbedding`, …) with gemma4-specific
//! features:
//!
//! - **Per-layer head_dim** (sliding=`key_length_swa`, global=`key_length`)
//! - **Per-layer RoPE** with `freq_base` (global) or `freq_base_swa` (sliding)
//! - **`rope_freqs.weight`** proportional rope for non-SWA layers
//! - **Partial RoPE** when `rope.dimension_count` < head_dim
//! - **Parameter-free V-norm** (tri-norm pattern)
//! - **`f_attention_scale = 1.0`** (no pre-attention scaling)
//! - **GeGLU** in the FFN (not SiLU like Llama/Qwen)
//! - **Final logit softcapping** (tanh-based)
//! - **Layer output scale** (per-layer learned scalar)
//! - **Per-layer embedding** for E4B variant
//! - **Shared KV layers** for the 31B variant (`n_layer_kv_from_start` < `n_layer`)
//! - **Pipeline-parallel layer split** (`from_gguf_multi_device`)
//!
//! GGUF arch string: `gemma4`.

use crate::models::quantized_blocks::*;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use crate::models::quantized_blocks::attention::gqa_attention_k_transposed;
use candle::quantized::{gguf_file, GgufBlob};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use rayon::prelude::*;
use std::sync::Arc;

/// Helper used by the H1 n_kv-padding code path: pull `seq_len` (dim 2) from
/// either the shared Q tensor or from x_norm.
fn q_precomputed_shape_or_none(
    shared: &Option<(Tensor, Tensor, Tensor)>,
    x_norm: &Tensor,
) -> Result<(usize, usize, usize, usize)> {
    match shared {
        Some((q, _, _)) => q.dims4(),
        None => {
            let (b, s, h) = x_norm.dims3()?;
            // Shape the attention eventually sees is (B, H_q, seq, D) — we
            // only need seq here (dim 2), reported as dim 2 of our tuple.
            Ok((b, 1, s, h))
        }
    }
}

fn q_precomputed_device_or_xnorm<'a>(
    shared: &'a Option<(Tensor, Tensor, Tensor)>,
    x_norm: &'a Tensor,
) -> &'a Device {
    match shared {
        Some((q, _, _)) => q.device(),
        None => x_norm.device(),
    }
}

pub const MAX_SEQ_LEN: usize = 131072;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

// ---------------------------------------------------------------------------
// Per-layer embedding (E4B-specific)
// ---------------------------------------------------------------------------

/// Per-layer embedding components (E4B-specific).
///
/// Each layer reads a slice from a global per-layer embedding tensor and
/// gates it through a learned `inp_gate → gelu → proj → post_norm` path.
struct PerLayerEmbed {
    inp_gate: super::with_tracing::QMatMul,
    proj: super::with_tracing::QMatMul,
    post_norm: RmsNorm,
}

/// Global per-layer embedding components (E4B only).
///
/// `token_embd` is the huge `per_layer_token_embd.weight` tensor (~11GB
/// dequantized for E4B). It's kept on CPU and only the looked-up rows are
/// moved to the model device per forward pass.
struct PerLayerEmbeddings {
    token_embd: Tensor,
    model_proj: super::with_tracing::QMatMul,
    proj_norm: RmsNorm,
    n_embd_per_layer: usize,
}

// ---------------------------------------------------------------------------
// LayerWeights
// ---------------------------------------------------------------------------

/// Optional MoE branch present in gemma4-A4B (26B) but absent in
/// gemma4-E4B (4B dense). When present, the FFN block is dual:
///   combined = post_norm_1(dense_ffn(norm_1(residual)))
///            + post_norm_2(moe_ffn(norm_2(residual), router(residual)))
///   out = residual + post_ffw_norm(combined)
///
/// THREE post-norms: _1 for dense, _2 for MoE, plain for combined.
/// Both FFN branches read from `residual` (the post-attention output).
struct MoeBranch {
    /// Post-norm for the dense FFN branch (post_ffw_norm_1).
    dense_post_norm: RmsNorm,
    /// Router weight matrix [n_experts, hidden_dim].
    gate_inp: QMatMul,
    /// Router input scale: learned per-element scale applied to the
    /// rms-normed input before the router matmul.
    gate_inp_s: Tensor,
    /// Fused expert gate+up [n_experts, 2*expert_ffn_dim, hidden_dim].
    gate_up_exps: std::sync::Arc<candle::quantized::QTensor>,
    /// Expert down [n_experts, hidden_dim, expert_ffn_dim].
    down_exps: std::sync::Arc<candle::quantized::QTensor>,
    /// Optional per-expert down projection scale (ffn_down_exps.scale).
    /// Applied element-wise to the down projection output before
    /// combining across topk experts.
    down_exps_scale: Option<Tensor>,
    /// Expert FFN intermediate size (per expert).
    expert_intermediate: usize,
    /// Number of experts used per token.
    num_experts_used: usize,
    /// Pre-MoE norm (pre_ffw_norm_2).
    pre_norm: RmsNorm,
    /// Post-MoE norm (post_ffw_norm_2).
    post_norm: RmsNorm,
    /// Embedding length for the 1/sqrt(n_embd) router scale.
    n_embd: usize,
    /// RMS norm epsilon (reused for the inline router rms_norm).
    eps: f32,
}

struct LayerWeights {
    attn: StandardAttention,
    attn_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: DenseMlp,
    /// Optional MoE expert branch for A4B-class models.
    moe: Option<MoeBranch>,
    /// Optional sliding-window mask radius (None = global attention).
    sliding_window_size: Option<usize>,
    /// True if this layer computes its own K/V (the first
    /// `n_layer_kv_from_start` layers). False for shared-KV layers
    /// that reuse an earlier layer's cache.
    has_kv: bool,
    /// Index of the source layer to borrow K/V cache from when `has_kv == false`.
    kv_source_idx: usize,
    /// Per-layer learned output scale (gemma4 only). When present, the residual
    /// stream after the FFN block is multiplied by this scalar.
    layer_output_scale: Option<Tensor>,
    /// Per-layer embedding components (E4B only).
    per_layer_embed: Option<PerLayerEmbed>,
    /// Device this layer's weights live on.
    device: Device,
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: super::with_tracing::QMatMul,
    final_logit_softcap: Option<f64>,
    /// Per-layer embedding components (E4B only).
    per_layer_embeddings: Option<PerLayerEmbeddings>,
    /// Per-layer KV cache. For shared-KV layers (`has_kv=false`) the entry is
    /// always `None` and the layer reads from `kv_caches[kv_source_idx]`.
    ///
    /// Uses the pre-allocated `KvCache` (slice-set into a fixed buffer)
    /// instead of `Tensor::cat`, which would otherwise allocate a new
    /// `cache_len + 1` buffer every decode step and copy the entire
    /// cache contents into it — O(N²) memory traffic on long contexts
    /// and the dominant decode cost on gemma4 (which slowed 31% on
    /// candle vs 3% on turbo at long-cache decode before this fix).
    kv_caches: Vec<Option<candle_nn::kv_cache::KvCache>>,
    /// G2/G3 decode replay state. See `quantized_llama::DecodeState`.
    /// Single-device only for now — multi-device pipeline-parallel layouts
    /// fall through to the normal forward path.
    #[cfg(feature = "hip")]
    decode_state: DecodeState,
}

/// Mirror of `quantized_llama::DecodeState` — kept private to this module
/// since it transitively references `decode_cache::{RecordedOp, DecodePlan,
/// DecodeGraph}`. See the llama path for the protocol.
#[cfg(feature = "hip")]
enum DecodeState {
    Init,
    WarmUp,
    /// Token 3: first recording captured. `input_ptrs` is the list of
    /// per-token external-input device pointers (1 element for dense
    /// Gemma4 = layer_in only; 2 elements for E4B = [layer_in,
    /// inp_per_layer]). `input_bytes` mirrors the byte size of each
    /// input; it must match between the two recordings.
    Recorded1 {
        ops: Vec<candle::hip_backend::decode_cache::RecordedOp>,
        input_ptrs: Vec<usize>,
        input_bytes: Vec<usize>,
    },
    Replay(candle::hip_backend::decode_cache::DecodePlan),
    Graph {
        plan: candle::hip_backend::decode_cache::DecodePlan,
        graph: candle::hip_backend::decode_cache::DecodeGraph,
        captured_l_k: usize,
    },
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
    /// Token embedding lives on `devices[0]`; output norm + lm_head live on
    /// the device of the last layer. Per-layer weights are loaded in
    /// parallel via rayon (mmap-backed `Gguf`).
    pub fn from_gguf_multi_device(
        ct: gguf_file::Content,
        blob: Arc<GgufBlob>,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("from_gguf_multi_device requires at least one device");
        }
        let dev0 = &devices[0];

        // ----- read all metadata up front via the shared GgufConfig --------
        let gg = Gguf::new(ct, blob, dev0.clone());
        let cfg = GgufConfig::from_metadata(gg.metadata())?;
        if cfg.arch != "gemma4" {
            candle::bail!("quantized_gemma4 expects arch=gemma4, got {}", cfg.arch);
        }
        let block_count = cfg.block_count;
        let head_count = cfg.head_count;
        let head_count_kv_default = *cfg.head_count_kv.get(0);
        let key_length = cfg.head_dim;
        let rms_norm_eps = cfg.rms_norm_eps;
        let embedding_length = cfg.hidden_size;

        // ----- gemma4-specific metadata fields not on GgufConfig -----------
        let metadata = gg.metadata();
        let md_get = |k: &str| metadata.get(&format!("gemma4.{k}"));
        let key_length_swa = md_get("attention.key_length_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(key_length);
        let sliding_window_size = md_get("attention.sliding_window")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(4096);
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_swa = md_get("rope.freq_base_swa")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);
        let rope_dim_count = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let rope_dim_count_swa = md_get("rope.dimension_count_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let final_logit_softcap = md_get("final_logit_softcapping")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let n_embd_per_layer = md_get("embedding_length_per_layer_input")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(0);

        // Per-layer kv heads (gemma4 may have an array)
        let head_count_kv_per_layer: Vec<usize> = (0..block_count)
            .map(|i| *cfg.head_count_kv.get(i))
            .collect();
        let _ = head_count_kv_default;

        // Sliding-window pattern: prefer the per-layer bool array; fall back
        // to the standard "every Nth layer is global" rule.
        let sliding_window_pattern: Option<Vec<bool>> = metadata
            .get("gemma4.attention.sliding_window_pattern")
            .and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    let mut bools = Vec::with_capacity(arr.len());
                    for item in arr {
                        match item {
                            gguf_file::Value::Bool(b) => bools.push(*b),
                            gguf_file::Value::U8(x) => bools.push(*x != 0),
                            _ => return None,
                        }
                    }
                    Some(bools)
                }
                _ => None,
            });
        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(6);

        // Shared-KV layers (31B variant): the LAST `shared_kv_layers` layers
        // reuse K/V from earlier layers via cache aliasing.
        let shared_kv_layers = md_get("attention.shared_kv_layers")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(0);
        let n_layer_kv_from_start = block_count.saturating_sub(shared_kv_layers);

        // ----- pipeline-parallel layer-to-device assignment ----------------
        let layer_to_device = split_layers_across_devices(block_count, devices.len());

        // Resolve sliding/global per layer.
        let is_sliding_per_layer: Vec<bool> = (0..block_count)
            .map(|i| {
                if let Some(ref pat) = sliding_window_pattern {
                    pat.get(i).copied().unwrap_or(false)
                } else {
                    (i + 1) % sliding_window_type > 0
                }
            })
            .collect();

        // For shared-KV layers, find the most recent same-pattern layer in
        // [0..n_layer_kv_from_start). Mirrors llama.cpp's iswa cache aliasing
        // (logged as `reuse layer N, is_swa = X`).
        let kv_source_per_layer: Vec<usize> = (0..block_count)
            .map(|il| {
                if il < n_layer_kv_from_start {
                    il
                } else {
                    let want = is_sliding_per_layer[il];
                    (0..n_layer_kv_from_start)
                        .rev()
                        .find(|&j| is_sliding_per_layer[j] == want)
                        .unwrap_or(0)
                }
            })
            .collect();

        // ----- shared rope_freqs (proportional rope for global layers) -----
        // Read from the dev0-targeted gg (no need for set_device since gg is
        // already on dev0).
        let rope_freqs: Option<Vec<f32>> = gg
            .try_dequantize("rope_freqs.weight")
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.flatten_all().ok())
            .and_then(|t| t.to_vec1::<f32>().ok());

        // ----- token embedding -------------------------------------------
        // For larger gemma4 variants (31B has hidden=5376, vocab=262144) the
        // dequantized F32 embedding is 5.6 GB and pushes dev0 over the 16 GB
        // MI50 limit. Keep the embedding on CPU and let the forward pass move
        // the looked-up rows to dev0 (a few KB per token).
        let cpu = candle::Device::Cpu;
        let cpu_gg = gg.with_device(cpu.clone());
        let tok_tensor = cpu_gg.tensor("token_embd.weight")?;
        let tok_embeddings = Embedding::new(tok_tensor.dequantize(&cpu)?, embedding_length);

        // ----- per-layer embedding global components (E4B only) ------------
        let per_layer_embeddings = if n_embd_per_layer > 0 {
            // Keep the huge per_layer_token_embd on CPU; the proj/norm live on dev0.
            let pl_embd = cpu_gg.try_dequantize("per_layer_token_embd.weight");
            let pl_proj = gg.try_qmatmul("per_layer_model_proj.weight");
            let pl_pn = gg.try_rms_norm("per_layer_proj_norm.weight", rms_norm_eps);
            // NOTE: we deliberately do NOT requantize `pl_proj` (the
            // global model_proj used in the prelude) even though it's an
            // F16 weight. Tested empirically — switching it to Q8_0 makes
            // the prelude's alloc pattern diverge from recording (extra
            // quantize_q8_1 buffer per call), which throws the
            // decode_alloc cursor out of sync with the captured plan and
            // causes `hipErrorNotReady` mid-replay. The per-layer
            // `inp_gate` / `proj` ARE requantized below — those are
            // inside the captured layer loop where the alloc pattern is
            // already part of the recording.
            match (pl_embd, pl_proj, pl_pn) {
                (Some(embd), Some(proj), Some(pn)) => Some(PerLayerEmbeddings {
                    token_embd: embd,
                    model_proj: proj,
                    proj_norm: pn,
                    n_embd_per_layer,
                }),
                _ => None,
            }
        } else {
            None
        };

        // ----- build layers in parallel via rayon --------------------------
        // Each rayon worker pulls a cheap Gguf clone for its layer's device,
        // then loads attn / norms / FFN / per-layer-embed from the shared
        // mmap'd blob in one shot. With 4 MI50s and gemma4 31B (~17 GB Q4_0)
        // this drops model build time from ~52 s (sequential) to ~6 s.
        let layers: Vec<LayerWeights> = (0..block_count)
            .into_par_iter()
            .map(|il| -> Result<LayerWeights> {
                let block_prefix = format!("blk.{il}");
                let layer_dev_idx = layer_to_device[il];
                let layer_device = devices[layer_dev_idx].clone();
                let lgg = gg.with_device(layer_device.clone());

                let is_sliding = is_sliding_per_layer[il];
                let has_kv = il < n_layer_kv_from_start;
                let kv_source_idx = kv_source_per_layer[il];

                // Per-layer head_dim (sliding=key_length_swa, global=key_length).
                let layer_head_dim = if is_sliding { key_length_swa } else { key_length };

                // Per-layer rotated dimension count.
                let rotated_dim = if is_sliding {
                    rope_dim_count_swa.unwrap_or(layer_head_dim)
                } else {
                    rope_dim_count.unwrap_or_else(|| {
                        let partial = (layer_head_dim as f64 * 0.25) as usize;
                        (partial & !1).max(2)
                    })
                };
                let layer_rope_freq = if is_sliding {
                    rope_freq_base_swa
                } else {
                    rope_freq_base
                };
                let layer_freq_factors: Option<&[f32]> =
                    if !is_sliding { rope_freqs.as_deref() } else { None };

                // CANDLE_MAX_CONTEXT (env var) caps the precomputed rotary
                // sin/cos table to avoid reserving VRAM for a full 128k
                // context when benchmarking at much shorter lengths. Gemma-4
                // 31B with 62 layers otherwise burns ~4 GiB on RoPE tables
                // alone (131072 × head_dim × 4 bytes × n_layers).
                let rope_cap = std::env::var("CANDLE_MAX_CONTEXT")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(MAX_SEQ_LEN);
                let rotary = Arc::new(RotaryEmbedding::new_with_freq_factors(
                    layer_rope_freq as f64,
                    rotated_dim,
                    rope_cap.min(cfg.max_seq_len()),
                    layer_freq_factors,
                    DType::F32,
                    &layer_device,
                )?);

                // Per-layer GgufConfig clone with the right head_dim/n_kv_head.
                let mut layer_cfg = cfg.clone();
                layer_cfg.head_dim = layer_head_dim;
                layer_cfg.head_count_kv = PerLayer::Uniform(head_count_kv_per_layer[il]);

                // Gemma4 attention: the Q weights have `1/sqrt(query_pre_attn_scalar)`
                // baked in, so inference scale is 1.0. This is true for both
                // E4B (256) and 26B-A4B (assumed same convention).
                //
                // TODO: the 26B-A4B model's GGUF doesn't have
                // `query_pre_attn_scalar` metadata — turbo reads it from
                // `n_embd_head_k`. Need to verify the correct scale for
                // the 26B variant. The `Some(1.0)` works for E4B.
                let attn_opts = StandardAttentionOpts {
                    use_v_norm: true,
                    use_gemma_norms: false,
                    attention_scale: Some(1.0),
                };
                let attn = StandardAttention::load_with_opts(
                    &lgg,
                    &block_prefix,
                    &layer_cfg,
                    il,
                    rotary,
                    attn_opts,
                )?;

                let attn_norm =
                    lgg.rms_norm(&format!("{block_prefix}.attn_norm.weight"), rms_norm_eps)?;
                let post_attention_norm = lgg.rms_norm(
                    &format!("{block_prefix}.post_attention_norm.weight"),
                    rms_norm_eps,
                )?;
                let ffn_norm =
                    lgg.rms_norm(&format!("{block_prefix}.ffn_norm.weight"), rms_norm_eps)?;
                let mut post_ffn_norm = lgg
                    .rms_norm(&format!("{block_prefix}.post_ffw_norm.weight"), rms_norm_eps)?;

                // Gemma4 FFN uses GeGLU.
                let mlp = DenseMlp::load_with_activation(
                    &lgg,
                    &block_prefix,
                    MlpActivation::Gelu,
                )?;

                // Detect MoE layer: present if ffn_gate_inp.weight exists.
                let moe = if lgg.has_tensor(&format!("{block_prefix}.ffn_gate_inp.weight")) {
                    let gate_inp = lgg.qmatmul(&format!("{block_prefix}.ffn_gate_inp.weight"))?;
                    let gate_inp_s = lgg
                        .try_dequantize(&format!("{block_prefix}.ffn_gate_inp.scale"))
                        .ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "MoE layer {block_prefix} has gate_inp but no gate_inp.scale"
                            ))
                        })?;
                    let gate_up_exps = std::sync::Arc::new(
                        lgg.tensor(&format!("{block_prefix}.ffn_gate_up_exps.weight"))?,
                    );
                    let down_exps = std::sync::Arc::new(
                        lgg.tensor(&format!("{block_prefix}.ffn_down_exps.weight"))?,
                    );
                    // Expert FFN intermediate: infer from gate_up_exps shape
                    // [n_experts, 2*intermediate, hidden] → intermediate = dim[1]/2
                    let expert_intermediate = gate_up_exps.shape().dims()[1] / 2;
                    let num_experts_used = cfg.expert_used_count.unwrap_or(8);
                    let pre_norm = lgg.rms_norm(
                        &format!("{block_prefix}.pre_ffw_norm_2.weight"),
                        rms_norm_eps,
                    )?;
                    let post_norm = lgg.rms_norm(
                        &format!("{block_prefix}.post_ffw_norm_2.weight"),
                        rms_norm_eps,
                    )?;
                    let down_exps_scale =
                        lgg.try_dequantize(&format!("{block_prefix}.ffn_down_exps.scale"));
                    // post_ffw_norm_1 is the dense branch post-norm.
                    // post_ffw_norm (plain, no suffix) stays as the THIRD
                    // norm applied to the COMBINED dense+MoE output.
                    let dense_post_norm = lgg.rms_norm(
                        &format!("{block_prefix}.post_ffw_norm_1.weight"),
                        rms_norm_eps,
                    )?;
                    Some(MoeBranch {
                        dense_post_norm,
                        gate_inp,
                        gate_inp_s,
                        gate_up_exps,
                        down_exps,
                        down_exps_scale,
                        expert_intermediate,
                        num_experts_used,
                        pre_norm,
                        post_norm,
                        n_embd: embedding_length,
                        eps: rms_norm_eps as f32,
                    })
                } else {
                    None
                };

                let layer_output_scale =
                    lgg.try_dequantize(&format!("{block_prefix}.layer_output_scale.weight"));
                if il < 2 {
                    if let Some(ref s) = layer_output_scale {
                        let v: Vec<f32> = s.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
                        eprintln!("[load L{il}] layer_output_scale = {v:?}");
                    }
                }

                let per_layer_embed = if n_embd_per_layer > 0 {
                    let mut inp_gate = lgg.try_qmatmul(&format!("{block_prefix}.inp_gate.weight"));
                    let mut proj = lgg.try_qmatmul(&format!("{block_prefix}.proj.weight"));
                    // E4B per_layer weights are stored as F16 in the GGUF →
                    // `QMatMul::from_arc` dequantizes them at load and
                    // forward dispatches to `Tensor::matmul` → rocBLAS,
                    // which the G2 launch recorder doesn't see (rocBLAS
                    // calls bypass `LaunchArgs::launch`). Force these
                    // matmuls onto the MMVQ kernel chain by re-quantizing
                    // to Q8_0 — that's what makes G2 replay actually
                    // capture the per_layer_embed projections. Cost is
                    // ~0.1% accuracy loss on per_layer (small weights),
                    // which is negligible vs the model's own Q4_0 main
                    // path. Opt-out via CANDLE_GEMMA4_PE_NO_REQUANT=1.
                    if std::env::var("CANDLE_GEMMA4_PE_NO_REQUANT").is_err() {
                        // Quantization choice for the requantize: default Q8_0
                        // (best precision among Q-types). User can override
                        // via CANDLE_GEMMA4_PE_REQUANT_DTYPE=q4_0|q4_1|q5_0|q8_0.
                        let dtype = match std::env::var("CANDLE_GEMMA4_PE_REQUANT_DTYPE")
                            .ok().as_deref()
                        {
                            Some("q4_0") => candle::quantized::GgmlDType::Q4_0,
                            Some("q4_1") => candle::quantized::GgmlDType::Q4_1,
                            Some("q5_0") => candle::quantized::GgmlDType::Q5_0,
                            _ => candle::quantized::GgmlDType::Q8_0,
                        };
                        if let Some(ref mut m) = inp_gate {
                            let _ = m.requantize_to(dtype);
                        }
                        if let Some(ref mut m) = proj {
                            let _ = m.requantize_to(dtype);
                        }
                    }
                    if std::env::var("CANDLE_GEMMA4_PE_DTYPE_DEBUG").is_ok() && il == 0 {
                        eprintln!(
                            "[PE-dtype L{}] inp_gate is_qtensor={:?} proj is_qtensor={:?}",
                            il,
                            inp_gate.as_ref().map(|m| m.is_qtensor()),
                            proj.as_ref().map(|m| m.is_qtensor()),
                        );
                    }
                    let pn = lgg.try_rms_norm(
                        &format!("{block_prefix}.post_norm.weight"),
                        rms_norm_eps,
                    );
                    match (inp_gate, proj, pn) {
                        (Some(inp_gate), Some(proj), Some(post_norm)) => Some(PerLayerEmbed {
                            inp_gate,
                            proj,
                            post_norm,
                        }),
                        _ => None,
                    }
                } else {
                    None
                };

                Ok(LayerWeights {
                    attn,
                    attn_norm,
                    post_attention_norm,
                    ffn_norm,
                    post_ffn_norm,
                    mlp,
                    moe,
                    sliding_window_size: if is_sliding { Some(sliding_window_size) } else { None },
                    has_kv,
                    kv_source_idx,
                    layer_output_scale,
                    per_layer_embed,
                    device: layer_device,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let _ = head_count; // silence "unused" if metadata reads more than we use

        // ----- output norm + lm_head live on the device of the last layer --
        let last_dev = devices[layer_to_device[block_count - 1]].clone();
        let last_gg = gg.with_device(last_dev.clone());
        let norm = last_gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // gemma4 uses tied word embeddings: the lm_head IS token_embd.
        // IMPORTANT: the 26B-A4B GGUF has `output.weight` that shares
        // the same data offset as `blk.0.attn_output.weight` — loading
        // it as the lm_head produces wrong logits. Always use
        // token_embd.weight for the lm_head.
        let output = if let Some(te) = last_gg.try_qmatmul("token_embd.weight") {
            eprintln!("[gemma4] lm_head loaded from token_embd.weight");
            te
        } else if let Some(ow) = last_gg.try_qmatmul("output.weight") {
            eprintln!("[gemma4] WARNING: lm_head from output.weight (token_embd not found)");
            ow
        } else {
            return Err(candle::Error::Msg(
                "missing lm_head weight (token_embd.weight or output.weight)".into(),
            ));
        };

        // Pre-allocated KV cache slot per layer. Slot is `None` for
        // shared-KV layers (which borrow from another layer's slot).
        // Initialized lazily on first append in the forward.
        let kv_caches: Vec<Option<candle_nn::kv_cache::KvCache>> =
            (0..block_count).map(|_| None).collect();

        Ok(Self {
            tok_embeddings: Embedding::new_unused(),
            embedding_length,
            layers,
            norm,
            output,
            final_logit_softcap,
            per_layer_embeddings,
            kv_caches,
            #[cfg(feature = "hip")]
            decode_state: DecodeState::Init,
        }
        .with_tok_embeddings(tok_embeddings))
    }

    fn with_tok_embeddings(mut self, e: Embedding) -> Self {
        self.tok_embeddings = e;
        self
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        // If we're in Replay/Graph state, resume decode_alloc BEFORE the
        // prelude so the prelude's allocations (layer_in, inp_per_layer,
        // SWA mask) return the same pool slots they got at recording
        // time. The CPU→GPU memcpys inside those builders refresh the
        // *content* of those slots; the captured replay kernels then
        // read fresh data from a stable address — no External patching
        // needed for slots that were stable across the two recordings.
        #[cfg(feature = "hip")]
        let g2_in_replay = matches!(
            self.decode_state,
            DecodeState::Replay(_) | DecodeState::Graph { .. }
        );
        // V2-3b: also resume on Recorded1 — second captured forward
        // (recording-#2 run) needs to consume the cursor positions
        // that recording #1 populated.
        // V2-5a: also resume on WarmUp because
        // `decode_alloc_start_record` was followed by an immediate
        // pause in the post-forward state-machine branch, so the
        // FIRST recorded forward needs to resume (in Recording mode)
        // before its prelude allocs.  Between-forward main-loop
        // allocs stay in the normal pool (decode_alloc Paused
        // outside the forward).
        #[cfg(feature = "hip")]
        let g2_in_post_warmup = matches!(
            self.decode_state,
            DecodeState::Recorded1 { .. }
        );
        #[cfg(feature = "hip")]
        let g2_in_warmup = matches!(
            self.decode_state,
            DecodeState::WarmUp
        );
        #[cfg(feature = "hip")]
        if g2_in_replay || g2_in_post_warmup {
            // Recorded1 → replay existing recorded entries,
            // Replay/Graph → same.
            candle::hip_backend::hipdarc::driver::decode_alloc_resume();
            if std::env::var("CANDLE_G2_ALLOC_TRACE").is_ok() {
                eprintln!("[G2-alloc] forward start (resume replaying): mode={:?}",
                    candle::hip_backend::hipdarc::driver::decode_alloc_get_mode());
            }
        } else if g2_in_warmup {
            // WarmUp (rec #1) → enter Recording mode for this forward
            // so in-forward allocs append to the per-device tables.
            candle::hip_backend::hipdarc::driver::decode_alloc_set_mode(
                candle::hip_backend::hipdarc::driver::DecodeAllocMode::Recording
            );
            if std::env::var("CANDLE_G2_ALLOC_TRACE").is_ok() {
                eprintln!("[G2-alloc] forward start (resume recording): mode={:?}",
                    candle::hip_backend::hipdarc::driver::decode_alloc_get_mode());
            }
        }

        // ── Phase timing (CANDLE_G2_PHASE_TIME=1) ───────────────────────
        //
        // Records wall-clock per phase of the forward, prints aggregated
        // averages at the end of decode. Used to localize the G2 vs
        // default decode-throughput gap to a specific phase (prelude
        // CPU embed, prelude per_layer compute, SWA mask rebuild, fast
        // path replay, etc).
        #[cfg(feature = "hip")]
        let phase_time = std::env::var("CANDLE_G2_PHASE_TIME").is_ok();
        #[cfg(feature = "hip")]
        let _t_forward_start = std::time::Instant::now();

        // ── G2/G3 eligibility check ──
        //
        // Gated by `CANDLE_G2_REPLAY=1` so the default forward path skips
        // both recording and the prelude pause overhead. The plan
        // captures kernel launches across all devices in order — each
        // captured op carries its stream which determines the launching
        // device — so multi-device pipeline-parallel works at the kernel
        // level. Cross-device tensor transfers (`to_device`) are NOT
        // captured (memcpy isn't in the recorder hook); instead the
        // forward continues to call them fresh each token. As long as
        // the destination buffer ends up at a stable decode_alloc pool
        // address, the captured kernels on the next device read fresh
        // data. Multi-device support is therefore opt-in and conditional
        // on `CANDLE_G2_MULTI_DEV=1` while it's still being validated.
        #[cfg(feature = "hip")]
        let g2_eligible = seq_len == 1
            && index_pos > 0
            && {
                let single_device = {
                    let d0 = &self.layers[0].device;
                    self.layers.iter().all(|l| device_eq(&l.device, d0))
                };
                single_device || std::env::var("CANDLE_G2_MULTI_DEV").is_ok()
            }
            && std::env::var("CANDLE_G2_REPLAY").is_ok()
            && std::env::var("CANDLE_G2_DISABLE").is_err();

        // Phase P auto-gate. T-major K + mat-vec decode kernel is now a
        // win both on its own (+3.9 %) AND under G2 replay (+2.5 % vs
        // legacy G2, after Q1 propagated dyn_lk into the mat-vec
        // `l_k_iter` arg). So default-on is safe.
        // `CANDLE_KV_TMAJOR=0|1` explicit override still works for
        // benchmarks and bisecting regressions.
        // Mirrors the `CANDLE_NKV_PAD` pattern used below. Computed
        // once here, captured by the KvCache-init closure and re-read
        // at the narrow + dispatch sites below.
        #[cfg(feature = "hip")]
        let k_is_canonical: bool = match std::env::var("CANDLE_KV_TMAJOR").ok().as_deref() {
            Some("0") | Some("false") => false,
            _ => true,
        };
        #[cfg(not(feature = "hip"))]
        let k_is_canonical: bool = false;

        // Token embedding lookup (lives on CPU to avoid VRAM blow-up on the
        // larger 31B variants). Move the looked-up rows to the first layer's
        // device — that's where the actual transformer work begins.
        //
        // For G2/G3: this entire chain runs OUTSIDE the captured plan (we
        // pause kernel recording while it executes). The to_device call
        // is a hipMemcpyAsync — not capturable by the launch recorder.
        // The follow-on `* sqrt` kernel WOULD be capturable, but anchoring
        // its output address as input #0 requires the `* sqrt` to *not*
        // be in the plan: the plan must start from the layer loop, which
        // reads from `layer_in` as an external input that we patch each
        // replay.
        let first_layer_dev = self.layers[0].device.clone();
        #[cfg(feature = "hip")]
        let _t_prelude_li_start = std::time::Instant::now();
        #[cfg(feature = "hip")]
        let layer_in = if g2_eligible {
            candle::hip_backend::decode_cache::with_recording_paused(|| -> Result<Tensor> {
                let cpu = candle::Device::Cpu;
                let x_cpu = if device_eq(x.device(), &cpu) {
                    x.clone()
                } else {
                    x.to_device(&cpu)?
                };
                let layer_in_cpu = self.tok_embeddings.forward(&x_cpu)?;
                let mut layer_in = layer_in_cpu.to_device(&first_layer_dev)?;
                layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
                Ok(layer_in)
            })?
        } else {
            let cpu = candle::Device::Cpu;
            let x_cpu = if device_eq(x.device(), &cpu) {
                x.clone()
            } else {
                x.to_device(&cpu)?
            };
            let layer_in_cpu = self.tok_embeddings.forward(&x_cpu)?;
            let mut layer_in = layer_in_cpu.to_device(&first_layer_dev)?;
            layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
            layer_in
        };
        #[cfg(not(feature = "hip"))]
        let layer_in = {
            let cpu = candle::Device::Cpu;
            let x_cpu = if device_eq(x.device(), &cpu) {
                x.clone()
            } else {
                x.to_device(&cpu)?
            };
            let layer_in_cpu = self.tok_embeddings.forward(&x_cpu)?;
            let mut layer_in = layer_in_cpu.to_device(&first_layer_dev)?;
            layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
            layer_in
        };
        let mut layer_in = layer_in;
        #[cfg(feature = "hip")]
        let t_prelude_li = _t_prelude_li_start.elapsed();
        #[cfg(feature = "hip")]
        let _t_prelude_pe_start = std::time::Instant::now();

        // ── Per-layer embedding compute (E4B) — hoisted above the G2 fast
        // path so its result can serve as a second G2 external anchor.
        //
        // When G2/G3 is active, run with kernel recording PAUSED so that
        // the per_layer compute kernels (model_proj MMVQ, norm, residual
        // add) stay outside the captured plan. Their CPU→GPU memcpy of
        // the tiny per-layer embedding lookup can't be replayed (memcpy
        // sources aren't captured), so the entire chain has to run fresh
        // on every call, with the *result* anchored as input #1.
        let inp_per_layer: Option<Tensor> = if let Some(ref ple) = self.per_layer_embeddings {
            let n_layer = self.layers.len();
            let n_embd_per_layer = ple.n_embd_per_layer;
            let model_device = layer_in.device().clone();

            #[cfg(feature = "hip")]
            let compute = || -> Result<Tensor> {
                let cpu = candle::Device::Cpu;
                let x_cpu = x.to_device(&cpu)?;
                let pl_tok_embd = Embedding::new(ple.token_embd.clone(), n_embd_per_layer * n_layer);
                let inp_pe_cpu = pl_tok_embd.forward(&x_cpu)?;
                let inp_pe_cpu = inp_pe_cpu.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
                let inp_pe = (inp_pe_cpu * (n_embd_per_layer as f64).sqrt())?
                    .to_device(&model_device)?;
                let proj_out = ple.model_proj.forward(&layer_in)?;
                let proj_out = (proj_out * (1.0 / (self.embedding_length as f64).sqrt()))?;
                let proj_out = proj_out.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
                let proj_out = ple.proj_norm.forward(&proj_out)?;
                let combined = ((proj_out + inp_pe)? * (1.0 / 2f64.sqrt()))?;
                Ok(combined)
            };
            #[cfg(feature = "hip")]
            {
                if g2_eligible {
                    Some(candle::hip_backend::decode_cache::with_recording_paused(compute)?)
                } else {
                    Some(compute()?)
                }
            }
            #[cfg(not(feature = "hip"))]
            {
                let cpu = candle::Device::Cpu;
                let x_cpu = x.to_device(&cpu)?;
                let pl_tok_embd = Embedding::new(ple.token_embd.clone(), n_embd_per_layer * n_layer);
                let inp_pe_cpu = pl_tok_embd.forward(&x_cpu)?;
                let inp_pe_cpu = inp_pe_cpu.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
                let inp_pe = (inp_pe_cpu * (n_embd_per_layer as f64).sqrt())?
                    .to_device(&model_device)?;
                let proj_out = ple.model_proj.forward(&layer_in)?;
                let proj_out = (proj_out * (1.0 / (self.embedding_length as f64).sqrt()))?;
                let proj_out = proj_out.reshape((b_sz, seq_len, n_layer, n_embd_per_layer))?;
                let proj_out = ple.proj_norm.forward(&proj_out)?;
                let combined = ((proj_out + inp_pe)? * (1.0 / 2f64.sqrt()))?;
                Some(combined)
            }
        } else {
            None
        };
        #[cfg(feature = "hip")]
        let t_prelude_pe = _t_prelude_pe_start.elapsed();
        #[cfg(feature = "hip")]
        let _t_swa_start = std::time::Instant::now();

        // Snapshot the GPU inputs used as G2/G3 anchors. `layer_in` is
        // always anchor #0; `inp_per_layer` (when present) is anchor #1.
        // Used both by the fast-path memcpy below and by the post-forward
        // state machine that records the second decode token.
        #[cfg(feature = "hip")]
        let layer_in_anchor: Option<Tensor> = if g2_eligible {
            Some(layer_in.clone())
        } else {
            None
        };
        #[cfg(feature = "hip")]
        let inp_per_layer_anchor: Option<Tensor> = if g2_eligible {
            inp_per_layer.clone()
        } else {
            None
        };

        // ── SWA mask precompute (G2 path only) ──
        //
        // The captured plan needs the mask `last_dim` to be stable across
        // recordings AND across replays — otherwise the kernel's recorded
        // arg `nrows_y = mask.last_dim` mismatches at runtime. We pad the
        // mask out to the same value `pad_t` that the K cache will be
        // padded to in the layer loop below, so both are at a fixed
        // size for the whole 256-token replay window.
        //
        // `with_recording_paused` keeps the mask compute outside the
        // captured plan and forces the result tensor into the normal
        // pool (per-call address → marked External when its address
        // shows up in captured args).
        #[cfg(feature = "hip")]
        let g2_pad_t: usize = if g2_eligible {
            std::env::var("CANDLE_NKV_PAD")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(256)
        } else {
            1
        };
        #[cfg(feature = "hip")]
        let swa_masks: std::collections::HashMap<(String, usize), Tensor> = if g2_eligible {
            use std::collections::HashSet;
            let mut want: HashSet<(String, usize)> = HashSet::new();
            for l in &self.layers {
                if let Some(sw) = l.sliding_window_size {
                    want.insert((format!("{:?}", l.device.location()), sw));
                }
            }
            if want.is_empty() {
                std::collections::HashMap::new()
            } else {
                let current_t = index_pos + seq_len;
                let pad_total = if g2_pad_t > 1 {
                    ((current_t + g2_pad_t - 1) / g2_pad_t) * g2_pad_t
                } else {
                    current_t
                };
                candle::hip_backend::decode_cache::with_recording_paused(|| -> Result<_> {
                    let mut out = std::collections::HashMap::new();
                    for l in &self.layers {
                        let sw = match l.sliding_window_size {
                            Some(s) => s,
                            None => continue,
                        };
                        let key = (format!("{:?}", l.device.location()), sw);
                        if out.contains_key(&key) {
                            continue;
                        }
                        let m = causal_mask_padded(
                            b_sz,
                            seq_len,
                            index_pos,
                            Some(sw),
                            layer_in.dtype(),
                            &l.device,
                            pad_total,
                        )?;
                        out.insert(key, m);
                    }
                    Ok(out)
                })?
            }
        } else {
            std::collections::HashMap::new()
        };

        // Deterministic anchor ordering for the post-forward state machine.
        #[cfg(feature = "hip")]
        let swa_mask_anchors: Vec<((String, usize), Tensor)> = if g2_eligible {
            let mut v: Vec<_> = swa_masks
                .iter()
                .map(|(k, t)| (k.clone(), t.clone()))
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        } else {
            Vec::new()
        };

        #[cfg(feature = "hip")]
        let t_prelude_swa = _t_swa_start.elapsed();
        #[cfg(feature = "hip")]
        let _t_fastpath_start = std::time::Instant::now();

        // Diagnostic: dump the address + first values of each anchored
        // input on replay calls to verify that decode_alloc is returning
        // the same slot AND that the prelude's memcpy refreshed the data.
        #[cfg(feature = "hip")]
        if g2_eligible && g2_in_replay && std::env::var("CANDLE_G2_REPLAY_TRACE").is_ok() {
            let head: Vec<f32> = layer_in.narrow(1, 0, 1)?.flatten_all()?
                .to_device(&candle::Device::Cpu)?.to_vec1()?;
            let head4: Vec<f32> = head.iter().take(4).copied().collect();
            eprintln!(
                "[G2-gemma4] replay layer_in ptr=0x{:x} first4={:?}",
                Self::hip_device_ptr(&layer_in)?, head4
            );
        }

        // ── G2/G3 fast path: replay captured kernel sequence ──
        //
        // `layer_in` (post CPU-embed + transfer + scale) is anchor #0,
        // `inp_per_layer` (when E4B is active) is anchor #1. Both are
        // freshly allocated each call → memcpy into the captured anchor
        // address + patch external arg slots to the anchor pointer. Same
        // trick the llama path uses, just with two inputs instead of one.
        #[cfg(feature = "hip")]
        if g2_eligible {
            let layer_in_param = layer_in_anchor.as_ref().unwrap().clone();
            let inp_per_layer_param = inp_per_layer_anchor.clone();
            // Build the per-call (input_idx, fresh_ptr) list once and reuse
            // it for both Replay and Graph branches.
            //   #0 = layer_in
            //   #1 = inp_per_layer (E4B only)
            //   #2.. = SWA masks (one per device + sliding-window pair)
            let fresh_inputs: Vec<(usize, usize)> = {
                let mut v = vec![(0usize, Self::hip_device_ptr(&layer_in_param)?)];
                if let Some(ref ipl) = inp_per_layer_param {
                    v.push((1, Self::hip_device_ptr(ipl)?));
                }
                let mask_idx_base = v.len();
                for (i, (_key, t)) in swa_mask_anchors.iter().enumerate() {
                    v.push((mask_idx_base + i, Self::hip_device_ptr(t)?));
                }
                v
            };

            // Common helper: patch the captured arg slots that read from
            // each external input to the live (per-call) device pointer.
            //
            // Both `layer_in` and `inp_per_layer` are computed with HIP
            // recording paused, so their producing ops are NOT in the
            // plan; the captured kernels read the addresses directly.
            // Across the two recordings the addresses differ → those
            // args were marked External; patching points them at the
            // live tensor each replay. The `Tensor` clones held in
            // scope keep the GPU buffers alive through replay.
            let dev_for_replay = Self::hip_device_from_tensor(&layer_in_param)?;
            let stage_inputs = |plan: &mut candle::hip_backend::decode_cache::DecodePlan,
                                _dev: &candle::hip_backend::HipDevice|
             -> Result<()> {
                for &(idx, fresh_ptr) in &fresh_inputs {
                    if plan.input_count() == 0 {
                        // Single-input legacy plan — patch every external
                        // slot to the live pointer.
                        plan.patch_all_externals(fresh_ptr);
                    } else {
                        plan.patch_external_input(idx, fresh_ptr);
                    }
                }
                Ok(())
            };

            match &mut self.decode_state {
                DecodeState::Graph { plan, graph, captured_l_k } => {
                    use candle::hip_backend::hipdarc;
                    let current_t = index_pos + seq_len;
                    let pad_default: usize = if std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 };
                    let pad_t = std::env::var("CANDLE_NKV_PAD")
                        .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(pad_default);
                    let live_l_k = if pad_t > 1 {
                        ((current_t + pad_t - 1) / pad_t) * pad_t
                    } else {
                        current_t
                    };
                    if live_l_k != *captured_l_k {
                        eprintln!("[G3-gemma4] pad boundary crossed (captured={} live={}), rebuilding",
                            captured_l_k, live_l_k);
                        self.decode_state = DecodeState::Init;
                    } else {
                        hipdarc::driver::decode_alloc_resume();
                        stage_inputs(plan, &dev_for_replay)?;
                        if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                            plan.advance_counters();
                        }
                        // Phase T2: refresh the device-resident counter
                        // buffer slots before launching the captured graph.
                        // L_k_iter for the captured gqa_decode_mv_fast_*_ctr
                        // kernels reads from CounterSlot::LkIter; without
                        // this update the kernels see whatever value the
                        // slot held at recording time and produce stale
                        // attention output.
                        candle::hip_backend::g3_counters::set(
                            &dev_for_replay,
                            candle::hip_backend::g3_counters::CounterSlot::LkIter,
                            current_t as u32,
                        )?;
                        // V2-1: same validator hook as Replay branch —
                        // graph capture is driven by the same plan so
                        // the same patch-integrity check applies.
                        plan.validate_inputs(&fresh_inputs)?;
                        unsafe { graph.patch_and_launch(plan, &dev_for_replay)?; }
                        if std::env::var("CANDLE_G2_SYNC_REPLAY").is_ok() {
                            let _ = dev_for_replay.stream().synchronize();
                        }
                        hipdarc::driver::decode_alloc_pause();
                        hipdarc::driver::decode_alloc_reset();
                        let out = plan.output_tensor(&dev_for_replay)?;
                        if phase_time {
                            let t_fp = _t_fastpath_start.elapsed();
                            let t_total = _t_forward_start.elapsed();
                            eprintln!(
                                "[G2-time graph] li={:>5}us pe={:>5}us swa={:>5}us fp={:>5}us total={:>6}us",
                                t_prelude_li.as_micros(), t_prelude_pe.as_micros(),
                                t_prelude_swa.as_micros(), t_fp.as_micros(),
                                t_total.as_micros(),
                            );
                        }
                        return Ok(out);
                    }
                }
                DecodeState::Replay(plan) => {
                    let max_replays = std::env::var("CANDLE_G2_REPLAY_MAX")
                        .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
                    if std::env::var("CANDLE_G2_REPLAY").is_ok()
                        && plan.replay_count() < max_replays
                    {
                        use candle::hip_backend::hipdarc;
                        hipdarc::driver::decode_alloc_resume();
                        stage_inputs(plan, &dev_for_replay)?;
                        if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                            plan.advance_counters();
                        }
                        // Phase T2: refresh device-resident counter buffer
                        // (same rationale as the Graph branch above —
                        // replay shortcuts the gqa_attention_decode_mv
                        // call site that would normally write the slot).
                        let current_t_for_ctr = index_pos + seq_len;
                        candle::hip_backend::g3_counters::set(
                            &dev_for_replay,
                            candle::hip_backend::g3_counters::CounterSlot::LkIter,
                            current_t_for_ctr as u32,
                        )?;
                        // V2-1: validate external patching before we
                        // launch — catches stale ptr / missed input_id
                        // before the kernels crash on garbage addresses.
                        plan.validate_inputs(&fresh_inputs)?;
                        if std::env::var("CANDLE_G2_ALLOC_CURSOR_TRACE").is_ok() {
                            let c = candle::hip_backend::hipdarc::driver::decode_alloc_cursors_by_device();
                            eprintln!("[V2-3] pre-replay cursors-by-dev: {:?}", c);
                            let h = candle::hip_backend::hipdarc::driver::decode_alloc_head_entries(5);
                            for (dev, head) in h {
                                eprintln!(
                                    "[V2-3] dev {} head entries: {:?}",
                                    dev,
                                    head.iter().map(|(s, p)| format!("({}, 0x{:x})", s, p)).collect::<Vec<_>>(),
                                );
                            }
                        }
                        unsafe { plan.replay(&dev_for_replay)?; }
                        // Phase-time: optional sync-here-to-measure-GPU.
                        // CANDLE_G2_SYNC_REPLAY=1 makes the fp timing
                        // include actual GPU execution; without this flag
                        // it only measures launch/dispatch overhead.
                        if std::env::var("CANDLE_G2_SYNC_REPLAY").is_ok() {
                            let _ = dev_for_replay.stream().synchronize();
                        }
                        // Diagnostic for K10: dump the first 8 logits to
                        // confirm whether the captured plan actually
                        // produces fresh output per replay or stays
                        // stuck. If they're byte-identical across
                        // multiple replays, the plan has at least one
                        // critical arg that wasn't recognized as
                        // dynamic (Counter/External), so its kernel
                        // re-runs with stale data.
                        if std::env::var("CANDLE_G2_REPLAY_TRACE").is_ok() {
                            let _ = dev_for_replay.stream().synchronize();
                            let out = plan.output_tensor(&dev_for_replay)?;
                            let head: Vec<f32> = out
                                .narrow(out.rank() - 1, 0, 8.min(out.dim(out.rank()-1)?))?
                                .flatten_all()?.to_vec1()?;
                            eprintln!(
                                "[G2-gemma4] replay#{} idx={} output head: {:?}",
                                plan.replay_count(), index_pos, head
                            );
                            // Also dump inp_per_layer first 4 if present —
                            // verify the prelude is actually refreshing the
                            // anchor the captured per-layer kernels read from.
                            if let Some(ref ipl) = inp_per_layer_param {
                                let ipl_head: Vec<f32> = ipl
                                    .narrow(ipl.rank() - 1, 0, 4.min(ipl.dim(ipl.rank()-1)?))?
                                    .flatten_all()?
                                    .to_vec1()?;
                                eprintln!(
                                    "[G2-gemma4] replay#{} inp_per_layer ptr=0x{:x} head4={:?}",
                                    plan.replay_count(),
                                    Self::hip_device_ptr(ipl)?,
                                    ipl_head
                                );
                            }
                        }
                        // K13 deeper diagnostic: peek at every captured op's
                        // output buffer. Print only on replays 1 and 2 so we
                        // can diff between them. Ops whose head is identical
                        // across both replays despite different inputs are
                        // the ones reading from stale state somewhere.
                        if std::env::var("CANDLE_G2_PEEK_OPS").is_ok()
                            && plan.replay_count() <= 2
                        {
                            let peeks = plan.peek_op_outputs(&dev_for_replay, 4);
                            for (i, name, head) in peeks {
                                eprintln!(
                                    "[G2-peek r{}] op[{:>4}] {:>40} {}",
                                    plan.replay_count(), i, name,
                                    match head {
                                        Some(v) => format!("head={:?}", v),
                                        None => "(unknown output arg)".to_string(),
                                    }
                                );
                            }
                        }
                        // Even deeper: dump per-op arg pointer lists once
                        // (replay 1 only). Look for buffer-chain breaks
                        // between consecutive ops.
                        if std::env::var("CANDLE_G2_PEEK_ARGS").is_ok()
                            && plan.replay_count() == 1
                        {
                            for (i, name, args, sizes) in plan.peek_op_args() {
                                let arg_strs: Vec<String> = args.iter().zip(sizes.iter())
                                    .map(|(v, sz)| if *sz == 8 {
                                        format!("0x{:012x}", v)
                                    } else {
                                        format!("{}", *v as i64)
                                    })
                                    .collect();
                                eprintln!(
                                    "[G2-args] op[{:>4}] {:>40} args=[{}]",
                                    i, name, arg_strs.join(", ")
                                );
                            }
                        }
                        let g3_enabled = std::env::var("CANDLE_G3_GRAPH").is_ok();
                        let g3_after = std::env::var("CANDLE_G3_AFTER")
                            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);
                        let should_capture = g3_enabled && plan.replay_count() == g3_after;
                        if should_capture {
                            let current_t = index_pos + seq_len;
                            let pad_t_env: usize = std::env::var("CANDLE_NKV_PAD")
                                .ok().and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(if g3_enabled || std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 });
                            let live_l_k = if pad_t_env > 1 {
                                ((current_t + pad_t_env - 1) / pad_t_env) * pad_t_env
                            } else {
                                current_t
                            };
                            eprintln!("[G3-gemma4] capturing graph after {} replays (L_k_padded={})",
                                g3_after, live_l_k);
                            hipdarc::driver::decode_alloc_pause();
                            hipdarc::driver::decode_alloc_reset();
                            let output = plan.output_tensor(&dev_for_replay)?;
                            let old = std::mem::replace(&mut self.decode_state, DecodeState::Init);
                            if let DecodeState::Replay(plan) = old {
                                match unsafe { candle::hip_backend::decode_cache::DecodeGraph::capture(&plan, &dev_for_replay) } {
                                    Ok(graph) => {
                                        eprintln!("[G3-gemma4] graph captured successfully");
                                        self.decode_state = DecodeState::Graph {
                                            plan, graph, captured_l_k: live_l_k,
                                        };
                                    }
                                    Err(e) => {
                                        eprintln!("[G3-gemma4] graph capture failed: {:?} — staying in Replay", e);
                                        self.decode_state = DecodeState::Replay(plan);
                                    }
                                }
                            }
                            return Ok(output);
                        }
                        hipdarc::driver::decode_alloc_pause();
                        hipdarc::driver::decode_alloc_reset();
                        let out = plan.output_tensor(&dev_for_replay)?;
                        if phase_time {
                            let t_fp = _t_fastpath_start.elapsed();
                            let t_total = _t_forward_start.elapsed();
                            eprintln!(
                                "[G2-time replay] li={:>5}us pe={:>5}us swa={:>5}us fp={:>5}us total={:>6}us",
                                t_prelude_li.as_micros(), t_prelude_pe.as_micros(),
                                t_prelude_swa.as_micros(), t_fp.as_micros(),
                                t_total.as_micros(),
                            );
                        }
                        return Ok(out);
                    }
                }
                _ => {}
            }
        }
        if std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
            let emb: Vec<f32> = layer_in.narrow(1, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
            eprintln!("[EMBED] first token (scaled): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]", emb[0], emb[1], emb[2], emb[3], emb[4]);
        }

        // (per-layer embedding compute hoisted up before the G2 fast path)

        // (SWA mask precompute hoisted up before the G2 fast path)

        // A1 — piecewise capture: start KERNEL recording HERE (after
        // prelude, after G2 fast-path early-return) so the captured plan
        // ONLY contains the transformer loop + final norm / LM-head /
        // softcap. The sampler / argmax that runs between forwards in the
        // generation loop stays EAGER (its kernel launches are not added
        // to the plan), exactly like vLLM / Aphrodite's PIECEWISE capture
        // mode. The decode_alloc table is still active across sampler +
        // prelude so that anchor addresses (layer_in, inp_per_layer, SWA
        // masks) stay stable across recordings and replays.
        //
        // Before A1, start_recording was called at end-of-forward N so the
        // recording spanned [end-of-N → end-of-N+1] and included all ops
        // launched between forwards (sampler, next-token embed).
        #[cfg(feature = "hip")]
        if g2_eligible
            && matches!(
                self.decode_state,
                DecodeState::WarmUp | DecodeState::Recorded1 { .. }
            )
            && !candle::hip_backend::decode_cache::is_recording()
        {
            candle::hip_backend::decode_cache::start_recording();
        }

        // ----- transformer block loop --------------------------------------
        for il in 0..self.layers.len() {
            let (has_kv, kv_source_idx, layer_device, sliding_window_size) = {
                let l = &self.layers[il];
                (l.has_kv, l.kv_source_idx, l.device.clone(), l.sliding_window_size)
            };

            // Pipeline-parallel: move residual stream to this layer's device.
            if !device_eq(layer_in.device(), &layer_device) {
                layer_in = layer_in.to_device(&layer_device)?;
            }

            // Build the per-device causal mask (sliding-aware).
            //
            // Skip the mask for the trivial decode-on-global-layer case
            // (single token, no SWA): a 1-row causal mask is all zeros and the
            // attention is correct without it. For sliding layers we MUST build
            // a mask even at decode, otherwise the query attends to keys older
            // than the window — see llama.cpp `gemma4-iswa.cpp` SWA mask path.
            //
            // G2 path: reuse the precomputed mask from `swa_masks` so the
            // captured kernel reads from a single anchored buffer (instead
            // of 32 freshly-allocated ones per token).
            #[cfg(feature = "hip")]
            let attention_mask = if g2_eligible {
                if let Some(sw) = sliding_window_size {
                    let key = (format!("{:?}", layer_device.location()), sw);
                    swa_masks.get(&key).cloned()
                } else {
                    None
                }
            } else if seq_len == 1 && sliding_window_size.is_none() {
                None
            } else {
                Some(causal_mask(
                    b_sz,
                    seq_len,
                    index_pos,
                    sliding_window_size,
                    layer_in.dtype(),
                    &layer_device,
                )?)
            };
            #[cfg(not(feature = "hip"))]
            let attention_mask = if seq_len == 1 && sliding_window_size.is_none() {
                None
            } else {
                Some(causal_mask(
                    b_sz,
                    seq_len,
                    index_pos,
                    sliding_window_size,
                    layer_in.dtype(),
                    &layer_device,
                )?)
            };

            // -------- attention block --------
            let residual = layer_in.clone();

            // Try the fused norm+Q8_1 path first: replaces `attn_norm.forward`
            // + `compute_qkv_shared_q8`'s internal quantize_q8_1 with a single
            // `rmsnorm_q8_fused` kernel. Saves 1 launch per layer per token.
            // Falls back to the unfused path when conditions aren't met
            // (large batch, non-quantized weights, etc.).
            #[cfg(feature = "hip")]
            let (shared_qkv, x_norm) = if has_kv {
                if let Some((q, k, v, x_norm)) = self.layers[il].attn
                    .compute_qkv_norm_q8_fused(
                        &layer_in,
                        self.layers[il].attn_norm.weight(),
                        self.layers[il].attn_norm.eps_f32() as f64,
                        index_pos,
                    )?
                {
                    (Some((q, k, v)), x_norm)
                } else {
                    let x_norm = self.layers[il].attn_norm.forward(&layer_in)?;
                    let shared = self.layers[il].attn.compute_qkv_shared_q8(&x_norm, index_pos)?;
                    (shared, x_norm)
                }
            } else {
                (None, self.layers[il].attn_norm.forward(&layer_in)?)
            };
            #[cfg(not(feature = "hip"))]
            let (shared_qkv, x_norm): (Option<(candle::Tensor, candle::Tensor, candle::Tensor)>, candle::Tensor) = (
                None,
                self.layers[il].attn_norm.forward(&layer_in)?,
            );

            // Compute fresh K/V if this layer owns its cache and append
            // them to the pre-allocated `KvCache`.
            if has_kv {
                let kv_result = if let Some((_, ref k_new, ref v_new)) = shared_qkv {
                    Some((k_new.clone(), v_new.clone()))
                } else {
                    self.layers[il].attn.compute_kv(&x_norm, index_pos)?
                };
                if let Some((k_new, v_new)) = kv_result
                {
                    // First-touch initialization at index_pos == 0:
                    // wipe any state from a previous generation. The
                    // `KvCache::reset` is `O(1)` and doesn't free the
                    // backing buffer.
                    if index_pos == 0 {
                        if let Some(ref mut c) = self.kv_caches[il] {
                            c.reset();
                        }
                    }
                    let cache = self.kv_caches[il]
                        .get_or_insert_with(|| {
                            // Phase P: K canonical (B, H_kv, T, D) with D
                            // contiguous — enables mat-vec decode attention
                            // (+3.9% on default path). Gated by
                            // `k_is_canonical` hoisted at forward() entry:
                            // on when G2 replay is NOT requested (explicit
                            // `CANDLE_KV_TMAJOR=0|1` always wins).
                            if k_is_canonical {
                                candle_nn::kv_cache::KvCache::new_k_canonical_stable(2, 4096)
                            } else {
                                candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096)
                            }
                        });
                    // KvCache::append needs contiguous sources.
                    let k_new = if k_new.is_contiguous() { k_new } else { k_new.contiguous()? };
                    let v_new = if v_new.is_contiguous() { v_new } else { v_new.contiguous()? };
                    let _ = cache.append(&k_new, &v_new)?;
                }
            }

            // Read K/V from this layer's slot or borrow from the source slot
            // (transferring across devices if needed).
            //
            // H1: pad attention sequence length to a multiple of
            // CANDLE_NKV_PAD (default 256 when G2 is enabled — keeps the
            // captured plan's K-cache args at a stable shape across the
            // 256-token replay window; off otherwise to avoid the extra
            // -inf padding work when not replaying).
            #[cfg(feature = "hip")]
            let pad_t = std::env::var("CANDLE_NKV_PAD")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(if g2_eligible { 256 } else { 1 });
            #[cfg(not(feature = "hip"))]
            let pad_t = std::env::var("CANDLE_NKV_PAD")
                .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
            let (_b_sz_q, _, seq_len_q, _) = q_precomputed_shape_or_none(&shared_qkv, &x_norm)?;
            let (k_use, v_use, pad_info) = {
                let src = if has_kv { il } else { kv_source_idx };
                let cache = self.kv_caches[src]
                    .as_ref()
                    .expect("KV cache must exist for has_kv source layer");
                let k_cache = cache.k_cache();
                let v_cache = cache.v_cache();
                let t_cur = k_cache.current_seq_len();
                let max_t = k_cache.max_seq_len();
                let l_k_padded = if pad_t > 1 {
                    (((t_cur + pad_t - 1) / pad_t) * pad_t).min(max_t)
                } else {
                    t_cur
                };
                let (k, v, padded) = if l_k_padded > t_cur && seq_len_q == 1 {
                    let k_full = k_cache.all_data().as_ref()
                        .ok_or_else(|| candle::Error::Msg("gemma4 k cache empty".into()))?;
                    let v_full = v_cache.all_data().as_ref()
                        .ok_or_else(|| candle::Error::Msg("gemma4 v cache empty".into()))?;
                    // Phase P canonical layout uses dim 2 (T axis);
                    // legacy D-major uses dim 3. Flag hoisted at
                    // forward() entry as `k_is_canonical`.
                    let k = if k_is_canonical {
                        k_full.narrow(2, 0, l_k_padded)?
                    } else {
                        k_full.narrow(3, 0, l_k_padded)?
                    };
                    // V stored (B, H_kv, maxT, D) → narrow dim 2 to l_k_padded.
                    let v = v_full.narrow(2, 0, l_k_padded)?;
                    (k, v, Some((t_cur, l_k_padded)))
                } else {
                    let k = cache.k()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                    let v = cache.v()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                    (k, v, None)
                };
                let k = if device_eq(k.device(), &layer_device) { k } else { k.to_device(&layer_device)? };
                let v = if device_eq(v.device(), &layer_device) { v } else { v.to_device(&layer_device)? };
                (k, v, padded)
            };

            // When padded, extend the attention_mask (or build a new one
            // for decode where mask is None) so positions [T_cur, L_k_pad)
            // get -inf and contribute nothing to softmax.
            let attention_mask = if let Some((t_cur, l_k_pad)) = pad_info {
                let dev = q_precomputed_device_or_xnorm(&shared_qkv, &x_norm);
                let head_dim_neg_inf = Tensor::full(f32::NEG_INFINITY, (1usize, 1, 1, l_k_pad - t_cur), dev)?;
                let zeros = Tensor::zeros((1usize, 1, 1, t_cur), DType::F32, dev)?;
                let pad_mask = Tensor::cat(&[&zeros, &head_dim_neg_inf], 3)?.contiguous()?;
                Some(pad_mask)
            } else {
                attention_mask
            };

            // Dump layer 0 attention inputs for turbo comparison
            if il == 0 && std::env::var("CANDLE_GEMMA4_DUMP_L0").is_ok() {
                let xn: Vec<f32> = x_norm.narrow(1, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
                eprintln!("[L0 attn_in] first tok [0..5]: {:.6?}", &xn[..5]);
                let kv = k_use.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                eprintln!("[L0 k_cache] first 5 vals: {:.6?} shape={:?}", &kv[..5.min(kv.len())], k_use.shape());
            }
            // Attack C: K is pre-transposed in the cache.
            // D2: if shared_qkv provided Q already, use it directly to skip
            // the redundant wq.forward(x_norm) + quantize_q8_1.
            //
            // G2 dynamic-L_k: when the KV cache is padded for replay
            // stability (pad_info present, seq_len_q=1), set the
            // per-call override so the flash-attn dispatch iterates
            // only `t_cur` positions instead of `l_k_padded`. The
            // override is a thread-local Cell so it vanishes right
            // after the attention call below.
            let dyn_lk_guard = if let Some((t_cur_dyn, _)) = pad_info {
                if std::env::var("CANDLE_G2_DYN_LK").ok().as_deref() != Some("0") {
                    crate::models::quantized_blocks::attention::set_flash_l_k_iter_override(
                        Some(t_cur_dyn),
                    );
                    true
                } else { false }
            } else { false };
            // Phase P dispatch. `k_is_canonical` (hoisted at forward()
            // entry) routes decode through the mat-vec kernel on
            // canonical K; legacy D-major K stays on
            // flash_attn_v2_fwd_ktvs.
            let attn = if let Some((ref q_precomputed, _, _)) = shared_qkv {
                let (b_sz, seq_len, _) = x_norm.dims3()?;
                let (_, _, _, k_dim3) = k_use.dims4()?;
                // Sanity: canonical K has dim3 == head_dim; D-major has dim3 == t_k.
                let head_dim = q_precomputed.dim(candle::D::Minus1)?;
                let attn_output = if k_is_canonical && k_dim3 == head_dim {
                    if seq_len == 1 {
                        // Decode: mat-vec kernel on T-major K.
                        // K/V are narrow'd views of the padded cache
                        // (non-contig); the mat-vec kernel requires
                        // contiguous inputs. Per-layer copy is
                        // l_k_padded*D*4 bytes.
                        //
                        // l_k_iter: when pad_info is Some (G2 replay
                        // active with fixed pad), the K buffer is padded
                        // past the real context length — iterate only
                        // t_cur positions, not the padded extent
                        // (Phase L dyn_lk equivalent for mat-vec).
                        let k_c = if k_use.is_contiguous() { k_use.clone() } else { k_use.contiguous()? };
                        let v_c = if v_use.is_contiguous() { v_use.clone() } else { v_use.contiguous()? };
                        let l_k_iter = pad_info.map(|(t_cur, _)| t_cur).unwrap_or(k_c.dim(2)?);
                        candle::hip_backend::gqa_attention_decode_mv(
                            q_precomputed, &k_c, &v_c,
                            attention_mask.as_ref(),
                            self.layers[il].attn.attn_scale,
                            l_k_iter,
                        )?
                    } else {
                        // Prefill: reuse the legacy D-major fast path by
                        // transposing T-major K to (B, H, D, T) once per
                        // call. Per-call cost is small vs the decode-wide
                        // win, and keeps the flash_attn_v2_kt_strided_v
                        // prefill kernel reachable. Long-term: prefill
                        // kernel rewrite for canonical K (Phase N).
                        let k_dmajor = k_use.transpose(2, 3)?.contiguous()?;
                        gqa_attention_k_transposed(
                            q_precomputed, &k_dmajor, &v_use,
                            attention_mask.as_ref(), self.layers[il].attn.attn_scale,
                        )?
                    }
                } else {
                    // Legacy D-major path
                    gqa_attention_k_transposed(
                        q_precomputed, &k_use, &v_use,
                        attention_mask.as_ref(), self.layers[il].attn.attn_scale,
                    )?
                };
                self.layers[il].attn.finish_attn(attn_output, b_sz, seq_len)?
            } else if k_is_canonical {
                // Phase P: K canonical (B, H_kv, T, D). For DECODE, route
                // Q through prepare_q and call decode_mv directly. For
                // PREFILL, transpose K to D-major and fall into the
                // legacy flash_attn_v2_fwd_ktvs path (via
                // forward_with_kv_transposed) — that handles d=512
                // correctly; the non-transposed canonical prefill path
                // has no d=512 kernel and bounces to rocBLAS with a
                // shape rocBLAS refuses.
                let (b_sz, seq_len_q, _) = x_norm.dims3()?;
                if seq_len_q == 1 {
                    let q = self.layers[il].attn.prepare_q(&x_norm, index_pos)?;
                    let k_c = if k_use.is_contiguous() { k_use.clone() } else { k_use.contiguous()? };
                    let v_c = if v_use.is_contiguous() { v_use.clone() } else { v_use.contiguous()? };
                    // l_k_iter: same dyn_lk story as the shared_qkv branch
                    // above — iterate t_cur, not the padded l_k_padded.
                    let l_k_iter = pad_info.map(|(t_cur, _)| t_cur).unwrap_or(k_c.dim(2)?);
                    let attn_output = candle::hip_backend::gqa_attention_decode_mv(
                        &q, &k_c, &v_c,
                        attention_mask.as_ref(),
                        self.layers[il].attn.attn_scale,
                        l_k_iter,
                    )?;
                    self.layers[il].attn.finish_attn(attn_output, b_sz, seq_len_q)?
                } else {
                    let k_dmajor = k_use.transpose(2, 3)?.contiguous()?;
                    self.layers[il].attn.forward_with_kv_transposed(
                        &x_norm,
                        &k_dmajor,
                        &v_use,
                        attention_mask.as_ref(),
                        index_pos,
                    )?
                }
            } else {
                self.layers[il].attn.forward_with_kv_transposed(
                    &x_norm,
                    &k_use,
                    &v_use,
                    attention_mask.as_ref(),
                    index_pos,
                )?
            };
            if dyn_lk_guard {
                crate::models::quantized_blocks::attention::set_flash_l_k_iter_override(None);
            }
            if il == 0 && std::env::var("CANDLE_GEMMA4_DUMP_L0").is_ok() {
                let ao: Vec<f32> = attn.narrow(1, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
                eprintln!("[L0 attn_out] first tok [0..5]: {:.6?} abs_mean={:.4}", &ao[..5], ao.iter().map(|v| v.abs()).sum::<f32>() / ao.len() as f32);
            }
            // Fused post-attn norm + residual add (Q0a). On HIP this is
            // one launch; falls back to rms_norm + add otherwise.
            let x = self.layers[il]
                .post_attention_norm
                .forward_post_residual(&attn, &residual)?;

            // -------- FFN block --------
            let skip_moe = std::env::var("CANDLE_GEMMA4_SKIP_MOE").is_ok();
            let mut x = if !skip_moe && self.layers[il].moe.is_some() {
                let moe_branch = self.layers[il].moe.as_ref().unwrap();
                // Dual dense+MoE: both branches read from `x` (the
                // attention residual output, a.k.a. attn_out in turbo).
                let attn_out = x;

                // Branch 1: Dense MLP
                let dense_in = self.layers[il].ffn_norm.forward(&attn_out)?;
                let dense_out = self.layers[il].mlp.forward(&dense_in)?;
                let dense_normed = moe_branch.dense_post_norm.forward(&dense_out)?;

                // Branch 2: MoE experts
                let moe_in = moe_branch.pre_norm.forward(&attn_out)?;

                // Custom router logits:
                //   tmp = rms_norm(attn_out) / sqrt(n_embd) * gate_inp_s
                //   logits = gate_inp @ tmp
                let router_input = {
                    let normed = crate::models::quantized_blocks::norms::v_norm(
                        &attn_out,
                        moe_branch.eps as f64,
                    )?;
                    let scaled = (normed * (1.0 / (moe_branch.n_embd as f64).sqrt()))?;
                    scaled.broadcast_mul(&moe_branch.gate_inp_s)?
                };
                let router_logits = moe_branch.gate_inp.forward(&router_input)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // TopK selection (route through CPU for argsort).
                let device = routing_weights.device().clone();
                let (b_sz_ff, seq_len_ff, _hidden) = moe_in.dims3()?;
                let flat_tokens = b_sz_ff * seq_len_ff;
                let n_experts = routing_weights.dim(candle::D::Minus1)?;
                let rw_flat = routing_weights.reshape((flat_tokens, n_experts))?;
                let topk_ids = {
                    let rw_cpu = rw_flat.to_device(&candle::Device::Cpu)?;
                    let ids = rw_cpu
                        .arg_sort_last_dim(false)?
                        .narrow(candle::D::Minus1, 0, moe_branch.num_experts_used)?
                        .contiguous()?;
                    // A3 / EPLB diagnostic — observe routing while ids live
                    // on CPU.  No-op when CANDLE_EPLB_PRINT / _DUMP unset.
                    if let Ok(v) = ids.flatten_all()?.to_vec1::<u32>() {
                        crate::models::quantized_blocks::eplb::observe(&v, il);
                    }
                    ids.to_device(&device)?
                };
                let topk_weights = rw_flat.gather(&topk_ids, candle::D::Minus1)?;
                let topk_weights = (&topk_weights
                    / topk_weights
                        .sum_keepdim(candle::D::Minus1)?
                        .broadcast_as(topk_weights.shape())?)?;

                // Expert computation via fused gate_up_exps
                let hidden = moe_in.dim(candle::D::Minus1)?;
                let moe_in_flat = moe_in.reshape((flat_tokens, hidden))?;
                let x_3d = moe_in_flat.unsqueeze(1)?.contiguous()?;
                if il == 0 && std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
                    eprintln!("[MoE L{il}] attn_out={:?} moe_in={:?} x_3d={:?} topk_ids={:?} gate_up_exps={:?} down_exps={:?} expert_intermediate={}",
                        attn_out.shape(), moe_in.shape(), x_3d.shape(), topk_ids.shape(),
                        moe_branch.gate_up_exps.shape(), moe_branch.down_exps.shape(),
                        moe_branch.expert_intermediate);
                    // Print first token's routing
                    let ids_cpu: Vec<u32> = topk_ids.to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    let wts_cpu: Vec<f32> = topk_weights.to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    eprintln!("[MoE L{il}] token0 expert_ids={:?} weights={:.4?}", &ids_cpu[..8.min(ids_cpu.len())], &wts_cpu[..8.min(wts_cpu.len())]);
                    eprintln!("[MoE L{il}] gate_inp_s={:?}", moe_branch.gate_inp_s.shape());
                    let rl_cpu: Vec<f32> = router_logits.narrow(0, 0, 1).unwrap().to_device(&candle::Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
                    let top5: Vec<(usize, f32)> = {
                        let mut indexed: Vec<_> = rl_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        indexed[..5].to_vec()
                    };
                    eprintln!("[MoE L{il}] router_logits top5={:?}", top5);
                }
                let gate_up = moe_branch
                    .gate_up_exps
                    .indexed_moe_forward(&x_3d, &topk_ids)?;
                // Debug: compare indexed_moe_forward result against CPU manual matmul
                if il == 0 && std::env::var("CANDLE_GEMMA4_DUMP_L0").is_ok() {
                    let gu_cpu: Vec<f32> = gate_up.narrow(0, 0, 1)?.narrow(1, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
                    eprintln!("[L0 MoE] gate_up[tok0,expert0,:5] = {:.6?} shape={:?}", &gu_cpu[..5.min(gu_cpu.len())], gate_up.shape());
                    // Manual: dequant expert 0's gate_up row 0, dot with input
                    let expert0_id: u32 = topk_ids.narrow(0, 0, 1)?.narrow(1, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?[0];
                    eprintln!("[L0 MoE] top expert for token 0: id={expert0_id}");
                    let input_vals: Vec<f32> = x_3d.narrow(0, 0, 1)?.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
                    eprintln!("[L0 MoE] input[0,:5] = {:.6?}", &input_vals[..5.min(input_vals.len())]);
                }
                let gate = gate_up
                    .narrow(candle::D::Minus1, 0, moe_branch.expert_intermediate)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(
                        candle::D::Minus1,
                        moe_branch.expert_intermediate,
                        moe_branch.expert_intermediate,
                    )?
                    .contiguous()?;
                let activated = gate.gelu()?.broadcast_mul(&up)?;
                let mut moe_out = moe_branch
                    .down_exps
                    .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?;
                // Apply per-expert down scale if present (ffn_down_exps.scale).
                // Shape: [n_experts]; gather the scales for the selected
                // topk experts and broadcast over the hidden dim.
                if let Some(ref scale) = moe_branch.down_exps_scale {
                    // scale: [n_experts=128], topk_ids: [tokens, topk=8]
                    // Gather: [tokens, topk] per-expert scales
                    let expert_scales = scale
                        .index_select(&topk_ids.flatten_all()?, 0)?
                        .reshape(topk_ids.shape())?;
                    // Broadcast to [tokens, topk, 1] for hidden-dim multiply
                    let expert_scales = expert_scales.unsqueeze(candle::D::Minus1)?;
                    moe_out = moe_out.broadcast_mul(&expert_scales)?;
                }

                // Weight + sum across topk
                let topk_weights = topk_weights.unsqueeze(candle::D::Minus1)?;
                let weighted = moe_out.broadcast_mul(&topk_weights)?;
                let moe_summed = weighted.sum(1)?; // sum across topk dim
                let moe_summed = moe_summed.reshape((b_sz_ff, seq_len_ff, hidden))?;
                let moe_normed = moe_branch.post_norm.forward(&moe_summed)?;

                // Debug: zero out outputs for isolation
                let moe_normed = if std::env::var("CANDLE_GEMMA4_ZERO_MOE").is_ok() {
                    Tensor::zeros_like(&moe_normed)?
                } else {
                    moe_normed
                };
                let dense_normed = if std::env::var("CANDLE_GEMMA4_ZERO_DENSE").is_ok() {
                    Tensor::zeros_like(&dense_normed)?
                } else {
                    dense_normed
                };

                // Combine dense + MoE, apply the THIRD post-norm
                // (`post_ffw_norm` = `post_feedforward_layernorm` in HF),
                // THEN add the residual.
                //
                // Both turbo (gemma4-iswa.cpp:184-190) AND HF apply this:
                //   combined = post_norm_1(dense) + post_norm_2(moe)
                //   cur = post_ffw_norm(combined)     ← third norm!
                //   cur = residual + cur
                //
                // `post_ffw_norm` is loaded from `post_ffw_norm.weight`
                // (the ORIGINAL, unsuffixed tensor). It's stored in
                // `self.layers[il].post_ffn_norm` because we did NOT
                // override it — we loaded `_1` into MoeBranch.dense_post_norm
                // separately.
                let combined = (&dense_normed + &moe_normed)?;
                self.layers[il]
                    .post_ffn_norm
                    .forward_post_residual(&combined, &attn_out)?
            } else {
                // Dense-only path (E4B and similar).
                let residual = x.clone();
                // HIP fused-decode FFN: combines `ffn_norm` + `quantize_q8_1`
                // into `rmsnorm_q8_fused`, then reuses the Q8_1 buffer for
                // both `gate_up.forward_preq8` and (after the activation)
                // `down.forward_preq8`. Saves 2 quantize_q8_1 launches per
                // layer per decode token. Opt-out via
                // `CANDLE_GEMMA4_FFN_FUSED_OFF=1`.
                let ffn_out: Option<Tensor> = {
                    #[cfg(feature = "hip")]
                    {
                        let is_decode = seq_len == 1
                            && matches!(x.device(), candle::Device::Hip(_))
                            && x.dtype() == DType::F32
                            && std::env::var("CANDLE_GEMMA4_FFN_FUSED_OFF").is_err();
                        if is_decode {
                            let (b_sz_ff, _, _) = x.dims3()?;
                            let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
                            match candle::hip_backend::rmsnorm_q8_fused(
                                &x_c,
                                self.layers[il].ffn_norm.weight(),
                                self.layers[il].ffn_norm.eps_f32() as f64,
                                false,
                            ) {
                                Ok((x_q8_buf, _)) => {
                                    let x_q8_view = x_q8_buf.slice(0..x_q8_buf.len());
                                    let rhs_shape = vec![b_sz_ff, 1usize, x.dim(candle::D::Minus1)?];
                                    self.layers[il].mlp.forward_preq8_decode(
                                        &x_q8_view, b_sz_ff, &rhs_shape,
                                    ).ok()
                                }
                                Err(_) => None,
                            }
                        } else { None }
                    }
                    #[cfg(not(feature = "hip"))]
                    { None }
                };
                let ffn_out = match ffn_out {
                    Some(o) => o,
                    None => {
                        let ffn_in = self.layers[il].ffn_norm.forward(&x)?;
                        self.layers[il].mlp.forward(&ffn_in)?
                    }
                };
                if il == 0 && std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
                    let d_abs = ffn_out.abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_vec0::<f32>().unwrap();
                    let r_abs = residual.abs().unwrap().mean_all().unwrap().to_device(&candle::Device::Cpu).unwrap().to_vec0::<f32>().unwrap();
                    eprintln!("[E4B L{il}] dense_ffn={d_abs:.3} residual={r_abs:.3}");
                }
                // Fused post-ffn norm + residual add (Q0a).
                self.layers[il]
                    .post_ffn_norm
                    .forward_post_residual(&ffn_out, &residual)?
            };

            // -------- per-layer embedding injection (E4B) --------
            // Diagnostic switch: CANDLE_GEMMA4_NO_PE_INJECT=1 skips the
            // per-layer embed injection entirely. Used to localize where
            // the G2 replay state divergence comes from. WARNING: model
            // output is wrong (drops a learned residual contribution),
            // so only useful for diff'ing replay vs baseline at the same
            // configuration.
            let pe_inject_off = std::env::var("CANDLE_GEMMA4_NO_PE_INJECT").is_ok();
            if !pe_inject_off {
            if let (Some(ref ple), Some(ref ipl)) =
                (&self.layers[il].per_layer_embed, &inp_per_layer)
            {
                let pe_in = x.clone();
                let pe_cur = ple.inp_gate.forward(&x)?;
                let pe_cur = pe_cur.gelu()?;
                let inp_this_layer = ipl.narrow(2, il, 1)?.squeeze(2)?;
                let inp_this_layer = if device_eq(inp_this_layer.device(), &layer_device) {
                    inp_this_layer
                } else {
                    inp_this_layer.to_device(&layer_device)?
                };
                let pe_cur = (pe_cur * inp_this_layer)?;
                let pe_cur = ple.proj.forward(&pe_cur)?;
                let pe_cur = ple.post_norm.forward(&pe_cur)?;
                x = (pe_in + pe_cur)?;
            }
            } // end pe_inject_off guard

            // -------- layer output scale (gemma4) --------
            if let Some(ref scale) = self.layers[il].layer_output_scale {
                x = x.broadcast_mul(scale)?;
            }

            // K13 diagnostic: dump first 4 floats of layer output per call.
            // Set CANDLE_LAYER_DUMP=1 to enable. Used to diff baseline+flash
            // vs G2-replay forward at the same index_pos.
            if std::env::var("CANDLE_LAYER_DUMP").is_ok() {
                let head: Vec<f32> = x
                    .narrow(x.rank() - 1, 0, 4.min(x.dim(x.rank()-1)?))?
                    .flatten_all()?
                    .to_device(&candle::Device::Cpu)?
                    .to_vec1()?;
                eprintln!("[LAYER idx={} L{}] head={:?}", index_pos, il, head);
            }

            layer_in = x;
        }

        // ----- final norm + lm_head + softcap ------------------------------
        let x = layer_in.i((.., seq_len - 1, ..))?;
        if std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
            let x_cpu: Vec<f32> = x.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
            let has_nan = x_cpu.iter().any(|v| v.is_nan());
            let has_inf = x_cpu.iter().any(|v| v.is_infinite());
            let abs_mean = x_cpu.iter().map(|v| v.abs()).sum::<f32>() / x_cpu.len() as f32;
            eprintln!("[FINAL] pre-norm: shape={:?} nan={has_nan} inf={has_inf} abs_mean={abs_mean:.4}", x.shape());
        }
        let x = self.norm.forward(&x)?;
        let logits = self.output.forward(&x)?;
        if std::env::var("CANDLE_GEMMA4_DEBUG_MOE").is_ok() {
            eprintln!("[FINAL] softcap={:?}", self.final_logit_softcap);
            let l_cpu: Vec<f32> = logits.to_device(&candle::Device::Cpu)?.flatten_all()?.to_vec1()?;
            let top5: Vec<(usize, f32)> = {
                let mut indexed: Vec<_> = l_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed[..5.min(indexed.len())].to_vec()
            };
            eprintln!("[FINAL] logits top5={top5:?} has_nan={}", l_cpu.iter().any(|v| v.is_nan()));
        }

        #[cfg(feature = "hip")]
        if std::env::var("CANDLE_G2_SYNC_REPLAY").is_ok() {
            if let candle::Device::Hip(d) = layer_in.device() {
                let _ = d.stream().synchronize();
            }
        }
        #[cfg(feature = "hip")]
        let _t_loop_end = std::time::Instant::now();

        let result = if let Some(soft_cap) = self.final_logit_softcap {
            // Compute softcap in f64 to preserve discrimination at high
            // logit magnitudes. At f32, tanh(x) for x > ~9 returns exactly
            // 1.0f, losing all contrast between top tokens. The 26B-A4B
            // model produces raw logits of 300+ (legitimate for Q8_0
            // embeddings with dim 2816) where f32 tanh saturates.
            let logits64 = logits.to_dtype(candle::DType::F64)?;
            let capped = ((&logits64 / soft_cap)?.tanh()? * soft_cap)?;
            capped.to_dtype(candle::DType::F32)?
        } else {
            logits
        };

        // ── G2: Post-forward recording state machine ────────────────────
        //
        // Mirrors `quantized_llama::forward`. Anchors `layer_in_anchor`
        // as input #0; for E4B, also anchors `inp_per_layer_anchor` as
        // input #1. The captured plan reads from both anchor addresses;
        // `Recorded1` stashes the per-input ptr tuple from the first
        // recording so it can pair with the second to derive the
        // External patch slots.
        #[cfg(feature = "hip")]
        if g2_eligible {
            use candle::hip_backend::decode_cache;
            use candle::hip_backend::hipdarc::driver as hdrv;

            let in_anchor = layer_in_anchor.as_ref().unwrap();
            let inp_per_layer_anchor_ref = inp_per_layer_anchor.as_ref();
            if decode_cache::is_recording() {
                if let Some(recording) = decode_cache::stop_recording() {
                    // Snapshot per-input pointers + sizes for this token.
                    // Order matches `fresh_inputs` in the fast path:
                    //   #0 layer_in
                    //   #1 inp_per_layer (E4B)
                    //   #2.. SWA masks
                    let mut cur_ptrs: Vec<usize> = vec![Self::hip_device_ptr(in_anchor)?];
                    let mut cur_bytes: Vec<usize> =
                        vec![in_anchor.elem_count() * in_anchor.dtype().size_in_bytes()];
                    if let Some(t) = inp_per_layer_anchor_ref {
                        cur_ptrs.push(Self::hip_device_ptr(t)?);
                        cur_bytes.push(t.elem_count() * t.dtype().size_in_bytes());
                    }
                    for (_key, t) in &swa_mask_anchors {
                        cur_ptrs.push(Self::hip_device_ptr(t)?);
                        cur_bytes.push(t.elem_count() * t.dtype().size_in_bytes());
                    }

                    match std::mem::replace(&mut self.decode_state, DecodeState::Init) {
                        DecodeState::WarmUp => {
                            // A1 — decode_alloc_start_replay arms the table
                            // for the next forward's use.  V2-3b: immediately
                            // PAUSE so that allocations made between now and
                            // the start of the next forward (main loop
                            // sampler + `Tensor::new(tokens, device)`) go
                            // through the normal pool instead of consuming
                            // recorded slots.  The next forward calls
                            // `decode_alloc_resume()` at the top to re-enter
                            // replay mode for its own allocations — keeping
                            // the cursor aligned between recording #2 and
                            // all subsequent replays.
                            hdrv::decode_alloc_start_replay();
                            hdrv::decode_alloc_pause();
                            eprintln!(
                                "[G2-gemma4] token 3 recorded: {} ops, {} alloc entries, ptrs={:x?}",
                                recording.len(),
                                hdrv::decode_alloc_entry_count(),
                                cur_ptrs,
                            );
                            self.decode_state = DecodeState::Recorded1 {
                                ops: recording,
                                input_ptrs: cur_ptrs,
                                input_bytes: cur_bytes,
                            };
                        }
                        DecodeState::Recorded1 { ops: first, input_ptrs: prev_ptrs, input_bytes: prev_bytes } => {
                            let output_ptr = Self::hip_device_ptr(&result)?;
                            let output_f32_count = result.elem_count();
                            let output_shape = result.dims().to_vec();

                            // Build the per-input anchor list. `bytes`
                            // mirrors the live-tensor size; if it's
                            // changed between the two recordings (it
                            // shouldn't for decode), bail.
                            if prev_ptrs.len() != cur_ptrs.len()
                                || prev_bytes != cur_bytes
                            {
                                eprintln!("[G2-gemma4] input shape changed between recordings — aborting");
                                hdrv::decode_alloc_stop();
                                self.decode_state = DecodeState::Init;
                            } else {
                                let inputs: Vec<decode_cache::ExternalInput> = prev_ptrs
                                    .iter()
                                    .zip(cur_ptrs.iter())
                                    .zip(cur_bytes.iter())
                                    .map(|((&a, &b), &n)| decode_cache::ExternalInput {
                                        first_ptr: a,
                                        second_ptr: b,
                                        bytes: n,
                                    })
                                    .collect();

                                if let Some(mut plan) = decode_cache::DecodePlan::from_two_recordings_with_inputs(
                                    &first,
                                    &recording,
                                    output_ptr,
                                    output_f32_count,
                                    output_shape,
                                    &inputs,
                                ) {
                                    for (i, &b) in cur_bytes.iter().enumerate() {
                                        plan.set_input_anchor_bytes_at(i, b);
                                    }
                                    eprintln!(
                                        "[G2-gemma4] plan ready: {} ops, {} dynamic/{} fixed, {} external patches, {} input anchors",
                                        plan.len(),
                                        plan.dynamic_arg_count(),
                                        plan.fixed_arg_count(),
                                        plan.external_patch_count(),
                                        plan.input_count(),
                                    );
                                    for i in 0..plan.input_count() {
                                        if let Some(a) = plan.input_anchor(i) {
                                            eprintln!("[G2-gemma4]   anchor #{}: ptr=0x{:x} bytes={}", i, a.ptr, a.bytes);
                                        }
                                    }
                                    hdrv::decode_alloc_pause();
                                    hdrv::decode_alloc_reset();
                                    self.decode_state = DecodeState::Replay(plan);
                                } else {
                                    eprintln!("[G2-gemma4] plan build failed — recordings diverged");
                                    hdrv::decode_alloc_stop();
                                    self.decode_state = DecodeState::Init;
                                }
                            }
                        }
                        _ => {
                            hdrv::decode_alloc_stop();
                            self.decode_state = DecodeState::Init;
                        }
                    }
                }
            } else {
                // A1 — kernel recording moves to TOP of next forward (after
                // prelude).  But `decode_alloc_start_record` fires here
                // at the Init → WarmUp transition so the FIRST recorded
                // forward's prelude allocates through the alloc table too,
                // keeping the anchor tensors (layer_in, inp_per_layer, SWA
                // masks) at stable pool slots across both recordings.
                //
                // V2-5a: immediately pause after start_record so the
                // main-loop allocations that fire BETWEEN this forward
                // and the next (e.g. `Tensor::new(tokens, device)` in
                // main.rs before the next forward call) do NOT land in
                // the decode_alloc table.  Without this, the between-
                // forwards 4-byte input becomes entries[0] on dev 0
                // and desyncs rec #2's cursor on the first 11264-byte
                // layer_in alloc.  The next forward calls
                // decode_alloc_resume() at its top (Recorded1 or Replay
                // state handling at the top of forward), so in-forward
                // allocs still hit the table in their correct order.
                if matches!(&self.decode_state, DecodeState::Init) {
                    hdrv::decode_alloc_start_record();
                    hdrv::decode_alloc_pause();
                    self.decode_state = DecodeState::WarmUp;
                }
            }
        }

        #[cfg(feature = "hip")]
        if phase_time {
            let t_total = _t_forward_start.elapsed();
            let t_loop = _t_loop_end.duration_since(_t_fastpath_start);
            eprintln!(
                "[G2-time normal] li={:>5}us pe={:>5}us swa={:>5}us layers={:>5}us total={:>6}us",
                t_prelude_li.as_micros(), t_prelude_pe.as_micros(),
                t_prelude_swa.as_micros(), t_loop.as_micros(),
                t_total.as_micros(),
            );
        }
        Ok(result)
    }

    pub fn clear_kv_cache(&mut self) {
        for slot in self.kv_caches.iter_mut() {
            if let Some(c) = slot.as_mut() {
                c.reset();
            }
        }
        // Reset G2/G3 replay state too — KV cache invalidation makes any
        // captured plan stale.
        #[cfg(feature = "hip")]
        {
            self.decode_state = DecodeState::Init;
        }
    }

    /// Extract the raw HIP device pointer from a tensor's storage. Used by
    /// the G2/G3 replay path to anchor the per-token input tensor.
    #[cfg(feature = "hip")]
    fn hip_device_ptr(t: &Tensor) -> Result<usize> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            candle::Storage::Hip(hip) => {
                let base = hip.slice.device_ptr() as usize;
                let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
                Ok(base + offset_bytes)
            }
            _ => candle::bail!("expected HIP storage for decode replay"),
        }
    }

    /// Extract the HipDevice from a tensor.
    #[cfg(feature = "hip")]
    fn hip_device_from_tensor(t: &Tensor) -> Result<candle::hip_backend::HipDevice> {
        match t.device() {
            candle::Device::Hip(dev) => Ok(dev.clone()),
            _ => candle::bail!("expected HIP device for decode replay"),
        }
    }
}

// ---------------------------------------------------------------------------
// `Embedding::new_unused` – placeholder used during construction
// ---------------------------------------------------------------------------
//
// `ModelWeights::from_gguf_multi_device` builds the struct in two stages so we
// can move the dequantized embedding tensor in *after* the rest of the fields
// are populated. This avoids cloning the (potentially large) embedding twice.

trait EmbeddingUnused {
    fn new_unused() -> Self;
}

impl EmbeddingUnused for Embedding {
    fn new_unused() -> Self {
        let dev = candle::Device::Cpu;
        let zero = Tensor::zeros((1, 1), DType::F32, &dev).unwrap();
        Embedding::new(zero, 1)
    }
}
