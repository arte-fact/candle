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

#[cfg(feature = "hip")]
use candle::hip_backend::decode_cache::DecodeState;

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
    kv_caches: Vec<Option<crate::models::quantized_blocks::any_kv_cache::AnyKvCache>>,
    /// `CANDLE_KV_Q8=1`: use block_q8_0 packed KV instead of F32. Matches
    /// llama.cpp-turbo's per-context VRAM exactly (1.88× smaller than F16).
    use_q8_kv: bool,
    /// `CANDLE_KV_TQ={2,3,4}`: TurboQuant variant (2.5/4/4.25 bits/val).
    /// Takes precedence over use_q8_kv when set. None = don't use.
    /// NB: quality requires phase TQ3 (WHT + rotation) — without those
    /// centroids don't match the K/V distribution and output will drift.
    tq_variant: Option<u8>,
    /// G2/G3 decode replay state (opt-in via `CANDLE_G2_REPLAY=1`).
    #[cfg(feature = "hip")]
    decode_state: candle::hip_backend::decode_cache::DecodeState,
    /// Number of distinct HIP devices spanned by this model. Used to
    /// auto-select the decomposed attention path on single-GPU
    /// (+1.9 % tg on E4B 1-GPU; -1 % on 4-GPU MoE per Task #10
    /// validation). Computed once at load.
    #[cfg(feature = "hip")]
    n_devices: usize,
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

        // ----- build layers in parallel via a SCOPED rayon pool ------------
        // Each rayon worker pulls a cheap Gguf clone for its layer's device,
        // then loads attn / norms / FFN / per-layer-embed from the shared
        // mmap'd blob in one shot. With 4 MI50s and gemma4 31B (~17 GB Q4_0)
        // this drops model build time from ~52 s (sequential) to ~6 s.
        //
        // IMPORTANT: use a LOCAL `ThreadPool` scoped to this function, not
        // the global `into_par_iter()` pool. The global pool spawns
        // `available_parallelism()` workers (~32 on a 16c/32t host) that
        // persist for the process lifetime and spin on futex waits during
        // decode — measured ~440 s of sys CPU across a 35 s decode run
        // (13.8 cores of host overhead for GPU work). Limiting to
        // `devices.len() * 2` and dropping the pool after load collapses
        // the post-load thread count to 0 extra workers. +19 % tg64 on
        // gemma4-26B Q8_0 4-GPU.
        let n_upload_threads = (devices.len() * 2).max(2);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_upload_threads)
            .build()
            .map_err(|e| candle::Error::Msg(format!("rayon pool build: {e}")))?;
        let layers: Vec<LayerWeights> = pool.install(|| {
            (0..block_count)
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
                    let mut gate_inp = lgg.qmatmul(&format!("{block_prefix}.ffn_gate_inp.weight"))?;
                    // Gemma-4 ships `ffn_gate_inp.weight` as F32 in all Q*_0 /
                    // Q*_1 GGUFs.  F32 matmul lowers to rocBLAS gemv, which the
                    // steady-state profiler shows at ~25 % of total decode on
                    // 4-GPU 26B-A4B (7.3 ms / 29 ms summed across 30 layers).
                    // Requantising to Q8_0 forces the router onto the MMVQ
                    // kernel chain (same fix as O4 for Qwen3-Coder-Next,
                    // memory `project_phase_o2_o4_qwen3next.md`).  The router
                    // head is tiny ([hidden, 128]) so Q8_0 is essentially
                    // lossless.  Opt-out: `CANDLE_GEMMA4_ROUTER_NO_REQUANT=1`,
                    // override dtype: `CANDLE_GEMMA4_ROUTER_REQUANT_DTYPE=q4_0|q4_1|q5_0|q8_0`.
                    if std::env::var("CANDLE_GEMMA4_ROUTER_NO_REQUANT").is_err() {
                        let dtype = match std::env::var("CANDLE_GEMMA4_ROUTER_REQUANT_DTYPE")
                            .ok().as_deref()
                        {
                            Some("q4_0") => candle::quantized::GgmlDType::Q4_0,
                            Some("q4_1") => candle::quantized::GgmlDType::Q4_1,
                            Some("q5_0") => candle::quantized::GgmlDType::Q5_0,
                            _ => candle::quantized::GgmlDType::Q8_0,
                        };
                        let _ = gate_inp.requantize_to(dtype);
                    }
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
            .collect::<Result<Vec<_>>>()
        })?;
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
        let kv_caches: Vec<Option<crate::models::quantized_blocks::any_kv_cache::AnyKvCache>> =
            (0..block_count).map(|_| None).collect();

        #[cfg(feature = "hip")]
        let n_devices = devices.len();
        let use_q8_kv = std::env::var("CANDLE_KV_Q8").ok().as_deref() == Some("1");
        let tq_variant: Option<u8> = std::env::var("CANDLE_KV_TQ")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .filter(|n| matches!(n, 2 | 3 | 4));
        Ok(Self {
            tok_embeddings: Embedding::new_unused(),
            embedding_length,
            layers,
            norm,
            output,
            final_logit_softcap,
            per_layer_embeddings,
            kv_caches,
            use_q8_kv,
            tq_variant,
            #[cfg(feature = "hip")]
            decode_state: candle::hip_backend::decode_cache::DecodeState::Init,
            #[cfg(feature = "hip")]
            n_devices,
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
        self.g2_pre_forward_alloc_setup();

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
        let g2_eligible = self.g2_is_eligible(seq_len, index_pos);

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
        // G2-only SWA mask cache (one per device, padded to nkv_pad).
        // Keyed by `(device_location, sliding_window_size)`; anchored
        // into the captured plan so replays read from a stable buffer.
        // Decode (seq_len = 1) on global-attention layers skips the
        // mask entirely.
        #[cfg(feature = "hip")]
        let swa_masks: std::collections::HashMap<(String, usize), Tensor> = if g2_eligible {
            use std::collections::HashSet;
            let mut want: HashSet<(String, Option<usize>)> = HashSet::new();
            for l in &self.layers {
                let sw = l.sliding_window_size;
                if sw.is_some() {
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
                    0
                };
                let dtype = layer_in.dtype();
                candle::hip_backend::decode_cache::with_recording_paused(|| -> Result<_> {
                    let mut out = std::collections::HashMap::new();
                    for l in &self.layers {
                        let sw = l.sliding_window_size;
                        if sw.is_none() { continue; }
                        let key = (format!("{:?}", l.device.location()), sw.unwrap_or(0));
                        if out.contains_key(&key) { continue; }
                        let m = causal_mask_padded(
                            b_sz, seq_len, index_pos, sw, dtype, &l.device, pad_total,
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
        // ── G2/G3 fast path: replay captured kernel sequence ──
        #[cfg(feature = "hip")]
        if g2_eligible {
            if let Some(out) = self.try_g2_replay(
                index_pos,
                seq_len,
                &layer_in,
                inp_per_layer.as_ref(),
                &swa_mask_anchors,
                t_prelude_li,
                t_prelude_pe,
                t_prelude_swa,
                _t_forward_start,
            )? {
                return Ok(out);
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

        // T1 profiling: per-sublayer wall-time when CANDLE_LAYER_PROFILE=1.
        // Tallies (attention, ffn, tail) across all layers per forward.
        #[cfg(feature = "hip")]
        let layer_profile = std::env::var("CANDLE_LAYER_PROFILE").is_ok();
        #[cfg(feature = "hip")]
        let mut t_xdev_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_mask_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_attn_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_ffn_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_tail_us = 0u128;
        // MoE sub-phase tallies.
        #[cfg(feature = "hip")]
        let mut t_moe_dense_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_router_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_topk_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_gate_up_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_activated_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_down_us = 0u128;
        #[cfg(feature = "hip")]
        let mut t_moe_combine_us = 0u128;

        // Per-device runtime cache for the non-G2 causal-mask path.
        // Built lazily on first encounter of each (device, sw) pair so
        // device-setup ordering matches the prior per-layer behaviour
        // (building all masks up-front before the first layer kernel
        // broke module loading with hipErrorInvalidImage — see K7).
        #[cfg(feature = "hip")]
        let mut mask_runtime_cache: std::collections::HashMap<(String, usize), Tensor> =
            std::collections::HashMap::new();

        // ----- transformer block loop --------------------------------------
        #[cfg(feature = "hip")]
        let _t_layer_loop_start = std::time::Instant::now();
        for il in 0..self.layers.len() {
            let (has_kv, kv_source_idx, layer_device, sliding_window_size) = {
                let l = &self.layers[il];
                (l.has_kv, l.kv_source_idx, l.device.clone(), l.sliding_window_size)
            };

            #[cfg(feature = "hip")]
            let _t_xdev_start = std::time::Instant::now();
            // Pipeline-parallel: move residual stream to this layer's device.
            if !device_eq(layer_in.device(), &layer_device) {
                layer_in = layer_in.to_device(&layer_device)?;
            }
            #[cfg(feature = "hip")]
            if layer_profile {
                if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                t_xdev_us += _t_xdev_start.elapsed().as_micros();
            }
            #[cfg(feature = "hip")]
            let _t_mask_start = std::time::Instant::now();

            // Build the per-device causal mask (sliding-aware).
            //
            // Skip the mask for the trivial decode-on-global-layer case
            // (single token, no SWA): a 1-row causal mask is all zeros and the
            // attention is correct without it. For sliding layers we MUST build
            // a mask even at decode, otherwise the query attends to keys older
            // than the window — see llama.cpp `gemma4-iswa.cpp` SWA mask path.
            //
            // G2 path: reuse the precomputed mask from `swa_masks` so the
            // captured kernel reads from a single anchored buffer.
            // Non-G2 path: lazily populate & reuse `mask_runtime_cache`
            // per-device so the same mask isn't rebuilt on every layer
            // (measured 18 % of prefill on Gemma-4-26B-A4B 4-GPU).
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
                let key = (
                    format!("{:?}", layer_device.location()),
                    sliding_window_size.unwrap_or(0),
                );
                if let Some(m) = mask_runtime_cache.get(&key) {
                    Some(m.clone())
                } else {
                    let m = causal_mask(
                        b_sz,
                        seq_len,
                        index_pos,
                        sliding_window_size,
                        layer_in.dtype(),
                        &layer_device,
                    )?;
                    mask_runtime_cache.insert(key, m.clone());
                    Some(m)
                }
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

            #[cfg(feature = "hip")]
            if layer_profile {
                if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                t_mask_us += _t_mask_start.elapsed().as_micros();
            }
            #[cfg(feature = "hip")]
            let _t_attn_start = std::time::Instant::now();

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
                    // Lazy allocation of this layer's cache. Use AnyKvCache
                    // so the gate swaps the underlying storage between F32
                    // KvCache, block_q8_0 QKvCache, or one of the TurboQuant
                    // variants (TQ2/TQ3/TQ4 — ~6.4×/4×/3.76× smaller than F16).
                    if self.kv_caches[il].is_none() {
                        use crate::models::quantized_blocks::any_kv_cache::AnyKvCache;
                        #[cfg(feature = "hip")]
                        let new_cache = if let Some(tq) = self.tq_variant {
                            let (b0, h_kv0, _, d0) = k_new.dims4()?;
                            let hip_dev = match k_new.device() {
                                candle::Device::Hip(d) => d.clone(),
                                _ => candle::bail!("CANDLE_KV_TQ: k_new must be on HIP"),
                            };
                            use candle::hip_backend::turbo_kv_cache::{QTurboKvCache, TbVariant};
                            let variant = match tq {
                                2 => TbVariant::Tq2_0,
                                3 => TbVariant::Tq3_0,
                                4 => TbVariant::Tq4_0,
                                _ => candle::bail!("CANDLE_KV_TQ={tq}: only 2/3/4 supported"),
                            };
                            let t_max = std::env::var("CANDLE_KV_MAX_T")
                                .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
                            AnyKvCache::Tb(QTurboKvCache::new(
                                &hip_dev, variant, b0, h_kv0, t_max, d0,
                            )?)
                        } else if self.use_q8_kv {
                            let (b0, h_kv0, _, d0) = k_new.dims4()?;
                            let hip_dev = match k_new.device() {
                                candle::Device::Hip(d) => d.clone(),
                                _ => candle::bail!("CANDLE_KV_Q8: k_new must be on HIP"),
                            };
                            let t_max = std::env::var("CANDLE_KV_MAX_T")
                                .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
                            AnyKvCache::Q8(
                                candle::hip_backend::q8_kv_cache::QKvCache::new(
                                    &hip_dev, b0, h_kv0, t_max, d0,
                                )?,
                            )
                        } else if k_is_canonical {
                            AnyKvCache::F32(
                                candle_nn::kv_cache::KvCache::new_k_canonical_stable(2, 4096),
                            )
                        } else {
                            AnyKvCache::F32(
                                candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096),
                            )
                        };
                        #[cfg(not(feature = "hip"))]
                        let new_cache = if k_is_canonical {
                            AnyKvCache::F32(
                                candle_nn::kv_cache::KvCache::new_k_canonical_stable(2, 4096),
                            )
                        } else {
                            AnyKvCache::F32(
                                candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096),
                            )
                        };
                        self.kv_caches[il] = Some(new_cache);
                    }
                    let cache = self.kv_caches[il].as_mut().unwrap();
                    // KvCache::append needs contiguous sources.
                    let k_new = if k_new.is_contiguous() { k_new } else { k_new.contiguous()? };
                    let v_new = if v_new.is_contiguous() { v_new } else { v_new.contiguous()? };
                    // CANDLE_KV_F16: align K/V to the cache's target
                    // dtype. Under CANDLE_KV_F16, decode's producer
                    // kernels (rmsnorm_rope_decode_f16out_f32,
                    // v_norm_f16out_f32) already emit F16 — this cast
                    // is a no-op in that hot path. For fallback paths
                    // (e.g. prefill with b_size > 8 that hit
                    // compute_qkv_shared_q8 instead of
                    // compute_qkv_norm_q8_fused) it fires ONCE per
                    // prefill call and keeps the whole cache at one
                    // consistent dtype.
                    #[cfg(feature = "hip")]
                    let (k_new, v_new) = if std::env::var("CANDLE_KV_F16").ok().as_deref().map(|s| s != "0").unwrap_or(false) {
                        let k_new = if k_new.dtype() == candle::DType::F16 { k_new } else { k_new.to_dtype(candle::DType::F16)? };
                        let v_new = if v_new.dtype() == candle::DType::F16 { v_new } else { v_new.to_dtype(candle::DType::F16)? };
                        (k_new, v_new)
                    } else { (k_new, v_new) };
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
                let any_cache = self.kv_caches[src]
                    .as_ref()
                    .expect("KV cache must exist for has_kv source layer");
                let t_cur = any_cache.current_seq_len();
                let max_t = any_cache.max_seq_len();
                // Compressed caches (Q8 / TurboQuant) skip the
                // padded-slab narrow (no F32 all_data to share —
                // dequantising the whole buffer each step would defeat
                // the VRAM win). Padding is a G2-replay feature; G2 is
                // no-gain on gfx906 anyway.
                let use_padding = !any_cache.is_compressed() && pad_t > 1;
                let l_k_padded = if use_padding {
                    (((t_cur + pad_t - 1) / pad_t) * pad_t).min(max_t)
                } else {
                    t_cur
                };

                let (k, v, padded) = if let Some(f32_cache) = any_cache.as_f32() {
                    let k_cache = f32_cache.k_cache();
                    let v_cache = f32_cache.v_cache();
                    if l_k_padded > t_cur && seq_len_q == 1 {
                        let k_full = k_cache.all_data().as_ref()
                            .ok_or_else(|| candle::Error::Msg("gemma4 k cache empty".into()))?;
                        let v_full = v_cache.all_data().as_ref()
                            .ok_or_else(|| candle::Error::Msg("gemma4 v cache empty".into()))?;
                        let k = if k_is_canonical {
                            k_full.narrow(2, 0, l_k_padded)?
                        } else {
                            k_full.narrow(3, 0, l_k_padded)?
                        };
                        let v = v_full.narrow(2, 0, l_k_padded)?;
                        (k, v, Some((t_cur, l_k_padded)))
                    } else {
                        let k = f32_cache.k()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                        let v = f32_cache.v()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
                        (k, v, None)
                    }
                } else {
                    // Compressed arm (Q8 or TurboQuant): dequantise the
                    // slab into fresh F32 tensors in canonical
                    // (B, H_kv, t_cur, D) layout. Transient on top of
                    // the at-rest VRAM win.
                    let k = any_cache.canonical_k_f32()?
                        .ok_or_else(|| candle::Error::Msg("gemma4 compressed k cache empty".into()))?;
                    let v = any_cache.canonical_v_f32()?
                        .ok_or_else(|| candle::Error::Msg("gemma4 compressed v cache empty".into()))?;
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
                        // Task #2 + #10: decomposed attention (gemv_qk → softmax → gemv_v).
                        // Validated +1.9 % on E4B 1-GPU, -1 % on 4-GPU MoE (the extra
                        // kernel launches cross device boundaries, adding sync cost).
                        // Default heuristic: enable on single-GPU, disable on multi-GPU.
                        // CANDLE_ATTN_DECOMPOSED={0,1} overrides the heuristic.
                        let env_override = std::env::var("CANDLE_ATTN_DECOMPOSED").ok();
                        let use_decomposed = match env_override.as_deref() {
                            Some("0") | Some("false") => false,
                            Some(_) => true,
                            None => self.n_devices == 1,
                        };
                        if use_decomposed {
                            candle::hip_backend::gqa_attention_decode_mv_decomposed(
                                q_precomputed, &k_c, &v_c,
                                attention_mask.as_ref(),
                                self.layers[il].attn.attn_scale,
                                l_k_iter,
                            )?
                        } else {
                            candle::hip_backend::gqa_attention_decode_mv(
                                q_precomputed, &k_c, &v_c,
                                attention_mask.as_ref(),
                                self.layers[il].attn.attn_scale,
                                l_k_iter,
                            )?
                        }
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

            #[cfg(feature = "hip")]
            if layer_profile {
                if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                t_attn_us += _t_attn_start.elapsed().as_micros();
            }
            #[cfg(feature = "hip")]
            let _t_ffn_start = std::time::Instant::now();

            // -------- FFN block --------
            let skip_moe = std::env::var("CANDLE_GEMMA4_SKIP_MOE").is_ok();
            let mut x = if !skip_moe && self.layers[il].moe.is_some() {
                let moe_branch = self.layers[il].moe.as_ref().unwrap();
                // Dual dense+MoE: both branches read from `x` (the
                // attention residual output, a.k.a. attn_out in turbo).
                let attn_out = x;

                #[cfg(feature = "hip")]
                let _t_dense_start = std::time::Instant::now();

                // Branch 1: Dense MLP
                let dense_in = self.layers[il].ffn_norm.forward(&attn_out)?;
                let dense_out = self.layers[il].mlp.forward(&dense_in)?;
                let dense_normed = moe_branch.dense_post_norm.forward(&dense_out)?;

                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_dense_us += _t_dense_start.elapsed().as_micros();
                }
                #[cfg(feature = "hip")]
                let _t_router_start = std::time::Instant::now();

                // Branch 2: MoE experts
                let moe_in = moe_branch.pre_norm.forward(&attn_out)?;

                // Custom router logits:
                //   tmp = rms_norm(attn_out) / sqrt(n_embd) * gate_inp_s
                //   logits = gate_inp @ tmp
                //
                // T1.1: when on HIP with f32 contiguous attn_out, drop into
                // a single fused kernel `gemma4_router_prep_f32` instead of
                // the 5-launch (sqr+mean+add+sqrt+bdiv → mul → bmul) chain.
                // Opt-out via `CANDLE_GEMMA4_ROUTER_FUSED_OFF=1` for A/B
                // bisect.
                let router_input = {
                    #[cfg(feature = "hip")]
                    {
                        if matches!(attn_out.device(), candle::Device::Hip(_))
                            && attn_out.dtype() == candle::DType::F32
                            && attn_out.is_contiguous()
                            && moe_branch.gate_inp_s.dtype() == candle::DType::F32
                            && moe_branch.gate_inp_s.is_contiguous()
                            && std::env::var("CANDLE_GEMMA4_ROUTER_FUSED_OFF").is_err()
                        {
                            candle::hip_backend::gemma4_router_prep_fused(
                                &attn_out,
                                &moe_branch.gate_inp_s,
                                moe_branch.eps as f64,
                            )?
                        } else {
                            let normed = crate::models::quantized_blocks::norms::v_norm(
                                &attn_out,
                                moe_branch.eps as f64,
                            )?;
                            let scaled = (normed * (1.0 / (moe_branch.n_embd as f64).sqrt()))?;
                            scaled.broadcast_mul(&moe_branch.gate_inp_s)?
                        }
                    }
                    #[cfg(not(feature = "hip"))]
                    {
                        let normed = crate::models::quantized_blocks::norms::v_norm(
                            &attn_out,
                            moe_branch.eps as f64,
                        )?;
                        let scaled = (normed * (1.0 / (moe_branch.n_embd as f64).sqrt()))?;
                        scaled.broadcast_mul(&moe_branch.gate_inp_s)?
                    }
                };
                let router_logits = moe_branch.gate_inp.forward(&router_input)?;

                // T1.2: fused softmax + topk + normalise via single GPU
                // kernel. Replaces the 7-step chain (softmax → CPU
                // arg_sort → narrow → HtoD → gather → sum_keepdim →
                // bdiv) with one launch. Produces compact
                // `(tokens, topk)` ids + weights directly. Opt-out via
                // `CANDLE_GEMMA4_TOPK_FUSED_OFF=1` for A/B bisect.
                let topk_fused: Option<(Tensor, Tensor)> = {
                    #[cfg(feature = "hip")]
                    {
                        if matches!(router_logits.device(), candle::Device::Hip(_))
                            && router_logits.dtype() == candle::DType::F32
                            && router_logits.is_contiguous()
                            && std::env::var("CANDLE_GEMMA4_TOPK_FUSED_OFF").is_err()
                        {
                            // router_logits shape is (b, seq, n_experts); flatten to (b*seq, n_experts).
                            let n_experts_dim = router_logits.dim(candle::D::Minus1)?;
                            let flat = router_logits.reshape(((), n_experts_dim))?;
                            match candle::hip_backend::topk_moe_norm_fused(
                                &flat,
                                moe_branch.num_experts_used,
                            ) {
                                Ok(pair) => Some(pair),
                                Err(_) => None,
                            }
                        } else { None }
                    }
                    #[cfg(not(feature = "hip"))]
                    { None }
                };

                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_router_us += _t_router_start.elapsed().as_micros();
                }
                #[cfg(feature = "hip")]
                let _t_topk_start = std::time::Instant::now();

                // TopK selection. T1.2 fast path (above) emits compact
                // `(tokens, topk)` ids + weights directly; fallback is
                // the legacy CPU arg_sort chain.
                let device = routing_weights.device().clone();
                let (b_sz_ff, seq_len_ff, _hidden) = moe_in.dims3()?;
                let flat_tokens = b_sz_ff * seq_len_ff;
                let n_experts = routing_weights.dim(candle::D::Minus1)?;
                let rw_flat = routing_weights.reshape((flat_tokens, n_experts))?;
                // W3 step 4: only sync topk_ids to CPU when W3 gather
                // is enabled — otherwise the D2H per layer has no
                // consumer and adds a stream drain for nothing.
                #[cfg(feature = "hip")]
                let w3_gather_on = std::env::var("CANDLE_W3_GATHER")
                    .map_or(false, |v| v != "0");
                #[cfg(not(feature = "hip"))]
                let w3_gather_on = false;

                let (topk_ids, topk_weights) = if let Some((w, ids)) = topk_fused {
                    #[cfg(feature = "hip")]
                    {
                        if w3_gather_on {
                            let ids_cpu: Vec<u32> =
                                ids.flatten_all()?.to_vec1::<u32>()?;
                            candle::quantized::hip::w3_cached_ids_set(ids_cpu);
                        }
                    }
                    (ids, w)
                } else {
                    let topk_ids = {
                        let rw_cpu = rw_flat.to_device(&candle::Device::Cpu)?;
                        let ids = rw_cpu
                            .arg_sort_last_dim(false)?
                            .narrow(candle::D::Minus1, 0, moe_branch.num_experts_used)?
                            .contiguous()?;
                        // A3 / EPLB diagnostic — observe routing while ids live
                        // on CPU.  No-op when CANDLE_EPLB_PRINT / _DUMP unset.
                        let v: Vec<u32> = ids
                            .flatten_all()?
                            .to_vec1::<u32>()
                            .unwrap_or_default();
                        if !v.is_empty() {
                            crate::models::quantized_blocks::eplb::observe(&v, il);
                        }
                        #[cfg(feature = "hip")]
                        {
                            if w3_gather_on && !v.is_empty() {
                                candle::quantized::hip::w3_cached_ids_set(v);
                            }
                        }
                        ids.to_device(&device)?
                    };
                    let topk_weights = rw_flat.gather(&topk_ids, candle::D::Minus1)?;
                    let topk_weights = (&topk_weights
                        / topk_weights
                            .sum_keepdim(candle::D::Minus1)?
                            .broadcast_as(topk_weights.shape())?)?;
                    (topk_ids, topk_weights)
                };

                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_topk_us += _t_topk_start.elapsed().as_micros();
                }
                #[cfg(feature = "hip")]
                let _t_gate_up_start = std::time::Instant::now();

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
                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_gate_up_us += _t_gate_up_start.elapsed().as_micros();
                }
                #[cfg(feature = "hip")]
                let _t_activated_start = std::time::Instant::now();

                // W3d: fused `gelu(gate) * up → quantize_q8_1 → down`
                // replaces the 5-step narrow/gelu/bmul/contiguous/quant
                // chain with one HIP kernel launch + a preq8 down
                // MMVQ. Eliminates 4-5 small kernel launches per MoE
                // layer on decode. Opt out with CANDLE_W3D=0.
                // W3d eligibility: limit to decode-shape batch (≤ 8).
                // The fused kernel launches `batch * topk * inter_padded/32`
                // blocks per call; at prefill batch=512 this balloons to
                // 327k blocks and dominates the unfused Rust pipeline.
                // The unfused path scales better at large batch because
                // bulk ops like gelu / bmul / contiguous amortise launch
                // overhead over millions of elements per kernel.
                #[cfg(feature = "hip")]
                let w3d_on = matches!(gate_up.device(), candle::Device::Hip(_))
                    && gate_up.dtype() == candle::DType::F32
                    && gate_up.is_contiguous()
                    && moe_branch.expert_intermediate % 32 == 0
                    && gate_up.dim(0).unwrap_or(usize::MAX) <= 8
                    && std::env::var("CANDLE_W3D").as_deref() != Ok("0");
                #[cfg(not(feature = "hip"))]
                let w3d_on = false;

                let mut moe_out = if w3d_on {
                    #[cfg(feature = "hip")]
                    {
                        let (q8_buf, activated_shape) =
                            candle::hip_backend::gelu_mul_quantize_q8_1_fused(
                                &gate_up,
                                moe_branch.expert_intermediate,
                            )?;
                        if layer_profile {
                            if let candle::Device::Hip(d) = &layer_device {
                                let _ = d.stream().synchronize();
                            }
                            t_moe_activated_us +=
                                _t_activated_start.elapsed().as_micros();
                        }
                        let q8_view = q8_buf.slice(0..q8_buf.len());
                        let out = moe_branch.down_exps.indexed_moe_forward_preq8(
                            &q8_view,
                            &activated_shape,
                            &topk_ids,
                        )?;
                        drop(q8_buf);
                        out
                    }
                    #[cfg(not(feature = "hip"))]
                    unreachable!()
                } else {
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

                    #[cfg(feature = "hip")]
                    if layer_profile {
                        if let candle::Device::Hip(d) = &layer_device {
                            let _ = d.stream().synchronize();
                        }
                        t_moe_activated_us += _t_activated_start.elapsed().as_micros();
                    }

                    moe_branch
                        .down_exps
                        .indexed_moe_forward(&activated.contiguous()?, &topk_ids)?
                };
                #[cfg(feature = "hip")]
                let _t_down_start = std::time::Instant::now();
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

                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_down_us += _t_down_start.elapsed().as_micros();
                }
                #[cfg(feature = "hip")]
                let _t_combine_start = std::time::Instant::now();

                // Weight + sum across topk.
                //
                // W3a: prefer the fused `moe_combine_weighted_sum_fused`
                // kernel when inputs are HIP/f32/contiguous — it replaces
                // `broadcast_mul + sum(1)` (one bmul_f32 + one fast_sum_f32
                // per layer, = 358+ ms aggregate in the 2026-04-16 profile)
                // with a single launch and no [T, K, H] intermediate.
                // Env kill-switch: CANDLE_W3A_COMBINE=0 restores the old
                // unfused path for debugging / regression isolation.
                let moe_summed = {
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
                    let moe_out_c = moe_out.contiguous()?;
                    let topk_w_c = topk_weights.contiguous()?;
                    if try_fused {
                        #[cfg(feature = "hip")]
                        {
                            candle::hip_backend::moe_combine_weighted_sum_fused(
                                &moe_out_c,
                                &topk_w_c,
                            )?
                        }
                        #[cfg(not(feature = "hip"))]
                        unreachable!()
                    } else {
                        let topk_w_u = topk_w_c.unsqueeze(candle::D::Minus1)?;
                        let weighted = moe_out_c.broadcast_mul(&topk_w_u)?;
                        weighted.sum(1)?
                    }
                };
                let moe_summed = moe_summed.reshape((b_sz_ff, seq_len_ff, hidden))?;

                // W3 cache cleanup: ensure no stale topk_ids leak to
                // the next MoE layer.
                #[cfg(feature = "hip")]
                candle::quantized::hip::w3_cached_ids_clear();

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
                let result = self.layers[il]
                    .post_ffn_norm
                    .forward_post_residual(&combined, &attn_out)?;

                #[cfg(feature = "hip")]
                if layer_profile {
                    if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                    t_moe_combine_us += _t_combine_start.elapsed().as_micros();
                }
                result
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

            #[cfg(feature = "hip")]
            if layer_profile {
                if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                t_ffn_us += _t_ffn_start.elapsed().as_micros();
            }
            #[cfg(feature = "hip")]
            let _t_tail_start = std::time::Instant::now();

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

            #[cfg(feature = "hip")]
            if layer_profile {
                if let candle::Device::Hip(d) = &layer_device { let _ = d.stream().synchronize(); }
                t_tail_us += _t_tail_start.elapsed().as_micros();
            }

            layer_in = x;
        }

        #[cfg(feature = "hip")]
        if layer_profile {
            let tot = t_xdev_us + t_mask_us + t_attn_us + t_ffn_us + t_tail_us;
            eprintln!(
                "[layer-profile] xdev={} mask={} attn={} ffn={} tail={} total={} (µs, summed across {} layers)",
                t_xdev_us, t_mask_us, t_attn_us, t_ffn_us, t_tail_us, tot,
                self.layers.len(),
            );
            let moe_tot = t_moe_dense_us + t_moe_router_us + t_moe_topk_us
                + t_moe_gate_up_us + t_moe_activated_us + t_moe_down_us + t_moe_combine_us;
            if moe_tot > 0 {
                eprintln!(
                    "[moe-profile]   dense={} router={} topk={} gate_up={} activated={} down={} combine={} sum={} (µs)",
                    t_moe_dense_us, t_moe_router_us, t_moe_topk_us,
                    t_moe_gate_up_us, t_moe_activated_us, t_moe_down_us, t_moe_combine_us, moe_tot,
                );
            }
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

        // ── G2: Post-forward recording state machine ──
        #[cfg(feature = "hip")]
        if g2_eligible {
            self.g2_post_forward(
                &result,
                layer_in_anchor.as_ref(),
                inp_per_layer_anchor.as_ref(),
                &swa_mask_anchors,
            )?;
        }

        #[cfg(feature = "hip")]
        if phase_time {
            let t_total = _t_forward_start.elapsed();
            let t_loop = _t_loop_end.duration_since(_t_layer_loop_start);
            eprintln!(
                "[G2-time normal] li={:>5}us pe={:>5}us swa={:>5}us layers={:>5}us total={:>6}us",
                t_prelude_li.as_micros(), t_prelude_pe.as_micros(),
                t_prelude_swa.as_micros(), t_loop.as_micros(),
                t_total.as_micros(),
            );
        }
        Ok(result)
    }

    // ── G2/G3 private helpers (extracted from forward) ────────────────────

    /// Resume or arm the decode_alloc table at the start of forward()
    /// based on the current `decode_state`. Must run BEFORE the prelude
    /// so prelude allocations land in the correct pool slots.
    #[cfg(feature = "hip")]
    fn g2_pre_forward_alloc_setup(&self) {
        let trace = std::env::var("CANDLE_G2_ALLOC_TRACE").is_ok();
        match &self.decode_state {
            DecodeState::Replay(_) | DecodeState::Graph { .. } | DecodeState::Recorded1 { .. } => {
                candle::hip_backend::hipdarc::driver::decode_alloc_resume();
                if trace {
                    eprintln!(
                        "[G2-alloc] forward start (resume replaying): mode={:?}",
                        candle::hip_backend::hipdarc::driver::decode_alloc_get_mode()
                    );
                }
            }
            DecodeState::WarmUp => {
                candle::hip_backend::hipdarc::driver::decode_alloc_set_mode(
                    candle::hip_backend::hipdarc::driver::DecodeAllocMode::Recording,
                );
                if trace {
                    eprintln!(
                        "[G2-alloc] forward start (resume recording): mode={:?}",
                        candle::hip_backend::hipdarc::driver::decode_alloc_get_mode()
                    );
                }
            }
            DecodeState::Init => {}
        }
    }

    /// Check whether G2/G3 replay is eligible for this forward call.
    #[cfg(feature = "hip")]
    fn g2_is_eligible(&self, seq_len: usize, index_pos: usize) -> bool {
        seq_len == 1
            && index_pos > 0
            && {
                let single_device = {
                    let d0 = &self.layers[0].device;
                    self.layers.iter().all(|l| device_eq(&l.device, d0))
                };
                single_device || std::env::var("CANDLE_G2_MULTI_DEV").is_ok()
            }
            && std::env::var("CANDLE_G2_REPLAY").is_ok()
            && std::env::var("CANDLE_G2_DISABLE").is_err()
    }

    /// G2/G3 fast path: if the decode_state is Graph or Replay, execute
    /// the captured plan directly and return `Some(output)`.  Returns
    /// `None` when the state machine hasn't reached replay yet (WarmUp,
    /// Recorded1, Init) and the caller should fall through to the normal
    /// layer loop.
    ///
    /// # Arguments
    /// * `index_pos`, `seq_len` — token position info
    /// * `layer_in` — anchor #0 tensor
    /// * `inp_per_layer` — anchor #1 tensor (E4B only)
    /// * `swa_mask_anchors` — anchors #2+ (SWA masks, sorted)
    /// * `t_prelude_li`, `t_prelude_pe`, `t_prelude_swa` — phase timings
    /// * `forward_start` — instant the forward call began
    #[cfg(feature = "hip")]
    fn try_g2_replay(
        &mut self,
        index_pos: usize,
        seq_len: usize,
        layer_in: &Tensor,
        inp_per_layer: Option<&Tensor>,
        swa_mask_anchors: &[((String, usize), Tensor)],
        t_prelude_li: std::time::Duration,
        t_prelude_pe: std::time::Duration,
        t_prelude_swa: std::time::Duration,
        forward_start: std::time::Instant,
    ) -> Result<Option<Tensor>> {
        let phase_time = std::env::var("CANDLE_G2_PHASE_TIME").is_ok();
        let fastpath_start = std::time::Instant::now();

        // Diagnostic: dump anchor addresses on replay calls.
        if matches!(self.decode_state, DecodeState::Replay(_) | DecodeState::Graph { .. })
            && std::env::var("CANDLE_G2_REPLAY_TRACE").is_ok()
        {
            let head: Vec<f32> = layer_in
                .narrow(1, 0, 1)?
                .flatten_all()?
                .to_device(&candle::Device::Cpu)?
                .to_vec1()?;
            let head4: Vec<f32> = head.iter().take(4).copied().collect();
            eprintln!(
                "[G2-gemma4] replay layer_in ptr=0x{:x} first4={:?}",
                candle::hip_backend::decode_cache::hip_device_ptr(layer_in)?,
                head4
            );
        }

        // Build the per-call (input_idx, fresh_ptr) list for external patching.
        let fresh_inputs: Vec<(usize, usize)> = {
            let mut v = vec![(
                0usize,
                candle::hip_backend::decode_cache::hip_device_ptr(layer_in)?,
            )];
            if let Some(ipl) = inp_per_layer {
                v.push((1, candle::hip_backend::decode_cache::hip_device_ptr(ipl)?));
            }
            let mask_idx_base = v.len();
            for (i, (_key, t)) in swa_mask_anchors.iter().enumerate() {
                v.push((
                    mask_idx_base + i,
                    candle::hip_backend::decode_cache::hip_device_ptr(t)?,
                ));
            }
            v
        };

        let dev_for_replay =
            candle::hip_backend::decode_cache::hip_device_from_tensor(layer_in)?;

        // Helper closure: patch external arg slots to live pointers.
        let stage_inputs =
            |plan: &mut candle::hip_backend::decode_cache::DecodePlan,
             _dev: &candle::hip_backend::HipDevice|
             -> Result<()> {
                for &(idx, fresh_ptr) in &fresh_inputs {
                    if plan.input_count() == 0 {
                        plan.patch_all_externals(fresh_ptr);
                    } else {
                        plan.patch_external_input(idx, fresh_ptr);
                    }
                }
                Ok(())
            };

        match &mut self.decode_state {
            DecodeState::Graph {
                plan,
                graph,
                captured_l_k,
            } => {
                use candle::hip_backend::hipdarc;
                let current_t = index_pos + seq_len;
                let pad_default: usize =
                    if std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 };
                let pad_t = std::env::var("CANDLE_NKV_PAD")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(pad_default);
                let live_l_k = if pad_t > 1 {
                    ((current_t + pad_t - 1) / pad_t) * pad_t
                } else {
                    current_t
                };
                if live_l_k != *captured_l_k {
                    eprintln!(
                        "[G3-gemma4] pad boundary crossed (captured={} live={}), rebuilding",
                        captured_l_k, live_l_k
                    );
                    self.decode_state = DecodeState::Init;
                    return Ok(None);
                }
                hipdarc::driver::decode_alloc_resume();
                stage_inputs(plan, &dev_for_replay)?;
                if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                    plan.advance_counters();
                }
                candle::hip_backend::g3_counters::set(
                    &dev_for_replay,
                    candle::hip_backend::g3_counters::CounterSlot::LkIter,
                    current_t as u32,
                )?;
                plan.validate_inputs(&fresh_inputs)?;
                unsafe {
                    graph.patch_and_launch(plan, &dev_for_replay)?;
                }
                if std::env::var("CANDLE_G2_SYNC_REPLAY").is_ok() {
                    let _ = dev_for_replay.stream().synchronize();
                }
                hipdarc::driver::decode_alloc_pause();
                hipdarc::driver::decode_alloc_reset();
                let out = plan.output_tensor(&dev_for_replay)?;
                if phase_time {
                    let t_fp = fastpath_start.elapsed();
                    let t_total = forward_start.elapsed();
                    eprintln!(
                        "[G2-time graph] li={:>5}us pe={:>5}us swa={:>5}us fp={:>5}us total={:>6}us",
                        t_prelude_li.as_micros(),
                        t_prelude_pe.as_micros(),
                        t_prelude_swa.as_micros(),
                        t_fp.as_micros(),
                        t_total.as_micros(),
                    );
                }
                return Ok(Some(out));
            }
            DecodeState::Replay(plan) => {
                let max_replays = std::env::var("CANDLE_G2_REPLAY_MAX")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                if std::env::var("CANDLE_G2_REPLAY").is_ok()
                    && plan.replay_count() < max_replays
                {
                    use candle::hip_backend::hipdarc;
                    hipdarc::driver::decode_alloc_resume();
                    stage_inputs(plan, &dev_for_replay)?;
                    if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                        plan.advance_counters();
                    }
                    let current_t_for_ctr = index_pos + seq_len;
                    candle::hip_backend::g3_counters::set(
                        &dev_for_replay,
                        candle::hip_backend::g3_counters::CounterSlot::LkIter,
                        current_t_for_ctr as u32,
                    )?;
                    plan.validate_inputs(&fresh_inputs)?;
                    if std::env::var("CANDLE_G2_ALLOC_CURSOR_TRACE").is_ok() {
                        let c =
                            candle::hip_backend::hipdarc::driver::decode_alloc_cursors_by_device();
                        eprintln!("[V2-3] pre-replay cursors-by-dev: {:?}", c);
                        let h =
                            candle::hip_backend::hipdarc::driver::decode_alloc_head_entries(5);
                        for (dev, head) in h {
                            eprintln!(
                                "[V2-3] dev {} head entries: {:?}",
                                dev,
                                head.iter()
                                    .map(|(s, p)| format!("({}, 0x{:x})", s, p))
                                    .collect::<Vec<_>>(),
                            );
                        }
                    }
                    unsafe {
                        plan.replay(&dev_for_replay)?;
                    }
                    if std::env::var("CANDLE_G2_SYNC_REPLAY").is_ok() {
                        let _ = dev_for_replay.stream().synchronize();
                    }
                    // K10/K13 diagnostics
                    if std::env::var("CANDLE_G2_REPLAY_TRACE").is_ok() {
                        let _ = dev_for_replay.stream().synchronize();
                        let out = plan.output_tensor(&dev_for_replay)?;
                        let head: Vec<f32> = out
                            .narrow(
                                out.rank() - 1,
                                0,
                                8.min(out.dim(out.rank() - 1)?),
                            )?
                            .flatten_all()?
                            .to_vec1()?;
                        eprintln!(
                            "[G2-gemma4] replay#{} idx={} output head: {:?}",
                            plan.replay_count(),
                            index_pos,
                            head
                        );
                        if let Some(ipl) = inp_per_layer {
                            let ipl_head: Vec<f32> = ipl
                                .narrow(
                                    ipl.rank() - 1,
                                    0,
                                    4.min(ipl.dim(ipl.rank() - 1)?),
                                )?
                                .flatten_all()?
                                .to_vec1()?;
                            eprintln!(
                                "[G2-gemma4] replay#{} inp_per_layer ptr=0x{:x} head4={:?}",
                                plan.replay_count(),
                                candle::hip_backend::decode_cache::hip_device_ptr(ipl)?,
                                ipl_head
                            );
                        }
                    }
                    if std::env::var("CANDLE_G2_PEEK_OPS").is_ok()
                        && plan.replay_count() <= 2
                    {
                        let peeks = plan.peek_op_outputs(&dev_for_replay, 4);
                        for (i, name, head) in peeks {
                            eprintln!(
                                "[G2-peek r{}] op[{:>4}] {:>40} {}",
                                plan.replay_count(),
                                i,
                                name,
                                match head {
                                    Some(v) => format!("head={:?}", v),
                                    None => "(unknown output arg)".to_string(),
                                }
                            );
                        }
                    }
                    if std::env::var("CANDLE_G2_PEEK_ARGS").is_ok()
                        && plan.replay_count() == 1
                    {
                        for (i, name, args, sizes) in plan.peek_op_args() {
                            let arg_strs: Vec<String> = args
                                .iter()
                                .zip(sizes.iter())
                                .map(|(v, sz): (&u64, &usize)| {
                                    if *sz == 8 {
                                        format!("0x{:012x}", v)
                                    } else {
                                        format!("{}", *v as i64)
                                    }
                                })
                                .collect();
                            eprintln!(
                                "[G2-args] op[{:>4}] {:>40} args=[{}]",
                                i,
                                name,
                                arg_strs.join(", ")
                            );
                        }
                    }
                    let g3_enabled = std::env::var("CANDLE_G3_GRAPH").is_ok();
                    let g3_after = std::env::var("CANDLE_G3_AFTER")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(2);
                    let should_capture =
                        g3_enabled && plan.replay_count() == g3_after;
                    if should_capture {
                        let current_t = index_pos + seq_len;
                        let pad_t_env: usize = std::env::var("CANDLE_NKV_PAD")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(
                                if g3_enabled
                                    || std::env::var("CANDLE_G2_REPLAY").is_ok()
                                {
                                    256
                                } else {
                                    1
                                },
                            );
                        let live_l_k = if pad_t_env > 1 {
                            ((current_t + pad_t_env - 1) / pad_t_env) * pad_t_env
                        } else {
                            current_t
                        };
                        eprintln!(
                            "[G3-gemma4] capturing graph after {} replays (L_k_padded={})",
                            g3_after, live_l_k
                        );
                        hipdarc::driver::decode_alloc_pause();
                        hipdarc::driver::decode_alloc_reset();
                        let output = plan.output_tensor(&dev_for_replay)?;
                        let old = std::mem::replace(
                            &mut self.decode_state,
                            DecodeState::Init,
                        );
                        if let DecodeState::Replay(plan) = old {
                            match unsafe {
                                candle::hip_backend::decode_cache::DecodeGraph::capture(
                                    &plan,
                                    &dev_for_replay,
                                )
                            } {
                                Ok(graph) => {
                                    eprintln!(
                                        "[G3-gemma4] graph captured successfully"
                                    );
                                    self.decode_state = DecodeState::Graph {
                                        plan,
                                        graph,
                                        captured_l_k: live_l_k,
                                    };
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[G3-gemma4] graph capture failed: {:?} — staying in Replay",
                                        e
                                    );
                                    self.decode_state =
                                        DecodeState::Replay(plan);
                                }
                            }
                        }
                        return Ok(Some(output));
                    }
                    hipdarc::driver::decode_alloc_pause();
                    hipdarc::driver::decode_alloc_reset();
                    let out = plan.output_tensor(&dev_for_replay)?;
                    if phase_time {
                        let t_fp = fastpath_start.elapsed();
                        let t_total = forward_start.elapsed();
                        eprintln!(
                            "[G2-time replay] li={:>5}us pe={:>5}us swa={:>5}us fp={:>5}us total={:>6}us",
                            t_prelude_li.as_micros(),
                            t_prelude_pe.as_micros(),
                            t_prelude_swa.as_micros(),
                            t_fp.as_micros(),
                            t_total.as_micros(),
                        );
                    }
                    return Ok(Some(out));
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Post-forward G2 recording state machine. Handles the Init→WarmUp→
    /// Recorded1→Replay transitions, capturing kernel recordings and
    /// building the `DecodePlan` from two consecutive forward calls.
    #[cfg(feature = "hip")]
    fn g2_post_forward(
        &mut self,
        result: &Tensor,
        layer_in_anchor: Option<&Tensor>,
        inp_per_layer_anchor: Option<&Tensor>,
        swa_mask_anchors: &[((String, usize), Tensor)],
    ) -> Result<()> {
        use candle::hip_backend::decode_cache;
        use candle::hip_backend::hipdarc::driver as hdrv;

        let in_anchor = layer_in_anchor.unwrap();

        if decode_cache::is_recording() {
            if let Some(recording) = decode_cache::stop_recording() {
                // Snapshot per-input pointers + sizes.
                let mut cur_ptrs: Vec<usize> =
                    vec![decode_cache::hip_device_ptr(in_anchor)?];
                let mut cur_bytes: Vec<usize> =
                    vec![in_anchor.elem_count() * in_anchor.dtype().size_in_bytes()];
                if let Some(t) = inp_per_layer_anchor {
                    cur_ptrs.push(decode_cache::hip_device_ptr(t)?);
                    cur_bytes.push(t.elem_count() * t.dtype().size_in_bytes());
                }
                for (_key, t) in swa_mask_anchors {
                    cur_ptrs.push(decode_cache::hip_device_ptr(t)?);
                    cur_bytes.push(t.elem_count() * t.dtype().size_in_bytes());
                }

                match std::mem::replace(&mut self.decode_state, DecodeState::Init) {
                    DecodeState::WarmUp => {
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
                    DecodeState::Recorded1 {
                        ops: first,
                        input_ptrs: prev_ptrs,
                        input_bytes: prev_bytes,
                    } => {
                        let output_ptr = decode_cache::hip_device_ptr(result)?;
                        let output_f32_count = result.elem_count();
                        let output_shape = result.dims().to_vec();

                        if prev_ptrs.len() != cur_ptrs.len() || prev_bytes != cur_bytes {
                            eprintln!(
                                "[G2-gemma4] input shape changed between recordings — aborting"
                            );
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

                            if let Some(mut plan) =
                                decode_cache::DecodePlan::from_two_recordings_with_inputs(
                                    &first,
                                    &recording,
                                    output_ptr,
                                    output_f32_count,
                                    output_shape,
                                    &inputs,
                                )
                            {
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
                                        eprintln!(
                                            "[G2-gemma4]   anchor #{}: ptr=0x{:x} bytes={}",
                                            i, a.ptr, a.bytes
                                        );
                                    }
                                }
                                hdrv::decode_alloc_pause();
                                hdrv::decode_alloc_reset();
                                self.decode_state = DecodeState::Replay(plan);
                            } else {
                                eprintln!(
                                    "[G2-gemma4] plan build failed — recordings diverged"
                                );
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
            // Init → WarmUp: arm the decode_alloc table for recording.
            if matches!(&self.decode_state, DecodeState::Init) {
                hdrv::decode_alloc_start_record();
                hdrv::decode_alloc_pause();
                self.decode_state = DecodeState::WarmUp;
            }
        }
        Ok(())
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
            self.decode_state = candle::hip_backend::decode_cache::DecodeState::Init;
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
