//! GGUF metadata-driven model configuration.
//!
//! Reads all model parameters from GGUF metadata using the `{arch}.` prefix convention.
//! No hardcoded values — everything is derived from what's in the file.

use candle::quantized::gguf_file;
use candle::{DType, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PerLayer<T>: uniform or per-layer array values from GGUF metadata
// ---------------------------------------------------------------------------

/// A value that can be uniform across all layers or vary per-layer.
/// GGUF metadata stores some fields as scalars and others as arrays.
#[derive(Debug, Clone)]
pub enum PerLayer<T> {
    Uniform(T),
    Array(Vec<T>),
}

impl<T: Clone> PerLayer<T> {
    /// Get the value for a specific layer index.
    pub fn get(&self, layer: usize) -> &T {
        match self {
            PerLayer::Uniform(v) => v,
            PerLayer::Array(v) => &v[layer],
        }
    }

    /// Get the uniform value, or the first element if per-layer.
    pub fn first(&self) -> &T {
        match self {
            PerLayer::Uniform(v) => v,
            PerLayer::Array(v) => &v[0],
        }
    }
}

/// Read a GGUF Value as a usize, handling multiple integer types.
fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    use gguf_file::Value::*;
    match v {
        U8(x) => Ok(*x as usize),
        I8(x) => Ok(*x as usize),
        U16(x) => Ok(*x as usize),
        I16(x) => Ok(*x as usize),
        U32(x) => Ok(*x as usize),
        I32(x) => Ok(*x as usize),
        U64(x) => Ok(*x as usize),
        I64(x) => Ok(*x as usize),
        F32(x) => Ok(*x as usize),
        F64(x) => Ok(*x as usize),
        Bool(x) => Ok(usize::from(*x)),
        _ => candle::bail!("cannot convert metadata value to usize: {v:?}"),
    }
}

/// Read a GGUF Value as PerLayer<usize> — handles both scalar and array metadata.
pub fn read_per_layer_usize(v: &gguf_file::Value, expected_len: usize) -> Result<PerLayer<usize>> {
    use gguf_file::Value::Array;
    match v {
        Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                out.push(value_to_usize(item)?);
            }
            if out.len() == expected_len {
                Ok(PerLayer::Array(out))
            } else if out.len() == 1 {
                Ok(PerLayer::Uniform(out[0]))
            } else {
                candle::bail!(
                    "per-layer array length mismatch: expected {expected_len}, got {}",
                    out.len()
                )
            }
        }
        _ => Ok(PerLayer::Uniform(value_to_usize(v)?)),
    }
}

/// Read a GGUF Value as Vec<usize> (for fixed-length arrays like rope_sections).
fn read_usize_vec(v: &gguf_file::Value) -> Result<Vec<usize>> {
    use gguf_file::Value::Array;
    match v {
        Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                out.push(value_to_usize(item)?);
            }
            Ok(out)
        }
        _ => Ok(vec![value_to_usize(v)?]),
    }
}

// ---------------------------------------------------------------------------
// Layer type detection via tensor probing
// ---------------------------------------------------------------------------

/// What kind of attention block a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Gated Delta Net / linear attention (qwen35 recurrent layers)
    DeltaNet,
    /// Standard GQA with separate Q/K/V projections (llama, gemma4, etc.)
    Standard,
    /// No attention in this block (pure SSM, pure MoE like nemotron odd blocks)
    None,
}

/// What kind of FFN block a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    /// Standard SwiGLU dense MLP
    Dense,
    /// Mixture of Experts (with optional shared expert)
    Moe,
    /// Dense MLP + MoE in parallel (gemma4 MoE layers)
    DualPath,
    /// No FFN in this block
    None,
}

/// Detected layer composition from GGUF tensor probing.
#[derive(Debug, Clone)]
pub struct LayerKind {
    pub attn: AttnKind,
    pub ffn: FfnKind,
    /// Full attention layers in qwen35 have Q+gate fused (attn_q outputs 2x head_dim)
    pub has_gated_q: bool,
    /// MoE layers with a shared expert path (ffn_*_shexp tensors)
    pub has_shared_expert: bool,
    /// Post-attention norm exists (qwen35, gemma4)
    pub has_post_attn_norm: bool,
    /// Post-FFN norm exists (gemma4)
    pub has_post_ffn_norm: bool,
    /// Layer output scale exists (gemma4)
    pub has_layer_output_scale: bool,
    /// QK norms exist (gemma4, qwen35 full-attn)
    pub has_qk_norms: bool,
}

/// Detect what type of layer block `il` is based on which tensors exist in the GGUF file.
pub fn detect_layer_kind(tensor_infos: &HashMap<String, gguf_file::TensorInfo>, il: usize) -> LayerKind {
    let prefix = format!("blk.{il}");
    let has = |name: &str| tensor_infos.contains_key(&format!("{prefix}.{name}"));

    let attn = if has("ssm_a") {
        AttnKind::DeltaNet
    } else if has("attn_q.weight") || has("attn_qkv.weight") {
        AttnKind::Standard
    } else {
        AttnKind::None
    };

    let has_moe_experts = has("ffn_gate_exps.weight") || has("ffn_gate_up_exps.weight");
    let has_dense_mlp = has("ffn_gate.weight");

    let ffn = if has_moe_experts && has_dense_mlp {
        FfnKind::DualPath
    } else if has_moe_experts || has("ffn_gate_inp.weight") {
        FfnKind::Moe
    } else if has_dense_mlp {
        FfnKind::Dense
    } else {
        FfnKind::None
    };

    LayerKind {
        attn,
        ffn,
        has_gated_q: has("attn_q.weight") && has("attn_q_norm.weight") && !has("ssm_a"),
        has_shared_expert: has("ffn_up_shexp.weight"),
        has_post_attn_norm: has("post_attention_norm.weight"),
        has_post_ffn_norm: has("post_ffw_norm.weight"),
        has_layer_output_scale: has("layer_output_scale.weight"),
        has_qk_norms: has("attn_q_norm.weight") || has("attn_k_norm.weight"),
    }
}

// ---------------------------------------------------------------------------
// GgufConfig: all model parameters from GGUF metadata
// ---------------------------------------------------------------------------

/// All model configuration derived from GGUF metadata.
/// No hardcoded values — everything read from the file.
#[derive(Debug, Clone)]
pub struct GgufConfig {
    /// Architecture string from `general.architecture` (e.g. "qwen35", "gemma4")
    pub arch: String,
    /// Hidden size: `{arch}.embedding_length`
    pub hidden_size: usize,
    /// Number of transformer blocks: `{arch}.block_count`
    pub block_count: usize,

    // -- Attention --
    /// Number of query heads: `{arch}.attention.head_count`
    pub head_count: usize,
    /// Number of KV heads per layer: `{arch}.attention.head_count_kv` (scalar or array)
    pub head_count_kv: PerLayer<usize>,
    /// Key/query head dimension: `{arch}.attention.key_length` (or hidden/heads)
    pub head_dim: usize,
    /// Value head dimension: `{arch}.attention.value_length` (defaults to head_dim)
    pub value_length: usize,
    /// RMS norm epsilon: `{arch}.attention.layer_norm_rms_epsilon`
    pub rms_norm_eps: f64,
    /// Attention scale override: `{arch}.attention.scale`
    pub attention_scale: Option<f64>,

    // -- RoPE --
    /// RoPE frequency base: `{arch}.rope.freq_base`
    pub rope_freq_base: Option<f64>,
    /// How many dims RoPE applies to: `{arch}.rope.dimension_count`
    pub rope_dimension_count: Option<usize>,
    /// Multi-frequency sections: `{arch}.rope.dimension_sections`
    pub rope_sections: Option<Vec<usize>>,

    // -- GDN / recurrent (None for pure-attention models) --
    /// SSM inner dimension: `{arch}.ssm.inner_size`
    pub ssm_d_inner: Option<usize>,
    /// SSM state dimension: `{arch}.ssm.state_size`
    pub ssm_d_state: Option<usize>,
    /// SSM group count: `{arch}.ssm.group_count`
    pub ssm_n_group: Option<usize>,
    /// SSM time step rank: `{arch}.ssm.time_step_rank`
    pub ssm_dt_rank: Option<usize>,
    /// SSM conv kernel size: `{arch}.ssm.conv_kernel`
    pub ssm_conv_kernel: Option<usize>,
    /// Every N-th layer is full attention: `{arch}.full_attention_interval`
    pub full_attention_interval: Option<usize>,

    // -- MoE (None for dense models) --
    /// Total expert count: `{arch}.expert_count`
    pub expert_count: Option<usize>,
    /// Experts used per token: `{arch}.expert_used_count`
    pub expert_used_count: Option<usize>,

    // -- Sliding window --
    /// Sliding window size: `{arch}.attention.sliding_window`
    pub sliding_window: Option<usize>,
    /// Sliding window pattern period: `{arch}.attention.sliding_window_type`
    pub sliding_window_type: Option<usize>,
    /// Per-layer sliding window pattern: `{arch}.attention.sliding_window_pattern`
    /// Bool array, true = sliding (local) attention, false = full attention
    pub sliding_window_pattern: Option<Vec<bool>>,

    // -- Misc --
    /// Feed-forward intermediate size: `{arch}.feed_forward_length`
    pub feed_forward_length: Option<usize>,
    /// Max context length: `{arch}.context_length`
    pub context_length: Option<usize>,
    /// Final logit softcap value (Gemma): `{arch}.final_logit_softcapping`
    pub final_logit_softcap: Option<f64>,
    /// Attention logit softcap value (Gemma2): `{arch}.attn_logit_softcapping`
    pub attn_logit_softcap: Option<f64>,
    /// Sliding window RoPE freq base: `{arch}.rope.freq_base_swa`
    pub rope_freq_base_swa: Option<f64>,
    /// Sliding window RoPE dimension count: `{arch}.rope.dimension_count_swa`
    pub rope_dimension_count_swa: Option<usize>,

    /// Compute dtype for activations
    pub dtype: DType,
}

impl GgufConfig {
    /// Read all configuration from GGUF metadata. No hardcoded values.
    pub fn from_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Result<Self> {
        let arch = metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Helper to read a metadata key with the arch prefix
        let md = |key: &str| -> Option<&gguf_file::Value> {
            metadata.get(&format!("{arch}.{key}"))
        };
        let md_req = |key: &str| -> Result<&gguf_file::Value> {
            md(key).ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata: {arch}.{key}")))
        };

        let hidden_size = md_req("embedding_length")?.to_u32()? as usize;
        let block_count = md_req("block_count")?.to_u32()? as usize;
        let head_count = md_req("attention.head_count")?.to_u32()? as usize;

        let head_count_kv = read_per_layer_usize(
            md_req("attention.head_count_kv")?,
            block_count,
        )?;

        let head_dim = md("attention.key_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / head_count);

        let value_length = md("attention.value_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rms_norm_eps = md_req("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let attention_scale = md("attention.scale")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);

        let rope_freq_base = md("rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let rope_dimension_count = md("rope.dimension_count")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let rope_sections = md("rope.dimension_sections")
            .and_then(|v| read_usize_vec(v).ok());

        let ssm_d_inner = md("ssm.inner_size").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let ssm_d_state = md("ssm.state_size").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let ssm_n_group = md("ssm.group_count").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let ssm_dt_rank = md("ssm.time_step_rank").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let ssm_conv_kernel = md("ssm.conv_kernel").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let full_attention_interval = md("full_attention_interval")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);

        let expert_count = md("expert_count").and_then(|v| v.to_u32().ok()).map(|v| v as usize);
        let expert_used_count = md("expert_used_count").and_then(|v| v.to_u32().ok()).map(|v| v as usize);

        let sliding_window = md("attention.sliding_window")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let sliding_window_type = md("attention.sliding_window_type")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let sliding_window_pattern = md("attention.sliding_window_pattern")
            .and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    let mut bools = Vec::with_capacity(arr.len());
                    for item in arr {
                        let b = match item {
                            gguf_file::Value::Bool(b) => *b,
                            gguf_file::Value::U8(x) => *x != 0,
                            _ => return None,
                        };
                        bools.push(b);
                    }
                    Some(bools)
                }
                _ => None,
            });

        let feed_forward_length = md("feed_forward_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let context_length = md("context_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);

        let final_logit_softcap = md("final_logit_softcapping")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let attn_logit_softcap = md("attn_logit_softcapping")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let rope_freq_base_swa = md("rope.freq_base_swa")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let rope_dimension_count_swa = md("rope.dimension_count_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);

        let dtype = match metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        Ok(Self {
            arch,
            hidden_size,
            block_count,
            head_count,
            head_count_kv,
            head_dim,
            value_length,
            rms_norm_eps,
            attention_scale,
            rope_freq_base,
            rope_dimension_count,
            rope_sections,
            ssm_d_inner,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_conv_kernel,
            full_attention_interval,
            expert_count,
            expert_used_count,
            sliding_window,
            sliding_window_type,
            sliding_window_pattern,
            feed_forward_length,
            context_length,
            final_logit_softcap,
            attn_logit_softcap,
            rope_freq_base_swa,
            rope_dimension_count_swa,
            dtype,
        })
    }

    /// Is layer `il` a recurrent (GDN) layer vs full attention?
    /// Based on `full_attention_interval`: recurrent layers are all except every N-th.
    pub fn is_recurrent(&self, il: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) if interval > 0 => (il + 1) % interval != 0,
            _ => false,
        }
    }

    /// Is layer `il` a sliding window layer? (gemma4 ISWA pattern)
    /// Checks `sliding_window_pattern` array first, falls back to `sliding_window_type`.
    pub fn is_sliding(&self, il: usize) -> bool {
        // Prefer the explicit per-layer pattern array
        if let Some(ref pattern) = self.sliding_window_pattern {
            if il < pattern.len() {
                return pattern[il];
            }
        }
        // Fall back to modulo pattern
        match self.sliding_window_type {
            Some(swt) if swt > 0 => (il + 1) % swt != 0,
            _ => false,
        }
    }

    /// Number of recurrent (GDN) layers in the model.
    pub fn num_recurrent_layers(&self) -> usize {
        (0..self.block_count).filter(|&il| self.is_recurrent(il)).count()
    }

    /// GDN conv channels: d_inner + 2 * n_group * d_state
    pub fn gdn_conv_channels(&self) -> usize {
        let d_inner = self.ssm_d_inner.unwrap_or(0);
        let n_group = self.ssm_n_group.unwrap_or(0);
        let d_state = self.ssm_d_state.unwrap_or(0);
        d_inner + 2 * n_group * d_state
    }

    /// GDN head_v_dim: d_inner / dt_rank
    pub fn gdn_head_v_dim(&self) -> usize {
        let d_inner = self.ssm_d_inner.unwrap_or(1);
        let dt_rank = self.ssm_dt_rank.unwrap_or(1);
        d_inner / dt_rank
    }

    /// Default attention scale: 1/sqrt(head_dim)
    pub fn default_attention_scale(&self) -> f64 {
        self.attention_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }

    /// Max sequence length for RoPE precomputation.
    pub fn max_seq_len(&self) -> usize {
        self.context_length.unwrap_or(131072).min(131072)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_layer_uniform() {
        let pl = PerLayer::Uniform(42usize);
        assert_eq!(*pl.get(0), 42);
        assert_eq!(*pl.get(99), 42);
        assert_eq!(*pl.first(), 42);
    }

    #[test]
    fn test_per_layer_array() {
        let pl = PerLayer::Array(vec![10, 20, 30]);
        assert_eq!(*pl.get(0), 10);
        assert_eq!(*pl.get(1), 20);
        assert_eq!(*pl.get(2), 30);
        assert_eq!(*pl.first(), 10);
    }

    #[test]
    fn test_read_per_layer_usize_scalar() {
        let v = gguf_file::Value::U32(8);
        let result = read_per_layer_usize(&v, 10).unwrap();
        match result {
            PerLayer::Uniform(val) => assert_eq!(val, 8),
            PerLayer::Array(_) => panic!("expected uniform"),
        }
    }

    #[test]
    fn test_read_per_layer_usize_array() {
        let v = gguf_file::Value::Array(vec![
            gguf_file::Value::U32(4),
            gguf_file::Value::U32(8),
            gguf_file::Value::U32(4),
        ]);
        let result = read_per_layer_usize(&v, 3).unwrap();
        match result {
            PerLayer::Array(vals) => assert_eq!(vals, vec![4, 8, 4]),
            PerLayer::Uniform(_) => panic!("expected array"),
        }
    }

    #[test]
    fn test_read_per_layer_usize_single_element_array() {
        let v = gguf_file::Value::Array(vec![gguf_file::Value::U32(16)]);
        let result = read_per_layer_usize(&v, 10).unwrap();
        match result {
            PerLayer::Uniform(val) => assert_eq!(val, 16),
            PerLayer::Array(_) => panic!("expected uniform from single-element array"),
        }
    }

    #[test]
    fn test_read_per_layer_usize_length_mismatch() {
        let v = gguf_file::Value::Array(vec![
            gguf_file::Value::U32(1),
            gguf_file::Value::U32(2),
        ]);
        assert!(read_per_layer_usize(&v, 5).is_err());
    }

    fn make_qwen35_metadata() -> HashMap<String, gguf_file::Value> {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), gguf_file::Value::String("qwen35".into()));
        m.insert("qwen35.embedding_length".into(), gguf_file::Value::U32(5120));
        m.insert("qwen35.block_count".into(), gguf_file::Value::U32(64));
        m.insert("qwen35.attention.head_count".into(), gguf_file::Value::U32(24));
        m.insert("qwen35.attention.head_count_kv".into(), gguf_file::Value::U32(4));
        m.insert("qwen35.attention.key_length".into(), gguf_file::Value::U32(256));
        m.insert("qwen35.attention.layer_norm_rms_epsilon".into(), gguf_file::Value::F32(1e-6));
        m.insert("qwen35.rope.freq_base".into(), gguf_file::Value::F32(1e7));
        m.insert("qwen35.rope.dimension_count".into(), gguf_file::Value::U32(64));
        m.insert("qwen35.full_attention_interval".into(), gguf_file::Value::U32(4));
        m.insert("qwen35.ssm.inner_size".into(), gguf_file::Value::U32(6144));
        m.insert("qwen35.ssm.state_size".into(), gguf_file::Value::U32(128));
        m.insert("qwen35.ssm.group_count".into(), gguf_file::Value::U32(16));
        m.insert("qwen35.ssm.time_step_rank".into(), gguf_file::Value::U32(48));
        m.insert("qwen35.ssm.conv_kernel".into(), gguf_file::Value::U32(4));
        m.insert("qwen35.context_length".into(), gguf_file::Value::U32(262144));
        m
    }

    #[test]
    fn test_gguf_config_from_metadata_qwen35() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();

        assert_eq!(cfg.arch, "qwen35");
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.block_count, 64);
        assert_eq!(cfg.head_count, 24);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(*cfg.head_count_kv.first(), 4);
        assert_eq!(cfg.ssm_d_inner, Some(6144));
        assert_eq!(cfg.ssm_d_state, Some(128));
        assert_eq!(cfg.ssm_n_group, Some(16));
        assert_eq!(cfg.ssm_dt_rank, Some(48));
        assert_eq!(cfg.full_attention_interval, Some(4));
        assert_eq!(cfg.rope_dimension_count, Some(64));
    }

    #[test]
    fn test_is_recurrent_qwen35() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();

        // full_attention_interval=4 → layer 3,7,11,... are full attention
        assert!(cfg.is_recurrent(0));   // block 0: recurrent (GDN)
        assert!(cfg.is_recurrent(1));   // block 1: recurrent
        assert!(cfg.is_recurrent(2));   // block 2: recurrent
        assert!(!cfg.is_recurrent(3));  // block 3: full attention ((3+1)%4==0)
        assert!(cfg.is_recurrent(4));   // block 4: recurrent
        assert!(!cfg.is_recurrent(7));  // block 7: full attention
    }

    #[test]
    fn test_num_recurrent_layers() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();
        // 64 layers, every 4th is full attention → 48 recurrent, 16 full
        assert_eq!(cfg.num_recurrent_layers(), 48);
    }

    #[test]
    fn test_gdn_conv_channels() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();
        // conv_channels = 6144 + 2*16*128 = 6144 + 4096 = 10240
        assert_eq!(cfg.gdn_conv_channels(), 10240);
    }

    #[test]
    fn test_gdn_head_v_dim() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();
        // head_v_dim = 6144 / 48 = 128
        assert_eq!(cfg.gdn_head_v_dim(), 128);
    }

    #[test]
    fn test_default_attention_scale() {
        let m = make_qwen35_metadata();
        let cfg = GgufConfig::from_metadata(&m).unwrap();
        let expected = 1.0 / (256.0f64).sqrt();
        assert!((cfg.default_attention_scale() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_is_sliding_gemma4_pattern() {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), gguf_file::Value::String("gemma4".into()));
        m.insert("gemma4.embedding_length".into(), gguf_file::Value::U32(5376));
        m.insert("gemma4.block_count".into(), gguf_file::Value::U32(60));
        m.insert("gemma4.attention.head_count".into(), gguf_file::Value::U32(32));
        m.insert("gemma4.attention.head_count_kv".into(), gguf_file::Value::U32(16));
        m.insert("gemma4.attention.layer_norm_rms_epsilon".into(), gguf_file::Value::F32(1e-6));
        m.insert("gemma4.attention.sliding_window_type".into(), gguf_file::Value::U32(6));
        let cfg = GgufConfig::from_metadata(&m).unwrap();

        // sliding_window_type=6 → every 6th layer is global, rest are sliding
        assert!(cfg.is_sliding(0));    // sliding
        assert!(cfg.is_sliding(4));    // sliding
        assert!(!cfg.is_sliding(5));   // global ((5+1)%6==0)
        assert!(cfg.is_sliding(6));    // sliding
        assert!(!cfg.is_sliding(11));  // global
    }

    fn dummy_tensor_info() -> gguf_file::TensorInfo {
        gguf_file::TensorInfo {
            shape: candle::Shape::from_dims(&[1]),
            offset: 0,
            ggml_dtype: candle::quantized::GgmlDType::F32,
        }
    }

    fn tensor_set(names: &[&str]) -> HashMap<String, gguf_file::TensorInfo> {
        names.iter().map(|n| (n.to_string(), dummy_tensor_info())).collect()
    }

    #[test]
    fn test_detect_layer_kind_gdn() {
        let tensor_infos = tensor_set(&[
            "blk.0.ssm_a", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        ]);
        let kind = detect_layer_kind(&tensor_infos, 0);
        assert_eq!(kind.attn, AttnKind::DeltaNet);
        assert_eq!(kind.ffn, FfnKind::Dense);
    }

    #[test]
    fn test_detect_layer_kind_standard_attn() {
        let tensor_infos = tensor_set(&[
            "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
            "blk.3.attn_output.weight", "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
            "blk.3.ffn_gate.weight",
        ]);
        let kind = detect_layer_kind(&tensor_infos, 3);
        assert_eq!(kind.attn, AttnKind::Standard);
        assert_eq!(kind.ffn, FfnKind::Dense);
        assert!(kind.has_qk_norms);
        assert!(kind.has_gated_q);
    }

    #[test]
    fn test_detect_layer_kind_moe_with_shared() {
        let tensor_infos = tensor_set(&[
            "blk.0.ssm_a", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_up_shexp.weight",
        ]);
        let kind = detect_layer_kind(&tensor_infos, 0);
        assert_eq!(kind.attn, AttnKind::DeltaNet);
        assert_eq!(kind.ffn, FfnKind::Moe);
        assert!(kind.has_shared_expert);
    }

    #[test]
    fn test_detect_layer_kind_dual_path() {
        let tensor_infos = tensor_set(&[
            "blk.0.attn_q.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight",
        ]);
        let kind = detect_layer_kind(&tensor_infos, 0);
        assert_eq!(kind.ffn, FfnKind::DualPath);
    }
}
