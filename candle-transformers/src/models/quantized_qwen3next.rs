//! Quantized Qwen3-Next model.
//!
//! Structurally identical to qwen35moe (hybrid GDN + gated full-attention + MoE FFN).
//! The only difference is the SSM parameterization: `ssm_ba` instead of
//! separate `ssm_alpha` + `ssm_beta`. This is handled automatically by
//! `DeltaNetLayer::load()` which probes for both tensor names.
//!
//! GGUF arch string: "qwen3next"

pub use super::quantized_qwen35_moe::ModelWeights;
