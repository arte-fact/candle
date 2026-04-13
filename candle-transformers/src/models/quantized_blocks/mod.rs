//! Modular building blocks for quantized GGUF model inference.
//!
//! This module provides composable, GGUF-metadata-driven components that can be
//! assembled into complete model implementations. All code is device-agnostic —
//! works on CPU, CUDA, and HIP with zero backend-specific code.
//!
//! # Architecture
//!
//! ```text
//! Layer 1: GgufConfig       — reads ALL metadata from GGUF, no hardcoded values
//! Layer 2: Composable blocks — Attention, GDN, MLP, MoE, RoPE, norms
//! Layer 3: Model assembler   — thin per-arch file that composes blocks from config
//! ```
//!
//! # Supported block types
//!
//! - [`StandardAttention`] — GQA with optional QK norms, V norm
//! - [`GatedAttention`] — Q+gate fused, sigmoid gating (qwen35 full-attn layers)
//! - [`DeltaNetLayer`] — Gated Delta Net linear attention (qwen35 recurrent layers)
//! - [`DenseMlp`] — SwiGLU feed-forward
//! - [`MoeExperts`] — Mixture of Experts with optional shared expert
//! - [`RotaryEmbedding`] — Standard, multi-frequency, or custom-frequency RoPE

pub mod gguf_config;
pub mod gguf_loader;
pub mod rope;
pub mod norms;
pub mod attention;
pub mod ffn;
pub mod delta_net;

pub use gguf_config::*;
pub use gguf_loader::Gguf;
pub use rope::RotaryEmbedding;
pub use norms::{causal_mask, causal_mask_padded, v_norm, GemmaRmsNorm};
pub use attention::{StandardAttention, StandardAttentionOpts, AttnNorm, GatedAttention};
pub use ffn::{DenseMlp, MlpActivation, MoeExperts};
pub use delta_net::{DeltaNetLayer, GdnState, GdnDimensions};

use candle::Device;

/// Compare two devices by location string.
///
/// `Device::same_device(&other)` would be cleaner but candle doesn't expose it,
/// so we compare the location strings (`Cpu` vs `Hip{gpu_id}` vs `Cuda{gpu_id}`).
/// All multi-device assemblers use this to decide whether to call `to_device()`.
pub fn device_eq(a: &Device, b: &Device) -> bool {
    format!("{:?}", a.location()) == format!("{:?}", b.location())
}

/// Compute the layer-to-device assignment for pipeline-parallel split mode.
///
/// Splits `block_count` layers into `devices.len()` contiguous chunks. The
/// first `block_count % devices.len()` devices get one extra layer when the
/// count doesn't divide evenly. Mirrors llama.cpp's `LLAMA_SPLIT_MODE_LAYER`.
///
/// Returns a vector of length `block_count` where entry `i` is the device
/// index that owns layer `i`.
pub fn split_layers_across_devices(block_count: usize, n_dev: usize) -> Vec<usize> {
    assert!(n_dev > 0, "split_layers_across_devices: at least one device required");
    let base = block_count / n_dev;
    let extra = block_count % n_dev;
    let mut v = Vec::with_capacity(block_count);
    for d in 0..n_dev {
        let count = base + if d < extra { 1 } else { 0 };
        for _ in 0..count {
            v.push(d);
        }
    }
    v
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_split_layers_even() {
        // 40 layers across 4 devices: 10 each
        let m = split_layers_across_devices(40, 4);
        assert_eq!(m.len(), 40);
        assert_eq!(&m[..10], &[0; 10]);
        assert_eq!(&m[10..20], &[1; 10]);
        assert_eq!(&m[20..30], &[2; 10]);
        assert_eq!(&m[30..40], &[3; 10]);
    }

    #[test]
    fn test_split_layers_uneven() {
        // 42 layers across 4 devices: 11, 11, 10, 10
        let m = split_layers_across_devices(42, 4);
        assert_eq!(m.len(), 42);
        assert_eq!(&m[..11], &[0; 11]);
        assert_eq!(&m[11..22], &[1; 11]);
        assert_eq!(&m[22..32], &[2; 10]);
        assert_eq!(&m[32..42], &[3; 10]);
    }

    #[test]
    fn test_split_layers_single_device() {
        let m = split_layers_across_devices(32, 1);
        assert_eq!(m, vec![0; 32]);
    }
}
