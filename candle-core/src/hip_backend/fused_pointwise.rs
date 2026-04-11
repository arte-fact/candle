//! Fused pointwise / reduction kernels — Rust launchers.
//!
//! Each function here wraps a single HIP kernel in `candle-hip-kernels`
//! and exposes a `&Tensor → Tensor` (or `&Tensor → new Tensor with same
//! shape`) API so model code can replace multi-op tensor chains with a
//! single kernel launch. On non-HIP builds these entry points are not
//! compiled in; callers gate via `#[cfg(feature = "hip")]` and fall
//! through to the unfused tensor-op chain when needed.
//!
//! Current kernels:
//!   - [`softplus_fused`]: replaces the 6-op `abs + maximum + neg + exp
//!     + log1p + add` chain used for numerically-stable softplus.
//!   - [`l2_norm_fused`]: replaces the 5-op `sqr + sum_keepdim + add_eps
//!     + sqrt + broadcast_div` chain used for L2 normalisation along
//!     the last axis.
//!
//! These are Phase 4 of the gfx906 optimisation roadmap. They target
//! the long-tail pointwise kernels in the post-P5 rocprofv3 breakdown
//! — each of the 10+ sub-100-ms ops that make up the 913 ms pointwise
//! category.

use crate::op::BackpropOp;
use crate::{DType, Device, Result, Shape, Storage, Tensor};
use hipdarc::driver::LaunchConfig;

use super::{kernels, HipStorage, WrapErr};

/// gfx906 Wave64 size — mirrored from `candle-hip-kernels/build.rs`.
const WARP_SIZE: u32 = 64;

/// Fused softplus: `out = log(1 + exp(x))` computed via the numerically
/// stable form `max(x, 0) + log1p(exp(-|x|))`.
///
/// Replaces an 8-launch tensor-op chain with 1 kernel launch. Used per
/// GDN layer in every forward step (qwen35 / qwen3next).
///
/// # Requirements
/// - `x` must be on a HIP device, contiguous, and f32.
/// - Shape is arbitrary — the kernel treats the input as a flat buffer.
///
/// # Returns
/// A fresh f32 tensor with the same shape as `x`.
pub fn softplus_fused(x: &Tensor) -> Result<Tensor> {
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("softplus_fused: input must be on a HIP device"),
    };
    if x.dtype() != DType::F32 {
        crate::bail!(
            "softplus_fused: input must be f32, got {:?}",
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        crate::bail!("softplus_fused: input must be contiguous");
    }

    let numel = x.elem_count();
    if numel == 0 {
        return x.clone().contiguous();
    }

    let (x_st, x_l) = x.storage_and_layout();
    let x_hip = match &*x_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("softplus_fused: storage is not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = match x_l.contiguous_offsets() {
        Some((lo, hi)) => x_slice.slice(lo..hi),
        None => crate::bail!("softplus_fused: non-contiguous storage"),
    };

    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(numel)? };

    let func = dev.get_or_load_func("softplus_f32", &kernels::UNARY)?;

    // Simple 1D grid: 256 threads per block, one element per thread
    // (with a stride-loop fallback for numel > grid * 256).
    const BLOCK: u32 = 256;
    let grid = ((numel as u32) + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    let numel_arg = numel;
    builder.arg(&numel_arg);
    builder.arg(&x_view);
    builder.arg(&out);
    // SAFETY: ffi — shapes/dtype/contig validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        x.shape().clone(),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}

/// Fused L2 normalisation along the last axis:
/// `out[..., i] = x[..., i] / sqrt(Σ_i x[..., i]² + eps)`.
///
/// Replaces a 5-op tensor-op chain (sqr → sum_keepdim → add_eps →
/// sqrt → broadcast_div) with one kernel launch. Used by GDN for Q/K
/// head normalisation per recurrent layer per forward step.
///
/// # Shapes
/// - `x` can be any rank; the last dimension is the reduction axis.
/// - Each "row" (flat offset along the last axis) must fit into one
///   warp in the current single-warp-per-row implementation — up to
///   `WARP_SIZE = 64` threads running a strided loop, so row_len up
///   to a few thousand elements is fine. For qwen35's 128-element
///   head_k_dim this is a 2-iteration strided loop per thread.
///
/// # Requirements
/// - `x` must be on a HIP device, contiguous, and f32.
/// - `x.rank() >= 1`.
pub fn l2_norm_fused(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("l2_norm_fused: input must be on a HIP device"),
    };
    if x.dtype() != DType::F32 {
        crate::bail!(
            "l2_norm_fused: input must be f32, got {:?}",
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        crate::bail!("l2_norm_fused: input must be contiguous");
    }
    if x.rank() < 1 {
        crate::bail!("l2_norm_fused: input must have rank >= 1");
    }

    let row_len = *x.dims().last().unwrap();
    let n_rows = x.elem_count() / row_len;
    if row_len == 0 || n_rows == 0 {
        return x.clone().contiguous();
    }

    let (x_st, x_l) = x.storage_and_layout();
    let x_hip = match &*x_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("l2_norm_fused: storage is not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = match x_l.contiguous_offsets() {
        Some((lo, hi)) => x_slice.slice(lo..hi),
        None => crate::bail!("l2_norm_fused: non-contiguous storage"),
    };

    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(x.elem_count())? };

    let func = dev.get_or_load_func("l2_norm_f32", &kernels::UNARY)?;

    // One block per row, one Wave64 warp per block. The kernel's inner
    // loop handles row_len > warp_size via thread-stride.
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    let row_len_arg = row_len;
    let eps_f32 = eps as f32;
    builder.arg(&row_len_arg);
    builder.arg(&eps_f32);
    builder.arg(&x_view);
    builder.arg(&out);
    // SAFETY: ffi — shapes/dtype/contig validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<Vec<usize>>>::from(x.dims().to_vec()),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}
