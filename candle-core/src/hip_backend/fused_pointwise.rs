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

use crate::backend::BackendDevice;
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

/// Fused gate/up split + silu_mul:
/// `out[..., j] = silu(gate_up[..., j]) * gate_up[..., j + N]` for j in [0, N).
///
/// Replaces the
/// `narrow(..,0,N).contiguous() + narrow(..,N,N).contiguous() + silu_mul`
/// chain in `DenseMlp::forward` with a single kernel launch that reads
/// the fused `gate_up` projection output directly from one contiguous
/// buffer. Saves 2 `.contiguous()` materialisations per FFN per layer
/// per token — on qwen35-9B decode that's ~10 240 ucopy_f32 launches
/// eliminated per session.
///
/// # Requirements
/// - `gate_up` must be on a HIP device, contiguous, and f32.
/// - `gate_up.dims().last()` must be even (the dim gets halved).
///
/// # Returns
/// A fresh f32 tensor with the same shape as `gate_up` except the last
/// dim is halved.
pub fn silu_mul_split_last_fused(gate_up: &Tensor) -> Result<Tensor> {
    let dev = match gate_up.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("silu_mul_split_last_fused: input must be on HIP"),
    };
    if gate_up.dtype() != DType::F32 {
        crate::bail!(
            "silu_mul_split_last_fused: input must be f32, got {:?}",
            gate_up.dtype()
        );
    }
    if !gate_up.is_contiguous() {
        crate::bail!("silu_mul_split_last_fused: input must be contiguous");
    }
    let dims = gate_up.dims();
    let last = *dims
        .last()
        .ok_or_else(|| crate::Error::Msg("silu_mul_split_last_fused: empty shape".into()))?;
    if last % 2 != 0 {
        crate::bail!(
            "silu_mul_split_last_fused: last dim must be even, got {last}"
        );
    }
    let half_row = last / 2;
    let out_numel: usize = dims[..dims.len() - 1].iter().product::<usize>() * half_row;
    if out_numel == 0 {
        let mut shape = dims.to_vec();
        shape[dims.len() - 1] = half_row;
        return Tensor::zeros(shape, DType::F32, gate_up.device());
    }

    let (x_st, x_l) = gate_up.storage_and_layout();
    let x_hip = match &*x_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("silu_mul_split_last_fused: storage is not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = match x_l.contiguous_offsets() {
        Some((lo, hi)) => x_slice.slice(lo..hi),
        None => crate::bail!("silu_mul_split_last_fused: non-contiguous storage"),
    };

    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(out_numel)? };

    let func = dev.get_or_load_func("silu_mul_split_last_f32", &kernels::UNARY)?;

    const BLOCK: u32 = 256;
    let grid = ((out_numel as u32) + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    let numel_arg = out_numel;
    let half_row_arg = half_row;
    builder.arg(&numel_arg);
    builder.arg(&half_row_arg);
    builder.arg(&x_view);
    builder.arg(&out);
    // SAFETY: ffi — shape / dtype / contiguous validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);

    let mut out_shape: Vec<usize> = dims.to_vec();
    out_shape[dims.len() - 1] = half_row;
    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<Vec<usize>>>::from(out_shape),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}

/// Fused masked softmax + scale.
///
/// Replaces the
/// ```text
///   att = att * scale
///   att = att + mask            // broadcast_add, additive f32 mask
///   att = softmax_last_dim(att)
/// ```
/// chain that appears in every attention forward after QK^T, saving 2
/// kernel launches and 2 full memory sweeps over the
/// `(B, H, L_q, L_k)` score tensor. The kernel folds the scale-multiply
/// and mask-add into pass 1 of a 3-pass online-softmax, so there is a
/// single store of the final output.
///
/// # Shapes
/// - `att` shape `(B, H, L_q, L_k)`, contiguous, f32.
/// - `mask` optional f32, broadcast-compatible with `(B, 1, L_q, L_k)`.
///   Supported: `(1, 1, 1, L_k)`, `(1, 1, L_q, L_k)`, `(B, 1, 1, L_k)`,
///   `(B, 1, L_q, L_k)`. Must be contiguous when provided.
///
/// # Returns
/// A fresh `(B, H, L_q, L_k)` f32 tensor = softmax(att * scale + mask).
///
/// # Errors
/// Any precondition failure (non-HIP, non-f32, non-contiguous, wrong
/// mask shape) returns an error so callers can fall through to the
/// chained tensor-ops path.
pub fn masked_softmax_scale_fused(
    att: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let dev = match att.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("masked_softmax_scale_fused: att must be on HIP"),
    };
    if att.dtype() != DType::F32 {
        crate::bail!(
            "masked_softmax_scale_fused: att must be f32, got {:?}",
            att.dtype()
        );
    }
    if !att.is_contiguous() {
        crate::bail!("masked_softmax_scale_fused: att must be contiguous");
    }
    let (b_sz, n_head, l_q, l_k) = att.dims4()?;

    if let Some(m) = mask {
        if !matches!(m.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!("masked_softmax_scale_fused: mask not on same HIP device");
        }
        if m.dtype() != DType::F32 {
            crate::bail!(
                "masked_softmax_scale_fused: mask must be f32, got {:?}",
                m.dtype()
            );
        }
        if !m.is_contiguous() {
            crate::bail!("masked_softmax_scale_fused: mask must be contiguous");
        }
    }

    // Derive (mask_b_stride, mask_lq_stride) from the mask shape. Mirrors
    // the pattern used by `flash_attn_fused`.
    let (mask_b_stride, mask_lq_stride): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            let mdims = m.dims();
            let last = *mdims.last().ok_or_else(|| {
                crate::Error::Msg("masked_softmax_scale_fused: mask has empty shape".into())
            })?;
            if last != l_k {
                crate::bail!(
                    "masked_softmax_scale_fused: mask last dim {last} != L_k {l_k}"
                );
            }
            let second_last = if mdims.len() >= 2 {
                mdims[mdims.len() - 2]
            } else {
                1
            };
            let lq_stride: i32 = if second_last == l_q {
                l_k as i32
            } else if second_last == 1 {
                0
            } else {
                crate::bail!(
                    "masked_softmax_scale_fused: mask second-to-last dim {second_last} not 1 or L_q {l_q}"
                );
            };
            let mask_batch_dim = if mdims.len() >= 4 { mdims[0] } else { 1 };
            let b_stride: i32 = if mask_batch_dim == b_sz {
                (second_last * l_k) as i32
            } else if mask_batch_dim == 1 {
                0
            } else {
                crate::bail!(
                    "masked_softmax_scale_fused: mask batch dim {mask_batch_dim} not 1 or B {b_sz}"
                );
            };
            (b_stride, lq_stride)
        }
    };

    // --- Storage handles --------------------------------------------
    let (a_st, a_l) = att.storage_and_layout();
    let a_hip = match &*a_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("masked_softmax_scale_fused: att storage is not HIP"),
    };
    let a_slice = a_hip.as_hip_slice::<f32>()?;
    let a_view = match a_l.contiguous_offsets() {
        Some((lo, hi)) => a_slice.slice(lo..hi),
        None => crate::bail!("masked_softmax_scale_fused: att non-contiguous"),
    };

    let mask_owned = mask.map(|m| m.storage_and_layout());
    let mask_view_owned = mask_owned.as_ref().map(|(st, l)| {
        let hip_st = match &**st {
            Storage::Hip(s) => s,
            _ => panic!("masked_softmax_scale_fused: mask not HIP (checked above)"),
        };
        let slice = hip_st.as_hip_slice::<f32>().expect("as_hip_slice");
        let (lo, hi) = l.contiguous_offsets().expect("contiguous");
        slice.slice(lo..hi)
    });

    let numel = b_sz * n_head * l_q * l_k;
    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(numel)? };

    let func = dev.get_or_load_func("masked_softmax_scale_f32", &kernels::REDUCE)?;

    // Match the existing `softmax_f32` launch geometry: one Wave64 warp
    // per row, with `block_dim.y` providing the WARP_SIZE lanes.
    let n_rows = (b_sz * n_head * l_q) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_rows, 1, 1),
        block_dim: (1, WARP_SIZE, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&a_view);
    if let Some(mv) = mask_view_owned.as_ref() {
        builder.arg(mv);
    } else {
        builder.arg(&hipdarc::driver::NullDevicePtr::default());
    }
    builder.arg(&out);
    let n_cols_arg = l_k as i32;
    let b_arg = b_sz as i32;
    let h_arg = n_head as i32;
    let lq_arg = l_q as i32;
    let scale_arg = scale as f32;
    builder.arg(&n_cols_arg);
    builder.arg(&b_arg);
    builder.arg(&h_arg);
    builder.arg(&lq_arg);
    builder.arg(&scale_arg);
    builder.arg(&mask_b_stride);
    builder.arg(&mask_lq_stride);
    // SAFETY: ffi — shapes/dtype/contig validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(a_st);
    drop(mask_owned);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<(usize, usize, usize, usize)>>::from((b_sz, n_head, l_q, l_k)),
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
