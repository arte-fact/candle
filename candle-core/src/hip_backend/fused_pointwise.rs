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

/// Fused post-norm residual: `dst = rms_norm(x, weight, eps) + residual`.
///
/// Replaces the `post_norm.forward(&attn)? + residual?` pattern used in
/// Gemma4 and other post-norm transformer blocks with a single kernel
/// launch. The gemma4 forward has two of these per layer (post-attn
/// residual, post-ffn residual) × 48 layers × 64 forwards ≈ 6,144
/// kernel launches folded into half as many.
///
/// Mirrors the geometry of the existing `rms_norm` and
/// `fused_residual_rms_norm` launchers in `candle-nn/src/ops.rs`:
/// one row per warp, `blockDim = (64, 1, 1)` or `(1024, 1, 1)` when
/// hidden dim ≥ 1024.
///
/// # Requirements
/// - `x`, `residual`, `weight` on the same HIP device, contiguous, f32
/// - `x.shape() == residual.shape()` and the last dim equals
///   `weight.dims1()`
///
/// # Returns
/// A fresh tensor with the same shape as `x`.
pub fn rms_norm_post_residual_fused(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("rms_norm_post_residual_fused: x must be on HIP"),
    };
    if !matches!(residual.device(), Device::Hip(d) if d.same_device(&dev))
        || !matches!(weight.device(), Device::Hip(d) if d.same_device(&dev))
    {
        crate::bail!("rms_norm_post_residual_fused: inputs must share a HIP device");
    }
    if x.dtype() != DType::F32 || residual.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        crate::bail!("rms_norm_post_residual_fused: all inputs must be f32");
    }
    if x.shape() != residual.shape() {
        crate::bail!(
            "rms_norm_post_residual_fused: x {:?} vs residual {:?} shape mismatch",
            x.shape(),
            residual.shape()
        );
    }
    if !x.is_contiguous() || !residual.is_contiguous() || !weight.is_contiguous() {
        crate::bail!("rms_norm_post_residual_fused: all inputs must be contiguous");
    }
    let hidden = *x.dims().last().unwrap_or(&0);
    let weight_dim = weight.dims1()?;
    if hidden != weight_dim {
        crate::bail!(
            "rms_norm_post_residual_fused: hidden {hidden} != weight {weight_dim}"
        );
    }

    let el = x.elem_count();
    let n_rows = el / hidden;
    if el == 0 {
        return x.clone().contiguous();
    }

    let (x_st, x_l) = x.storage_and_layout();
    let (r_st, r_l) = residual.storage_and_layout();
    let (w_st, w_l) = weight.storage_and_layout();

    macro_rules! slice_of {
        ($st:expr, $l:expr, $label:literal) => {{
            let hip_st = match &*$st {
                Storage::Hip(s) => s,
                _ => crate::bail!("rms_norm_post_residual_fused: {} not HIP", $label),
            };
            let slice = hip_st.as_hip_slice::<f32>()?;
            match $l.contiguous_offsets() {
                Some((lo, hi)) => slice.slice(lo..hi),
                None => crate::bail!("rms_norm_post_residual_fused: {} non-contiguous", $label),
            }
        }};
    }

    let x_view = slice_of!(x_st, x_l, "x");
    let r_view = slice_of!(r_st, r_l, "residual");
    let w_view = slice_of!(w_st, w_l, "weight");

    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(el)? };

    let func = dev.get_or_load_func("rmsnorm_post_residual_f32", &kernels::REDUCE)?;

    // Match the standalone `rms_norm` launch geometry.
    let block_size: u32 = if hidden >= 1024 { 1024 } else { 64 };
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&x_view);
    builder.arg(&r_view);
    builder.arg(&out);
    builder.arg(&w_view);
    let n_cols_i32 = hidden as i32;
    let block_size_i32 = block_size as i32;
    builder.arg(&n_cols_i32);
    builder.arg(&block_size_i32);
    builder.arg(&eps);
    // SAFETY: ffi — shapes/dtype/contig validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);
    drop(r_st);
    drop(w_st);

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

/// Phase T1.2 — fused MoE topk: softmax + top-k argmax + normalise in
/// a single kernel.
///
/// Replaces the 7-step Rust chain at
/// `quantized_gemma4.rs:1762-1779`:
///   softmax → to_device(Cpu) → arg_sort_last_dim → narrow → to_device(Hip)
///          → gather → sum_keepdim → broadcast_div
///
/// Kernel ports turbo's `topk_moe_cuda` (gemma4-specialised: with_norm=
/// true, no bias, no sigmoid).  Outputs are COMPACT
/// `(n_rows × n_expert_used)` rather than turbo's full-strided layout.
///
/// # Args
/// - `router_logits`: `(n_rows, n_experts)` f32 contiguous on HIP.
///   `n_experts` MUST match a compiled instantiation (currently 64,
///   128, 512).
/// - `n_expert_used`: top-k value (≤ n_experts; for gemma4 this is 8).
///
/// # Returns
/// `(weights, ids)` — both `(n_rows, n_expert_used)`:
///   * `weights`: f32 normalised top-k routing probabilities (sum=1
///     per row).
///   * `ids`: u32 expert indices (`unsigned int` on the HIP side).
pub fn topk_moe_norm_fused(
    router_logits: &Tensor,
    n_expert_used: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = match router_logits.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("topk_moe_norm_fused: input must be HIP"),
    };
    if router_logits.dtype() != DType::F32 {
        crate::bail!(
            "topk_moe_norm_fused: input must be f32, got {:?}",
            router_logits.dtype()
        );
    }
    if !router_logits.is_contiguous() {
        crate::bail!("topk_moe_norm_fused: input must be contiguous");
    }
    if router_logits.rank() < 1 {
        crate::bail!("topk_moe_norm_fused: input must have rank >= 1");
    }
    let n_experts = *router_logits.dims().last().unwrap();
    let n_rows = router_logits.elem_count() / n_experts;
    if n_expert_used == 0 || n_expert_used > n_experts {
        crate::bail!(
            "topk_moe_norm_fused: n_expert_used={} out of (0, n_experts={}]",
            n_expert_used, n_experts
        );
    }

    // Pick the compiled kernel instance.
    let kernel_name = match n_experts {
        64 => "topk_moe_norm_f32_n64",
        128 => "topk_moe_norm_f32_n128",
        512 => "topk_moe_norm_f32_n512",
        _ => crate::bail!(
            "topk_moe_norm_fused: no compiled kernel for n_experts={} (have 64, 128, 512)",
            n_experts
        ),
    };

    let (logits_st, logits_l) = router_logits.storage_and_layout();
    let logits_hip = match &*logits_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("topk_moe_norm_fused: logits storage not HIP"),
    };
    let logits_slice = logits_hip.as_hip_slice::<f32>()?;
    let logits_view = match logits_l.contiguous_offsets() {
        Some((lo, hi)) => logits_slice.slice(lo..hi),
        None => crate::bail!("topk_moe_norm_fused: non-contiguous logits"),
    };

    // Output buffers: weights f32, ids u32; both (n_rows, n_expert_used).
    // SAFETY: kernel writes every (row, k).
    let weights_out = unsafe { dev.alloc::<f32>(n_rows * n_expert_used)? };
    let ids_out = unsafe { dev.alloc::<u32>(n_rows * n_expert_used)? };

    let func = dev.get_or_load_func(kernel_name, &kernels::TOPK_MOE)?;

    // Grid: ceil(n_rows / 4) blocks, each block has 4 wavefronts.
    const WARPS_PER_BLOCK: u32 = 4;
    let cfg = LaunchConfig {
        grid_dim: (((n_rows as u32) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1),
        block_dim: (WARP_SIZE, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };
    let n_rows_arg = n_rows as i32;
    let n_used_arg = n_expert_used as i32;
    let mut builder = func.builder();
    builder.arg(&logits_view);
    builder.arg(&weights_out);
    builder.arg(&ids_out);
    builder.arg(&n_rows_arg);
    builder.arg(&n_used_arg);
    // SAFETY: ffi — shapes/dtype validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(logits_st);

    let out_shape = vec![n_rows, n_expert_used];
    let weights_storage = HipStorage::wrap_hip_slice(weights_out, dev.clone());
    let ids_storage = HipStorage::wrap_hip_slice(ids_out, dev);
    let weights = Tensor::from_storage(
        Storage::Hip(weights_storage),
        <Shape as From<Vec<usize>>>::from(out_shape.clone()),
        BackpropOp::none(),
        false,
    );
    let ids = Tensor::from_storage(
        Storage::Hip(ids_storage),
        <Shape as From<Vec<usize>>>::from(out_shape),
        BackpropOp::none(),
        false,
    );
    Ok((weights, ids))
}

/// W3a: Fused MoE-combine weighted sum over the top-k axis.
///
/// Replaces the two-kernel Rust chain in
/// `quantized_gemma4.rs:2021-2023`:
/// ```ignore
/// let topk_weights = topk_weights.unsqueeze(D::Minus1)?;  // shape -> [T, K, 1]
/// let weighted = moe_out.broadcast_mul(&topk_weights)?;   // bmul_f32 kernel
/// let moe_summed = weighted.sum(1)?;                      // fast_sum_f32 kernel
/// ```
/// with a single HIP kernel launch per MoE layer.
///
/// # Perf evidence
/// From the 2026-04-16 rocprofv3 baseline
/// (`bench/results/kfr_gemma4-26B-A4B-Q8_0.json`), `fast_sum_f32` is
/// called 7,800× at 46 µs/call = **358 ms aggregate GPU time**.
/// The kernel is launch-overhead bound because `fast_sum` launches one
/// block per output element with a 1024-wide default block — totally
/// over-provisioned for K=8 reductions. Plus the preceding
/// `broadcast_mul` adds bmul_f32 time and allocates a [T, K, H]
/// intermediate tensor every layer. This fused replacement reads
/// `moe_out` once, broadcasts the weights via LDS, and writes
/// `combined` — no intermediate tensor, one kernel launch.
///
/// # Shape contract
/// - `moe_out`: `[T, K, H]` f32 contiguous on HIP.
/// - `topk_weights`: `[T, K]` f32 contiguous on the same HIP device.
/// - returns: `[T, H]` f32.
///
/// `T` (tokens, prompt+decode), `K` (top-k experts used, ≤ 32 today),
/// `H` (hidden dim) are recovered from the tensor shapes.
pub fn moe_combine_weighted_sum_fused(
    moe_out: &Tensor,
    topk_weights: &Tensor,
) -> Result<Tensor> {
    let dev = match moe_out.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("moe_combine_weighted_sum_fused: moe_out must be HIP"),
    };
    if !matches!(topk_weights.device(), Device::Hip(d) if d.same_device(&dev)) {
        crate::bail!(
            "moe_combine_weighted_sum_fused: moe_out and topk_weights must be on the same HIP device"
        );
    }
    if moe_out.dtype() != DType::F32 || topk_weights.dtype() != DType::F32 {
        crate::bail!(
            "moe_combine_weighted_sum_fused: both inputs must be f32 (got moe_out={:?}, weights={:?})",
            moe_out.dtype(), topk_weights.dtype()
        );
    }
    if !moe_out.is_contiguous() {
        crate::bail!("moe_combine_weighted_sum_fused: moe_out must be contiguous");
    }
    if !topk_weights.is_contiguous() {
        crate::bail!("moe_combine_weighted_sum_fused: topk_weights must be contiguous");
    }
    if moe_out.rank() != 3 {
        crate::bail!(
            "moe_combine_weighted_sum_fused: moe_out must have rank 3 [T, K, H], got {:?}",
            moe_out.shape()
        );
    }
    if topk_weights.rank() != 2 {
        crate::bail!(
            "moe_combine_weighted_sum_fused: topk_weights must have rank 2 [T, K], got {:?}",
            topk_weights.shape()
        );
    }
    let (t_out, k_out, h_out) = moe_out.dims3()?;
    let (t_w, k_w) = topk_weights.dims2()?;
    if t_out != t_w || k_out != k_w {
        crate::bail!(
            "moe_combine_weighted_sum_fused: shape mismatch — moe_out [{},{},{}] vs weights [{},{}]",
            t_out, k_out, h_out, t_w, k_w
        );
    }
    // Kernel's compile-time MOE_COMBINE_K_MAX is 32.
    if k_out > 32 {
        crate::bail!(
            "moe_combine_weighted_sum_fused: K={} exceeds compiled MOE_COMBINE_K_MAX=32",
            k_out
        );
    }

    let (out_st, out_l) = moe_out.storage_and_layout();
    let out_hip = match &*out_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("moe_combine_weighted_sum_fused: moe_out storage not HIP"),
    };
    let out_view = match out_l.contiguous_offsets() {
        Some((lo, hi)) => out_hip.as_hip_slice::<f32>()?.slice(lo..hi),
        None => crate::bail!("moe_combine_weighted_sum_fused: non-contiguous moe_out"),
    };

    let (w_st, w_l) = topk_weights.storage_and_layout();
    let w_hip = match &*w_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("moe_combine_weighted_sum_fused: topk_weights storage not HIP"),
    };
    let w_view = match w_l.contiguous_offsets() {
        Some((lo, hi)) => w_hip.as_hip_slice::<f32>()?.slice(lo..hi),
        None => crate::bail!("moe_combine_weighted_sum_fused: non-contiguous topk_weights"),
    };

    // SAFETY: kernel writes every (t, h).
    let combined_out = unsafe { dev.alloc::<f32>(t_out * h_out)? };

    let func = dev.get_or_load_func(
        "moe_combine_weighted_sum_f32",
        &kernels::TOPK_MOE,
    )?;

    // Must match MOE_COMBINE_BH in topk_moe.cu.
    const BH: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (((h_out as u32) + BH - 1) / BH, t_out as u32, 1),
        block_dim: (BH, 1, 1),
        shared_mem_bytes: 0,
    };
    let t_arg = t_out as i32;
    let k_arg = k_out as i32;
    let h_arg = h_out as i32;
    let mut builder = func.builder();
    builder.arg(&out_view);
    builder.arg(&w_view);
    builder.arg(&combined_out);
    builder.arg(&t_arg);
    builder.arg(&k_arg);
    builder.arg(&h_arg);
    // SAFETY: ffi — shapes/dtype validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(out_st);
    drop(w_st);

    let out_shape: Shape = vec![t_out, h_out].into();
    let storage = HipStorage::wrap_hip_slice(combined_out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(storage),
        out_shape,
        BackpropOp::none(),
        false,
    ))
}

/// Fused MoE router pre-amble: `out = v_norm(x, eps) * (1/sqrt(n_embd)) * gate_s`.
///
/// Phase T1.1 — replaces 5 separate launches in
/// `quantized_gemma4.rs:1745-1752`:
///   1. `v_norm(attn_out, eps)`     (sqr + mean_keepdim + add + sqrt + bdiv)
///   2. `normed * (1/sqrt(n_embd))` (scalar mul)
///   3. `scaled.broadcast_mul(gate_s)` (per-channel mul)
/// with a single fused kernel `gemma4_router_prep_f32`.
///
/// On Gemma-4-26B-A4B 4-GPU decode, the T1 profile measured the router
/// pre-amble at ~227 µs/layer × 30 layers = 6.8 ms/token (33 % of MoE
/// FFN time). Each saved launch removes ~50 µs of HIP dispatch
/// overhead, so collapsing 5 → 1 nominally saves 4 launches × 30 layers
/// × 50 µs ≈ 6 ms/token (~20 % decode speedup).
///
/// # Requirements
/// - `x` and `gate_s` must be on the same HIP device, f32, contiguous.
/// - `gate_s` length must equal `x`'s last dim (== `n_embd`).
pub fn gemma4_router_prep_fused(x: &Tensor, gate_s: &Tensor, eps: f64) -> Result<Tensor> {
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("gemma4_router_prep_fused: input must be HIP"),
    };
    if x.dtype() != DType::F32 || gate_s.dtype() != DType::F32 {
        crate::bail!(
            "gemma4_router_prep_fused: x ({:?}) and gate_s ({:?}) must be f32",
            x.dtype(), gate_s.dtype()
        );
    }
    if !x.is_contiguous() {
        crate::bail!("gemma4_router_prep_fused: x must be contiguous");
    }
    if !gate_s.is_contiguous() {
        crate::bail!("gemma4_router_prep_fused: gate_s must be contiguous");
    }
    if x.rank() < 1 {
        crate::bail!("gemma4_router_prep_fused: x must have rank >= 1");
    }

    let row_len = *x.dims().last().unwrap();
    if gate_s.elem_count() != row_len {
        crate::bail!(
            "gemma4_router_prep_fused: gate_s len ({}) != x last dim ({})",
            gate_s.elem_count(), row_len
        );
    }
    let n_rows = x.elem_count() / row_len;
    if row_len == 0 || n_rows == 0 {
        return x.clone().contiguous();
    }

    let (x_st, x_l) = x.storage_and_layout();
    let x_hip = match &*x_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gemma4_router_prep_fused: x storage not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = match x_l.contiguous_offsets() {
        Some((lo, hi)) => x_slice.slice(lo..hi),
        None => crate::bail!("gemma4_router_prep_fused: non-contiguous storage"),
    };

    let (gs_st, gs_l) = gate_s.storage_and_layout();
    let gs_hip = match &*gs_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gemma4_router_prep_fused: gate_s storage not HIP"),
    };
    let gs_slice = gs_hip.as_hip_slice::<f32>()?;
    let gs_view = match gs_l.contiguous_offsets() {
        Some((lo, hi)) => gs_slice.slice(lo..hi),
        None => crate::bail!("gemma4_router_prep_fused: non-contiguous gate_s"),
    };

    // SAFETY: populated by the kernel below.
    let out = unsafe { dev.alloc::<f32>(x.elem_count())? };

    let func = dev.get_or_load_func("gemma4_router_prep_f32", &kernels::UNARY)?;

    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let row_len_arg = row_len;
    let eps_f32 = eps as f32;
    let inv_sqrt_n: f32 = 1.0_f32 / (row_len as f32).sqrt();
    let mut builder = func.builder();
    builder.arg(&row_len_arg);
    builder.arg(&eps_f32);
    builder.arg(&inv_sqrt_n);
    builder.arg(&x_view);
    builder.arg(&gs_view);
    builder.arg(&out);
    // SAFETY: ffi — shapes/dtype/contig validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);
    drop(gs_st);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<Vec<usize>>>::from(x.dims().to_vec()),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}

/// W3d: Fused `gelu(gate) * up → quantize_q8_1` for MoE gate_up → down.
///
/// Replaces the 5-step chain in `quantized_gemma4.rs:2005-2014`:
/// ```ignore
/// let gate = gate_up.narrow(D::Minus1, 0, inter)?.contiguous()?;
/// let up   = gate_up.narrow(D::Minus1, inter, inter)?.contiguous()?;
/// let activated = gate.gelu()?.broadcast_mul(&up)?;
/// // down.indexed_moe_forward internally quantizes activated to Q8_1
/// ```
/// with a single HIP kernel (`gelu_mul_quantize_q8_1_f32` in
/// `candle-hip-kernels/src/quantized.cu`) producing a Q8_1 buffer
/// suitable for direct consumption by
/// `QHipStorage::indexed_moe_forward_preq8`. Eliminates 4-5 small
/// kernel launches per MoE layer on decode (narrow × 2, contiguous ×
/// 2, gelu, bmul, quantize) at ~5 µs each = ~25 µs CPU per layer =
/// ~750 µs per decode forward.
///
/// # Inputs
/// - `gate_up`: HIP f32 tensor of shape `[batch, topk, 2 * inter]`,
///   row-major contiguous (exactly the output of
///   `gate_up_exps.indexed_moe_forward` with a fused gate_up weight).
/// - `inter`: half the last-dim of `gate_up` (= `expert_intermediate`).
///
/// # Output
/// An owned `HipSlice<u8>` holding `batch * topk * inter / QK8_1`
/// `block_q8_1` entries (36 bytes each). The caller is responsible for
/// keeping it alive for the duration of the subsequent
/// `indexed_moe_forward_preq8` call and passes it as a `HipView<u8>`.
///
/// The output layout matches what candle's existing `quantize_q8_1`
/// kernel writes — per-32-element blocks with `ds.x = delta` and
/// `ds.y = sum(raw_floats)` — so the downstream MMVQ kernel reads
/// without any extra stride acrobatics.
pub fn gelu_mul_quantize_q8_1_fused(
    gate_up: &Tensor,
    inter: usize,
) -> Result<(hipdarc::driver::HipSlice<u8>, crate::Shape)> {
    use crate::hip_backend::HipDevice;
    let dev: HipDevice = match gate_up.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("gelu_mul_quantize_q8_1_fused: gate_up must be on HIP"),
    };
    if gate_up.dtype() != DType::F32 {
        crate::bail!(
            "gelu_mul_quantize_q8_1_fused: gate_up must be f32, got {:?}",
            gate_up.dtype()
        );
    }
    if !gate_up.is_contiguous() {
        crate::bail!("gelu_mul_quantize_q8_1_fused: gate_up must be contiguous");
    }
    if gate_up.rank() != 3 {
        crate::bail!(
            "gelu_mul_quantize_q8_1_fused: expected rank 3 [batch, topk, 2*inter], got shape {:?}",
            gate_up.shape()
        );
    }
    let (batch, topk, last) = gate_up.dims3()?;
    if last != 2 * inter {
        crate::bail!(
            "gelu_mul_quantize_q8_1_fused: last dim {} != 2*inter {}",
            last, 2 * inter
        );
    }
    if inter % 32 != 0 {
        crate::bail!(
            "gelu_mul_quantize_q8_1_fused: inter={} must be a multiple of QK8_1=32",
            inter
        );
    }

    let (st, l) = gate_up.storage_and_layout();
    let hip = match &*st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gelu_mul_quantize_q8_1_fused: storage not HIP"),
    };
    let slice = hip.as_hip_slice::<f32>()?;
    let gate_up_view = match l.contiguous_offsets() {
        Some((lo, hi)) => slice.slice(lo..hi),
        None => crate::bail!("gelu_mul_quantize_q8_1_fused: non-contiguous gate_up"),
    };

    // Pad `inter` up to MATRIX_ROW_PADDING so the downstream MMVQ's
    // `k_padded / QK8_1` block stride reads initialised data (zero
    // blocks in the padding range).
    let inter_padded = crate::quantized::hip::pad(inter, crate::quantized::hip::MATRIX_ROW_PADDING);

    // Output buffer: batch * topk * (inter_padded / 32) Q8_1 blocks.
    const Q8_1_BLOCK_BYTES: usize = 36;
    let n_blocks = batch * topk * (inter_padded / 32);
    let q8_bytes = n_blocks * Q8_1_BLOCK_BYTES;
    // SAFETY: kernel writes every block's qs[0..32] + ds (padded blocks
    // get zero val → q=0, d=0, sum=0).
    let out = unsafe { dev.alloc::<u8>(q8_bytes)? };

    let func = dev.get_or_load_func(
        "gelu_mul_quantize_q8_1_f32",
        &kernels::QUANTIZED,
    )?;
    // Grid: (inter_padded / QK8_1, topk, batch). Block: (32, 1, 1).
    let cfg = LaunchConfig {
        grid_dim: ((inter_padded / 32) as u32, topk as u32, batch as u32),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let inter_arg = inter as i32;
    let inter_padded_arg = inter_padded as i32;
    let topk_arg = topk as i32;
    let batch_arg = batch as i32;
    let mut builder = func.builder();
    builder.arg(&gate_up_view);
    builder.arg(&out);
    builder.arg(&inter_arg);
    builder.arg(&inter_padded_arg);
    builder.arg(&topk_arg);
    builder.arg(&batch_arg);
    // SAFETY: shape validated; all kernel args match kernel signature.
    unsafe { builder.launch(cfg) }.w()?;

    drop(st);

    let out_shape: crate::Shape = vec![batch, topk, inter].into();
    Ok((out, out_shape))
}

/// Fused RMSNorm + Scale + Q8_1 quantize.
///
/// Combines three operations into a single kernel launch:
///   1. `norm = rmsnorm(x, eps)`
///   2. `scaled = norm * alpha` (per-channel weight)
///   3. `q8 = quantize_q8_1(scaled)` (Q8_1 quantization)
///
/// Returns `(q8_buffer, optional_f32_output)`:
///   - `q8_buffer`: raw Q8_1 blocks as a byte buffer, shape = (nrows * blocks_per_row * 36) bytes
///   - `f32_output`: if `need_f32` is true, also returns the f32 scaled output for residual use
///
/// This eliminates 2 kernel launches + 1 HBM round-trip per call.
/// On a layer with 3 MMQ consumers (Q/K/V), the Q8_1 is produced once
/// and consumed 3 times instead of each consumer independently quantizing.
///
/// # Requirements
/// - `x` must be on a HIP device, f32, contiguous, rank >= 2.
/// - `alpha` must be f32, contiguous, length == last dim of x.
pub fn rmsnorm_q8_fused(
    x: &Tensor,
    alpha: &Tensor,
    eps: f64,
    need_f32: bool,
) -> Result<(hipdarc::driver::HipSlice<u8>, Option<Tensor>)> {
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("rmsnorm_q8_fused: input must be on a HIP device"),
    };
    if x.dtype() != DType::F32 {
        crate::bail!("rmsnorm_q8_fused: input must be f32");
    }
    if !x.is_contiguous() {
        crate::bail!("rmsnorm_q8_fused: input must be contiguous");
    }

    let dims = x.dims();
    let ncols = *dims.last().unwrap();
    let nrows = x.elem_count() / ncols;

    if ncols % 32 != 0 {
        crate::bail!("rmsnorm_q8_fused: ncols ({}) must be a multiple of 32 (QK8_1)", ncols);
    }

    let blocks_per_row = ncols / 32;
    let q8_bytes = nrows * blocks_per_row * 36; // 36 bytes per Q8_1 block

    let (x_st, x_l) = x.storage_and_layout();
    let x_hip = match &*x_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("rmsnorm_q8_fused: storage is not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = match x_l.contiguous_offsets() {
        Some((lo, hi)) => x_slice.slice(lo..hi),
        None => crate::bail!("rmsnorm_q8_fused: non-contiguous storage"),
    };

    let (a_st, a_l) = alpha.storage_and_layout();
    let a_hip = match &*a_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("rmsnorm_q8_fused: alpha not HIP"),
    };
    let a_slice = a_hip.as_hip_slice::<f32>()?;
    let a_view = match a_l.contiguous_offsets() {
        Some((lo, hi)) => a_slice.slice(lo..hi),
        None => crate::bail!("rmsnorm_q8_fused: alpha non-contiguous"),
    };

    // Allocate Q8_1 output buffer
    let q8_out = unsafe { dev.alloc::<u8>(q8_bytes)? };

    // Optionally allocate f32 output
    let f32_out = if need_f32 {
        Some(unsafe { dev.alloc::<f32>(x.elem_count())? })
    } else {
        None
    };

    let func = dev.get_or_load_func("rmsnorm_q8_fused_f32", &kernels::REDUCE)?;

    let block_size: u32 = if ncols >= 1024 { 1024 } else { 64 };
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;
    let block_size_i32 = block_size as i32;
    let eps_f32 = eps as f32;

    let mut builder = func.builder();
    builder.arg(&x_view);
    builder.arg(&a_view);
    builder.arg(&q8_out);
    match &f32_out {
        Some(buf) => { builder.arg(buf); }
        None => {
            let null: usize = 0;
            builder.arg(&null);
        }
    }
    builder.arg(&ncols_i32);
    builder.arg(&block_size_i32);
    builder.arg(&eps_f32);

    unsafe { builder.launch(cfg) }.w()?;

    drop(x_st);
    drop(a_st);

    let f32_tensor = if let Some(buf) = f32_out {
        let st = HipStorage::wrap_hip_slice(buf, dev.clone());
        Some(Tensor::from_storage(
            Storage::Hip(st),
            x.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    } else {
        None
    };

    Ok((q8_out, f32_tensor))
}

/// Fused FFN decode: rmsnorm → gate matmul → up matmul → silu*gate → down matmul → residual add.
///
/// Eliminates 7 kernel launches per layer by fusing the entire FFN path into
/// a single persistent kernel. The hidden state stays in registers between
/// operations; only weight data is streamed from HBM.
///
/// # Requirements
/// - `h` must be contiguous f32 on HIP, shape `(1, 1, hidden_dim)` or `(hidden_dim,)`
/// - All weight tensors must be Q4_0 quantized on the same device
/// - `hidden_dim` must be a multiple of 64 (WARP_SIZE)
/// - `intermediate_dim` must fit in LDS (max ~8704 for 64KB LDS)
///
/// Modifies `h` in-place (residual add included).
pub fn fused_ffn_decode_q4_0(
    h: &mut Tensor,
    ffn_norm_w: &Tensor,
    w_gate: &crate::quantized::QTensor,
    w_up: &crate::quantized::QTensor,
    w_down: &crate::quantized::QTensor,
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f64,
) -> Result<()> {
    let dev = match h.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("fused_ffn_decode_q4_0: h must be HIP"),
    };
    if h.dtype() != DType::F32 || !h.is_contiguous() {
        crate::bail!("fused_ffn_decode_q4_0: h must be f32 contiguous");
    }
    if h.elem_count() != hidden_dim {
        crate::bail!("fused_ffn_decode_q4_0: h elem count {} != hidden_dim {hidden_dim}", h.elem_count());
    }
    if intermediate_dim > 8704 {
        crate::bail!("fused_ffn_decode_q4_0: intermediate_dim {intermediate_dim} > 8704 (LDS limit)");
    }

    let (h_st, h_l) = h.storage_and_layout();
    let h_hip = match &*h_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("fused_ffn_decode_q4_0: h not HIP"),
    };
    let h_slice = h_hip.as_hip_slice::<f32>()?;
    let h_view = match h_l.contiguous_offsets() {
        Some((lo, hi)) => h_slice.slice(lo..hi),
        None => crate::bail!("fused_ffn_decode_q4_0: h non-contiguous"),
    };

    let (nw_st, nw_l) = ffn_norm_w.storage_and_layout();
    let nw_hip = match &*nw_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("fused_ffn_decode_q4_0: norm weight not HIP"),
    };
    let nw_slice = nw_hip.as_hip_slice::<f32>()?;
    let nw_view = match nw_l.contiguous_offsets() {
        Some((lo, hi)) => nw_slice.slice(lo..hi),
        None => crate::bail!("fused_ffn_decode_q4_0: norm weight non-contiguous"),
    };

    // Get raw pointers to quantized weight data
    let gate_ptr = w_gate.device_ptr()? as usize;
    let up_ptr = w_up.device_ptr()? as usize;
    let down_ptr = w_down.device_ptr()? as usize;

    let func = dev.get_or_load_func("fused_ffn_decode_q4_0", &kernels::FUSED_FFN_DECODE)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: (intermediate_dim * 4 * 2) as u32, // gate_lds + up_lds
    };

    let hdim = hidden_dim as i32;
    let idim = intermediate_dim as i32;
    let eps_f32 = eps as f32;

    let mut builder = func.builder();
    builder.arg(&h_view);
    builder.arg(&nw_view);
    builder.arg(&gate_ptr);
    builder.arg(&up_ptr);
    builder.arg(&down_ptr);
    builder.arg(&hdim);
    builder.arg(&idim);
    builder.arg(&eps_f32);
    unsafe { builder.launch(cfg) }.w()?;

    drop(h_st);
    drop(nw_st);
    Ok(())
}
