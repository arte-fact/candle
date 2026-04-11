//! Flash-attention forward — HIP launcher.
//!
//! Replaces the `matmul + softmax + matmul` chain in
//! `candle-transformers/src/models/quantized_blocks/attention.rs::
//! gqa_attention` with a single kernel launch via
//! `candle-hip-kernels/src/flash_attn.cu`. Targets llama-class dense
//! models where attention is the dominant GPU-time category
//! (TinyLlama at ~30 % of GPU time on prefill).
//!
//! Phase 1 scope:
//!   - f32 Q, K, V, output
//!   - llama-style GQA via `n_rep = n_head / n_kv_head`
//!   - Optional additive mask shape `(B, 1, L_q, L_k)` (or
//!     broadcast-compatible)
//!   - Head dim 64 (TinyLlama, gemma4-small) and 128 (llama-70B,
//!     gemma4-4B, qwen3)
//!
//! The kernel takes Q at `(B, n_head, L_q, D)` and K/V at
//! `(B, n_kv_head, L_k, D)` — no expand on Q. It does the GQA
//! broadcast internally via `h_kv = h_idx / n_rep` (the llama
//! division convention, NOT the modulo/tile convention the GDN
//! kernel uses).

use crate::backend::BackendDevice;
use crate::op::BackpropOp;
use crate::{DType, Device, Result, Shape, Storage, Tensor};
use hipdarc::driver::LaunchConfig;

use super::{kernels, HipError, HipStorage, WrapErr};

const WARP_SIZE: u32 = 64;

/// Runs a single flash-attention forward pass on HIP.
///
/// # Shapes
/// - `q`: `(B, n_head, L_q, D)` f32, contiguous
/// - `k`, `v`: `(B, n_kv_head, L_k, D)` f32, contiguous
/// - `mask`: optional, shape `(1, 1, L_q, L_k)` or broadcast-compatible;
///    must be contiguous f32 when provided. Pass `-inf` for masked
///    positions, `0` for allowed.
///
/// # Returns
/// A fresh `(B, n_head, L_q, D)` f32 tensor.
///
/// # Errors
/// Any precondition failure (non-HIP, non-f32, non-contiguous,
/// unsupported head dim) returns an error so the caller can fall
/// back to the rocBLAS path.
pub fn flash_attn_fused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    // --- Device ------------------------------------------------------
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_fused: q must be on a HIP device"),
    };
    for (name, t) in [("k", k), ("v", v)] {
        if !matches!(t.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!("flash_attn_fused: {name} is on a different device than q");
        }
    }
    if let Some(m) = mask {
        if !matches!(m.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!("flash_attn_fused: mask is on a different device than q");
        }
    }

    // --- Dtype -------------------------------------------------------
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.dtype() != DType::F32 {
            crate::bail!("flash_attn_fused: {name} must be f32, got {:?}", t.dtype());
        }
    }
    if let Some(m) = mask {
        if m.dtype() != DType::F32 {
            crate::bail!("flash_attn_fused: mask must be f32, got {:?}", m.dtype());
        }
    }

    // --- Shapes ------------------------------------------------------
    let (b_sz, n_head, l_q, d) = q.dims4()?;
    let (kb, n_kv, l_k, kd) = k.dims4()?;
    if (kb, kd) != (b_sz, d) {
        crate::bail!(
            "flash_attn_fused: k shape {:?} incompatible with q {:?}",
            k.dims(),
            q.dims()
        );
    }
    let v_dims = v.dims4()?;
    if v_dims != (kb, n_kv, l_k, kd) {
        crate::bail!(
            "flash_attn_fused: v shape {:?} != k shape {:?}",
            v.dims(),
            k.dims()
        );
    }
    if n_head % n_kv != 0 {
        crate::bail!(
            "flash_attn_fused: n_head ({n_head}) not divisible by n_kv_head ({n_kv})"
        );
    }
    let n_rep = n_head / n_kv;

    // --- Supported head dim ------------------------------------------
    let kernel_name = match d {
        64 => "flash_attn_fwd_d64_f32",
        128 => "flash_attn_fwd_d128_f32",
        _ => {
            return Err(HipError::InternalError(Box::leak(
                format!("flash_attn_fused: unsupported head dim {d} (supported: 64, 128)")
                    .into_boxed_str(),
            ))
            .into())
        }
    };

    // --- Contiguity --------------------------------------------------
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if !t.is_contiguous() {
            crate::bail!("flash_attn_fused: {name} must be contiguous");
        }
    }
    let mask_contig = mask.map(|m| m.is_contiguous()).unwrap_or(true);
    if !mask_contig {
        crate::bail!("flash_attn_fused: mask must be contiguous when provided");
    }

    // Derive the mask B stride and L_q stride from the mask shape.
    // Supported shapes: (1, 1, 1, L_k), (1, 1, L_q, L_k),
    // (B, 1, 1, L_k), (B, 1, L_q, L_k). Last dim must be L_k.
    let (mask_b_stride, mask_l_q_stride): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            let mdims = m.dims();
            let last = *mdims.last().ok_or_else(|| {
                crate::Error::Msg("flash_attn_fused: mask has empty shape".into())
            })?;
            if last != l_k {
                crate::bail!(
                    "flash_attn_fused: mask last dim {last} != L_k {l_k}"
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
                    "flash_attn_fused: mask second-to-last dim {second_last} not 1 or L_q {l_q}"
                );
            };
            // Figure out the B-stride. The first dim of the mask
            // that's not of size 1 is the B dim.
            let mask_batch_dim = if mdims.len() >= 4 { mdims[0] } else { 1 };
            let b_stride: i32 = if mask_batch_dim == b_sz {
                // Per-batch mask: stride = second_last_size × L_k.
                (second_last * l_k) as i32
            } else if mask_batch_dim == 1 {
                0
            } else {
                crate::bail!(
                    "flash_attn_fused: mask batch dim {mask_batch_dim} not 1 or B {b_sz}"
                );
            };
            (b_stride, lq_stride)
        }
    };

    // --- Storage handles ---------------------------------------------
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let mask_owned = mask.map(|m| m.storage_and_layout());

    macro_rules! hip_view_f32 {
        ($storage:expr, $layout:expr, $label:literal) => {{
            let hip_st = match &*$storage {
                Storage::Hip(s) => s,
                _ => crate::bail!("flash_attn_fused: {} not a HIP storage", $label),
            };
            let slice = hip_st.as_hip_slice::<f32>()?;
            match $layout.contiguous_offsets() {
                Some((lo, hi)) => slice.slice(lo..hi),
                None => crate::bail!("flash_attn_fused: {} non-contiguous", $label),
            }
        }};
    }

    let q_view = hip_view_f32!(q_st, q_l, "q");
    let k_view = hip_view_f32!(k_st, k_l, "k");
    let v_view = hip_view_f32!(v_st, v_l, "v");
    let mask_view_owned = mask_owned.as_ref().map(|(st, l)| {
        let hip_st = match &**st {
            Storage::Hip(s) => s,
            _ => panic!("flash_attn_fused: mask not HIP (checked above)"),
        };
        let slice = hip_st.as_hip_slice::<f32>().expect("as_hip_slice");
        let (lo, hi) = l.contiguous_offsets().expect("contiguous");
        slice.slice(lo..hi)
    });

    // --- Output allocation -------------------------------------------
    let out_len = b_sz * n_head * l_q * d;
    // SAFETY: the kernel fully populates `out` before any Rust code
    // reads it via `Tensor::from_storage` below.
    let out = unsafe { dev.alloc::<f32>(out_len)? };

    // --- Launch ------------------------------------------------------
    let func = dev.get_or_load_func(kernel_name, &kernels::FLASH_ATTN)?;

    // Grid: (L_q, n_head, B). Block: 1 Wave64 warp.
    let cfg = LaunchConfig {
        grid_dim: (l_q as u32, n_head as u32, b_sz as u32),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&q_view);
    builder.arg(&k_view);
    builder.arg(&v_view);
    // Mask: optional pointer. Pass null when absent.
    if let Some(mv) = mask_view_owned.as_ref() {
        builder.arg(mv);
    } else {
        builder.arg(&hipdarc::driver::NullDevicePtr::default());
    }
    builder.arg(&out);
    let b_arg = b_sz as i32;
    let n_head_arg = n_head as i32;
    let n_kv_arg = n_kv as i32;
    let l_q_arg = l_q as i32;
    let l_k_arg = l_k as i32;
    let scale_arg = scale as f32;
    let n_rep_arg = n_rep as i32;
    builder.arg(&b_arg);
    builder.arg(&n_head_arg);
    builder.arg(&n_kv_arg);
    builder.arg(&l_q_arg);
    builder.arg(&l_k_arg);
    builder.arg(&scale_arg);
    builder.arg(&n_rep_arg);
    builder.arg(&mask_b_stride);
    builder.arg(&mask_l_q_stride);
    // SAFETY: ffi — preconditions validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(q_st);
    drop(k_st);
    drop(v_st);
    drop(mask_owned);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<(usize, usize, usize, usize)>>::from((b_sz, n_head, l_q, d)),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}

// ======================================================================
// Flash-attention v2 — BR=4 LDS-tiled, D in {64, 128, 256}, prefill path.
// ======================================================================

/// Minimum L_q to route through the v2 kernel. Below this the BR=4
/// block geometry under-utilises CUs; decode (L_q=1) should fall back
/// to rocBLAS via the caller's else branch.
const FLASH_V2_MIN_L_Q: usize = 4;

const FLASH_V2_BR: u32 = 4;

/// Flash-attention forward v2 — BR=4 LDS-tiled.
///
/// Replaces the `rocBLAS QK^T + softmax + rocBLAS SV` chain in
/// `gqa_attention` for prefill (L_q >= FLASH_V2_MIN_L_Q). Covers D ∈
/// {64, 128, 256} which includes gemma4 (D=256). Takes the same
/// additive f32 mask convention as `flash_attn_fused`.
///
/// # Shapes
/// - `q`: `(B, n_head, L_q, D)` f32 contiguous
/// - `k`, `v`: `(B, n_kv_head, L_k, D)` f32 contiguous
/// - `mask`: optional additive f32, broadcast-compatible with
///   `(B, 1, L_q, L_k)` (same shape rules as `flash_attn_fused`)
///
/// # Returns
/// A fresh `(B, n_head, L_q, D)` f32 tensor.
///
/// # Errors
/// Any precondition failure (non-HIP, non-f32, non-contiguous,
/// unsupported head dim, or L_q < FLASH_V2_MIN_L_Q) returns an
/// error so the caller can fall through to the rocBLAS path.
pub fn flash_attn_v2_fused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    // --- Device ------------------------------------------------------
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_v2_fused: q must be on a HIP device"),
    };
    for (name, t) in [("k", k), ("v", v)] {
        if !matches!(t.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!("flash_attn_v2_fused: {name} is on a different device than q");
        }
    }
    if let Some(m) = mask {
        if !matches!(m.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!("flash_attn_v2_fused: mask is on a different device than q");
        }
    }

    // --- Dtype -------------------------------------------------------
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.dtype() != DType::F32 {
            crate::bail!("flash_attn_v2_fused: {name} must be f32, got {:?}", t.dtype());
        }
    }
    if let Some(m) = mask {
        if m.dtype() != DType::F32 {
            crate::bail!("flash_attn_v2_fused: mask must be f32, got {:?}", m.dtype());
        }
    }

    // --- Shapes ------------------------------------------------------
    let (b_sz, n_head, l_q, d) = q.dims4()?;
    let (kb, n_kv, l_k, kd) = k.dims4()?;
    if (kb, kd) != (b_sz, d) {
        crate::bail!(
            "flash_attn_v2_fused: k shape {:?} incompatible with q {:?}",
            k.dims(),
            q.dims()
        );
    }
    let v_dims = v.dims4()?;
    if v_dims != (kb, n_kv, l_k, kd) {
        crate::bail!(
            "flash_attn_v2_fused: v shape {:?} != k shape {:?}",
            v.dims(),
            k.dims()
        );
    }
    if n_head % n_kv != 0 {
        crate::bail!(
            "flash_attn_v2_fused: n_head ({n_head}) not divisible by n_kv_head ({n_kv})"
        );
    }
    let n_rep = n_head / n_kv;

    // --- L_q gate + head dim --------------------------------------
    if l_q < FLASH_V2_MIN_L_Q {
        crate::bail!(
            "flash_attn_v2_fused: L_q={l_q} below minimum {}",
            FLASH_V2_MIN_L_Q
        );
    }
    let kernel_name = match d {
        64 => "flash_attn_v2_fwd_d64_f32",
        128 => "flash_attn_v2_fwd_d128_f32",
        256 => "flash_attn_v2_fwd_d256_f32",
        _ => {
            return Err(HipError::InternalError(Box::leak(
                format!("flash_attn_v2_fused: unsupported head dim {d} (supported: 64, 128, 256)")
                    .into_boxed_str(),
            ))
            .into())
        }
    };

    // --- Contiguity --------------------------------------------------
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if !t.is_contiguous() {
            crate::bail!("flash_attn_v2_fused: {name} must be contiguous");
        }
    }
    if let Some(m) = mask {
        if !m.is_contiguous() {
            crate::bail!("flash_attn_v2_fused: mask must be contiguous when provided");
        }
    }

    // Mask stride derivation — identical to v1.
    let (mask_b_stride, mask_l_q_stride): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            let mdims = m.dims();
            let last = *mdims.last().ok_or_else(|| {
                crate::Error::Msg("flash_attn_v2_fused: mask has empty shape".into())
            })?;
            if last != l_k {
                crate::bail!(
                    "flash_attn_v2_fused: mask last dim {last} != L_k {l_k}"
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
                    "flash_attn_v2_fused: mask second-to-last dim {second_last} not 1 or L_q {l_q}"
                );
            };
            let mask_batch_dim = if mdims.len() >= 4 { mdims[0] } else { 1 };
            let b_stride: i32 = if mask_batch_dim == b_sz {
                (second_last * l_k) as i32
            } else if mask_batch_dim == 1 {
                0
            } else {
                crate::bail!(
                    "flash_attn_v2_fused: mask batch dim {mask_batch_dim} not 1 or B {b_sz}"
                );
            };
            (b_stride, lq_stride)
        }
    };

    // --- Storage handles ---------------------------------------------
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let mask_owned = mask.map(|m| m.storage_and_layout());

    macro_rules! hip_view_f32_v2 {
        ($storage:expr, $layout:expr, $label:literal) => {{
            let hip_st = match &*$storage {
                Storage::Hip(s) => s,
                _ => crate::bail!("flash_attn_v2_fused: {} not a HIP storage", $label),
            };
            let slice = hip_st.as_hip_slice::<f32>()?;
            match $layout.contiguous_offsets() {
                Some((lo, hi)) => slice.slice(lo..hi),
                None => crate::bail!("flash_attn_v2_fused: {} non-contiguous", $label),
            }
        }};
    }

    let q_view = hip_view_f32_v2!(q_st, q_l, "q");
    let k_view = hip_view_f32_v2!(k_st, k_l, "k");
    let v_view = hip_view_f32_v2!(v_st, v_l, "v");
    let mask_view_owned = mask_owned.as_ref().map(|(st, l)| {
        let hip_st = match &**st {
            Storage::Hip(s) => s,
            _ => panic!("flash_attn_v2_fused: mask not HIP (checked above)"),
        };
        let slice = hip_st.as_hip_slice::<f32>().expect("as_hip_slice");
        let (lo, hi) = l.contiguous_offsets().expect("contiguous");
        slice.slice(lo..hi)
    });

    // --- Output allocation -------------------------------------------
    let out_len = b_sz * n_head * l_q * d;
    // SAFETY: kernel fully populates `out` for q_idx < L_q before the
    // Tensor::from_storage wrap below. Rows at block boundaries past
    // L_q are handled by the kernel's `q_in_range` early-exit and
    // don't write to `out`, so their contents remain uninitialised.
    // The output shape matches L_q exactly, so no reader sees those
    // uninitialised slots.
    let out = unsafe { dev.alloc::<f32>(out_len)? };

    // --- Launch ------------------------------------------------------
    let func = dev.get_or_load_func(kernel_name, &kernels::FLASH_ATTN_V2)?;

    // Grid: (ceil(L_q / BR), n_head, B). Block: (WARP_SIZE, BR, 1).
    let grid_x = ((l_q as u32) + FLASH_V2_BR - 1) / FLASH_V2_BR;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_head as u32, b_sz as u32),
        block_dim: (WARP_SIZE, FLASH_V2_BR, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&q_view);
    builder.arg(&k_view);
    builder.arg(&v_view);
    if let Some(mv) = mask_view_owned.as_ref() {
        builder.arg(mv);
    } else {
        builder.arg(&hipdarc::driver::NullDevicePtr::default());
    }
    builder.arg(&out);
    let b_arg = b_sz as i32;
    let n_head_arg = n_head as i32;
    let n_kv_arg = n_kv as i32;
    let l_q_arg = l_q as i32;
    let l_k_arg = l_k as i32;
    let scale_arg = scale as f32;
    let n_rep_arg = n_rep as i32;
    builder.arg(&b_arg);
    builder.arg(&n_head_arg);
    builder.arg(&n_kv_arg);
    builder.arg(&l_q_arg);
    builder.arg(&l_k_arg);
    builder.arg(&scale_arg);
    builder.arg(&n_rep_arg);
    builder.arg(&mask_b_stride);
    builder.arg(&mask_l_q_stride);
    // SAFETY: ffi — preconditions validated above.
    unsafe { builder.launch(cfg) }.w()?;

    drop(q_st);
    drop(k_st);
    drop(v_st);
    drop(mask_owned);

    let out_storage = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<(usize, usize, usize, usize)>>::from((b_sz, n_head, l_q, d)),
        BackpropOp::none(),
        /* is_variable */ false,
    ))
}
