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

/// Flash attention v2 with K-transposed layout `(B, H_kv, D, L_k)`.
pub fn flash_attn_v2_kt_fused(
    q: &Tensor, k_t: &Tensor, v: &Tensor, mask: Option<&Tensor>, scale: f64,
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_v2_kt: q must be HIP"),
    };
    for (n, t) in [("q", q), ("k_t", k_t), ("v", v)] {
        if t.dtype() != DType::F32 { crate::bail!("flash_attn_v2_kt: {n} must be f32"); }
        if !t.is_contiguous() { crate::bail!("flash_attn_v2_kt: {n} must be contiguous"); }
    }
    let (b, nh, lq, d) = q.dims4()?;
    let (_, nkv, kd, lk) = k_t.dims4()?;
    if kd != d { crate::bail!("flash_attn_v2_kt: k_t D mismatch"); }
    if nh % nkv != 0 { crate::bail!("flash_attn_v2_kt: nh%nkv!=0"); }
    let nrep = nh / nkv;
    if lq < FLASH_V2_MIN_L_Q { crate::bail!("flash_attn_v2_kt: lq<min"); }
    let kn = match d {
        64 => "flash_attn_v2_fwd_kt_d64_f32",
        128 => "flash_attn_v2_fwd_kt_d128_f32",
        256 => "flash_attn_v2_fwd_kt_d256_f32",
        _ => crate::bail!("flash_attn_v2_kt: unsupported D={d}"),
    };
    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            if m.dtype() != DType::F32 || !m.is_contiguous() {
                crate::bail!("flash_attn_v2_kt: mask bad");
            }
            let md = m.dims();
            let last = *md.last().unwrap();
            if last != lk { crate::bail!("flash_attn_v2_kt: mask last!=lk"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb==b { (sl*lk) as i32 } else { 0 }, if sl==lq { lk as i32 } else { 0 })
        }
    };
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k_t.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let mo = mask.map(|m| m.storage_and_layout());
    macro_rules! hv { ($s:expr,$l:expr) => {{
        let h = match &*$s { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl = h.as_hip_slice::<f32>()?;
        let (lo,hi) = $l.contiguous_offsets().unwrap(); sl.slice(lo..hi)
    }}; }
    let qv=hv!(qs,ql); let kv=hv!(ks,kl); let vv=hv!(vs,vl);
    let mv = mo.as_ref().map(|(st,l)| {
        let h=match &**st { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl=h.as_hip_slice::<f32>().unwrap();
        let(lo,hi)=l.contiguous_offsets().unwrap(); sl.slice(lo..hi)
    });
    let out = unsafe { dev.alloc::<f32>(b*nh*lq*d)? };
    let func = dev.get_or_load_func(kn, &kernels::FLASH_ATTN_V2)?;
    let gx = ((lq as u32)+FLASH_V2_BR-1)/FLASH_V2_BR;
    let cfg = LaunchConfig {
        grid_dim:(gx, nh as u32, b as u32),
        block_dim:(WARP_SIZE, FLASH_V2_BR, 1),
        shared_mem_bytes:0,
    };
    let mut bld = func.builder();
    bld.arg(&qv); bld.arg(&kv); bld.arg(&vv);
    match mv.as_ref() { Some(v)=>{bld.arg(v);} None=>{bld.arg(&hipdarc::driver::NullDevicePtr::default());} }
    bld.arg(&out);
    let(ba,nha,nka,lqa,lka)=(b as i32,nh as i32,nkv as i32,lq as i32,lk as i32);
    let(sa,nra)=(scale as f32,nrep as i32);
    bld.arg(&ba);bld.arg(&nha);bld.arg(&nka);bld.arg(&lqa);bld.arg(&lka);
    bld.arg(&sa);bld.arg(&nra);bld.arg(&mbs);bld.arg(&mls);
    unsafe { bld.launch(cfg) }.w()?;
    drop(qs);drop(ks);drop(vs);drop(mo);
    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(Storage::Hip(os),
        <Shape as From<(usize,usize,usize,usize)>>::from((b,nh,lq,d)),
        BackpropOp::none(), false))
}

/// Decode attention with strided K/V — for KvCache pre-allocated buffers.
///
/// Q is `(B, n_head, L_q, D)` contiguous.
/// K and V are accessed via explicit strides per head:
///   K[head][j][d] = k_ptr[head * k_head_stride + j * k_stride_j + d * k_stride_d]
///   V[head][j][d] = v_ptr[head * v_head_stride + j * v_stride_j + d * v_stride_d]
///
/// For KvCache V stored as (D, max_T) per head:
///   v_stride_j = 1, v_stride_d = max_T
/// For KvCache K stored as (D, max_T) per head:
///   k_stride_j = 1, k_stride_d = max_T
///
/// This replaces rocBLAS for the attention matmul during decode,
/// reading ONLY the actual T_cur elements from V (not max_T).
pub fn flash_attn_decode_strided(
    q: &Tensor,       // (B, n_head, L_q, D) contiguous
    k: &Tensor,       // (B, n_kv_head, ...) with arbitrary strides
    v: &Tensor,       // (B, n_kv_head, ...) with arbitrary strides
    mask: Option<&Tensor>,
    scale: f64,
    l_k: usize,       // actual number of KV positions (T_cur)
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_decode_strided: q must be HIP"),
    };

    let (b_sz, n_head, l_q, d) = q.dims4()?;
    if q.dtype() != DType::F32 { crate::bail!("flash_attn_decode_strided: q must be f32"); }

    let n_kv = k.dim(1)?;
    if n_head % n_kv != 0 { crate::bail!("flash_attn_decode_strided: n_head % n_kv != 0"); }
    let n_rep = n_head / n_kv;

    let kernel_name = match d {
        64 => "flash_attn_decode_d64_f32",
        128 => "flash_attn_decode_d128_f32",
        256 => "flash_attn_decode_d256_f32",
        512 => "flash_attn_decode_d512_f32",
        _ => crate::bail!("flash_attn_decode_strided: unsupported D={d}"),
    };

    // Mask strides
    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            let md = m.dims();
            let last = *md.last().unwrap();
            if last != l_k { crate::bail!("flash_attn_decode_strided: mask last != l_k"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb == b_sz { (sl*l_k) as i32 } else { 0 },
             if sl == l_q { l_k as i32 } else { 0 })
        }
    };

    // Extract raw pointers and strides
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let mask_owned = mask.map(|m| m.storage_and_layout());

    macro_rules! hv {
        ($s:expr, $l:expr) => {{
            let h = match &*$s { Storage::Hip(s) => s, _ => panic!("not HIP") };
            let sl = h.as_hip_slice::<f32>()?;
            let off = $l.start_offset();
            sl.slice(off..sl.len())
        }};
    }

    let q_v = hv!(q_st, q_l);
    let k_v = hv!(k_st, k_l);
    let v_v = hv!(v_st, v_l);
    let m_v = mask_owned.as_ref().map(|(st, l)| {
        let h = match &**st { Storage::Hip(s) => s, _ => panic!("not HIP") };
        let sl = h.as_hip_slice::<f32>().unwrap();
        let off = l.start_offset();
        sl.slice(off..sl.len())
    });

    // K strides: K shape (B, n_kv, D, T) from k_cache.current_data() (dim=3
    // last). strides = (H*D*maxT, D*maxT, maxT, 1). Dim 2 = D, dim 3 = T.
    let k_strides = k_l.stride();
    let k_head_stride = k_strides[1] as i32;
    let k_stride_d = k_strides[2] as i32;  // stride along D dim (= maxT)
    let k_stride_j = k_strides[3] as i32;  // stride along T dim (= 1)

    // V strides: V shape (B, H, T, D) from narrow(dim=3,T)+transpose(2,3).
    // strides = (H*D*maxT, D*maxT, 1, maxT). Dim 2 = T, dim 3 = D.
    let v_strides = v_l.stride();
    let v_head_stride = v_strides[1] as i32;
    let v_stride_j = v_strides[2] as i32;  // stride along T dim (= 1)
    let v_stride_d = v_strides[3] as i32;  // stride along D dim (= maxT)

    let out_len = b_sz * n_head * l_q * d;
    let out = unsafe { dev.alloc::<f32>(out_len)? };
    let func = dev.get_or_load_func(kernel_name, &kernels::FLASH_ATTN)?;

    let cfg = LaunchConfig {
        grid_dim: (l_q as u32, n_head as u32, b_sz as u32),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut bld = func.builder();
    bld.arg(&q_v); bld.arg(&k_v); bld.arg(&v_v);
    match m_v.as_ref() {
        Some(v) => { bld.arg(v); }
        None => { bld.arg(&hipdarc::driver::NullDevicePtr::default()); }
    }
    bld.arg(&out);
    let (ba, nha, nka, lqa, lka) = (b_sz as i32, n_head as i32, n_kv as i32, l_q as i32, l_k as i32);
    let (sa, nra) = (scale as f32, n_rep as i32);
    bld.arg(&ba); bld.arg(&nha); bld.arg(&nka); bld.arg(&lqa); bld.arg(&lka);
    bld.arg(&sa); bld.arg(&nra); bld.arg(&mbs); bld.arg(&mls);
    bld.arg(&k_head_stride); bld.arg(&k_stride_j); bld.arg(&k_stride_d);
    bld.arg(&v_head_stride); bld.arg(&v_stride_j); bld.arg(&v_stride_d);
    unsafe { bld.launch(cfg) }.w()?;

    drop(q_st); drop(k_st); drop(v_st); drop(mask_owned);

    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(os),
        <Shape as From<(usize,usize,usize,usize)>>::from((b_sz, n_head, l_q, d)),
        BackpropOp::none(), false,
    ))
}

/// Split-K decode flash attention. Partitions the K dimension across
/// multiple workgroups per head to fill gfx906's 60 CUs.
///
/// Phase 1 kernel: grid=(num_chunks, n_head, B), each workgroup produces
/// unnormalized partial (o, m, l) for its K-chunk.
/// Phase 2 (merge): grid=(n_head, B, 1), combines partials via log-sum-exp.
///
/// Used for decode (L_q=1) when T_k is large enough for splitting to help.
pub fn flash_attn_decode_strided_split_k(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    l_k: usize,
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_decode_strided_split_k: q must be HIP"),
    };

    let (b_sz, n_head, l_q, d) = q.dims4()?;
    if l_q != 1 {
        crate::bail!("flash_attn_decode_strided_split_k: requires L_q=1, got {l_q}");
    }
    if q.dtype() != DType::F32 {
        crate::bail!("flash_attn_decode_strided_split_k: q must be f32");
    }

    let n_kv = k.dim(1)?;
    if n_head % n_kv != 0 {
        crate::bail!("flash_attn_decode_strided_split_k: n_head % n_kv != 0");
    }
    let n_rep = n_head / n_kv;

    // Prefer the LDS-tiled variant (cooperative K/V load into shared memory,
    // amortising the uncoalesced K stride over BC reuse). Fall back to the
    // plain direct-stream variant via env var CANDLE_SPLITK_NOLDS=1.
    let use_lds = std::env::var("CANDLE_SPLITK_NOLDS").is_err();
    let (phase1_name, merge_name) = match (d, use_lds) {
        (64, true) => ("flash_attn_decode_split_k_lds_phase1_d64_f32",
                       "flash_attn_decode_split_k_merge_d64_f32"),
        (128, true) => ("flash_attn_decode_split_k_lds_phase1_d128_f32",
                        "flash_attn_decode_split_k_merge_d128_f32"),
        (64, false) => ("flash_attn_decode_split_k_phase1_d64_f32",
                        "flash_attn_decode_split_k_merge_d64_f32"),
        (128, false) => ("flash_attn_decode_split_k_phase1_d128_f32",
                         "flash_attn_decode_split_k_merge_d128_f32"),
        _ => crate::bail!("flash_attn_decode_strided_split_k: unsupported D={d}"),
    };

    // Pick num_chunks: gfx906 has 60 CUs. n_head=32 heads per B=1 means
    // num_chunks=2 gives 64 workgroups (saturates), =4 gives 128 (2-deep),
    // =8 gives 256 (4-deep — reduces per-chunk serial work at cost of
    // more merges). With LDS tiling each chunk runs BC=64 positions per
    // tile, so chunk_size should be >= BC to keep the inner loop busy.
    let num_chunks: i32 = if let Ok(v) = std::env::var("CANDLE_SPLITK_CHUNKS") {
        v.parse().unwrap_or(1)
    } else if l_k < 64 {
        1
    } else if l_k < 128 {
        2
    } else if l_k < 512 {
        4
    } else if l_k < 2048 {
        8
    } else {
        16
    };

    // Mask strides (same as non-split kernel).
    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            let md = m.dims();
            let last = *md.last().unwrap();
            if last != l_k { crate::bail!("split_k: mask last != l_k"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb == b_sz { (sl*l_k) as i32 } else { 0 },
             if sl == l_q { l_k as i32 } else { 0 })
        }
    };

    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let mask_owned = mask.map(|m| m.storage_and_layout());

    macro_rules! hv {
        ($s:expr, $l:expr) => {{
            let h = match &*$s { Storage::Hip(s) => s, _ => panic!("not HIP") };
            let sl = h.as_hip_slice::<f32>()?;
            let off = $l.start_offset();
            sl.slice(off..sl.len())
        }};
    }

    let q_v = hv!(q_st, q_l);
    let k_v = hv!(k_st, k_l);
    let v_v = hv!(v_st, v_l);
    let m_v = mask_owned.as_ref().map(|(st, l)| {
        let h = match &**st { Storage::Hip(s) => s, _ => panic!("not HIP") };
        let sl = h.as_hip_slice::<f32>().unwrap();
        let off = l.start_offset();
        sl.slice(off..sl.len())
    });

    // K shape (B, n_kv, D, T), V shape (B, H, T, D) from narrow+transpose.
    // See flash_attn_decode_strided for the stride derivation.
    let k_strides = k_l.stride();
    let k_head_stride = k_strides[1] as i32;
    let k_stride_d = k_strides[2] as i32;  // maxT
    let k_stride_j = k_strides[3] as i32;  // 1

    let v_strides = v_l.stride();
    let v_head_stride = v_strides[1] as i32;
    let v_stride_j = v_strides[2] as i32;  // 1
    let v_stride_d = v_strides[3] as i32;  // maxT

    // Partial buffer: [num_chunks, B, n_head, D+2] f32. ~16*1*32*66*4 = 132KB.
    let partial_len = num_chunks as usize * b_sz * n_head * (d + 2);
    let partial = unsafe { dev.alloc::<f32>(partial_len)? };
    let out = unsafe { dev.alloc::<f32>(b_sz * n_head * l_q * d)? };

    let debug = std::env::var("CANDLE_SPLITK_DEBUG").is_ok();
    if debug {
        eprintln!(
            "[splitK] l_k={} num_chunks={} partial_len={} B={} n_head={} D={}",
            l_k, num_chunks, partial_len, b_sz, n_head, d
        );
    }

    let phase1 = dev.get_or_load_func(phase1_name, &kernels::FLASH_ATTN)?;
    let merge  = dev.get_or_load_func(merge_name,  &kernels::FLASH_ATTN)?;

    // Phase 1 launch: (num_chunks, n_head, B)
    // LDS variant uses 2 warps per block (NW=2) for cooperative K tile load.
    // Non-LDS variant uses 1 warp per block.
    {
        let block_y = if use_lds { 2 } else { 1 };
        let cfg = LaunchConfig {
            grid_dim: (num_chunks as u32, n_head as u32, b_sz as u32),
            block_dim: (WARP_SIZE, block_y, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = phase1.builder();
        bld.arg(&q_v); bld.arg(&k_v); bld.arg(&v_v);
        match m_v.as_ref() {
            Some(v) => { bld.arg(v); }
            None => { bld.arg(&hipdarc::driver::NullDevicePtr::default()); }
        }
        bld.arg(&partial);
        let (ba, nha, nka, lka) = (b_sz as i32, n_head as i32, n_kv as i32, l_k as i32);
        let (sa, nra) = (scale as f32, n_rep as i32);
        let nc = num_chunks as i32;
        bld.arg(&ba); bld.arg(&nha); bld.arg(&nka); bld.arg(&lka);
        bld.arg(&sa); bld.arg(&nra); bld.arg(&mbs); bld.arg(&mls);
        bld.arg(&k_head_stride); bld.arg(&k_stride_j); bld.arg(&k_stride_d);
        bld.arg(&v_head_stride); bld.arg(&v_stride_j); bld.arg(&v_stride_d);
        bld.arg(&nc);
        unsafe { bld.launch(cfg) }.w()?;
    }

    // Phase 2 (merge): (n_head, B, 1)
    {
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, b_sz as u32, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = merge.builder();
        bld.arg(&partial); bld.arg(&out);
        let (ba, nha, nc) = (b_sz as i32, n_head as i32, num_chunks as i32);
        bld.arg(&ba); bld.arg(&nha); bld.arg(&nc);
        unsafe { bld.launch(cfg) }.w()?;
    }

    drop(q_st); drop(k_st); drop(v_st); drop(mask_owned);

    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(os),
        <Shape as From<(usize,usize,usize,usize)>>::from((b_sz, n_head, l_q, d)),
        BackpropOp::none(), false,
    ))
}

/// Flash attention v2 with K-transposed AND strided V, with a runtime-
/// patchable "real L_k" that cuts iteration below the allocated K extent.
///
/// `l_k_alloc` is the K buffer's dim(3) — used for K's stride along D and
/// for the mask row stride. `l_k_iter` is how many sequence positions to
/// actually attend to. Under G2 replay, `l_k_alloc` is kept at the
/// captured padded value (stable across replays) while `l_k_iter` is
/// classified as a Counter(+1) arg by the plan — one kernel arg ticks
/// forward per decode token.
///
/// When `l_k_iter == l_k_alloc`, the behavior matches
/// `flash_attn_v2_kt_strided_v`.
pub fn flash_attn_v2_kt_strided_v_dyn(
    q: &Tensor, k_t: &Tensor, v: &Tensor, mask: Option<&Tensor>,
    scale: f64, l_k_alloc: usize, l_k_iter: usize,
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_v2_kt_strided_v_dyn: q must be HIP"),
    };
    let (b, nh, lq, d) = q.dims4()?;
    let nkv = k_t.dim(1)?;
    if nh % nkv != 0 { crate::bail!("nh%nkv!=0"); }
    let nrep = nh / nkv;
    if l_k_iter > l_k_alloc {
        crate::bail!("l_k_iter ({l_k_iter}) > l_k_alloc ({l_k_alloc})");
    }

    let kn = match d {
        64 => "flash_attn_v2_fwd_ktvs_dyn_d64_f32",
        128 => "flash_attn_v2_fwd_ktvs_dyn_d128_f32",
        256 => "flash_attn_v2_fwd_ktvs_dyn_d256_f32",
        512 => "flash_attn_v2_fwd_ktvs_dyn_d512_f32",
        _ => crate::bail!("unsupported D={d}"),
    };

    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            if m.dtype() != DType::F32 || !m.is_contiguous() {
                crate::bail!("mask bad");
            }
            let md = m.dims();
            let last = *md.last().unwrap();
            if last < l_k_iter { crate::bail!("mask last<l_k_iter"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb==b { (sl*last) as i32 } else { 0 }, if sl==lq { last as i32 } else { 0 })
        }
    };

    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k_t.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let mo = mask.map(|m| m.storage_and_layout());

    macro_rules! hv { ($s:expr,$l:expr) => {{
        let h = match &*$s { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl = h.as_hip_slice::<f32>()?;
        let off = $l.start_offset(); sl.slice(off..sl.len())
    }}; }
    let qv=hv!(qs,ql); let kv=hv!(ks,kl); let vv=hv!(vs,vl);
    let mv = mo.as_ref().map(|(st,l)| {
        let h=match &**st { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl=h.as_hip_slice::<f32>().unwrap();
        let off=l.start_offset(); sl.slice(off..sl.len())
    });

    let v_strides = vl.stride();
    let v_head_stride = v_strides[1] as i32;
    let v_stride_j = v_strides[2] as i32;
    let v_stride_d = v_strides[3] as i32;
    // Explicit K strides so the caller can pass a narrow'd (non-contiguous)
    // K view. For gemma4's KvCache K layout (B, H_kv, D, maxT) narrow'd to
    // (B, H_kv, D, l_k_alloc), the strides are
    //   (H_kv*D*maxT, D*maxT, maxT, 1)
    // i.e. stride-along-D = maxT (not l_k_alloc), stride-along-seq = 1.
    // The kernel needs both so it can index K[d * maxT + row] without the
    // caller having to .contiguous() first.
    let k_strides = kl.stride();
    let k_head_stride = k_strides[1] as i32;
    let k_stride_d = k_strides[2] as i32;
    let k_stride_j = k_strides[3] as i32;

    let out = unsafe { dev.alloc::<f32>(b*nh*lq*d)? };
    let func = dev.get_or_load_func(kn, &kernels::FLASH_ATTN_V2)?;

    let gx = ((lq as u32)+FLASH_V2_BR-1)/FLASH_V2_BR;
    let cfg = LaunchConfig {
        grid_dim:(gx, nh as u32, b as u32),
        block_dim:(WARP_SIZE, FLASH_V2_BR, 1),
        shared_mem_bytes:0,
    };
    let mut bld = func.builder();
    bld.arg(&qv); bld.arg(&kv); bld.arg(&vv);
    match mv.as_ref() { Some(v)=>{bld.arg(v);} None=>{bld.arg(&hipdarc::driver::NullDevicePtr::default());} }
    bld.arg(&out);
    let(ba,nha,nka,lqa,lka_alloc)=(b as i32,nh as i32,nkv as i32,lq as i32,l_k_alloc as i32);
    let(sa,nra)=(scale as f32,nrep as i32);
    let lka_iter = l_k_iter as i32;
    bld.arg(&ba);bld.arg(&nha);bld.arg(&nka);bld.arg(&lqa);bld.arg(&lka_alloc);
    bld.arg(&sa);bld.arg(&nra);bld.arg(&mbs);bld.arg(&mls);
    bld.arg(&v_head_stride);bld.arg(&v_stride_j);bld.arg(&v_stride_d);
    bld.arg(&lka_iter);
    bld.arg(&k_head_stride);bld.arg(&k_stride_d);bld.arg(&k_stride_j);
    unsafe { bld.launch(cfg) }.w()?;
    drop(qs);drop(ks);drop(vs);drop(mo);
    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(Storage::Hip(os),
        <Shape as From<(usize,usize,usize,usize)>>::from((b,nh,lq,d)),
        BackpropOp::none(), false))
}

/// Flash attention v2 with K-transposed AND strided V.
/// K is (B, H_kv, D, L_k) contiguous. V is (B, H_kv, T, D) non-contiguous
/// from KvCache narrow+transpose with strides (H*D*maxT, D*maxT, 1, maxT).
pub fn flash_attn_v2_kt_strided_v(
    q: &Tensor, k_t: &Tensor, v: &Tensor, mask: Option<&Tensor>,
    scale: f64, l_k: usize,
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("flash_attn_v2_kt_strided_v: q must be HIP"),
    };
    let (b, nh, lq, d) = q.dims4()?;
    let nkv = k_t.dim(1)?;
    if nh % nkv != 0 { crate::bail!("nh%nkv!=0"); }
    let nrep = nh / nkv;
    // Strided variant supports L_q=1 (decode) via q_in_range guard in the kernel.

    let kn = match d {
        64 => "flash_attn_v2_fwd_ktvs_d64_f32",
        128 => "flash_attn_v2_fwd_ktvs_d128_f32",
        256 => "flash_attn_v2_fwd_ktvs_d256_f32",
        512 => "flash_attn_v2_fwd_ktvs_d512_f32",
        _ => crate::bail!("unsupported D={d}"),
    };

    // Mask shape: typically (1,1,L_q,L_k_pad) where L_k_pad may be >= l_k.
    // For G2 dynamic-l_k (the captured plan advances l_k each replay while
    // mask stays allocated at L_k_pad), `last >= l_k` — the kernel iterates
    // [0..l_k) in the K dim and only touches mask entries up to l_k, so
    // L_k_pad > l_k is safe as long as the per-row stride reflects the
    // allocation, not the loop bound.
    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            if m.dtype() != DType::F32 || !m.is_contiguous() {
                crate::bail!("mask bad");
            }
            let md = m.dims();
            let last = *md.last().unwrap();
            if last < l_k { crate::bail!("mask last<lk"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb==b { (sl*last) as i32 } else { 0 }, if sl==lq { last as i32 } else { 0 })
        }
    };

    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k_t.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let mo = mask.map(|m| m.storage_and_layout());

    macro_rules! hv { ($s:expr,$l:expr) => {{
        let h = match &*$s { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl = h.as_hip_slice::<f32>()?;
        let off = $l.start_offset(); sl.slice(off..sl.len())
    }}; }
    let qv=hv!(qs,ql); let kv=hv!(ks,kl); let vv=hv!(vs,vl);
    let mv = mo.as_ref().map(|(st,l)| {
        let h=match &**st { Storage::Hip(s)=>s, _=>panic!("!hip") };
        let sl=h.as_hip_slice::<f32>().unwrap();
        let off=l.start_offset(); sl.slice(off..sl.len())
    });

    // V strides for the kernel.
    let v_strides = vl.stride();
    let v_head_stride = v_strides[1] as i32;
    // V is (B, H, T, D) non-contiguous from narrow+transpose:
    // strides = (H*D*maxT, D*maxT, 1, maxT)
    // The kernel needs: v_stride_j (stride along T/seq) and v_stride_d (stride along D).
    // V[j][d] = v_ptr[j * stride[2] + d * stride[3]]
    let v_stride_j = v_strides[2] as i32;  // stride along T (= 1 for transposed)
    let v_stride_d = v_strides[3] as i32;  // stride along D (= maxT for transposed)

    let out = unsafe { dev.alloc::<f32>(b*nh*lq*d)? };
    let func = dev.get_or_load_func(kn, &kernels::FLASH_ATTN_V2)?;

    let gx = ((lq as u32)+FLASH_V2_BR-1)/FLASH_V2_BR;
    let cfg = LaunchConfig {
        grid_dim:(gx, nh as u32, b as u32),
        block_dim:(WARP_SIZE, FLASH_V2_BR, 1),
        shared_mem_bytes:0,
    };
    let mut bld = func.builder();
    bld.arg(&qv); bld.arg(&kv); bld.arg(&vv);
    match mv.as_ref() { Some(v)=>{bld.arg(v);} None=>{bld.arg(&hipdarc::driver::NullDevicePtr::default());} }
    bld.arg(&out);
    let(ba,nha,nka,lqa,lka)=(b as i32,nh as i32,nkv as i32,lq as i32,l_k as i32);
    let(sa,nra)=(scale as f32,nrep as i32);
    bld.arg(&ba);bld.arg(&nha);bld.arg(&nka);bld.arg(&lqa);bld.arg(&lka);
    bld.arg(&sa);bld.arg(&nra);bld.arg(&mbs);bld.arg(&mls);
    bld.arg(&v_head_stride);bld.arg(&v_stride_j);bld.arg(&v_stride_d);
    unsafe { bld.launch(cfg) }.w()?;
    drop(qs);drop(ks);drop(vs);drop(mo);
    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(Storage::Hip(os),
        <Shape as From<(usize,usize,usize,usize)>>::from((b,nh,lq,d)),
        BackpropOp::none(), false))
}

/// Decode-only L_q=1 attention via rocBLAS strided-batched sgemv. Mirrors
/// the kernel turbo's llama.cpp dispatch (`mul_mat_vec_f<float,float,...>`)
/// uses for E4B-style wide-head decode (D=256/512). For each `q_row`
/// (n_rep entries per kv_head), one batched sgemv per direction:
///
/// 1. attn[h, r, :]  = K[h]^T · q[h, r, :]  (m=t_k, n=D, NoTrans)
/// 2. softmax(scale * attn + mask)
/// 3. o[h, r, :]     = V[h]^T · attn[h, r, :]  (m=t_k, n=D, Trans)
///
/// Shapes:
/// - `q`:    (B, n_head,    L_q=1, D) contiguous f32
/// - `k_t`:  (B, n_kv_head, D, T_pad) — narrow'd KvCache view
///   (strides `(H*D*max_T, D*max_T, max_T, 1)`)
/// - `v`:    (B, n_kv_head, T_pad, D) — narrow'd KvCache transpose view
///   (strides `(H*D*max_T, D*max_T, 1, max_T)`)
/// - `mask`: optional `(1, 1, 1, T_mask)` f32 with `T_mask >= t_k`
///
/// `t_k` is the actual attended length (≤ T_pad). The kernel reads K[..., 0..t_k]
/// and V[0..t_k, ...] only.
///
/// Returns `(B, n_head, L_q=1, D)`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_gemv(
    q: &Tensor,
    k_t: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    t_k: usize,
) -> Result<Tensor> {
    use hipdarc::rocblas::{sgemv_strided_batched, GemmOp, StridedBatchedSgemvConfig};

    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("gqa_attention_decode_gemv: q must be HIP"),
    };
    let (b, n_head, l_q, d) = q.dims4()?;
    if l_q != 1 {
        crate::bail!("gqa_attention_decode_gemv: requires L_q=1, got {l_q}");
    }
    if q.dtype() != DType::F32 {
        crate::bail!("gqa_attention_decode_gemv: q must be f32");
    }
    if !q.is_contiguous() {
        crate::bail!("gqa_attention_decode_gemv: q must be contiguous");
    }

    let n_kv = k_t.dim(1)?;
    if n_head % n_kv != 0 {
        crate::bail!("gqa_attention_decode_gemv: n_head % n_kv != 0");
    }
    let n_rep = n_head / n_kv;

    let kd = k_t.dim(2)?;
    if kd != d {
        crate::bail!("gqa_attention_decode_gemv: K dim2 ({kd}) != q dim3 ({d})");
    }

    // K strides: (H*D*max_T, D*max_T, max_T, 1) — see callsite docs.
    // We need head stride and dim-D stride; the kernel reads K[d * max_T + j].
    let k_strides = k_t.layout().stride();
    let k_head_stride = k_strides[1] as i64;
    let k_lda = k_strides[2] as i32; // = max_T

    // V layout depends on the cache: gemma4 keeps V canonical (T_pad, D)
    // row-major, so strides are (H*D*T_pad, D*T_pad, D, 1). Some other
    // models pre-transpose V to (D, T_pad) — strides (..., 1, T_pad).
    // Detect by checking which of stride[2]/stride[3] is == 1 (the
    // contiguous dim).
    let v_strides = v.layout().stride();
    let v_head_stride = v_strides[1] as i64;
    let v_t_contig = v_strides[2] == 1; // V is (D, T_pad) with T-axis contiguous
    let v_lda = if v_t_contig {
        v_strides[3] as i32 // T_pad
    } else {
        v_strides[2] as i32 // D, V is (T_pad, D) D-axis contiguous
    };

    // Storage + base pointers for Q/K/V.
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k_t.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();

    let q_hip = match &*q_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gqa_attention_decode_gemv: q storage not HIP"),
    };
    let k_hip = match &*k_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gqa_attention_decode_gemv: k storage not HIP"),
    };
    let v_hip = match &*v_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gqa_attention_decode_gemv: v storage not HIP"),
    };
    let q_slice = q_hip.as_hip_slice::<f32>()?;
    let k_slice = k_hip.as_hip_slice::<f32>()?;
    let v_slice = v_hip.as_hip_slice::<f32>()?;

    let q_base = q_slice.slice(q_l.start_offset()..q_slice.len());
    let k_base = k_slice.slice(k_l.start_offset()..k_slice.len());
    let v_base = v_slice.slice(v_l.start_offset()..v_slice.len());

    // Mask handling — only used in step 2's masked_softmax_scale_fused.
    // Validate shape compat now so we bail before allocating.
    if let Some(m) = mask {
        let md = m.dims();
        let last = *md.last().unwrap();
        if last < t_k {
            crate::bail!("gqa_attention_decode_gemv: mask last ({last}) < t_k ({t_k})");
        }
    }

    let blas = dev.rocblas_handle()?;

    // Per-head Q layout under reshape (B, n_kv, n_rep*L_q=n_rep, D):
    //   q[b][h_kv][r][d] = q_orig[b][h_kv*n_rep + r][0][d]
    // Q is contiguous so memory is unchanged: stride is (H_kv*n_rep*D, n_rep*D, D, 1).
    // For sgemv batched on h_kv with fixed r:
    //   per-batch x base = q_base + h_kv * (n_rep*D) + r * D
    //   stride_x (per batch step) = n_rep * D
    let q_per_row_stride = (n_rep * d) as i64;

    // Allocate attn_weights buffer (B, H_kv, n_rep, t_k) contiguous f32.
    let attn_total_elts = b * n_kv * n_rep * t_k;
    let attn_buf = unsafe { dev.alloc::<f32>(attn_total_elts)? };
    let attn_view = attn_buf.slice(0..attn_buf.len());

    // Step 1: attn[h, r, :] = K[h]^T · q[h, r, :]
    // rocBLAS NoTrans on A=K_h (rocblas-view (T, D), lda=max_T) gives
    // y = A · x : T-vector. We want this for each (h, r). Stride between
    // batches (varying h, fixed r) for output: per-h stride = n_rep * t_k.
    let cfg_qk = StridedBatchedSgemvConfig {
        trans: GemmOp::NoTrans,
        m: t_k as i32,
        n: d as i32,
        lda: k_lda,
        stride_a: k_head_stride,
        incx: 1,
        stride_x: q_per_row_stride,
        incy: 1,
        stride_y: (n_rep * t_k) as i64,
        batch_count: n_kv as i32,
    };
    for r in 0..n_rep {
        let q_off = r * d;
        let y_off = r * t_k;
        unsafe {
            sgemv_strided_batched(
                blas,
                &cfg_qk,
                1.0,
                k_base.device_ptr() as *const f32,
                (q_base.device_ptr() as *const f32).wrapping_add(q_off),
                0.0,
                (attn_view.device_ptr() as *mut f32).wrapping_add(y_off),
            )
            .map_err(|e| HipError::InternalError(
                Box::leak(format!("sgemv K^T·Q failed: {e:?}").into_boxed_str())
            ))?;
        }
    }

    // Step 2: softmax(scale * attn + mask). Reshape attn to (B, n_head, 1, t_k)
    // — zero-copy since memory layout matches.
    let attn_storage = HipStorage::wrap_hip_slice(attn_buf, dev.clone());
    let attn_tensor = Tensor::from_storage(
        Storage::Hip(attn_storage),
        <Shape as From<(usize, usize, usize, usize)>>::from((b, n_head, 1usize, t_k)),
        BackpropOp::none(),
        false,
    );
    let attn_softmaxed = super::masked_softmax_scale_fused(&attn_tensor, mask, scale)?;

    // Step 3: o[h, r, :] = V[h]^T · attn[h, r, :]
    // Output layout (B, H_kv, n_rep, D) contiguous → reshape (B, n_head, 1, D).
    let out_total_elts = b * n_kv * n_rep * d;
    let out_buf = unsafe { dev.alloc::<f32>(out_total_elts)? };
    let out_view = out_buf.slice(0..out_buf.len());

    // Re-borrow attn after softmax (it may be a fresh tensor).
    let (att_st, att_l) = attn_softmaxed.storage_and_layout();
    let att_hip = match &*att_st {
        Storage::Hip(s) => s,
        _ => crate::bail!("gqa_attention_decode_gemv: softmax storage not HIP"),
    };
    let att_slice = att_hip.as_hip_slice::<f32>()?;
    let att_base = att_slice.slice(att_l.start_offset()..att_slice.len());

    // For V step: O = V^T · attn (D-vec).
    // - If V is (T_pad, D) canonical (D-contig): A_rocblas = V^T view via
    //   col-major with lda=D, op=NoTrans, m=D, n=t_k.
    // - If V is (D, T_pad) pre-transposed (T-contig): A_rocblas = V via
    //   col-major with lda=T_pad, op=Trans, m=t_k, n=D.
    let cfg_av = if v_t_contig {
        StridedBatchedSgemvConfig {
            trans: GemmOp::Trans,
            m: t_k as i32,
            n: d as i32,
            lda: v_lda,
            stride_a: v_head_stride,
            incx: 1,
            stride_x: (n_rep * t_k) as i64,
            incy: 1,
            stride_y: (n_rep * d) as i64,
            batch_count: n_kv as i32,
        }
    } else {
        StridedBatchedSgemvConfig {
            trans: GemmOp::NoTrans,
            m: d as i32,
            n: t_k as i32,
            lda: v_lda,
            stride_a: v_head_stride,
            incx: 1,
            stride_x: (n_rep * t_k) as i64,
            incy: 1,
            stride_y: (n_rep * d) as i64,
            batch_count: n_kv as i32,
        }
    };
    for r in 0..n_rep {
        let x_off = r * t_k;
        let y_off = r * d;
        unsafe {
            sgemv_strided_batched(
                blas,
                &cfg_av,
                1.0,
                v_base.device_ptr() as *const f32,
                (att_base.device_ptr() as *const f32).wrapping_add(x_off),
                0.0,
                (out_view.device_ptr() as *mut f32).wrapping_add(y_off),
            )
            .map_err(|e| HipError::InternalError(
                Box::leak(format!("sgemv V^T·attn failed: {e:?}").into_boxed_str())
            ))?;
        }
    }

    drop(q_st);
    drop(k_st);
    drop(v_st);
    drop(att_st);

    let out_storage = HipStorage::wrap_hip_slice(out_buf, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(out_storage),
        <Shape as From<(usize, usize, usize, usize)>>::from((b, n_head, 1usize, d)),
        BackpropOp::none(),
        false,
    ))
}

/// Phase P — Decode attention via mat-vec kernel (L_q=1 only).
///
/// Assumes K and V are both stored **row-major (B, H_kv, T_pad, D)**
/// with D as the contiguous axis — llama.cpp-turbo's layout. Use this
/// only when the KvCache was created with `new_k_canonical_stable`.
///
/// One block per (b, h_q). 64 threads (one warp). Threads stride D
/// coalesced via `lane + i * WARP_SIZE`. No LDS K/V tiling — direct
/// global reads per t, warp-reduce on the partial dot product.
///
/// `l_k_alloc` is K/V's T-axis size (the padded or allocated extent);
/// `l_k_iter` is how many real positions to attend to (≤ l_k_alloc).
/// Pass `l_k_iter == l_k_alloc` when not using dynamic-L_k.
///
/// Returns `(B, n_head, 1, D)`.
pub fn gqa_attention_decode_mv(
    q: &Tensor,                 // (B, n_head, 1, D) f32 contiguous
    k: &Tensor,                 // (B, n_kv_head, L_k_alloc, D) f32 contiguous
    v: &Tensor,                 // (B, n_kv_head, L_k_alloc, D) f32 contiguous
    mask: Option<&Tensor>,      // optional (1,1,1,T_mask) or (B,1,1,T_mask) f32
    scale: f64,
    l_k_iter: usize,
) -> Result<Tensor> {
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("gqa_attention_decode_mv: q must be HIP"),
    };
    let (b, n_head, l_q, d) = q.dims4()?;
    if l_q != 1 {
        crate::bail!("gqa_attention_decode_mv: requires L_q=1, got {l_q}");
    }
    if !q.is_contiguous() || q.dtype() != DType::F32 {
        crate::bail!("gqa_attention_decode_mv: q must be f32 contiguous");
    }

    let (kb, n_kv, l_k_alloc, kd) = k.dims4()?;
    if (kb, kd) != (b, d) {
        crate::bail!(
            "gqa_attention_decode_mv: k shape {:?} incompatible with q {:?}",
            k.dims(), q.dims()
        );
    }
    if !k.is_contiguous() || k.dtype() != DType::F32 {
        crate::bail!("gqa_attention_decode_mv: k must be f32 contiguous");
    }
    let (vb, v_nkv, v_lk, vd) = v.dims4()?;
    if (vb, v_nkv, v_lk, vd) != (b, n_kv, l_k_alloc, d) {
        crate::bail!(
            "gqa_attention_decode_mv: v shape {:?} != expected {:?}",
            v.dims(), (b, n_kv, l_k_alloc, d)
        );
    }
    if !v.is_contiguous() || v.dtype() != DType::F32 {
        crate::bail!("gqa_attention_decode_mv: v must be f32 contiguous");
    }
    if n_head % n_kv != 0 {
        crate::bail!("gqa_attention_decode_mv: n_head % n_kv != 0");
    }
    let n_rep = n_head / n_kv;
    if l_k_iter > l_k_alloc {
        crate::bail!(
            "gqa_attention_decode_mv: l_k_iter ({l_k_iter}) > l_k_alloc ({l_k_alloc})"
        );
    }

    // Phase Q2: `gqa_decode_mv_fast_d{256,512}_f32` uses float2 loads +
    // 128-thread block + cross-warp LDS reduce. Targets the 73 μs/call
    // vs turbo's 9 μs gap. Fast kernel is the default on compatible
    // shapes (D ∈ {256, 512}); smaller heads fall back to the original
    // scalar 64-thread kernel where the gain is negligible.
    // Opt-out via `CANDLE_DECODE_MV_FAST=0` for bisecting regressions.
    let use_fast = matches!(d, 256 | 512)
        && std::env::var("CANDLE_DECODE_MV_FAST")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true);

    // Phase T2: when G2 is enabled, route to the `_ctr` (counter-buffer)
    // variant. The kernel reads L_k_iter from a stable device buffer slot
    // instead of taking it as a value arg — eliminates per-replay
    // hipGraphExecKernelNodeSetParams calls for this kernel (was 42 of
    // 326 dynamic ops on E4B Q4_0 = ~13 % of the patch storm).
    //
    // Both recording and replay must use the same kernel sig, so we
    // gate on G2-enabled (recording always happens then) rather than
    // G3-enabled-and-currently-replaying.
    let g2_enabled = std::env::var("CANDLE_G2_REPLAY")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false);
    let (kernel_name, block_dim_x): (&str, u32) = if use_fast {
        match (d, g2_enabled) {
            (256, true)  => ("gqa_decode_mv_fast_d256_f32_ctr", WARP_SIZE),
            (256, false) => ("gqa_decode_mv_fast_d256_f32",     WARP_SIZE),
            (512, true)  => ("gqa_decode_mv_fast_d512_f32_ctr", WARP_SIZE),
            (512, false) => ("gqa_decode_mv_fast_d512_f32",     WARP_SIZE),
            _ => unreachable!(),
        }
    } else {
        match d {
            64 => ("gqa_decode_mv_d64_f32", WARP_SIZE),
            128 => ("gqa_decode_mv_d128_f32", WARP_SIZE),
            256 => ("gqa_decode_mv_d256_f32", WARP_SIZE),
            512 => ("gqa_decode_mv_d512_f32", WARP_SIZE),
            _ => crate::bail!("gqa_attention_decode_mv: unsupported D={d}"),
        }
    };
    let use_ctr = use_fast && g2_enabled;

    // Mask: (1,1,1,T_mask) or (B,1,1,T_mask). T_mask must be ≥ l_k_iter.
    let mbs: i32 = match mask {
        None => 0,
        Some(m) => {
            if m.dtype() != DType::F32 || !m.is_contiguous() {
                crate::bail!("gqa_attention_decode_mv: mask must be f32 contiguous");
            }
            let md = m.dims();
            let last = *md.last().unwrap();
            if last < l_k_iter {
                crate::bail!(
                    "gqa_attention_decode_mv: mask last dim {last} < l_k_iter {l_k_iter}"
                );
            }
            // Per-batch stride: for (B,1,1,T) give T; for broadcast (1,...) give 0.
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            if mb == b { last as i32 } else { 0 }
        }
    };

    // Extract device slices via layout.
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let m_owned = mask.map(|m| m.storage_and_layout());

    macro_rules! hv {
        ($s:expr, $l:expr) => {{
            let h = match &*$s { Storage::Hip(s) => s, _ => panic!("not HIP") };
            let sl = h.as_hip_slice::<f32>()?;
            let off = $l.start_offset();
            sl.slice(off..sl.len())
        }};
    }
    let q_v = hv!(q_st, q_l);
    let k_v = hv!(k_st, k_l);
    let v_v = hv!(v_st, v_l);
    let m_v = m_owned.as_ref().map(|(st, l)| {
        let h = match &**st { Storage::Hip(s) => s, _ => panic!("not HIP") };
        let sl = h.as_hip_slice::<f32>().unwrap();
        let off = l.start_offset();
        sl.slice(off..sl.len())
    });

    let out = unsafe { dev.alloc::<f32>(b * n_head * d)? };

    let (ba, nha, nka, lka) = (b as i32, n_head as i32, n_kv as i32, l_k_alloc as i32);
    let sa = scale as f32;
    let nra = n_rep as i32;
    let lk_it = l_k_iter as i32;

    // Phase R1: split-L_k attention (Flash-Decoding pattern).
    // At low H_q the unsplit kernel only launches 8 wavefronts → 1 wave
    // per active SIMD on MI50, no latency hiding for the loop-carried
    // softmax state. Splitting T into chunks puts CHUNKS×H_q×B waves on
    // the GPU. Validated by arXiv 2311.01282 (3.93× decode on AMD).
    //
    // Default chunk_t = 32 (E4B sweep: chunk=16 helps long but regresses
    // short, chunk=64 is the inverse — 32 is the balanced sweet spot).
    // Threshold = 96: below this, kernel-launch overhead of the 2-kernel
    // sequence exceeds the gain.
    //
    // Default-ON. Best-of-3 on E4B Q4_0:
    //   Short prompt: 60 → 65 t/s (+8.7%)
    //   Long prompt:  48 → 63 t/s (+31%)
    // Opt-out via CANDLE_FLASH_SPLIT_LK=0 for bisecting regressions.
    let split_lk_enabled = std::env::var("CANDLE_FLASH_SPLIT_LK")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(true);
    let chunk_t: usize = std::env::var("CANDLE_FLASH_SPLIT_CHUNK_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let split_threshold: usize = std::env::var("CANDLE_FLASH_SPLIT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    // Phase T1: compute n_chunks from l_k_alloc (the padded extent that
    // stays constant across a pad window) instead of l_k_iter (which
    // advances per token). Why: the captured G2/G3 graph freezes
    // grid_dim at recording time; if l_k_iter advances and changes
    // n_chunks, runtime grid mismatches the captured kernel-node
    // grid → silent miscompute or skipped chunks. The split kernel
    // already early-exits chunks where `t0 >= L_k`, so excess chunks
    // launched against l_k_alloc cost only ~5 µs each but produce
    // correct output. Scratch is also sized once per pad window.
    //
    // For non-G2 default-path callers, this just rounds n_chunks up to
    // the pad multiple — same correctness, ≤ pad_t/chunk_t extra
    // empty-chunk launches per call (worst case 8 for pad=256/chunk=32).
    let n_chunks = (l_k_alloc + chunk_t - 1) / chunk_t;
    let use_split = use_fast
        && split_lk_enabled
        && l_k_iter >= split_threshold
        && n_chunks >= 2;

    if use_split {
        // Scratch: partial_o[B, n_head, n_chunks, D]; partial_m,l[B, n_head, n_chunks].
        let scratch_o = unsafe { dev.alloc::<f32>(b * n_head * n_chunks * d)? };
        let scratch_m = unsafe { dev.alloc::<f32>(b * n_head * n_chunks)? };
        let scratch_l = unsafe { dev.alloc::<f32>(b * n_head * n_chunks)? };

        let split_name = match d {
            256 => "gqa_decode_split_chunk_d256_f32",
            512 => "gqa_decode_split_chunk_d512_f32",
            _ => unreachable!(),
        };
        let combine_name = match d {
            256 => "gqa_decode_split_combine_d256_f32",
            512 => "gqa_decode_split_combine_d512_f32",
            _ => unreachable!(),
        };

        let split_func   = dev.get_or_load_func(split_name,   &kernels::FLASH_ATTN_V2)?;
        let combine_func = dev.get_or_load_func(combine_name, &kernels::FLASH_ATTN_V2)?;

        let split_cfg = LaunchConfig {
            grid_dim: (n_chunks as u32, n_head as u32, b as u32),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let combine_cfg = LaunchConfig {
            grid_dim: (n_head as u32, b as u32, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let chunk_t_i = chunk_t as i32;
        let n_chunks_i = n_chunks as i32;

        let mut bld = split_func.builder();
        bld.arg(&q_v); bld.arg(&k_v); bld.arg(&v_v);
        match m_v.as_ref() {
            Some(v) => { bld.arg(v); }
            None => { bld.arg(&hipdarc::driver::NullDevicePtr::default()); }
        }
        bld.arg(&scratch_o); bld.arg(&scratch_m); bld.arg(&scratch_l);
        bld.arg(&ba); bld.arg(&nha); bld.arg(&nka); bld.arg(&lka);
        bld.arg(&sa); bld.arg(&nra); bld.arg(&mbs);
        bld.arg(&chunk_t_i); bld.arg(&n_chunks_i);
        unsafe { bld.launch(split_cfg) }.w()?;

        let mut bld = combine_func.builder();
        bld.arg(&scratch_o); bld.arg(&scratch_m); bld.arg(&scratch_l);
        bld.arg(&out);
        bld.arg(&ba); bld.arg(&nha); bld.arg(&n_chunks_i);
        unsafe { bld.launch(combine_cfg) }.w()?;
    } else {
        let func = dev.get_or_load_func(kernel_name, &kernels::FLASH_ATTN_V2)?;
        let cfg = LaunchConfig {
            grid_dim: (1, n_head as u32, b as u32),
            block_dim: (block_dim_x, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase T2: prepare counter-buffer slot for L_k_iter when using
        // the `_ctr` kernel variant. The slot pointer is stable across
        // calls; the value is updated by `g3_counters::set` BEFORE this
        // launch (in-stream async memcpy on the device's primary stream).
        // Captured by G3 as a constant pointer arg → zero per-replay
        // set_kernel_node_params for this op.
        if use_ctr {
            crate::hip_backend::g3_counters::set(
                &dev,
                crate::hip_backend::g3_counters::CounterSlot::LkIter,
                l_k_iter as u32,
            )?;
        }
        let lk_iter_ptr: usize = if use_ctr {
            crate::hip_backend::g3_counters::slot_ptr(
                &dev,
                crate::hip_backend::g3_counters::CounterSlot::LkIter,
            ) as usize
        } else {
            0
        };

        let mut bld = func.builder();
        bld.arg(&q_v); bld.arg(&k_v); bld.arg(&v_v);
        match m_v.as_ref() {
            Some(v) => { bld.arg(v); }
            None => { bld.arg(&hipdarc::driver::NullDevicePtr::default()); }
        }
        bld.arg(&out);
        bld.arg(&ba); bld.arg(&nha); bld.arg(&nka); bld.arg(&lka);
        bld.arg(&sa); bld.arg(&nra); bld.arg(&mbs);
        if use_ctr {
            // Pass the device pointer as a usize (8 bytes on 64-bit) —
            // the kernel signature has `const int * L_k_iter_ptr`;
            // bld.arg(&usize) writes the pointer value which the kernel
            // dereferences on the device.
            bld.arg(&lk_iter_ptr);
        } else {
            bld.arg(&lk_it);
        }
        unsafe { bld.launch(cfg) }.w()?;
    }

    drop(q_st); drop(k_st); drop(v_st); drop(m_owned);

    let os = HipStorage::wrap_hip_slice(out, dev);
    Ok(Tensor::from_storage(
        Storage::Hip(os),
        <Shape as From<(usize, usize, usize, usize)>>::from((b, n_head, 1usize, d)),
        BackpropOp::none(),
        false,
    ))
}
