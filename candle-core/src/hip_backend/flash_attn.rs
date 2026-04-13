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

    let (mbs, mls): (i32, i32) = match mask {
        None => (0, 0),
        Some(m) => {
            if m.dtype() != DType::F32 || !m.is_contiguous() {
                crate::bail!("mask bad");
            }
            let md = m.dims();
            let last = *md.last().unwrap();
            if last != l_k { crate::bail!("mask last!=lk"); }
            let sl = if md.len() >= 2 { md[md.len()-2] } else { 1 };
            let mb = if md.len() >= 4 { md[0] } else { 1 };
            (if mb==b { (sl*l_k) as i32 } else { 0 }, if sl==lq { l_k as i32 } else { 0 })
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
