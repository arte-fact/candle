//! Fused gated delta net (GDN) recurrent step — HIP launcher.
//!
//! Replaces the ~8-op tensor-op chain in
//! `candle-transformers/src/models/quantized_blocks/delta_net.rs::
//! delta_net_step_vectorized` with a single `gated_delta_net.cu` kernel
//! invocation per call site.
//!
//! Phase 1 scope: f32, `S_v = 128`, KDA=false (scalar gate per head).
//! Matches every qwen35 / qwen3next variant the project currently runs.
//! Other `S_v` values and dtypes are a mechanical extension of the .cu
//! templates — add a new `extern "C"` wrapper and an entry in
//! [`KERNEL_FOR_S_V`].
//!
//! The kernel consumes `(B, H, L, S_v)` q / k / v, `(B, H, L)` gate /
//! beta, and the old `(B, H, S_v, S_v)` state. It produces a fresh
//! attention output tensor **and** a fresh state tensor, and replaces
//! the caller's `state: &mut Tensor` with the new one. Allocating a
//! fresh state buffer instead of mutating the existing one keeps the
//! Rust-side borrow structure simple — every tensor is read through a
//! plain read lock, no `storage_mut_and_layout` gymnastics — and the
//! workspace pool returns the old buffer for reuse on drop.
//!
//! All inputs must be f32 and contiguous — validation is explicit so
//! shape errors fail fast at the Rust side instead of silently
//! corrupting device memory.

use crate::backend::BackendDevice;
use crate::op::BackpropOp;
use crate::{DType, Device, Result, Shape, Storage, Tensor};
use hipdarc::driver::LaunchConfig;

use super::{kernels, HipError, HipStorage, WrapErr};

/// gfx906 Wave64 size. Mirrored from `candle-hip-kernels/build.rs` which
/// passes `-DWARP_SIZE=64` to `hipcc`.
const WARP_SIZE: u32 = 64;

/// Warps per block. Matches the upstream turbo layout — one Wave64 warp
/// per output column, four warps per SIMD so the compiler has room to
/// overlap issue across columns of the same `(b, h)`.
const NUM_WARPS: u32 = 4;

/// Run one fused gated-delta-net recurrent step for `L` tokens on HIP.
///
/// # Shapes
/// - `q`, `k`, `v`: `(B, H, L, S_v)` — f32, contiguous
/// - `gate`, `beta`: `(B, H, L, 1)` or `(B, H, L)` — f32, contiguous.
///   One scalar per `(b, h, t)`; the kernel reads them flat so either
///   rank is fine.
/// - `state`: `(B, H, S_v, S_v)` — f32, contiguous. On return
///   `*state` points to a fresh tensor with the updated state; the old
///   backing buffer is returned to the workspace pool by its `Drop`.
///
/// # Returns
/// A fresh tensor `attn_out: (B, H, L, S_v)` containing the per-token
/// attention output rows. The caller is responsible for any downstream
/// reshape / norm / gate.
///
/// # Errors
/// - Any tensor is on a non-HIP device, or not f32, or non-contiguous.
/// - `S_v` is not a supported size (Phase 1 only handles 128).
/// - Any tensor's shape disagrees with the inferred `(B, H, L, S_v)`.
pub fn gated_delta_net_step_fused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    // --- Device ------------------------------------------------------
    let dev = match q.device() {
        Device::Hip(d) => d.clone(),
        _ => crate::bail!("gated_delta_net_step_fused: q must be on a HIP device"),
    };
    for (name, t) in [
        ("k", k),
        ("v", v),
        ("gate", gate),
        ("beta", beta),
        ("state", &*state),
    ] {
        if !matches!(t.device(), Device::Hip(d) if d.same_device(&dev)) {
            crate::bail!(
                "gated_delta_net_step_fused: {name} is on a different device than q"
            );
        }
    }

    // --- Dtype -------------------------------------------------------
    for (name, t) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("gate", gate),
        ("beta", beta),
        ("state", &*state),
    ] {
        if t.dtype() != DType::F32 {
            crate::bail!(
                "gated_delta_net_step_fused: {name} must be f32, got {:?}",
                t.dtype()
            );
        }
    }

    // --- Shapes ------------------------------------------------------
    let (b_sz, h_v, seq_len, s_v) = q.dims4()?;
    if (b_sz, h_v, seq_len, s_v) != k.dims4()? {
        crate::bail!(
            "gated_delta_net_step_fused: k shape {:?} != q shape {:?}",
            k.dims(),
            q.dims()
        );
    }
    if (b_sz, h_v, seq_len, s_v) != v.dims4()? {
        crate::bail!(
            "gated_delta_net_step_fused: v shape {:?} != q shape {:?}",
            v.dims(),
            q.dims()
        );
    }
    // gate / beta: elem count must be B*H*L. Accept any rank as long as
    // total elements match — the kernel reads a flat buffer.
    let gb_elems = b_sz * h_v * seq_len;
    if gate.elem_count() != gb_elems {
        crate::bail!(
            "gated_delta_net_step_fused: gate has {} elements, expected B*H*L = {}",
            gate.elem_count(),
            gb_elems
        );
    }
    if beta.elem_count() != gb_elems {
        crate::bail!(
            "gated_delta_net_step_fused: beta has {} elements, expected B*H*L = {}",
            beta.elem_count(),
            gb_elems
        );
    }
    // State: (B, H, S_v, S_v)
    let (sb, sh, s1, s2) = state.dims4()?;
    if (sb, sh, s1, s2) != (b_sz, h_v, s_v, s_v) {
        crate::bail!(
            "gated_delta_net_step_fused: state shape {:?} != (B={b_sz}, H={h_v}, S_v={s_v}, S_v={s_v})",
            state.dims()
        );
    }

    // --- Supported S_v ----------------------------------------------
    let kernel_name = kernel_for_s_v(s_v).ok_or_else(|| {
        HipError::InternalError(Box::leak(
            format!("gated_delta_net_step_fused: unsupported S_v={s_v} (Phase 1 supports {{128}})")
                .into_boxed_str(),
        ))
    })?;

    // --- Contiguity --------------------------------------------------
    //
    // The kernel walks every buffer as a flat `contiguous_offsets()`
    // range. Non-contiguous inputs would either need a strided variant
    // or an explicit `.contiguous()` call by the caller — we want the
    // caller to be explicit, so we refuse here instead of silently
    // materialising a copy that defeats the whole fusion goal.
    for (name, t) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("gate", gate),
        ("beta", beta),
        ("state", &*state),
    ] {
        if !t.is_contiguous() {
            crate::bail!(
                "gated_delta_net_step_fused: {name} must be contiguous"
            );
        }
    }

    // --- Storage handles ---------------------------------------------
    //
    // Every input is read-only. `state` is read as `state_in`; the
    // kernel writes a fresh `state_out` buffer, which we wrap as a new
    // Tensor and assign to the caller's `*state` at the end.
    let (q_st, q_l) = q.storage_and_layout();
    let (k_st, k_l) = k.storage_and_layout();
    let (v_st, v_l) = v.storage_and_layout();
    let (g_st, g_l) = gate.storage_and_layout();
    let (b_st, b_l) = beta.storage_and_layout();
    let (s_st, s_l) = state.storage_and_layout();

    macro_rules! hip_slice_f32 {
        ($storage:expr, $layout:expr, $label:literal) => {{
            let hip_st = match &*$storage {
                Storage::Hip(s) => s,
                _ => crate::bail!(
                    "gated_delta_net_step_fused: {} is not a HIP storage",
                    $label
                ),
            };
            let slice = hip_st.as_hip_slice::<f32>()?;
            let view = match $layout.contiguous_offsets() {
                Some((lo, hi)) => slice.slice(lo..hi),
                None => {
                    crate::bail!(
                        "gated_delta_net_step_fused: {} has non-contiguous storage",
                        $label
                    )
                }
            };
            view
        }};
    }

    let q_view = hip_slice_f32!(q_st, q_l, "q");
    let k_view = hip_slice_f32!(k_st, k_l, "k");
    let v_view = hip_slice_f32!(v_st, v_l, "v");
    let g_view = hip_slice_f32!(g_st, g_l, "gate");
    let b_view = hip_slice_f32!(b_st, b_l, "beta");
    let s_in_view = hip_slice_f32!(s_st, s_l, "state");

    // Fresh output buffers. Both hit the workspace pool on the second
    // forward onwards (same shapes every step for a given model).
    let attn_out_len = b_sz * h_v * seq_len * s_v;
    let state_out_len = b_sz * h_v * s_v * s_v;
    // SAFETY: both buffers are fully populated by the kernel before any
    // Rust code reads them.
    let attn_out = unsafe { dev.alloc::<f32>(attn_out_len)? };
    let state_out = unsafe { dev.alloc::<f32>(state_out_len)? };

    // --- Launch ------------------------------------------------------
    let func = dev.get_or_load_func(kernel_name, &kernels::GATED_DELTA_NET)?;

    // Grid: (H, B, ceil(S_v / NUM_WARPS))
    // Block: (WARP_SIZE, NUM_WARPS, 1)
    let grid_z = (s_v as u32).div_ceil(NUM_WARPS);
    let cfg = LaunchConfig {
        grid_dim: (h_v as u32, b_sz as u32, grid_z),
        block_dim: (WARP_SIZE, NUM_WARPS, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&q_view);
    builder.arg(&k_view);
    builder.arg(&v_view);
    builder.arg(&g_view);
    builder.arg(&b_view);
    builder.arg(&s_in_view);
    builder.arg(&state_out);
    builder.arg(&attn_out);
    let b_arg = b_sz as i32;
    let h_arg = h_v as i32;
    let l_arg = seq_len as i32;
    builder.arg(&b_arg);
    builder.arg(&h_arg);
    builder.arg(&l_arg);
    // SAFETY: ffi — kernel contract is validated above (shapes, dtype, contiguous).
    unsafe { builder.launch(cfg) }.w()?;

    // Drop read locks before constructing the output Tensors so any
    // downstream op that borrows these storages doesn't block on our guards.
    drop(q_st);
    drop(k_st);
    drop(v_st);
    drop(g_st);
    drop(b_st);
    drop(s_st);

    // Wrap the fresh buffers as Tensors.
    let attn_tensor = Tensor::from_storage(
        Storage::Hip(HipStorage::wrap_hip_slice(attn_out, dev.clone())),
        <Shape as From<(usize, usize, usize, usize)>>::from((b_sz, h_v, seq_len, s_v)),
        BackpropOp::none(),
        /* is_variable */ false,
    );
    let state_tensor = Tensor::from_storage(
        Storage::Hip(HipStorage::wrap_hip_slice(state_out, dev)),
        <Shape as From<(usize, usize, usize, usize)>>::from((b_sz, h_v, s_v, s_v)),
        BackpropOp::none(),
        /* is_variable */ false,
    );
    // Replace the caller's state with the new buffer. The old buffer's
    // Drop routes it back to the workspace pool.
    *state = state_tensor;
    Ok(attn_tensor)
}

/// Map `S_v` → exported kernel name. Phase 1 only supports `S_v = 128`;
/// adding a new entry requires writing a matching `extern "C"` wrapper
/// in `candle-hip-kernels/src/gated_delta_net.cu` and no other change.
fn kernel_for_s_v(s_v: usize) -> Option<&'static str> {
    match s_v {
        128 => Some("gated_delta_net_step_s128_f32"),
        _ => None,
    }
}
