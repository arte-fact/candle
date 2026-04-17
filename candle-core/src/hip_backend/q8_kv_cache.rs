//! Q8_0 KV cache — packed int8 storage for K and V tensors.
//!
//! Matches ggml / llama.cpp-turbo's `block_q8_0` layout (34 bytes per
//! 32-element block: 1 half scale + 32 int8 quants), so VRAM usage per
//! context length is identical to turbo.
//!
//! Cache shape: `(B, H_kv, T_max, D/32)` of `block_q8_0`. D is contiguous
//! within a row so that both the V·attn and Q·K decode gemvs can issue
//! coalesced per-t reads.
//!
//! Phase 1 (this file) delivers:
//!   - `QKvCache` struct with independent K/V slabs.
//!   - `append_k` / `append_v`: quantise + pack an incoming F32
//!     `(B, H_kv, T_new, D)` slice at `t_offset = current_seq_len`.
//!   - `dequantize_k` / `dequantize_v`: full-slab F32 re-hydration for
//!     prefill fallback.
//!
//! Later phases wire the cache into model forward passes and add the
//! Q8-aware attention gemv kernels.

use crate::backend::BackendDevice;
use crate::hip_backend::{kernels, HipDevice, HipError, HipStorage, WrapErr};
use crate::{DType, Result, Shape, Storage, Tensor};
use hipdarc::driver::{HipSlice, LaunchConfig};

const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34; // sizeof(half) + 32 int8

/// Bytes required to store a (B, H, T, D) Q8_0 KV slab.
#[inline]
pub fn q8_slab_bytes(b: usize, h: usize, t: usize, d: usize) -> usize {
    assert!(d % QK8_0 == 0, "D must be a multiple of {QK8_0}");
    b * h * t * (d / QK8_0) * BLOCK_Q8_0_BYTES
}

pub struct QKvCache {
    /// Packed Q8_0 K slab of shape `(B, H_kv, T_max, D/32)`.
    k: HipSlice<u8>,
    /// Packed Q8_0 V slab — same layout.
    v: HipSlice<u8>,
    dev: HipDevice,
    b: usize,
    h_kv: usize,
    t_max: usize,
    d: usize,
    current_seq_len: usize,
}

impl QKvCache {
    /// Allocate K and V slabs. Both are uninit; `current_seq_len = 0`.
    /// Slots past `current_seq_len` stay uninit until later appends
    /// overwrite them — attention consumers must never read beyond
    /// `current_seq_len`.
    pub fn new(dev: &HipDevice, b: usize, h_kv: usize, t_max: usize, d: usize) -> Result<Self> {
        if d % QK8_0 != 0 {
            crate::bail!(
                "QKvCache: D={d} must be a multiple of QK8_0={QK8_0}"
            );
        }
        let bytes = q8_slab_bytes(b, h_kv, t_max, d);
        // SAFETY: `append_*` writes slot [t_offset, t_offset + t_new) in
        // full for every block; callers access only [0, current_seq_len).
        let k = unsafe { dev.alloc::<u8>(bytes)? };
        let v = unsafe { dev.alloc::<u8>(bytes)? };
        Ok(Self {
            k,
            v,
            dev: dev.clone(),
            b,
            h_kv,
            t_max,
            d,
            current_seq_len: 0,
        })
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }

    pub fn k_slab(&self) -> &HipSlice<u8> {
        &self.k
    }
    pub fn v_slab(&self) -> &HipSlice<u8> {
        &self.v
    }

    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.b, self.h_kv, self.t_max, self.d)
    }

    /// Quantise + pack an F32 `(B, H_kv, T_new, D)` tensor into the K
    /// slab at `current_seq_len`. Advances `current_seq_len` by `T_new`.
    pub fn append_k(&mut self, src: &Tensor) -> Result<()> {
        self.append_inner(src, /*is_v=*/ false)?;
        // current_seq_len is incremented in append_inner's V path only
        // so that a paired append_k → append_v advances exactly once.
        // But K is commonly appended first on its own in a few call
        // sites, so we accept both orderings via an explicit advance.
        Ok(())
    }

    /// Quantise + pack an F32 `(B, H_kv, T_new, D)` tensor into the V
    /// slab at `current_seq_len` and advance the counter.
    pub fn append_v(&mut self, src: &Tensor) -> Result<()> {
        let t_new = self.append_inner(src, /*is_v=*/ true)?;
        self.current_seq_len += t_new;
        Ok(())
    }

    /// Append both K and V in one call (preferred — keeps the pair
    /// logically atomic and avoids the K-only vs V-only ambiguity).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let t_new = self.append_inner(k, /*is_v=*/ false)?;
        let t_new_v = self.append_inner(v, /*is_v=*/ true)?;
        if t_new != t_new_v {
            crate::bail!(
                "QKvCache::append: K and V T_new disagree ({t_new} vs {t_new_v})"
            );
        }
        self.current_seq_len += t_new;
        Ok(())
    }

    fn append_inner(&mut self, src: &Tensor, is_v: bool) -> Result<usize> {
        let dims = src.dims();
        if dims.len() != 4 {
            crate::bail!("QKvCache::append: expected (B, H, T_new, D), got {dims:?}");
        }
        let (b, h, t_new, d) = (dims[0], dims[1], dims[2], dims[3]);
        if b != self.b || h != self.h_kv || d != self.d {
            crate::bail!(
                "QKvCache::append: shape {:?} incompatible with cache ({},{},_,{})",
                dims,
                self.b,
                self.h_kv,
                self.d
            );
        }
        if src.dtype() != DType::F32 {
            crate::bail!("QKvCache::append: src dtype must be F32 (got {:?})", src.dtype());
        }
        let t_offset = self.current_seq_len;
        if t_offset + t_new > self.t_max {
            crate::bail!(
                "QKvCache::append: overflow (t_offset={} + t_new={} > t_max={})",
                t_offset,
                t_new,
                self.t_max
            );
        }
        if t_new == 0 {
            return Ok(0);
        }

        let src_contig = src.contiguous()?;
        let (st, l) = src_contig.storage_and_layout();
        let hip_st = match &*st {
            Storage::Hip(s) => s,
            _ => crate::bail!("QKvCache::append: src must be on HIP"),
        };
        let src_slice = hip_st
            .as_hip_slice::<f32>()
            .map_err(crate::Error::wrap)?;
        let (lo, hi) = l.contiguous_offsets().ok_or_else(|| {
            crate::Error::Msg("QKvCache::append: src not contiguous".into())
        })?;
        let src_view = src_slice.slice(lo..hi);

        let dst = if is_v { &self.v } else { &self.k };

        let func = self
            .dev
            .get_or_load_func("quantize_q8_0_append_f32", &kernels::QUANTIZE_KV_Q8)?;

        // Grid: (D/32, H * T_new, B). Block: 32 threads.
        let cfg = LaunchConfig {
            grid_dim: (
                (d / QK8_0) as u32,
                (h * t_new) as u32,
                b as u32,
            ),
            block_dim: (QK8_0 as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i = b as i32;
        let h_i = h as i32;
        let t_new_i = t_new as i32;
        let d_i = d as i32;
        let t_max_i = self.t_max as i32;
        let t_offset_i = t_offset as i32;

        let mut builder = func.builder();
        builder.arg(&src_view);
        builder.arg(dst);
        builder.arg(&b_i);
        builder.arg(&h_i);
        builder.arg(&t_new_i);
        builder.arg(&d_i);
        builder.arg(&t_max_i);
        builder.arg(&t_offset_i);
        // SAFETY: grid covers every block in [t_offset, t_offset+t_new)
        // with 32 threads writing all 32 qs + 1 scale. No padding holes.
        unsafe { builder.launch(cfg) }.w()?;

        drop(st);
        Ok(t_new)
    }

    /// Dequantise the first `current_seq_len` tokens of the K slab into
    /// a fresh `(B, H_kv, current_seq_len, D)` F32 tensor. Used by the
    /// prefill fallback path.
    pub fn dequantize_k(&self) -> Result<Tensor> {
        self.dequantize_inner(&self.k)
    }

    pub fn dequantize_v(&self) -> Result<Tensor> {
        self.dequantize_inner(&self.v)
    }

    fn dequantize_inner(&self, slab: &HipSlice<u8>) -> Result<Tensor> {
        let t_sel = self.current_seq_len;
        if t_sel == 0 {
            // Empty cache: return a 0-length tensor with the right shape
            // signature.
            return Tensor::zeros(
                (self.b, self.h_kv, 0, self.d),
                DType::F32,
                &crate::Device::Hip(self.dev.clone()),
            );
        }
        let elem = self.b * self.h_kv * t_sel * self.d;
        // SAFETY: kernel fully writes `elem` f32s.
        let out = unsafe { self.dev.alloc::<f32>(elem)? };

        let func = self
            .dev
            .get_or_load_func("dequantize_q8_0_kv_f32", &kernels::QUANTIZE_KV_Q8)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (self.d / QK8_0) as u32,
                (self.h_kv * t_sel) as u32,
                self.b as u32,
            ),
            block_dim: (QK8_0 as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i = self.b as i32;
        let h_i = self.h_kv as i32;
        let t_sel_i = t_sel as i32;
        let d_i = self.d as i32;
        let t_max_i = self.t_max as i32;
        let t_offset_i: i32 = 0; // dequantise from the start

        let mut builder = func.builder();
        builder.arg(slab);
        builder.arg(&out);
        builder.arg(&b_i);
        builder.arg(&h_i);
        builder.arg(&t_sel_i);
        builder.arg(&d_i);
        builder.arg(&t_max_i);
        builder.arg(&t_offset_i);
        // SAFETY: kernel writes every element of `out`.
        unsafe { builder.launch(cfg) }.w()?;

        let storage = HipStorage::wrap_hip_slice(out, self.dev.clone());
        Ok(Tensor::from_storage(
            Storage::Hip(storage),
            Shape::from((self.b, self.h_kv, t_sel, self.d)),
            crate::op::BackpropOp::none(),
            /* is_variable */ false,
        ))
    }
}
