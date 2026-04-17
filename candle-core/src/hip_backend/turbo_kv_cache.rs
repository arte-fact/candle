//! TurboQuant KV cache — TQ2_0, TQ3_0, TQ4_0 variants.
//!
//! Companion to `q8_kv_cache.rs`. Same interface (alloc, append, dequantize)
//! but for the three TurboQuant variants ported from llama.cpp-turbo
//! (`ggml-cuda/turbo-quant.cu` + `dequantize.cuh` + `cpy-utils.cuh`).
//!
//! Block sizes per 32 elements (TQ4 is per 128):
//! - TQ2_0: 10 B  → 2.50 bits/val, 6.40× smaller than F16
//! - TQ3_0: 16 B  → 4.00 bits/val, 4.00× smaller than F16
//! - TQ4_0: 68 B /128 elems → 4.25 bits/val, 3.76× smaller than F16
//!
//! NB: the centroid tables are Lloyd-Max codebooks tuned for unit-norm
//! rotated-Gaussian inputs. Without the graph-level WHT + dense rotation
//! (phase TQ3), quality on non-Gaussian K/V is poor. These kernels cover
//! the plumbing; phase TQ3 adds the preconditioning that makes the
//! codebooks actually work well.

use crate::backend::BackendDevice;
use crate::hip_backend::{kernels, HipDevice, HipError, HipStorage, WrapErr};
use crate::{DType, Result, Shape, Storage, Tensor};
use hipdarc::driver::{HipSlice, LaunchConfig};

/// When this is set, `QTurboKvCache::append` applies the WHT + sign
/// preconditioning from turbo (`turbo_wht.cu`) to K and V before
/// quantising, and `dequantize_{k,v}` applies the inverse WHT after
/// dequantising. Matches turbo's fattn-q3 scheme — the Lloyd-Max
/// codebooks expect unit-Gaussian distributions, which the signed
/// FWHT produces from most K/V distributions.
fn wht_enabled() -> bool {
    std::env::var("CANDLE_KV_TQ_NO_WHT").ok().as_deref() != Some("1")
}

/// Apply the signed 128-element FWHT to a `(B, H_kv, T, D)` F32 tensor
/// in place of an out-of-place copy. `kind` selects K vs V sign tables;
/// `inverse=true` uses the sign-reversed kernel variant that undoes
/// the forward transform.
fn apply_wht_128(
    t: &Tensor,
    dev: &HipDevice,
    kind_is_v: bool,
    inverse: bool,
) -> Result<Tensor> {
    let dims = t.dims();
    if dims.len() != 4 {
        crate::bail!("apply_wht_128: expected 4D, got {dims:?}");
    }
    let d = dims[3];
    if d % 128 != 0 {
        // WHT block size is 128; head_dims that aren't a multiple of
        // 128 bypass preconditioning (e.g. head_dim=64 TinyLlama).
        return Ok(t.clone());
    }
    let n_elem: usize = dims.iter().product();
    let n_groups = n_elem / 128;

    let t_contig = t.contiguous()?;
    let (st, l) = t_contig.storage_and_layout();
    let hip_st = match &*st {
        Storage::Hip(s) => s,
        _ => crate::bail!("apply_wht_128: tensor must be on HIP"),
    };
    let src_slice = hip_st.as_hip_slice::<f32>().map_err(crate::Error::wrap)?;
    let (lo, hi) = l.contiguous_offsets().ok_or_else(|| {
        crate::Error::Msg("apply_wht_128: tensor not contiguous".into())
    })?;
    let src_view = src_slice.slice(lo..hi);

    let out = unsafe { dev.alloc::<f32>(n_elem)? };
    let kn = match (kind_is_v, inverse) {
        (false, false) => "turbo_wht_k_forward",
        (false, true)  => "turbo_wht_k_inverse",
        (true,  false) => "turbo_wht_v_forward",
        (true,  true)  => "turbo_wht_v_inverse",
    };
    let func = dev.get_or_load_func(kn, &kernels::TURBO_WHT)?;

    let cfg = LaunchConfig {
        grid_dim: (n_groups as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_groups_i = n_groups as i64;
    let mut builder = func.builder();
    builder.arg(&src_view);
    builder.arg(&out);
    builder.arg(&n_groups_i);
    unsafe { builder.launch(cfg) }.w()?;

    drop(st);

    let storage = HipStorage::wrap_hip_slice(out, dev.clone());
    Ok(Tensor::from_storage(
        Storage::Hip(storage),
        Shape::from(dims),
        crate::op::BackpropOp::none(),
        /* is_variable */ false,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbVariant {
    Tq2_0,
    Tq3_0,
    Tq4_0,
}

impl TbVariant {
    pub fn block_size_elems(self) -> usize {
        match self {
            TbVariant::Tq2_0 => 32,
            TbVariant::Tq3_0 => 32,
            TbVariant::Tq4_0 => 128,
        }
    }

    pub fn block_size_bytes(self) -> usize {
        match self {
            TbVariant::Tq2_0 => 10,
            TbVariant::Tq3_0 => 16,
            TbVariant::Tq4_0 => 68,
        }
    }

    pub fn bits_per_val(self) -> f32 {
        (self.block_size_bytes() as f32 * 8.0) / self.block_size_elems() as f32
    }

    pub fn quantize_kernel(self) -> &'static str {
        match self {
            TbVariant::Tq2_0 => "quantize_turbo2_0_append_f32",
            TbVariant::Tq3_0 => "quantize_turbo3_0_append_f32",
            TbVariant::Tq4_0 => "quantize_turbo4_0_append_f32",
        }
    }

    pub fn dequantize_kernel(self) -> &'static str {
        match self {
            TbVariant::Tq2_0 => "dequantize_turbo2_0_kv_f32",
            TbVariant::Tq3_0 => "dequantize_turbo3_0_kv_f32",
            TbVariant::Tq4_0 => "dequantize_turbo4_0_kv_f32",
        }
    }

    /// Thread-block size (= block_size_elems); each thread handles one value.
    pub fn block_dim_x(self) -> u32 {
        self.block_size_elems() as u32
    }
}

/// Bytes required to store a (B, H, T, D) TurboQuant KV slab.
#[inline]
pub fn tb_slab_bytes(v: TbVariant, b: usize, h: usize, t: usize, d: usize) -> usize {
    let qk = v.block_size_elems();
    assert!(d % qk == 0, "D={d} must be a multiple of block size {qk}");
    b * h * t * (d / qk) * v.block_size_bytes()
}

#[derive(Debug)]
pub struct QTurboKvCache {
    k: HipSlice<u8>,
    v: HipSlice<u8>,
    dev: HipDevice,
    variant: TbVariant,
    b: usize,
    h_kv: usize,
    t_max: usize,
    d: usize,
    current_seq_len: usize,
}

impl QTurboKvCache {
    pub fn new(
        dev: &HipDevice,
        variant: TbVariant,
        b: usize,
        h_kv: usize,
        t_max: usize,
        d: usize,
    ) -> Result<Self> {
        let qk = variant.block_size_elems();
        if d % qk != 0 {
            crate::bail!("QTurboKvCache: D={d} must be a multiple of block size {qk} for {:?}", variant);
        }
        let bytes = tb_slab_bytes(variant, b, h_kv, t_max, d);
        // SAFETY: kernel writes every byte of every block it touches;
        // callers read only [0, current_seq_len).
        let k = unsafe { dev.alloc::<u8>(bytes)? };
        let v = unsafe { dev.alloc::<u8>(bytes)? };
        Ok(Self {
            k,
            v,
            dev: dev.clone(),
            variant,
            b,
            h_kv,
            t_max,
            d,
            current_seq_len: 0,
        })
    }

    pub fn variant(&self) -> TbVariant { self.variant }
    pub fn current_seq_len(&self) -> usize { self.current_seq_len }
    pub fn reset(&mut self) { self.current_seq_len = 0; }
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.b, self.h_kv, self.t_max, self.d)
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        // WHT preconditioning (if enabled + head_dim%128==0): maps K/V
        // to a near-Gaussian distribution the Lloyd-Max centroids expect.
        // Independent sign tables for K vs V avoid coupling their error
        // distributions.
        let (k_pre, v_pre) = if wht_enabled() {
            let kk = apply_wht_128(k, &self.dev, /*kind_is_v=*/ false, /*inverse=*/ false)?;
            let vv = apply_wht_128(v, &self.dev, /*kind_is_v=*/ true,  /*inverse=*/ false)?;
            (kk, vv)
        } else {
            (k.clone(), v.clone())
        };

        let t_new = self.append_inner(&k_pre, /*is_v=*/ false)?;
        let t_new_v = self.append_inner(&v_pre, /*is_v=*/ true)?;
        if t_new != t_new_v {
            crate::bail!("QTurboKvCache::append: K/V T_new disagree ({t_new} vs {t_new_v})");
        }
        self.current_seq_len += t_new;
        Ok(())
    }

    fn append_inner(&mut self, src: &Tensor, is_v: bool) -> Result<usize> {
        let dims = src.dims();
        if dims.len() != 4 {
            crate::bail!("QTurboKvCache::append: expected (B, H, T_new, D), got {dims:?}");
        }
        let (b, h, t_new, d) = (dims[0], dims[1], dims[2], dims[3]);
        if b != self.b || h != self.h_kv || d != self.d {
            crate::bail!(
                "QTurboKvCache::append: shape {:?} incompatible with cache ({},{},_,{})",
                dims, self.b, self.h_kv, self.d
            );
        }
        if src.dtype() != DType::F32 {
            crate::bail!("QTurboKvCache::append: src dtype must be F32");
        }
        let t_offset = self.current_seq_len;
        if t_offset + t_new > self.t_max {
            crate::bail!(
                "QTurboKvCache::append: overflow (offset={} + new={} > max={})",
                t_offset, t_new, self.t_max
            );
        }
        if t_new == 0 {
            return Ok(0);
        }

        let src_contig = src.contiguous()?;
        let (st, l) = src_contig.storage_and_layout();
        let hip_st = match &*st {
            Storage::Hip(s) => s,
            _ => crate::bail!("QTurboKvCache::append: src must be on HIP"),
        };
        let src_slice = hip_st.as_hip_slice::<f32>().map_err(crate::Error::wrap)?;
        let (lo, hi) = l.contiguous_offsets().ok_or_else(|| {
            crate::Error::Msg("QTurboKvCache::append: src not contiguous".into())
        })?;
        let src_view = src_slice.slice(lo..hi);
        let dst = if is_v { &self.v } else { &self.k };

        let func = self.dev.get_or_load_func(
            self.variant.quantize_kernel(),
            &kernels::TURBO_KV,
        )?;

        let qk = self.variant.block_size_elems();
        let cfg = LaunchConfig {
            grid_dim: ((d / qk) as u32, (h * t_new) as u32, b as u32),
            block_dim: (self.variant.block_dim_x(), 1, 1),
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
        unsafe { builder.launch(cfg) }.w()?;

        drop(st);
        Ok(t_new)
    }

    pub fn dequantize_k(&self) -> Result<Tensor> {
        let raw = self.dequantize_inner(&self.k, 0, self.current_seq_len)?;
        if wht_enabled() && self.d % 128 == 0 && self.current_seq_len > 0 {
            apply_wht_128(&raw, &self.dev, /*kind_is_v=*/ false, /*inverse=*/ true)
        } else {
            Ok(raw)
        }
    }

    pub fn dequantize_v(&self) -> Result<Tensor> {
        let raw = self.dequantize_inner(&self.v, 0, self.current_seq_len)?;
        if wht_enabled() && self.d % 128 == 0 && self.current_seq_len > 0 {
            apply_wht_128(&raw, &self.dev, /*kind_is_v=*/ true, /*inverse=*/ true)
        } else {
            Ok(raw)
        }
    }

    fn dequantize_inner(
        &self,
        slab: &HipSlice<u8>,
        t_start: usize,
        t_sel: usize,
    ) -> Result<Tensor> {
        if t_start + t_sel > self.current_seq_len {
            crate::bail!(
                "QTurboKvCache::dequantize: [{t_start}, {t_start}+{t_sel}) > current_seq_len={}",
                self.current_seq_len
            );
        }
        if t_sel == 0 {
            return Tensor::zeros(
                (self.b, self.h_kv, 0, self.d),
                DType::F32,
                &crate::Device::Hip(self.dev.clone()),
            );
        }
        let elem = self.b * self.h_kv * t_sel * self.d;
        let out = unsafe { self.dev.alloc::<f32>(elem)? };

        let func = self.dev.get_or_load_func(
            self.variant.dequantize_kernel(),
            &kernels::TURBO_KV,
        )?;

        let qk = self.variant.block_size_elems();
        let cfg = LaunchConfig {
            grid_dim: ((self.d / qk) as u32, (self.h_kv * t_sel) as u32, self.b as u32),
            block_dim: (self.variant.block_dim_x(), 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i = self.b as i32;
        let h_i = self.h_kv as i32;
        let t_sel_i = t_sel as i32;
        let d_i = self.d as i32;
        let t_max_i = self.t_max as i32;
        let t_offset_i = t_start as i32;

        let mut builder = func.builder();
        builder.arg(slab);
        builder.arg(&out);
        builder.arg(&b_i);
        builder.arg(&h_i);
        builder.arg(&t_sel_i);
        builder.arg(&d_i);
        builder.arg(&t_max_i);
        builder.arg(&t_offset_i);
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
