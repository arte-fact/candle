//! Implementation of Backend traits for HIP/ROCm device
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
pub use candle_hip_kernels as kernels;
pub use hipdarc;
use hipdarc::driver::{DeviceRepr, HipSlice, LaunchConfig, NullDevicePtr, PushKernelArg, ValidAsZeroBits};
use hipdarc::rocblas::{GemmDataType, GemmOp, StridedBatchedGemmConfig};
use half::{bf16, f16};

mod device;
mod error;
mod utils;
pub use device::{HipDevice, DeviceId};
pub use error::{HipError, WrapErr};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

/// Convenience macro to push scalar kernel arguments.
macro_rules! barg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

pub enum SlicePtrOrNull<T> {
    Ptr(HipSlice<T>),
    Null(NullDevicePtr),
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a>(&'a self, builder: &mut hipdarc::driver::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => {
                builder.arg(slice);
            }
            SlicePtrOrNull::Null(ref null) => {
                builder.arg(null);
            }
        };
    }
}

impl crate::scalar::Scalar {
    pub fn hip_builder_arg<'a>(&'a self, builder: &mut hipdarc::driver::LaunchArgs<'a>) {
        use crate::scalar::Scalar;
        match self {
            Scalar::U8(v) => { builder.arg(v); }
            Scalar::U32(v) => { builder.arg(v); }
            Scalar::I16(v) => { builder.arg(v); }
            Scalar::I32(v) => { builder.arg(v); }
            Scalar::I64(v) => { builder.arg(v); }
            Scalar::F32(v) => { builder.arg(v); }
            Scalar::F64(v) => { builder.arg(v); }
            Scalar::F16(v) => { builder.arg(v); }
            Scalar::BF16(v) => { builder.arg(v); }
            Scalar::F8E4M3(v) => { builder.arg(v); }
        };
    }
}

impl SlicePtrOrNull<usize> {
    pub fn params_from_layout(dev: &HipDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null(NullDevicePtr::new())
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&[l.dims(), l.stride()].concat())?)
        };
        Ok(ds)
    }
}

#[derive(Debug)]
pub enum HipStorageSlice {
    U8(HipSlice<u8>),
    U32(HipSlice<u32>),
    I16(HipSlice<i16>),
    I32(HipSlice<i32>),
    I64(HipSlice<i64>),
    BF16(HipSlice<bf16>),
    F16(HipSlice<f16>),
    F32(HipSlice<f32>),
    F64(HipSlice<f64>),
}

struct Clone;
impl Map1 for Clone {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        s: &HipSlice<T>,
        dev: &HipDevice,
        _: &Layout,
    ) -> Result<HipSlice<T>> {
        dev.clone_dtod(s)
    }
}

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

struct Affine(f64, f64);
impl Map1 for Affine {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("affine"), &kernels::AFFINE)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, T::from_f64(self.0));
        barg!(builder, T::from_f64(self.1));
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Elu(f64);
impl Map1 for Elu {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("uelu"), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        barg!(builder, T::from_f64(self.0));
        builder.arg(&src);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

#[allow(unused)]
struct Im2Col1D {
    l_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col1D {
    #[allow(unused)]
    fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let l_out = self.l_out(dims[2]);
        let threads = dims[0] * l_out * dims[1];
        let cfg = LaunchConfig::for_num_elems(threads as u32);
        let ds = dev.clone_htod(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("im2col1d"), &kernels::CONV)?;
        // SAFETY: Set later by running the kernel.
        let dst = unsafe { dev.alloc::<T>(threads * self.l_k)? };
        let mut builder = func.builder();
        barg!(builder, threads);
        barg!(builder, l_out);
        barg!(builder, self.l_k);
        barg!(builder, self.stride);
        barg!(builder, self.padding);
        barg!(builder, self.dilation);
        builder.arg(&ds);
        builder.arg(&src);
        builder.arg(&dst);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(dst)
    }
}

#[allow(unused)]
struct Im2Col {
    h_k: usize,
    w_k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Im2Col {
    #[allow(unused)]
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let (h_out, w_out) = self.hw_out(dims[2], dims[3]);
        let dst_el = dims[0] * h_out * w_out * dims[1] * self.h_k * self.w_k;
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let ds = dev.clone_htod(&[dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("im2col"), &kernels::CONV)?;
        // SAFETY: Set later by running the kernel.
        let dst = unsafe { dev.alloc::<T>(dst_el)? };
        let mut builder = func.builder();
        barg!(builder, dst_el);
        barg!(builder, h_out);
        barg!(builder, w_out);
        barg!(builder, self.h_k);
        barg!(builder, self.w_k);
        barg!(builder, self.stride);
        barg!(builder, self.padding);
        barg!(builder, self.dilation);
        builder.arg(&ds);
        builder.arg(&src);
        builder.arg(&dst);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(dst)
    }
}

struct Powf(f64);
impl Map1 for Powf {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("upowf"), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        barg!(builder, T::from_f64(self.0));
        builder.arg(&src);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct FastReduce<'a>(&'a [usize], ReduceOp);
impl Map1Any for FastReduce<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(HipSlice<T>) -> S>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S> {
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
        let cfg = LaunchConfig {
            grid_dim: (dst_el as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let ds = dev.clone_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..src.len());
        let (name, check_empty, return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false, false),
            ReduceOp::Min => ("fast_min", true, false),
            ReduceOp::Max => ("fast_max", true, false),
            ReduceOp::ArgMin => ("fast_argmin", true, true),
            ReduceOp::ArgMax => ("fast_argmax", true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::REDUCE)?;
        if return_index {
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<u32>(dst_el)? };
            let mut builder = func.builder();
            barg!(builder, src_el);
            barg!(builder, el_to_sum_per_block);
            barg!(builder, src_dims.len());
            builder.arg(&ds);
            builder.arg(&src);
            builder.arg(&out);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(S::U32(out))
        } else {
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<T>(dst_el)? };
            let mut builder = func.builder();
            barg!(builder, src_el);
            barg!(builder, el_to_sum_per_block);
            barg!(builder, src_dims.len());
            builder.arg(&ds);
            builder.arg(&src);
            builder.arg(&out);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(wrap(out))
        }
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el_count)? };
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(&src);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

/// A raw device pointer that can be pushed as a kernel argument.
/// This is used when we need to pass a device pointer from an offset into
/// a `HipSlice` as a kernel argument (mirroring the CUDA `slice_ptr` pattern).
#[derive(Clone, Copy)]
struct DevPtr(*mut std::ffi::c_void);

unsafe impl PushKernelArg for DevPtr {
    fn as_kernel_arg(&self) -> (*const std::ffi::c_void, usize) {
        (
            &self.0 as *const *mut std::ffi::c_void as *const std::ffi::c_void,
            std::mem::size_of::<*mut std::ffi::c_void>(),
        )
    }
}

/// Returns the raw device pointer for a slice at the given offset, wrapped
/// so it can be passed as a kernel argument.
fn slice_ptr<T>(v: &HipSlice<T>, lo: usize) -> DevPtr {
    let view = v.slice(lo..v.len());
    DevPtr(view.device_ptr())
}

struct IndexSelect<'a>(&'a HipStorage, &'a Layout, usize);
impl Map1 for IndexSelect<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        src_l: &Layout,
    ) -> Result<HipSlice<T>> {
        let ids_l = &self.1;
        let (name, ids) = match &self.0.slice {
            HipStorageSlice::U32(slice) => ("is_u32", slice_ptr(slice, ids_l.start_offset())),
            HipStorageSlice::U8(slice) => ("is_u8", slice_ptr(slice, ids_l.start_offset())),
            HipStorageSlice::I64(slice) => ("is_i64", slice_ptr(slice, ids_l.start_offset())),
            _ => Err(HipError::UnexpectedDType {
                msg: "index_select ids should be u8, u32, or i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            .w()?,
        };
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ds = dev.clone_htod(&[ids_dims, ids_l.stride()].concat())?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let src_dim_size = src_l.dims()[self.2];
        let ids_dim_size = ids_shape.elem_count();
        let dst_el = ids_shape.elem_count() * left_size * right_size;
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let mut builder = func.builder();
        barg!(builder, dst_el);
        barg!(builder, ids_dims.len());
        builder.arg(&ds);
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, left_size);
        barg!(builder, src_dim_size);
        barg!(builder, ids_dim_size);
        barg!(builder, right_size);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Gather<'a>(&'a HipStorage, &'a Layout, usize);
impl Map1 for Gather<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        src_l: &Layout,
    ) -> Result<HipSlice<T>> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            HipStorageSlice::U32(slice) => ("gather_u32", slice_ptr(slice, ids_o1)),
            HipStorageSlice::U8(slice) => ("gather_u8", slice_ptr(slice, ids_o1)),
            HipStorageSlice::I64(slice) => ("gather_i64", slice_ptr(slice, ids_o1)),
            _ => Err(HipError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let el = ids_l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, left_sz);
        barg!(builder, src_dim_sz);
        barg!(builder, ids_dim_sz);
        barg!(builder, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct IndexAdd<'a>(&'a HipStorage, &'a Layout, usize);
impl Map2InPlace for IndexAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut HipSlice<T>,
        dst_l: &Layout,
        src: &HipSlice<T>,
        src_l: &Layout,
        dev: &HipDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            HipStorageSlice::U32(slice) => ("ia_u32", slice_ptr(slice, ids_o1)),
            HipStorageSlice::I64(slice) => ("ia_i64", slice_ptr(slice, ids_o1)),
            HipStorageSlice::U8(slice) => ("ia_u8", slice_ptr(slice, ids_o1)),
            _ => Err(HipError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst_view = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let src_view = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        barg!(builder, ids_dim_sz);
        builder.arg(&src_view);
        builder.arg(&dst_view);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

struct Scatter<'a>(&'a HipStorage, &'a Layout, usize);
impl Map2InPlace for Scatter<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut HipSlice<T>,
        dst_l: &Layout,
        src: &HipSlice<T>,
        src_l: &Layout,
        dev: &HipDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            HipStorageSlice::U32(slice) => ("s_u32", slice_ptr(slice, ids_o1)),
            HipStorageSlice::I64(slice) => ("s_i64", slice_ptr(slice, ids_o1)),
            HipStorageSlice::U8(slice) => ("s_u8", slice_ptr(slice, ids_o1)),
            _ => Err(HipError::UnexpectedDType {
                msg: "scatter ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst_view = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let src_view = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        builder.arg(&src_view);
        builder.arg(&dst_view);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

struct ScatterAdd<'a>(&'a HipStorage, &'a Layout, usize);
impl Map2InPlace for ScatterAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut HipSlice<T>,
        dst_l: &Layout,
        src: &HipSlice<T>,
        src_l: &Layout,
        dev: &HipDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            HipStorageSlice::U32(slice) => ("sa_u32", slice_ptr(slice, ids_o1)),
            HipStorageSlice::I64(slice) => ("sa_i64", slice_ptr(slice, ids_o1)),
            HipStorageSlice::U8(slice) => ("sa_u8", slice_ptr(slice, ids_o1)),
            _ => Err(HipError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst_view = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let src_view = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        builder.arg(&src_view);
        builder.arg(&dst_view);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

struct Conv1D<'a>(&'a crate::conv::ParamsConv1D);
impl Map2 for Conv1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        inp_l: &Layout,
        k: &HipSlice<T>,
        k_l: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let p = &self.0;
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let k = &k.slice(k_l.start_offset()..k.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv1d"), &kernels::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.clone_htod(&ds)?;
        let mut builder = func.builder();
        barg!(builder, el, l_out, p.stride, p.padding, p.dilation);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&k);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Conv2D<'a>(&'a crate::conv::ParamsConv2D);
impl Map2 for Conv2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        inp_l: &Layout,
        k: &HipSlice<T>,
        k_l: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let k = &k.slice(k_l.start_offset()..k.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv2d"), &kernels::CONV)?;
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.clone_htod(&ds)?;
        let mut builder = func.builder();
        barg!(builder, el, out_w, out_h, p.stride, p.padding, p.dilation);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&k);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Col2Im1D {
    stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        col: &HipSlice<T>,
        dev: &HipDevice,
        l: &Layout,
    ) -> Result<HipSlice<T>> {
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let dst_el = b_size * c_out * l_out;
        let im = unsafe { dev.alloc::<T>(dst_el)? };

        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("col2im1d"), &kernels::CONV)?;
        let mut builder = func.builder();
        barg!(builder, dst_el, l_out, l_in, c_out, k_size, stride);
        builder.arg(col);
        builder.arg(&im);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(im)
    }
}

struct ConvTranspose1D<'a>(&'a crate::conv::ParamsConvTranspose1D);
impl Map2 for ConvTranspose1D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        inp_l: &Layout,
        k: &HipSlice<T>,
        k_l: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let p = &self.0;
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let k = &k.slice(k_l.start_offset()..k.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv_transpose1d"), &kernels::CONV)?;
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose1d {dims:?}")
        };
        let ds = dev.clone_htod(&ds)?;
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, l_out);
        barg!(builder, p.stride);
        barg!(builder, p.padding);
        barg!(builder, p.output_padding);
        barg!(builder, p.dilation);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&k);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct ConvTranspose2D<'a>(&'a crate::conv::ParamsConvTranspose2D);
impl Map2 for ConvTranspose2D<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        inp_l: &Layout,
        k: &HipSlice<T>,
        k_l: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let k = &k.slice(k_l.start_offset()..k.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv_transpose2d"), &kernels::CONV)?;
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        };
        let ds = dev.clone_htod(&ds)?;
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, out_w);
        barg!(builder, out_h);
        barg!(builder, p.stride);
        barg!(builder, p.padding);
        barg!(builder, p.output_padding);
        barg!(builder, p.dilation);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&k);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

enum PoolOp {
    Max,
    Avg,
}

struct Pool2D {
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    op: PoolOp,
}

impl Map1 for Pool2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        dev: &HipDevice,
        inp_l: &Layout,
    ) -> Result<HipSlice<T>> {
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for pool {dims:?}")
        };
        let el = shape.elem_count();
        let out_w = (dims[2] - self.w_k) / self.w_stride + 1;
        let out_h = (dims[3] - self.h_k) / self.h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let kname = match self.op {
            PoolOp::Max => "max_pool2d",
            PoolOp::Avg => "avg_pool2d",
        };
        let func = dev.get_or_load_func(&kernel_name::<T>(kname), &kernels::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.clone_htod(&ds)?;
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, self.w_k);
        barg!(builder, self.h_k);
        barg!(builder, self.w_stride);
        barg!(builder, self.h_stride);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct UpsampleNearest2D(usize, usize);
impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        dev: &HipDevice,
        inp_l: &Layout,
    ) -> Result<HipSlice<T>> {
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        };
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("upsample_nearest2d"), &kernels::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.clone_htod(&ds)?;
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;
        let mut builder = func.builder();
        barg!(builder, out_w);
        barg!(builder, out_h);
        barg!(builder, scale_w);
        barg!(builder, scale_h);
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct UpsampleBilinear2D {
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h_factor: Option<f64>,
    scale_w_factor: Option<f64>,
}

impl Map1 for UpsampleBilinear2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &HipSlice<T>,
        dev: &HipDevice,
        inp_l: &Layout,
    ) -> Result<HipSlice<T>> {
        let inp = &inp.slice(inp_l.start_offset()..inp.len());
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for upsample_bilinear2d {dims:?}")
        };

        let (out_w, out_h) = (self.out_w, self.out_h);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func =
            dev.get_or_load_func(&kernel_name::<T>("upsample_bilinear2d"), &kernels::CONV)?;

        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.clone_htod(&ds)?;

        let mut builder = func.builder();
        barg!(builder, out_w);
        barg!(builder, out_h);
        barg!(builder, self.align_corners);
        barg!(builder, self.scale_h_factor.is_some());
        barg!(builder, self.scale_h_factor.unwrap_or(0.0));
        barg!(builder, self.scale_w_factor.is_some());
        barg!(builder, self.scale_w_factor.unwrap_or(0.0));
        builder.arg(&ds);
        builder.arg(&inp);
        builder.arg(&out);

        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct WhereCond<'a>(&'a HipStorage, &'a Layout);
impl Map2 for WhereCond<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        t: &HipSlice<T>,
        layout_t: &Layout,
        f: &HipSlice<T>,
        layout_f: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let ids_l = &self.1;
        let (ids, name) = match &self.0.slice {
            HipStorageSlice::U8(slice) => {
                (slice_ptr(slice, ids_l.start_offset()), "where_u8")
            }
            HipStorageSlice::U32(slice) => {
                (slice_ptr(slice, ids_l.start_offset()), "where_u32")
            }
            HipStorageSlice::I64(slice) => {
                (slice_ptr(slice, ids_l.start_offset()), "where_i64")
            }
            _ => Err(HipError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            .w()?,
        };
        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds =
            dev.clone_htod(&[dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat())?;
        let t = &t.slice(layout_t.start_offset()..t.len());
        let f = &f.slice(layout_f.start_offset()..f.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::TERNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        builder.arg(&ds);
        barg!(builder, ids);
        builder.arg(&t);
        builder.arg(&f);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &HipSlice<T>,
        lhs_l: &Layout,
        rhs: &HipSlice<T>,
        rhs_l: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            SlicePtrOrNull::Null(NullDevicePtr::new())
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
        let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::BINARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(elem_count)? };
        let mut builder = func.builder();
        barg!(builder, elem_count);
        barg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(&lhs);
        builder.arg(&rhs);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Cmp(CmpOp);
impl Map2Any for Cmp {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &HipSlice<T>,
        lhs_l: &Layout,
        rhs: &HipSlice<T>,
        rhs_l: &Layout,
        dev: &HipDevice,
    ) -> Result<S> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            SlicePtrOrNull::Null(NullDevicePtr::new())
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
        let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
        let name = match self.0 {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::BINARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<u8>(elem_count)? };
        let mut builder = func.builder();
        barg!(builder, elem_count);
        barg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(&lhs);
        builder.arg(&rhs);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(S::U8(out))
    }
}

#[derive(Debug)]
pub struct HipStorage {
    pub slice: HipStorageSlice,
    pub device: HipDevice,
}

pub trait HipDType: Sized {
    fn as_hip_slice(s: &HipStorage) -> Result<&HipSlice<Self>>;
    fn as_hip_slice_mut(s: &mut HipStorage) -> Result<&mut HipSlice<Self>>;
    fn wrap_hip_slice(s: HipSlice<Self>, dev: HipDevice) -> HipStorage;
}

macro_rules! hip_dtype {
    ($ty:ty, $dtype:ident) => {
        impl HipDType for $ty {
            fn as_hip_slice(s: &HipStorage) -> Result<&HipSlice<Self>> {
                match &s.slice {
                    HipStorageSlice::$dtype(data) => Ok(&data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn as_hip_slice_mut(s: &mut HipStorage) -> Result<&mut HipSlice<Self>> {
                match s.slice {
                    HipStorageSlice::$dtype(ref mut data) => Ok(data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn wrap_hip_slice(slice: HipSlice<Self>, device: HipDevice) -> HipStorage {
                let slice = HipStorageSlice::$dtype(slice);
                HipStorage { slice, device }
            }
        }
    };
}
hip_dtype!(u8, U8);
hip_dtype!(u32, U32);
hip_dtype!(i16, I16);
hip_dtype!(i32, I32);
hip_dtype!(i64, I64);
hip_dtype!(f16, F16);
hip_dtype!(bf16, BF16);
hip_dtype!(f32, F32);
hip_dtype!(f64, F64);

impl HipStorage {
    pub fn wrap_hip_slice<T: HipDType>(slice: HipSlice<T>, device: HipDevice) -> HipStorage {
        T::wrap_hip_slice(slice, device)
    }

    pub fn as_hip_slice<T: HipDType>(&self) -> Result<&HipSlice<T>> {
        T::as_hip_slice(self)
    }

    pub fn as_hip_slice_mut<T: HipDType>(&mut self) -> Result<&mut HipSlice<T>> {
        T::as_hip_slice_mut(self)
    }
}

fn gemm_config(
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<(StridedBatchedGemmConfig, GemmOp, GemmOp)> {
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    // The a tensor has dims batching, k, n (rhs)
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, GemmOp::NoTrans)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, GemmOp::Trans)
    } else {
        Err(HipError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The b tensor has dims batching, m, k (lhs)
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, GemmOp::NoTrans)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, GemmOp::Trans)
    } else {
        Err(HipError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => Err(HipError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => Err(HipError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };

    let cfg = StridedBatchedGemmConfig {
        trans_a: transa,
        trans_b: transb,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        stride_a: stride_a as i64,
        ldb,
        stride_b: stride_b as i64,
        ldc: n as i32,
        stride_c: (m * n) as i64,
        batch_count: b as i32,
        // These will be overridden per-dtype.
        ab_type: GemmDataType::F32,
        c_type: GemmDataType::F32,
        compute_type: GemmDataType::F32,
    };
    Ok((cfg, transa, transb))
}

// Default for the reduced precision setting is false, similar to pytorch.
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

impl BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let slice = Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> DType {
        match self.slice {
            HipStorageSlice::U8(_) => DType::U8,
            HipStorageSlice::U32(_) => DType::U32,
            HipStorageSlice::I16(_) => DType::I16,
            HipStorageSlice::I32(_) => DType::I32,
            HipStorageSlice::I64(_) => DType::I64,
            HipStorageSlice::BF16(_) => DType::BF16,
            HipStorageSlice::F16(_) => DType::F16,
            HipStorageSlice::F32(_) => DType::F32,
            HipStorageSlice::F64(_) => DType::F64,
        }
    }

    fn device(&self) -> &HipDevice {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let dev = &self.device;
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src_o = layout.start_offset();
        let (src, kernel_name) = match &mut self.slice {
            S::U8(s) => (slice_ptr(s, src_o), "const_set_u8"),
            S::U32(s) => (slice_ptr(s, src_o), "const_set_u32"),
            S::I16(s) => (slice_ptr(s, src_o), "const_set_i16"),
            S::I32(s) => (slice_ptr(s, src_o), "const_set_i32"),
            S::I64(s) => (slice_ptr(s, src_o), "const_set_i64"),
            S::BF16(s) => (slice_ptr(s, src_o), "const_set_bf16"),
            S::F16(s) => (slice_ptr(s, src_o), "const_set_f16"),
            S::F32(s) => (slice_ptr(s, src_o), "const_set_f32"),
            S::F64(s) => (slice_ptr(s, src_o), "const_set_f64"),
        };

        let func = dev.get_or_load_func(kernel_name, &kernels::FILL)?;
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        s.hip_builder_arg(&mut builder);
        barg!(builder, src);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let start_o = layout.start_offset();
        let inp = match &self.slice {
            HipStorageSlice::U8(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::U32(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::I16(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::I32(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::I64(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::BF16(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::F16(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::F32(inp) => slice_ptr(inp, start_o),
            HipStorageSlice::F64(inp) => slice_ptr(inp, start_o),
        };
        let inp = &inp;

        let kname = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
        let func = dev.get_or_load_func(&kname, &kernels::CAST)?;
        let slice = match dtype {
            DType::U8 => {
                let out = unsafe { dev.alloc::<u8>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::U8(out)
            }
            DType::U32 => {
                let out = unsafe { dev.alloc::<u32>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::U32(out)
            }
            DType::I64 => {
                let out = unsafe { dev.alloc::<i64>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::I64(out)
            }
            DType::BF16 => {
                let out = unsafe { dev.alloc::<bf16>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::BF16(out)
            }
            DType::F16 => {
                let out = unsafe { dev.alloc::<f16>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::F16(out)
            }
            DType::F32 => {
                let out = unsafe { dev.alloc::<f32>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::F32(out)
            }
            DType::F64 => {
                let out = unsafe { dev.alloc::<f64>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                HipStorageSlice::F64(out)
            }
            DType::I16 | DType::I32 => {
                return Err(HipError::InternalError("i16,i32 dtypes are not supported").into())
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    HipError::InternalError("F8/F6/F4 types not supported in HIP backend").into(),
                )
            }
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = Cmp(op).map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = U::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            HipStorageSlice::U8(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::U8(cpu_storage))
            }
            HipStorageSlice::U32(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::U32(cpu_storage))
            }
            HipStorageSlice::I16(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::I16(cpu_storage))
            }
            HipStorageSlice::I32(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::I32(cpu_storage))
            }
            HipStorageSlice::I64(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::I64(cpu_storage))
            }
            HipStorageSlice::BF16(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::BF16(cpu_storage))
            }
            HipStorageSlice::F16(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::F16(cpu_storage))
            }
            HipStorageSlice::F32(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            HipStorageSlice::F64(slice) => {
                let cpu_storage = self.device.clone_dtoh(slice)?;
                Ok(CpuStorage::F64(cpu_storage))
            }
        }
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = WhereCond(self, layout).map(&t.slice, t_l, &f.slice, f_l, &device)?;
        Ok(Self { slice, device })
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV1D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV1D {
            let slice = Conv1D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
        }

        let col = Im2Col1D {
            l_k: params.k_size,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(&self.slice, &device, l)?;
        let col = Self { slice: col, device };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        const USE_COL2IM_CONV1D_TR: bool = true;

        let device = self.device().clone();
        let can_use_col2im = kernel_l.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        let slice = if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = l.shape().dims3()?;
            let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
            if !kernel_l.is_contiguous() {
                crate::bail!(
                    "convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}"
                )
            }
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    l.shape(),
                    kernel_l.shape()
                )
            }
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1],
                    kernel_l.start_offset(),
                );
                self.matmul(
                    kernel,
                    (
                        b_size,
                        /* m */ l_in,
                        /* n */ c_out * k_size,
                        /* k */ c_in,
                    ),
                    &l.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
            Col2Im1D {
                stride: params.stride,
            }
            .map(&col.slice, &device, &col_l)?
        } else {
            ConvTranspose1D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?
        };
        Ok(Self { slice, device })
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV2D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV2D {
            let slice = Conv2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
        }

        let col = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(&self.slice, &device, l)?;
        let col = Self { slice: col, device };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice =
            ConvTranspose2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
        Ok(Self { slice, device })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Avg,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Max,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on hip")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = UpsampleNearest2D(out_w, out_h).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = UpsampleBilinear2D {
            out_w,
            out_h,
            align_corners,
            scale_h_factor: scale_h,
            scale_w_factor: scale_w,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        Scatter(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        ScatterAdd(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, l, &src.slice, src_l, &device)?;
        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let slice = match (&self.slice, &rhs.slice) {
            (HipStorageSlice::BF16(lhs), HipStorageSlice::BF16(rhs)) => {
                let lhs_view = lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs_view = rhs.slice(rhs_l.start_offset()..rhs.len());
                let (mut cfg, _, _) = gemm_config((b, m, n, k), lhs_l, rhs_l)?;
                cfg.ab_type = GemmDataType::BF16;
                cfg.c_type = GemmDataType::BF16;
                // gfx906: always use F32 compute for bf16
                cfg.compute_type = GemmDataType::F32;
                let out = unsafe { dev.alloc::<bf16>(elem_count)? };
                let alpha_f32: f32 = 1.0;
                let beta_f32: f32 = 0.0;
                unsafe {
                    hipdarc::rocblas::gemm_strided_batched_ex(
                        &dev.blas,
                        &cfg,
                        &alpha_f32 as *const f32 as *const _,
                        rhs_view.device_ptr() as *const _,
                        lhs_view.device_ptr() as *const _,
                        &beta_f32 as *const f32 as *const _,
                        out.device_ptr() as *mut _,
                    )
                }
                .w()?;
                HipStorageSlice::BF16(out)
            }
            (HipStorageSlice::F16(lhs), HipStorageSlice::F16(rhs)) => {
                let lhs_view = lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs_view = rhs.slice(rhs_l.start_offset()..rhs.len());
                let (mut cfg, _, _) = gemm_config((b, m, n, k), lhs_l, rhs_l)?;
                cfg.ab_type = GemmDataType::F16;
                cfg.c_type = GemmDataType::F16;
                let (compute_type, alpha_ptr, beta_ptr): (GemmDataType, *const std::ffi::c_void, *const std::ffi::c_void);
                let alpha_f16 = f16::ONE;
                let beta_f16 = f16::ZERO;
                let alpha_f32: f32 = 1.0;
                let beta_f32: f32 = 0.0;
                if gemm_reduced_precision_f16() {
                    compute_type = GemmDataType::F16;
                    alpha_ptr = &alpha_f16 as *const f16 as *const _;
                    beta_ptr = &beta_f16 as *const f16 as *const _;
                } else {
                    compute_type = GemmDataType::F32;
                    alpha_ptr = &alpha_f32 as *const f32 as *const _;
                    beta_ptr = &beta_f32 as *const f32 as *const _;
                }
                cfg.compute_type = compute_type;
                let out = unsafe { dev.alloc::<f16>(elem_count)? };
                unsafe {
                    hipdarc::rocblas::gemm_strided_batched_ex(
                        &dev.blas,
                        &cfg,
                        alpha_ptr,
                        rhs_view.device_ptr() as *const _,
                        lhs_view.device_ptr() as *const _,
                        beta_ptr,
                        out.device_ptr() as *mut _,
                    )
                }
                .w()?;
                HipStorageSlice::F16(out)
            }
            (HipStorageSlice::F32(lhs), HipStorageSlice::F32(rhs)) => {
                let lhs_view = lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs_view = rhs.slice(rhs_l.start_offset()..rhs.len());
                let (mut cfg, _, _) = gemm_config((b, m, n, k), lhs_l, rhs_l)?;
                cfg.ab_type = GemmDataType::F32;
                cfg.c_type = GemmDataType::F32;
                cfg.compute_type = GemmDataType::F32;
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let out = unsafe { dev.alloc::<f32>(elem_count)? };
                unsafe {
                    hipdarc::rocblas::gemm_strided_batched_ex(
                        &dev.blas,
                        &cfg,
                        &alpha as *const f32 as *const _,
                        rhs_view.device_ptr() as *const _,
                        lhs_view.device_ptr() as *const _,
                        &beta as *const f32 as *const _,
                        out.device_ptr() as *mut _,
                    )
                }
                .w()?;
                HipStorageSlice::F32(out)
            }
            (HipStorageSlice::F64(lhs), HipStorageSlice::F64(rhs)) => {
                let lhs_view = lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs_view = rhs.slice(rhs_l.start_offset()..rhs.len());
                let (mut cfg, _, _) = gemm_config((b, m, n, k), lhs_l, rhs_l)?;
                cfg.ab_type = GemmDataType::F64;
                cfg.c_type = GemmDataType::F64;
                cfg.compute_type = GemmDataType::F64;
                let alpha: f64 = 1.0;
                let beta: f64 = 0.0;
                let out = unsafe { dev.alloc::<f64>(elem_count)? };
                unsafe {
                    hipdarc::rocblas::gemm_strided_batched_ex(
                        &dev.blas,
                        &cfg,
                        &alpha as *const f64 as *const _,
                        rhs_view.device_ptr() as *const _,
                        lhs_view.device_ptr() as *const _,
                        &beta as *const f64 as *const _,
                        out.device_ptr() as *mut _,
                    )
                }
                .w()?;
                HipStorageSlice::F64(out)
            }
            _ => Err(HipError::InternalError("dtype mismatch in matmul op"))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let dev = &self.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let (src, dst_ptr, kname) = match (&self.slice, &mut dst.slice) {
            (S::U8(s), S::U8(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u8"),
            (S::U32(s), S::U32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u32"),
            (S::I16(s), S::I16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i16"),
            (S::I32(s), S::I32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i32"),
            (S::I64(s), S::I64(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i64"),
            (S::BF16(s), S::BF16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_bf16"),
            (S::F16(s), S::F16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f16"),
            (S::F32(s), S::F32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f32"),
            (S::F64(s), S::F64(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f64"),
            _ => Err(HipError::InternalError("dtype mismatch in copy2d"))?,
        };
        let func = dev.get_or_load_func(kname, &kernels::FILL)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let mut builder = func.builder();
        barg!(builder, src);
        barg!(builder, dst_ptr);
        barg!(builder, d1);
        barg!(builder, d2);
        builder.arg(&src_s);
        builder.arg(&dst_s);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = SlicePtrOrNull::params_from_layout(dev, src_l)?;

        // Helper: launch a ucopy kernel to copy src (potentially strided) into
        // a contiguous region of dst starting at dst_offset.  When the layout
        // is contiguous the ds parameter is Null and the kernel degenerates to
        // a plain element-wise copy.
        macro_rules! copy_strided_impl {
            ($src_slice:expr, $dst_slice:expr, $kname:expr) => {{
                let src = &$src_slice.slice(src_l.start_offset()..$src_slice.len());
                let dst_view = $dst_slice.slice(dst_offset..dst_offset + el_count);
                let func = dev.get_or_load_func($kname, &kernels::UNARY)?;
                let mut builder = func.builder();
                barg!(builder, el_count);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                builder.arg(&src);
                builder.arg(&dst_view);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
            }};
        }

        match (&self.slice, &mut dst.slice) {
            (HipStorageSlice::U8(src), HipStorageSlice::U8(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_u8");
            }
            (HipStorageSlice::U32(src), HipStorageSlice::U32(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_u32");
            }
            (HipStorageSlice::I16(src), HipStorageSlice::I16(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_i16");
            }
            (HipStorageSlice::I32(src), HipStorageSlice::I32(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_i32");
            }
            (HipStorageSlice::I64(src), HipStorageSlice::I64(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_i64");
            }
            (HipStorageSlice::BF16(src), HipStorageSlice::BF16(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_bf16");
            }
            (HipStorageSlice::F16(src), HipStorageSlice::F16(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_f16");
            }
            (HipStorageSlice::F32(src), HipStorageSlice::F32(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_f32");
            }
            (HipStorageSlice::F64(src), HipStorageSlice::F64(dst)) => {
                copy_strided_impl!(src, dst, "ucopy_f64");
            }
            _ => Err(HipError::InternalError(
                "dtype mismatch in copy_strided op",
            ))?,
        }
        Ok(())
    }
}
