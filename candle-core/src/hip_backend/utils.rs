/// Helper traits to dispatch HIP kernels across dtypes.
use crate::{Layout, Result, WithDType};
use hipdarc::driver::{DeviceRepr, HipSlice, ValidAsZeroBits};

use super::{HipDevice, HipError, WrapErr};

/// Alias for the storage slice enum.
pub type S = super::HipStorageSlice;

/// Dispatch a unary kernel across all supported dtypes.
pub trait Map1 {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
    ) -> Result<HipSlice<T>>;

    fn map(&self, s: &S, d: &HipDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I16(s) => S::I16(self.f(s, d, l)?),
            S::I32(s) => S::I32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
        };
        Ok(out)
    }
}

/// Dispatch a binary kernel across matching dtype pairs.
pub trait Map2 {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &HipSlice<T>,
        layout1: &Layout,
        src2: &HipSlice<T>,
        layout2: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &HipDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => S::U8(self.f(s1, l1, s2, l2, d)?),
            (S::U32(s1), S::U32(s2)) => S::U32(self.f(s1, l1, s2, l2, d)?),
            (S::I16(s1), S::I16(s2)) => S::I16(self.f(s1, l1, s2, l2, d)?),
            (S::I32(s1), S::I32(s2)) => S::I32(self.f(s1, l1, s2, l2, d)?),
            (S::I64(s1), S::I64(s2)) => S::I64(self.f(s1, l1, s2, l2, d)?),
            (S::BF16(s1), S::BF16(s2)) => S::BF16(self.f(s1, l1, s2, l2, d)?),
            (S::F16(s1), S::F16(s2)) => S::F16(self.f(s1, l1, s2, l2, d)?),
            (S::F32(s1), S::F32(s2)) => S::F32(self.f(s1, l1, s2, l2, d)?),
            (S::F64(s1), S::F64(s2)) => S::F64(self.f(s1, l1, s2, l2, d)?),
            _ => Err(HipError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}

/// Dispatch a ternary kernel across matching dtype triples.
pub trait Map3 {
    #[allow(clippy::too_many_arguments)]
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &HipSlice<T>,
        layout1: &Layout,
        src2: &HipSlice<T>,
        layout2: &Layout,
        src3: &HipSlice<T>,
        layout3: &Layout,
        dev: &HipDevice,
    ) -> Result<HipSlice<T>>;

    #[allow(clippy::too_many_arguments)]
    fn map(
        &self,
        s1: &S,
        l1: &Layout,
        s2: &S,
        l2: &Layout,
        s3: &S,
        l3: &Layout,
        d: &HipDevice,
    ) -> Result<S> {
        let out = match (s1, s2, s3) {
            (S::U8(a), S::U8(b), S::U8(c)) => S::U8(self.f(a, l1, b, l2, c, l3, d)?),
            (S::U32(a), S::U32(b), S::U32(c)) => S::U32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I64(a), S::I64(b), S::I64(c)) => S::I64(self.f(a, l1, b, l2, c, l3, d)?),
            (S::BF16(a), S::BF16(b), S::BF16(c)) => S::BF16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F16(a), S::F16(b), S::F16(c)) => S::F16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F32(a), S::F32(b), S::F32(c)) => S::F32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F64(a), S::F64(b), S::F64(c)) => S::F64(self.f(a, l1, b, l2, c, l3, d)?),
            _ => Err(HipError::InternalError("dtype mismatch in ternary op"))?,
        };
        Ok(out)
    }
}

/// Dispatch an in-place binary kernel across matching dtype pairs.
pub trait Map2InPlace {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut HipSlice<T>,
        dst_l: &Layout,
        src: &HipSlice<T>,
        src_l: &Layout,
        dev: &HipDevice,
    ) -> Result<()>;

    fn map(
        &self,
        dst: &mut S,
        dst_l: &Layout,
        src: &S,
        src_l: &Layout,
        d: &HipDevice,
    ) -> Result<()> {
        match (dst, src) {
            (S::U8(dst), S::U8(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::U32(dst), S::U32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I16(dst), S::I16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I32(dst), S::I32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_l, src, src_l, d),
            _ => Err(HipError::InternalError("dtype mismatch in binary op"))?,
        }
    }
}

/// Dispatch a unary kernel that may produce a different output dtype.
pub trait Map1Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(HipSlice<T>) -> S>(
        &self,
        src: &HipSlice<T>,
        dev: &HipDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &HipDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I16(s) => self.f(s, d, l, S::I16)?,
            S::I32(s) => self.f(s, d, l, S::I32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
        };
        Ok(out)
    }
}

/// Dispatch a binary kernel that may produce a different output dtype.
pub trait Map2Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &HipSlice<T>,
        layout1: &Layout,
        src2: &HipSlice<T>,
        layout2: &Layout,
        dev: &HipDevice,
    ) -> Result<S>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &HipDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(a), S::U8(b)) => self.f(a, l1, b, l2, d)?,
            (S::U32(a), S::U32(b)) => self.f(a, l1, b, l2, d)?,
            (S::I64(a), S::I64(b)) => self.f(a, l1, b, l2, d)?,
            (S::BF16(a), S::BF16(b)) => self.f(a, l1, b, l2, d)?,
            (S::F16(a), S::F16(b)) => self.f(a, l1, b, l2, d)?,
            (S::F32(a), S::F32(b)) => self.f(a, l1, b, l2, d)?,
            (S::F64(a), S::F64(b)) => self.f(a, l1, b, l2, d)?,
            _ => Err(HipError::InternalError("dtype mismatch in binary op")).w()?,
        };
        Ok(out)
    }
}
