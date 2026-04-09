#![allow(unused)]
use super::GgmlDType;
use crate::{Error, HipDevice, HipStorage, Result};

pub struct QHipStorage {
    dtype: GgmlDType,
    device: HipDevice,
}

impl QHipStorage {
    pub fn zeros(_: &HipDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<HipStorage> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<HipStorage> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn quantize(&mut self, _src: &HipStorage) -> Result<()> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &HipStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &HipStorage,
        _layout: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithHipSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &HipStorage,
        _: &crate::Layout,
        _: &HipStorage,
        _: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        Err(Error::NotCompiledWithHipSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &HipDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithHipSupport)
}
