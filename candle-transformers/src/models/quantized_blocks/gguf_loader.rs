//! Generic GGUF tensor loader.
//!
//! Extracted from quantized_qwen3.rs to be reused by all quantized model assemblers.

use super::super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, QTensor};
use candle::{Device, Result, Tensor};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Generic GGUF tensor loader. Device-agnostic — works on CPU, CUDA, and HIP.
pub struct Gguf<R: Read + Seek> {
    pub ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Switch the target device for subsequent tensor loads.
    /// Used by pipeline-parallel loaders to place each layer's weights on a
    /// specific GPU.
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }

    pub fn metadata(&self) -> &HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    /// Check if a tensor exists in the GGUF file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Load a tensor as QMatMul (quantized matrix multiply).
    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    /// Try to load a tensor as QMatMul. Returns None if the tensor doesn't exist.
    pub fn try_qmatmul(&mut self, name: &str) -> Option<QMatMul> {
        if self.has_tensor(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }

    /// Load a tensor as RmsNorm.
    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    /// Try to load a tensor as RmsNorm. Returns None if the tensor doesn't exist.
    pub fn try_rms_norm(&mut self, name: &str, eps: f64) -> Option<RmsNorm> {
        if self.has_tensor(name) {
            self.rms_norm(name, eps).ok()
        } else {
            None
        }
    }

    /// Load a raw QTensor.
    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Try to load a raw QTensor. Returns None if the tensor doesn't exist.
    pub fn try_tensor(&mut self, name: &str) -> Option<QTensor> {
        if self.has_tensor(name) {
            self.tensor(name).ok()
        } else {
            None
        }
    }

    /// Load a tensor and immediately dequantize to f32 on the target device.
    /// Use for small tensors (biases, scalar params) that don't benefit from quantization.
    pub fn dequantize(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.ct.tensor(&mut self.reader, name, &self.device)?;
        qt.dequantize(&self.device)
    }

    /// Try to load and dequantize. Returns None if the tensor doesn't exist.
    pub fn try_dequantize(&mut self, name: &str) -> Option<Tensor> {
        if self.has_tensor(name) {
            self.dequantize(name).ok()
        } else {
            None
        }
    }
}
