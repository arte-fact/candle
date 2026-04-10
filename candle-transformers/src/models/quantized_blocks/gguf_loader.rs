//! Generic GGUF tensor loader.
//!
//! Memory-mapped, `&self`, and `Clone`-cheap so model assemblers can fan
//! per-layer tensor loads out across rayon workers without any reader
//! contention.
//!
//! ## Why mmap + `&self`?
//!
//! The previous reader-based loader (`Gguf<R: Read + Seek>`) held an
//! exclusive `&mut R` for the duration of every tensor read. That made
//! parallel layer loading impossible — every worker would have had to
//! serialize on the reader. Loading a 17 GB GGUF on 4 GPUs took ~17 s.
//!
//! With mmap + `&self`, the [`GgufBlob`] is shared via `Arc` and every
//! tensor lookup is just a slice into that shared `&[u8]` view. The same
//! [`Gguf`] can be cloned per worker thread (cheap — only Arc clones), and
//! [`Self::with_device`] retargets a clone to a different GPU without
//! touching the underlying blob or [`Content`].

use super::super::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{
    gguf_file::{self, GgufBlob},
    QTensor,
};
use candle::{Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Generic GGUF tensor loader. Device-agnostic — works on CPU, CUDA, and HIP.
///
/// All accessors take `&self`, so `Gguf` can be wrapped in `Arc` and shared
/// across rayon workers. [`Self::with_device`] returns a cheap clone with a
/// different target device for pipeline-parallel loading.
#[derive(Clone)]
pub struct Gguf {
    /// Shared GGUF metadata + tensor info table.
    pub ct: Arc<gguf_file::Content>,
    /// Memory-mapped tensor data, shared across all clones.
    blob: Arc<GgufBlob>,
    /// Target device for tensor loads on this clone.
    device: Device,
}

impl Gguf {
    /// Construct a loader from a parsed [`Content`] + a memory-mapped blob.
    ///
    /// `ct` and `blob` are wrapped in `Arc` so cloning the loader is cheap
    /// (no metadata or byte-buffer copies).
    pub fn new(ct: gguf_file::Content, blob: Arc<GgufBlob>, device: Device) -> Self {
        Self {
            ct: Arc::new(ct),
            blob,
            device,
        }
    }

    /// Returns a cheap clone of this loader retargeted to `device`. Used by
    /// pipeline-parallel loaders to assign each layer's weights to a specific
    /// GPU. The blob and tensor info table are shared via `Arc`.
    pub fn with_device(&self, device: Device) -> Self {
        Self {
            ct: self.ct.clone(),
            blob: self.blob.clone(),
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn metadata(&self) -> &HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    /// Check if a tensor exists in the GGUF file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Load a tensor as QMatMul (quantized matrix multiply).
    pub fn qmatmul(&self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor_from_blob(&self.blob, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    /// Try to load a tensor as QMatMul. Returns None if the tensor doesn't exist.
    pub fn try_qmatmul(&self, name: &str) -> Option<QMatMul> {
        if self.has_tensor(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }

    /// Load several quantized weight tensors and concatenate them along
    /// dim 0 (the output / row dim) into a single fused QMatMul.
    ///
    /// All input tensors must share the same `ggml_dtype` and the same
    /// in_features (last dim). The fused output has shape
    /// `(sum_of_out_features, in_features)`.
    ///
    /// This is the on-load fusion used to collapse the QKV projections
    /// (and the gate+up FFN projections) into a single matmul launch on
    /// the forward path. For Q4_0, Q4_1, Q5_K, Q6_K and friends, the
    /// per-row block layout means concatenating along the row dim is a
    /// pure byte concatenation of the underlying storage — no scale
    /// re-computation, no requantization.
    ///
    /// Loads the bytes of each constituent tensor from the mmap blob,
    /// concatenates them in host memory, and uploads the combined
    /// buffer to the target device as one allocation. Cheaper than
    /// loading three tensors separately and then trying to merge them
    /// device-side.
    pub fn qmatmul_concat_rows(&self, names: &[&str]) -> Result<QMatMul> {
        if names.is_empty() {
            candle::bail!("qmatmul_concat_rows: empty name list");
        }
        // Look up tensor info for each constituent tensor and validate
        // they're all compatible for byte concatenation along dim 0.
        let infos: Vec<&gguf_file::TensorInfo> = names
            .iter()
            .map(|n| {
                self.ct
                    .tensor_infos
                    .get(*n)
                    .ok_or_else(|| candle::Error::Msg(format!("missing tensor: {n}")))
            })
            .collect::<Result<_>>()?;

        let dtype = infos[0].ggml_dtype;
        let dims0 = infos[0].shape.dims();
        if dims0.len() != 2 {
            candle::bail!(
                "qmatmul_concat_rows: expected rank-2 tensors, got {dims0:?} for {}",
                names[0]
            );
        }
        let in_features = dims0[1];
        for (info, name) in infos.iter().zip(names.iter()).skip(1) {
            if info.ggml_dtype != dtype {
                candle::bail!(
                    "qmatmul_concat_rows: dtype mismatch ({:?} vs {:?}) for {name}",
                    dtype,
                    info.ggml_dtype
                );
            }
            let d = info.shape.dims();
            if d.len() != 2 || d[1] != in_features {
                candle::bail!(
                    "qmatmul_concat_rows: shape mismatch (in_features={in_features}) \
                     vs {d:?} for {name}"
                );
            }
        }
        let total_out: usize = infos.iter().map(|i| i.shape.dims()[0]).sum();

        // Read each constituent's raw bytes from the blob and append.
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let mut combined: Vec<u8> = Vec::with_capacity(
            (total_out * in_features / block_size) * type_size,
        );
        for info in &infos {
            let elems = info.shape.elem_count();
            if elems % block_size != 0 {
                candle::bail!(
                    "qmatmul_concat_rows: elems {elems} not divisible by block_size {block_size}"
                );
            }
            let bytes = elems / block_size * type_size;
            let abs_offset = self.ct.tensor_data_offset + info.offset;
            let raw = self.blob.read_to_vec(abs_offset, bytes)?;
            combined.extend_from_slice(&raw);
        }

        // Build a single QTensor from the concatenated bytes.
        let combined_dims = vec![total_out, in_features];
        let qt = candle::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &combined,
            combined_dims,
            &self.device,
        )?;
        QMatMul::from_weights(qt.into())
    }

    /// Load a tensor as RmsNorm.
    pub fn rms_norm(&self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor_from_blob(&self.blob, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    /// Try to load a tensor as RmsNorm. Returns None if the tensor doesn't exist.
    pub fn try_rms_norm(&self, name: &str, eps: f64) -> Option<RmsNorm> {
        if self.has_tensor(name) {
            self.rms_norm(name, eps).ok()
        } else {
            None
        }
    }

    /// Load a raw QTensor.
    pub fn tensor(&self, name: &str) -> Result<QTensor> {
        self.ct.tensor_from_blob(&self.blob, name, &self.device)
    }

    /// Try to load a raw QTensor. Returns None if the tensor doesn't exist.
    pub fn try_tensor(&self, name: &str) -> Option<QTensor> {
        if self.has_tensor(name) {
            self.tensor(name).ok()
        } else {
            None
        }
    }

    /// Load a tensor and immediately dequantize to f32 on the target device.
    /// Use for small tensors (biases, scalar params) that don't benefit from quantization.
    pub fn dequantize(&self, name: &str) -> Result<Tensor> {
        let qt = self.ct.tensor_from_blob(&self.blob, name, &self.device)?;
        qt.dequantize(&self.device)
    }

    /// Try to load and dequantize. Returns None if the tensor doesn't exist.
    pub fn try_dequantize(&self, name: &str) -> Option<Tensor> {
        if self.has_tensor(name) {
            self.dequantize(name).ok()
        } else {
            None
        }
    }
}
