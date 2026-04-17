//! X3.4 correctness check: `QTensor::narrow_experts` round-trip.
//!
//! Build a known quantized [E, H, I] tensor, slice experts [start, start+count),
//! dequantize both, verify the sliced tensor matches rows
//! [start, start+count) of the full dequantized tensor bit-for-bit.

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

const E: usize = 16;
const H: usize = 256;
const I: usize = 512;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;

    // Build a reproducible [E, H, I] F32 tensor and quantize to Q8_0.
    let n = E * H * I;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as i64 * 37 + 11) % 251) as f32 / 125.0 - 1.0)
        .collect();
    let full_f32 = Tensor::new(data.as_slice(), &dev)?.reshape((E, H, I))?;

    let full_q = QTensor::quantize(&full_f32, GgmlDType::Q8_0)?;
    println!(
        "full QTensor: shape={:?}  dtype={:?}  storage_bytes={}",
        full_q.shape().dims(),
        full_q.dtype(),
        full_q.storage_size_in_bytes()
    );

    // Slice a contiguous range of experts.
    let start = 4;
    let count = 7;
    let sliced = full_q.narrow_experts(start, count)?;
    println!(
        "sliced QTensor: shape={:?}  storage_bytes={}",
        sliced.shape().dims(),
        sliced.storage_size_in_bytes()
    );
    let expected_bytes =
        count * H * I / GgmlDType::Q8_0.block_size() * GgmlDType::Q8_0.type_size();
    assert_eq!(
        sliced.storage_size_in_bytes(),
        expected_bytes,
        "sliced storage bytes"
    );

    // Dequantize both and compare the overlapping region.
    let full_dq = full_q.dequantize(&dev)?;
    let sliced_dq = sliced.dequantize(&dev)?;

    let full_slice = full_dq.narrow(0, start, count)?;
    let full_vec: Vec<f32> = full_slice
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let sliced_vec: Vec<f32> = sliced_dq
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    assert_eq!(
        full_vec.len(),
        sliced_vec.len(),
        "dequantized element count mismatch"
    );
    let max_abs = full_vec
        .iter()
        .zip(sliced_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max_abs_err (sliced vs full-narrow) = {:.3e}", max_abs);
    // Dequantized values come from IDENTICAL Q8_0 bytes in both paths,
    // so they should match bit-for-bit up to fp32 rounding.
    assert!(
        max_abs < 1e-6,
        "sliced QTensor dequant diverged from full: max_abs={max_abs:.3e}"
    );

    // Also verify boundary cases: slice all experts + empty slice.
    let full_slice = full_q.narrow_experts(0, E)?;
    assert_eq!(full_slice.storage_size_in_bytes(), full_q.storage_size_in_bytes());
    assert_eq!(full_slice.shape().dims(), full_q.shape().dims());

    let empty = full_q.narrow_experts(E, 0)?;
    assert_eq!(empty.storage_size_in_bytes(), 0);
    assert_eq!(empty.shape().dims(), &[0, H, I]);

    // Expected-fail: out of range.
    let bad = full_q.narrow_experts(E - 1, 5);
    assert!(bad.is_err(), "out-of-range narrow_experts should error");

    println!("OK");
    Ok(())
}
