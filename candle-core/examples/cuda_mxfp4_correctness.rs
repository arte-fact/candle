//! Correctness test for the CUDA MXFP4 dequant kernel.
//!
//! Builds a synthetic byte buffer of `BlockMxfp4` blocks (1 byte E8M0 shared
//! exponent + 16 bytes of packed e2m1 quants per 32-element block), exercises
//! the full 16-value lookup table across several exponent scales, and asserts
//! the CUDA kernel output matches the CPU reference bit-for-bit.
use std::borrow::Cow;

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{Device, Shape};

fn main() -> Result<()> {
    let cuda = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    // 64 blocks × 32 elements = 2048 elements total. Scales span 2^-8..2^+7.
    // Nibble sequence cycles through every index 0..15 so every KVALUES_MXFP4
    // entry is touched.
    // Exponents chosen to hit every code path in `e8m0_to_fp32_half`:
    //   0, 1        — denormal branch (x < 2)
    //   2, 127      — transition / unit scale
    //   254, 255    — largest normalized values
    //   120..=135   — typical model weight scales
    // 64 blocks total × 32 elements = 2048 outputs; covers every KVALUES_MXFP4
    // nibble and spans >1 CUDA grid block (nb=8).
    let exponents: Vec<u8> = [0u8, 1, 2, 126, 127, 128, 254, 255]
        .into_iter()
        .chain((120u8..=135u8).cycle().take(56))
        .collect();
    let n_blocks = exponents.len();
    assert_eq!(n_blocks, 64);
    let mut bytes = Vec::with_capacity(n_blocks * 17);
    for (b, &e) in exponents.iter().enumerate() {
        bytes.push(e);
        for i in 0..16 {
            let lo = ((b + i) % 16) as u8;
            let hi = ((b + i + 7) % 16) as u8;
            bytes.push((hi << 4) | lo);
        }
    }
    let shape: Shape = (n_blocks * 32,).into();

    let cpu_storage = QStorage::from_data(Cow::Borrowed(&bytes), &cpu, GgmlDType::Mxfp4)?;
    let cpu_qt = QTensor::new(cpu_storage, shape.clone())?;
    let cpu_out = cpu_qt.dequantize(&cpu)?.to_vec1::<f32>()?;

    let cuda_storage = QStorage::from_data(Cow::Borrowed(&bytes), &cuda, GgmlDType::Mxfp4)?;
    let cuda_qt = QTensor::new(cuda_storage, shape)?;
    let cuda_out = cuda_qt.dequantize(&cuda)?.to_vec1::<f32>()?;

    assert_eq!(cpu_out.len(), cuda_out.len());
    let mut max_abs = 0.0f32;
    let mut mismatches = 0usize;
    for (a, b) in cpu_out.iter().zip(cuda_out.iter()) {
        let d = (a - b).abs();
        if d > 0.0 {
            mismatches += 1;
        }
        if d > max_abs {
            max_abs = d;
        }
    }

    println!("n_elements: {}", cpu_out.len());
    println!("max_abs_diff: {max_abs}");
    println!("mismatches:   {mismatches}/{}", cpu_out.len());

    if mismatches == 0 {
        println!("PASS — CUDA MXFP4 dequant matches CPU bit-for-bit");
    } else {
        for i in 0..std::cmp::min(16, cpu_out.len()) {
            println!("  [{i}] cpu={} cuda={}", cpu_out[i], cuda_out[i]);
        }
        anyhow::bail!("CUDA MXFP4 dequant diverges from CPU reference");
    }

    // Also round-trip the f16 path. Skip elements whose true value is outside
    // f16 range (the extreme e=254/255 test blocks overflow by design).
    const F16_MAX: f32 = 65504.0;
    let cuda_f16 = cuda_qt.dequantize_f16(&cuda)?.to_vec1::<half::f16>()?;
    let mut max_rel_f16 = 0.0f32;
    let mut compared = 0usize;
    for (a, b) in cpu_out.iter().zip(cuda_f16.iter()) {
        if a.abs() > F16_MAX || !a.is_finite() {
            continue;
        }
        compared += 1;
        let bf = b.to_f32();
        let denom = a.abs().max(1.0);
        let rel = ((a - bf).abs()) / denom;
        if rel > max_rel_f16 {
            max_rel_f16 = rel;
        }
    }
    println!("f16 max_rel_diff (over {compared} in-range elems): {max_rel_f16}");
    // f16 has 10-bit mantissa → ~2^-10 ≈ 1e-3 relative error ceiling.
    assert!(
        max_rel_f16 < 2e-3,
        "f16 dequant relative error too large: {max_rel_f16}"
    );
    println!("PASS — CUDA MXFP4 f16 dequant within f16 precision");

    Ok(())
}
