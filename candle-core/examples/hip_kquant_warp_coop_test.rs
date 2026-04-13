//! Compare K-quant warp-coop vs template MMVQ kernels for correctness.
//!
//! Runs the same Q4_K/Q5_K/Q6_K vector-matrix multiply twice — once with the
//! template kernel and once with the gfx906 warp-cooperative kernel — and
//! reports L2 norm of the difference.
//!
//! Usage:
//!   cargo run --release --features hip --example hip_kquant_warp_coop_test

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn test_dtype(dtype: GgmlDType, device: &Device) -> Result<()> {
    // Shape matching typical lm_head: cols=2560 (multiple of 256 for K-quants), rows=4096.
    let cols = 2560usize;
    let rows = 4096usize;

    // Seeded deterministic weights so both runs see identical quantization.
    let weight_data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 31 + 17) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), device)?.reshape((rows, cols))?;
    let qweight = QTensor::quantize(&weight, dtype)?;
    let qmatmul = QMatMul::from_qtensor(qweight)?;

    // Single-row input → triggers b_size=1 warp-coop dispatch.
    let input_data: Vec<f32> = (0..cols)
        .map(|i| ((i * 7 + 3) % 31) as f32 / 15.0 - 1.0)
        .collect();
    let input = Tensor::new(input_data.as_slice(), device)?.reshape((1, cols))?;

    // Run with warp-coop disabled (template kernel = reference).
    std::env::remove_var("CANDLE_KQUANT_WARP_COOP");
    let out_template = qmatmul.forward(&input)?;
    let out_template_v = out_template.to_vec2::<f32>()?[0].clone();

    // Run with warp-coop enabled.
    std::env::set_var("CANDLE_KQUANT_WARP_COOP", "1");
    let out_wc = qmatmul.forward(&input)?;
    let out_wc_v = out_wc.to_vec2::<f32>()?[0].clone();
    std::env::remove_var("CANDLE_KQUANT_WARP_COOP");

    // Compute L2 norm of difference, relative L2, and max abs diff.
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..rows {
        let d = (out_template_v[i] - out_wc_v[i]) as f64;
        diff_sq += d * d;
        ref_sq += (out_template_v[i] as f64).powi(2);
        max_abs = max_abs.max(d.abs());
    }
    let l2_diff = diff_sq.sqrt();
    let l2_ref = ref_sq.sqrt();
    let rel_l2 = if l2_ref > 0.0 { l2_diff / l2_ref } else { 0.0 };

    println!(
        "{:?}: L2 ref = {:.4}, L2 diff = {:.4e}, rel L2 = {:.4e}, max abs = {:.4e}",
        dtype, l2_ref, l2_diff, rel_l2, max_abs
    );

    // Spot-check a few elements.
    println!(
        "  template[0..4] = {:?}",
        &out_template_v[..4]
    );
    println!("  warpcoop[0..4] = {:?}", &out_wc_v[..4]);

    // Accept if rel L2 < 1e-4 (numerical noise threshold for quantized dp4a with same data).
    if rel_l2 > 1e-4 {
        eprintln!("  ⚠ WARP-COOP DIVERGES from template by > 1e-4");
    } else {
        println!("  ✓ warp-coop matches template");
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_hip(0)?;
    println!("HIP device: {:?}\n", device.location());

    println!("=== K-quant warp-coop vs template MMVQ correctness ===\n");

    for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
        test_dtype(dtype, &device)?;
    }

    Ok(())
}
