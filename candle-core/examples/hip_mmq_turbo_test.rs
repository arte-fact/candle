//! Bit-exact test for the Q4_0 MMQ turbo port.
//!
//! Runs the same Q4_0 QMatMul twice — once with the baseline path
//! (`CANDLE_MMQ_TURBO_PORT=0`) and once with the turbo port
//! (`CANDLE_MMQ_TURBO_PORT=1`) — and diffs the outputs.
//!
//! Usage:   cargo run --release --example hip_mmq_turbo_test --features hip
use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn run_once(
    gate: &str,
    dtype: GgmlDType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // SAFETY: we're single-threaded in main.
    unsafe {
        std::env::set_var("CANDLE_MMQ_TURBO_PORT", gate);
    }

    let device = Device::new_hip(0)?;
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 200.0)
        .collect();
    let w = Tensor::new(weight_data.as_slice(), &device)?.reshape((n, k))?;
    let qw = QTensor::quantize(&w, dtype)?;
    let qmm = QMatMul::from_qtensor(qw)?;

    let input_data: Vec<f32> = (0..(m * k))
        .map(|i| ((i * 41 % 199) as f32 - 99.0) / 150.0)
        .collect();
    let x = Tensor::new(input_data.as_slice(), &device)?.reshape((m, k))?;

    let y = qmm.forward(&x)?;
    let flat: Vec<f32> = y.flatten_all()?.to_vec1::<f32>()?;

    // Also return a sample of the input to confirm both runs see the same data.
    let x_sample = x.flatten_all()?.to_vec1::<f32>()?;
    Ok((flat, x_sample))
}

fn main() -> Result<()> {
    // (label, dtype, m, k, n).  Q4_0 + Q4_1 covered by M2/M3a; mmq_x
    // selection: <=8→x8, <=16→x16, <=32→x32, else x64.
    let shapes = [
        ("Q4_0 m13  k2048 n2048", GgmlDType::Q4_0, 13, 2048, 2048),
        ("Q4_0 m128 k2048 n512",  GgmlDType::Q4_0, 128, 2048, 512),
        ("Q4_0 m513 k2048 n256",  GgmlDType::Q4_0, 513, 2048, 256),
        // Q4_1 aligned-K shapes — exercise the M3a turbo kernel.
        ("Q4_1 m13  k2048 n2048", GgmlDType::Q4_1, 13, 2048, 2048),
        ("Q4_1 m513 k2048 n2048", GgmlDType::Q4_1, 513, 2048, 2048),
        // Qwen3.5-9B FFN-down (K=15360 = 60×256, aligned; N=2880).
        ("Q4_1 m513 k15360 n2880", GgmlDType::Q4_1, 513, 15360, 2880),
        // Non-aligned K: gates out (M3b TODO). Should land ~0 diff because
        // the dispatcher routes to baseline.
        ("Q4_1 m13  k2880 n2880", GgmlDType::Q4_1, 13, 2880, 2880),
    ];

    for (label, dtype, m, k, n) in shapes {
        println!("\n=== {label} ===");

        let (out0, x0) = run_once("0", dtype, m, k, n)?;
        let (out1, x1) = run_once("1", dtype, m, k, n)?;

        assert_eq!(out0.len(), out1.len(), "output length mismatch");
        // Sanity check: same inputs.
        let xs_diff = x0
            .iter()
            .zip(x1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("input max-abs diff between runs: {xs_diff:.6e}");
        assert!(xs_diff < 1e-6, "inputs diverged");

        let (mut mad, mut mse, mut n_mismatch) = (0.0f32, 0.0f64, 0usize);
        for (a, b) in out0.iter().zip(out1.iter()) {
            let d = (a - b).abs();
            mad = mad.max(d);
            mse += (d as f64) * (d as f64);
            if d > 1e-3 {
                n_mismatch += 1;
            }
        }
        let rmse = (mse / out0.len() as f64).sqrt();
        let m0 = out0.iter().cloned().fold(0.0f32, f32::max);
        let m1 = out1.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "max-abs-diff={mad:.4e}  rmse={rmse:.4e}  n_mismatch(>1e-3)={n_mismatch}/{}  \
             max(out_baseline)={m0:.3}  max(out_turbo)={m1:.3}",
            out0.len()
        );
        println!(
            "sample out_base[0..6] = {:?}",
            &out0[..6.min(out0.len())]
        );
        println!(
            "sample out_turbo[0..6] = {:?}",
            &out1[..6.min(out1.len())]
        );
    }

    Ok(())
}
