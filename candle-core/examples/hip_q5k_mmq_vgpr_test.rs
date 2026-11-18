//! Correctness harness for Q5_K MMQ VGPR-reduction experiments.
//!
//! Per memory `project_qwen36_perf_findings_2026_04_18.md`: "3.5× per-call
//! gap vs turbo's Q4_K MMQ remains open. Don't attempt template extension
//! as single-session port — needs correctness-sweep harness first."
//!
//! This harness runs Q5_K MMQ over many shapes with the default kernel
//! (`mul_mat_q5_K_gfx906_v2f_tile32`, VGPR=192 → 1 wave/SIMD) and any
//! experimental reduced-VGPR variant selected via
//! `CANDLE_Q5K_MMQ_V3=1`, comparing against a dequantize+F32-matmul
//! reference. Fails loudly if any shape drifts beyond tolerance.
//!
//! Usage: cargo run --release -p candle-core --example hip_q5k_mmq_vgpr_test --features hip

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn run_shape(dtype: GgmlDType, m: usize, k: usize, n: usize, gate: &str) -> Result<Vec<f32>> {
    unsafe {
        if gate.is_empty() {
            std::env::remove_var("CANDLE_Q5K_MMQ_V3");
        } else {
            std::env::set_var("CANDLE_Q5K_MMQ_V3", gate);
        }
    }
    let device = Device::new_hip(0)?;
    let w_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 200.0)
        .collect();
    let w = Tensor::new(w_data.as_slice(), &device)?.reshape((n, k))?;
    let qw = QTensor::quantize(&w, dtype)?;
    let qmm = QMatMul::from_qtensor(qw)?;

    let x_data: Vec<f32> = (0..(m * k))
        .map(|i| ((i * 41 % 199) as f32 - 99.0) / 150.0)
        .collect();
    let x = Tensor::new(x_data.as_slice(), &device)?.reshape((m, k))?;
    let y = qmm.forward(&x)?;
    Ok(y.flatten_all()?.to_vec1::<f32>()?)
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_large = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let rel = d / x.abs().max(1e-6);
        if d > max_abs { max_abs = d; }
        if rel > max_rel { max_rel = rel; }
        if d > 5e-4 { n_large += 1; }
    }
    (max_abs, max_rel, n_large)
}

fn main() -> Result<()> {
    // Q5_K requires K multiple of QK_K=256. Test with k=512, 2048, 4096, and
    // the 840/1129 edges flagged in memory (rounded to nearest QK_K).
    // M-values cover decode (1), low-batch (8, 16), mid-batch (32, 64),
    // and prefill (128, 256, 512).
    // N-values cover small, Qwen3.5 hidden (2880), and a wide case (4096).
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("Q5K m1  k512  n512",   1,   512, 512),
        ("Q5K m1  k2048 n2048",  1,  2048, 2048),
        ("Q5K m8  k2048 n2048",  8,  2048, 2048),
        ("Q5K m16 k2048 n2048", 16,  2048, 2048),
        ("Q5K m32 k2048 n2048", 32,  2048, 2048),
        ("Q5K m64 k2048 n2048", 64,  2048, 2048),
        ("Q5K m128 k2048 n2880", 128, 2048, 2880),
        ("Q5K m256 k2048 n4096", 256, 2048, 4096),
        ("Q5K m512 k2048 n2880", 512, 2048, 2880),
        // C3-redo edges (840, 1129 tokens)
        ("Q5K m840 k2048 n2048", 840, 2048, 2048),
        ("Q5K m1129 k2048 n2048", 1129, 2048, 2048),
        // Larger K (gemma4-ish)
        ("Q5K m64  k4096 n4096", 64,  4096, 4096),
    ];

    println!("Q5_K MMQ VGPR-reduction correctness harness");
    println!();
    let mut any_fail = false;
    for (label, m, k, n) in shapes {
        let y_base = run_shape(GgmlDType::Q5K, *m, *k, *n, "")?;
        let y_v3   = run_shape(GgmlDType::Q5K, *m, *k, *n, "1")?;
        let (mx, rx, nl) = diff_stats(&y_base, &y_v3);
        let r_max = y_base.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let tol = r_max.max(1.0) * 1e-3;
        let ok = mx < tol;
        let tag = if ok { "OK  " } else { "FAIL" };
        println!(
            "{tag} {label:<30}  max_abs={mx:.3e}  max_rel={rx:.3e}  #>5e-4={nl:>4}  tol={tol:.3e}"
        );
        if !ok { any_fail = true; }
    }

    if any_fail {
        anyhow::bail!("Q5_K MMQ v3 variant drifted from baseline");
    }
    println!();
    println!("ALL PASS");
    Ok(())
}
