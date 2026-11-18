//! Correctness harness for the D1 4-warp LDS-tiled MMQ port (Q4_K + Q6_K).
//!
//! Per `project_devstral_prefill_gap_elucidation_2026_04_20.md`: Devstral
//! pp512 gap vs llama.cpp is 97% in Q4_K (2.81× slower) and Q6_K (4.50×
//! slower) MMQ kernels. Root cause is candle's single-warp turbo_dense
//! vs llama.cpp's 4-warp cooperative LDS-tiled block. Closing this is a
//! multi-session structural rewrite.
//!
//! Per `feedback_measure_before_claim.md` and the Q5_K harness pattern:
//! land a correctness sweep BEFORE the new kernel so any drift in the
//! 4-warp variant is caught at shape-level (not at pp-bench-level).
//!
//! The harness runs Q4_K + Q6_K MMQ over the shapes Devstral/Mistral3
//! hits at pp{128,512,2048} (attention + FFN projections), with both
//! the baseline (turbo_dense_x8) and the gated new kernel (under
//! `CANDLE_Q4K_MMQ_4WARP` / `CANDLE_Q6K_MMQ_4WARP`). Each variant is
//! compared against an F32 dequant+matmul reference. Fails loudly if
//! any shape drifts beyond quant-noise tolerance.
//!
//! Usage:
//!   cargo run --release -p candle-core --example hip_q4k_q6k_mmq_4warp_test --features hip

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

/// Run one MMQ with the given env gate setting and return the flat output.
fn run_shape(
    dtype: GgmlDType,
    m: usize,
    k: usize,
    n: usize,
    gate_name: &str,
    gate_val: &str,
) -> Result<Vec<f32>> {
    unsafe {
        if gate_val.is_empty() {
            std::env::remove_var(gate_name);
        } else {
            std::env::set_var(gate_name, gate_val);
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

/// Reference: dequantize weight to F32 then F32 matmul — bypasses all MMQ paths.
fn run_reference(dtype: GgmlDType, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let device = Device::new_hip(0)?;
    let w_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 200.0)
        .collect();
    let w = Tensor::new(w_data.as_slice(), &device)?.reshape((n, k))?;
    let qw = QTensor::quantize(&w, dtype)?;
    let w_dq = qw.dequantize(&device)?;
    let x_data: Vec<f32> = (0..(m * k))
        .map(|i| ((i * 41 % 199) as f32 - 99.0) / 150.0)
        .collect();
    let x = Tensor::new(x_data.as_slice(), &device)?.reshape((m, k))?;
    let y = x.matmul(&w_dq.t()?)?;
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
        if d > 5e-3 { n_large += 1; }
    }
    (max_abs, max_rel, n_large)
}

/// Devstral-24B (mistral3) + Mistral dense shapes, plus a Qwen3.5-27B cross-check.
/// M values: prefill cases 128, 512, 2048 (primary targets). K/N covers
/// q/k/v/o (embed=5120), gate/up/down (ffn=32768), and a 4096 edge.
fn q4k_shapes() -> &'static [(&'static str, usize, usize, usize)] {
    &[
        // attention q/o projection (square-ish)
        ("Q4K m128  k5120 n5120",  128, 5120, 5120),
        ("Q4K m512  k5120 n5120",  512, 5120, 5120),
        ("Q4K m2048 k5120 n5120", 2048, 5120, 5120),
        // k/v projection (narrow) — mistral3 GQA (8 kv heads)
        ("Q4K m512  k5120 n1024",  512, 5120, 1024),
        // FFN gate/up
        ("Q4K m128  k5120 n32768", 128, 5120, 32768),
        ("Q4K m512  k5120 n32768", 512, 5120, 32768),
        // edge shapes — Qwen3.5 hidden 2880 / head 128
        ("Q4K m64   k2048 n2880",   64, 2048, 2880),
        ("Q4K m512  k4096 n4096",  512, 4096, 4096),
        // need-check edges — nrows not a multiple of 64
        ("Q4K m256  k5120 n1025",  256, 5120, 1025),
        ("Q4K m512  k5120 n31999", 512, 5120, 31999),
    ]
}

/// Q6_K used in Q4_K_M for attn.v + ffn.down. down_proj is the big one.
fn q6k_shapes() -> &'static [(&'static str, usize, usize, usize)] {
    &[
        // attn.v projection (mistral3 GQA)
        ("Q6K m512  k5120 n1024",  512, 5120, 1024),
        // ffn.down — 32k → 5k, dominates Q6_K cost
        ("Q6K m128  k32768 n5120", 128, 32768, 5120),
        ("Q6K m512  k32768 n5120", 512, 32768, 5120),
        ("Q6K m2048 k32768 n5120", 2048, 32768, 5120),
        // square edges
        ("Q6K m256  k5120 n5120",  256, 5120, 5120),
        // need-check edge
        ("Q6K m512  k32768 n5119", 512, 32768, 5119),
    ]
}

fn sweep(dtype: GgmlDType, gate_name: &str, shapes: &[(&'static str, usize, usize, usize)]) -> Result<bool> {
    let mut any_fail = false;
    println!();
    println!("=== {:?}  gate={} ===", dtype, gate_name);
    for (label, m, k, n) in shapes {
        let y_ref  = run_reference(dtype, *m, *k, *n)?;
        let y_base = run_shape(dtype, *m, *k, *n, gate_name, "0")?;
        let y_new  = run_shape(dtype, *m, *k, *n, gate_name, "1")?;

        // Tolerance scales with output magnitude (quant noise is relative).
        let r_max = y_ref.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let tol_quant = r_max.max(1.0) * 5e-2;  // quant vs F32 ref: 5% of max
        let tol_xk    = r_max.max(1.0) * 1e-3;  // kernel vs kernel: 0.1%

        let (mb_a, mb_r, _) = diff_stats(&y_ref, &y_base);
        let (mn_a, mn_r, _) = diff_stats(&y_ref, &y_new);
        let (xk_a, xk_r, xk_n) = diff_stats(&y_base, &y_new);

        let ok_base = mb_a < tol_quant;
        let ok_new  = mn_a < tol_quant;
        let ok_xk   = xk_a < tol_xk;

        let tag = if ok_base && ok_new && ok_xk { "OK  " } else { "FAIL" };
        println!(
            "{tag} {label:<28}  base-vs-ref={mb_a:.3e} (r{mb_r:.2e})  new-vs-ref={mn_a:.3e} (r{mn_r:.2e})  base-vs-new={xk_a:.3e} (r{xk_r:.2e}) n>{}={}",
            5e-3, xk_n
        );
        if !ok_base || !ok_new || !ok_xk { any_fail = true; }
    }
    Ok(!any_fail)
}

fn main() -> Result<()> {
    println!("D1 MMQ 4-warp port correctness harness (Q4_K + Q6_K)");
    let ok_q4k = sweep(GgmlDType::Q4K, "CANDLE_Q4K_MMQ_4WARP", q4k_shapes())?;
    let ok_q6k = sweep(GgmlDType::Q6K, "CANDLE_Q6K_MMQ_4WARP", q6k_shapes())?;

    println!();
    if !ok_q4k || !ok_q6k {
        anyhow::bail!("D1 MMQ 4-warp correctness sweep had failures");
    }
    println!("ALL PASS");
    Ok(())
}
