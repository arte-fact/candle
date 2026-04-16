//! Attention decode kernel micro-bench (Task #2).
//!
//! Compares the current fused `gqa_attention_decode_mv` against the
//! (future) decomposed 3-kernel path (Q·K gemv → softmax → V·attn gemv)
//! on matched synthetic inputs.
//!
//! Shapes mirror gemma4-E4B SWA layer decode:
//!   B=1, H_q=8, H_kv=2 (n_rep=4), L_q=1, L_k=256, D=256.
//!
//! Usage:
//!   cargo run --release --features hip --example hip_attention_bench
//!
//! Numbers are per-call µs at steady-state. Run 1: warmup. Runs 2-21: measured.

use anyhow::Result;
use candle_core::hip_backend::{
    gqa_attention_decode_mv, gqa_attention_decode_mv_decomposed,
};
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_Q: usize = 8;
const H_KV: usize = 2;
const L_Q: usize = 1;
const L_K: usize = 256;
const D: usize = 256;

const WARMUP_ITERS: usize = 3;
const MEASURE_ITERS: usize = 20;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let dev_hip = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    println!("HIP device: {:?}", dev.location());
    println!("Attention shape: B={B}, H_q={H_Q}, H_kv={H_KV}, L_k={L_K}, D={D}");
    let n_rep = H_Q / H_KV;
    println!("  n_rep = H_q/H_kv = {n_rep}");

    // Synthetic Q, K, V, mask — deterministic RNG for reproducibility.
    let q_data: Vec<f32> = (0..B * H_Q * L_Q * D)
        .map(|i| ((i * 31 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let k_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i * 17 + 3) % 119) as f32 / 59.0 - 1.0)
        .collect();
    let v_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i * 13 + 11) % 113) as f32 / 57.0 - 1.0)
        .collect();

    let q = Tensor::new(q_data.as_slice(), &dev)?
        .reshape((B, H_Q, L_Q, D))?;
    let k = Tensor::new(k_data.as_slice(), &dev)?
        .reshape((B, H_KV, L_K, D))?;
    let v = Tensor::new(v_data.as_slice(), &dev)?
        .reshape((B, H_KV, L_K, D))?;

    let scale = 1.0 / (D as f64).sqrt();
    let l_k_iter = L_K;

    // Warmup + 1 correctness capture.
    let out_fused_ref = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
    for _ in 0..WARMUP_ITERS {
        let _ = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
    }
    dev_hip.stream().synchronize()?;

    // Fused path timing — use hipEventRecord-like pattern via wall-clock
    // bracketed by explicit stream.sync() to exclude host overhead.
    let mut fused_times_us = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        dev_hip.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
        dev_hip.stream().synchronize()?;
        let dt = t0.elapsed().as_micros() as f64;
        fused_times_us.push(dt);
    }

    fused_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_fused = fused_times_us[MEASURE_ITERS / 2];
    let min_fused = fused_times_us[0];
    let p90_fused = fused_times_us[MEASURE_ITERS * 9 / 10];

    println!("\n=== fused (gqa_attention_decode_mv) ===");
    println!(
        "  min={:.2} µs  median={:.2} µs  p90={:.2} µs  (per full attention op)",
        min_fused, median_fused, p90_fused
    );

    // Per-call latency budget: rocprofv3 said 29.9 µs for the underlying
    // kernel — Rust wrapper overhead (arg marshalling, HipDevice::stream()
    // lookups, etc.) adds to the wall-clock we measure here.
    println!("  reference (rocprofv3 kernel-only): 29.93 µs");

    let head_fused: Vec<f32> = out_fused_ref
        .flatten_all()?
        .narrow(0, 0, 4)?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    println!("  out[0..4]: {head_fused:?}");

    // ------------------------------------------------------------------
    // Decomposed path: gemv_qk → softmax → gemv_v.
    // ------------------------------------------------------------------
    let out_decomp_ref = gqa_attention_decode_mv_decomposed(
        &q, &k, &v, None, scale, L_K,
    )?;
    for _ in 0..WARMUP_ITERS {
        let _ = gqa_attention_decode_mv_decomposed(&q, &k, &v, None, scale, L_K)?;
    }
    dev_hip.stream().synchronize()?;

    let mut decomp_times_us = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        dev_hip.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_attention_decode_mv_decomposed(&q, &k, &v, None, scale, L_K)?;
        dev_hip.stream().synchronize()?;
        let dt = t0.elapsed().as_micros() as f64;
        decomp_times_us.push(dt);
    }
    decomp_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_decomp = decomp_times_us[MEASURE_ITERS / 2];
    let min_decomp = decomp_times_us[0];
    let p90_decomp = decomp_times_us[MEASURE_ITERS * 9 / 10];

    println!("\n=== decomposed (gemv_qk → softmax → gemv_v) ===");
    println!(
        "  min={:.2} µs  median={:.2} µs  p90={:.2} µs  (per full attention op)",
        min_decomp, median_decomp, p90_decomp
    );

    let head_decomp: Vec<f32> = out_decomp_ref
        .flatten_all()?
        .narrow(0, 0, 4)?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    println!("  out[0..4]: {head_decomp:?}");

    // Numerical diff between fused and decomposed outputs.
    let fused_all: Vec<f32> = out_fused_ref
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let decomp_all: Vec<f32> = out_decomp_ref
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let max_abs_err: f32 = fused_all
        .iter()
        .zip(decomp_all.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let mean_abs_err: f32 = fused_all
        .iter()
        .zip(decomp_all.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / fused_all.len() as f32;
    println!(
        "\n  max_abs_err = {:.3e}   mean_abs_err = {:.3e}",
        max_abs_err, mean_abs_err
    );

    println!("\n=== Summary ===");
    let delta_us = median_fused - median_decomp;
    let pct = 100.0 * delta_us / median_fused;
    println!(
        "  fused median:      {:>6.2} µs",
        median_fused
    );
    println!(
        "  decomposed median: {:>6.2} µs",
        median_decomp
    );
    println!(
        "  delta:             {:>+6.2} µs ({:+.1}%)",
        delta_us, pct
    );

    Ok(())
}
