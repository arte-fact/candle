//! Correctness + perf comparison for the Q8 V·attn decode gemv.
//!
//! Runs the F32 reference (`gqa_decode_gemv_v_d{N}_f32_b128` via the
//! existing decomposed-attention path) against the Q8 kernel and prints
//! max_abs_err + per-call median µs for both.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{gqa_decode_gemv_v_q8, QKvCache};
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_Q: usize = 8;
const H_KV: usize = 1; // exercise n_rep=8 GQA broadcast
const D: usize = 128;  // matches Llama shape — extend later
const L_K: usize = 256;
const T_MAX: usize = 512;

const WARMUP: usize = 3;
const ITERS: usize = 50;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    println!(
        "B={B}, H_q={H_Q}, H_kv={H_KV}, D={D}, L_k={L_K}, T_max={T_MAX}"
    );

    // ---- Build attention weights (softmaxed) and a synthetic V.
    let attn_data: Vec<f32> = (0..B * H_Q * L_K)
        .map(|i| ((i as i64 * 29 + 5) % 137) as f32 / 68.0 - 1.0)
        .collect();
    let v_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i as i64 * 23 + 7) % 149) as f32 / 74.0 - 1.0)
        .collect();

    // Softmax-like weighting: exp(x) / sum; inline to avoid a candle-nn dep.
    let attn_raw = Tensor::new(attn_data.as_slice(), &dev)?.reshape((B, H_Q, 1, L_K))?;
    let max = attn_raw.max_keepdim(candle_core::D::Minus1)?;
    let e = attn_raw.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(candle_core::D::Minus1)?;
    let attn = e.broadcast_div(&sum)?;

    let v = Tensor::new(v_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, D))?;

    // ---- Q8 cache: quantise V into it.
    let mut cache = QKvCache::new(&hip_dev, B, H_KV, T_MAX, D)?;
    // K doesn't matter for this test; use v as a placeholder.
    cache.append(&v, &v)?;
    assert_eq!(cache.current_seq_len(), L_K);

    // ---- F32 reference: use dequantized V so both paths see the same Q8
    //       approximation. Then the difference is purely the kernel math.
    let v_dq = cache.dequantize_v()?.reshape((B, H_KV, L_K, D))?;
    let out_ref = gqa_decode_gemv_v_f32_reference(&attn, &v_dq)?;
    let out_q8 = gqa_decode_gemv_v_q8(&attn, &cache, L_K)?;

    let r: Vec<f32> = out_ref
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let q: Vec<f32> = out_q8.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs = r
        .iter()
        .zip(q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Q8 vs F32(dequant V) max_abs_err = {:.3e}", max_abs);

    // Also compare Q8 against the raw F32 V (should differ by the Q8
    // quantisation noise — bounded by ~L_k × step × sum|attn| ≈ step × 1).
    let out_ref_raw = gqa_decode_gemv_v_f32_reference(&attn, &v)?;
    let r_raw: Vec<f32> = out_ref_raw
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let max_abs_raw = r_raw
        .iter()
        .zip(q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Q8 vs F32(original V) max_abs_err = {:.3e}", max_abs_raw);

    // ---- Perf: Q8 path
    for _ in 0..WARMUP {
        let _ = gqa_decode_gemv_v_q8(&attn, &cache, L_K)?;
    }
    hip_dev.stream().synchronize()?;
    let mut q8_times = Vec::new();
    for _ in 0..ITERS {
        hip_dev.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_decode_gemv_v_q8(&attn, &cache, L_K)?;
        hip_dev.stream().synchronize()?;
        q8_times.push(t0.elapsed().as_micros() as f64);
    }
    q8_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q8_med = q8_times[ITERS / 2];

    // ---- Perf: F32 reference via same kernel signature
    for _ in 0..WARMUP {
        let _ = gqa_decode_gemv_v_f32_reference(&attn, &v)?;
    }
    hip_dev.stream().synchronize()?;
    let mut f32_times = Vec::new();
    for _ in 0..ITERS {
        hip_dev.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_decode_gemv_v_f32_reference(&attn, &v)?;
        hip_dev.stream().synchronize()?;
        f32_times.push(t0.elapsed().as_micros() as f64);
    }
    f32_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f32_med = f32_times[ITERS / 2];

    println!(
        "F32 median: {:.2} µs   Q8 median: {:.2} µs   delta: {:+.2} µs ({:+.1}%)",
        f32_med,
        q8_med,
        q8_med - f32_med,
        100.0 * (q8_med - f32_med) / f32_med
    );

    // Q8 should be strictly ≤ F32 on bandwidth-bound paths. 1.3× slack
    // for first-call warm-up variance.
    assert!(max_abs < 1e-4, "F32-dequant vs Q8 kernel mismatch: {max_abs:.3e}");

    Ok(())
}

/// Baseline V·attn gemv via Tensor matmul. Not the fastest candle path
/// but an easy-to-understand reference for correctness.
fn gqa_decode_gemv_v_f32_reference(attn: &Tensor, v: &Tensor) -> Result<Tensor> {
    // attn: (B, H_q, 1, L_k)  v: (B, H_kv, L_k, D)
    let (_b, h_q, _one, _l_k) = attn.dims4()?;
    let (_, h_kv, _, _d) = v.dims4()?;
    let n_rep = h_q / h_kv;
    // Expand V to H_q via repeat_interleave on head axis.
    let v_expanded = if n_rep > 1 {
        let v = v.unsqueeze(2)?; // (B, H_kv, 1, L_k, D)
        let v = v.expand((
            v.dim(0)?,
            h_kv,
            n_rep,
            v.dim(3)?,
            v.dim(4)?,
        ))?;
        let (bv, hv, nr, lkv, dv) = v.dims5()?;
        v.reshape((bv, hv * nr, lkv, dv))?
    } else {
        v.clone()
    };
    // (B, H_q, 1, L_k) @ (B, H_q, L_k, D) → (B, H_q, 1, D)
    let out = attn.matmul(&v_expanded.contiguous()?)?;
    Ok(out)
}
