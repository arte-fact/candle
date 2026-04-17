//! Correctness + perf for the Q8 Q·K decode gemv.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{gqa_decode_gemv_qk_q8, QKvCache};
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_Q: usize = 8;
const H_KV: usize = 1;
const D: usize = 128;
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
    println!("B={B}, H_q={H_Q}, H_kv={H_KV}, D={D}, L_k={L_K}, T_max={T_MAX}");

    let q_data: Vec<f32> = (0..B * H_Q * D)
        .map(|i| ((i as i64 * 19 + 3) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let k_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i as i64 * 13 + 5) % 131) as f32 / 65.0 - 1.0)
        .collect();

    let q = Tensor::new(q_data.as_slice(), &dev)?.reshape((B, H_Q, 1, D))?;
    let k = Tensor::new(k_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, D))?;

    let mut cache = QKvCache::new(&hip_dev, B, H_KV, T_MAX, D)?;
    cache.append(&k, &k)?;

    let k_dq = cache.dequantize_k()?.reshape((B, H_KV, L_K, D))?;
    let scores_ref = qk_reference(&q, &k_dq)?;
    let scores_q8 = gqa_decode_gemv_qk_q8(&q, &cache, L_K)?;

    let r: Vec<f32> = scores_ref.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let s: Vec<f32> = scores_q8.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs = r
        .iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Q8 Q·K vs F32(dequant K) max_abs_err = {:.3e}", max_abs);

    // Dequant K is what both paths see — expect only float-rounding noise.
    // Values are sums of D products of range [-1,1]² so ~O(D) ≈ O(128).
    // Rel tol: 1e-5 * value_range.
    let r_max = r.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = r_max * 1e-4;
    assert!(max_abs < tol, "Q8 kernel math drift: {max_abs:.3e} > {tol:.3e}");

    // Perf
    for _ in 0..WARMUP {
        let _ = gqa_decode_gemv_qk_q8(&q, &cache, L_K)?;
    }
    hip_dev.stream().synchronize()?;
    let mut q8_times = Vec::new();
    for _ in 0..ITERS {
        hip_dev.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_decode_gemv_qk_q8(&q, &cache, L_K)?;
        hip_dev.stream().synchronize()?;
        q8_times.push(t0.elapsed().as_micros() as f64);
    }
    q8_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = q8_times[ITERS / 2];
    println!("Q8 Q·K median: {:.2} µs  (L_k={L_K}, D={D})", med);

    Ok(())
}

fn qk_reference(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    // q: (B, H_q, 1, D)  k: (B, H_kv, L_k, D)
    let (_, h_q, _, _) = q.dims4()?;
    let (_, h_kv, _, _) = k.dims4()?;
    let n_rep = h_q / h_kv;
    let k_exp = if n_rep > 1 {
        let k5 = k.unsqueeze(2)?;
        let (_, hv, _, lkv, dv) = k5.dims5()?;
        let k5 = k5.expand((k5.dim(0)?, hv, n_rep, lkv, dv))?;
        let (bv, hv, nr, lkv, dv) = k5.dims5()?;
        k5.reshape((bv, hv * nr, lkv, dv))?
    } else {
        k.clone()
    };
    // Q·K^T: (B, H_q, 1, D) @ (B, H_q, D, L_k) → (B, H_q, 1, L_k)
    let kt = k_exp.contiguous()?.transpose(2, 3)?.contiguous()?;
    Ok(q.matmul(&kt)?)
}
