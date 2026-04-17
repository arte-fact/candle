//! Head_dim=128 variant of hip_attention_bench, covering Llama-2/3 and
//! Qwen3.5 attention shapes. Validates the decomposed path on head_dim=128.

use anyhow::Result;
use candle_core::hip_backend::{
    gqa_attention_decode_mv, gqa_attention_decode_mv_decomposed,
};
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_Q: usize = 32;     // Llama-2-7B has 32 heads
const H_KV: usize = 32;    // no GQA on Llama-2-7B; Llama-3-8B has 8 KV, so n_rep=4
const L_Q: usize = 1;
const L_K: usize = 256;
const D: usize = 128;

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
    println!("  n_rep = {n_rep}");

    let q_data: Vec<f32> = (0..B * H_Q * L_Q * D)
        .map(|i| ((i * 31 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let k_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i * 17 + 3) % 119) as f32 / 59.0 - 1.0)
        .collect();
    let v_data: Vec<f32> = (0..B * H_KV * L_K * D)
        .map(|i| ((i * 13 + 11) % 113) as f32 / 57.0 - 1.0)
        .collect();
    let q = Tensor::new(q_data.as_slice(), &dev)?.reshape((B, H_Q, L_Q, D))?;
    let k = Tensor::new(k_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, D))?;
    let v = Tensor::new(v_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, D))?;

    let scale = 1.0 / (D as f64).sqrt();
    let l_k_iter = L_K;

    let out_fused_ref = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
    for _ in 0..WARMUP_ITERS {
        let _ = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
    }
    dev_hip.stream().synchronize()?;
    let mut fused_times = Vec::new();
    for _ in 0..MEASURE_ITERS {
        dev_hip.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_attention_decode_mv(&q, &k, &v, None, scale, l_k_iter)?;
        dev_hip.stream().synchronize()?;
        fused_times.push(t0.elapsed().as_micros() as f64);
    }
    fused_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fused_med = fused_times[MEASURE_ITERS / 2];

    let out_decomp_ref = gqa_attention_decode_mv_decomposed(&q, &k, &v, None, scale, L_K)?;
    for _ in 0..WARMUP_ITERS {
        let _ = gqa_attention_decode_mv_decomposed(&q, &k, &v, None, scale, L_K)?;
    }
    dev_hip.stream().synchronize()?;
    let mut decomp_times = Vec::new();
    for _ in 0..MEASURE_ITERS {
        dev_hip.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let _ = gqa_attention_decode_mv_decomposed(&q, &k, &v, None, scale, L_K)?;
        dev_hip.stream().synchronize()?;
        decomp_times.push(t0.elapsed().as_micros() as f64);
    }
    decomp_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let decomp_med = decomp_times[MEASURE_ITERS / 2];

    let fused_all: Vec<f32> = out_fused_ref
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let decomp_all: Vec<f32> = out_decomp_ref
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let max_err = fused_all
        .iter()
        .zip(decomp_all.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "fused median: {:.2} µs   decomposed median: {:.2} µs   delta: {:+.2} µs ({:+.1}%)",
        fused_med,
        decomp_med,
        fused_med - decomp_med,
        100.0 * (fused_med - decomp_med) / fused_med,
    );
    println!("max_abs_err = {:.3e}", max_err);
    Ok(())
}
