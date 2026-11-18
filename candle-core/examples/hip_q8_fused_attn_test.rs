//! Correctness test for the fused Q8-KV attention kernel
//! `gqa_decode_mv_fast_q8_d{128,256,512}_f32`.
//!
//! The fused kernel does Q·K + online-softmax + V·attn in a single
//! launch, reading Q8_0 blocks directly (no F32 dequant). This test
//! validates against the F32-reference attention computed on the
//! dequantised K/V from the same cache.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{
    gqa_decode_mv_fast_q8, QKvCache,
};
use candle_core::{Device, Tensor, D};

fn run_case(dev: &Device, b: usize, h_q: usize, h_kv: usize, dim: usize, l_k: usize) -> Result<()> {
    let hip_dev = match dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    let t_max = l_k.max(64);
    let scale = 1.0 / (dim as f64).sqrt();

    let q_data: Vec<f32> = (0..b * h_q * dim)
        .map(|i| ((i as i64 * 19 + 3) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let k_data: Vec<f32> = (0..b * h_kv * l_k * dim)
        .map(|i| ((i as i64 * 13 + 5) % 131) as f32 / 65.0 - 1.0)
        .collect();
    let v_data: Vec<f32> = (0..b * h_kv * l_k * dim)
        .map(|i| ((i as i64 * 11 + 2) % 113) as f32 / 56.0 - 1.0)
        .collect();
    let q = Tensor::new(q_data.as_slice(), dev)?.reshape((b, h_q, 1, dim))?;
    let k = Tensor::new(k_data.as_slice(), dev)?.reshape((b, h_kv, l_k, dim))?;
    let v = Tensor::new(v_data.as_slice(), dev)?.reshape((b, h_kv, l_k, dim))?;

    let mut cache = QKvCache::new(&hip_dev, b, h_kv, t_max, dim)?;
    cache.append(&k, &v)?;

    // Fused kernel output (no mask for simplicity)
    let out_fused = gqa_decode_mv_fast_q8(&q, &cache, l_k, None, scale)?;

    // Reference: pipeline the same Q8 K/V through dequant + f32 attention.
    let k_dq = cache.dequantize_k()?.reshape((b, h_kv, l_k, dim))?;
    let v_dq = cache.dequantize_v()?.reshape((b, h_kv, l_k, dim))?;
    let out_ref = f32_reference(&q, &k_dq, &v_dq, scale)?;

    let r: Vec<f32> = out_ref.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let s: Vec<f32> = out_fused.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs = r
        .iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let r_max = r.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    // dp4a variant (CANDLE_Q8_ATTN_DP4A=1) adds Q-quantisation error on top
    // of the K/V quant. ~2e-4 absolute is typical; ~1e-8 for the non-dp4a.
    let dp4a_on = std::env::var("CANDLE_Q8_ATTN_DP4A").ok().as_deref() == Some("1");
    let tol = if dp4a_on { r_max.max(1.0) * 2e-3 } else { r_max.max(1.0) * 5e-5 };
    println!(
        "B={b} H_q={h_q} H_kv={h_kv} D={dim} L_k={l_k}  max_abs_err={max_abs:.3e}  tol={tol:.3e}"
    );
    assert!(
        max_abs < tol,
        "fused Q8 vs F32-dequant drift too large: {max_abs:.3e} > {tol:.3e}"
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;

    // D=128 / 256 / 512 — the supported head_dims.
    run_case(&dev, 1, 8, 1, 128, 64)?;
    run_case(&dev, 1, 8, 1, 128, 256)?;
    run_case(&dev, 1, 8, 1, 256, 64)?;
    run_case(&dev, 1, 8, 1, 256, 256)?;
    run_case(&dev, 1, 4, 2, 256, 128)?;   // GQA: n_rep = 2
    run_case(&dev, 1, 2, 1, 512, 64)?;
    run_case(&dev, 1, 2, 1, 512, 256)?;

    println!();
    println!("Microbench: fused Q8 vs dequant+F32 reference (100 reps)");
    bench_shape(&dev, 1, 8, 1, 256, 256)?;  // gemma4-A4B SWA shape
    bench_shape(&dev, 1, 2, 1, 512, 256)?;  // gemma4-A4B global shape
    bench_shape(&dev, 1, 8, 1, 256, 1024)?; // longer context

    println!("OK");
    Ok(())
}

fn bench_shape(dev: &Device, b: usize, h_q: usize, h_kv: usize, dim: usize, l_k: usize) -> Result<()> {
    use std::time::Instant;
    let hip_dev = match dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    let scale = 1.0 / (dim as f64).sqrt();
    let q_data: Vec<f32> = (0..b * h_q * dim).map(|i| (i as f32 * 0.017).sin()).collect();
    let k_data: Vec<f32> = (0..b * h_kv * l_k * dim).map(|i| (i as f32 * 0.013).sin()).collect();
    let v_data: Vec<f32> = (0..b * h_kv * l_k * dim).map(|i| (i as f32 * 0.011).cos()).collect();
    let q = Tensor::new(q_data.as_slice(), dev)?.reshape((b, h_q, 1, dim))?;
    let k = Tensor::new(k_data.as_slice(), dev)?.reshape((b, h_kv, l_k, dim))?;
    let v = Tensor::new(v_data.as_slice(), dev)?.reshape((b, h_kv, l_k, dim))?;
    let mut cache = QKvCache::new(&hip_dev, b, h_kv, l_k.max(64), dim)?;
    cache.append(&k, &v)?;

    let reps = 100;
    // Warm-up
    for _ in 0..5 { let _ = gqa_decode_mv_fast_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps { let _ = gqa_decode_mv_fast_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let us_fused = t0.elapsed().as_micros() as f64 / reps as f64;

    use candle_core::hip_backend::gqa_attention_decode_mv;
    // Production F32 reference: dequant the slab, then run the SAME fused
    // F32 kernel gemma4 uses on multi-GPU (gqa_decode_mv_fast_d*_f32 via
    // gqa_attention_decode_mv). This is the real target to beat.
    for _ in 0..5 {
        let k_dq = cache.dequantize_k()?.reshape((b, h_kv, l_k, dim))?;
        let v_dq = cache.dequantize_v()?.reshape((b, h_kv, l_k, dim))?;
        let _ = gqa_attention_decode_mv(&q, &k_dq, &v_dq, None, scale, l_k)?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        let k_dq = cache.dequantize_k()?.reshape((b, h_kv, l_k, dim))?;
        let v_dq = cache.dequantize_v()?.reshape((b, h_kv, l_k, dim))?;
        let _ = gqa_attention_decode_mv(&q, &k_dq, &v_dq, None, scale, l_k)?;
    }
    dev.synchronize()?;
    let us_ref = t0.elapsed().as_micros() as f64 / reps as f64;

    println!(
        "  D={} L_k={:4}  fused_Q8={:7.2}µs  ref(dequant+fused_F32)={:7.2}µs  speedup={:.2}x",
        dim, l_k, us_fused, us_ref, us_ref / us_fused
    );
    Ok(())
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?)
}

fn f32_reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let (_, h_q, _, _) = q.dims4()?;
    let (_, h_kv, _, _) = k.dims4()?;
    let n_rep = h_q / h_kv;
    let (k_exp, v_exp) = if n_rep > 1 {
        let expand = |t: &Tensor| -> Result<Tensor> {
            let t5 = t.unsqueeze(2)?;
            let (bv, hv, _one, lkv, dv) = t5.dims5()?;
            let t5 = t5.expand((bv, hv, n_rep, lkv, dv))?;
            let (bv, hv, nr, lkv, dv) = t5.dims5()?;
            Ok(t5.reshape((bv, hv * nr, lkv, dv))?)
        };
        (expand(k)?, expand(v)?)
    } else {
        (k.clone(), v.clone())
    };
    let kt = k_exp.contiguous()?.transpose(2, 3)?.contiguous()?;
    let s = q.matmul(&kt)?.affine(scale, 0.0)?;
    let p = softmax_last_dim(&s)?;
    Ok(p.matmul(&v_exp.contiguous()?)?)
}
