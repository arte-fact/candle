//! F4-M1 correctness test: `gqa_decode_fattn_tile_q8_d128_n4_f32`.
//!
//! Validates against the F32 dequant reference at D=128, n_rep=4
//! (Devstral/Gemma-31B GQA ratio), no mask, across L_k ∈ {64,256,1024}.
//! Exit criteria: max_abs_err < 5e-3. Perf is M2's problem.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{gqa_decode_fattn_tile_q8, QKvCache};
use candle_core::{Device, Tensor, D};

fn run_case(dev: &Device, b: usize, h_kv: usize, l_k: usize) -> Result<()> {
    let dim = 128usize;
    let n_rep = 4usize;
    let h_q = h_kv * n_rep;

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

    let out_tile = gqa_decode_fattn_tile_q8(&q, &cache, l_k, None, scale)?;

    let k_dq = cache.dequantize_k()?.reshape((b, h_kv, l_k, dim))?;
    let v_dq = cache.dequantize_v()?.reshape((b, h_kv, l_k, dim))?;
    let out_ref = f32_reference(&q, &k_dq, &v_dq, scale)?;

    let r: Vec<f32> = out_ref.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let s: Vec<f32> = out_tile.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs = r
        .iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let r_max = r.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = r_max.max(1.0) * 5e-3;
    println!(
        "B={b} H_q={h_q} H_kv={h_kv} D={dim} L_k={l_k}  max_abs_err={max_abs:.3e}  tol={tol:.3e}"
    );
    assert!(
        max_abs < tol,
        "tile-Q fused Q8 vs F32-dequant drift too large: {max_abs:.3e} > {tol:.3e}"
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;

    // M1 correctness.
    run_case(&dev, 1, 8, 64)?;
    run_case(&dev, 1, 8, 256)?;
    run_case(&dev, 1, 8, 1024)?;
    run_case(&dev, 2, 4, 256)?;

    // M2 perf gate: compare tile kernel vs (a) existing single-Q fused Q8
    // kernel and (b) dequant + F32-fused reference. Decision gate is whether
    // the tile float kernel already wins at L_k≥1024 before we invest in
    // dp4a (M2b) or LDS K-tile (M2c).
    println!();
    println!("M2 perf: tile vs gqa_decode_mv_fast_q8 (single-Q) vs dequant+F32-fused");
    for l_k in [64usize, 512, 1024, 2048, 4096] {
        bench_shape(&dev, 1, 8, l_k)?;
    }

    println!("OK");
    Ok(())
}

fn bench_shape(dev: &Device, b: usize, h_kv: usize, l_k: usize) -> Result<()> {
    use candle_core::hip_backend::gqa_attention_decode_mv;
    use candle_core::hip_backend::q8_kv_cache::gqa_decode_mv_fast_q8;
    use std::time::Instant;

    let dim = 128usize;
    let n_rep = 4usize;
    let h_q = h_kv * n_rep;

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

    let reps = 200usize;
    for _ in 0..10 { let _ = gqa_decode_fattn_tile_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps { let _ = gqa_decode_fattn_tile_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let us_tile = t0.elapsed().as_micros() as f64 / reps as f64;

    for _ in 0..10 { let _ = gqa_decode_mv_fast_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps { let _ = gqa_decode_mv_fast_q8(&q, &cache, l_k, None, scale)?; }
    dev.synchronize()?;
    let us_single = t0.elapsed().as_micros() as f64 / reps as f64;

    for _ in 0..10 {
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
        "  L_k={:4}  tile_Q8={:7.2}µs  single_Q8={:7.2}µs  dequant+F32_fused={:7.2}µs   tile/single={:.2}x  tile/F32={:.2}x",
        l_k, us_tile, us_single, us_ref, us_single / us_tile, us_ref / us_tile
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
