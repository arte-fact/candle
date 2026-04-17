//! End-to-end Q8 KV attention pipeline:
//!   F32 Q → Q8_K·Q → softmax(scale·scores + mask) → Q8_V·attn → F32 out
//!
//! Compared against F32-dequant reference over the same Q8 KV. Delta
//! should be purely float-rounding noise across the matmul chain.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{
    gqa_decode_gemv_qk_q8, gqa_decode_gemv_v_q8, QKvCache,
};
use candle_core::{Device, Tensor, D};

const B: usize = 1;
const H_Q: usize = 8;
const H_KV: usize = 1;
const DIM: usize = 128;
const L_K: usize = 256;
const T_MAX: usize = 512;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    println!("B={B}, H_q={H_Q}, H_kv={H_KV}, D={DIM}, L_k={L_K}");
    let scale = 1.0 / (DIM as f64).sqrt();

    let q_data: Vec<f32> = (0..B * H_Q * DIM)
        .map(|i| ((i as i64 * 19 + 3) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let k_data: Vec<f32> = (0..B * H_KV * L_K * DIM)
        .map(|i| ((i as i64 * 13 + 5) % 131) as f32 / 65.0 - 1.0)
        .collect();
    let v_data: Vec<f32> = (0..B * H_KV * L_K * DIM)
        .map(|i| ((i as i64 * 11 + 2) % 113) as f32 / 56.0 - 1.0)
        .collect();
    let q = Tensor::new(q_data.as_slice(), &dev)?.reshape((B, H_Q, 1, DIM))?;
    let k = Tensor::new(k_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, DIM))?;
    let v = Tensor::new(v_data.as_slice(), &dev)?.reshape((B, H_KV, L_K, DIM))?;

    let mut cache = QKvCache::new(&hip_dev, B, H_KV, T_MAX, DIM)?;
    cache.append(&k, &v)?;

    // --- Q8 pipeline -----------------------------------------------------
    let scores_q8 = gqa_decode_gemv_qk_q8(&q, &cache, L_K)?;
    let scores_q8_scaled = scores_q8.affine(scale, 0.0)?;
    let p_q8 = softmax_last_dim(&scores_q8_scaled)?;
    let out_q8 = gqa_decode_gemv_v_q8(&p_q8, &cache, L_K)?;

    // --- F32-dequant reference with the SAME quantized K/V so the only
    //     source of difference is float rounding in the reduction.
    let k_dq = cache.dequantize_k()?.reshape((B, H_KV, L_K, DIM))?;
    let v_dq = cache.dequantize_v()?.reshape((B, H_KV, L_K, DIM))?;
    let out_ref = f32_reference(&q, &k_dq, &v_dq, scale)?;

    let r: Vec<f32> = out_ref.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let s: Vec<f32> = out_q8.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs = r
        .iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("pipeline(Q8) vs pipeline(F32-dequant) max_abs_err = {:.3e}", max_abs);

    // Also vs ORIGINAL F32 inputs — shows the full Q8-quantisation error.
    let out_ref_raw = f32_reference(&q, &k, &v, scale)?;
    let r_raw: Vec<f32> = out_ref_raw.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let max_abs_raw = r_raw
        .iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("pipeline(Q8) vs pipeline(F32 raw)      max_abs_err = {:.3e}", max_abs_raw);

    // Kernel math drift bound: Q·K partial reduction + V·attn partial
    // reduction, both float32 — expect O(D·L_k·ε) ≈ 1e-5 of output scale.
    let r_max = r.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let tol = r_max.max(1.0) * 5e-5;
    assert!(
        max_abs < tol,
        "pipeline kernel-math drift too large: {max_abs:.3e} > {tol:.3e}"
    );

    println!("OK");
    Ok(())
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?)
}

fn f32_reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    // Pure-tensor decode attention.
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
