//! Simulated decode loop with the Q8 KV cache:
//!   - Seed the cache with a prefill block (L_prefill tokens).
//!   - Run N decode steps, each appending a single (K, V) row and
//!     running Q8 Q·K → softmax → Q8 V·attn.
//!   - Compare the decode outputs against an F32 reference that
//!     uses the DEQUANTISED cache (isolates kernel math from Q8 noise).
//!
//! This is the template `StandardAttention::gqa_attention` will follow
//! when Q8 mode is on. Phase 5 (proper) still needs to swap the F32
//! KvCache for QKvCache inside the gemma4/llama model structs and
//! thread through the decode dispatch. That's ~10 sites + Rotating
//! cache support — deferred to a follow-up.

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::{
    gqa_decode_gemv_qk_q8, gqa_decode_gemv_v_q8, QKvCache,
};
use candle_core::{Device, Tensor, D};

const B: usize = 1;
const H_Q: usize = 8;
const H_KV: usize = 1;
const DIM: usize = 128;
const L_PREFILL: usize = 128;
const N_DECODE: usize = 16;
const T_MAX: usize = 512;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    println!("B={B}, H_q={H_Q}, H_kv={H_KV}, D={DIM}, prefill={L_PREFILL}, decode={N_DECODE}");
    let scale = 1.0 / (DIM as f64).sqrt();

    let mut cache = QKvCache::new(&hip_dev, B, H_KV, T_MAX, DIM)?;

    // --- Prefill: append L_PREFILL rows in one go.
    let pre_k = make_tensor(&dev, B * H_KV * L_PREFILL * DIM, 13, 0.02)?
        .reshape((B, H_KV, L_PREFILL, DIM))?;
    let pre_v = make_tensor(&dev, B * H_KV * L_PREFILL * DIM, 17, 0.03)?
        .reshape((B, H_KV, L_PREFILL, DIM))?;
    cache.append(&pre_k, &pre_v)?;
    assert_eq!(cache.current_seq_len(), L_PREFILL);

    let mut worst_err = 0.0f32;
    let mut worst_step = 0usize;
    for step in 0..N_DECODE {
        let seed = 100 + step as i32;
        let q = make_tensor(&dev, B * H_Q * DIM, seed, 0.05)?
            .reshape((B, H_Q, 1, DIM))?;
        let new_k = make_tensor(&dev, B * H_KV * DIM, seed + 7, 0.03)?
            .reshape((B, H_KV, 1, DIM))?;
        let new_v = make_tensor(&dev, B * H_KV * DIM, seed + 11, 0.04)?
            .reshape((B, H_KV, 1, DIM))?;
        cache.append(&new_k, &new_v)?;

        let l_k = cache.current_seq_len();

        // Q8 path
        let s_q8 = gqa_decode_gemv_qk_q8(&q, &cache, l_k)?;
        let p_q8 = softmax_last_dim(&s_q8.affine(scale, 0.0)?)?;
        let out_q8 = gqa_decode_gemv_v_q8(&p_q8, &cache, l_k)?;

        // Reference from DEQUANTISED Q8 cache (isolate kernel math).
        let k_dq = cache.dequantize_k()?.reshape((B, H_KV, l_k, DIM))?;
        let v_dq = cache.dequantize_v()?.reshape((B, H_KV, l_k, DIM))?;
        let out_ref = f32_reference(&q, &k_dq, &v_dq, scale)?;

        let r: Vec<f32> = out_ref.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
        let s: Vec<f32> = out_q8.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
        let err = r
            .iter()
            .zip(s.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if err > worst_err {
            worst_err = err;
            worst_step = step;
        }
    }

    println!(
        "worst Q8-vs-F32(dequant) over {} steps: max_abs_err = {:.3e} at step {}",
        N_DECODE, worst_err, worst_step
    );
    assert!(worst_err < 1e-4, "decode-loop kernel drift: {worst_err:.3e}");

    // VRAM comparison for the filled cache.
    let t_filled = cache.current_seq_len();
    let q8_bytes =
        candle_core::hip_backend::q8_kv_cache::q8_slab_bytes(B, H_KV, T_MAX, DIM) * 2;
    let f32_bytes_used = B * H_KV * t_filled * DIM * 4 * 2;
    let f16_bytes_used = B * H_KV * t_filled * DIM * 2 * 2;
    println!(
        "filled={} tok  VRAM: Q8_0 slab={} B (filled {}/{})  F32-per-tok={} B  F16-per-tok={} B",
        t_filled,
        q8_bytes,
        t_filled,
        T_MAX,
        f32_bytes_used,
        f16_bytes_used
    );

    Ok(())
}

fn make_tensor(dev: &Device, n: usize, seed: i32, step: f32) -> Result<Tensor> {
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as i64 + seed as i64 * 311) % 251) as f32 * step - 1.0)
        .collect();
    Ok(Tensor::new(data.as_slice(), dev)?)
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&m)?.exp()?;
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
