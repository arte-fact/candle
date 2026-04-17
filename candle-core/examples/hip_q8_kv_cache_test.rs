//! Round-trip test for the Q8_0 KV cache:
//!   F32 tensor → QKvCache::append → dequantize → compare vs original.
//!
//! Validates packing/unpacking correctness; expected rel err ≤ 1e-2
//! (Q8_0 worst-case per-block amax/127 step).

use anyhow::Result;
use candle_core::hip_backend::q8_kv_cache::QKvCache;
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_KV: usize = 8;
const T_MAX: usize = 256;
const D: usize = 128;
const T_NEW: usize = 4;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };

    println!("B={B}, H_kv={H_KV}, T_max={T_MAX}, D={D}, T_new={T_NEW}");

    let n_elem_new = B * H_KV * T_NEW * D;
    let ref_data: Vec<f32> = (0..n_elem_new)
        .map(|i| ((i as i64 * 37 + 11) % 251) as f32 / 125.0 - 1.0)
        .collect();
    let k = Tensor::new(ref_data.as_slice(), &dev)?.reshape((B, H_KV, T_NEW, D))?;
    let v = k.affine(0.5, 0.25)?; // distinct payload for V

    let mut cache = QKvCache::new(&hip_dev, B, H_KV, T_MAX, D)?;

    // Two appends to exercise t_offset math.
    cache.append(&k, &v)?;
    cache.append(&k, &v)?;
    assert_eq!(cache.current_seq_len(), 2 * T_NEW);

    let k_dq = cache.dequantize_k()?.to_device(&Device::Cpu)?;
    let v_dq = cache.dequantize_v()?.to_device(&Device::Cpu)?;
    let k_dq: Vec<f32> = k_dq.flatten_all()?.to_vec1()?;
    let v_dq: Vec<f32> = v_dq.flatten_all()?.to_vec1()?;

    // Reference mirrors the cache's (B, H, T_SEL=2*T_NEW, D) layout:
    // for each (b, h) pair, the T_NEW rows of the first append are
    // concatenated with the T_NEW rows of the second (identical) append.
    // Plain `ref_data.chain(ref_data)` would be (H, T, D) || (H, T, D)
    // — wrong axis order.
    let mut k_ref: Vec<f32> = Vec::with_capacity(B * H_KV * 2 * T_NEW * D);
    for b in 0..B {
        for h in 0..H_KV {
            let base = ((b * H_KV + h) * T_NEW) * D;
            let chunk = &ref_data[base..base + T_NEW * D];
            k_ref.extend_from_slice(chunk);
            k_ref.extend_from_slice(chunk);
        }
    }
    let v_ref: Vec<f32> = k_ref.iter().map(|x| 0.5 * x + 0.25).collect();

    // Q8_0 worst case per block: step = amax/127. Max abs err is
    // bounded by (amax_block / 254) — call the block amax `A`, then
    // per-element abs err ≤ A/254. Absolute metric is the clean
    // correctness proxy; rel-err explodes on near-zero values.
    let k_abs = max_abs_err(&k_ref, &k_dq);
    let v_abs = max_abs_err(&v_ref, &v_dq);
    // Max abs value in the reference acts as the worst-case amax.
    let k_max = k_ref.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let v_max = v_ref.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let k_bound = k_max / 254.0;
    let v_bound = v_max / 254.0;

    println!("K: max_abs_err = {:.3e}   Q8 bound (k_max/254) = {:.3e}", k_abs, k_bound);
    println!("V: max_abs_err = {:.3e}   Q8 bound (v_max/254) = {:.3e}", v_abs, v_bound);
    // 2× slack over the theoretical bound covers rounding.
    assert!(k_abs < 2.0 * k_bound, "K abs err {k_abs:.3e} > 2× bound {k_bound:.3e}");
    assert!(v_abs < 2.0 * v_bound, "V abs err {v_abs:.3e} > 2× bound {v_bound:.3e}");

    // VRAM sanity check.
    let bytes = candle_core::hip_backend::q8_kv_cache::q8_slab_bytes(B, H_KV, T_MAX, D);
    let f32_bytes = B * H_KV * T_MAX * D * 4;
    let f16_bytes = B * H_KV * T_MAX * D * 2;
    println!(
        "VRAM: Q8_0={} B, F16={} B ({:.2}×), F32={} B ({:.2}×)",
        bytes,
        f16_bytes,
        f16_bytes as f32 / bytes as f32,
        f32_bytes,
        f32_bytes as f32 / bytes as f32,
    );

    println!("OK");
    Ok(())
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

