//! Round-trip test for the TurboQuant KV cache (TQ2/TQ3/TQ4).
//!
//! Input distribution is synthetic unit-norm Gaussian — matches what the
//! Lloyd-Max codebooks are tuned for. Real-model K/V need WHT + dense
//! rotation preconditioning (phase TQ3) to hit this distribution; this
//! test validates the bit-pack / unpack math only.

use anyhow::Result;
use candle_core::hip_backend::turbo_kv_cache::{QTurboKvCache, TbVariant};
use candle_core::{Device, Tensor};

const B: usize = 1;
const H_KV: usize = 4;
const T_MAX: usize = 512;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };

    // TQ2/TQ3 use 32-elem blocks; TQ4 uses 128-elem blocks.
    // Use D=128 so all three variants fit.
    const D: usize = 128;
    const T_NEW: usize = 4;

    // Gaussian-ish synthetic data, normalised per 32-elem group so each
    // block has unit norm (what the centroids expect).
    let n_elem = B * H_KV * T_NEW * D;
    let mut data: Vec<f32> = (0..n_elem)
        .map(|i| {
            // Approximate Gaussian via sum of 3 uniform samples in [-1, 1].
            let mut x = 0.0f32;
            for k in 0..3 {
                let seed = (i as u64 * 2654435761).wrapping_add(k as u64 * 1597334677);
                x += ((seed % 2000) as f32) / 1000.0 - 1.0;
            }
            x
        })
        .collect();

    // Normalise each 32-element slab so its L2 norm matches the expected
    // codebook domain (centroid range ~0.19 for TQ3, 0.13 for TQ2). The
    // kernels compute per-block norm at quantize time, so we just need
    // the relative distribution to look like a rotated Gaussian.
    for slab_start in (0..n_elem).step_by(32) {
        let slab = &mut data[slab_start..slab_start + 32];
        let sq: f32 = slab.iter().map(|x| x * x).sum();
        let norm = sq.sqrt().max(1e-6);
        for x in slab.iter_mut() {
            *x /= norm;
        }
    }

    let ref_f32 = Tensor::new(data.as_slice(), &dev)?.reshape((B, H_KV, T_NEW, D))?;

    for variant in [TbVariant::Tq2_0, TbVariant::Tq3_0, TbVariant::Tq4_0] {
        println!("\n=== {:?} ({} bits/val, {}× vs F16) ===",
                 variant,
                 variant.bits_per_val(),
                 16.0 / variant.bits_per_val());

        let mut cache = QTurboKvCache::new(&hip_dev, variant, B, H_KV, T_MAX, D)?;
        cache.append(&ref_f32, &ref_f32)?;
        assert_eq!(cache.current_seq_len(), T_NEW);

        let k_dq: Vec<f32> = cache.dequantize_k()?
            .flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
        let v_dq: Vec<f32> = cache.dequantize_v()?
            .flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;

        // Per-block norms of the DEQUANTIZED values — should roughly
        // match the input per-block norms (unit in our test case).
        let k_abs_err = max_abs_err(&data, &k_dq);
        let v_abs_err = max_abs_err(&data, &v_dq);
        let k_rms = rms_err(&data, &k_dq);
        let v_rms = rms_err(&data, &v_dq);

        println!("K: max_abs_err = {:.3e}   rms_err = {:.3e}", k_abs_err, k_rms);
        println!("V: max_abs_err = {:.3e}   rms_err = {:.3e}", v_abs_err, v_rms);

        // Without the WHT + dense rotation preconditioning, outliers
        // beyond the outermost centroid get clamped, so max_abs_err is
        // dominated by those outliers — not a kernel-correctness signal.
        // RMS error is the right metric: ~half the centroid gap for
        // well-distributed inputs. TQ2 gap ≈ 0.09, TQ3 gap ≈ 0.04,
        // TQ4 with residual ≈ 0.05 (residual step helps but unrotated
        // input defeats much of it).
        let rms_bound = match variant {
            TbVariant::Tq2_0 => 0.12,
            TbVariant::Tq3_0 => 0.07,
            TbVariant::Tq4_0 => 0.12,
        };
        assert!(k_rms < rms_bound, "K {:?} rms_err {:.3e} > {:.3e}", variant, k_rms, rms_bound);
        assert!(v_rms < rms_bound, "V {:?} rms_err {:.3e} > {:.3e}", variant, v_rms, rms_bound);
    }

    // VRAM comparison vs F32 / F16 / Q8.
    println!("\n=== VRAM footprint for (B={B}, H_kv={H_KV}, T={T_MAX}, D={D}) ===");
    let f32_bytes = B * H_KV * T_MAX * D * 4;
    let f16_bytes = B * H_KV * T_MAX * D * 2;
    let q8_bytes = B * H_KV * T_MAX * (D / 32) * 34;
    println!("F32       : {} B", f32_bytes);
    println!("F16       : {} B ({:.2}× vs F32)", f16_bytes, f32_bytes as f32 / f16_bytes as f32);
    println!("Q8_0      : {} B ({:.2}× vs F32)", q8_bytes, f32_bytes as f32 / q8_bytes as f32);
    for v in [TbVariant::Tq2_0, TbVariant::Tq3_0, TbVariant::Tq4_0] {
        let tb_bytes = candle_core::hip_backend::turbo_kv_cache::tb_slab_bytes(v, B, H_KV, T_MAX, D);
        println!("{:?}: {} B ({:.2}× vs F32, {:.2}× vs F16)",
                 v, tb_bytes,
                 f32_bytes as f32 / tb_bytes as f32,
                 f16_bytes as f32 / tb_bytes as f32);
    }

    println!("\nOK");
    Ok(())
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn rms_err(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ss: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (ss / n).sqrt()
}
