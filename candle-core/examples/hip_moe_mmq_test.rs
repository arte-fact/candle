//! Bit-exact test for the Q8_0 MoE MMQ turbo port (M6) + W3 per-expert
//! gather.
//!
//! Two harnesses:
//!   - Prefill M6 harness: `CANDLE_MMQ_TURBO_MOE=0` (fused) vs `=1`
//!     (turbo gather MMQ). Existing shapes tokens∈{32, 128, 512}.
//!   - Decode W3 harness: `CANDLE_W3_GATHER=0` (fused) vs `=1`
//!     (per-expert warp-coop MMVQ loop). Shapes tokens=1, topk=8,
//!     n_experts=128.
//!
//! W3 uses different summation trees than the fused kernel
//! (warp-coop uses 1 wave/2 rows; fused uses nwarps=4 per row), so the
//! expected tolerance is rel-err ≤ 1e-3 — not bit-exact.
//!
//! Usage:  cargo run --release --example hip_moe_mmq_test --features hip
use anyhow::Result;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

fn run_once_with_env(
    env_key: &str,
    env_val: &str,
    dtype: GgmlDType,
    tokens: usize,
    topk: usize,
    n_experts: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    // SAFETY: single-threaded main.
    unsafe {
        std::env::set_var("CANDLE_MMQ_TURBO_PORT", "1");
        std::env::set_var(env_key, env_val);
    }

    let device = Device::new_hip(0)?;

    let weight_data: Vec<f32> = (0..(n_experts * n * k))
        .map(|i| (((i * 31 + 7) % 251) as f32 - 125.0) / 200.0)
        .collect();
    let w = Tensor::new(weight_data.as_slice(), &device)?.reshape((n_experts, n, k))?;
    let qw = QTensor::quantize(&w, dtype)?;

    let input_data: Vec<f32> = (0..(tokens * k))
        .map(|i| (((i * 41) % 199) as f32 - 99.0) / 150.0)
        .collect();
    let x = Tensor::new(input_data.as_slice(), &device)?.reshape((tokens, 1, k))?;

    let ids_data: Vec<u32> = (0..(tokens * topk))
        .map(|i| {
            let tok = i / topk;
            let slot = i % topk;
            ((tok * 101 + slot * 13 + 7) as u32) % (n_experts as u32)
        })
        .collect();
    let ids = Tensor::new(ids_data.as_slice(), &device)?.reshape((tokens, topk))?;

    // indexed_moe_forward returns [tokens, topk, n]
    let y = qw.indexed_moe_forward(&x, &ids)?;
    let flat: Vec<f32> = y.flatten_all()?.to_vec1::<f32>()?;
    let x_sample: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;
    let ids_out: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?;
    Ok((flat, x_sample, ids_out))
}

fn run_case(
    label: &str,
    env_key: &str,
    dtype: GgmlDType,
    tokens: usize,
    topk: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    fail_mad: f32,
    fail_mismatch_frac: f32,
) -> Result<bool> {
    println!("\n=== {label} — dtype={dtype:?} tokens={tokens} topk={topk} n_experts={n_experts} n={n} k={k} ===");
    // Disable the OTHER toggle so we isolate just the flag under test.
    let other_key = if env_key == "CANDLE_MMQ_TURBO_MOE" { "CANDLE_W3_GATHER" } else { "CANDLE_MMQ_TURBO_MOE" };
    // SAFETY: single-threaded main.
    unsafe { std::env::set_var(other_key, "0"); }
    let (out_ref, x_ref, ids_ref) = run_once_with_env(env_key, "0", dtype, tokens, topk, n_experts, n, k)?;
    let (out_new, x_new, ids_new) = run_once_with_env(env_key, "1", dtype, tokens, topk, n_experts, n, k)?;
    assert_eq!(out_ref.len(), out_new.len(), "output length mismatch");

    let xs_diff = x_ref
        .iter()
        .zip(x_new.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ids_equal = ids_ref == ids_new;
    println!("input max-abs diff={xs_diff:.3e}  ids_equal={ids_equal}");
    assert!(xs_diff < 1e-6, "inputs diverged between runs");

    let (mut mad, mut mse, mut n_mismatch) = (0.0f32, 0.0f64, 0usize);
    for (a, b) in out_ref.iter().zip(out_new.iter()) {
        let d = (a - b).abs();
        mad = mad.max(d);
        mse += (d as f64) * (d as f64);
        let rel_tol = 1e-3_f32.max(a.abs() * 1e-3);
        if d > rel_tol {
            n_mismatch += 1;
        }
    }
    let rmse = (mse / out_ref.len() as f64).sqrt();
    let m0 = out_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let m1 = out_new.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "max-abs-diff={mad:.4e}  rmse={rmse:.4e}  n_mismatch={n_mismatch}/{}",
        out_ref.len()
    );
    println!("max(ref)={m0:.3}  max(new)={m1:.3}");
    println!("ref[0..6]  = {:?}", &out_ref[..6.min(out_ref.len())]);
    println!("new[0..6]  = {:?}", &out_new[..6.min(out_new.len())]);
    let mismatch_limit = ((out_ref.len() as f32) * fail_mismatch_frac) as usize;
    if mad > fail_mad || n_mismatch > mismatch_limit {
        println!("*** FAIL: mad={mad} (>{fail_mad}) or n_mismatch={n_mismatch} (>{mismatch_limit}) ***");
        return Ok(true);
    }
    Ok(false)
}

fn main() -> Result<()> {
    // ---- M6 (turbo MMQ gather) — prefill shapes ----
    // (label, dtype, tokens, topk, n_experts, n, k)
    let m6_shapes = [
        ("Q8_0 tiny (tok=32 n=64 k=128)",   GgmlDType::Q8_0, 32, 4, 8,   64,  128),
        ("Q4_0 tiny (tok=32 n=64 k=128)",   GgmlDType::Q4_0, 32, 4, 8,   64,  128),
        ("Q4_1 tiny (tok=32 n=64 k=128)",   GgmlDType::Q4_1, 32, 4, 8,   64,  128),
        // gemma-4-26B-A4B shapes (Q8_0, n_experts=128, topk=8).
        ("Q8_0 gemma4-26B gate_up",         GgmlDType::Q8_0, 512, 8, 128, 4864, 2560),
        ("Q8_0 gemma4-26B down",            GgmlDType::Q8_0, 512, 8, 128, 2560, 2432),
        // Qwen3-Coder-30B-A3B shapes (Q4_0, n_experts=128, topk=8).
        ("Q4_0 qwen-coder-30B gate_up",     GgmlDType::Q4_0, 512, 8, 128, 1536, 5120),
        ("Q4_0 qwen-coder-30B down",        GgmlDType::Q4_0, 512, 8, 128, 5120, 768),
        // Q4_1 stress (non-aligned K).
        ("Q4_1 stress m128 k2560 n1024",    GgmlDType::Q4_1, 128, 4, 32,  1024, 2560),
    ];

    // ---- W3 (per-expert MMVQ gather) — decode shapes (tokens=1) ----
    let w3_shapes = [
        // gemma-4-26B-A4B decode: one token, topk=8, 128 experts.
        ("W3 Q8_0 gemma4-26B decode gate_up", GgmlDType::Q8_0, 1, 8, 128, 4864, 2560),
        ("W3 Q8_0 gemma4-26B decode down",    GgmlDType::Q8_0, 1, 8, 128, 2560, 2432),
        // Qwen3-Coder-30B-A3B decode: topk=8, 128 experts.
        ("W3 Q4_0 qwen-coder-30B decode gate_up", GgmlDType::Q4_0, 1, 8, 128, 1536, 5120),
        ("W3 Q4_0 qwen-coder-30B decode down",    GgmlDType::Q4_0, 1, 8, 128, 5120, 768),
        // Tiny sanity — catches expert-stride off-by-ones fast.
        ("W3 Q8_0 tiny decode (tok=1 n=64 k=128)", GgmlDType::Q8_0, 1, 4, 8, 64, 128),
        ("W3 Q4_0 tiny decode (tok=1 n=64 k=128)", GgmlDType::Q4_0, 1, 4, 8, 64, 128),
    ];

    let mut any_fail = false;

    println!("\n######## M6: CANDLE_MMQ_TURBO_MOE=0 vs =1 (prefill gather MMQ) ########");
    for (label, dtype, tokens, topk, n_experts, n, k) in m6_shapes {
        any_fail |= run_case(
            label, "CANDLE_MMQ_TURBO_MOE", dtype, tokens, topk, n_experts, n, k,
            /*fail_mad=*/ 1e-1,
            /*fail_mismatch_frac=*/ 0.05,
        )?;
    }

    println!("\n######## W3: CANDLE_W3_GATHER=0 vs =1 (decode per-expert MMVQ) ########");
    for (label, dtype, tokens, topk, n_experts, n, k) in w3_shapes {
        any_fail |= run_case(
            label, "CANDLE_W3_GATHER", dtype, tokens, topk, n_experts, n, k,
            /*fail_mad=*/ 1e-2,  // tighter; only sum-order drift expected
            /*fail_mismatch_frac=*/ 0.01,
        )?;
    }

    if any_fail {
        std::process::exit(1);
    }
    Ok(())
}
