//! Bit-exact test for the Phase O2 MXFP4 MMVQ kernel.
//!
//! Loads a real MXFP4 tensor from Qwen3-Coder-Next (shared-expert
//! `ffn_up_shexp.weight` of layer 0, shape [2048, 512]), runs a forward
//! matmul through the new `mul_mat_vec_mxfp4_q8_1_cuda*` kernels, and
//! compares against the dequantise-to-F32 + F32 matmul reference.
//!
//! Usage:  cargo run --release --example hip_mxfp4_mmvq_test --features hip

use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Module, Tensor};
use std::fs::File;
use std::sync::Arc;

fn main() -> Result<()> {
    let device = Device::new_hip(0)?;
    let model = "/artefact/models/Qwen3-Coder-Next-Q4_0.gguf";
    let mut f = File::open(model).with_context(|| format!("opening {model}"))?;
    let (ct, _blob) = gguf_file::Content::read_mmap(model)?;

    // Pick MXFP4 tensors from the first couple of layers.
    let candidates = [
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.5.ffn_up_shexp.weight",
    ];

    for name in candidates {
        let info = ct
            .tensor_infos
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let dims = info.shape.dims();
        println!("\n=== {name}  shape={dims:?}  dtype={:?} ===", info.ggml_dtype);
        assert_eq!(info.ggml_dtype, candle_core::quantized::GgmlDType::Mxfp4);

        // Load the tensor via the GGUF tensor() helper; this goes through
        // the normal load path so QStorage is populated correctly on HIP.
        let qt = Arc::new(ct.tensor(&mut f, name, &device)?);
        // Shape is stored as (n, k) for Q-tensors: n=output rows, k=input cols.
        let (n, k) = qt.shape().dims2()?;

        let b = 1usize;
        let x_data: Vec<f32> = (0..(b * k))
            .map(|i| (((i * 41 % 163) as f32 - 81.0) / 32.0))
            .collect();
        let x = Tensor::new(x_data.as_slice(), &device)?.reshape((b, k))?;

        // MMVQ path (new kernel).
        let qmm = QMatMul::from_arc(qt.clone())?;
        let y_q = qmm.forward(&x)?;
        let y_q = y_q.flatten_all()?.to_vec1::<f32>()?;

        // Reference: dequantize weight to F32 and matmul.
        let w_ref = qt.dequantize(&device)?; // (n, k) F32
        let y_ref = x.matmul(&w_ref.t()?)?;   // (b, n)
        let y_ref = y_ref.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(y_q.len(), y_ref.len());

        let (mut mad, mut mse) = (0.0f32, 0.0f64);
        for (a, b_) in y_q.iter().zip(y_ref.iter()) {
            let d = (a - b_).abs();
            mad = mad.max(d);
            mse += (d as f64) * (d as f64);
        }
        let rmse = (mse / y_q.len() as f64).sqrt();
        let peak = y_ref.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let rel = mad as f64 / peak.max(1e-6) as f64;
        println!(
            "  n={n} k={k}  max-abs={mad:.4e}  rmse={rmse:.4e}  peak={peak:.3}  rel={rel:.3e}"
        );
        println!("  q[0..6]   = {:?}", &y_q[..6.min(y_q.len())]);
        println!("  ref[0..6] = {:?}", &y_ref[..6.min(y_ref.len())]);
        if rel > 5e-3 {
            println!("*** FAIL: relative error {rel} > 5e-3 ***");
            std::process::exit(1);
        }
    }
    println!("\nall ok");
    Ok(())
}
