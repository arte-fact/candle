//! X5 correctness: `Content::tensor_{expert,row,column}_range_from_blob`
//! round-trip against "load full → narrow on device".
//!
//! Opens a real GGUF (Qwen3-Coder-30B Q4_0) and, for one MoE expert
//! tensor and one 2D attention weight, verifies that reading only a
//! byte-range from the mmap'd blob produces a QTensor whose dequantised
//! values equal the corresponding slice of the full dequantised tensor.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, Tensor};

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    let av: Vec<f32> = a.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    let bv: Vec<f32> = b.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(av.len(), bv.len(), "elem count mismatch");
    Ok(av
        .iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max))
}

fn main() -> Result<()> {
    let path = std::env::var("CANDLE_X5_GGUF")
        .unwrap_or_else(|_| "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf".into());
    println!("GGUF: {path}");

    let dev = Device::new_hip(0)?;
    let (content, blob) = Content::read_mmap(&path)?;

    // --- 1. MoE expert-axis slice.
    let moe_name = "blk.0.ffn_gate_exps.weight";
    let info = content
        .tensor_infos
        .get(moe_name)
        .with_context(|| format!("missing {moe_name}"))?;
    let dims = info.shape.dims().to_vec();
    println!("\n[expert-range] {moe_name} shape={:?} dtype={:?}", dims, info.ggml_dtype);
    assert_eq!(dims.len(), 3, "expected 3D MoE tensor");
    let (e_total, d1, d2) = (dims[0], dims[1], dims[2]);
    let e_start = 32;
    let e_count = 16;
    assert!(e_start + e_count <= e_total);

    let full_q = content.tensor_from_blob(&blob, moe_name, &dev)?;
    let sliced_q = content.tensor_expert_range_from_blob(&blob, moe_name, e_start, e_count, &dev)?;
    assert_eq!(sliced_q.shape().dims(), &[e_count, d1, d2]);
    println!(
        "  full  bytes={}  sliced bytes={}  ratio={:.3}",
        full_q.storage_size_in_bytes(),
        sliced_q.storage_size_in_bytes(),
        sliced_q.storage_size_in_bytes() as f64 / full_q.storage_size_in_bytes() as f64,
    );

    let full_dq = full_q.dequantize(&dev)?;
    let sliced_dq = sliced_q.dequantize(&dev)?;
    let want = full_dq.narrow(0, e_start, e_count)?;
    let err = max_abs(&want, &sliced_dq)?;
    println!("  max_abs(slice vs full.narrow(0,..)) = {err:.3e}");
    assert!(err < 1e-6, "expert-range diverged: max_abs={err:.3e}");

    // --- 2. 2D row-axis slice on attn_q.
    let name_2d = "blk.0.attn_q.weight";
    let info = content
        .tensor_infos
        .get(name_2d)
        .with_context(|| format!("missing {name_2d}"))?;
    let dims2 = info.shape.dims().to_vec();
    println!("\n[row-range] {name_2d} shape={:?} dtype={:?}", dims2, info.ggml_dtype);
    assert_eq!(dims2.len(), 2);
    let (rows, cols) = (dims2[0], dims2[1]);
    let r_start = rows / 4;
    let r_count = rows / 4;

    let full_2d = content.tensor_from_blob(&blob, name_2d, &dev)?;
    let row_slice = content.tensor_row_range_from_blob(&blob, name_2d, r_start, r_count, &dev)?;
    assert_eq!(row_slice.shape().dims(), &[r_count, cols]);
    println!(
        "  full  bytes={}  sliced bytes={}",
        full_2d.storage_size_in_bytes(),
        row_slice.storage_size_in_bytes(),
    );
    let err = max_abs(
        &full_2d.dequantize(&dev)?.narrow(0, r_start, r_count)?,
        &row_slice.dequantize(&dev)?,
    )?;
    println!("  max_abs = {err:.3e}");
    assert!(err < 1e-6, "row-range diverged: max_abs={err:.3e}");

    // --- 3. 2D column-axis slice on the same tensor.
    // Requires col_start and col_count to be block-aligned; Q4_0 block
    // size is 32 elements, and `cols` is already a multiple of 32 for
    // any real LLM weight.
    let c_start = cols / 4;
    let c_count = cols / 4;
    let col_slice =
        content.tensor_column_range_from_blob(&blob, name_2d, c_start, c_count, &dev)?;
    println!("\n[col-range] {name_2d}  col_start={c_start} col_count={c_count}");
    assert_eq!(col_slice.shape().dims(), &[rows, c_count]);
    let err = max_abs(
        &full_2d.dequantize(&dev)?.narrow(1, c_start, c_count)?,
        &col_slice.dequantize(&dev)?,
    )?;
    println!("  max_abs = {err:.3e}");
    assert!(err < 1e-6, "column-range diverged: max_abs={err:.3e}");

    // --- 4. Boundary cases: full-range + empty + out-of-range.
    let full_via_range = content.tensor_expert_range_from_blob(&blob, moe_name, 0, e_total, &dev)?;
    assert_eq!(
        full_via_range.storage_size_in_bytes(),
        full_q.storage_size_in_bytes()
    );
    let err = max_abs(&full_via_range.dequantize(&dev)?, &full_dq)?;
    assert!(err < 1e-6, "full-via-range diverged: {err:.3e}");

    let empty = content.tensor_expert_range_from_blob(&blob, moe_name, e_total, 0, &dev)?;
    assert_eq!(empty.shape().dims(), &[0, d1, d2]);
    assert_eq!(empty.storage_size_in_bytes(), 0);

    let bad = content.tensor_expert_range_from_blob(&blob, moe_name, e_total - 1, 5, &dev);
    assert!(bad.is_err(), "out-of-range expected to error");

    println!("\nOK");
    Ok(())
}
