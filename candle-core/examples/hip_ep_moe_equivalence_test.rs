//! X3.6 primitive validation: the sum of per-rank partial MoE outputs
//! computed via `narrow_experts + remapped topk_ids + masked weights`
//! equals the full-expert-set MoE output bit-for-bit.
//!
//! This isolates the EP math (X3.4 narrow + X3.5 remap/mask) from
//! full-model integration. If this passes, X3.6 model-level
//! validation is "plumb everything and bench" — the arithmetic
//! itself is already proven correct.

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Shape, Tensor, D};

const E: usize = 8;        // num_experts (must split cleanly across W)
const H: usize = 128;      // hidden dim (must be % 32 for Q8_0)
const I: usize = 256;      // intermediate dim
const B: usize = 3;        // batch size (decode-like)
const K: usize = 4;        // top-k
const W: usize = 4;        // number of virtual ranks

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;

    // Build a synthetic [E, I, H] gate_exps tensor (matmul contract:
    // expert weights [I, H] produce [B, I] from input [B, H]).
    let gate_data: Vec<f32> = (0..E * I * H)
        .map(|i| ((i as i64 * 29 + 17) % 239) as f32 / 119.0 - 1.0)
        .collect();
    let gate_f32 = Tensor::new(gate_data.as_slice(), &dev)?.reshape((E, I, H))?;
    let gate_q = QTensor::quantize(&gate_f32, GgmlDType::Q8_0)?;
    let gate_q = std::sync::Arc::new(gate_q);

    // Synthetic input x: [B, H] → reshape as the indexed_moe_forward_preq8
    // expects `in_shape = [b_size, 1, hidden]` (the "1" = topk_or_1, since
    // x is shared across top-k experts for the preq8 variant).
    let x_data: Vec<f32> = (0..B * H)
        .map(|i| ((i as i64 * 13 + 7) % 199) as f32 / 99.0 - 1.0)
        .collect();
    let x = Tensor::new(x_data.as_slice(), &dev)?.reshape((B, H))?;

    // Quantize x to Q8_1 once.
    use candle_core::quantized::hip::{pad, quantize_q8_1, MATRIX_ROW_PADDING};
    let hip_dev = match &dev {
        Device::Hip(d) => d.clone(),
        _ => unreachable!(),
    };
    let x_contig = x.contiguous()?;
    let (x_st, x_l) = x_contig.storage_and_layout();
    let x_hip = match &*x_st {
        candle_core::Storage::Hip(s) => s,
        _ => unreachable!(),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let (lo, hi) = x_l.contiguous_offsets().unwrap();
    let x_view = x_slice.slice(lo..hi);
    let ncols_padded = pad(H, MATRIX_ROW_PADDING);
    let q8_bytes =
        B * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8 = unsafe { hip_dev.alloc::<u8>(q8_bytes)? };
    quantize_q8_1(&x_view, &mut y_q8, H, B, &hip_dev)?;
    let q8_view = y_q8.slice(0..y_q8.len());
    drop(x_st);
    let in_shape = Shape::from((B, 1usize, H));

    // Random topk_ids in [0, E) and matching topk_weights (B × K each).
    let mut ids_cpu: Vec<u32> = Vec::with_capacity(B * K);
    for b in 0..B {
        // Force each row to contain DISTINCT expert IDs (no duplicates)
        // so the full-expert-set MoE computation covers exactly K slots.
        let mut remaining: Vec<u32> = (0..E as u32).collect();
        for k in 0..K {
            let idx = ((b * 11 + k * 7 + 3) % remaining.len() as usize) as usize;
            ids_cpu.push(remaining.remove(idx));
        }
    }
    let topk_ids_global = Tensor::from_vec(ids_cpu.clone(), (B, K), &dev)?;
    let weights_cpu: Vec<f32> = (0..B * K)
        .map(|i| ((i as i64 * 7 + 5) % 31) as f32 / 31.0 + 0.1)
        .collect();
    let topk_weights = Tensor::from_vec(weights_cpu.clone(), (B, K), &dev)?;
    // Normalise per row so weights sum to 1 (matches real routing).
    let topk_weights = (&topk_weights
        / topk_weights.sum_keepdim(D::Minus1)?.broadcast_as((B, K))?)?;

    // === FULL-EXPERT-SET FORWARD ===
    let gate_full_out = gate_q.indexed_moe_forward_preq8(&q8_view, &in_shape, &topk_ids_global)?;
    // Sum over top-k with weights: gate_full_out shape = [B, K, I]
    let w_full = topk_weights.unsqueeze(D::Minus1)?;
    let full_out = gate_full_out.broadcast_mul(&w_full)?.sum(1)?; // [B, I]
    let full_vec: Vec<f32> = full_out.flatten_all()?.to_device(&Device::Cpu)?.to_vec1()?;
    println!(
        "full_out: shape={:?}  sum={:.4}",
        full_out.shape().dims(),
        full_vec.iter().sum::<f32>()
    );

    // === PER-RANK PARTIAL FORWARDS ===
    let block = (E + W - 1) / W;
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(W);
    for rank in 0..W {
        let local_start = rank * block;
        let local_count = (E - local_start).min(block);
        if local_count == 0 {
            partials.push(vec![0.0; B * I]);
            continue;
        }
        let local_exps = gate_q.narrow_experts(local_start, local_count)?;

        // Remap topk_ids to local range; mask out-of-range weights.
        let mut local_ids = Vec::with_capacity(B * K);
        let mut mask = Vec::with_capacity(B * K);
        for id in &ids_cpu {
            let in_range = *id as usize >= local_start
                && (*id as usize) < local_start + local_count;
            if in_range {
                local_ids.push(id - local_start as u32);
                mask.push(1.0f32);
            } else {
                local_ids.push(0);
                mask.push(0.0f32);
            }
        }
        let local_ids_t = Tensor::from_vec(local_ids, (B, K), &dev)?;
        let mask_t = Tensor::from_vec(mask, (B, K), &dev)?;
        let masked_weights = topk_weights.mul(&mask_t)?;

        let gate_local_out =
            local_exps.indexed_moe_forward_preq8(&q8_view, &in_shape, &local_ids_t)?;
        let w_local = masked_weights.unsqueeze(D::Minus1)?;
        let partial = gate_local_out.broadcast_mul(&w_local)?.sum(1)?;
        let partial_vec: Vec<f32> = partial
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        println!(
            "rank {}: experts [{}, {}) partial sum = {:.4}",
            rank,
            local_start,
            local_start + local_count,
            partial_vec.iter().sum::<f32>()
        );
        partials.push(partial_vec);
    }

    // Sum partials element-wise.
    let mut summed = vec![0.0f32; B * I];
    for partial in &partials {
        for (i, v) in partial.iter().enumerate() {
            summed[i] += v;
        }
    }

    // Compare to full_out.
    let max_abs = full_vec
        .iter()
        .zip(summed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = {
        let n = full_vec.len() as f32;
        let ss: f32 = full_vec
            .iter()
            .zip(summed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (ss / n).sqrt()
    };
    println!(
        "\n|full - sum_ranks|:  max_abs = {:.3e},  rms = {:.3e}",
        max_abs, rms
    );
    // Q8_0 quantisation error is ~1e-3; full path and per-rank path go through
    // the SAME Q8_0 bytes, so max_abs should be at float-rounding level.
    assert!(
        max_abs < 1e-5,
        "EP math drift: full vs sum_ranks max_abs = {max_abs:.3e}"
    );

    println!("\nOK");
    Ok(())
}
