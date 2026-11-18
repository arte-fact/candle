//! X8.3 structural test: `DenseMlp::load_tp` on gemma4-26B-A4B.
//!
//! Loads `blk.0` dense FFN (gate/up/down) in two forms:
//!   1. Full reference via `load_with_activation`.
//!   2. Per-rank TP shards (W=2) via `load_tp`.
//!
//! Asserts:
//!   - Per-rank intermediate_size = full / W.
//!   - Σ_rank storage bytes ≈ full storage bytes (within padding slack).

use anyhow::{Context, Result};
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{Device, HipDevice};
use candle_transformers::models::quantized_blocks::{
    DenseMlp, Gguf, MlpActivation,
};

fn main() -> Result<()> {
    let path = std::env::var("CANDLE_X8_GGUF")
        .unwrap_or_else(|_| "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf".into());
    println!("GGUF: {path}");

    let mut devs = Vec::<HipDevice>::new();
    for ord in 0..16 {
        if let Ok(Device::Hip(d)) = Device::new_hip(ord) {
            devs.push(d);
        } else {
            break;
        }
    }
    let world = devs.len().min(2);
    if world < 2 {
        println!("Need at least 2 HIP devices; found {world}");
        return Ok(());
    }
    devs.truncate(world);
    println!("World size = {world}");

    let cluster = std::sync::Arc::new(HipCluster::new(devs.clone())?);

    let dev_ref = Device::Hip(devs[0].clone());
    let (ct_ref, blob_ref) = Content::read_mmap(&path)?;
    let prefix = "blk.0";
    let gg_ref = Gguf::new(ct_ref, blob_ref, dev_ref.clone());
    let ref_mlp = DenseMlp::load_with_activation(&gg_ref, prefix, MlpActivation::Gelu)
        .with_context(|| "reference DenseMlp::load_with_activation")?;
    let ref_bytes = ref_mlp.q_weight_bytes_total();
    println!(
        "reference  intermediate={} weight_bytes={}",
        ref_mlp.intermediate_size(),
        ref_bytes
    );

    let mut sum_bytes = 0usize;
    for rank in 0..world {
        let dev = Device::Hip(devs[rank].clone());
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, dev.clone());
        let mlp = DenseMlp::load_tp(&gg, prefix, MlpActivation::Gelu, &cluster, rank)
            .with_context(|| format!("load_tp rank={rank}"))?;
        let r_bytes = mlp.q_weight_bytes_total();
        println!(
            "rank {rank}   intermediate={} weight_bytes={}",
            mlp.intermediate_size(),
            r_bytes
        );
        assert_eq!(
            mlp.intermediate_size(),
            ref_mlp.intermediate_size() / world,
            "rank {rank} intermediate"
        );
        sum_bytes += r_bytes;
    }

    // 2 matmuls × padding slack (< 1 block per QStorage's MATRIX_ROW_PADDING
    // tail) → up to ~4 KB slack per rank per weight.
    let slack = world * 2 * 4096;
    let diff = (sum_bytes as i64 - ref_bytes as i64).abs() as usize;
    println!(
        "Σ_rank = {sum_bytes}   reference = {ref_bytes}   diff = {diff}   slack ≤ {slack}"
    );
    assert!(
        diff <= slack,
        "TP per-rank byte sum diverged: diff={diff} > slack={slack}"
    );

    println!("\nOK");
    Ok(())
}
