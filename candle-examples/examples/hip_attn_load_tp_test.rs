//! X8.1 structural test: `StandardAttention::load_tp` on gemma4-26B-A4B.
//!
//! Loads layer `blk.0` attention in two forms:
//!   1. Full reference (non-TP) via `load_with_opts`.
//!   2. Per-rank TP shards (W=2) via `load_tp`.
//!
//! Asserts:
//!   - Each rank's n_head = full / W, n_kv_head = full / W.
//!   - head_dim unchanged.
//!   - `Σ_rank q_weight_bytes_total(rank)` ≈ `q_weight_bytes_total(full)`
//!     (within one block's padding slack per weight × 4 weights).

use anyhow::{Context, Result};
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{DType, Device, HipDevice};
use candle_transformers::models::quantized_blocks::{
    GgufConfig, Gguf, RotaryEmbedding, StandardAttention, StandardAttentionOpts,
};

fn build_rotary(dev: &Device) -> Result<std::sync::Arc<RotaryEmbedding>> {
    // Stub RoPE — the structural test never exercises it.
    Ok(std::sync::Arc::new(RotaryEmbedding::new(
        10_000.0, 64, 128, DType::F32, dev,
    )?))
}

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

    // --- Reference full attention on rank 0 ---
    let dev_ref = Device::Hip(devs[0].clone());
    let (ct_ref, blob_ref) = Content::read_mmap(&path)?;
    let cfg = GgufConfig::from_metadata(&ct_ref.metadata)?;
    let prefix = "blk.0";
    let gg_ref = Gguf::new(ct_ref, blob_ref, dev_ref.clone());
    let opts = StandardAttentionOpts {
        use_v_norm: true,
        use_gemma_norms: true,
        attention_scale: Some(1.0),
    };
    let ref_attn = StandardAttention::load_with_opts(
        &gg_ref, prefix, &cfg, 0, build_rotary(&dev_ref)?, opts,
    )?;
    let ref_bytes = ref_attn.q_weight_bytes_total();
    println!(
        "reference  n_head={}  n_kv_head={}  head_dim={}  weight_bytes={}",
        ref_attn.n_head, ref_attn.n_kv_head, ref_attn.head_dim, ref_bytes
    );

    // --- Per-rank TP attention ---
    let mut sum_bytes = 0usize;
    for rank in 0..world {
        let dev = Device::Hip(devs[rank].clone());
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, dev.clone());
        let attn = StandardAttention::load_tp(
            &gg, prefix, &cfg, 0,
            build_rotary(&dev)?,
            opts,
            &cluster, rank,
        )
        .with_context(|| format!("load_tp rank={rank}"))?;
        let r_bytes = attn.q_weight_bytes_total();
        println!(
            "rank {rank}   n_head={}  n_kv_head={}  head_dim={}  weight_bytes={}",
            attn.n_head, attn.n_kv_head, attn.head_dim, r_bytes
        );
        assert_eq!(attn.n_head, ref_attn.n_head / world, "rank {rank} n_head");
        assert_eq!(attn.n_kv_head, ref_attn.n_kv_head / world, "rank {rank} n_kv_head");
        assert_eq!(attn.head_dim, ref_attn.head_dim);
        sum_bytes += r_bytes;
    }

    // Allow small slack per weight from the MATRIX_ROW_PADDING tail zero
    // at the end of each QStorage (< 1 block per weight × 4 weights).
    let slack = 4 * 4096;
    let diff = (sum_bytes as i64 - ref_bytes as i64).abs() as usize;
    println!(
        "Σ_rank = {sum_bytes}   reference = {ref_bytes}   diff = {diff}   slack ≤ {slack}"
    );
    assert!(
        diff <= slack,
        "TP per-rank byte sum diverged from reference: diff={diff} > slack={slack}"
    );

    println!("\nOK");
    Ok(())
}
