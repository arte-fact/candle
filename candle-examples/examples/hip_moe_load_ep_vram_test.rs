//! X5 end-to-end: `MoeExperts::load_ep` on a real GGUF actually allocates
//! only the per-rank shard (1/W of expert bytes). No 2× VRAM spike.
//!
//! Builds a HipCluster from the first 2 HIP devices, opens
//! Qwen3-Coder-30B-A3B Q4_0, calls `MoeExperts::load_ep` for rank=0 and
//! rank=1 on `blk.0`, and asserts:
//!   1. Each rank's expert tensor shape[0] equals its local expert count.
//!   2. Each rank's expert tensor byte-size equals
//!      `local_count / num_experts × full_bytes`.

use anyhow::{Context, Result};
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{Device, HipDevice};
use candle_transformers::models::quantized_blocks::{GgufConfig, Gguf, MoeExperts};

fn main() -> Result<()> {
    let path = std::env::var("CANDLE_X5_GGUF")
        .unwrap_or_else(|_| "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf".into());
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
    println!("World size = {world} (ranks 0..{world})");

    let cluster = std::sync::Arc::new(HipCluster::new(devs.clone())?);

    let (content, blob) = Content::read_mmap(&path)?;
    let cfg = GgufConfig::from_metadata(&content.metadata)?;
    println!(
        "arch={} num_experts={} expert_used={} expert_ffn={}",
        cfg.arch,
        cfg.expert_count.unwrap_or(0),
        cfg.expert_used_count.unwrap_or(0),
        cfg.feed_forward_length.unwrap_or(0),
    );

    let prefix = "blk.0";
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let info = content
        .tensor_infos
        .get(&down_name)
        .with_context(|| format!("missing {down_name}"))?;
    let num_experts = info.shape.dims()[0];
    let full_down_bytes = {
        let elems = info.shape.elem_count();
        elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()
    };
    println!(
        "full {down_name}  num_experts={num_experts}  bytes={}",
        full_down_bytes,
    );

    let block = (num_experts + world - 1) / world;

    drop(content);
    drop(blob);

    for rank in 0..world {
        let device = Device::Hip(devs[rank].clone());
        // Fresh (Content, blob) per rank — all pread-backed mmaps on
        // the same file, ~zero-cost to re-open.
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, device.clone());
        let me = MoeExperts::load_ep(&gg, prefix, &cfg, cluster.clone(), rank)?;

        let local_start = rank * block;
        let local_count = num_experts.saturating_sub(local_start).min(block);
        let expected_bytes = (local_count * full_down_bytes) / num_experts;

        let down = me.down_exps();
        let down_dims = down.shape().dims().to_vec();
        let down_bytes = down.storage_size_in_bytes();
        println!(
            "rank={rank}  local=[{local_start}..{}]  down_exps shape={:?}  bytes={down_bytes}  expected≈{expected_bytes}",
            local_start + local_count, down_dims,
        );
        assert_eq!(down_dims[0], local_count, "rank {rank} down_exps dim0");
        let bytes_diff = (down_bytes as i64 - expected_bytes as i64).abs() as usize;
        assert_eq!(
            down_bytes, expected_bytes,
            "rank {rank}: down_exps bytes {down_bytes} != expected {expected_bytes} (diff {bytes_diff})"
        );

        let part = me.ep_partition().expect("ep_partition present");
        assert_eq!(part.rank, rank);
        assert_eq!(part.local_start, local_start);
        assert_eq!(part.local_count, local_count);
        assert_eq!(part.num_experts, num_experts);
    }

    // --- Part 2: gemma4 MoeBranch::load_ep ---
    let gemma_path = std::env::var("CANDLE_X5_GEMMA_GGUF")
        .unwrap_or_else(|_| "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf".into());
    if std::path::Path::new(&gemma_path).exists() {
        println!("\n=== gemma4 MoeBranch::load_ep ===");
        println!("GGUF: {gemma_path}");
        let (ct_g, _blob_g) = Content::read_mmap(&gemma_path)?;
        let cfg_g = GgufConfig::from_metadata(&ct_g.metadata)?;
        println!(
            "arch={} num_experts={} expert_used={} expert_ffn={}",
            cfg_g.arch,
            cfg_g.expert_count.unwrap_or(0),
            cfg_g.expert_used_count.unwrap_or(0),
            cfg_g.feed_forward_length.unwrap_or(0),
        );
        if cfg_g.arch != "gemma4" {
            println!("(skip: arch != gemma4, got {})", cfg_g.arch);
        } else {
            let g_prefix = "blk.0";
            let g_gate_up = format!("{g_prefix}.ffn_gate_up_exps.weight");
            let info_g = ct_g
                .tensor_infos
                .get(&g_gate_up)
                .with_context(|| format!("missing {g_gate_up}"))?;
            let num_experts_g = info_g.shape.dims()[0];
            let full_gate_up_bytes = {
                let e = info_g.shape.elem_count();
                e / info_g.ggml_dtype.block_size() * info_g.ggml_dtype.type_size()
            };
            println!(
                "full {g_gate_up}  num_experts={num_experts_g}  bytes={full_gate_up_bytes}",
            );
            let rms_eps = cfg_g.rms_norm_eps;
            let hidden = cfg_g.hidden_size;
            let block_g = (num_experts_g + world - 1) / world;
            drop(ct_g);

            for rank in 0..world {
                let device = Device::Hip(devs[rank].clone());
                let (ct, blob) = Content::read_mmap(&gemma_path)?;
                let gg = Gguf::new(ct, blob, device.clone());
                let mb = candle_transformers::models::quantized_gemma4::MoeBranch::load_ep(
                    &gg, g_prefix, &cfg_g, hidden, rms_eps, cluster.clone(), rank,
                )?;
                let local_start = rank * block_g;
                let local_count = num_experts_g.saturating_sub(local_start).min(block_g);
                let expected_bytes = (local_count * full_gate_up_bytes) / num_experts_g;

                let gu = mb.gate_up_exps();
                let gu_bytes = gu.storage_size_in_bytes();
                let gu_dims = gu.shape().dims().to_vec();
                println!(
                    "rank={rank}  local=[{}..{}]  gate_up shape={:?}  bytes={gu_bytes}  expected={expected_bytes}",
                    local_start, local_start + local_count, gu_dims
                );
                assert_eq!(gu_dims[0], local_count, "gemma4 gate_up dim0");
                assert_eq!(gu_bytes, expected_bytes, "gemma4 gate_up bytes");

                if let Some(scale) = mb.down_exps_scale() {
                    let scale_dims = scale.shape().dims().to_vec();
                    println!("  down_exps_scale shape={:?}", scale_dims);
                    assert_eq!(scale_dims[0], local_count, "gemma4 down_exps_scale dim0");
                }
                let part = mb.ep_partition().expect("ep_partition present");
                assert_eq!(part.rank, rank);
                assert_eq!(part.local_start, local_start);
                assert_eq!(part.local_count, local_count);
                assert_eq!(part.num_experts, num_experts_g);
            }
        }
    } else {
        println!("\n(skip gemma4 part: {gemma_path} not found)");
    }

    println!("\nOK");
    Ok(())
}
