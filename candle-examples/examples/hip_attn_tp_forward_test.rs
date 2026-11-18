//! X8.4 correctness foundation: TP attention forward parity.
//!
//! For a single layer's `StandardAttention`, verify that
//!   Σ_{r=0..W} rank_r.forward_with_kv(x, k_r, v_r) == ref.forward_with_kv(x, k_full, v_full)
//! to float-rounding precision.
//!
//! This is the mathematical invariant that makes TP+EP forward correct:
//! row-sharded Q/K/V + column-sharded out_proj + all-reduce produces
//! the same output as the full (non-TP) forward.

use anyhow::Result;
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{DType, Device, HipDevice, Tensor};
use candle_transformers::models::quantized_blocks::{
    GgufConfig, Gguf, RotaryEmbedding, StandardAttention, StandardAttentionOpts,
};

fn make_x(dev: &Device, b: usize, s: usize, hidden: usize) -> Result<Tensor> {
    let data: Vec<f32> = (0..b * s * hidden)
        .map(|i| ((i as i64 * 17 + 5) % 211) as f32 / 105.0 - 1.0)
        .collect();
    Ok(Tensor::from_vec(data, (b, s, hidden), dev)?)
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

    // --- Shared setup: GgufConfig + fake x on every rank ---
    let dev0 = Device::Hip(devs[0].clone());
    let (ct_ref, blob_ref) = Content::read_mmap(&path)?;
    let cfg = GgufConfig::from_metadata(&ct_ref.metadata)?;
    let prefix = "blk.0";
    let hidden = cfg.hidden_size;
    let b_sz = 1usize;
    let seq_len = 1usize;

    let opts = StandardAttentionOpts {
        use_v_norm: true,
        use_gemma_norms: true,
        attention_scale: Some(1.0),
    };
    // Peek at attn_k weight to get the per-layer head_dim (gemma4 uses
    // different head_dim per layer for SWA vs global; cfg.head_dim is
    // only a default).
    let ref_gg = Gguf::new(ct_ref, blob_ref, dev0.clone());
    let k_dims = ref_gg
        .ct
        .tensor_infos
        .get(&format!("{prefix}.attn_k.weight"))
        .unwrap()
        .shape
        .dims()
        .to_vec();
    let n_kv_head_cfg = *cfg.head_count_kv.get(0);
    let head_dim_local = k_dims[0] / n_kv_head_cfg;
    println!("derived head_dim={head_dim_local}");
    let rotary_ref = std::sync::Arc::new(RotaryEmbedding::new(
        1_000_000.0, head_dim_local, 8192, DType::F32, &dev0,
    )?);
    let ref_attn = StandardAttention::load_with_opts(
        &ref_gg, prefix, &cfg, 0, rotary_ref.clone(), opts,
    )?;
    println!(
        "reference  n_head={} n_kv_head={} head_dim={}",
        ref_attn.n_head, ref_attn.n_kv_head, ref_attn.head_dim
    );

    // --- Reference forward ---
    let x_ref = make_x(&dev0, b_sz, seq_len, hidden)?;
    let (k_full, v_full) = ref_attn
        .compute_kv(&x_ref, 0)?
        .expect("attn_k/attn_v present on blk.0");
    let full_out = ref_attn.forward_with_kv(&x_ref, &k_full, &v_full, None, 0)?;
    let full_vec: Vec<f32> = full_out
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let full_sum: f32 = full_vec.iter().sum();
    println!("full_out shape={:?} sum={full_sum:.4}", full_out.shape().dims());

    // --- Per-rank forward, sum on CPU ---
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(world);
    for rank in 0..world {
        let dev = Device::Hip(devs[rank].clone());
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, dev.clone());
        let rotary = std::sync::Arc::new(RotaryEmbedding::new(
            1_000_000.0, head_dim_local, 8192, DType::F32, &dev,
        )?);
        let attn = StandardAttention::load_tp(
            &gg, prefix, &cfg, 0, rotary, opts, &cluster, rank,
        )?;
        let x = make_x(&dev, b_sz, seq_len, hidden)?;
        let (k, v) = attn.compute_kv(&x, 0)?.expect("attn_k/v present");
        let partial = attn.forward_with_kv(&x, &k, &v, None, 0)?;
        let partial_vec: Vec<f32> = partial
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let p_sum: f32 = partial_vec.iter().sum();
        println!(
            "rank {rank}  n_head={} n_kv_head={} partial sum={:.4}",
            attn.n_head, attn.n_kv_head, p_sum
        );
        partials.push(partial_vec);
    }

    // Sum partials element-wise.
    let n = full_vec.len();
    let mut summed = vec![0.0f32; n];
    for p in &partials {
        assert_eq!(p.len(), n);
        for i in 0..n {
            summed[i] += p[i];
        }
    }

    // Compare.
    let max_abs = full_vec
        .iter()
        .zip(summed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = {
        let ss: f32 = full_vec
            .iter()
            .zip(summed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (ss / n as f32).sqrt()
    };
    println!("\n|full - sum_ranks|:  max_abs = {max_abs:.3e},  rms = {rms:.3e}");

    // Q8_0 weights go through two DIFFERENT matmul paths here (rocBLAS on
    // full n_head vs rocBLAS/MMVQ on n_head/W), so expect small float
    // rounding divergence. 1e-3 is generous float32 GQA slack.
    assert!(
        max_abs < 1e-3,
        "TP attention forward diverged from reference: max_abs={max_abs:.3e}"
    );

    println!("\nOK");
    Ok(())
}
