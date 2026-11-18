//! X8.4 composite test: TP attention forward with the barrier all-reduce
//! wired INSIDE `finish_attn` via `set_tp_context`.
//!
//! Two concurrent worker threads each run their rank's attention forward;
//! the all-reduce happens transparently at the output projection. The
//! forward's return value is the FULL (reduced) output on each rank.
//! We verify both ranks see the same full output and it matches the
//! single-GPU reference.

use anyhow::Result;
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{DType, Device, HipDevice, Tensor};
use candle_transformers::models::quantized_blocks::{
    GgufConfig, Gguf, RotaryEmbedding, StandardAttention, StandardAttentionOpts, TpContext,
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
    let prefix = "blk.0";
    let b_sz = 1usize;
    let seq_len = 1usize;

    // --- Reference (non-TP) forward ---
    let dev0 = Device::Hip(devs[0].clone());
    let (ct_ref, blob_ref) = Content::read_mmap(&path)?;
    let cfg = GgufConfig::from_metadata(&ct_ref.metadata)?;
    let k_dims = ct_ref
        .tensor_infos
        .get(&format!("{prefix}.attn_k.weight"))
        .unwrap()
        .shape
        .dims()
        .to_vec();
    let head_dim_local = k_dims[0] / *cfg.head_count_kv.get(0);
    let gg_ref = Gguf::new(ct_ref, blob_ref, dev0.clone());
    let opts = StandardAttentionOpts {
        use_v_norm: true,
        use_gemma_norms: true,
        attention_scale: Some(1.0),
    };
    let rotary_ref = std::sync::Arc::new(RotaryEmbedding::new(
        1_000_000.0, head_dim_local, 8192, DType::F32, &dev0,
    )?);
    let ref_attn = StandardAttention::load_with_opts(
        &gg_ref, prefix, &cfg, 0, rotary_ref, opts,
    )?;
    let hidden = cfg.hidden_size;
    let x_ref = make_x(&dev0, b_sz, seq_len, hidden)?;
    let (k_full, v_full) = ref_attn.compute_kv(&x_ref, 0)?.unwrap();
    let full_out = ref_attn.forward_with_kv(&x_ref, &k_full, &v_full, None, 0)?;
    let full_vec: Vec<f32> = full_out
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let full_sum: f32 = full_vec.iter().sum();
    println!("reference full_out sum={full_sum:.4}");

    // --- Two threads running TP attention, each with tp_ctx attached. ---
    // Build per-rank attention structs up front.
    let mut rank_attns: Vec<StandardAttention> = Vec::with_capacity(world);
    for rank in 0..world {
        let dev = Device::Hip(devs[rank].clone());
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, dev.clone());
        let rotary = std::sync::Arc::new(RotaryEmbedding::new(
            1_000_000.0, head_dim_local, 8192, DType::F32, &dev,
        )?);
        let mut attn = StandardAttention::load_tp(
            &gg, prefix, &cfg, 0, rotary, opts, &cluster, rank,
        )?;
        attn.set_tp_context(TpContext {
            cluster: cluster.clone(),
            rank,
        });
        rank_attns.push(attn);
    }

    let cluster_c = cluster.clone();
    let cfg_hidden = hidden;
    // Pair (rank, attn, dev) so each closure owns its inputs cleanly.
    let rank_data: Vec<(usize, StandardAttention, HipDevice)> = rank_attns
        .into_iter()
        .enumerate()
        .map(|(rank, attn)| (rank, attn, devs[rank].clone()))
        .collect();
    let results: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = rank_data
            .into_iter()
            .map(|(rank, attn, dev_h)| {
                s.spawn(move || {
                    let dev = Device::Hip(dev_h.clone());
                    // Pin HIP context for this worker.
                    candle::hip_backend::hipdarc::driver::HipDevice::new(dev_h.ordinal())
                        .unwrap()
                        .set_current()
                        .unwrap();
                    let x = make_x(&dev, b_sz, seq_len, cfg_hidden).unwrap();
                    let (k, v) = attn.compute_kv(&x, 0).unwrap().unwrap();
                    let out = attn.forward_with_kv(&x, &k, &v, None, 0).unwrap();
                    let v: Vec<f32> = out
                        .flatten_all()
                        .unwrap()
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    (rank, v)
                })
            })
            .collect();
        let mut results: Vec<(usize, Vec<f32>)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        results.sort_by_key(|r| r.0);
        results.into_iter().map(|(_, v)| v).collect()
    });
    let _ = cluster_c;

    // Every rank must see the same full output (post-all-reduce).
    let n = full_vec.len();
    for (rank, got) in results.iter().enumerate() {
        assert_eq!(got.len(), n);
        let max_abs = full_vec
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rank_sum: f32 = got.iter().sum();
        println!("rank {rank}  sum={rank_sum:.4}  max_abs(vs ref)={max_abs:.3e}");
        assert!(
            max_abs < 1e-3,
            "rank {rank} barrier-TP forward diverged from reference: max_abs={max_abs:.3e}"
        );
    }
    println!("\nOK");
    Ok(())
}
