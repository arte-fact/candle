//! X8.3 forward parity: Σ_r rank_mlp.forward(x) == ref_mlp.forward(x).
//!
//! Validates the mathematical invariant that row-sharded gate/up +
//! column-sharded down + sum across ranks produces the same output
//! as the full (non-TP) dense FFN.

use anyhow::Result;
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file::Content;
use candle::{Device, HipDevice, Module, Tensor};
use candle_transformers::models::quantized_blocks::{
    DenseMlp, Gguf, MlpActivation,
};

fn make_x(dev: &Device, b: usize, s: usize, hidden: usize) -> Result<Tensor> {
    let data: Vec<f32> = (0..b * s * hidden)
        .map(|i| ((i as i64 * 19 + 7) % 193) as f32 / 96.0 - 1.0)
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
    let prefix = "blk.0";
    let b_sz = 1usize;
    let seq_len = 1usize;

    let dev0 = Device::Hip(devs[0].clone());
    let (ct_ref, blob_ref) = Content::read_mmap(&path)?;
    let gg_ref = Gguf::new(ct_ref, blob_ref, dev0.clone());

    // Peek hidden from attn_q.
    let hidden = gg_ref
        .ct
        .tensor_infos
        .get("blk.0.attn_q.weight")
        .unwrap()
        .shape
        .dims()[1];
    println!("hidden={hidden}");

    // --- Reference dense FFN ---
    let ref_mlp = DenseMlp::load_with_activation(&gg_ref, prefix, MlpActivation::Gelu)?;
    let x_ref = make_x(&dev0, b_sz, seq_len, hidden)?;
    let full_out = ref_mlp.forward(&x_ref)?;
    let full_vec: Vec<f32> = full_out
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let full_sum: f32 = full_vec.iter().sum();
    println!("full_out shape={:?} sum={full_sum:.4}", full_out.shape().dims());

    // --- Per-rank dense FFN ---
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(world);
    for rank in 0..world {
        let dev = Device::Hip(devs[rank].clone());
        let (ct, blob) = Content::read_mmap(&path)?;
        let gg = Gguf::new(ct, blob, dev.clone());
        let mlp = DenseMlp::load_tp(&gg, prefix, MlpActivation::Gelu, &cluster, rank)?;
        let x = make_x(&dev, b_sz, seq_len, hidden)?;
        let partial = mlp.forward(&x)?;
        let partial_vec: Vec<f32> = partial
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let p_sum: f32 = partial_vec.iter().sum();
        println!(
            "rank {rank}  intermediate={}  partial sum={:.4}",
            mlp.intermediate_size(), p_sum,
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
    assert!(
        max_abs < 1e-3,
        "TP dense MLP forward diverged from reference: max_abs={max_abs:.3e}"
    );
    println!("\nOK");
    Ok(())
}
