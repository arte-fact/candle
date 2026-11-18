//! X8.5 — End-to-end TP+EP bench on gemma4-26B-A4B.
//!
//! Loads the model twice:
//!   1. `ModelWeights::from_gguf_multi_device` with a single device (reference).
//!   2. `ModelWeightsTp::from_gguf_tp_ep` across W devices (TP+EP).
//!
//! Compares per-token logits on a synthetic prompt for correctness,
//! then benchmarks decode throughput.
//!
//! Env:
//!   CANDLE_X8_GGUF       — path to GGUF (default: gemma-4-26B-A4B Q8_0)
//!   CANDLE_X8_WORLD      — TP world size (default: 2)
//!   CANDLE_X8_SKIP_REF   — "1" to skip the reference correctness run
//!   CANDLE_X8_DECODE_N   — decode tokens to bench (default: 32)

use anyhow::{Context, Result};
use candle::quantized::gguf_file::Content;
use candle::{Device, HipDevice, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use candle_transformers::models::quantized_gemma4_tp::ModelWeightsTp;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::var("CANDLE_X8_GGUF")
        .unwrap_or_else(|_| "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf".into());
    let world: usize = std::env::var("CANDLE_X8_WORLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let skip_ref = std::env::var("CANDLE_X8_SKIP_REF").ok().as_deref() == Some("1");
    let decode_n: usize = std::env::var("CANDLE_X8_DECODE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    // Ensure G2/G3 replay is off — multi-threaded forward breaks
    // single-threaded-recorder assumptions. The code tests
    // `env::var(..).is_ok()` so setting "0" is NOT sufficient; must
    // remove the var entirely.
    std::env::remove_var("CANDLE_G2_REPLAY");
    std::env::remove_var("CANDLE_G3_GRAPH");

    println!("GGUF: {path}");
    println!("World: {world}  decode_n: {decode_n}");

    let mut devs = Vec::<HipDevice>::new();
    for ord in 0..16 {
        if let Ok(Device::Hip(d)) = Device::new_hip(ord) {
            devs.push(d);
        } else {
            break;
        }
    }
    if devs.len() < world {
        println!(
            "Need {world} HIP devices; found {}. Bailing.",
            devs.len()
        );
        return Ok(());
    }
    devs.truncate(world);
    let devices: Vec<Device> = devs.iter().map(|d| Device::Hip(d.clone())).collect();

    // Synthetic prompt tokens used for both correctness + decode bench.
    let prompt_ids: Vec<u32> = vec![2, 1, 107, 108, 5, 10];

    // --- Capture reference logits FIRST (on a fresh single-device model).
    // We then drop the reference model so its VRAM is freed before loading
    // the TP+EP replicas — otherwise rank 0's device holds the full model
    // AND the TP shard concurrently, and HIP runtime contention can stall
    // the rayon load of the second model.
    let mut ref_logits: Option<Vec<f32>> = None;
    if !skip_ref {
        let ref_device = devices[0].clone();
        let (ct, blob) = Content::read_mmap(&path)?;
        let t0 = Instant::now();
        eprintln!("loading reference model on rank-0 device …");
        let mut refm = ModelWeights::from_gguf(ct, blob, &ref_device)
            .with_context(|| "ModelWeights::from_gguf reference")?;
        println!("Reference load: {:.1} s", t0.elapsed().as_secs_f64());
        let last_tok = *prompt_ids.last().unwrap();
        let x_ref = Tensor::from_vec(vec![last_tok], (1, 1), &ref_device)?;
        let tf = Instant::now();
        let logits_ref = refm.forward(&x_ref, 0)?;
        eprintln!(
            "reference forward: {} ms, logits shape={:?}",
            tf.elapsed().as_millis(),
            logits_ref.shape().dims()
        );
        let v: Vec<f32> = logits_ref
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        ref_logits = Some(v);
        // Drop refm before loading TP+EP replicas.
        drop(refm);
    }

    // --- Build TP+EP model ---
    let t_load = Instant::now();
    let tp_model = ModelWeightsTp::from_gguf_tp_ep(&path, &devices)
        .with_context(|| "ModelWeightsTp::from_gguf_tp_ep")?;
    let load_tp_secs = t_load.elapsed().as_secs_f64();
    println!(
        "TP+EP load: {load_tp_secs:.1} s  (W={})",
        tp_model.world_size()
    );

    // --- Correctness: compare logits at index_pos 0 on the last prompt token ---
    if let Some(ref_vec) = ref_logits {
        println!("\n=== correctness: TP+EP logits vs reference ===");
        let last_tok = *prompt_ids.last().unwrap();

        // TP+EP: forward the same token on rank 0's device (dispatcher
        // moves to each rank's device internally).
        let x_tp = Tensor::from_vec(vec![last_tok], (1, 1), &devices[0])?;
        let t0 = Instant::now();
        let logits_tp = tp_model.forward(&x_tp, 0)?;
        let dt_tp = t0.elapsed().as_millis();
        let tp_vec: Vec<f32> = logits_tp
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;

        assert_eq!(ref_vec.len(), tp_vec.len());
        let max_abs = ref_vec
            .iter()
            .zip(tp_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms = {
            let n = ref_vec.len() as f32;
            let ss: f32 = ref_vec
                .iter()
                .zip(tp_vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (ss / n).sqrt()
        };
        let ref_norm = ref_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel = rms / (ref_norm / (ref_vec.len() as f32).sqrt()).max(1e-6);
        println!(
            "tp dt={dt_tp}ms  max_abs={max_abs:.3e} rms={rms:.3e} rel={rel:.3e}"
        );

        let ref_argmax = argmax(&ref_vec);
        let tp_argmax = argmax(&tp_vec);
        println!(
            "argmax: ref={ref_argmax} tp={tp_argmax} match={}",
            ref_argmax == tp_argmax
        );
        if rel > 1e-2 {
            println!(
                "WARNING: rel logit error {rel:.3e} > 1e-2 — check TP math (reduce, head split, etc.)"
            );
        }
    }

    // --- Decode bench ---
    println!("\n=== decode bench ({decode_n} tokens, TP+EP) ===");
    tp_model.clear_kv_cache();
    // Prefill the prompt.
    let mut t_prefill = 0u128;
    let t0 = Instant::now();
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let x = Tensor::from_vec(vec![tok], (1, 1), &devices[0])?;
        let _ = tp_model.forward(&x, i)?;
    }
    t_prefill = t0.elapsed().as_millis();
    println!("prefill {} tokens: {t_prefill} ms", prompt_ids.len());

    // Decode loop. Feed argmax of last logits.
    let mut last = prompt_ids.last().copied().unwrap_or(1);
    let t0 = Instant::now();
    for step in 0..decode_n {
        let x = Tensor::from_vec(vec![last], (1, 1), &devices[0])?;
        let logits = tp_model.forward(&x, prompt_ids.len() + step)?;
        let v: Vec<f32> = logits
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        last = argmax(&v) as u32;
    }
    let t_decode = t0.elapsed().as_millis();
    let tg = (decode_n as f64) * 1000.0 / (t_decode as f64);
    println!("decode {decode_n} tokens: {t_decode} ms  → tg = {tg:.2} t/s");

    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}
