//! Quantized Gemma4 inference example.
//!
//! Supports gemma4 dense (31B), MoE (26B-A4B), and small (E4B) variants.
//! Auto-detects variant from GGUF metadata.
//!
//! Usage:
//!   cargo run --example quantized-gemma4 --release -- \
//!     --model path/to/gemma-4-E4B-it-Q4_0.gguf \
//!     --prompt "Hello, world!"
//!
//! Tokenizer is read from the GGUF file automatically. Override with --tokenizer if needed.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use std::io::Write;
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_gemma4::ModelWeights;

const DEFAULT_PROMPT: &str = "Explain the concept of attention in neural networks.";

#[derive(Parser, Debug)]
#[command(author, version, about = "Quantized Gemma4 GGUF inference")]
struct Args {
    /// GGUF model file path
    #[arg(long)]
    model: String,

    /// Tokenizer JSON file path (optional — reads from GGUF if not provided)
    #[arg(long)]
    tokenizer: Option<String>,

    /// Input prompt
    #[arg(long)]
    prompt: Option<String>,

    /// Max tokens to generate
    #[arg(short = 'n', long, default_value_t = 200)]
    sample_len: usize,

    /// Temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-K sampling
    #[arg(long)]
    top_k: Option<usize>,

    /// Top-P (nucleus) sampling
    #[arg(long)]
    top_p: Option<f64>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Run on CPU
    #[arg(long)]
    cpu: bool,

    /// Number of GPUs to split the model across (pipeline-parallel layer split).
    /// 1 = single GPU (default). When > 1, uses Hip(0)..Hip(N-1).
    #[arg(long, default_value_t = 1)]
    n_gpus: usize,

    /// Repeat penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Repeat penalty context
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Build the device list. For multi-GPU, allocate Hip(0)..Hip(n_gpus-1).
    let devices: Vec<candle::Device> = if args.cpu {
        vec![candle::Device::Cpu]
    } else if args.n_gpus > 1 {
        (0..args.n_gpus)
            .map(candle::Device::new_hip)
            .collect::<candle::Result<Vec<_>>>()?
    } else {
        vec![candle_examples::device(false)?]
    };
    let device = devices[0].clone();
    println!("devices: {:?}", devices.iter().map(|d| d.location()).collect::<Vec<_>>());

    let model_path = std::path::PathBuf::from(&args.model);
    let start = std::time::Instant::now();

    // mmap-backed loader: handles single-file GGUFs and split
    // `*-NNNNN-of-MMMMM.gguf` multi-file GGUFs. The returned `GgufBlob` is
    // shared via `Arc` across rayon workers in `from_gguf_multi_device`,
    // which loads layer weights in parallel.
    let (ct, blob) = gguf_file::Content::read_mmap(&model_path)
        .map_err(|e| e.with_path(&model_path))?;

    let name = ct.metadata.get("general.name")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let total_size: usize = ct.tensor_infos.iter()
        .map(|(_, t)| t.shape.elem_count() * t.ggml_dtype.type_size() / t.ggml_dtype.block_size())
        .sum();
    println!(
        "name: {name}, tensors: {}, size: {:.2}GB, loaded in {:.2}s",
        ct.tensor_infos.len(),
        total_size as f64 / 1e9,
        start.elapsed().as_secs_f32(),
    );

    // Build tokenizer: from GGUF metadata (default) or external file
    let tokenizer = match &args.tokenizer {
        Some(path) => Tokenizer::from_file(path).map_err(anyhow::Error::msg)?,
        None => {
            println!("reading tokenizer from GGUF metadata...");
            Tokenizer::from_gguf(&ct).map_err(anyhow::Error::msg)?
        }
    };

    let mut model = if devices.len() > 1 {
        ModelWeights::from_gguf_multi_device(ct, blob, &devices)?
    } else {
        ModelWeights::from_gguf(ct, blob, &device)?
    };
    println!("model built in {:.2}s", start.elapsed().as_secs_f32());
    let mut tos = TokenOutputStream::new(tokenizer);

    let prompt_str = args.prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    // Gemma4 chat template (different from gemma3): <bos><|turn>role\n…<turn|>\n<|turn>model\n
    // <|turn> = id 105, <turn|> = id 106 in the gemma4 vocab.
    let prompt_str = format!("<bos><|turn>user\n{prompt_str}<turn|>\n<|turn>model\n");
    print!("prompt: {prompt_str}");

    let tokens = tos.tokenizer().encode(prompt_str, true).map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();
    println!("({} tokens)", tokens.len());

    let mut logits_processor = {
        let sampling = if args.temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature: args.temperature },
                (Some(k), None) => Sampling::TopK { k, temperature: args.temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature: args.temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature: args.temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    // Q2: warmup pass to force rocBLAS Tensile JIT out of the timed
    // window. Mirrors `llama-bench`'s default warmup.
    {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0)?;
        let input1 = Tensor::new(&tokens[..1], &device)?.unsqueeze(0)?;
        let _ = model.forward(&input1, tokens.len())?;
        model.clear_kv_cache();
        device.synchronize()?;
    }
    let start_gen = std::time::Instant::now();
    let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;

    let mut next_token = logits_processor.sample(&logits)?;
    let prompt_dt = start_gen.elapsed();

    println!(
        "prompt processed: {:.2} token/s",
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );

    let mut all_tokens = vec![next_token];
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let vocab = tos.tokenizer().get_vocab(true);
    // Gemma4 ends model turns with <turn|> (id 106 in this vocab).
    // <eos> (id 1) is the legacy stop token; we accept either.
    let eos_token = vocab.get("<turn|>")
        .or_else(|| vocab.get("<eos>"))
        .copied()
        .unwrap_or(u32::MAX);

    let start_decode = std::time::Instant::now();
    let mut sampled = 0;

    for index in 0..args.sample_len.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits, args.repeat_penalty, &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        }
    }

    if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;

    let dt = start_decode.elapsed();
    println!(
        "\n\n{:4} prompt tokens: {:.2} t/s",
        tokens.len(),
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} t/s",
        sampled as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
