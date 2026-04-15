//! `candle-bench meta` — dump GGUF metadata.
//!
//! Parses the GGUF header via `candle::quantized::gguf_file::Content::read`
//! (no tensor loading) so this is ~instant on any size model.
//! Extracts the fields I've repeatedly fished out by hand with `struct.unpack`:
//! arch, size_label, n_experts, n_experts_used, block counts, head counts.

use crate::output::Envelope;
use anyhow::Result;
use candle::quantized::gguf_file;
use serde::Serialize;
use std::fs::File;
use std::path::Path;

#[derive(Serialize, Clone)]
pub struct Meta {
    pub path: String,
    pub size_bytes: u64,
    pub tensor_count: usize,
    pub kv_count: usize,
    pub arch: Option<String>,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub quantized_by: Option<String>,
    pub license: Option<String>,

    // Model structure (populated when the arch exposes them).
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub head_dim: Option<u64>,
    pub context_length: Option<u64>,

    // MoE-specific.
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub expert_feed_forward_length: Option<u64>,

    // Hybrid (GDN / SSM) hints — we just report presence.
    pub is_hybrid: bool,

    // Tensor dtype histogram — useful to spot F16/BF16 mixed quants.
    pub dtype_histogram: std::collections::BTreeMap<String, usize>,
}

pub fn read(path: &Path) -> Result<Meta> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let content = gguf_file::Content::read(&mut f)?;

    // For each key family, try with and without an architecture prefix.
    // Candle's quantized-qwen35 example uses the HF convention:
    // `<arch>.block_count`, etc.
    let arch_str = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned());
    let prefix = arch_str.as_deref().unwrap_or("");

    let get = |key: &str| content.metadata.get(key);
    let getp = |suffix: &str| -> Option<&gguf_file::Value> {
        if prefix.is_empty() {
            None
        } else {
            content.metadata.get(&format!("{prefix}.{suffix}"))
        }
    };

    let s = |k: &str| get(k).and_then(|v| v.to_string().ok().cloned());
    let u = |v: &gguf_file::Value| -> Option<u64> {
        // Different models pack these as u32 / u64 / i32 — try all.
        v.to_u64()
            .or_else(|_| v.to_u32().map(|x| x as u64))
            .or_else(|_| v.to_i64().map(|x| x as u64))
            .or_else(|_| v.to_i32().map(|x| x as u64))
            .ok()
    };
    let un = |key: &str| -> Option<u64> {
        get(key).and_then(u).or_else(|| getp(key).and_then(u))
    };

    let block_count = un("block_count");
    let embedding_length = un("embedding_length");
    let feed_forward_length = un("feed_forward_length");
    let head_count = un("attention.head_count");
    let head_count_kv = un("attention.head_count_kv");
    let head_dim = un("attention.key_length"); // common stand-in
    let context_length = un("context_length");
    let expert_count = un("expert_count");
    let expert_used_count = un("expert_used_count");
    let expert_feed_forward_length = un("expert_feed_forward_length");

    // `ssm_*` keys flag GatedDeltaNet / Mamba-style hybrid archs.
    let is_hybrid = content
        .metadata
        .keys()
        .any(|k| k.contains("ssm_") || k.contains("gated_delta"));

    // Tensor dtype histogram via the already-parsed tensor infos.
    let mut dtype_histogram: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for (_name, info) in content.tensor_infos.iter() {
        *dtype_histogram
            .entry(format!("{:?}", info.ggml_dtype))
            .or_insert(0) += 1;
    }

    Ok(Meta {
        path: path.display().to_string(),
        size_bytes: size,
        tensor_count: content.tensor_infos.len(),
        kv_count: content.metadata.len(),
        arch: arch_str,
        name: s("general.name"),
        size_label: s("general.size_label"),
        quantized_by: s("general.quantized_by"),
        license: s("general.license"),
        block_count,
        embedding_length,
        feed_forward_length,
        head_count,
        head_count_kv,
        head_dim,
        context_length,
        expert_count,
        expert_used_count,
        expert_feed_forward_length,
        is_hybrid,
        dtype_histogram,
    })
}

pub fn cmd(path: &Path, json: bool) -> Result<()> {
    let meta = read(path)?;
    if json {
        Envelope::new("meta", &meta).print_json()
    } else {
        print_md(&meta);
        Ok(())
    }
}

pub fn print_md(m: &Meta) {
    println!("# {}", m.path);
    println!("- size:             {:.2} GiB ({} B)",
             (m.size_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
             m.size_bytes);
    println!("- arch:             {}", m.arch.as_deref().unwrap_or("?"));
    if let Some(ref n) = m.name {
        println!("- name:             {n}");
    }
    if let Some(ref q) = m.quantized_by {
        println!("- quantized_by:     {q}");
    }
    if let Some(ref s) = m.size_label {
        println!("- size_label:       {s}");
    }
    println!("- tensor_count:     {}", m.tensor_count);
    println!("- kv_count:         {}", m.kv_count);
    if let Some(v) = m.block_count {
        println!("- block_count:      {v}");
    }
    if let Some(v) = m.embedding_length {
        println!("- embedding_length: {v}");
    }
    if let Some(v) = m.feed_forward_length {
        println!("- ffn_length:       {v}");
    }
    if let Some(v) = m.head_count {
        println!("- head_count:       {v}");
    }
    if let Some(v) = m.head_count_kv {
        println!("- head_count_kv:    {v}");
    }
    if let Some(v) = m.context_length {
        println!("- context_length:   {v}");
    }
    if let Some(v) = m.expert_count {
        println!("- n_experts:        {v}");
    }
    if let Some(v) = m.expert_used_count {
        println!("- topk:             {v}");
    }
    if let Some(v) = m.expert_feed_forward_length {
        println!("- expert_ffn_len:   {v}");
    }
    if m.is_hybrid {
        println!("- hybrid:           true  (SSM / GatedDeltaNet keys present)");
    }
    if !m.dtype_histogram.is_empty() {
        print!("- dtype_histogram:  ");
        let mut first = true;
        for (k, v) in m.dtype_histogram.iter() {
            if !first {
                print!(", ");
            }
            print!("{k}={v}");
            first = false;
        }
        println!();
    }
}
