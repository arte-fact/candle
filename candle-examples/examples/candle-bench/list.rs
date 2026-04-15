//! `candle-bench list` — enumerate .gguf files in a directory with metadata.
//!
//! Uses `meta::read` for each file so the output has the same shape.

use crate::{meta, output::Envelope};
use anyhow::Result;
use std::path::Path;

pub fn cmd(dir: &Path, json: bool) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    entries.sort();

    let mut rows = Vec::<meta::Meta>::new();
    let mut errs = std::collections::BTreeMap::<String, String>::new();
    for p in &entries {
        match meta::read(p) {
            Ok(m) => rows.push(m),
            Err(e) => {
                errs.insert(p.display().to_string(), e.to_string());
            }
        }
    }

    if json {
        let mut env = Envelope::new("list", &rows);
        env.errors = errs;
        env.print_json()?;
    } else {
        println!("| file | arch | size | params | quant mix | context | moe |");
        println!("|------|------|-----:|--------|-----------|--------:|-----|");
        for m in &rows {
            let name = std::path::Path::new(&m.path)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or(&m.path);
            let size_gib = (m.size_bytes as f64) / 1024.0 / 1024.0 / 1024.0;
            let arch = m.arch.as_deref().unwrap_or("?");
            let params = m.size_label.as_deref().unwrap_or("?");
            let quants = {
                let mut v: Vec<_> = m.dtype_histogram.iter().collect();
                v.sort_by(|a, b| b.1.cmp(a.1));
                v.into_iter()
                    .take(3)
                    .map(|(k, c)| format!("{k}×{c}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let ctx = m
                .context_length
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into());
            let moe = match (m.expert_count, m.expert_used_count) {
                (Some(e), Some(k)) => format!("{e}×top{k}"),
                _ => "—".into(),
            };
            println!(
                "| {name} | {arch} | {size_gib:.2} GiB | {params} | {quants} | {ctx} | {moe} |"
            );
        }
        if !errs.is_empty() {
            println!("\n## errors");
            for (k, v) in &errs {
                println!("- {k}: {v}");
            }
        }
    }
    Ok(())
}
