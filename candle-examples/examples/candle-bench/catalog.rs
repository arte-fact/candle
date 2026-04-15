//! Built-in model catalog.  Mirrors `scripts/download-models.sh` plus
//! per-model metadata that `matrix` needs (candle binary, expected
//! keys, extra CLI flags, known blockers).
//!
//! Loaded in-place rather than from disk so an out-of-repo invocation
//! still works.  Override via `candle-bench matrix --catalog FILE` where
//! FILE is a JSON array of `CatalogEntry`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CatalogEntry {
    pub label: String,
    pub model: PathBuf,
    /// Candle example binary (matches `target/release/examples/<bin>`).
    pub bin: String,
    /// GPU list, same string candle / llama-bench will see.
    pub gpus: String,
    /// Bench keys to run on this entry.
    pub keys: Vec<String>,
    /// Extra CLI args appended to the candle binary (e.g. `--split-prompt`).
    pub extras: Vec<String>,
    /// Free-form tags: arch family, quant type, dense/MoE, etc.
    pub tags: Vec<String>,
    /// Per-key EXPECTED failures — so regressions vs known-broken are
    /// distinguishable.  If a key is listed here and actually succeeds,
    /// great; if a key NOT listed here fails, that's a real problem.
    pub blockers: BTreeMap<String, String>,
}

fn entry(
    label: &str,
    file: &str,
    bin: &str,
    gpus: &str,
    keys: &[&str],
    extras: &[&str],
    tags: &[&str],
    blockers: &[(&str, &str)],
) -> CatalogEntry {
    CatalogEntry {
        label: label.into(),
        model: PathBuf::from(std::env::var("MODELS_DIR")
            .unwrap_or_else(|_| "/artefact/models".into()))
            .join(file),
        bin: bin.into(),
        gpus: gpus.into(),
        keys: keys.iter().map(|s| s.to_string()).collect(),
        extras: extras.iter().map(|s| s.to_string()).collect(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        blockers: blockers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

pub fn builtin() -> Vec<CatalogEntry> {
    vec![
        // gemma4 family.
        entry(
            "gemma4-E4B-Q8_0",
            "gemma-4-E4B-it-Q8_0.gguf",
            "quantized-gemma4",
            "0",
            &["pp128", "pp512", "tg64"],
            &[],
            &["gemma4", "dense", "q8_0"],
            &[],
        ),
        entry(
            "gemma4-26B-A4B-Q8_0",
            "gemma-4-26B-A4B-it-Q8_0.gguf",
            "quantized-gemma4",
            "0,1,2,3",
            &["pp128", "pp512", "tg64"],
            &[],
            &["gemma4", "moe", "q8_0"],
            &[],
        ),
        entry(
            "gemma4-31B-Q4_K_M",
            "gemma-4-31B-it-Q4_K_M.gguf",
            "quantized-gemma4",
            "0,1,2,3",
            &["pp128", "pp512", "tg64"],
            &[],
            &["gemma4", "dense", "q4_k"],
            &[],
        ),
        // Qwen3 / Qwen3.5 dense.
        entry(
            "qwen35-9B-BF16",
            "Qwen3.5-9B-BF16.gguf",
            "quantized-qwen35",
            "0",
            &["pp128", "pp512", "tg64"],
            &[],
            &["qwen35", "dense", "bf16"],
            &[
                ("pp128", "no BF16 MMQ path in candle (dequant -> rocBLAS)"),
                ("pp512", "no BF16 MMQ path"),
            ],
        ),
        entry(
            "qwen35-27B-Q4_1",
            "Qwen3.5-27B-Q4_1.gguf",
            "quantized-qwen35",
            "0,1,2,3",
            &["pp128", "pp512", "tg64"],
            &[],
            &["qwen35", "dense", "q4_1"],
            &[],
        ),
        // Qwen3 MoE.
        entry(
            "qwen3-coder-30B-A3B-Q4_0",
            "Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf",
            "quantized-qwen35",
            "0,1,2,3",
            &["pp128", "pp512", "tg64"],
            &[],
            &["qwen3moe", "moe", "q4_0"],
            &[
                ("pp128", "rocBLAS Tensile kernel miss (Cijk_Alik...ISA906)"),
                ("pp512", "rocBLAS Tensile kernel miss"),
            ],
        ),
        // Qwen3-Next (recurrent/GDN hybrid).  Prefill requires sequential
        // split-prompt so the pp benches aren't comparable — decode only.
        entry(
            "qwen3-coder-next-Q4_0",
            "Qwen3-Coder-Next-Q4_0.gguf",
            "quantized-qwen35",
            "0,1,2,3",
            &["tg64"],
            &["--split-prompt"],
            &["qwen3next", "moe", "gdn", "q4_0"],
            &[],
        ),
        entry(
            "qwen35-35B-A3B-MXFP4",
            "Qwen3.5-35B-A3B-MXFP4_MOE.gguf",
            "quantized-qwen35",
            "0,1,2,3",
            &["pp128", "pp512", "tg64"],
            &[],
            &["qwen3moe", "moe", "mxfp4"],
            &[
                ("pp128", "no MXFP4 MMQ kernel in candle yet"),
                ("pp512", "no MXFP4 MMQ kernel in candle yet"),
                ("tg64", "no MXFP4 MMQ kernel in candle yet"),
            ],
        ),
    ]
}
