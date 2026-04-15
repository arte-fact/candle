//! `candle-bench matrix` — iterate the catalog, run candle vs llamacpp per
//! entry, emit a matrix.
//!
//! Sequential by default (backends within an entry, entries sequentially);
//! `--parallel` runs candle + llamacpp concurrently per-entry since they
//! share the same GPU set and serialize badly otherwise.  Parallelism
//! *across entries* is not implemented in v1 — most entries use the same
//! all-4 GPU set, so parallel entries would thrash.

use crate::{bench, catalog, output::Envelope};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Serialize)]
pub struct MatrixRow {
    pub label: String,
    pub tags: Vec<String>,
    pub gpus: String,
    /// Per-backend, per-key rate (t/s).
    pub rates: BTreeMap<String, BTreeMap<String, f32>>, // rates["candle"]["pp128"]
    /// Per-backend, per-key error message.
    pub errors: BTreeMap<String, BTreeMap<String, String>>,
    /// Known-blocker map (from the catalog entry).
    pub blockers: BTreeMap<String, String>,
}

pub fn cmd(
    catalog_path: Option<&Path>,
    tag: Option<&str>,
    only: Option<&str>,
    _parallel: bool,
    reps: usize,
    timeout: u64,
    json: bool,
) -> Result<()> {
    // Load the catalog (either builtin or user-specified JSON).
    let entries: Vec<catalog::CatalogEntry> = if let Some(p) = catalog_path {
        serde_json::from_reader(std::fs::File::open(p)?)?
    } else {
        catalog::builtin()
    };

    let picks: Vec<&catalog::CatalogEntry> = entries
        .iter()
        .filter(|e| {
            (only.is_none_or(|lbl| e.label == lbl))
                && (tag.is_none_or(|t| e.tags.iter().any(|x| x == t)))
                && e.model.exists()
        })
        .collect();

    if picks.is_empty() {
        eprintln!("no catalog entries matched the filters (or model files missing)");
    }

    let mut rows = Vec::<MatrixRow>::new();
    for e in &picks {
        let mut row = MatrixRow {
            label: e.label.clone(),
            tags: e.tags.clone(),
            gpus: e.gpus.clone(),
            rates: BTreeMap::new(),
            errors: BTreeMap::new(),
            blockers: e.blockers.clone(),
        };

        // Derive prompt_lens & tg_len from entry.keys.
        let mut prompt_lens = Vec::<usize>::new();
        let mut tg_len = 0usize;
        for k in &e.keys {
            if let Some(rest) = k.strip_prefix("pp") {
                if let Ok(n) = rest.parse::<usize>() {
                    prompt_lens.push(n);
                }
            } else if let Some(rest) = k.strip_prefix("tg") {
                if let Ok(n) = rest.parse::<usize>() {
                    tg_len = n;
                }
            }
        }

        let base_args = bench::BenchArgs {
            model: e.model.clone(),
            backend: "candle".into(),
            bin: Some(e.bin.clone()),
            gpus: e.gpus.clone(),
            prompt_lens: prompt_lens.clone(),
            tg_len,
            envs: vec!["CANDLE_MMQ_TURBO_PORT=1".into()],
            extra_args: e.extras.clone(),
            reps,
            timeout,
            json: false,
            log_dir: None,
        };

        // candle (respect blockers by skipping the blocked keys).
        {
            let mut cargs = base_args.clone();
            cargs.prompt_lens.retain(|p| !e.blockers.contains_key(&format!("pp{p}")));
            if e.blockers.contains_key(&format!("tg{}", tg_len)) {
                cargs.tg_len = 0;
            }
            match bench::run(&cargs) {
                Ok(r) => {
                    let mut m = BTreeMap::new();
                    for (k, v) in r.pp {
                        m.insert(k, v);
                    }
                    if let Some(t) = r.tg {
                        m.insert(format!("tg{}", tg_len), t);
                    }
                    row.rates.insert("candle".into(), m);
                    if !r.sub_errors.is_empty() {
                        row.errors.insert("candle".into(), r.sub_errors);
                    }
                }
                Err(err) => {
                    row.errors.insert(
                        "candle".into(),
                        std::iter::once(("_call".into(), err.to_string())).collect(),
                    );
                }
            }
        }

        // llamacpp — usually fine regardless of candle blockers.
        {
            let mut largs = base_args.clone();
            largs.backend = "llamacpp".into();
            largs.bin = None;
            largs.extra_args = Vec::new();
            largs.envs = Vec::new();
            match bench::run(&largs) {
                Ok(r) => {
                    let mut m = BTreeMap::new();
                    for (k, v) in r.pp {
                        m.insert(k, v);
                    }
                    if let Some(t) = r.tg {
                        m.insert(format!("tg{}", tg_len), t);
                    }
                    row.rates.insert("llamacpp".into(), m);
                    if !r.sub_errors.is_empty() {
                        row.errors.insert("llamacpp".into(), r.sub_errors);
                    }
                }
                Err(err) => {
                    row.errors.insert(
                        "llamacpp".into(),
                        std::iter::once(("_call".into(), err.to_string())).collect(),
                    );
                }
            }
        }

        rows.push(row);
    }

    if json {
        Envelope::new("matrix", &rows).print_json()
    } else {
        print_md(&rows);
        Ok(())
    }
}

fn print_md(rows: &[MatrixRow]) {
    println!("| model | bench | candle | llamacpp | c/llama | note |");
    println!("|-------|-------|-------:|---------:|--------:|------|");
    for r in rows {
        let mut keys: Vec<String> = r
            .rates
            .values()
            .flat_map(|m| m.keys())
            .cloned()
            .collect();
        keys.extend(r.blockers.keys().cloned());
        keys.sort();
        keys.dedup();
        for k in &keys {
            let c = r.rates.get("candle").and_then(|m| m.get(k)).copied();
            let l = r.rates.get("llamacpp").and_then(|m| m.get(k)).copied();
            let mut notes = Vec::<String>::new();
            if let Some(b) = r.blockers.get(k) {
                notes.push(format!("blocker: {b}"));
            }
            if let Some(e) = r.errors.get("candle").and_then(|m| m.get(k)) {
                notes.push(format!("candle: {}", short(e)));
            }
            if let Some(e) = r.errors.get("llamacpp").and_then(|m| m.get(k)) {
                notes.push(format!("llama: {}", short(e)));
            }
            println!(
                "| {} | {k} | {} | {} | {} | {} |",
                r.label,
                crate::output::fmt_rate(c),
                crate::output::fmt_rate(l),
                crate::output::fmt_ratio(c, l),
                notes.join(" / ")
            );
        }
    }
}

fn short(s: &str) -> String {
    s.chars().take(60).collect::<String>()
}
