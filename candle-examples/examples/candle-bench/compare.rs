//! `candle-bench compare` — A/B two configs on the same model.
//!
//! Takes the same flags as `bench`, plus `--a ENV` and `--b ENV` (each a
//! space-separated list of `KEY=VAL` pairs applied on top of `--env`).
//! Emits both results and the per-key ratio (b / a), so:
//!   --a 'CANDLE_MMQ_TURBO_PORT=0' --b 'CANDLE_MMQ_TURBO_PORT=1'
//! gives the speedup factor for the turbo port vs the baseline.

use crate::{bench, output::Envelope};
use anyhow::Result;
use clap::Args;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Args, Clone, Debug)]
pub struct CompareArgs {
    #[command(flatten)]
    pub base: bench::BenchArgs,

    /// Env for config A.  Space-separated `KEY=VAL` pairs.
    #[arg(long, default_value = "")]
    pub a: String,
    /// Env for config B.
    #[arg(long, default_value = "")]
    pub b: String,
    /// Label for config A in the output.
    #[arg(long, default_value = "A")]
    pub a_label: String,
    /// Label for config B in the output.
    #[arg(long, default_value = "B")]
    pub b_label: String,
}

#[derive(Serialize)]
struct CompareOut {
    a_label: String,
    b_label: String,
    a: bench::BenchResult,
    b: bench::BenchResult,
    /// `ratio["pp512"] = b.pp512 / a.pp512`.
    ratio: BTreeMap<String, f32>,
}

pub fn cmd(args: &CompareArgs) -> Result<()> {
    let mk_env = |extra: &str| -> Vec<String> {
        let mut v = args.base.envs.clone();
        for tok in extra.split_whitespace() {
            v.push(tok.to_string());
        }
        v
    };

    let mut base_a = args.base.clone();
    base_a.envs = mk_env(&args.a);
    let mut base_b = args.base.clone();
    base_b.envs = mk_env(&args.b);

    let a = bench::run(&base_a)?;
    let b = bench::run(&base_b)?;

    let mut ratio = BTreeMap::new();
    for (k, av) in &a.pp {
        if let Some(bv) = b.pp.get(k) {
            if *av > 0.0 {
                ratio.insert(k.clone(), *bv / *av);
            }
        }
    }
    if let (Some(av), Some(bv)) = (a.tg, b.tg) {
        if av > 0.0 {
            ratio.insert("tg".into(), bv / av);
        }
    }

    let out = CompareOut {
        a_label: args.a_label.clone(),
        b_label: args.b_label.clone(),
        a,
        b,
        ratio,
    };

    if args.base.json {
        Envelope::new("compare", &out).print_json()
    } else {
        print_md(&out);
        Ok(())
    }
}

fn print_md(c: &CompareOut) {
    let model = std::path::Path::new(&c.a.model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&c.a.model);
    println!("# compare — {model}  ({} vs {})", c.a_label, c.b_label);
    println!(
        "- {}: env={:?}",
        c.a_label,
        c.a.env
            .iter()
            .filter(|(k, _)| k.starts_with("CANDLE_") || !["LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"].contains(&k.as_str()))
            .collect::<Vec<_>>()
    );
    println!(
        "- {}: env={:?}",
        c.b_label,
        c.b.env
            .iter()
            .filter(|(k, _)| k.starts_with("CANDLE_") || !["LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"].contains(&k.as_str()))
            .collect::<Vec<_>>()
    );
    println!();
    println!("| bench | {} | {} | {}/{} |", c.a_label, c.b_label, c.b_label, c.a_label);
    println!("|-------|----:|----:|------:|");
    let mut keys: Vec<String> = c.a.pp.keys().chain(c.b.pp.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    for k in &keys {
        let av = c.a.pp.get(k);
        let bv = c.b.pp.get(k);
        println!(
            "| {k} | {} | {} | {} |",
            crate::output::fmt_rate(av.copied()),
            crate::output::fmt_rate(bv.copied()),
            crate::output::fmt_ratio(bv.copied(), av.copied()),
        );
    }
    if c.a.tg.is_some() || c.b.tg.is_some() {
        println!(
            "| tg | {} | {} | {} |",
            crate::output::fmt_rate(c.a.tg),
            crate::output::fmt_rate(c.b.tg),
            crate::output::fmt_ratio(c.b.tg, c.a.tg),
        );
    }
}
