//! candle-bench — bench / investigation tool for the candle project.
//!
//! Subcommands:
//!   meta      Parse GGUF header → arch, n_experts, topk, sizes.
//!   list      Enumerate .gguf files in a directory with metadata.
//!   fetch     Download a GGUF from a direct URL via curl (resume + size-verify).
//!   bench     Run one bench (candle / llama.cpp / turbo) and emit JSON or md.
//!   compare   A/B-compare two env configs on the same model.
//!   matrix    Iterate a built-in catalog with filters; parallel optional.
//!   kernels   Wrap llvm-readobj to list kernels + VGPR counts in a .hsaco.
//!
//! Everything has `--json` for machine-readable stdout.  See
//! `.claude/skills/candle-bench/SKILL.md` for usage from Claude Code.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod bench;
mod catalog;
mod compare;
mod fetch;
mod kernels;
mod list;
mod matrix;
mod meta;
mod output;

#[derive(Parser, Debug)]
#[command(name = "candle-bench",
          about = "candle-project bench / investigation tool",
          version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Dump GGUF metadata for one model.
    Meta {
        model: PathBuf,
        #[arg(long)]
        json: bool,
    },

    /// List .gguf files in a directory with metadata.  Defaults to
    /// `$MODELS_DIR` (env) or `/artefact/models`.
    List {
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },

    /// Download a model from a direct URL (curl -C -, size-verified).
    Fetch {
        /// Any direct URL; typical form is
        /// https://huggingface.co/OWNER/REPO/resolve/main/FILE.gguf
        url: String,
        #[arg(long)]
        dir: Option<PathBuf>,
        /// If set, fail unless the downloaded file matches this size.
        #[arg(long)]
        expected_bytes: Option<u64>,
        /// Override destination filename (default: trailing URL segment).
        #[arg(long)]
        out_name: Option<String>,
    },

    /// Run one bench.
    Bench(bench::BenchArgs),

    /// A/B compare two configs on the same model.
    Compare(compare::CompareArgs),

    /// Iterate the model catalog; emit a matrix of backends × keys.
    Matrix {
        #[arg(long)]
        catalog: Option<PathBuf>,
        /// Filter catalog entries by tag (e.g. "moe", "q4_0", "gemma4").
        #[arg(long)]
        tag: Option<String>,
        /// Filter catalog entries by exact label.
        #[arg(long)]
        only: Option<String>,
        /// Run entries on disjoint GPU sets concurrently where possible.
        #[arg(long)]
        parallel: bool,
        #[arg(long, default_value_t = 1)]
        reps: usize,
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        #[arg(long)]
        json: bool,
    },

    /// List kernel symbols and VGPR counts in a compiled .hsaco.
    Kernels {
        hsaco: PathBuf,
        /// Substring filter on kernel name.
        #[arg(long)]
        filter: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn models_dir(arg: Option<PathBuf>) -> PathBuf {
    arg.or_else(|| std::env::var_os("MODELS_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/artefact/models"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Meta { model, json } => meta::cmd(&model, json),
        Cmd::List { dir, json } => list::cmd(&models_dir(dir), json),
        Cmd::Fetch { url, dir, expected_bytes, out_name } => {
            fetch::cmd(&url, &models_dir(dir), expected_bytes, out_name.as_deref())
        }
        Cmd::Bench(args) => bench::cmd(&args),
        Cmd::Compare(args) => compare::cmd(&args),
        Cmd::Matrix { catalog, tag, only, parallel, reps, timeout, json } => {
            matrix::cmd(catalog.as_deref(), tag.as_deref(), only.as_deref(),
                        parallel, reps, timeout, json)
        }
        Cmd::Kernels { hsaco, filter, json } => {
            kernels::cmd(&hsaco, filter.as_deref(), json)
        }
    }
}
