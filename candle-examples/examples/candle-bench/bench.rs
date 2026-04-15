//! `candle-bench bench` — run one bench (candle / llama.cpp / turbo).
//!
//! This is an orchestrator: it spawns the appropriate binary, sets env vars,
//! parses the trailing `N prompt tokens: X t/s` / `tokens generated: Y t/s`
//! lines, and returns a structured result.  No tensors are loaded here.

use crate::output::Envelope;
use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

pub const ROCM_LIB: &str = "/opt/rocm-7.1.1/core-7.13/lib";
pub const ROCM_EXTRA_LIB: &str = "/opt/rocm/lib";
pub const LLAMACPP_BIN_DIR: &str = "/artefact/llama.cpp/build/bin";
pub const TURBO_BIN_DIR: &str = "/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin";

#[derive(Args, Clone, Debug)]
pub struct BenchArgs {
    #[arg(long)]
    pub model: PathBuf,

    /// candle / llamacpp / turbo.
    #[arg(long, default_value = "candle")]
    pub backend: String,

    /// Candle example binary (ignored for llamacpp/turbo).  If unset, use
    /// `quantized-gemma4` — the default that works for the gemma4 family.
    /// Set e.g. `quantized-qwen35` for Qwen3 / Qwen3.5 / Qwen3Next.
    #[arg(long)]
    pub bin: Option<String>,

    /// Comma-separated GPU ids (passed via HIP_VISIBLE_DEVICES; candle
    /// also receives `--n-gpus N` for pipeline-parallel split).
    #[arg(long, default_value = "0")]
    pub gpus: String,

    /// Prompt lengths to benchmark (repeatable).  One run per length.
    #[arg(long = "prompt-len", default_values_t = [128usize, 512usize])]
    pub prompt_lens: Vec<usize>,

    /// Token-generation length for the decode bench.
    #[arg(long = "tg-len", default_value_t = 64)]
    pub tg_len: usize,

    /// `KEY=VALUE` pairs (repeatable).  Applied on top of the tool defaults.
    /// Common: `CANDLE_MMQ_TURBO_PORT=1`.
    #[arg(long = "env")]
    pub envs: Vec<String>,

    /// Extra CLI args to pass through to the candle binary (e.g.
    /// `--split-prompt` for qwen3next).  Repeatable.
    #[arg(long = "extra-arg")]
    pub extra_args: Vec<String>,

    #[arg(long, default_value_t = 1)]
    pub reps: usize,
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub log_dir: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchResult {
    pub backend: String,
    pub bin: String,
    pub model: String,
    pub gpus: String,
    pub reps: usize,
    pub wall_s: f64,
    pub env: BTreeMap<String, String>,
    /// "pp128" / "pp512" / ... mapped to best-of-reps t/s.
    pub pp: BTreeMap<String, f32>,
    /// Decode t/s for tg_len tokens.
    pub tg: Option<f32>,
    /// Per-subbench error messages (key = "pp128", "tg64", ...).
    pub sub_errors: BTreeMap<String, String>,
    /// Where the full stdout/stderr went, if a log_dir was given.
    pub log: Option<String>,
}

pub fn cmd(args: &BenchArgs) -> Result<()> {
    let res = run(args)?;
    if args.json {
        Envelope::new("bench", &res).print_json()
    } else {
        print_md(&res);
        Ok(())
    }
}

pub fn run(args: &BenchArgs) -> Result<BenchResult> {
    let t0 = Instant::now();
    let log_path = args
        .log_dir
        .as_ref()
        .map(|d| {
            std::fs::create_dir_all(d).ok();
            d.join("bench.log")
        });
    if let Some(ref p) = log_path {
        std::fs::write(p, b"").ok();
    }

    let res = match args.backend.as_str() {
        "candle" => run_candle(args, log_path.as_deref()),
        "llamacpp" => run_llama_bench(args, LLAMACPP_BIN_DIR, log_path.as_deref()),
        "turbo" => run_llama_bench(args, TURBO_BIN_DIR, log_path.as_deref()),
        other => bail!("unknown backend {other:?}"),
    }?;
    let mut out = res;
    out.wall_s = t0.elapsed().as_secs_f64();
    out.log = log_path.map(|p| p.display().to_string());
    Ok(out)
}

fn env_base(gpus: &str, extra_ld: &[&str]) -> Vec<(String, String)> {
    let mut env = Vec::new();
    env.push(("HIP_VISIBLE_DEVICES".into(), gpus.into()));
    // Unset ROCR_VISIBLE_DEVICES since HIP double-filters otherwise.
    env.push(("ROCR_VISIBLE_DEVICES".into(), String::new()));
    let mut lib = format!("{ROCM_LIB}:{ROCM_EXTRA_LIB}");
    for x in extra_ld {
        lib.push(':');
        lib.push_str(x);
    }
    env.push(("LD_LIBRARY_PATH".into(), lib));
    env
}

fn parse_kvs(raw: &[String]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for s in raw {
        if let Some((k, v)) = s.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        }
    }
    out
}

fn append_log(path: Option<&Path>, header: &str, out: &str) {
    if let Some(p) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(p) {
            use std::io::Write;
            let _ = writeln!(f, "=== {header} ===\n{out}\n");
        }
    }
}

/// Parse `N prompt tokens: X t/s` (pp) or `N tokens generated: X t/s` (tg).
fn parse_rate(out: &str, needle: &str) -> Option<f32> {
    for line in out.lines() {
        if let Some(idx) = line.find(needle) {
            // After the needle comes "X t/s" possibly with surrounding spaces.
            let rest = &line[idx + needle.len()..];
            for tok in rest.split_whitespace() {
                if let Ok(v) = tok.parse::<f32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn run_candle(args: &BenchArgs, log: Option<&Path>) -> Result<BenchResult> {
    let bin = args
        .bin
        .clone()
        .unwrap_or_else(|| "quantized-gemma4".to_string());
    let repo_root = repo_root();
    let bin_path = repo_root
        .join("target/release/examples")
        .join(&bin);
    if !bin_path.exists() {
        bail!(
            "candle binary not found: {} (build it with `cargo build --release -p candle-examples --example {}`)",
            bin_path.display(),
            bin
        );
    }
    let n_gpus = args.gpus.split(',').filter(|s| !s.is_empty()).count().max(1);

    let mut base_env = env_base(&args.gpus, &[]);
    base_env.push(("CANDLE_MMQ_TURBO_PORT".into(), "1".into()));
    for (k, v) in parse_kvs(&args.envs) {
        // CLI --env overrides defaults.
        base_env.retain(|(kk, _)| kk != &k);
        base_env.push((k, v));
    }

    let mut pp = BTreeMap::new();
    let mut sub_errors = BTreeMap::new();

    // Prefill runs.
    for &p_len in &args.prompt_lens {
        let prompt = "word ".repeat(p_len).trim_end().to_string();
        let mut runs = Vec::<f32>::new();
        let mut last_err = None;
        for rep in 0..args.reps {
            let (rc, out) = run_cmd(&bin_path, &[
                    "--model", args.model.to_str().unwrap(),
                    "--prompt", &prompt,
                    "--sample-len", "1",
                    "--temperature", "0",
                    "--n-gpus", &n_gpus.to_string(),
                ], &args.extra_args, &base_env, args.timeout);
            append_log(log, &format!("candle pp{p_len} rep{rep} rc={rc}"), &out);
            if rc != 0 {
                last_err = Some(format!("rc={rc} {}", tail_err(&out)));
                continue;
            }
            if let Some(v) = parse_rate(&out, "prompt tokens:") {
                runs.push(v);
            } else {
                last_err = Some(format!("rc=0 no pp rate parsed; tail={}", tail_err(&out)));
            }
        }
        if let Some(v) = runs.iter().cloned().fold(None::<f32>, |a, x| {
            Some(a.map_or(x, |prev| prev.max(x)))
        }) {
            pp.insert(format!("pp{p_len}"), v);
        } else {
            sub_errors.insert(format!("pp{p_len}"), last_err.unwrap_or("no runs".into()));
        }
    }

    // Decode run.
    let tg = if args.tg_len > 0 {
        let mut runs = Vec::new();
        let mut last_err = None;
        for rep in 0..args.reps {
            let (rc, out) = run_cmd(&bin_path, &[
                    "--model", args.model.to_str().unwrap(),
                    "--prompt", "hi",
                    "--sample-len", &args.tg_len.to_string(),
                    "--temperature", "0",
                    "--n-gpus", &n_gpus.to_string(),
                ], &args.extra_args, &base_env, args.timeout);
            append_log(log, &format!("candle tg rep{rep} rc={rc}"), &out);
            if rc != 0 {
                last_err = Some(format!("rc={rc} {}", tail_err(&out)));
                continue;
            }
            if let Some(v) = parse_rate(&out, "tokens generated:") {
                runs.push(v);
            } else {
                last_err = Some(format!("rc=0 no tg rate parsed; tail={}", tail_err(&out)));
            }
        }
        runs.iter().cloned().fold(None::<f32>, |a, x| Some(a.map_or(x, |p| p.max(x))))
            .or_else(|| {
                if let Some(e) = last_err {
                    sub_errors.insert(format!("tg{}", args.tg_len), e);
                }
                None
            })
    } else {
        None
    };

    Ok(BenchResult {
        backend: "candle".into(),
        bin,
        model: args.model.display().to_string(),
        gpus: args.gpus.clone(),
        reps: args.reps,
        wall_s: 0.0,
        env: base_env.into_iter().collect(),
        pp,
        tg,
        sub_errors,
        log: None,
    })
}

fn run_llama_bench(args: &BenchArgs, bin_dir: &str, log: Option<&Path>) -> Result<BenchResult> {
    let bin_path = PathBuf::from(bin_dir).join("llama-bench");
    if !bin_path.exists() {
        bail!("llama-bench not found at {}", bin_path.display());
    }
    let mut base_env = env_base(
        &args.gpus,
        &[
            bin_dir,
            &format!("{bin_dir}/../src"),
            &format!("{bin_dir}/../ggml/src"),
        ],
    );
    for (k, v) in parse_kvs(&args.envs) {
        base_env.retain(|(kk, _)| kk != &k);
        base_env.push((k, v));
    }

    let mut pp = BTreeMap::new();
    let mut sub_errors = BTreeMap::new();

    // Prefill: single llama-bench invocation per reps, many -p values.
    if !args.prompt_lens.is_empty() {
        let mut llama_args: Vec<String> = vec![
            "-m".into(),
            args.model.to_string_lossy().into(),
            "-ngl".into(),
            "99".into(),
            "-sm".into(),
            "layer".into(),
            "-r".into(),
            args.reps.to_string(),
            "-o".into(),
            "json".into(),
            "-n".into(),
            "0".into(),
        ];
        for &p in &args.prompt_lens {
            llama_args.push("-p".into());
            llama_args.push(p.to_string());
        }
        let argv: Vec<&str> = llama_args.iter().map(|s| s.as_str()).collect();
        let (rc, out) = run_cmd(&bin_path, &argv, &[], &base_env, args.timeout);
        append_log(log, &format!("llama-bench pp rc={rc}"), &out);
        if rc == 0 {
            if let Some(json) = extract_json_array(&out) {
                if let Ok(vals) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(arr) = vals.as_array() {
                        for rec in arr {
                            let n_gen = rec.get("n_gen").and_then(|v| v.as_i64()).unwrap_or(1);
                            let n_prompt = rec.get("n_prompt").and_then(|v| v.as_i64());
                            let avg = rec.get("avg_ts").and_then(|v| v.as_f64());
                            if n_gen == 0 {
                                if let (Some(p), Some(t)) = (n_prompt, avg) {
                                    pp.insert(format!("pp{p}"), t as f32);
                                }
                            }
                        }
                    }
                }
            }
            for &p_len in &args.prompt_lens {
                if !pp.contains_key(&format!("pp{p_len}")) {
                    sub_errors
                        .insert(format!("pp{p_len}"), format!("no pp{p_len} in json"));
                }
            }
        } else {
            for &p_len in &args.prompt_lens {
                sub_errors
                    .insert(format!("pp{p_len}"), format!("rc={rc} {}", tail_err(&out)));
            }
        }
    }

    // Decode.
    let tg = if args.tg_len > 0 {
        let reps_s = args.reps.to_string();
        let tg_s = args.tg_len.to_string();
        let argv: Vec<&str> = vec![
            "-m",
            args.model.to_str().unwrap(),
            "-ngl",
            "99",
            "-sm",
            "layer",
            "-r",
            &reps_s,
            "-o",
            "json",
            "-p",
            "0",
            "-n",
            &tg_s,
        ];
        let (rc, out) = run_cmd(&bin_path, &argv, &[], &base_env, args.timeout);
        append_log(log, &format!("llama-bench tg rc={rc}"), &out);
        if rc == 0 {
            if let Some(json) = extract_json_array(&out) {
                if let Ok(vals) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(arr) = vals.as_array() {
                        let mut tg = None;
                        for rec in arr {
                            if rec.get("n_gen").and_then(|v| v.as_i64())
                                == Some(args.tg_len as i64)
                            {
                                tg = rec.get("avg_ts").and_then(|v| v.as_f64()).map(|x| x as f32);
                            }
                        }
                        if tg.is_none() {
                            sub_errors.insert(format!("tg{}", args.tg_len), "no tg in json".into());
                        }
                        tg
                    } else { None }
                } else { None }
            } else { None }
        } else {
            sub_errors.insert(format!("tg{}", args.tg_len), format!("rc={rc} {}", tail_err(&out)));
            None
        }
    } else {
        None
    };

    Ok(BenchResult {
        backend: if bin_dir == TURBO_BIN_DIR {
            "turbo".into()
        } else {
            "llamacpp".into()
        },
        bin: "llama-bench".into(),
        model: args.model.display().to_string(),
        gpus: args.gpus.clone(),
        reps: args.reps,
        wall_s: 0.0,
        env: base_env.into_iter().collect(),
        pp,
        tg,
        sub_errors,
        log: None,
    })
}

fn extract_json_array(s: &str) -> Option<String> {
    // llama-bench prints a [...] array mixed with stderr chatter.
    let start = s.find("[\n").or_else(|| s.find("[ "))?;
    let mut depth = 0;
    let bytes = s.as_bytes();
    for i in start..bytes.len() {
        match bytes[i] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn tail_err(out: &str) -> String {
    out.lines()
        .rev()
        .find(|s| !s.trim().is_empty() && s.len() < 300)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn run_cmd(
    bin: &Path,
    args: &[&str],
    extra: &[String],
    env: &[(String, String)],
    timeout_s: u64,
) -> (i32, String) {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    for e in extra {
        cmd.arg(e);
    }
    for (k, v) in env {
        if v.is_empty() {
            cmd.env_remove(k);
        } else {
            cmd.env(k, v);
        }
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let _ = timeout_s; // TODO: enforce via std thread + child.kill if needed.
    match cmd.output() {
        Ok(o) => {
            let mut s = String::new();
            s.push_str(&String::from_utf8_lossy(&o.stdout));
            s.push('\n');
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            (o.status.code().unwrap_or(-1), s)
        }
        Err(e) => (-2, e.to_string()),
    }
}

fn repo_root() -> PathBuf {
    // Env override → $CARGO_MANIFEST_DIR ancestor → /artefact/candle.
    if let Ok(p) = std::env::var("CANDLE_ROOT") {
        return PathBuf::from(p);
    }
    if let Ok(mdir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(mdir);
        if let Some(parent) = p.parent() {
            return parent.to_path_buf();
        }
    }
    PathBuf::from("/artefact/candle")
}

pub fn print_md(r: &BenchResult) {
    println!("# {} / {} / {}",
             r.backend, r.bin, std::path::Path::new(&r.model).file_name().and_then(|s| s.to_str()).unwrap_or(&r.model));
    println!("- gpus={} reps={} wall={:.1}s", r.gpus, r.reps, r.wall_s);
    if !r.env.is_empty() {
        print!("- env: ");
        let mut first = true;
        for (k, v) in &r.env {
            if !first {
                print!(" ");
            }
            print!("{k}={v}");
            first = false;
        }
        println!();
    }
    println!("\n| bench | t/s |");
    println!("|-------|----:|");
    for (k, v) in &r.pp {
        println!("| {k} | {v:.1} |");
    }
    if let Some(tg) = r.tg {
        println!("| tg | {tg:.1} |");
    }
    if !r.sub_errors.is_empty() {
        println!("\n## errors");
        for (k, v) in &r.sub_errors {
            println!("- {k}: {v}");
        }
    }
}
