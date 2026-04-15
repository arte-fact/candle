//! Phase A3 — EPLB (Expert Parallelism Load Balancer) diagnostic stats.
//!
//! Aphrodite-engine ships `aphrodite/distributed/eplb/eplb_state.py` with
//! full logical→physical expert rebalancing across EP ranks.  For candle
//! on single-node MoE the rebalancing machinery isn't useful yet — all
//! experts fit in 4×16 GB and there's only one EP group.  What IS useful
//! is the **observation** piece: count which experts each decode step
//! routes to, so we can see whether a prompt exercises a wide expert
//! spread or keeps hammering the same few.
//!
//! This is intentionally lightweight:
//!   * atomic per-expert counter array (at most `MAX_EXPERTS = 512`),
//!   * `observe(&[u32])` called from `MoeExperts::forward*` right after
//!     topk_ids lands on CPU,
//!   * `snapshot()` returns a `Vec<u64>` for dumping.
//!
//! Env-gated — zero cost when disabled:
//!   * `CANDLE_EPLB_DUMP=<path>` : append one JSON line per decode step,
//!     `{step, layer, counts}`.
//!   * `CANDLE_EPLB_PRINT=1`     : print hottest/coldest summary on drop
//!     (i.e. at process exit).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// Upper bound on `n_experts` across all supported GGUFs today:
///   Gemma-4-26B-A4B = 128,  Qwen3-Coder-30B-A3B = 128,
///   Qwen3-Coder-Next = 512 (header cap).  512 is a comfortable ceiling.
const MAX_EXPERTS: usize = 512;

static ENABLED: OnceLock<bool> = OnceLock::new();
static DUMP_PATH: OnceLock<Option<String>> = OnceLock::new();

struct ExpertLoadStats {
    counts: [AtomicU64; MAX_EXPERTS],
    step: AtomicU64,
    print_on_drop: AtomicBool,
}

impl ExpertLoadStats {
    const fn zero() -> Self {
        // Can't `[AtomicU64::new(0); 512]` directly — AtomicU64 isn't Copy.
        // Use a static-initialised array via a helper const.
        // (This is the trick `ArrayVec` uses; see also unstable
        // `array::from_fn`.)
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            counts: [INIT; MAX_EXPERTS],
            step: AtomicU64::new(0),
            print_on_drop: AtomicBool::new(false),
        }
    }

    fn observe(&self, topk_ids: &[u32]) {
        for &id in topk_ids {
            let i = id as usize;
            if i < MAX_EXPERTS {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.step.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> Vec<u64> {
        self.counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }
}

impl Drop for ExpertLoadStats {
    fn drop(&mut self) {
        if !self.print_on_drop.load(Ordering::Relaxed) {
            return;
        }
        let counts = self.snapshot();
        // Find hottest / coldest and total.
        let total: u64 = counts.iter().sum();
        let active_n = counts.iter().filter(|&&c| c > 0).count();
        let mut idx: Vec<usize> = (0..counts.len()).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
        eprintln!(
            "[eplb] total_routes={} active_experts={}/{} step={}",
            total,
            active_n,
            MAX_EXPERTS,
            self.step.load(Ordering::Relaxed),
        );
        eprintln!("[eplb] top-8 hot:");
        for &i in idx.iter().take(8) {
            let c = counts[i];
            if c == 0 {
                break;
            }
            let pct = 100.0 * c as f64 / total.max(1) as f64;
            eprintln!("  expert {:>3}  count={:>7}  {:5.1}%", i, c, pct);
        }
    }
}

static STATS: OnceLock<ExpertLoadStats> = OnceLock::new();

fn stats() -> &'static ExpertLoadStats {
    STATS.get_or_init(|| {
        let s = ExpertLoadStats::zero();
        s.print_on_drop.store(
            std::env::var("CANDLE_EPLB_PRINT").is_ok(),
            Ordering::Relaxed,
        );
        s
    })
}

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("CANDLE_EPLB_PRINT").is_ok()
            || std::env::var("CANDLE_EPLB_DUMP").is_ok()
    })
}

fn dump_path() -> Option<&'static str> {
    DUMP_PATH
        .get_or_init(|| std::env::var("CANDLE_EPLB_DUMP").ok())
        .as_deref()
}

/// Record that this decode step routed to `topk_ids` (flat list of
/// expert ids).  No-op when disabled.  Call from `MoeExperts::forward*`
/// after the topk argmax materialises on CPU.
pub fn observe(topk_ids: &[u32], layer: usize) {
    if !enabled() {
        return;
    }
    stats().observe(topk_ids);
    if let Some(path) = dump_path() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let step = stats().step.load(Ordering::Relaxed);
            let ids_json: String = topk_ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let _ = writeln!(
                f,
                r#"{{"step":{},"layer":{},"topk_ids":[{}]}}"#,
                step, layer, ids_json
            );
        }
    }
}

/// Return the current per-expert count snapshot (length `MAX_EXPERTS`).
/// Useful for tests / external bench harnesses.
pub fn snapshot() -> Vec<u64> {
    stats().snapshot()
}

/// Print the hottest-8 summary to stderr.  Statics do not run `Drop`
/// at process exit in Rust, so callers must invoke this explicitly
/// (e.g. at end of `generate()`) when `CANDLE_EPLB_PRINT=1`.
pub fn print_summary() {
    if !enabled() {
        return;
    }
    let counts = snapshot();
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return;
    }
    let active_n = counts.iter().filter(|&&c| c > 0).count();
    let mut idx: Vec<usize> = (0..counts.len()).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    eprintln!(
        "[eplb] total_routes={} active_experts={}/{} steps={}",
        total,
        active_n,
        MAX_EXPERTS,
        stats().step.load(Ordering::Relaxed),
    );
    eprintln!("[eplb] top-8 hot:");
    for &i in idx.iter().take(8) {
        let c = counts[i];
        if c == 0 {
            break;
        }
        let pct = 100.0 * c as f64 / total.max(1) as f64;
        eprintln!("  expert {:>3}  count={:>7}  {:5.1}%", i, c, pct);
    }
}
