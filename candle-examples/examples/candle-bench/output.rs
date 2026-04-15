//! Shared output formatting + JSON envelope.

use serde::Serialize;
use std::collections::BTreeMap;

/// Stable JSON envelope for every subcommand.
#[derive(Serialize)]
pub struct Envelope<T: Serialize> {
    pub tool: &'static str,
    pub version: u32,
    pub cmd: &'static str,
    pub ts: String,
    pub result: T,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub errors: BTreeMap<String, String>,
}

impl<T: Serialize> Envelope<T> {
    pub fn new(cmd: &'static str, result: T) -> Self {
        Self {
            tool: "candle-bench",
            version: 1,
            cmd,
            ts: chrono::Utc::now().to_rfc3339(),
            result,
            errors: BTreeMap::new(),
        }
    }

    pub fn print_json(&self) -> anyhow::Result<()> {
        println!("{}", serde_json::to_string_pretty(self)?);
        Ok(())
    }
}

/// Format a (possibly-empty) float as a table cell.
pub fn fmt_rate(v: Option<f32>) -> String {
    match v {
        Some(x) => format!("{:.1}", x),
        None => "—".to_string(),
    }
}

/// Format a ratio as `1.39×` or `—`.
pub fn fmt_ratio(a: Option<f32>, b: Option<f32>) -> String {
    match (a, b) {
        (Some(a), Some(b)) if b > 0.0 => format!("{:.2}×", a / b),
        _ => "—".to_string(),
    }
}
