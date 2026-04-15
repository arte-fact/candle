//! `candle-bench fetch URL` — download a GGUF via curl with resume + size verify.
//!
//! Mirrors scripts/download-models.sh but in Rust so it integrates with the
//! rest of the tool.  We shell out to `curl` (fallback `wget`) rather than
//! depending on `hf-hub` — simpler build, works with any direct URL.

use crate::output::Envelope;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize)]
pub struct FetchResult {
    pub url: String,
    pub path: String,
    pub bytes: u64,
}

pub fn cmd(
    url: &str,
    dir: &Path,
    expected_bytes: Option<u64>,
    out_name: Option<&str>,
) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;

    let filename = out_name
        .map(|s| s.to_string())
        .or_else(|| url.rsplit('/').next().map(|s| s.to_string()))
        .filter(|s| !s.is_empty())
        .context("cannot derive filename from URL (pass --out-name)")?;
    let dest: PathBuf = dir.join(&filename);

    // Short-circuit if the local file already matches expected size.
    if let Some(want) = expected_bytes {
        if let Ok(m) = std::fs::metadata(&dest) {
            if m.len() == want {
                let r = FetchResult {
                    url: url.to_string(),
                    path: dest.display().to_string(),
                    bytes: want,
                };
                return Envelope::new("fetch", &r).print_json();
            }
        }
    }

    let tool = which_download_tool().context("neither curl nor wget found")?;
    eprintln!("[fetch] {filename}  <-  {url}  (tool: {tool:?})");

    let status = match tool {
        DownloadTool::Curl => Command::new("curl")
            .args([
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--retry-connrefused",
                "-C",
                "-",
                "--progress-bar",
            ])
            .arg("-o")
            .arg(&dest)
            .arg(url)
            .status()?,
        DownloadTool::Wget => Command::new("wget")
            .args(["-c", "--tries=5", "--retry-connrefused"])
            .arg("-O")
            .arg(&dest)
            .arg(url)
            .status()?,
    };

    if !status.success() {
        bail!("downloader exit {}", status);
    }
    let got = std::fs::metadata(&dest)?.len();
    if let Some(want) = expected_bytes {
        if got != want {
            bail!("size mismatch: got {got}, expected {want}");
        }
    }
    let r = FetchResult {
        url: url.to_string(),
        path: dest.display().to_string(),
        bytes: got,
    };
    Envelope::new("fetch", &r).print_json()
}

#[derive(Debug)]
enum DownloadTool {
    Curl,
    Wget,
}

fn which_download_tool() -> Option<DownloadTool> {
    fn has(cmd: &str) -> bool {
        Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    if has("curl") {
        Some(DownloadTool::Curl)
    } else if has("wget") {
        Some(DownloadTool::Wget)
    } else {
        None
    }
}
