//! `candle-bench kernels <HSACO>` — wrap llvm-readobj to extract kernel
//! symbol names and VGPR counts from a compiled HSACO / ELF.
//!
//! Locates llvm-readobj under a short list of likely ROCm paths.  Parses
//! lines of the form:
//!     Name: kernel_name.num_vgpr (73)
//! into `(kernel_name, num_vgpr)` pairs.  Also extracts private_seg_size
//! (scratch), numbered_sgpr, and a few other counters people look at
//! during kernel occupancy analysis.

use crate::output::Envelope;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_readobj() -> Option<PathBuf> {
    for p in [
        "/opt/rocm/llvm/bin/llvm-readobj",
        "/opt/rocm/bin/llvm-readobj",
        "/usr/bin/llvm-readobj",
        "/usr/local/bin/llvm-readobj",
    ] {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join("llvm-readobj"))
            .find(|p| p.exists())
    })
}

#[derive(Serialize, Clone, Default)]
pub struct KernelInfo {
    pub name: String,
    pub num_vgpr: Option<u32>,
    pub numbered_sgpr: Option<u32>,
    pub num_agpr: Option<u32>,
    pub private_seg_size: Option<u32>,
}

pub fn list(hsaco: &Path, filter: Option<&str>) -> Result<Vec<KernelInfo>> {
    let readobj = find_readobj().context("llvm-readobj not found in ROCm / PATH")?;
    let out = Command::new(&readobj)
        .arg("--symbols")
        .arg(hsaco)
        .output()
        .with_context(|| format!("running {}", readobj.display()))?;
    if !out.status.success() {
        bail!(
            "llvm-readobj failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Lines like:  "    Name: mul_mat_q8_0_turbo_moe_x32_unchecked.num_vgpr (73)"
    // We collect per-kernel fields into a map, then flatten.
    let mut by_name: BTreeMap<String, KernelInfo> = BTreeMap::new();
    for line in stdout.lines() {
        let l = line.trim();
        let rest = match l.strip_prefix("Name: ") {
            Some(r) => r,
            None => continue,
        };
        // "kernel.field (value)" or "kernel (value)" — only care about dot variants.
        let (body, _value_suffix) = match rest.rsplit_once(" (") {
            Some(v) => v,
            None => continue,
        };
        let (kname, field) = match body.rsplit_once('.') {
            Some(v) => v,
            None => continue,
        };
        if let Some(pat) = filter {
            if !kname.contains(pat) {
                continue;
            }
        }
        // Parse the trailing "...(N)" where N is an integer index into
        // the SymTab string table (NOT the value we want).  The counters
        // llvm-readobj exposes for AMDGPU kernels come from AMDGPU MsgPack
        // metadata and land in the symbol name itself, in a separate
        // pass we'd have to decode.  So: record the presence of each
        // field even if we can't give the scalar value from --symbols
        // alone.  We still use --notes to harvest scalars below.
        let _ = field; // Recorded implicitly via entry-or-default below.
        by_name
            .entry(kname.to_string())
            .or_insert_with(|| KernelInfo {
                name: kname.to_string(),
                ..Default::default()
            });
    }

    // Second pass: parse the MsgPack-ish --notes output to get real VGPR /
    // SGPR / AGPR / scratch sizes.  llvm-readobj --notes prints a block
    // with `.num_vgpr:    73` etc.  If the tool lacks --notes, we silently
    // leave the fields None.
    if let Ok(notes_out) = Command::new(&readobj).arg("--notes").arg(hsaco).output() {
        if notes_out.status.success() {
            let notes = String::from_utf8_lossy(&notes_out.stdout);
            let mut cur: Option<String> = None;
            for raw in notes.lines() {
                let line = raw.trim();
                if let Some(rest) = line.strip_prefix("- Name:") {
                    cur = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix(".name:") {
                    cur = Some(rest.trim().trim_matches('\'').to_string());
                } else if let Some(ref name) = cur {
                    if let Some(k) = by_name.get_mut(name) {
                        if let Some(v) = line
                            .strip_prefix(".vgpr_count:")
                            .and_then(|s| s.trim().parse().ok())
                        {
                            k.num_vgpr = Some(v);
                        } else if let Some(v) = line
                            .strip_prefix(".sgpr_count:")
                            .and_then(|s| s.trim().parse().ok())
                        {
                            k.numbered_sgpr = Some(v);
                        } else if let Some(v) = line
                            .strip_prefix(".agpr_count:")
                            .and_then(|s| s.trim().parse().ok())
                        {
                            k.num_agpr = Some(v);
                        } else if let Some(v) = line
                            .strip_prefix(".private_segment_fixed_size:")
                            .and_then(|s| s.trim().parse().ok())
                        {
                            k.private_seg_size = Some(v);
                        }
                    }
                }
            }
        }
    }

    let mut out: Vec<KernelInfo> = by_name.into_values().collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn cmd(hsaco: &Path, filter: Option<&str>, json: bool) -> Result<()> {
    let kernels = list(hsaco, filter)?;
    if json {
        Envelope::new("kernels", &kernels).print_json()
    } else {
        println!("| kernel | vgpr | sgpr | agpr | scratch |");
        println!("|--------|-----:|-----:|-----:|--------:|");
        for k in &kernels {
            println!(
                "| {} | {} | {} | {} | {} |",
                k.name,
                k.num_vgpr.map(|v| v.to_string()).unwrap_or("?".into()),
                k.numbered_sgpr.map(|v| v.to_string()).unwrap_or("?".into()),
                k.num_agpr.map(|v| v.to_string()).unwrap_or("?".into()),
                k.private_seg_size
                    .map(|v| v.to_string())
                    .unwrap_or("?".into()),
            );
        }
        println!("\n{} kernel(s) total", kernels.len());
        Ok(())
    }
}
