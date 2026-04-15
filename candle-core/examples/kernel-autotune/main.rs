//! Phase V1 — offline autotune harness for Q8_0 MMVQ kernel variants.
//!
//! For each `(dtype, b_size, ncols_x, nrows_x)` shape in the input JSON,
//! bench every compiled variant of `mul_mat_vec_q8_0_q8_1_cuda{1..8}`
//! plus the gfx906 warp-coop kernel, pick the fastest correct one, and
//! emit a JSON config that `candle-core::quantized::autotune` loads at
//! runtime.
//!
//! Usage:
//!   HIP_VISIBLE_DEVICES=0 \
//!   LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib \
//!   cargo run --release --features hip --example kernel-autotune -- \
//!       --shapes candle-examples/examples/kernel-autotune/bench_shapes.json \
//!       --out /tmp/v1_out.json

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_core::hip_backend::WrapErr;
use candle_core::hip_backend::hipdarc::{self, driver::LaunchConfig};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

const WARP_SIZE: u32 = 64;

// -- Shape / config types (mirror candle-core/src/quantized/autotune.rs) --

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Shape {
    /// Batch size (number of columns of Y). 1 for pure decode; up to 8 for
    /// chunked MMVQ at small prefill / MoE per-expert.
    b_size: usize,
    /// K — weight columns (hidden dim).
    ncols_x: usize,
    /// N — weight rows (output dim).
    nrows_x: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantResult {
    variant: String,
    time_us: f64,
    correct: bool,
    max_rel_err: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShapeBench {
    shape: Shape,
    variants: Vec<VariantResult>,
    winner: String,
    winner_time_us: f64,
    speedup_vs_cuda1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AutotuneOutput {
    schema_version: u32,
    target: String,
    generated_at: String,
    dtype: String,
    shapes: Vec<ShapeBench>,
}

// -- Harness --

#[derive(Parser, Debug)]
#[command(about = "Offline autotune harness for Q8_0 MMVQ kernel variants")]
struct Args {
    /// JSON file with an array of `Shape` objects.
    #[arg(long)]
    shapes: String,

    /// Output JSON path.
    #[arg(long)]
    out: String,

    /// Which kernel family to autotune.
    ///   `q8_0_mmvq` = Q8_0 decode MMVQ (V1; negative-result baseline).
    ///   `q4_0_mmq`  = Q4_0 prefill MMQ — tile_n ∈ {8, 16, 32}.
    ///   `q4_1_mmq`  = Q4_1 prefill MMQ — tile_n ∈ {8, 16, 32, 64} × {v2f, v2d, legacy}.
    ///   `q8_0_mmq`  = Q8_0 prefill MMQ — tile_n ∈ {8, 16, 32}.
    #[arg(long, default_value = "q8_0_mmvq")]
    kernel: String,

    /// Warmup iterations before each timed run.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Timed iterations per variant.
    #[arg(long, default_value_t = 20)]
    reps: usize,

    /// GPU ordinal (forwarded to `Device::new_hip`; respect `HIP_VISIBLE_DEVICES`).
    #[arg(long, default_value_t = 0)]
    gpu: usize,

    /// Correctness tolerance (relative error).
    #[arg(long, default_value_t = 5e-3)]
    tol: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let shapes: Vec<Shape> = {
        let raw = std::fs::read_to_string(&args.shapes)
            .with_context(|| format!("reading shapes file {}", args.shapes))?;
        serde_json::from_str(&raw).context("parsing shapes JSON")?
    };

    let device = Device::new_hip(args.gpu)?;
    let dev_hip = match &device {
        Device::Hip(d) => d.clone(),
        _ => unreachable!("new_hip should return a Hip device"),
    };
    println!("HIP device: {:?}", device.location());
    println!("kernel: {}", args.kernel);

    let mut out_shapes: Vec<ShapeBench> = Vec::with_capacity(shapes.len());

    for (i, shape) in shapes.iter().enumerate() {
        println!(
            "\n[{}/{}] shape b_size={} K={} N={}",
            i + 1, shapes.len(), shape.b_size, shape.ncols_x, shape.nrows_x,
        );
        let result = match args.kernel.as_str() {
            "q8_0_mmvq" => bench_shape(*shape, &device, &dev_hip, args.warmup, args.reps, args.tol),
            "q4_0_mmq" => bench_mmq(*shape, GgmlDType::Q4_0, &device, &dev_hip, args.warmup, args.reps, args.tol),
            "q4_1_mmq" => bench_mmq(*shape, GgmlDType::Q4_1, &device, &dev_hip, args.warmup, args.reps, args.tol),
            "q8_0_mmq" => bench_mmq(*shape, GgmlDType::Q8_0, &device, &dev_hip, args.warmup, args.reps, args.tol),
            other => anyhow::bail!("unknown --kernel {other:?}"),
        };
        match result {
            Ok(b) => {
                println!(
                    "  → winner: {:<40} {:>7.1} µs/call ({:.2}× ref)",
                    b.winner, b.winner_time_us, b.speedup_vs_cuda1,
                );
                out_shapes.push(b);
            }
            Err(e) => {
                eprintln!("  bench failed: {e}");
            }
        }
    }

    let out = AutotuneOutput {
        schema_version: 1,
        target: detect_target(&dev_hip).unwrap_or_else(|| "gfx906".to_string()),
        generated_at: chrono_like_ts(),
        dtype: args.kernel.clone(),
        shapes: out_shapes,
    };
    let json = serde_json::to_string_pretty(&out).context("encoding output")?;
    std::fs::write(&args.out, &json).with_context(|| format!("writing {}", args.out))?;
    println!("\nwrote {} ({} bytes)", args.out, json.len());
    Ok(())
}

fn detect_target(_: &candle_core::hip_backend::HipDevice) -> Option<String> {
    Some("gfx906".to_string())
}

fn chrono_like_ts() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{now}")
}

fn bench_shape(
    shape: Shape,
    device: &Device,
    dev_hip: &candle_core::hip_backend::HipDevice,
    warmup: usize,
    reps: usize,
    tol: f64,
) -> Result<ShapeBench> {
    let k = shape.ncols_x;
    let n = shape.nrows_x;
    let b = shape.b_size;

    // Allocate random f32 weight (N, K), quantize to Q8_0.
    let weight_data: Vec<f32> = (0..n * k)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), device)?.reshape((n, k))?;
    let qweight = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
    // Copy Q8_0 bytes to a padded device buffer (see hip::PaddedHipSlice for
    // production layout; here we use a flat slice — the kernel only reads
    // `nrows_x × blocks_per_row` blocks so padding isn't needed for bench).
    let q_host = qweight.data()?.to_vec();
    let mut q_dev = unsafe { dev_hip.alloc::<u8>(q_host.len())? };
    unsafe {
        let rc = hipdarc::sys::hipMemcpy(
            q_dev.device_ptr() as *mut std::ffi::c_void,
            q_host.as_ptr() as *const std::ffi::c_void,
            q_host.len(),
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
        if rc != hipdarc::sys::hipError_t::hipSuccess {
            anyhow::bail!("hipMemcpy weight: {:?}", rc);
        }
    }

    // Y input: random f32 (b, K), quantize to Q8_1.
    let y_data: Vec<f32> = (0..b * k)
        .map(|i| ((i.wrapping_mul(17).wrapping_add(11)) % 31) as f32 / 15.0 - 1.0)
        .collect();
    let y_t = Tensor::new(y_data.as_slice(), device)?.reshape((b, k))?;
    let (y_st, _y_l) = y_t.storage_and_layout();
    let y_view_f32 = match &*y_st {
        candle_core::Storage::Hip(s) => {
            let sl = s.as_hip_slice::<f32>()?;
            sl.slice(0..sl.len())
        }
        _ => anyhow::bail!("y not on HIP"),
    };
    use candle_core::quantized::hip::{quantize_q8_1, pad, MATRIX_ROW_PADDING};
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    const Q8_1_BLOCK_BYTES: usize = 36;
    const QK8_1: usize = 32;
    let y_q8_bytes = b * k_padded * Q8_1_BLOCK_BYTES / QK8_1;
    let mut y_q8 = unsafe { dev_hip.alloc::<u8>(y_q8_bytes)? };
    quantize_q8_1(&y_view_f32, &mut y_q8, k, b, dev_hip)?;
    drop(y_st);

    let y_view_q8 = y_q8.slice(0..y_q8.len());
    let q_view = q_dev.slice(0..q_dev.len());

    // Dst buffer — (N, b) f32.
    let mut dst = unsafe { dev_hip.alloc::<f32>(n * b)? };
    let dst_view = dst.slice(0..dst.len());

    // Per-b_size variants.
    //
    // IMPORTANT: the `mul_mat_vec_q<N, ...>` template has `N = ncols_y`
    // baked into the kernel — it reads `ncols_y` columns of the Q8_1
    // input and writes `ncols_y × nrows` outputs. Launching `cuda5`
    // with a b_size=1 input reads 5 columns of Y (OOB), writes 5
    // rows of dst (OOB) → device-state corruption that surfaces as
    // `hipErrorNotReady` on the NEXT shape.
    //
    // Therefore the variant set for autotune is NOT {cuda1..cuda8}
    // across all b_sizes — it's ONE variant per b_size:
    //   b_size=1 → cuda1 (warp-coop, hand-tuned)
    //   b_size=N (N=2..8) → cudaN (template, ncols_y=N)
    //
    // This means Q8_0 MMVQ has no autotune config space per shape: for
    // any given (b_size, K, N) there is exactly one correct kernel to
    // dispatch. The harness still records the time for that single
    // variant — useful as a measurement baseline and to confirm the
    // current dispatcher picks the right kernel.
    let only_variant = match b {
        1 => "mul_mat_vec_q8_0_q8_1_cuda1",
        2 => "mul_mat_vec_q8_0_q8_1_cuda2",
        3 => "mul_mat_vec_q8_0_q8_1_cuda3",
        4 => "mul_mat_vec_q8_0_q8_1_cuda4",
        5 => "mul_mat_vec_q8_0_q8_1_cuda5",
        6 => "mul_mat_vec_q8_0_q8_1_cuda6",
        7 => "mul_mat_vec_q8_0_q8_1_cuda7",
        8 => "mul_mat_vec_q8_0_q8_1_cuda8",
        _ => anyhow::bail!("b_size {b} out of 1..=8"),
    };
    let variants: Vec<&str> = vec![only_variant];
    let ref_variant = only_variant;

    // Helper: launch a variant by name. Returns elapsed µs per iter (median).
    let launch_and_time = |name: &str, timed_iters: usize, warmup_iters: usize|
        -> Result<f64>
    {
        let cfg = launch_cfg_for(name, b, n as u32);
        let func = dev_hip
            .get_or_load_func(name, &candle_hip_kernels::QUANTIZED)?;
        let ncols_x_i = k as i32;
        let nrows_x_i = n as i32;
        let nrows_y_i = k_padded as i32;
        let nrows_dst_i = n as i32;
        // Warmup.
        for _ in 0..warmup_iters {
            let mut bld = func.builder();
            bld.arg(&q_view);
            bld.arg(&y_view_q8);
            bld.arg(&dst_view);
            bld.arg(&ncols_x_i);
            bld.arg(&nrows_x_i);
            bld.arg(&nrows_y_i);
            bld.arg(&nrows_dst_i);
            unsafe { bld.launch(cfg) }.w()?;
        }
        let _ = dev_hip.stream().synchronize();
        // Timed: one big batch, divide.
        let t0 = std::time::Instant::now();
        for _ in 0..timed_iters {
            let mut bld = func.builder();
            bld.arg(&q_view);
            bld.arg(&y_view_q8);
            bld.arg(&dst_view);
            bld.arg(&ncols_x_i);
            bld.arg(&nrows_x_i);
            bld.arg(&nrows_y_i);
            bld.arg(&nrows_dst_i);
            unsafe { bld.launch(cfg) }.w()?;
        }
        let _ = dev_hip.stream().synchronize();
        let dt = t0.elapsed().as_micros() as f64 / timed_iters as f64;
        Ok(dt)
    };

    // Helper: pull a patch of dst to host.
    let snapshot_dst = || -> Result<Vec<f32>> {
        let mut host = vec![0f32; n * b];
        unsafe {
            let rc = hipdarc::sys::hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dst.device_ptr() as *const std::ffi::c_void,
                host.len() * 4,
                hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
            );
            if rc != hipdarc::sys::hipError_t::hipSuccess {
                anyhow::bail!("dst DtoH: {:?}", rc);
            }
        }
        Ok(host)
    };

    // Compute reference output.
    let _ = launch_and_time(ref_variant, 1, 0)?;
    let _ = dev_hip.stream().synchronize();
    let reference = snapshot_dst()?;

    let mut variant_results: Vec<VariantResult> = Vec::with_capacity(variants.len());
    for name in &variants {
        // Launch once, snapshot for correctness.
        let _ = launch_and_time(name, 1, 0)?;
        let _ = dev_hip.stream().synchronize();
        let got = match snapshot_dst() {
            Ok(v) => v,
            Err(e) => {
                variant_results.push(VariantResult {
                    variant: name.to_string(),
                    time_us: f64::INFINITY,
                    correct: false,
                    max_rel_err: f64::INFINITY,
                });
                eprintln!("    {name}: snapshot failed: {e}");
                continue;
            }
        };
        let max_rel = max_rel_err(&reference, &got);
        let correct = max_rel <= tol;

        // Time the variant.
        let t_us = match launch_and_time(name, reps, warmup) {
            Ok(t) => t,
            Err(e) => {
                variant_results.push(VariantResult {
                    variant: name.to_string(),
                    time_us: f64::INFINITY,
                    correct,
                    max_rel_err: max_rel,
                });
                eprintln!("    {name}: timing failed: {e}");
                continue;
            }
        };
        println!(
            "    {:<34} {:>7.1} µs  max_rel_err={:>.2e}  {}",
            name, t_us, max_rel, if correct { "ok" } else { "WRONG" }
        );
        variant_results.push(VariantResult {
            variant: name.to_string(),
            time_us: t_us,
            correct,
            max_rel_err: max_rel,
        });
    }

    // Pick winner: fastest correct variant.
    let winner = variant_results
        .iter()
        .filter(|v| v.correct)
        .min_by(|a, b| a.time_us.partial_cmp(&b.time_us).unwrap())
        .cloned()
        .unwrap_or_else(|| VariantResult {
            variant: "mul_mat_vec_q8_0_q8_1_cuda1".to_string(),
            time_us: f64::INFINITY,
            correct: false,
            max_rel_err: f64::INFINITY,
        });
    let cuda1_time = variant_results
        .iter()
        .find(|v| v.variant == "mul_mat_vec_q8_0_q8_1_cuda1")
        .map(|v| v.time_us)
        .unwrap_or(f64::NAN);
    let speedup = if cuda1_time.is_finite() && winner.time_us > 0.0 {
        cuda1_time / winner.time_us
    } else {
        1.0
    };

    Ok(ShapeBench {
        shape,
        winner: winner.variant.clone(),
        winner_time_us: winner.time_us,
        speedup_vs_cuda1: speedup,
        variants: variant_results,
    })
}

fn launch_cfg_for(name: &str, b_size: usize, nrows: u32) -> LaunchConfig {
    // Mirrors candle-core/src/quantized/hip.rs:398-423 exactly.
    // cuda1 is gfx906 warp-coop: grid=(ceil(nrows/2), 1, 1), block=(64, 1, 1).
    // cuda2..cuda8 are templates: grid depends on b_size (see hip.rs:413-418).
    let warp_coop = name == "mul_mat_vec_q8_0_q8_1_cuda1" && b_size == 1;
    if warp_coop {
        LaunchConfig {
            grid_dim: (nrows.div_ceil(2), 1, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        let (nblocks, nwarps) = match b_size {
            1 => (nrows, 4),
            2..=4 => (nrows.div_ceil(2), 4),
            5..=8 => (nrows.div_ceil(2), 2),
            _ => unreachable!(),
        };
        LaunchConfig {
            grid_dim: (nblocks, 1, 1),
            block_dim: (WARP_SIZE, nwarps, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Q4_0 / Q4_1 / Q8_0 MMQ (prefill) per-tile_n autotune. Mirrors the
/// dispatcher at `candle-core/src/quantized/hip.rs:743-927`.
///
/// Variants per dtype come directly from that dispatcher's match block:
///   Q4_0 → v2 (tile8), v2_tile16, v2f_tile32, v2_tile32 (legacy)
///   Q4_1 → v2 (tile8), v2_tile16, v2f_tile32, v2d_tile32, v2_tile32, v2_tile64
///   Q8_0 → v2 (tile8), v2_tile16, v2f_tile32, v2_tile32 (legacy)
///
/// For MMQ, `b_size` is interpreted as `total_b` (the M dimension / query
/// count per forward, i.e. prompt-len × batch for prefill). Launch config
/// is `grid=(ceil(nrows/TILE_M=64), ceil(total_b/tile_n), 1)`,
/// `block=(64, 1, 1)`, `shared=0`. Y padded to `ceil(total_b/tile_n) * tile_n`.
fn bench_mmq(
    shape: Shape,
    dtype: GgmlDType,
    device: &Device,
    dev_hip: &candle_core::hip_backend::HipDevice,
    warmup: usize,
    reps: usize,
    tol: f64,
) -> Result<ShapeBench> {
    let k = shape.ncols_x;
    let n = shape.nrows_x;
    let total_b = shape.b_size;   // M dim for MMQ

    // Variants: (kernel_name, tile_n).
    let variants: Vec<(&str, usize)> = match dtype {
        GgmlDType::Q4_0 => vec![
            ("mul_mat_q4_0_gfx906_v2",         8),
            ("mul_mat_q4_0_gfx906_v2_tile16",  16),
            ("mul_mat_q4_0_gfx906_v2f_tile32", 32),
            ("mul_mat_q4_0_gfx906_v2_tile32",  32),  // legacy
        ],
        GgmlDType::Q4_1 => vec![
            ("mul_mat_q4_1_gfx906_v2",         8),
            ("mul_mat_q4_1_gfx906_v2_tile16",  16),
            ("mul_mat_q4_1_gfx906_v2f_tile32", 32),
            ("mul_mat_q4_1_gfx906_v2d_tile32", 32),
            ("mul_mat_q4_1_gfx906_v2_tile32",  32),
            ("mul_mat_q4_1_gfx906_v2_tile64",  64),
        ],
        GgmlDType::Q8_0 => vec![
            ("mul_mat_q8_0_gfx906_v2",         8),
            ("mul_mat_q8_0_gfx906_v2_tile16",  16),
            ("mul_mat_q8_0_gfx906_v2f_tile32", 32),
            ("mul_mat_q8_0_gfx906_v2_tile32",  32),
        ],
        _ => anyhow::bail!("unsupported MMQ dtype {dtype:?}"),
    };

    // Weight: (N, K) f32 → quantize to `dtype`.
    let weight_data: Vec<f32> = (0..n * k)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), device)?.reshape((n, k))?;
    let qweight = QTensor::quantize(&weight, dtype)?;
    let q_host = qweight.data()?.to_vec();
    let q_dev = unsafe { dev_hip.alloc::<u8>(q_host.len())? };
    unsafe {
        let rc = hipdarc::sys::hipMemcpy(
            q_dev.device_ptr() as *mut std::ffi::c_void,
            q_host.as_ptr() as *const std::ffi::c_void,
            q_host.len(),
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
        if rc != hipdarc::sys::hipError_t::hipSuccess {
            anyhow::bail!("hipMemcpy weight: {:?}", rc);
        }
    }
    let q_view = q_dev.slice(0..q_dev.len());

    // Y input: (total_b, K) f32.
    let y_data: Vec<f32> = (0..total_b * k)
        .map(|i| ((i.wrapping_mul(17).wrapping_add(11)) % 31) as f32 / 15.0 - 1.0)
        .collect();
    let y_t = Tensor::new(y_data.as_slice(), device)?.reshape((total_b, k))?;
    let (y_st, _y_l) = y_t.storage_and_layout();
    let y_view_f32 = match &*y_st {
        candle_core::Storage::Hip(s) => {
            let sl = s.as_hip_slice::<f32>()?;
            sl.slice(0..sl.len())
        }
        _ => anyhow::bail!("y not on HIP"),
    };
    use candle_core::quantized::hip::{quantize_q8_1, pad, MATRIX_ROW_PADDING};
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    const Q8_1_BLOCK_BYTES: usize = 36;
    const QK8_1: usize = 32;

    // Per-variant tile_n needs its OWN Y-padded buffer; we allocate the
    // largest (tile_n=64 → total_b_padded=ceil(total_b/64)*64) once and
    // reuse by writing only the in-range rows (padding rows stay zero
    // since we alloc_zeros for the v2f safety).
    let max_tile_n = variants.iter().map(|(_, t)| *t).max().unwrap();
    let total_b_padded_max = total_b.div_ceil(max_tile_n) * max_tile_n;
    let y_q8_bytes = total_b_padded_max * k_padded * Q8_1_BLOCK_BYTES / QK8_1;
    // alloc_zeros so v2f OOB reads see zeros.
    let mut y_q8 = dev_hip.alloc_zeros::<u8>(y_q8_bytes)?;
    quantize_q8_1(&y_view_f32, &mut y_q8, k, total_b, dev_hip)?;
    drop(y_st);
    let y_view_q8 = y_q8.slice(0..y_q8.len());

    // Dst: (total_b, N) f32.
    let dst = unsafe { dev_hip.alloc::<f32>(total_b * n)? };
    let dst_view = dst.slice(0..dst.len());

    const TILE_M: u32 = 64;

    let launch = |name: &str, tile_n: usize, timed: usize, warm: usize| -> Result<f64> {
        let func = dev_hip.get_or_load_func(name, &candle_hip_kernels::QUANTIZED)?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(TILE_M),
                (total_b as u32).div_ceil(tile_n as u32),
                1,
            ),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let ncols_x = k as i32;
        let nrows_x = n as i32;
        let ncols_y = total_b as i32;
        let nrows_y = k_padded as i32;
        let nrows_dst = n as i32;
        for _ in 0..warm {
            let mut bld = func.builder();
            bld.arg(&q_view);
            bld.arg(&y_view_q8);
            bld.arg(&dst_view);
            bld.arg(&ncols_x);
            bld.arg(&nrows_x);
            bld.arg(&ncols_y);
            bld.arg(&nrows_y);
            bld.arg(&nrows_dst);
            unsafe { bld.launch(cfg) }.w()?;
        }
        let _ = dev_hip.stream().synchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..timed {
            let mut bld = func.builder();
            bld.arg(&q_view);
            bld.arg(&y_view_q8);
            bld.arg(&dst_view);
            bld.arg(&ncols_x);
            bld.arg(&nrows_x);
            bld.arg(&ncols_y);
            bld.arg(&nrows_y);
            bld.arg(&nrows_dst);
            unsafe { bld.launch(cfg) }.w()?;
        }
        let _ = dev_hip.stream().synchronize();
        Ok(t0.elapsed().as_micros() as f64 / timed as f64)
    };

    let snapshot = || -> Result<Vec<f32>> {
        let mut host = vec![0f32; total_b * n];
        unsafe {
            let rc = hipdarc::sys::hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dst.device_ptr() as *const std::ffi::c_void,
                host.len() * 4,
                hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
            );
            if rc != hipdarc::sys::hipError_t::hipSuccess {
                anyhow::bail!("dst DtoH: {:?}", rc);
            }
        }
        Ok(host)
    };

    // Reference = first variant.
    let ref_name = variants[0].0;
    let ref_tile = variants[0].1;
    let _ = launch(ref_name, ref_tile, 1, 0)?;
    let _ = dev_hip.stream().synchronize();
    let reference = snapshot()?;

    let mut results: Vec<VariantResult> = Vec::with_capacity(variants.len());
    for (name, tile_n) in &variants {
        let _ = launch(name, *tile_n, 1, 0)?;
        let _ = dev_hip.stream().synchronize();
        let got = match snapshot() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("    {name}: snapshot failed: {e}");
                results.push(VariantResult {
                    variant: name.to_string(),
                    time_us: f64::INFINITY,
                    correct: false,
                    max_rel_err: f64::INFINITY,
                });
                continue;
            }
        };
        let max_rel = max_rel_err(&reference, &got);
        let correct = max_rel <= tol;
        let t_us = match launch(name, *tile_n, reps, warmup) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("    {name}: timing failed: {e}");
                results.push(VariantResult {
                    variant: name.to_string(),
                    time_us: f64::INFINITY,
                    correct,
                    max_rel_err: max_rel,
                });
                continue;
            }
        };
        println!(
            "    {:<40} {:>7.1} µs  max_rel_err={:>.2e}  {}",
            name, t_us, max_rel, if correct { "ok" } else { "WRONG" }
        );
        results.push(VariantResult {
            variant: name.to_string(),
            time_us: t_us,
            correct,
            max_rel_err: max_rel,
        });
    }

    let winner = results
        .iter()
        .filter(|v| v.correct)
        .min_by(|a, b| a.time_us.partial_cmp(&b.time_us).unwrap())
        .cloned()
        .unwrap_or_else(|| VariantResult {
            variant: ref_name.to_string(),
            time_us: f64::INFINITY,
            correct: false,
            max_rel_err: f64::INFINITY,
        });
    let ref_time = results
        .iter()
        .find(|v| v.variant == ref_name)
        .map(|v| v.time_us)
        .unwrap_or(f64::NAN);
    let speedup = if ref_time.is_finite() && winner.time_us > 0.0 {
        ref_time / winner.time_us
    } else {
        1.0
    };

    Ok(ShapeBench {
        shape,
        winner: winner.variant.clone(),
        winner_time_us: winner.time_us,
        speedup_vs_cuda1: speedup,
        variants: results,
    })
}

fn max_rel_err(reference: &[f32], got: &[f32]) -> f64 {
    let mut max = 0f64;
    for (&a, &b) in reference.iter().zip(got.iter()) {
        let a = a as f64;
        let b = b as f64;
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(1e-6);
        let rel = diff / denom;
        if rel > max {
            max = rel;
        }
    }
    max
}
