use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, hip_backend::WrapErr};
use crate::{Result};
use half::f16;

use hipdarc::driver::{HipSlice, HipView};

use crate::hip_backend::HipDevice;
use crate::HipStorage;

/// Convenience macro to push scalar kernel arguments (mirrors the one in
/// hip_backend/mod.rs which is not exported).
macro_rules! barg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

#[derive(Debug)]
struct PaddedHipSlice {
    inner: HipSlice<u8>,
    len: usize,
}

#[derive(Debug)]
pub struct QHipStorage {
    data: PaddedHipSlice,
    dtype: GgmlDType,
    device: HipDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 64;
pub const MMQ_X_Q4_0_GFX906: usize = 64;
pub const MMQ_Y_Q4_0_GFX906: usize = 64;
pub const NWARPS_Q4_0_GFX906: usize = 8;
// Must match GGML_CUDA_DMMV_X in quantized.cu (hardcoded 32, not WARP_SIZE).
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &HipView<'_, f32>,
    dst: &mut HipSlice<u8>,
    k: usize,
    ky: usize,
    dev: &HipDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_hip_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = hipdarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (GGML_CUDA_MMV_X as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    let dst_view = dst.slice(0..dst.len());
    launch_mul_mat_vec_q8_1_chunk(
        data,
        &y_q8_1.slice(0..y_q8_1.len()),
        &dst_view,
        dtype,
        ncols,
        nrows,
        b_size,
        ncols_padded,
        dev,
    )?;
    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

/// Launch one specialized `mul_mat_vec_<dtype>_q8_1_cuda<b_size>` kernel
/// against an externally-managed (already-quantized) Q8_1 input view and an
/// externally-allocated output view. Used by both [`mul_mat_vec_via_q8_1`]
/// (which allocates everything itself) and [`mul_mat_via_q8_1_chunked`]
/// (which loops this function across slices of a single big buffer).
#[allow(clippy::too_many_arguments)]
fn launch_mul_mat_vec_q8_1_chunk(
    data: &PaddedHipSlice,
    y_q8_1: &hipdarc::driver::HipView<'_, u8>,
    dst: &hipdarc::driver::HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    ncols_padded: usize,
    dev: &HipDevice,
) -> Result<()> {
    if b_size == 0 || b_size > 8 {
        crate::bail!("launch_mul_mat_vec_q8_1_chunk: bsize must be 1..=8, got {b_size}")
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_hip_kernels::QUANTIZED)?;

    // For decode (b_size=1) on Q4_0/Q4_1/Q8_0 we use the gfx906
    // warp-cooperative kernel (1 wavefront, 2 rows per block, half-warp
    // per row). The K-quant b_size=1 kernels and all b_size>1 paths still
    // use the original ggml-cuda template which expects (WARP_SIZE,4,1)
    // or (WARP_SIZE,2,1) blocks.
    let warp_coop = b_size == 1
        && matches!(
            dtype,
            GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q8_0
        );
    let cfg = if warp_coop {
        hipdarc::driver::LaunchConfig {
            grid_dim: ((nrows as u32).div_ceil(2), 1, 1),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
        let (nblocks, nwarps) = match b_size {
            1 => (nrows as u32, 4),
            2..=4 => ((nrows as u32).div_ceil(2), 4),
            5..=8 => ((nrows as u32).div_ceil(2), 2),
            _ => unreachable!(),
        };
        hipdarc::driver::LaunchConfig {
            grid_dim: (nblocks, 1, 1),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        }
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y_q8_1);
    builder.arg(dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(())
}

/// Phase 2b/2d MMQ: single-launch quantized × Q8_1 matrix matmul for the
/// `b*m ≥ 9` prefill path on gfx906.
///
/// Replaces `mul_mat_via_q8_1_chunked`'s `ceil(total_b / 8)` vector launches
/// with one launch of a dtype-specific `mul_mat_<dtype>_gfx906_v2` kernel.
/// Each kernel owns one output row per thread, 64 threads per Wave64 block,
/// TILE_N=8 columns per block. Correctness-equivalent to the chunked path
/// to within Q4_0/Q4_1/Q8_0 × Q8_1 quantization tolerance.
///
/// Supported dtypes (Phase 2d): `Q4_0`, `Q4_1`, `Q8_0`. K-quants still fall
/// through to the chunked path until Phase 2f. The launcher returns `None`
/// when the dtype isn't supported so the caller can fall back.
///
/// Input `y`: f32 `(total_b, ncols)` row-major. Output: f32 `(total_b, nrows)`
/// row-major — same layout as `mul_mat_via_q8_1_chunked` so call sites don't
/// need to care which path serviced the request.
#[allow(clippy::too_many_arguments)]
fn mul_mat_q_v2(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    total_b: usize,
    dev: &HipDevice,
) -> Result<Option<HipStorage>> {
    // Phase 2c: widened N-tile variants. Default is TILE_N=32, which gives
    // ~1.35× prefill on qwen35-9B Q4_1 vs the Phase 2d baseline of TILE_N=8.
    // The wider tile amortises the per-K X load across more output columns
    // per thread, raising arithmetic intensity without growing the block
    // beyond 1 warp or adding LDS staging. Measured on MI50 1 GPU, 1149-tok
    // prompt (pp ~145 → ~196 t/s); TILE_N=64 regressed below baseline on
    // the same workload (too few blocks for CU saturation on a 1024-col
    // projection). The `CANDLE_MMQ_TILE_N=8|16|32|64` env var selects a
    // specific variant for A/B diagnostics.
    let tile_n: usize = std::env::var("CANDLE_MMQ_TILE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    mul_mat_q_v2_with_tile_n(data, y, dtype, ncols, nrows, total_b, dev, tile_n)
}

/// Internal variant of [`mul_mat_q_v2`] that takes an explicit `tile_n`.
/// Used by the public entry (which reads `CANDLE_MMQ_TILE_N`) and by the
/// regression test (which iterates every supported tile size).
#[allow(clippy::too_many_arguments)]
fn mul_mat_q_v2_with_tile_n(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    total_b: usize,
    dev: &HipDevice,
    tile_n_hint: usize,
) -> Result<Option<HipStorage>> {
    // Pick the kernel name + tile size. The fallback (`_`) clamps any
    // unsupported hint (or 0) to the default TILE_N=32.
    let (kernel_name, tile_n): (&str, usize) = match dtype {
        GgmlDType::Q4_0 => match tile_n_hint {
            8 => ("mul_mat_q4_0_gfx906_v2", 8),
            16 => ("mul_mat_q4_0_gfx906_v2_tile16", 16),
            _ => ("mul_mat_q4_0_gfx906_v2_tile32", 32),
        },
        GgmlDType::Q4_1 => match tile_n_hint {
            8 => ("mul_mat_q4_1_gfx906_v2", 8),
            16 => ("mul_mat_q4_1_gfx906_v2_tile16", 16),
            64 => ("mul_mat_q4_1_gfx906_v2_tile64", 64),
            _ => ("mul_mat_q4_1_gfx906_v2_tile32", 32),
        },
        GgmlDType::Q8_0 => match tile_n_hint {
            8 => ("mul_mat_q8_0_gfx906_v2", 8),
            16 => ("mul_mat_q8_0_gfx906_v2_tile16", 16),
            _ => ("mul_mat_q8_0_gfx906_v2_tile32", 32),
        },
        // P5: K-quant MMQ. Q5_K is the qwen35-9B output head tensor and
        // sits at ~35% of total GPU time in the chunked-vector fallback.
        // Each super-block is 256 K-elements with 8 sub-blocks — the
        // kernel loads one super-block per K iteration, precomputes
        // 8 sub-scales/mins via get_scale_min_k4, and unrolls the
        // 8-sub-block dp4a loop.
        GgmlDType::Q5K => match tile_n_hint {
            8 => ("mul_mat_q5_K_gfx906_v2", 8),
            16 => ("mul_mat_q5_K_gfx906_v2_tile16", 16),
            _ => ("mul_mat_q5_K_gfx906_v2_tile32", 32),
        },
        _ => return Ok(None),
    };

    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!(
            "mul_mat_q_v2: unexpected data size {}, ncols {ncols} nrows {nrows}",
            data_elems
        )
    }
    if y.len() != ncols * total_b {
        crate::bail!(
            "mul_mat_q_v2: unexpected y size {}, ncols {ncols} total_b {total_b}",
            y.len()
        )
    }
    if total_b == 0 {
        return Ok(Some(HipStorage::wrap_hip_slice(
            dev.alloc_zeros::<f32>(0)?,
            dev.clone(),
        )));
    }

    // 1. Quantize all `total_b` input rows to q8_1 in one shot. The layout
    //    produced by `quantize_q8_1` is column-major blocks: `y_q8_1[col * blocks_per_col_y + ib]`
    //    which is exactly what the kernel expects.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let bytes_per_row = ncols_padded / q8_1_block_size * q8_1_type_size;
    let y_size_in_bytes = total_b * bytes_per_row;
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, total_b, dev)?;

    // 2. Allocate output buffer — row-major `(total_b, nrows)`, matching the
    //    chunked path's convention.
    let dst = dev.alloc_zeros::<f32>(total_b * nrows)?;

    // 3. Launch. Grid covers `ceil(nrows / 64)` Y-tiles × `ceil(total_b / tile_n)`
    //    X-tiles (one Wave64 warp per tile). Block is a single 64-thread warp.
    const TILE_M: usize = 64;
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(nrows, TILE_M) as u32,
            ceil_div(total_b, tile_n) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* ncols_y */ total_b as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    Ok(Some(HipStorage::wrap_hip_slice(dst, dev.clone())))
}

/// Quantized matrix-matrix multiply built by chunking the input into
/// `b_size <= 8` slices and dispatching the well-tested vector kernel
/// (`mul_mat_vec_<dtype>_q8_1_cuda<b_size>`) for each chunk.
///
/// **Why this exists.** The integer-MMQ matrix kernels (`mul_mat_q*` in
/// `candle-hip-kernels/src/quantized.cu`) produce numerically wrong
/// results on gfx906 once `b*m >= 9` — see task #22. Until those kernels
/// are fixed/replaced (the upstream `mul_mat_q` rewrite is much larger),
/// chunking the call into multiple vector-path launches gives correct
/// results at a small launch-overhead cost. For typical prefill (L=15)
/// this is 2 launches instead of 1, far cheaper than the previous
/// dequant+f32-gemm fallback.
///
/// Layout: input is `(total_b, ncols)` row-major f32. Output is
/// `(total_b, nrows)` row-major f32 (`dst[i*nrows + j]` = row i × col j).
#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1_chunked(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    total_b: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!(
            "unexpected data size {}, ncols {ncols} nrows {nrows}",
            data_elems
        )
    }
    if y.len() != ncols * total_b {
        crate::bail!("unexpected y size {}, ncols {ncols} total_b {total_b}", y.len())
    }
    if total_b == 0 {
        return Ok(HipStorage::wrap_hip_slice(
            dev.alloc_zeros::<f32>(0)?,
            dev.clone(),
        ));
    }

    // 1. Quantize ALL input rows to q8_1 in one shot. quantize_q8_1
    //    operates row-by-row so the cost is the same whether we call it
    //    once with total_b rows or N times with chunks of 8 — and the
    //    single call is fewer kernel launches.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let bytes_per_row = ncols_padded / q8_1_block_size * q8_1_type_size;
    let y_size_in_bytes = total_b * bytes_per_row;
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, total_b, dev)?;

    // 2. Pre-allocate the full output buffer.
    let dst = dev.alloc_zeros::<f32>(total_b * nrows)?;

    // 3. Loop chunks of up to 8 rows. Each iteration launches one vector
    //    kernel against a slice of the quantized input and writes into a
    //    slice of the pre-allocated output buffer.
    const MAX_CHUNK: usize = 8;
    let mut chunk_start = 0usize;
    while chunk_start < total_b {
        let chunk = (total_b - chunk_start).min(MAX_CHUNK);
        // Slice the quantized input: chunk's bytes start at
        // chunk_start * bytes_per_row.
        let q_chunk = y_q8_1.slice(
            chunk_start * bytes_per_row..(chunk_start + chunk) * bytes_per_row,
        );
        // Slice the output: chunk's f32 floats start at
        // chunk_start * nrows.
        let dst_chunk =
            dst.slice(chunk_start * nrows..(chunk_start + chunk) * nrows);
        launch_mul_mat_vec_q8_1_chunk(
            data,
            &q_chunk,
            &dst_chunk,
            dtype,
            ncols,
            nrows,
            chunk,
            ncols_padded,
            dev,
        )?;
        chunk_start += chunk;
    }

    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

/// Integer-MMQ quantized matrix-matrix multiply (input quantized to q8_1).
///
/// **Currently unused — kept here as scaffolding for a future fix.**
///
/// The `mul_mat_q*` kernels in `candle-hip-kernels/src/quantized.cu` produce
/// numerically wrong results on gfx906 (Wave64) for any `b*m >= 9`. The bug
/// shows up as catastrophic divergence in qwen3-family multi-token prefill
/// (task #22). Until the kernels are fixed, [`QHipStorage::dequantize_matmul`]
/// always dequantizes the weight to f32 and dispatches through the regular
/// rocBLAS gemm, which is well-tested. This costs one weight-sized f32
/// allocation per call, freed when the call returns.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn mul_mat_via_q8_1(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &HipDevice,
) -> Result<HipStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 64),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 64),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 64, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 64, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 64, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 64),
        GgmlDType::Q3K => ("mul_mat_q3_K", 64, 64),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 64),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 64),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(x_rows * y_cols)?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 8, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(HipStorage::wrap_hip_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
/// MoE forward pass for F16 weights with F32 input.
///
/// Skips the q8_1 input quantization step that the K-quant path uses, since
/// F16 weights can be multiplied directly by F32 inputs in the kernel.
fn indexed_moe_forward_f16_f32(
    weight: &HipView<'_, u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    input: &HipSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &HipView<'_, u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &HipDevice,
) -> Result<(HipStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];
    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Output buffer: [batch, topk, n]
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let func =
        dev.get_or_load_func("indexed_moe_forward_f16_f32", &candle_hip_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(input);
    builder.arg(ids);
    builder.arg(&out);
    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        HipStorage::wrap_hip_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

fn indexed_moe_forward_fused_q8_1_input(
    weight: &HipView<'_, u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &HipSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &HipView<'_, u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &HipDevice,
) -> Result<(HipStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = dev.alloc_zeros::<u8>(y_size_in_bytes)?;

    let input_view = input.slice(0..input.len());
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        GgmlDType::Q4_0 => "indexed_moe_forward_q4_0_q8_1",
        GgmlDType::Q4_1 => "indexed_moe_forward_q4_1_q8_1",
        GgmlDType::Q5_0 => "indexed_moe_forward_q5_0_q8_1",
        GgmlDType::Q5_1 => "indexed_moe_forward_q5_1_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_hip_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        HipStorage::wrap_hip_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QHipStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &HipStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &HipStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_hip_slice::<f32>()?;
            let ids_storage = ids.as_hip_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..self.data.inner.len()),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..ids_storage.len()),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else if self.dtype() == GgmlDType::F16 {
            // Mixed-precision GGUFs (e.g. Unsloth Dynamic Q8_K_XL) leave some
            // expert tensors as raw F16. Skip the q8_1 input quantization and
            // dispatch directly to the F16 weight × F32 input kernel.
            let input_storage = input.as_hip_slice::<f32>()?;
            let ids_storage = ids.as_hip_slice::<u32>()?;
            indexed_moe_forward_f16_f32(
                &self.data.inner.slice(0..self.data.inner.len()),
                self_shape,
                input_storage,
                input_l.shape(),
                &ids_storage.slice(0..ids_storage.len()),
                ids_l.shape(),
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &HipDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QHipStorage {
            data: PaddedHipSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<HipStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = {
            let full = self.device.clone_dtoh(&self.data.inner)?;
            full[..self.data.len].to_vec()
        };
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
            GgmlDType::Mxfp4 => deq::<crate::quantized::BlockMxfp4>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<HipStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &HipStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::hip_backend::HipStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .stream()
            .memcpy_htod(&mut inner, &data).w()?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &HipStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::hip_backend::HipStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .stream()
            .memcpy_htod(&mut inner, &data).w()?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .stream()
            .memcpy_htod(&mut inner, &data).w()?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .stream()
            .memcpy_htod(&mut inner, &data).w()?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &HipStorage,
        layout: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let full = self.device.clone_dtoh(&self.data.inner)?;
        Ok(full[..self.data.len].to_vec())
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.device_ptr() as *const u8)
    }
}

impl QHipStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &HipStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_hip_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &HipStorage,
        layout: &crate::Layout,
    ) -> Result<(HipStorage, crate::Shape)> {
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        // Phase 2b/2d MMQ dispatch. For Q4_0/Q4_1/Q8_0 with `b*m >= 9` we use
        // the new single-launch `mul_mat_<dtype>_gfx906_v2` kernels. Smaller
        // batches still use the chunked-vector path (correct, already fast).
        // K-quants fall through to the chunked path until Phase 2f.
        //
        // The old broken `mul_mat_q*` kernels (task #22) are kept in the HSACO
        // for now but are not dispatched — when the full stream-K port lands
        // they'll be deleted.
        let storage = storage.as_hip_slice::<f32>()?;
        let storage = match layout.contiguous_offsets() {
            Some((o1, o2)) => storage.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-matmul",
            }
            .bt())?,
        };
        let total_b = b * m;
        let out = if total_b >= 9 {
            if let Some(out) = mul_mat_q_v2(
                &self.data,
                &storage,
                self.dtype,
                /* ncols */ k,
                /* nrows */ n,
                total_b,
                self.device(),
            )? {
                out
            } else {
                mul_mat_via_q8_1_chunked(
                    &self.data,
                    &storage,
                    self.dtype,
                    k,
                    n,
                    total_b,
                    self.device(),
                )?
            }
        } else {
            mul_mat_via_q8_1_chunked(
                &self.data,
                &storage,
                self.dtype,
                k,
                n,
                total_b,
                self.device(),
            )?
        };

        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &HipDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device
        .stream()
        .memcpy_htod(&mut inner, data).w()?;
    Ok(QStorage::Hip(QHipStorage {
        data: PaddedHipSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn hip_quantize_q8_1() -> Result<()> {
        let dev = HipDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.slice(0..y.len()), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn hip_mmv_q8_1() -> Result<()> {
        let dev = HipDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QHipStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&HipStorage::wrap_hip_slice(y.try_clone(dev.stream()).w()?, dev.clone()))?;
        let hip_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.slice(0..y.len()),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = hip_storage.as_hip_slice::<f32>()?;
        let vs = dev.clone_dtoh(vs)?;
        assert_eq!(vs.len(), 1);

        let hip_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.slice(0..y.len()),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = hip_storage.as_hip_slice::<f32>()?;
        let vs = dev.clone_dtoh(vs)?;
        assert_eq!(vs.len(), 1);
        Ok(())
    }

    #[test]
    fn hip_mm_q8_1() -> Result<()> {
        let dev = HipDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QHipStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&HipStorage::wrap_hip_slice(y.try_clone(dev.stream()).w()?, dev.clone()))?;
        let hip_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.slice(0..y.len()),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = hip_storage.as_hip_slice::<f32>()?;
        let _vs = dev.clone_dtoh(vs)?;
        Ok(())
    }

    #[test]
    fn hip_mm_q8_1_pad() -> Result<()> {
        let dev = HipDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QHipStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&HipStorage::wrap_hip_slice(y.try_clone(dev.stream()).w()?, dev.clone()))?;
        let hip_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.slice(0..y.len()),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = hip_storage.as_hip_slice::<f32>()?;
        let _vs = dev.clone_dtoh(vs)?;
        Ok(())
    }

    /// Helper: quantize an f32 weight matrix `(nrows × ncols)` to the given
    /// Q4_0 / Q4_1 / Q8_0 dtype, upload to HIP as a `PaddedHipSlice`, and
    /// return the wrapper needed by the launcher functions.
    fn make_q_weight(
        dev: &HipDevice,
        weight_f32: &[f32],
        nrows: usize,
        ncols: usize,
        dtype: GgmlDType,
    ) -> Result<PaddedHipSlice> {
        use crate::quantized::k_quants::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ8_0, QK_K};
        assert_eq!(weight_f32.len(), nrows * ncols);

        // K-quants need ncols divisible by 256; non-K quants need 32.
        let is_k_quant = matches!(dtype, GgmlDType::Q5K);
        let min_div = if is_k_quant { QK_K } else { 32 };
        assert!(
            ncols % min_div == 0,
            "dtype {dtype:?} requires ncols % {min_div} == 0, got {ncols}"
        );

        // Quantize on CPU via the existing block types.
        let (data_bytes_vec, type_size) = match dtype {
            GgmlDType::Q4_0 => {
                let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); nrows * ncols / 32];
                BlockQ4_0::from_float(weight_f32, &mut blocks);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        core::mem::size_of_val(blocks.as_slice()),
                    )
                }
                .to_vec();
                (bytes, GgmlDType::Q4_0.type_size())
            }
            GgmlDType::Q4_1 => {
                let mut blocks: Vec<BlockQ4_1> = vec![BlockQ4_1::zeros(); nrows * ncols / 32];
                BlockQ4_1::from_float(weight_f32, &mut blocks);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        core::mem::size_of_val(blocks.as_slice()),
                    )
                }
                .to_vec();
                (bytes, GgmlDType::Q4_1.type_size())
            }
            GgmlDType::Q8_0 => {
                let mut blocks: Vec<BlockQ8_0> = vec![BlockQ8_0::zeros(); nrows * ncols / 32];
                BlockQ8_0::from_float(weight_f32, &mut blocks);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        core::mem::size_of_val(blocks.as_slice()),
                    )
                }
                .to_vec();
                (bytes, GgmlDType::Q8_0.type_size())
            }
            GgmlDType::Q5K => {
                // Q5_K super-block = 256 elements → nrows * ncols / QK_K blocks.
                let n_blocks = nrows * ncols / QK_K;
                let mut blocks: Vec<BlockQ5K> = vec![
                    BlockQ5K {
                        d: half::f16::from_f32(0.0),
                        dmin: half::f16::from_f32(0.0),
                        scales: [0u8; 12],
                        qh: [0u8; 32],
                        qs: [0u8; 128],
                    };
                    n_blocks
                ];
                BlockQ5K::from_float(weight_f32, &mut blocks);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        core::mem::size_of_val(blocks.as_slice()),
                    )
                }
                .to_vec();
                (bytes, GgmlDType::Q5K.type_size())
            }
            _ => crate::bail!("make_q_weight: unsupported dtype {dtype:?}"),
        };

        // Reserve MATRIX_ROW_PADDING elements of trailing zeros as a
        // safety margin for tile-aligned reads. `block_size` is 32 for
        // Q4_0/Q4_1/Q8_0 and 256 for K-quants like Q5_K.
        let block_size = dtype.block_size();
        let pad_blocks = MATRIX_ROW_PADDING.div_ceil(block_size);
        let padded_len = data_bytes_vec.len() + pad_blocks * type_size;
        let mut inner = dev.alloc_zeros::<u8>(padded_len)?;
        dev.stream().memcpy_htod(&mut inner, &data_bytes_vec).w()?;
        Ok(PaddedHipSlice {
            inner,
            len: data_bytes_vec.len(),
        })
    }

    /// Phase 2b/2d regression test: `mul_mat_q_v2` must agree with the
    /// chunked-vector path on every (dtype, shape) combination below.
    /// `m = 9` is the critical boundary — this is where the old
    /// `mul_mat_q<QK4_0, ...>` kernel produced catastrophic divergence
    /// (task #22).
    #[test]
    fn hip_mmq_v2_matches_chunked() -> Result<()> {
        let dev = HipDevice::new(0)?;

        // (m, n, k) — cover the regression target m=9, plus a spread of
        // realistic prefill shapes. K is divisible by 32 for the
        // non-K quants; K-quants (Q5K) need ncols divisible by 256.
        let shapes_common: &[(usize, usize, usize)] = &[
            (9, 64, 128),    // boundary
            (9, 1024, 4096), // qwen35-9B projection shape
            (16, 64, 128),
            (16, 1024, 4096),
            (17, 128, 256),  // non-aligned m
            (64, 64, 128),
            (128, 2048, 2048),
            (512, 1024, 4096), // pp512 prefill
        ];
        // K-quant shapes: k must be divisible by QK_K=256. All other
        // dims can match the common shapes.
        let shapes_k: &[(usize, usize, usize)] = &[
            (9, 64, 256),    // boundary, one super-block
            (9, 1024, 4096), // qwen35-9B projection shape (k=4096=16 super-blocks)
            (16, 1024, 4096),
            (17, 128, 512),  // non-aligned m, 2 super-blocks
            (64, 64, 256),
            (128, 2048, 2048),
            (512, 1024, 4096), // pp512 prefill
        ];
        let dtypes = [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q8_0,
            GgmlDType::Q5K,
        ];

        // Tile sizes to exercise. Q4_1 also covers tile_n=64 because the
        // template has that instantiation (even though it's slower than
        // tile_n=32 on qwen35 shapes — the kernel must still be correct).
        let tile_ns_common: &[usize] = &[8, 16, 32];
        let tile_ns_q4_1: &[usize] = &[8, 16, 32, 64];

        for &dtype in &dtypes {
            let tile_ns = if dtype == GgmlDType::Q4_1 {
                tile_ns_q4_1
            } else {
                tile_ns_common
            };
            let shapes: &[(usize, usize, usize)] = if dtype == GgmlDType::Q5K {
                shapes_k
            } else {
                shapes_common
            };
            for &tile_n_hint in tile_ns {
                for &(m, n, k) in shapes {
                    // Deterministic weights and inputs. Small magnitudes so
                    // quantization noise stays well below 1.0.
                    let weight_f32: Vec<f32> = (0..n * k)
                        .map(|i| ((i as f32) * 0.0001).sin() * 0.3)
                        .collect();
                    let input_f32: Vec<f32> = (0..m * k)
                        .map(|i| ((i as f32) * 0.0003).cos() * 0.3)
                        .collect();

                    let data = make_q_weight(&dev, &weight_f32, n, k, dtype)?;
                    let y_dev = dev.clone_htod(&input_f32)?;

                    // Force a specific tile_n so each variant is exercised
                    // regardless of CANDLE_MMQ_TILE_N env state.
                    let out_v2 = mul_mat_q_v2_with_tile_n(
                        &data,
                        &y_dev.slice(0..y_dev.len()),
                        dtype,
                        /* ncols */ k,
                        /* nrows */ n,
                        /* total_b */ m,
                        &dev,
                        tile_n_hint,
                    )?
                    .unwrap_or_else(|| {
                        panic!("mul_mat_q_v2 should return Some for {dtype:?}")
                    });

                    // Known-good chunked-vector path (same inputs).
                    let out_chunk = mul_mat_via_q8_1_chunked(
                        &data,
                        &y_dev.slice(0..y_dev.len()),
                        dtype,
                        k,
                        n,
                        m,
                        &dev,
                    )?;

                    let v2_cpu = dev.clone_dtoh(out_v2.as_hip_slice::<f32>()?)?;
                    let chunk_cpu = dev.clone_dtoh(out_chunk.as_hip_slice::<f32>()?)?;
                    assert_eq!(
                        v2_cpu.len(),
                        chunk_cpu.len(),
                        "length mismatch for {dtype:?} tile={tile_n_hint} {:?}",
                        (m, n, k)
                    );

                    // Both paths quantize Y to Q8_1 the same way and use the same
                    // per-block formulas. The only drift source is FMA accumulation
                    // rounding. Allow 1e-4 absolute + 1e-3 relative.
                    let mut max_abs = 0.0f32;
                    let mut max_rel = 0.0f32;
                    let mut ref_max = 0.0f32;
                    for (a, b) in v2_cpu.iter().zip(chunk_cpu.iter()) {
                        let diff = (a - b).abs();
                        if diff > max_abs {
                            max_abs = diff;
                        }
                        ref_max = ref_max.max(b.abs());
                        if b.abs() > 1e-6 {
                            max_rel = max_rel.max(diff / b.abs());
                        }
                    }
                    assert!(
                        max_abs < 1e-4 + 1e-3 * ref_max,
                        "mmq_v2 vs chunked mismatch for {dtype:?} tile={tile_n_hint} at {:?}: \
                         max_abs={max_abs} max_rel={max_rel} ref_max={ref_max}",
                        (m, n, k)
                    );
                }
            }
        }
        Ok(())
    }
}
