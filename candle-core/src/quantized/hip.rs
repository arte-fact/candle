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
pub struct PaddedHipSlice {
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

pub fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Quantize f32 input to Q8_1 format for use with MMVQ/MMQ kernels.
/// Public so that model code can pre-quantize once and reuse across
/// multiple matmul consumers (D1/D2 optimization from the roadmap).
pub fn quantize_q8_1(
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
    // Start by quantizing y. The quantize_q8_1 kernel covers the full
    // `(kx_padded/QK8_1, b_size)` grid and writes every Q8_1 block in the
    // buffer (the tail col reads past `ncols` are gated to 0 inside the
    // kernel, see quantize_q8_1.cu:3433). So we can skip the zero fill.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // SAFETY: the quantize_q8_1 kernel launched on the next line fully
    // overwrites every block of this buffer (see the safety note above).
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    // SAFETY: launch_mul_mat_vec_q8_1_chunk fully overwrites its output.
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
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
/// externally-allocated output view. Public for D1/D2 cache optimization.
/// externally-allocated output view. Used by both [`mul_mat_vec_via_q8_1`]
/// (which allocates everything itself) and [`mul_mat_via_q8_1_chunked`]
/// (which loops this function across slices of a single big buffer).
#[allow(clippy::too_many_arguments)]
pub fn launch_mul_mat_vec_q8_1_chunk(
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
        GgmlDType::Mxfp4 => "mul_mat_vec_mxfp4_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    // Q4K/Q5K warp-coop are correct as of Phase B3-redo (2026-04-13).
    // Q6K still diverges (distinct 4+2 packing; kernel layout not yet ported).
    // Opt into Q6K warp-coop via CANDLE_KQUANT_WARP_COOP=1 (still broken).
    let q6_warp_coop = std::env::var("CANDLE_KQUANT_WARP_COOP").is_ok();
    let kernel_name = if b_size == 1
        && matches!(dtype, GgmlDType::Q4K | GgmlDType::Q5K)
    {
        match dtype {
            GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_wc".to_string(),
            GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_wc".to_string(),
            _ => unreachable!(),
        }
    } else if b_size == 1 && q6_warp_coop && matches!(dtype, GgmlDType::Q6K) {
        "mul_mat_vec_q6_K_q8_1_wc".to_string()
    } else {
        format!("{kernel_name}{b_size}")
    };
    let func = dev.get_or_load_func(&kernel_name, &candle_hip_kernels::QUANTIZED)?;

    // For decode (b_size=1) on Q4_0/Q4_1/Q8_0 we use the gfx906
    // warp-cooperative kernel (1 wavefront, 2 rows per block, half-warp
    // per row). The K-quant b_size=1 kernels and all b_size>1 paths still
    // use the original ggml-cuda template which expects (WARP_SIZE,4,1)
    // or (WARP_SIZE,2,1) blocks.
    // Q4K/Q5K warp-coop kernels correct after Phase B3-redo (pair-layout fix
    // landed 2026-04-13). Q6K still diverges; opt in via
    // CANDLE_KQUANT_WARP_COOP=1 once fixed.
    let q6_warp_coop = std::env::var("CANDLE_KQUANT_WARP_COOP").is_ok();
    let warp_coop = b_size == 1
        && (matches!(
                dtype,
                GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q8_0
                | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Mxfp4
            )
            || (q6_warp_coop && matches!(dtype, GgmlDType::Q6K)));
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

/// Fused gate+up MMVQ for Q4_0 decode. Runs ONE kernel that produces both
/// `W_gate @ x_q8_1` and `W_up @ x_q8_1` into separate output buffers,
/// sharing the Q8_1 vector reads. Saves one kernel launch per FFN layer and
/// cuts L2 traffic on the input vector roughly in half.
///
/// Preconditions: both weights are Q4_0, same shape `(nrows, ncols)`. Output
/// buffers must each hold `nrows` f32 elements.
pub fn launch_mul_mat_vec_q4_0_gate_up_fused(
    w_gate: &PaddedHipSlice,
    w_up: &PaddedHipSlice,
    y_q8_1: &hipdarc::driver::HipView<'_, u8>,
    dst_gate: &hipdarc::driver::HipView<'_, f32>,
    dst_up: &hipdarc::driver::HipView<'_, f32>,
    ncols: usize,
    nrows: usize,
    dev: &HipDevice,
) -> Result<()> {
    let func = dev.get_or_load_func(
        "mul_mat_vec_q4_0_q8_1_gate_up_fused",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(2), 1, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&w_gate.inner);
    builder.arg(&w_up.inner);
    builder.arg(y_q8_1);
    builder.arg(dst_gate);
    builder.arg(dst_up);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(())
}

/// Fused down+residual MMVQ for Q4_0 decode. Computes `W_down @ hidden_q8_1
/// + residual` in a single kernel. Saves one kernel launch per FFN layer.
pub fn launch_mul_mat_vec_q4_0_down_residual(
    w_down: &PaddedHipSlice,
    y_q8_1: &hipdarc::driver::HipView<'_, u8>,
    residual: &hipdarc::driver::HipView<'_, f32>,
    dst: &hipdarc::driver::HipView<'_, f32>,
    ncols: usize,
    nrows: usize,
    dev: &HipDevice,
) -> Result<()> {
    let func = dev.get_or_load_func(
        "mul_mat_vec_q4_0_q8_1_down_residual",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: ((nrows as u32).div_ceil(2), 1, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&w_down.inner);
    builder.arg(y_q8_1);
    builder.arg(residual);
    builder.arg(dst);
    barg!(builder, ncols as i32, nrows as i32);
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

    // MMQ turbo-port gate. Q4_0 (M2), Q4_1 (M3a), Q8_0 (M3c) covered.
    // See candle-hip-kernels/src/mmq_turbo.cu.
    let turbo_on = std::env::var("CANDLE_MMQ_TURBO_PORT")
        .map(|v| v == "1")
        .unwrap_or(false);
    if turbo_on
        && (dtype == GgmlDType::Q4_0
            || dtype == GgmlDType::Q4_1
            || dtype == GgmlDType::Q8_0)
    {
        if let Some(out) = mul_mat_q40q41_turbo(data, y, dtype, ncols, nrows, total_b, dev)? {
            return Ok(Some(out));
        }
    }

    mul_mat_q_v2_with_tile_n(data, y, dtype, ncols, nrows, total_b, dev, tile_n)
}

/// Turbo-port path: Q4_0 (M2) / Q4_1 (M3a) MMQ with turbo's tile geometry
/// (mmq_y=128, mmq_x∈{8,16,32,64}, nwarps=4). Uses the mmq Q8_1 layout
/// (`block_q8_1_mmq`, 144 B/block, ordered (big_block, col)) rather
/// than candle's row-major per-col Q8_1 — the port provides its own
/// `quantize_q8_1_mmq_q4_0` kernel (reusable across Q4_0 and Q4_1).
fn mul_mat_q40q41_turbo(
    data: &PaddedHipSlice,
    y: &HipView<'_, f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    total_b: usize,
    dev: &HipDevice,
) -> Result<Option<HipStorage>> {
    const MMQ_Y_DEFAULT: usize = 128;
    // U1: Q8_0 uses mmq_y=64 so two blocks/CU fit in 64 KB LDS (x_qs is 2×
    // larger per row than Q4_0/Q4_1 and would otherwise push total to
    // 46 720 B/block — 2 blocks = 93 KB, over the 64 KB gfx906 limit).
    const MMQ_Y_Q8: usize = 64;
    const MMQ_NWARPS: u32 = 4;
    const MMQ_TILE_NE_K: usize = 32;
    const Q8_1_MMQ_BYTES: usize = 144;
    const BIG_BLOCK_ELEMS: usize = 128;
    const MMQ_TILE_Y_K: usize = MMQ_TILE_NE_K + MMQ_TILE_NE_K / 8; // 36 ints

    let dtype_tag = match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q8_0 => "q8_0",
        _ => return Ok(None),
    };

    let mmq_y: usize = match dtype {
        GgmlDType::Q8_0 => MMQ_Y_Q8,
        _ => MMQ_Y_DEFAULT,
    };

    if total_b == 0 {
        return Ok(Some(HipStorage::wrap_hip_slice(
            dev.alloc_zeros::<f32>(0)?,
            dev.clone(),
        )));
    }
    if y.len() != ncols * total_b {
        crate::bail!(
            "mul_mat_{}_turbo: unexpected y size {}, ncols {ncols} total_b {total_b}",
            dtype_tag, y.len()
        )
    }
    // K (ncols) only needs to be a multiple of QK=32 (the Q4_0/Q4_1 block
    // size). M3b: the kernel handles non-multiple-of-256 K via in-kernel
    // OOB clamp in load_tiles (kb_remaining param zeroes d/dm contributions
    // for OOB blocks).
    const QK: usize = 32;
    if ncols % QK != 0 {
        return Ok(None);
    }
    // Non-aligned M (nrows): use the `_checked` kernel variant which
    // clamps OOB i values during load_tiles and gates the write-back
    // by row_g < nrows_x.  Only requires that the X buffer has room
    // for at least one valid row past the last legitimate one — true
    // because Q4_0/Q4_1 weight buffers are PaddedHipSlice-sized to
    // dtype-block alignment.
    let need_check = nrows % mmq_y != 0;

    // Pick mmq_x adaptive to batch.  Temporary: env override for debug.
    let mmq_x: usize = if let Ok(s) = std::env::var("CANDLE_MMQ_TURBO_X") {
        s.parse().unwrap_or(0)
    } else {
        match total_b {
            0 => return Ok(None),
            1..=8 => 8,
            9..=16 => 16,
            17..=32 => 32,
            _ => 64,
        }
    };
    let suffix = if need_check { "checked" } else { "unchecked" };
    let kernel_name: String = match mmq_x {
        8 | 16 | 32 | 64 => format!("mul_mat_{dtype_tag}_turbo_x{mmq_x}_{suffix}"),
        _ => return Ok(None),
    };

    // Allocate + populate the turbo-layout Q8_1 buffer.
    // n_big_blocks_real: ceil(K/128) — number of big-blocks the quantize
    //   kernel will write (last one may have partial valid range; the
    //   kernel's `ki < ncols` guard zeroes the K-tail).
    // n_big_blocks_alloc: + 1 extra so the K-loop's "second half" read at
    //   the last kb0 iter has a valid (zeroed) destination — the M3b
    //   kernel reads big-blocks (kb0/4) and (kb0/4 + 1).
    let n_big_blocks_real = ncols.div_ceil(BIG_BLOCK_ELEMS);
    let n_big_blocks_alloc = n_big_blocks_real + 1;
    let total_b_padded = total_b.div_ceil(mmq_x) * mmq_x;
    let y_mmq_size_bytes = n_big_blocks_alloc * total_b_padded * Q8_1_MMQ_BYTES;

    // Zero-fill: OOB cols (total_b..total_b_padded) and the tail big-block
    // both stay zero, contributing nothing to vec_dot.
    let mut y_mmq = dev.alloc_zeros::<u8>(y_mmq_size_bytes)?;

    // Quantize f32 y → block_q8_1_mmq laid out (big_block, col).
    //   grid = (n_big_blocks_real, total_b, 1), block = (128, 1, 1).
    let qkernel = dev.get_or_load_func(
        "quantize_q8_1_mmq_q4_0",
        &candle_hip_kernels::MMQ_TURBO,
    )?;
    let qcfg = hipdarc::driver::LaunchConfig {
        grid_dim: (n_big_blocks_real as u32, total_b as u32, 1),
        block_dim: (BIG_BLOCK_ELEMS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut qbuilder = qkernel.builder();
    qbuilder.arg(/* x */ y);
    qbuilder.arg(/* vy */ &y_mmq);
    barg!(
        qbuilder,
        /* ncols */ ncols as i32,
        /* total_b */ total_b_padded as i32
    );
    unsafe { qbuilder.launch(qcfg) }.w()?;

    // Allocate dst.
    let dst = unsafe { dev.alloc::<f32>(total_b * nrows)? };

    // LDS size for the chosen mmq_x:
    //   tile_y = PAD(mmq_x * MMQ_TILE_Y_K, nwarps*warp_size)  ints
    //   x_qs   = MMQ_Y * (MMQ_TILE_NE_K + 1)                  ints
    //   x_df   = MMQ_Y * (MMQ_TILE_NE_K / QI4_0) + MMQ_Y/QI4_0 floats
    let tile_y_ints = {
        let raw = mmq_x * MMQ_TILE_Y_K;
        let pad = MMQ_NWARPS as usize * WARP_SIZE;
        raw.div_ceil(pad) * pad
    };
    // Q8_0 LDS has 2× x_qs (full int per 4 values, not 0.5 int per 4 nibble-
    // pairs) but mmq_y=64 halves the row count → total x_qs 16 640 B (vs Q4_0
    // 16 896 B). x_df formula is mmq_y*8 + mmq_y/4 in all cases.
    let x_qs_ints = match dtype {
        GgmlDType::Q8_0 => mmq_y * (2 * MMQ_TILE_NE_K + 1),
        _ => mmq_y * (MMQ_TILE_NE_K + 1),
    };
    let x_df_flts = mmq_y * 8 + mmq_y / 4;
    let shared_mem_bytes = (tile_y_ints + x_qs_ints) * 4 + x_df_flts * 4;

    // Launch the MMQ kernel.
    let blocks_per_row_x = ncols / 32; // Q4_0/Q4_1/Q8_0 all have 32 elems/block.
    let func = dev.get_or_load_func(&kernel_name, &candle_hip_kernels::MMQ_TURBO)?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(nrows, mmq_y) as u32,
            ceil_div(total_b, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, MMQ_NWARPS, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_mmq);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* ncols_y */ total_b as i32,
        /* stride_col_y */ total_b_padded as i32,
        /* stride_row_x */ blocks_per_row_x as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    Ok(Some(HipStorage::wrap_hip_slice(dst, dev.clone())))
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
        GgmlDType::Q4_0 => {
            // Phase 2d v2f: same lane-scratch fix as Q4_1 — drop the
            // per-col bounds check, pad Y to TILE_N.
            let want_legacy = std::env::var("CANDLE_MMQ_VARIANT")
                .map(|s| s == "v2_legacy")
                .unwrap_or(false);
            match tile_n_hint {
                8 => ("mul_mat_q4_0_gfx906_v2", 8),
                16 => ("mul_mat_q4_0_gfx906_v2_tile16", 16),
                _ if want_legacy => ("mul_mat_q4_0_gfx906_v2_tile32", 32),
                _ => ("mul_mat_q4_0_gfx906_v2f_tile32", 32),
            }
        }
        GgmlDType::Q4_1 => {
            // Phase 2d: `CANDLE_MMQ_VARIANT` picks the Q4_1 TILE_N=32
            // variant. See BENCH-PMC-VALU-VMEM-2026-04-11.md for the
            // motivation — the legacy v2 kernel had ~574 lane-scratch
            // instructions (27 % of total) from the compiler's
            // implementation of `if (col >= ncols_y) break` inside a
            // fully-unrolled col loop.
            //   v2f (default for tile_n=32): fast path only, no per-col
            //        bounds check. 1425 instructions, ZERO lane moves.
            //        14.6 % faster prefill than legacy v2 on qwen35-9B.
            //        The Y quant buffer is padded to
            //        `ceil(total_b / tile_n) * tile_n` so OOB reads
            //        land on zeros.
            //   v2d: hoisted bounds check (full-tile fast path +
            //        boundary slow path). 3564 instructions — slower in
            //        practice because both branches are emitted and the
            //        compiler can't consolidate them.
            //   v2_legacy: the original per-col-checked kernel, kept
            //              behind `CANDLE_MMQ_VARIANT=v2_legacy` for
            //              A/B ablation.
            let variant = std::env::var("CANDLE_MMQ_VARIANT").ok();
            let want_legacy = variant.as_deref() == Some("v2_legacy");
            let want_v2d = variant.as_deref() == Some("v2d");
            match tile_n_hint {
                8 => ("mul_mat_q4_1_gfx906_v2", 8),
                16 => ("mul_mat_q4_1_gfx906_v2_tile16", 16),
                64 => ("mul_mat_q4_1_gfx906_v2_tile64", 64),
                _ if want_legacy => ("mul_mat_q4_1_gfx906_v2_tile32", 32),
                _ if want_v2d => ("mul_mat_q4_1_gfx906_v2d_tile32", 32),
                _ => ("mul_mat_q4_1_gfx906_v2f_tile32", 32),
            }
        }
        GgmlDType::Q8_0 => {
            let want_legacy = std::env::var("CANDLE_MMQ_VARIANT")
                .map(|s| s == "v2_legacy")
                .unwrap_or(false);
            match tile_n_hint {
                8 => ("mul_mat_q8_0_gfx906_v2", 8),
                16 => ("mul_mat_q8_0_gfx906_v2_tile16", 16),
                _ if want_legacy => ("mul_mat_q8_0_gfx906_v2_tile32", 32),
                _ => ("mul_mat_q8_0_gfx906_v2f_tile32", 32),
            }
        }
        // P5: K-quant MMQ. Q5_K is the qwen35-9B output head tensor and
        // sits at ~35% of total GPU time in the chunked-vector fallback.
        // Each super-block is 256 K-elements with 8 sub-blocks — the
        // kernel loads one super-block per K iteration, precomputes
        // 8 sub-scales/mins via get_scale_min_k4, and unrolls the
        // 8-sub-block dp4a loop.
        GgmlDType::Q5K => {
            // Phase 2d v2f: Q5_K had the biggest absolute lane-scratch
            // footprint (3608 lane moves vs Q4_1's 574) because the
            // unrolled inner body is 8× larger (8 sub-blocks per K
            // super-block). Dropping the per-col bounds check should
            // give the biggest absolute savings here.
            let want_legacy = std::env::var("CANDLE_MMQ_VARIANT")
                .map(|s| s == "v2_legacy")
                .unwrap_or(false);
            match tile_n_hint {
                8 => ("mul_mat_q5_K_gfx906_v2", 8),
                16 => ("mul_mat_q5_K_gfx906_v2_tile16", 16),
                _ if want_legacy => ("mul_mat_q5_K_gfx906_v2_tile32", 32),
                _ => ("mul_mat_q5_K_gfx906_v2f_tile32", 32),
            }
        }
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
    //
    // Phase 2d: the v2f fast-path kernel drops the per-col bounds check
    // inside the inner loop, which eliminates the ~27 % lane-scratch
    // predicate overhead — but it requires the Y quant buffer to have
    // enough slots for the full TILE_N column tile past `total_b`. We
    // pad `y_size_in_bytes` to `ceil(total_b / tile_n) * tile_n` so
    // OOB col reads land on zeroed memory (alloc_zeros) and contribute
    // nothing to the dp4a sums. Writebacks are still gated on
    // `col < ncols_y == total_b` so the OOB cols are never stored.
    let total_b_padded = total_b.div_ceil(tile_n) * tile_n;
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let bytes_per_row = ncols_padded / q8_1_block_size * q8_1_type_size;
    let y_size_in_bytes = total_b_padded * bytes_per_row;
    // The v2f kernel reads rows `total_b..total_b_padded` as zero-padded
    // OOB so that path still needs `alloc_zeros`. When `total_b` is already
    // tile-aligned (common when `total_b == 1` is padded up or when the
    // prefill length happens to be a tile multiple), only the quantizer
    // writes matter.
    // SAFETY (true branch): quantize_q8_1 fully writes this buffer when
    // total_b == total_b_padded. False branch keeps the zero-fill for
    // the v2f OOB row requirement.
    let mut y_q8_1 = if total_b == total_b_padded {
        unsafe { dev.alloc::<u8>(y_size_in_bytes)? }
    } else {
        dev.alloc_zeros::<u8>(y_size_in_bytes)?
    };
    quantize_q8_1(y, &mut y_q8_1, ncols, total_b, dev)?;

    // 2. Allocate output buffer — row-major `(total_b, nrows)`, matching the
    //    chunked path's convention. The MMQ kernel fully overwrites each
    //    valid `(row, col)`; no zero-fill needed.
    // SAFETY: the mul_mat_q_v2f kernel launched below writes every valid
    // dst element (`col < ncols_y == total_b`) and the buffer size equals
    // that valid range.
    let dst = unsafe { dev.alloc::<f32>(total_b * nrows)? };

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
    //
    //    No padding beyond `total_b` here (unlike the MMQ v2f path), so
    //    quantize_q8_1 writes every block and we can skip the zero fill.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let bytes_per_row = ncols_padded / q8_1_block_size * q8_1_type_size;
    let y_size_in_bytes = total_b * bytes_per_row;
    // SAFETY: quantize_q8_1 fully overwrites all `total_b * bytes_per_row`
    // bytes (no padding beyond total_b in this path).
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, total_b, dev)?;

    // 2. Pre-allocate the full output buffer. The vec kernels fully
    //    overwrite each `(chunk, row)` slot; no zero-fill needed.
    // SAFETY: the chunk loop below writes every (row, col) of dst.
    let dst = unsafe { dev.alloc::<f32>(total_b * nrows)? };

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
    // SAFETY: the MMQ kernel fully writes all x_rows × y_cols output elements.
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
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

    // Output buffer: [batch, topk, n]. The kernel grid is `(n, batch, topk)`
    // with every block writing one row of `n` elements — full coverage,
    // no zero-fill needed.
    // SAFETY: the indexed_moe_forward_f16_f32 kernel fully writes `out`.
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

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

/// M6/M7: Gate Q4_0 / Q4_1 / Q8_0 MoE prefill through the turbo-tile MMQ
/// kernel family (mul_mat_{q4_0,q4_1,q8_0}_turbo_moe_*) instead of the
/// per-token MMVQ kernel.  Returns `Ok(Some(..))` when the fast path
/// handled the call, else `Ok(None)`.
///
/// Entered when ALL of:
///   CANDLE_MMQ_TURBO_PORT=1  (or just CANDLE_MMQ_TURBO_MOE=1 alone)
///   w_dtype ∈ {Q4_0, Q4_1, Q8_0}
///   input_dim1 == 1           (pre-topk activation, [tokens,1,k])
///   k % 32 == 0               (shared QK block size)
///   tokens * topk / n_experts >= MOE_MMQ_MIN_AVG_TOKENS (small batches go MMVQ)
///
/// Pipeline: bucket prep (count / prefix / scatter) → quantize_q8_1_mmq_gather
/// (physically sorts Q8_1 by expert) → mul_mat_{dtype}_turbo_moe.
/// Scatter kernel writes into the same `[tokens, topk, n]` row-major layout
/// the MMVQ path produces, so this is a drop-in replacement.
fn indexed_moe_forward_turbo_moe(
    weight: &HipView<'_, u8>,
    w_shape: &crate::Shape,     // [n_experts, n, k]
    w_dtype: GgmlDType,
    input: &HipSlice<f32>,      // raw f32 [tokens, 1, k] contiguous
    in_shape: &crate::Shape,
    ids: &HipView<'_, u32>,     // [tokens, topk]
    idx_shape: &crate::Shape,
    dev: &HipDevice,
) -> Result<Option<(HipStorage, crate::Shape)>> {
    const MOE_MMQ_MIN_AVG_TOKENS: usize = 4;
    const MMQ_NWARPS: u32 = 4;
    const MMQ_TILE_NE_K: usize = 32;
    const MMQ_TILE_Y_K: usize = MMQ_TILE_NE_K + MMQ_TILE_NE_K / 8; // 36
    const Q8_1_MMQ_BYTES: usize = 144;
    const BIG_BLOCK_ELEMS: usize = 128;

    // Per-dtype MMQ tile row count (see mmq_turbo.cu moe_dtype_traits).
    // Q8_0 uses 64-row tile (U1 LDS fit); Q4_0/Q4_1 keep llama.cpp's 128.
    let (dtype_tag, mmq_y_rows): (&str, usize) = match w_dtype {
        GgmlDType::Q4_0 => ("q4_0", 128),
        GgmlDType::Q4_1 => ("q4_1", 128),
        GgmlDType::Q8_0 => ("q8_0", 64),
        _ => return Ok(None),
    };

    // Gate: CANDLE_MMQ_TURBO_PORT=1 enables M2/M3/M6/M7; CANDLE_MMQ_TURBO_MOE=0
    // can disable just the MoE path without disabling the regular MMQ port.
    let port_on = std::env::var("CANDLE_MMQ_TURBO_PORT").map(|v| v == "1").unwrap_or(false);
    let moe_on = std::env::var("CANDLE_MMQ_TURBO_MOE").map(|v| v != "0").unwrap_or(true);
    if !(port_on && moe_on) {
        return Ok(None);
    }

    let (n_experts, n, k) = w_shape.dims3()?;
    let tokens = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];
    let topk = idx_shape.dims()[1];

    // Shape constraints.  input_dim1==1 because gemma4/qwen reshape to
    // [tokens,1,k] before this call; the bucket prep expects flat
    // [tokens, topk] ids.
    if input_dim1 != 1 || k % 32 != 0 || n_experts > 512 {
        return Ok(None);
    }
    let total_pos = tokens * topk;
    if total_pos == 0 {
        // Empty — let caller handle.
        return Ok(None);
    }
    // Average tokens per expert. Small batches (decode, b=1) benefit more
    // from MMVQ (one block per output row with one token each).
    if total_pos < MOE_MMQ_MIN_AVG_TOKENS * n_experts {
        return Ok(None);
    }

    // Pick mmq_x based on expected bucket fill. Small fill wastes a tile;
    // large fill shrinks grid. mmq_x=32 is a safe default at ~32 tok/expert.
    let avg = total_pos / n_experts;
    let mmq_x: usize = if avg >= 48 {
        64
    } else if avg >= 24 {
        32
    } else if avg >= 12 {
        16
    } else {
        8
    };

    // ---- Bucket prep: counts / expert_bounds / ids_src1 / ids_dst ----
    // counts[n_experts] — zero init via alloc_zeros (3 int atomics don't need warmup).
    let counts = dev.alloc_zeros::<i32>(n_experts)?;
    // total_pos fits comfortably in 256 threads per block.
    {
        let func = dev.get_or_load_func("moe_bucket_count", &candle_hip_kernels::MMQ_TURBO)?;
        let cfg = hipdarc::driver::LaunchConfig {
            grid_dim: (ceil_div(total_pos, 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(ids);
        barg!(b, total_pos as i32);
        b.arg(&counts);
        unsafe { b.launch(cfg) }.w()?;
    }

    // expert_bounds[n_experts+1]. Single-block scan (512 threads covers ≤ 512 experts).
    let expert_bounds = dev.alloc_zeros::<i32>(n_experts + 1)?;
    {
        let func = dev.get_or_load_func("moe_bucket_prefix", &candle_hip_kernels::MMQ_TURBO)?;
        let cfg = hipdarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&counts);
        barg!(b, n_experts as i32);
        b.arg(&expert_bounds);
        unsafe { b.launch(cfg) }.w()?;
    }

    // ids_src1 / ids_dst / cursor (zeroed). Scatter via atomicAdd(cursor[e], 1).
    let cursor = dev.alloc_zeros::<i32>(n_experts)?;
    let ids_src1 = unsafe { dev.alloc::<i32>(total_pos)? };
    let ids_dst = unsafe { dev.alloc::<i32>(total_pos)? };
    {
        let func = dev.get_or_load_func("moe_bucket_scatter", &candle_hip_kernels::MMQ_TURBO)?;
        let cfg = hipdarc::driver::LaunchConfig {
            grid_dim: (ceil_div(total_pos, 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(ids);
        barg!(b, total_pos as i32, topk as i32);
        b.arg(&expert_bounds);
        b.arg(&cursor);
        b.arg(&ids_src1);
        b.arg(&ids_dst);
        unsafe { b.launch(cfg) }.w()?;
    }

    // ---- Gather-quantize f32 input → block_q8_1_mmq, PHYSICALLY SORTED by expert.
    // One bucket_pos per output col; each reads the original token via
    // `ids_src1[bucket_pos]` and writes to `Y[big_block, bucket_pos]`.  After
    // this kernel, Y is contiguous per expert — the MMQ kernel reads stride-1.
    // +1 trailing big-block is zeroed so the K-loop second-half read at the
    // last kb0 iter lands on zero (harmless, d_y=0).
    let n_big_blocks_real = ceil_div(k, BIG_BLOCK_ELEMS);
    let n_big_blocks_alloc = n_big_blocks_real + 1;
    let y_mmq_size_bytes = n_big_blocks_alloc * total_pos * Q8_1_MMQ_BYTES;
    let y_mmq = dev.alloc_zeros::<u8>(y_mmq_size_bytes)?;
    {
        let qkernel =
            dev.get_or_load_func("quantize_q8_1_mmq_gather", &candle_hip_kernels::MMQ_TURBO)?;
        let qcfg = hipdarc::driver::LaunchConfig {
            grid_dim: (n_big_blocks_real as u32, total_pos as u32, 1),
            block_dim: (BIG_BLOCK_ELEMS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut qb = qkernel.builder();
        let input_view = input.slice(0..input.len());
        qb.arg(&input_view);
        qb.arg(&ids_src1);
        qb.arg(&y_mmq);
        barg!(qb, k as i32, total_pos as i32);
        unsafe { qb.launch(qcfg) }.w()?;
    }

    // ---- Output: [tokens, topk, n] row-major. Every (token, topk_slot, row)
    // is written by the MMQ kernel via ids_dst scatter (one bucket pos per slot). ----
    // SAFETY: the MoE MMQ kernel writes to every dst[ids_dst_shared[j] * n + row_g]
    // for bucket_pos ∈ [0, total_pos) (the scatter covers all of ids_dst)
    // and row_g ∈ [0, n); the `need_check` variant gates row_g < nrows_x for
    // the last row-tile.
    let outsize = tokens * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

    // ---- MMQ MoE launch ----
    let need_check = n % mmq_y_rows != 0;
    let suffix = if need_check { "checked" } else { "unchecked" };
    let kernel_name = format!("mul_mat_{dtype_tag}_turbo_moe_x{mmq_x}_{suffix}");

    let tile_y_ints = {
        let raw = mmq_x * MMQ_TILE_Y_K;
        let pad_u = MMQ_NWARPS as usize * WARP_SIZE;
        raw.div_ceil(pad_u) * pad_u
    };
    // Q8_0 doubles x_qs per row (8 ints/block, not 4 after nibble pack);
    // Q4_0 and Q4_1 use the compact layout.  See moe_dtype_traits in
    // mmq_turbo.cu.
    let x_qs_ints = if matches!(w_dtype, GgmlDType::Q8_0) {
        mmq_y_rows * (2 * MMQ_TILE_NE_K + 1)
    } else {
        mmq_y_rows * (MMQ_TILE_NE_K + 1)
    };
    let x_df_flts = mmq_y_rows * 8 + mmq_y_rows / 4;
    let shared_mem_bytes = (tile_y_ints + x_qs_ints) * 4 + x_df_flts * 4;

    let blocks_per_row_x = k / 32;
    let stride_expert_blocks = n * blocks_per_row_x;
    let grid_y = ceil_div(total_pos, mmq_x) as u32;

    let func = dev.get_or_load_func(&kernel_name, &candle_hip_kernels::MMQ_TURBO)?;
    let cfg = hipdarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(n, mmq_y_rows) as u32,
            grid_y,
            n_experts as u32,
        ),
        block_dim: (WARP_SIZE as u32, MMQ_NWARPS, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    };
    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&y_mmq);
    builder.arg(&ids_dst);
    builder.arg(&expert_bounds);
    builder.arg(&out);
    barg!(
        builder,
        k as i32,
        n as i32,
        total_pos as i32,           // stride_col_y (big-block stride in Y = total bucket count)
        blocks_per_row_x as i32,
        stride_expert_blocks as i32,
        n as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok(Some((
        HipStorage::wrap_hip_slice(out, dev.clone()),
        out_shape.into(),
    )))
}

/// Launch the dtype-specific `indexed_moe_forward_*_q8_1` kernel given a
/// pre-built Q8_1 activation buffer.  Shared between the public preq8 API
/// (where the caller has already quantised the input — saves a kernel
/// dispatch when the same activation feeds multiple matmuls, e.g. router +
/// gate_up_exps in a MoE FFN) and `indexed_moe_forward_fused_q8_1_input`.
///
/// The Q8_1 layout is the standard per-row `block_q8_1` × `num_blocks_per_row`
/// produced by `quantize_q8_1` — identical to what `fwd_with_preq8` consumes.
#[allow(clippy::too_many_arguments)]
fn launch_indexed_moe_forward_with_q8(
    weight: &HipView<'_, u8>,
    w_shape: &crate::Shape,
    w_dtype: GgmlDType,
    input_q8_1: &HipView<'_, u8>,
    in_shape: &crate::Shape,           // [batch, topk_or_1, k]
    ids: &HipView<'_, u32>,
    idx_shape: &crate::Shape,          // [batch, topk]
    dev: &HipDevice,
) -> Result<(HipStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];
    let topk = idx_shape.dims()[1];
    let k_padded = pad(k, MATRIX_ROW_PADDING);

    // SAFETY: indexed_moe_forward_*_q8_1 writes every element of `out`.
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

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
    builder.arg(input_q8_1);
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

    // M6/M7: Q4_0 / Q4_1 / Q8_0 MoE MMQ fast path (turbo-tile gather-by-expert).
    // Bypasses the MMVQ per-token kernels for prefill-shaped MoE calls.  Returns
    // None when the path is disabled or the shape doesn't qualify — fall through.
    if matches!(w_dtype, GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q8_0) {
        if let Some(out) = indexed_moe_forward_turbo_moe(
            weight, w_shape, w_dtype, input, in_shape, ids, idx_shape, dev,
        )? {
            return Ok(out);
        }
    }

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;

    // SAFETY: quantize_q8_1 fully writes input_quant.
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    let input_view = input.slice(0..input.len());
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    let q8_view = input_quant.slice(0..input_quant.len());
    launch_indexed_moe_forward_with_q8(
        weight, w_shape, w_dtype, &q8_view, in_shape, ids, idx_shape, dev,
    )
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

    /// MoE matmul that consumes a pre-quantized Q8_1 activation buffer
    /// (Phase O1 — saves the per-call `quantize_q8_1` dispatch when the
    /// same activation feeds multiple matmuls, e.g. router + gate + up).
    ///
    /// The Q8_1 layout matches `quantize_q8_1`: `(batch * input_dim1)` rows
    /// of `k_padded / QK8_1` blocks of 36 bytes each.  Caller is responsible
    /// for producing it (e.g. via `quantize_q8_1` on `x_flat`) and ensuring
    /// `k` matches `self_shape`'s last dim.
    ///
    /// Falls back to the regular re-quantising path for F16 weights and
    /// returns an error for any other unsupported dtype.
    pub fn indexed_moe_forward_preq8(
        &self,
        self_shape: &crate::Shape,
        input_q8_1: &HipView<'_, u8>,
        in_shape: &crate::Shape,
        ids: &HipView<'_, u32>,
        idx_shape: &crate::Shape,
    ) -> Result<(HipStorage, crate::Shape)> {
        if !matches!(
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
            crate::bail!(
                "indexed_moe_forward_preq8: dtype {:?} not supported (use indexed_moe_forward)",
                self.dtype()
            );
        }
        launch_indexed_moe_forward_with_q8(
            &self.data.inner.slice(0..self.data.inner.len()),
            self_shape,
            self.dtype(),
            input_q8_1,
            in_shape,
            ids,
            idx_shape,
            &self.device,
        )
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
            GgmlDType::Iq4Xs => deq::<crate::quantized::BlockIq4Xs>(&buffer, block_len, &mut out),
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

    pub fn data_padded(&self) -> &PaddedHipSlice {
        &self.data
    }

    /// Mat-vec multiply using a **pre-quantized Q8_1** activation buffer.
    ///
    /// Skips the per-call `quantize_q8_1` dispatch — the caller is responsible
    /// for producing `y_q8_1` (e.g. via `rmsnorm_q8_fused` or a manual
    /// `quantize_q8_1` call) and ensuring it matches `ncols` / `ncols_padded`.
    ///
    /// Returns `(output_storage, output_shape)`.
    pub fn fwd_with_preq8(
        &self,
        self_shape: &crate::Shape,
        y_q8_1: &hipdarc::driver::HipView<'_, u8>,
        b_size: usize,
        rhs_shape: &[usize],
    ) -> Result<(HipStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);

        if b_size == 0 || b_size > 8 {
            crate::bail!("fwd_with_preq8: b_size must be 1..=8, got {b_size}");
        }

        let dst = unsafe { self.device.alloc::<f32>(nrows * b_size)? };
        let dst_view = dst.slice(0..dst.len());
        launch_mul_mat_vec_q8_1_chunk(
            &self.data,
            y_q8_1,
            &dst_view,
            self.dtype,
            ncols,
            nrows,
            b_size,
            ncols_padded,
            &self.device,
        )?;

        let mut out_shape = rhs_shape.to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((HipStorage::wrap_hip_slice(dst, self.device.clone()), out_shape.into()))
    }

    /// Fused gate+up Q4_0 MMVQ on pre-quantized Q8_1 input. Produces two
    /// output tensors from a single kernel launch (vs two separate MMVQs).
    /// `self` is the gate weight; `w_up` is the up weight. Returns
    /// `(gate_out_storage, up_out_storage)` each shape `(b_size, nrows)`.
    pub fn fwd_gate_up_preq8(
        &self,
        w_up: &QHipStorage,
        self_shape: &crate::Shape,
        y_q8_1: &hipdarc::driver::HipView<'_, u8>,
    ) -> Result<(HipStorage, HipStorage)> {
        let (nrows, ncols) = self_shape.dims2()?;
        if self.dtype != GgmlDType::Q4_0 || w_up.dtype != GgmlDType::Q4_0 {
            crate::bail!("fwd_gate_up_preq8: both weights must be Q4_0");
        }
        let dst_gate = unsafe { self.device.alloc::<f32>(nrows)? };
        let dst_up = unsafe { self.device.alloc::<f32>(nrows)? };
        let v_gate = dst_gate.slice(0..dst_gate.len());
        let v_up = dst_up.slice(0..dst_up.len());
        launch_mul_mat_vec_q4_0_gate_up_fused(
            &self.data, &w_up.data, y_q8_1,
            &v_gate, &v_up,
            ncols, nrows, &self.device,
        )?;
        Ok((
            HipStorage::wrap_hip_slice(dst_gate, self.device.clone()),
            HipStorage::wrap_hip_slice(dst_up, self.device.clone()),
        ))
    }

    /// Fused W_down @ hidden_q8_1 + residual. The output overwrites `dst`;
    /// caller supplies the pre-allocated (nrows-shaped) destination.
    pub fn fwd_down_residual_preq8(
        &self,
        self_shape: &crate::Shape,
        y_q8_1: &hipdarc::driver::HipView<'_, u8>,
        residual: &hipdarc::driver::HipView<'_, f32>,
    ) -> Result<HipStorage> {
        let (nrows, ncols) = self_shape.dims2()?;
        if self.dtype != GgmlDType::Q4_0 {
            crate::bail!("fwd_down_residual_preq8: weight must be Q4_0");
        }
        let dst = unsafe { self.device.alloc::<f32>(nrows)? };
        let dst_view = dst.slice(0..dst.len());
        launch_mul_mat_vec_q4_0_down_residual(
            &self.data, y_q8_1, residual, &dst_view,
            ncols, nrows, &self.device,
        )?;
        Ok(HipStorage::wrap_hip_slice(dst, self.device.clone()))
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
    let tail_len = MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let padded_len = data.len() + tail_len;
    // Phase O3: alloc_uninit + targeted tail zero-fill.  The body (data.len()
    // bytes) is immediately overwritten by `memcpy_htod`, so zeroing it was
    // pure waste (the rocprofv3 trace showed `__amd_rocclr_fillBufferAligned`
    // from model-load `alloc_zeros` as 18 % of total GPU time on a cold
    // 24-tok decode — > 500 ms of wasted fill on a 42 GB model).  Only the
    // trailing padding tail needs to stay zero — MMQ / MMVQ kernels pad
    // OOB K-reads into it and rely on d=0 / qs=0 for no contribution.
    // SAFETY: we fully write `[0, data.len())` via memcpy_htod and
    // `[data.len(), padded_len)` via the tail memset below before returning.
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    device.stream().memcpy_htod(&mut inner, data).w()?;
    if tail_len > 0 {
        let tail_ptr = unsafe { (inner.device_ptr() as *mut u8).add(data.len()) };
        let rc = unsafe {
            hipdarc::sys::hipMemsetAsync(
                tail_ptr as *mut std::ffi::c_void,
                0,
                tail_len,
                device.stream().raw(),
            )
        };
        if rc != hipdarc::sys::hipError_t::hipSuccess {
            crate::bail!("hipMemsetAsync(tail) failed: {rc:?}");
        }
    }
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
