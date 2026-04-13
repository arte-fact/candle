use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
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
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
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
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

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

        let cfg = cudarc::driver::LaunchConfig {
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
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
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
        GgmlDType::Mxfp4 => ("dequantize_block_mxfp4_f32", false, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
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
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
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
        GgmlDType::Mxfp4 => ("dequantize_block_mxfp4_f16", false, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
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
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
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
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
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
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
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
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(x_rows * y_cols)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
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
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
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

    let input_view = input.slice(0..);
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
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = cudarc::driver::LaunchConfig {
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
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
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

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Keep BF16 weights on GPU without upcasting to F32. Returns an owned
    /// `CudaStorage<BF16>` by transmuting the device byte buffer to
    /// `CudaSlice<bf16>` and cloning into a fresh allocation. ~2 ms/GiB D2D
    /// on an RTX 3090 — a one-time cost, vs. the ~15 s CPU roundtrip that
    /// also doubles VRAM.
    pub fn as_bf16_storage(&self, elem_count: usize) -> Result<CudaStorage> {
        if self.dtype != GgmlDType::BF16 {
            crate::bail!("as_bf16_storage: dtype must be BF16, got {:?}", self.dtype);
        }
        let n_bytes = elem_count * std::mem::size_of::<half::bf16>();
        if n_bytes > self.data.len {
            crate::bail!(
                "as_bf16_storage: elem_count ({elem_count}) exceeds buffer bytes ({})",
                self.data.len
            );
        }
        // SAFETY: `inner` stores raw BF16 bytes (candle wrote BF16 data into
        // the u8 buffer at load time). Reinterpreting as CudaView<bf16> of the
        // matching length is safe since bf16 is 2-byte aligned and the buffer
        // has padded length >= `n_bytes`.
        let view: CudaView<half::bf16> = unsafe {
            self.data
                .inner
                .slice(..n_bytes)
                .transmute::<half::bf16>(elem_count)
                .ok_or_else(|| {
                    crate::Error::Msg("as_bf16_storage: CudaSlice::transmute failed".into())
                })?
        };
        let stream = self.device.cuda_stream();
        let owned: CudaSlice<half::bf16> = stream.clone_dtod(&view).w()?;
        Ok(CudaStorage::wrap_cuda_slice(owned, self.device.clone()))
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
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
                | GgmlDType::Mxfp4
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
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

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
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
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
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
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
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
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
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
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
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
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
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
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
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
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

// Thread-local *double-buffered* pinned staging for H→D copies during
// weight loading. Two buffers ping-pong: while buffer A is draining its
// async H→D, buffer B fills the next tensor. The CudaEvent recorded after
// each memcpy_htod is checked the next time that buffer is reused — only
// then do we sync, so successive uploads overlap with their CPU-side prep.
//
// With one buffer (single-buffered) we had to call `stream.synchronize()`
// after every copy to avoid overwriting the buffer while the GPU was
// still reading from it; that cost ~5–8 s on an 851-tensor load. The
// ping-pong eliminates that hot stall — each sync hits an event that's
// already complete (its copy finished while the other buffer was working).
struct PinnedSlot {
    buf: Option<cudarc::driver::PinnedHostSlice<u8>>,
    in_flight: Option<cudarc::driver::CudaEvent>,
}
thread_local! {
    static PINNED_BUFS: std::cell::RefCell<Option<[PinnedSlot; 2]>>
        = const { std::cell::RefCell::new(None) };
    static NEXT_IDX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Pick the next ping-pong slot, sync any in-flight H→D from it, and
/// (re)allocate the pinned buffer if it can't fit `n_bytes`. Returns the
/// chosen index on success, `None` if pinned alloc failed (caller falls
/// back to pageable). Caller must, in a subsequent `PINNED_BUFS.with`
/// borrow, fill the buffer, queue the H→D, and record the new event into
/// `PinnedSlot::in_flight`.
fn next_pinned_slot(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n_bytes: usize,
) -> Result<Option<usize>> {
    let idx = NEXT_IDX.with(|c| {
        let i = c.get();
        c.set((i + 1) % 2);
        i
    });
    PINNED_BUFS.with(|cell| -> Result<Option<usize>> {
        let mut slot_array = cell.borrow_mut();
        if slot_array.is_none() {
            *slot_array = Some([
                PinnedSlot { buf: None, in_flight: None },
                PinnedSlot { buf: None, in_flight: None },
            ]);
        }
        let slot = &mut slot_array.as_mut().unwrap()[idx];
        // Drain any prior H→D from this buffer before we overwrite it.
        // Reused event — re-record on next memcpy via cuEventRecord.
        if let Some(ev) = slot.in_flight.as_ref() {
            ev.synchronize().w()?;
        }
        let need_realloc = slot.buf.as_ref().map_or(true, |b| b.len() < n_bytes);
        if need_realloc {
            let new_cap = match &slot.buf {
                Some(b) => std::cmp::max(n_bytes, b.len() + b.len() / 2),
                None => n_bytes,
            };
            slot.buf = None;
            match unsafe { stream.context().alloc_pinned::<u8>(new_cap) } {
                Ok(b) => slot.buf = Some(b),
                Err(_) => return Ok(None),
            }
        }
        Ok(Some(idx))
    })
}

/// Record/re-record the event on this slot's stream so the next reuse
/// will wait on it. Allocates the event lazily on first use, then reuses
/// (cuEventRecord re-captures the stream's current point each call).
///
/// Default flags = `DISABLE_TIMING` only, so `synchronize` busy-spins
/// rather than OS-sleeps. For loads where the event is almost always
/// already complete by the next reuse (CPU prep takes longer than H→D
/// drain), spinning is faster than the BLOCKING_SYNC wakeup path.
fn record_in_flight_event(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    idx: usize,
) -> Result<()> {
    PINNED_BUFS.with(|cell| -> Result<()> {
        let mut slots = cell.borrow_mut();
        let arr = slots.as_mut().expect("slot array initialized in next_pinned_slot");
        let slot = &mut arr[idx];
        // Reuse this slot's existing event if any; otherwise allocate one
        // for the lifetime of this thread.
        if slot.in_flight.is_none() {
            slot.in_flight = Some(stream.context().new_event(None).w()?);
        }
        slot.in_flight.as_ref().unwrap().record(stream).w()?;
        Ok(())
    })
}

/// Stage `data` through one of the two ping-pong pinned buffers and enqueue
/// an async `memcpy_htod` to `dst[0..data.len())`. Returns `Ok(Some(()))` on
/// the pinned fast path, `Ok(None)` if pinned alloc isn't available.
///
/// Per-call sync is replaced with per-buffer event sync inside
/// `next_pinned_slot`: on average the event has already completed when we
/// next pick that buffer, so the wait is free.
fn pinned_staged_copy(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    data: &[u8],
    dst: &mut cudarc::driver::CudaSlice<u8>,
) -> Result<Option<()>> {
    let n = data.len();
    if n == 0 {
        return Ok(Some(()));
    }
    let idx = match next_pinned_slot(stream, n)? {
        Some(i) => i,
        None => return Ok(None),
    };
    PINNED_BUFS.with(|cell| -> Result<()> {
        let mut slots = cell.borrow_mut();
        let slot = &mut slots.as_mut().unwrap()[idx];
        let pinned = slot.buf.as_mut().expect("just allocated");
        // SAFETY: pinned alive, len() ≥ n, exclusively borrowed via RefCell.
        let pinned_dst = unsafe { std::slice::from_raw_parts_mut(pinned.as_mut_ptr().w()?, n) };
        pinned_dst.copy_from_slice(data);
        let pinned_src: &[u8] =
            unsafe { std::slice::from_raw_parts(pinned.as_ptr().w()?, n) };
        let mut dst_view = dst.slice_mut(..n);
        stream.memcpy_htod(pinned_src, &mut dst_view).w()?;
        Ok(())
    })?;
    record_in_flight_event(stream, idx)?;
    Ok(Some(()))
}

/// Concat-load: pread N segments directly into a single pinned staging
/// buffer, then one async H→D copy of the concatenated bytes. Used by
/// `qmatmul_concat_rows` to fuse QKV / gate+up loads — saves an extra
/// CPU-side Vec assembly that would otherwise dominate load time for
/// transformer models where every layer has fused projections.
pub fn load_quantized_concat_from_blob(
    device: &CudaDevice,
    blob: &super::gguf_file::GgufBlob,
    segments: &[(u64, usize)],
    total_bytes: usize,
    dtype: GgmlDType,
    block_size: usize,
    type_size: usize,
) -> Result<super::QStorage> {
    let padded_len = total_bytes + MATRIX_ROW_PADDING * type_size / block_size;
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    let stream = device.cuda_stream();

    let idx = next_pinned_slot(&stream, total_bytes)?;
    let used_pinned = if let Some(idx) = idx {
        PINNED_BUFS.with(|cell| -> Result<()> {
            let mut slots = cell.borrow_mut();
            let slot = &mut slots.as_mut().unwrap()[idx];
            let pinned = slot.buf.as_mut().expect("just allocated");
            let pinned_dst = unsafe {
                std::slice::from_raw_parts_mut(pinned.as_mut_ptr().w()?, total_bytes)
            };
            let mut cursor = 0usize;
            for &(off, n) in segments {
                blob.read_into(off, &mut pinned_dst[cursor..cursor + n])?;
                cursor += n;
            }
            let pinned_src: &[u8] =
                unsafe { std::slice::from_raw_parts(pinned.as_ptr().w()?, total_bytes) };
            let mut dst_view = inner.slice_mut(..total_bytes);
            stream.memcpy_htod(pinned_src, &mut dst_view).w()?;
            Ok(())
        })?;
        record_in_flight_event(&stream, idx)?;
        true
    } else {
        false
    };
    if !used_pinned {
        let mut combined: Vec<u8> = Vec::with_capacity(total_bytes);
        for &(off, n) in segments {
            let raw = blob.read_to_vec(off, n)?;
            combined.extend_from_slice(&raw);
        }
        device.memcpy_htod(combined.as_slice(), &mut inner.slice_mut(..total_bytes))?;
    }
    if padded_len > total_bytes {
        let mut tail = inner.slice_mut(total_bytes..padded_len);
        stream.memset_zeros(&mut tail).w()?;
    }
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: total_bytes,
        },
        device: device.clone(),
        dtype,
    }))
}

/// Fused read-and-upload path: pread directly from `blob` at `offset` into
/// the thread-local pinned staging buffer, then async H→D copy. Skips the
/// intermediate `Vec<u8>` that `GgufBlob::read_to_vec` would allocate, which
/// strace shows accounts for ~20 s of pread time on a 17 GB Q4_1 load (one
/// linear copy from page cache into a fresh anonymous Vec per tensor).
pub fn load_quantized_from_blob(
    device: &CudaDevice,
    blob: &super::gguf_file::GgufBlob,
    offset: u64,
    n_bytes: usize,
    dtype: GgmlDType,
    block_size: usize,
    type_size: usize,
) -> Result<super::QStorage> {
    let padded_len = n_bytes + MATRIX_ROW_PADDING * type_size / block_size;
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    let stream = device.cuda_stream();

    // Ping-pong pinned: pick a buffer (sync prior in-flight H→D from this
    // slot if any), pread into it, queue async H→D, record event for the
    // next reuse to wait on. The other buffer drains in parallel.
    let idx = next_pinned_slot(&stream, n_bytes)?;
    let used_pinned = if let Some(idx) = idx {
        PINNED_BUFS.with(|cell| -> Result<()> {
            let mut slots = cell.borrow_mut();
            let slot = &mut slots.as_mut().unwrap()[idx];
            let pinned = slot.buf.as_mut().expect("just allocated");
            // SAFETY: just-allocated, exclusively owned via RefCell, len ≥ n_bytes.
            let pinned_dst =
                unsafe { std::slice::from_raw_parts_mut(pinned.as_mut_ptr().w()?, n_bytes) };
            blob.read_into(offset, pinned_dst)?;
            let pinned_src: &[u8] =
                unsafe { std::slice::from_raw_parts(pinned.as_ptr().w()?, n_bytes) };
            let mut dst_view = inner.slice_mut(..n_bytes);
            stream.memcpy_htod(pinned_src, &mut dst_view).w()?;
            Ok(())
        })?;
        record_in_flight_event(&stream, idx)?;
        true
    } else {
        false
    };
    if !used_pinned {
        let raw = blob.read_to_vec(offset, n_bytes)?;
        device.memcpy_htod(raw.as_slice(), &mut inner.slice_mut(..n_bytes))?;
    }
    if padded_len > n_bytes {
        let mut tail = inner.slice_mut(n_bytes..padded_len);
        stream.memset_zeros(&mut tail).w()?;
    }
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: n_bytes,
        },
        device: device.clone(),
        dtype,
    }))
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    // Skip the full-buffer zero-fill that `alloc_zeros` does — we overwrite
    // `[0..data.len())` with the tensor bytes immediately, and only the
    // small `[data.len()..padded_len)` tail needs to be zero-initialized for
    // kernels that read a rounded-up row stride. For 851 tensors this saves
    // ~450 ms of pointless `cudaMemset` on a 17 GB Q4_1 load.
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    // Pinned-staging fast path with a thread-local reusable buffer.
    //
    // `cudaMemcpyHtoDAsync` from pageable host memory bounces through a
    // driver-managed pinned pool that's both small and serialized on a
    // context lock — measured at ~0.5 GB/s for a 17 GB Qwen3.5-27B load.
    // Copying first into our own pinned buffer lets the H→D leg DMA at
    // full PCIe bandwidth.
    //
    // Per-call `cuMemHostAlloc` is too expensive to amortize (851 × ~10 ms
    // adds 8 s of pure overhead). A thread-local buffer that grows to the
    // largest tensor seen reuses the same pinned region for the whole load.
    // Each rayon worker has its own buffer so concurrent uploads don't
    // contend on a mutex.
    let stream = device.cuda_stream();
    if pinned_staged_copy(&stream, data, &mut inner)?.is_none() {
        // Pinned staging unavailable (driver out of pinned host memory):
        // fall back to the slow pageable path so loads still complete.
        device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    }
    if padded_len > data.len() {
        let mut tail = inner.slice_mut(data.len()..padded_len);
        stream.memset_zeros(&mut tail).w()?;
    }
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
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
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
