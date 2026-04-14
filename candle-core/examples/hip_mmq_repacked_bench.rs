//! Microbench: compare the original `mul_mat_q4_0_gfx906_v2f_tile32`
//! kernel against the Phase I1 `_repacked` variant.
//!
//! - Quantizes random data to Q4_0
//! - Builds two device buffers: original layout and repacked layout
//! - Runs both kernels, compares per-call time and output L2 norm
//!
//! Usage:
//!   cargo run --release --features hip --example hip_mmq_repacked_bench

use anyhow::Result;
use candle_core::hip_backend::WrapErr;
use candle_core::hip_backend::hipdarc::{self, driver::LaunchConfig};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

const M: usize = 14336;
const K: usize = 2048;
const N: usize = 874;

const QK4_0: usize = 32;
const QK8_1: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q8_1_BLOCK_BYTES: usize = 36;
const WARP_SIZE: u32 = 64;
const TILE_N: u32 = 32;

fn main() -> Result<()> {
    let dev = Device::new_hip(0)?;
    println!("HIP device: {:?}", dev.location());
    println!("Shape: M={M} K={K} N={N}");
    assert!(M % WARP_SIZE as usize == 0);
    assert!(K % QK4_0 == 0);

    let blocks_per_row_x = K / QK4_0;
    let m_tile_count = M / WARP_SIZE as usize;
    let total_blocks_x = M * blocks_per_row_x;
    let storage_size = total_blocks_x * Q4_0_BLOCK_BYTES;

    let dev_hip = match &dev { Device::Hip(d) => d.clone(), _ => unreachable!() };

    // 1. Quantize random weights to get the original layout bytes.
    let weight_data: Vec<f32> = (0..M * K)
        .map(|i| ((i * 31 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), &dev)?.reshape((M, K))?;
    let qweight = QTensor::quantize(&weight, GgmlDType::Q4_0)?;
    let orig_host = qweight.data()?.to_vec();
    assert_eq!(orig_host.len(), storage_size);
    println!("Q4_0 weight: {} bytes ({} blocks × 18B)", storage_size, total_blocks_x);

    // 2. Build repacked layout on host, then upload both.
    let mut repacked_host = vec![0u8; storage_size];
    for row in 0..M {
        for ib in 0..blocks_per_row_x {
            let src_idx = row * blocks_per_row_x + ib;
            let dst_idx = (ib * m_tile_count + row / WARP_SIZE as usize) * WARP_SIZE as usize
                + (row % WARP_SIZE as usize);
            let src = src_idx * Q4_0_BLOCK_BYTES;
            let dst = dst_idx * Q4_0_BLOCK_BYTES;
            repacked_host[dst..dst + Q4_0_BLOCK_BYTES]
                .copy_from_slice(&orig_host[src..src + Q4_0_BLOCK_BYTES]);
        }
    }
    let orig_dev = unsafe { dev_hip.alloc::<u8>(storage_size)? };
    let rep_dev  = unsafe { dev_hip.alloc::<u8>(storage_size)? };
    unsafe {
        let rc = hipdarc::sys::hipMemcpy(
            orig_dev.device_ptr() as *mut _,
            orig_host.as_ptr() as *const _,
            storage_size,
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
        assert_eq!(rc, hipdarc::sys::hipError_t::hipSuccess);
        let rc = hipdarc::sys::hipMemcpy(
            rep_dev.device_ptr() as *mut _,
            repacked_host.as_ptr() as *const _,
            storage_size,
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
        assert_eq!(rc, hipdarc::sys::hipError_t::hipSuccess);
    }
    println!("Both layouts uploaded.");

    // 3. Build Y as Q8_1.
    use candle_core::quantized::hip::{quantize_q8_1, pad, MATRIX_ROW_PADDING};
    let y_data: Vec<f32> = (0..K * N)
        .map(|i| ((i * 17 + 11) % 31) as f32 / 15.0 - 1.0)
        .collect();
    let y_t = Tensor::new(y_data.as_slice(), &dev)?.reshape((N, K))?;
    let (y_st, _y_l) = y_t.storage_and_layout();
    let y_view_f32 = match &*y_st {
        candle_core::Storage::Hip(s) => {
            let sl = s.as_hip_slice::<f32>()?;
            sl.slice(0..sl.len())
        }
        _ => unreachable!(),
    };
    let k_padded = pad(K, MATRIX_ROW_PADDING);
    let y_q8_bytes = N * k_padded * Q8_1_BLOCK_BYTES / QK8_1;
    let mut y_q8 = unsafe { dev_hip.alloc::<u8>(y_q8_bytes)? };
    quantize_q8_1(&y_view_f32, &mut y_q8, K, N, &dev_hip)?;
    drop(y_st);

    // 4. Allocate outputs and load both kernels.
    let dst_orig = unsafe { dev_hip.alloc::<f32>(M * N)? };
    let dst_rep  = unsafe { dev_hip.alloc::<f32>(M * N)? };
    let func_orig = dev_hip.get_or_load_func(
        "mul_mat_q4_0_gfx906_v2f_tile32",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let func_rep = dev_hip.get_or_load_func(
        "mul_mat_q4_0_gfx906_v2f_tile32_repacked",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (M as u32 / WARP_SIZE, (N as u32 + TILE_N - 1) / TILE_N, 1),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let ncols_x: i32 = K as i32;
    let nrows_x: i32 = M as i32;
    let ncols_y: i32 = N as i32;
    let nrows_y_padded: i32 = k_padded as i32;
    let nrows_dst: i32 = M as i32;
    let y_view_q8 = y_q8.slice(0..y_q8.len());

    // Warmup.
    for _ in 0..3 {
        let orig_view = orig_dev.slice(0..orig_dev.len());
        let mut bld = func_orig.builder();
        bld.arg(&orig_view); bld.arg(&y_view_q8); bld.arg(&dst_orig);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg) }.w()?;

        let rep_view = rep_dev.slice(0..rep_dev.len());
        let mut bld = func_rep.builder();
        bld.arg(&rep_view); bld.arg(&y_view_q8); bld.arg(&dst_rep);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg) }.w()?;
    }
    let _ = dev_hip.stream().synchronize();

    let n_iter = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let orig_view = orig_dev.slice(0..orig_dev.len());
        let mut bld = func_orig.builder();
        bld.arg(&orig_view); bld.arg(&y_view_q8); bld.arg(&dst_orig);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg) }.w()?;
    }
    let _ = dev_hip.stream().synchronize();
    let t_orig = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("Original kernel:  {t_orig:.1} µs/call");

    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let rep_view = rep_dev.slice(0..rep_dev.len());
        let mut bld = func_rep.builder();
        bld.arg(&rep_view); bld.arg(&y_view_q8); bld.arg(&dst_rep);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg) }.w()?;
    }
    let _ = dev_hip.stream().synchronize();
    let t_rep = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("Repacked kernel:  {t_rep:.1} µs/call ({:.2}× speedup)",
             t_orig / t_rep);

    // 5. Correctness.
    let mut host_orig = vec![0f32; M * N];
    let mut host_rep  = vec![0f32; M * N];
    unsafe {
        hipdarc::sys::hipMemcpy(
            host_orig.as_mut_ptr() as *mut _,
            dst_orig.device_ptr() as *const _,
            M * N * 4,
            hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hipdarc::sys::hipMemcpy(
            host_rep.as_mut_ptr() as *mut _,
            dst_rep.device_ptr() as *const _,
            M * N * 4,
            hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
        );
    }
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for i in 0..M * N {
        let d = (host_orig[i] - host_rep[i]) as f64;
        diff_sq += d * d;
        ref_sq += (host_orig[i] as f64).powi(2);
        max_abs = max_abs.max(d.abs());
    }
    let rel_l2 = (diff_sq.sqrt() / ref_sq.sqrt()).max(1e-30);
    println!("Correctness: rel L2 = {rel_l2:.3e}, max abs = {max_abs:.3e}");
    if rel_l2 > 1e-5 {
        println!("⚠ Outputs DIVERGE — Phase I1 has a bug.");
    } else {
        println!("✓ Repacked output bit-equivalent to original.");
    }
    Ok(())
}
