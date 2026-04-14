//! Phase I3 bench: dequant-to-f16 + rocBLAS sgemm vs custom MMQ Q4_0 kernel.
//!
//! Compares end-to-end time of:
//!   A. mul_mat_q4_0_gfx906_v2f_tile32 (custom MMQ on Q4_0 directly)
//!   B. dequantize_block_q4_0_f16 → rocblas_gemm_strided_batched_ex
//!      (W f16 × x f16 → f32 accumulate)
//!
//! Per-call measurement excludes the one-time dequant cost (which happens
//! once at GGUF load time, not per matmul).
//!
//! Usage:
//!   cargo run --release --features hip --example hip_mmq_rocblas_bench

use anyhow::Result;
use candle_core::hip_backend::WrapErr;
use candle_core::hip_backend::hipdarc::{
    self,
    driver::LaunchConfig,
    rocblas::{GemmDataType, GemmOp, RocBlas, StridedBatchedGemmConfig, gemm_strided_batched_ex},
};
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
    let dev_hip = match &dev { Device::Hip(d) => d.clone(), _ => unreachable!() };

    // 1. Quantize a (M, K) f32 weight to Q4_0.
    let weight_data: Vec<f32> = (0..M * K)
        .map(|i| ((i * 31 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), &dev)?.reshape((M, K))?;
    let qweight = QTensor::quantize(&weight, GgmlDType::Q4_0)?;
    let q_host = qweight.data()?.to_vec();
    let storage_size = q_host.len();
    println!("Q4_0 weight: {storage_size} bytes ({} blocks × 18 B)",
             storage_size / Q4_0_BLOCK_BYTES);

    let q_dev = unsafe { dev_hip.alloc::<u8>(storage_size)? };
    unsafe {
        let rc = hipdarc::sys::hipMemcpy(
            q_dev.device_ptr() as *mut _,
            q_host.as_ptr() as *const _,
            storage_size,
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
        assert_eq!(rc, hipdarc::sys::hipError_t::hipSuccess);
    }

    // 2. Build Y as f32 input for both paths.
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

    // ------------------------------------------------------------------
    // Path A: custom MMQ Q4_0 kernel (existing baseline).
    // ------------------------------------------------------------------
    use candle_core::quantized::hip::{quantize_q8_1, pad, MATRIX_ROW_PADDING};
    let k_padded = pad(K, MATRIX_ROW_PADDING);
    let y_q8_bytes = N * k_padded * Q8_1_BLOCK_BYTES / QK8_1;
    let mut y_q8 = unsafe { dev_hip.alloc::<u8>(y_q8_bytes)? };
    quantize_q8_1(&y_view_f32, &mut y_q8, K, N, &dev_hip)?;
    drop(y_st);

    let dst_mmq = unsafe { dev_hip.alloc::<f32>(M * N)? };
    let func_mmq = dev_hip.get_or_load_func(
        "mul_mat_q4_0_gfx906_v2f_tile32",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let cfg_mmq = LaunchConfig {
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
    let q_view = q_dev.slice(0..q_dev.len());

    // Warmup MMQ.
    for _ in 0..3 {
        let mut bld = func_mmq.builder();
        bld.arg(&q_view); bld.arg(&y_view_q8); bld.arg(&dst_mmq);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg_mmq) }.w()?;
    }
    let _ = dev_hip.stream().synchronize();

    let n_iter = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut bld = func_mmq.builder();
        bld.arg(&q_view); bld.arg(&y_view_q8); bld.arg(&dst_mmq);
        bld.arg(&ncols_x); bld.arg(&nrows_x); bld.arg(&ncols_y);
        bld.arg(&nrows_y_padded); bld.arg(&nrows_dst);
        unsafe { bld.launch(cfg_mmq) }.w()?;
    }
    let _ = dev_hip.stream().synchronize();
    let t_mmq = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("Path A — custom MMQ Q4_0:                     {t_mmq:.1} µs/call");

    // ------------------------------------------------------------------
    // Path B: dequant Q4_0 → f16, then rocBLAS gemm_ex (f16 × f16 → f32).
    // ------------------------------------------------------------------
    // Dequant kernel signature: dequantize_block_q4_0_f16(vx, y, k)
    // Uses 32-thread blocks; each block dequantizes 8 q4_0 blocks (256
    // values). Grid = (nb32 + 7) / 8 where nb32 is the q4_0 block count.
    let total_blocks_x = M * K / QK4_0;
    let nb32 = total_blocks_x as i32;
    let w_f16_bytes = M * K * 2;
    let w_f16_dev = unsafe { dev_hip.alloc::<u8>(w_f16_bytes)? };

    let func_deq = dev_hip.get_or_load_func(
        "dequantize_block_q4_0_f16",
        &candle_hip_kernels::QUANTIZED,
    )?;
    let cfg_deq = LaunchConfig {
        grid_dim: (((nb32 as u32) + 7) / 8, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_view_b = q_dev.slice(0..q_dev.len());
    let mut bld = func_deq.builder();
    bld.arg(&q_view_b); bld.arg(&w_f16_dev); bld.arg(&nb32);
    let t_deq_start = std::time::Instant::now();
    unsafe { bld.launch(cfg_deq) }.w()?;
    let _ = dev_hip.stream().synchronize();
    let t_deq = t_deq_start.elapsed().as_micros() as f64;
    println!("    (one-time Q4_0 → f16 dequant: {t_deq:.1} µs, amortized over many calls)");

    // Convert input y (f32) to f16. For the bench, do it on host.
    let y_f16_bytes = K * N * 2;
    let y_f16_dev = unsafe { dev_hip.alloc::<u8>(y_f16_bytes)? };
    let y_f16_host: Vec<u16> = y_data.iter().map(|f| half::f16::from_f32(*f).to_bits()).collect();
    unsafe {
        hipdarc::sys::hipMemcpy(
            y_f16_dev.device_ptr() as *mut _,
            y_f16_host.as_ptr() as *const _,
            y_f16_bytes,
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
        );
    }

    // Allocate f32 output for rocBLAS path.
    let dst_blas = unsafe { dev_hip.alloc::<f32>(M * N)? };

    // rocBLAS gemm setup. For row-major C[m,n] = sum_k W[m,k] * x[n,k]:
    // In col-major view, compute C_cm[n,m] = x_cm^T * W_cm where:
    //   - x is (N, K) row-major = x_cm (K, N), trans_a = Trans → (N, K)
    //   - W is (M, K) row-major = W_cm (K, M), trans_b = NoTrans → (K, M)
    //   - rocBLAS m = N, n = M, k = K; lda=K, ldb=K, ldc=N
    let blas = RocBlas::new(dev_hip.stream())?;
    let cfg_blas = StridedBatchedGemmConfig {
        trans_a: GemmOp::Trans,
        trans_b: GemmOp::NoTrans,
        m: N as i32,
        n: M as i32,
        k: K as i32,
        lda: K as i32,
        stride_a: 0,
        ldb: K as i32,
        stride_b: 0,
        ldc: N as i32,
        stride_c: 0,
        batch_count: 1,
        ab_type: GemmDataType::F16,
        c_type: GemmDataType::F32,
        compute_type: GemmDataType::F32,
    };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // Warmup rocBLAS.
    for _ in 0..3 {
        unsafe {
            gemm_strided_batched_ex(
                &blas, &cfg_blas,
                &alpha as *const f32 as *const _,
                y_f16_dev.device_ptr() as *const _,
                w_f16_dev.device_ptr() as *const _,
                &beta as *const f32 as *const _,
                dst_blas.device_ptr() as *mut _,
            ).w()?;
        }
    }
    let _ = dev_hip.stream().synchronize();

    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        unsafe {
            gemm_strided_batched_ex(
                &blas, &cfg_blas,
                &alpha as *const f32 as *const _,
                y_f16_dev.device_ptr() as *const _,
                w_f16_dev.device_ptr() as *const _,
                &beta as *const f32 as *const _,
                dst_blas.device_ptr() as *mut _,
            ).w()?;
        }
    }
    let _ = dev_hip.stream().synchronize();
    let t_blas = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("Path B — rocBLAS gemm_ex f16×f16 → f32:        {t_blas:.1} µs/call");
    println!("                                       speedup: {:.2}× ({})",
             t_mmq / t_blas,
             if t_blas < t_mmq { "rocBLAS WINS" } else { "MMQ wins" });

    // Bytes transferred: rocBLAS reads M*K*2 + K*N*2 bytes (half) vs MMQ
    // reads M*K*0.5 + N*K_padded*36/32 bytes (Q4_0+Q8_1).
    println!("    rocBLAS HBM read estimate: {:.1} MB",
             (M * K * 2 + K * N * 2) as f64 / 1e6);
    println!("    MMQ     HBM read estimate: {:.1} MB",
             (storage_size + y_q8_bytes) as f64 / 1e6);

    // ------------------------------------------------------------------
    // Correctness sanity — dequantize MMQ output and compare relative L2.
    // (Not bit-exact: f16-mul vs Q4_0+Q8_1 dp4a paths produce different
    //  numerical noise even from the same input.)
    // ------------------------------------------------------------------
    let mut host_mmq = vec![0f32; M * N];
    let mut host_blas_t = vec![0f32; M * N];  // rocBLAS writes in (N, M) col-major = (M, N) row-major if transposed
    unsafe {
        hipdarc::sys::hipMemcpy(
            host_mmq.as_mut_ptr() as *mut _,
            dst_mmq.device_ptr() as *const _,
            M * N * 4,
            hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hipdarc::sys::hipMemcpy(
            host_blas_t.as_mut_ptr() as *mut _,
            dst_blas.device_ptr() as *const _,
            M * N * 4,
            hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToHost,
        );
    }
    // host_mmq is (M, N) row-major: dst[col * nrows_dst + row] (col-major M-fast)
    // host_blas_t is C_cm (N, M) col-major = (M, N) row-major IF we
    // also store in the same (col-fast) order. Let's just compare the
    // first few values directly assuming both are (M, N) col-major
    // (rocBLAS writes col-major; the MMQ kernel also writes
    // dst[col * nrows_dst + row] = col-major).
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for i in 0..M * N {
        let d = (host_mmq[i] - host_blas_t[i]) as f64;
        diff_sq += d * d;
        ref_sq += (host_mmq[i] as f64).powi(2);
        max_abs = max_abs.max(d.abs());
    }
    let rel_l2 = (diff_sq.sqrt() / ref_sq.sqrt().max(1e-30)).max(1e-30);
    println!("Correctness (cross-precision): rel L2 = {rel_l2:.3e}, max abs = {max_abs:.3e}");
    if rel_l2 > 1e-2 {
        println!("⚠ Outputs DIVERGE significantly — check rocBLAS gemm config.");
    } else {
        println!("✓ Outputs in same ballpark (differences are quant precision noise).");
    }

    Ok(())
}
