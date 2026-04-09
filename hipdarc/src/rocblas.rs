//! Safe wrapper around rocBLAS for matrix operations.

use crate::driver::HipStream;
use crate::error::{check_rocblas, RocblasError};
use crate::sys;

/// A rocBLAS handle bound to a HIP stream.
pub struct RocBlas {
    handle: sys::rocblas_handle,
}

unsafe impl Send for RocBlas {}
unsafe impl Sync for RocBlas {}

impl RocBlas {
    /// Create a new rocBLAS handle on the given stream.
    ///
    /// # Errors
    /// Returns `RocblasError` if handle creation or stream binding fails.
    pub fn new(stream: &HipStream) -> Result<Self, RocblasError> {
        stream
            .device()
            .set_current()
            .map_err(|_| RocblasError::Status(sys::rocblas_status::rocblas_status_internal_error))?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check_rocblas(sys::rocblas_create_handle(&mut handle))?;
            check_rocblas(sys::rocblas_set_stream(handle, stream.raw()))?;
        }
        Ok(Self { handle })
    }

    /// Raw rocBLAS handle for advanced use.
    pub fn raw(&self) -> sys::rocblas_handle {
        self.handle
    }
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::rocblas_destroy_handle(self.handle);
            }
        }
    }
}

impl std::fmt::Debug for RocBlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocBlas").finish()
    }
}

// ---------------------------------------------------------------------------
// GEMM configuration types
// ---------------------------------------------------------------------------

/// Transpose operation for GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmOp {
    NoTrans,
    Trans,
    ConjTrans,
}

impl GemmOp {
    fn to_raw(self) -> sys::rocblas_operation {
        match self {
            Self::NoTrans => sys::rocblas_operation::rocblas_operation_none,
            Self::Trans => sys::rocblas_operation::rocblas_operation_transpose,
            Self::ConjTrans => sys::rocblas_operation::rocblas_operation_conjugate_transpose,
        }
    }
}

/// Data type for GEMM operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmDataType {
    F16,
    BF16,
    F32,
    F64,
}

impl GemmDataType {
    fn to_raw(self) -> sys::rocblas_datatype {
        match self {
            Self::F16 => sys::rocblas_datatype::rocblas_datatype_f16_r,
            Self::BF16 => sys::rocblas_datatype::rocblas_datatype_bf16_r,
            Self::F32 => sys::rocblas_datatype::rocblas_datatype_f32_r,
            Self::F64 => sys::rocblas_datatype::rocblas_datatype_f64_r,
        }
    }
}

/// Full configuration for a strided-batched GEMM call.
#[derive(Debug, Clone)]
pub struct StridedBatchedGemmConfig {
    pub trans_a: GemmOp,
    pub trans_b: GemmOp,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub stride_a: i64,
    pub ldb: i32,
    pub stride_b: i64,
    pub ldc: i32,
    pub stride_c: i64,
    pub batch_count: i32,
    pub ab_type: GemmDataType,
    pub c_type: GemmDataType,
    pub compute_type: GemmDataType,
}

/// Execute a strided-batched GEMM: C = alpha * op(A) * op(B) + beta * C.
///
/// # Safety
/// Device pointers `a`, `b`, `c` must be valid and correctly sized for the
/// given configuration. `alpha` and `beta` must point to values of
/// `compute_type`.
pub unsafe fn gemm_strided_batched_ex(
    blas: &RocBlas,
    cfg: &StridedBatchedGemmConfig,
    alpha: *const libc::c_void,
    a: *const libc::c_void,
    b: *const libc::c_void,
    beta: *const libc::c_void,
    c: *mut libc::c_void,
) -> Result<(), RocblasError> {
    check_rocblas(sys::rocblas_gemm_strided_batched_ex(
        blas.handle,
        cfg.trans_a.to_raw(),
        cfg.trans_b.to_raw(),
        cfg.m,
        cfg.n,
        cfg.k,
        alpha,
        a,
        cfg.ab_type.to_raw(),
        cfg.lda,
        cfg.stride_a,
        b,
        cfg.ab_type.to_raw(),
        cfg.ldb,
        cfg.stride_b,
        beta,
        c,
        cfg.c_type.to_raw(),
        cfg.ldc,
        cfg.stride_c,
        c,                    // D = C (in-place)
        cfg.c_type.to_raw(),  // D type = C type
        cfg.ldc,
        cfg.stride_c,
        cfg.batch_count,
        cfg.compute_type.to_raw(),
        sys::rocblas_gemm_algo::rocblas_gemm_algo_standard,
        0, // solution_index
        0, // flags
    ))
}
