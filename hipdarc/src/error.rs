use crate::sys;

#[derive(thiserror::Error, Debug)]
pub enum DriverError {
    #[error("HIP error: {0:?}")]
    Hip(sys::hipError_t),
    #[error("{0}")]
    Message(String),
}

#[derive(thiserror::Error, Debug)]
pub enum RocblasError {
    #[error("rocBLAS error: {0:?}")]
    Status(sys::rocblas_status),
}

#[derive(thiserror::Error, Debug)]
pub enum HiprandError {
    #[error("hiprand error: {0:?}")]
    Status(sys::hiprandStatus_t),
}

/// Check a HIP runtime call, returning Err on failure.
pub(crate) fn check_hip(status: sys::hipError_t) -> Result<(), DriverError> {
    if status == sys::hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(DriverError::Hip(status))
    }
}

/// Check a rocBLAS call, returning Err on failure.
pub(crate) fn check_rocblas(status: sys::rocblas_status) -> Result<(), RocblasError> {
    if status == sys::rocblas_status::rocblas_status_success {
        Ok(())
    } else {
        Err(RocblasError::Status(status))
    }
}

/// Check a hiprand call, returning Err on failure.
pub(crate) fn check_hiprand(status: sys::hiprandStatus_t) -> Result<(), HiprandError> {
    if status == sys::hiprandStatus_t::HIPRAND_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HiprandError::Status(status))
    }
}
