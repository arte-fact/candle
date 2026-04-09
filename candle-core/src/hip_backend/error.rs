use crate::{DType, Layout};

/// HIP/ROCm related errors.
#[derive(thiserror::Error, Debug)]
pub enum HipError {
    #[error(transparent)]
    Driver(#[from] hipdarc::error::DriverError),

    #[error(transparent)]
    Rocblas(#[from] hipdarc::error::RocblasError),

    #[error(transparent)]
    Hiprand(#[from] hipdarc::error::HiprandError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{driver} when loading {module_name}")]
    Load {
        driver: hipdarc::error::DriverError,
        module_name: String,
    },
}

impl From<HipError> for crate::Error {
    fn from(val: HipError) -> Self {
        crate::Error::Hip(Box::new(val)).bt()
    }
}

/// Extension trait to convert hipdarc results into candle results.
pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<HipError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Hip(Box::new(e.into())).bt())
    }
}
