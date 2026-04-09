//! Safe wrapper around RCCL (ROCm Communication Collectives Library).
//!
//! RCCL is API-compatible with NCCL — same function names, same types,
//! just links against `librccl.so` instead of `libnccl.so`.

use crate::driver::HipStream;
use crate::sys::hipStream_t;
use libc::{c_int, c_void, size_t};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FFI bindings (RCCL uses NCCL symbol names)
// ---------------------------------------------------------------------------

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    pub internal: [u8; NCCL_UNIQUE_ID_BYTES],
}

pub type NcclComm = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
    NumResults = 8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DataType {
    Int8 = 0,
    Uint8 = 2,
    Int32 = 3,
    Uint32 = 4,
    Int64 = 5,
    Uint64 = 6,
    Float16 = 7,
    Float32 = 8,
    Float64 = 9,
    Bfloat16 = 10,
}

extern "C" {
    fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult;
    fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: c_int,
        id: NcclUniqueId,
        rank: c_int,
    ) -> NcclResult;
    fn ncclCommDestroy(comm: NcclComm) -> NcclResult;
    fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: size_t,
        datatype: DataType,
        op: ReduceOp,
        comm: NcclComm,
        stream: hipStream_t,
    ) -> NcclResult;
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[derive(thiserror::Error, Debug)]
pub enum RcclError {
    #[error("RCCL error: {0:?}")]
    Status(NcclResult),
}

fn check(result: NcclResult) -> Result<(), RcclError> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        Err(RcclError::Status(result))
    }
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// A unique ID used to initialize RCCL communicators across processes.
/// One process generates the ID, then all processes use it.
impl NcclUniqueId {
    pub fn new() -> Result<Self, RcclError> {
        let mut id = NcclUniqueId {
            internal: [0u8; NCCL_UNIQUE_ID_BYTES],
        };
        unsafe { check(ncclGetUniqueId(&mut id))? };
        Ok(id)
    }

    /// Create from raw bytes (for passing between processes via env var or socket).
    pub fn from_bytes(bytes: &[u8; NCCL_UNIQUE_ID_BYTES]) -> Self {
        NcclUniqueId { internal: *bytes }
    }

    /// Get raw bytes for serialization.
    pub fn as_bytes(&self) -> &[u8; NCCL_UNIQUE_ID_BYTES] {
        &self.internal
    }
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        NcclUniqueId {
            internal: [0u8; NCCL_UNIQUE_ID_BYTES],
        }
    }
}

impl std::fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclUniqueId").finish()
    }
}

/// An RCCL communicator bound to a specific GPU rank and stream.
pub struct Comm {
    raw: NcclComm,
    _stream: Arc<HipStream>,
}

unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    /// Initialize a communicator for the given rank.
    ///
    /// All `nranks` processes must call this with the same `id` and
    /// unique `rank` values. This call blocks until all ranks have joined.
    pub fn new(nranks: usize, id: NcclUniqueId, rank: usize, stream: &Arc<HipStream>) -> Result<Self, RcclError> {
        stream.device().set_current().map_err(|_| RcclError::Status(NcclResult::InternalError))?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(ncclCommInitRank(
                &mut raw,
                nranks as c_int,
                id,
                rank as c_int,
            ))?;
        }
        Ok(Self {
            raw,
            _stream: stream.clone(),
        })
    }

    /// In-place all-reduce: sendbuff and recvbuff can be the same pointer.
    ///
    /// # Safety
    /// `send` and `recv` must be valid device pointers with at least `count`
    /// elements of the given datatype.
    pub unsafe fn all_reduce(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
        stream: &HipStream,
    ) -> Result<(), RcclError> {
        check(ncclAllReduce(
            send,
            recv,
            count as size_t,
            dtype,
            op,
            self.raw,
            stream.raw(),
        ))
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = ncclCommDestroy(self.raw);
            }
        }
    }
}

impl std::fmt::Debug for Comm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Comm").finish()
    }
}
