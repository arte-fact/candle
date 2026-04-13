//! Safe wrappers around the HIP driver API.
//!
//! Provides `HipDevice`, `HipStream`, `HipSlice<T>`, `HipModule`, `HipFunction`,
//! `LaunchConfig`, and kernel launch helpers that mirror the cudarc API surface
//! used by Candle.

use crate::error::{check_hip, DriverError};
use crate::sys;
use libc::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Marker trait for types that can be represented on the device.
///
/// # Safety
/// The type must be `Copy` and have a well-defined bit representation that is
/// compatible with HIP device memory.
pub unsafe trait DeviceRepr: Copy {}

unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
unsafe impl DeviceRepr for half::f16 {}
unsafe impl DeviceRepr for half::bf16 {}
unsafe impl DeviceRepr for float8::F8E4M3 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for bool {}

/// Marker trait for types where all-zero bytes is a valid value.
///
/// # Safety
/// All-zero bytes must produce a valid value of the type.
pub unsafe trait ValidAsZeroBits: DeviceRepr {}

unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
unsafe impl ValidAsZeroBits for half::f16 {}
unsafe impl ValidAsZeroBits for half::bf16 {}
unsafe impl ValidAsZeroBits for float8::F8E4M3 {}
unsafe impl ValidAsZeroBits for usize {}

// ---------------------------------------------------------------------------
// HipDevice
// ---------------------------------------------------------------------------

/// Represents a HIP-capable GPU device.
#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: i32,
}

impl HipDevice {
    /// Create a handle to the GPU with the given ordinal.
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let ordinal = ordinal as i32;
        unsafe {
            check_hip(sys::hipInit(0))?;
            let mut count = 0i32;
            check_hip(sys::hipGetDeviceCount(&mut count))?;
            if ordinal >= count {
                return Err(DriverError::Message(format!(
                    "device ordinal {ordinal} out of range (found {count} devices)"
                )));
            }
            check_hip(sys::hipSetDevice(ordinal))?;
        }
        Ok(Self { ordinal })
    }

    /// Return the device ordinal.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Make this device the current device for the calling thread.
    pub fn set_current(&self) -> Result<(), DriverError> {
        unsafe { check_hip(sys::hipSetDevice(self.ordinal)) }
    }

    /// Query device properties.
    pub fn properties(&self) -> Result<sys::hipDeviceProp_t, DriverError> {
        unsafe {
            let mut prop = std::mem::zeroed::<sys::hipDeviceProp_t>();
            check_hip(sys::hipGetDeviceProperties(&mut prop, self.ordinal))?;
            Ok(prop)
        }
    }

    /// Synchronize the device (wait for all work to complete).
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.set_current()?;
        unsafe { check_hip(sys::hipDeviceSynchronize()) }
    }
}

// ---------------------------------------------------------------------------
// HipStream
// ---------------------------------------------------------------------------

/// A HIP stream for ordering GPU operations.
pub struct HipStream {
    raw: sys::hipStream_t,
    device: HipDevice,
}

// SAFETY: HIP streams can be shared across threads; the HIP runtime handles
// thread safety for stream operations internally.
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Create a new stream on the given device.
    pub fn new(device: &HipDevice) -> Result<Self, DriverError> {
        device.set_current()?;
        let mut raw = std::ptr::null_mut();
        unsafe { check_hip(sys::hipStreamCreate(&mut raw))? };
        Ok(Self {
            raw,
            device: device.clone(),
        })
    }

    /// Get the raw stream handle (for FFI).
    pub fn raw(&self) -> sys::hipStream_t {
        self.raw
    }

    /// Get a reference to the device this stream belongs to.
    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    /// Wait for all operations on this stream to complete.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        unsafe { check_hip(sys::hipStreamSynchronize(self.raw)) }
    }

    // -- Allocation helpers --------------------------------------------------

    /// Allocate `len` elements of type `T` on the device (uninitialized).
    ///
    /// Uses stream-ordered `hipMallocAsync` (ROCm 5.1+) which is backed by
    /// an internal memory pool. Several **hundred microseconds faster per
    /// alloc** than the synchronous `hipMalloc` for the small temporary
    /// buffers candle creates by the hundred per decoded token. Falls back
    /// to `hipMalloc` if the async API isn't supported.
    pub fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<HipSlice<T>, DriverError> {
        self.device.set_current()?;
        let ordinal = self.device.ordinal;
        if len == 0 {
            return Ok(HipSlice {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                device_ordinal: -1,
                _marker: PhantomData,
            });
        }
        let size = len * std::mem::size_of::<T>();

        // Decode-alloc replay: serve from pre-recorded (padded) table.
        if let Some(ptr) = decode_alloc_try_take(size) {
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: DECODE_ALLOC_SENTINEL,
                _marker: PhantomData,
            });
        }
        let recording = decode_alloc_is_recording();

        // During decode-alloc recording, allocate at padded size to
        // accommodate kv_len growth on the next token.
        let alloc_size = if recording { pad_decode_size(size) } else { size };

        // Pool fast path.
        if let Some(ptr) = pool_try_take(ordinal, alloc_size) {
            if recording {
                decode_alloc_record(alloc_size, ptr);
                return Ok(HipSlice {
                    ptr,
                    len,
                    free_stream: std::ptr::null_mut(),
                    device_ordinal: DECODE_ALLOC_SENTINEL,
                    _marker: PhantomData,
                });
            }
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: ordinal,
                _marker: PhantomData,
            });
        }
        let mut ptr = std::ptr::null_mut();
        let mut free_stream: sys::hipStream_t = std::ptr::null_mut();
        unsafe {
            let rc = sys::hipMallocAsync(&mut ptr, alloc_size, self.raw);
            if rc != sys::hipError_t::hipSuccess {
                check_hip(sys::hipMalloc(&mut ptr, alloc_size))?;
            } else {
                free_stream = self.raw;
            }
        };
        if recording {
            decode_alloc_record(alloc_size, ptr);
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: DECODE_ALLOC_SENTINEL,
                _marker: PhantomData,
            });
        }
        Ok(HipSlice {
            ptr,
            len,
            free_stream,
            device_ordinal: ordinal,
            _marker: PhantomData,
        })
    }

    /// Allocate `len` elements of type `T` on the device, zeroed.
    pub fn alloc_zeros<T: ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<HipSlice<T>, DriverError> {
        self.device.set_current()?;
        let ordinal = self.device.ordinal;
        if len == 0 {
            return Ok(HipSlice {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                device_ordinal: -1,
                _marker: PhantomData,
            });
        }
        let size = len * std::mem::size_of::<T>();

        // Decode-alloc replay: serve from padded table, zero for correctness.
        if let Some(ptr) = decode_alloc_try_take(size) {
            unsafe { check_hip(sys::hipMemsetAsync(ptr, 0, size, self.raw))?; }
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: DECODE_ALLOC_SENTINEL,
                _marker: PhantomData,
            });
        }
        let recording = decode_alloc_is_recording();
        let alloc_size = if recording { pad_decode_size(size) } else { size };

        // Pool fast path.
        if let Some(ptr) = pool_try_take(ordinal, alloc_size) {
            unsafe {
                check_hip(sys::hipMemsetAsync(ptr, 0, size, self.raw))?;
            }
            if recording {
                decode_alloc_record(alloc_size, ptr);
                return Ok(HipSlice {
                    ptr,
                    len,
                    free_stream: std::ptr::null_mut(),
                    device_ordinal: DECODE_ALLOC_SENTINEL,
                    _marker: PhantomData,
                });
            }
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: ordinal,
                _marker: PhantomData,
            });
        }
        let mut ptr = std::ptr::null_mut();
        let mut free_stream: sys::hipStream_t = std::ptr::null_mut();
        unsafe {
            let rc = sys::hipMallocAsync(&mut ptr, alloc_size, self.raw);
            if rc != sys::hipError_t::hipSuccess {
                check_hip(sys::hipMalloc(&mut ptr, alloc_size))?;
            } else {
                free_stream = self.raw;
            }
            check_hip(sys::hipMemsetAsync(ptr, 0, size, self.raw))?;
        }
        if recording {
            decode_alloc_record(alloc_size, ptr);
            return Ok(HipSlice {
                ptr,
                len,
                free_stream: std::ptr::null_mut(),
                device_ordinal: DECODE_ALLOC_SENTINEL,
                _marker: PhantomData,
            });
        }
        Ok(HipSlice {
            ptr,
            len,
            free_stream,
            device_ordinal: ordinal,
            _marker: PhantomData,
        })
    }

    // -- Copy helpers --------------------------------------------------------

    /// Copy from host slice to a new device allocation.
    pub fn clone_htod<T: DeviceRepr>(&self, src: &[T]) -> Result<HipSlice<T>, DriverError> {
        let dst = self.alloc::<T>(src.len())?;
        if src.is_empty() {
            return Ok(dst);
        }
        let size = std::mem::size_of_val(src);
        unsafe {
            check_hip(sys::hipMemcpyAsync(
                dst.ptr,
                src.as_ptr() as *const c_void,
                size,
                sys::hipMemcpyKind::hipMemcpyHostToDevice,
                self.raw,
            ))?;
        }
        Ok(dst)
    }

    /// Copy device memory to a new host Vec.
    pub fn clone_dtoh<T: DeviceRepr>(&self, src: &HipSlice<T>) -> Result<Vec<T>, DriverError> {
        if src.len == 0 {
            return Ok(Vec::new());
        }
        let mut dst = Vec::with_capacity(src.len);
        let size = src.len * std::mem::size_of::<T>();
        unsafe {
            check_hip(sys::hipMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr as *const c_void,
                size,
                sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                self.raw,
            ))?;
            // Must synchronize before host reads the data.
            check_hip(sys::hipStreamSynchronize(self.raw))?;
            dst.set_len(src.len);
        }
        Ok(dst)
    }

    /// Copy from one device allocation to another (must be same length).
    pub fn memcpy_dtod<T: DeviceRepr>(
        &self,
        dst: &mut HipSlice<T>,
        src: &HipSlice<T>,
    ) -> Result<(), DriverError> {
        assert_eq!(dst.len, src.len, "dtod copy length mismatch");
        if src.len == 0 {
            return Ok(());
        }
        let size = src.len * std::mem::size_of::<T>();
        unsafe {
            check_hip(sys::hipMemcpyAsync(
                dst.ptr,
                src.ptr as *const c_void,
                size,
                sys::hipMemcpyKind::hipMemcpyDeviceToDevice,
                self.raw,
            ))?;
        }
        Ok(())
    }

    /// Clone a device allocation (allocate + copy).
    pub fn clone_dtod<T: DeviceRepr>(
        &self,
        src: &HipSlice<T>,
    ) -> Result<HipSlice<T>, DriverError> {
        let mut dst = self.alloc::<T>(src.len)?;
        self.memcpy_dtod(&mut dst, src)?;
        Ok(dst)
    }

    /// Copy from host slice into an existing device allocation.
    pub fn memcpy_htod<T: DeviceRepr>(
        &self,
        dst: &mut HipSlice<T>,
        src: &[T],
    ) -> Result<(), DriverError> {
        assert!(src.len() <= dst.len, "htod copy: src larger than dst");
        if src.is_empty() {
            return Ok(());
        }
        let size = std::mem::size_of_val(src);
        unsafe {
            check_hip(sys::hipMemcpyAsync(
                dst.ptr,
                src.as_ptr() as *const c_void,
                size,
                sys::hipMemcpyKind::hipMemcpyHostToDevice,
                self.raw,
            ))?;
        }
        Ok(())
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::hipStreamDestroy(self.raw);
            }
        }
    }
}

impl std::fmt::Debug for HipStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipStream")
            .field("device", &self.device.ordinal)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// HIP graphs — capture-replay for per-token decode
// ---------------------------------------------------------------------------
//
// Used to amortise per-kernel driver-call overhead. The model captures
// its per-token decode forward into a `HipGraphExec` once, then replays
// it for every subsequent token via `HipGraphExec::launch(stream)`.
//
// Restrictions during capture:
// - Every operation on the stream must be stream-ordered (no sync calls).
//   This is why hipdarc switched to `hipMallocAsync`. Sync `hipFree` on
//   `Drop` would also kill capture, so allocations made during capture
//   must outlive the capture and free outside it.
// - The kernel arguments and memory layout must be deterministic across
//   replays. For decode this means a pre-allocated KV cache (no growing
//   `Tensor::cat`) and a fixed input slot for the next-token id.

/// RAII handle for an in-progress stream capture. Calling
/// [`Self::end`] (or letting it drop) ends the capture and returns the
/// captured graph. Failing to do either leaves the stream stuck in
/// capture mode and is a programmer error.
pub struct HipGraphCapture<'s> {
    stream: &'s HipStream,
    finished: bool,
}

impl<'s> HipGraphCapture<'s> {
    /// Begin capturing every operation submitted to `stream`.
    pub fn begin(stream: &'s HipStream) -> Result<Self, DriverError> {
        unsafe {
            check_hip(sys::hipStreamBeginCapture(
                stream.raw,
                sys::HIP_STREAM_CAPTURE_MODE_GLOBAL,
            ))?;
        }
        Ok(Self {
            stream,
            finished: false,
        })
    }

    /// End the capture and return the resulting graph. Must be called
    /// exactly once per `HipGraphCapture`.
    pub fn end(mut self) -> Result<HipGraph, DriverError> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check_hip(sys::hipStreamEndCapture(self.stream.raw, &mut raw))?;
        }
        self.finished = true;
        Ok(HipGraph { raw })
    }
}

impl Drop for HipGraphCapture<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // The capture wasn't ended explicitly. Try to end it now to
            // avoid leaving the stream stuck in capture mode; we discard
            // the resulting graph since the caller obviously doesn't
            // want it.
            let mut raw = std::ptr::null_mut();
            unsafe {
                let rc = sys::hipStreamEndCapture(self.stream.raw, &mut raw);
                if rc == sys::hipError_t::hipSuccess && !raw.is_null() {
                    let _ = sys::hipGraphDestroy(raw);
                }
            }
        }
    }
}

/// Owned captured graph. Not yet executable — call
/// [`Self::instantiate`] to compile it into a `HipGraphExec`.
pub struct HipGraph {
    raw: sys::hipGraph_t,
}

unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl HipGraph {
    pub fn raw(&self) -> sys::hipGraph_t {
        self.raw
    }

    /// Enumerate every node in the captured graph in topological order.
    /// The handles returned identify the same nodes inside any
    /// `HipGraphExec` produced by `instantiate`, so they can be passed
    /// to `HipGraphExec::set_kernel_node_params`.
    pub fn nodes(&self) -> Result<Vec<sys::hipGraphNode_t>, DriverError> {
        // First call: query the count with null pointer.
        let mut n: usize = 0;
        unsafe {
            check_hip(sys::hipGraphGetNodes(
                self.raw,
                std::ptr::null_mut(),
                &mut n as *mut usize,
            ))?;
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut nodes: Vec<sys::hipGraphNode_t> = vec![std::ptr::null_mut(); n];
        let mut n2: usize = n;
        unsafe {
            check_hip(sys::hipGraphGetNodes(
                self.raw,
                nodes.as_mut_ptr(),
                &mut n2 as *mut usize,
            ))?;
        }
        nodes.truncate(n2);
        Ok(nodes)
    }

    /// Compile this graph into an executable that can be replayed
    /// repeatedly via [`HipGraphExec::launch`].
    pub fn instantiate(&self) -> Result<HipGraphExec, DriverError> {
        let mut exec = std::ptr::null_mut();
        unsafe {
            check_hip(sys::hipGraphInstantiate(
                &mut exec,
                self.raw,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ))?;
        }
        Ok(HipGraphExec { raw: exec })
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::hipGraphDestroy(self.raw);
            }
        }
    }
}

/// Compiled executable graph. Replay with [`Self::launch`].
pub struct HipGraphExec {
    raw: sys::hipGraphExec_t,
}

unsafe impl Send for HipGraphExec {}
unsafe impl Sync for HipGraphExec {}

impl HipGraphExec {
    pub fn raw(&self) -> sys::hipGraphExec_t {
        self.raw
    }

    /// Launch the captured graph on `stream`. This submits the entire
    /// kernel sequence with one driver call instead of N
    /// `hipModuleLaunchKernel` calls.
    pub fn launch(&self, stream: &HipStream) -> Result<(), DriverError> {
        unsafe { check_hip(sys::hipGraphLaunch(self.raw, stream.raw)) }
    }

    /// Patch the launch parameters of a single kernel node on this
    /// instantiated graph. The node handle comes from [`HipGraph::nodes`].
    ///
    /// `kernel_params` is an array of pointers to the argument values,
    /// same convention as `hipModuleLaunchKernel`'s `kernelParams` — each
    /// entry must point to an argument value that stays alive until the
    /// next `hipGraphLaunch` completes. `grid`/`block`/`shared_mem`/`func`
    /// must match the values the node was captured with.
    ///
    /// # Safety
    /// `kernel_params` must point to valid argument storage for the
    /// kernel's expected signature; `func` must match the captured kernel.
    pub unsafe fn set_kernel_node_params(
        &self,
        node: sys::hipGraphNode_t,
        func: sys::hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        kernel_params: *mut *mut std::ffi::c_void,
    ) -> Result<(), DriverError> {
        let params = sys::hipKernelNodeParams {
            blockDim: sys::dim3 { x: block.0, y: block.1, z: block.2 },
            extra: std::ptr::null_mut(),
            func: func as *mut std::ffi::c_void,
            gridDim: sys::dim3 { x: grid.0, y: grid.1, z: grid.2 },
            kernelParams: kernel_params,
            sharedMemBytes: shared_mem,
        };
        check_hip(sys::hipGraphExecKernelNodeSetParams(
            self.raw,
            node,
            &params as *const sys::hipKernelNodeParams,
        ))
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::hipGraphExecDestroy(self.raw);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HipSlice — owned device memory
// ---------------------------------------------------------------------------

/// Owned device memory allocation of `len` elements of type `T`.
pub struct HipSlice<T> {
    pub(crate) ptr: sys::hipDeviceptr_t,
    pub(crate) len: usize,
    /// Stream this allocation was taken from when created via
    /// `hipMallocAsync`. Used by the capture-mode `Drop` path to
    /// record a `hipFreeAsync` graph node on the right stream.
    #[allow(dead_code)]
    pub(crate) free_stream: sys::hipStream_t,
    /// Device ordinal this allocation lives on. Stored so `Drop` can
    /// route the buffer back to the right per-device workspace pool
    /// without needing to call `hipGetDevice` (which is per-thread
    /// state and would force a `set_device` round-trip in `Drop`).
    /// `-1` for the empty / null sentinel.
    pub(crate) device_ordinal: i32,
    pub(crate) _marker: PhantomData<T>,
}

// SAFETY: Device pointers can be sent across threads; the HIP runtime is
// thread-safe for memory operations.
unsafe impl<T> Send for HipSlice<T> {}
unsafe impl<T> Sync for HipSlice<T> {}

impl<T> HipSlice<T> {
    /// Create a non-owning view into existing device memory for decode
    /// replay. Uses `DECODE_ALLOC_SENTINEL` so `Drop` is a no-op.
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer to at least `len` elements of `T`.
    /// The memory must outlive this `HipSlice` (guaranteed when it belongs
    /// to the decode allocator table).
    pub unsafe fn decode_view(ptr: sys::hipDeviceptr_t, len: usize) -> Self {
        Self {
            ptr,
            len,
            free_stream: std::ptr::null_mut(),
            device_ordinal: DECODE_ALLOC_SENTINEL,
            _marker: PhantomData,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer.
    pub fn device_ptr(&self) -> sys::hipDeviceptr_t {
        self.ptr
    }

    /// Create a borrowed view into a sub-range of this allocation.
    pub fn slice(&self, range: std::ops::Range<usize>) -> HipView<'_, T> {
        assert!(range.end <= self.len, "slice out of bounds");
        let offset = range.start * std::mem::size_of::<T>();
        HipView {
            ptr: unsafe { (self.ptr as *const u8).add(offset) as sys::hipDeviceptr_t },
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }

    /// Clone this allocation (requires a stream for the copy).
    pub fn try_clone(&self, stream: &HipStream) -> Result<Self, DriverError>
    where
        T: DeviceRepr,
    {
        stream.clone_dtod(self)
    }
}

/// Sentinel ordinal for decode-allocated HipSlice. `Drop` becomes a
/// no-op so the buffer stays alive for replay across decode tokens.
pub const DECODE_ALLOC_SENTINEL: i32 = -2;

impl<T> Drop for HipSlice<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.device_ordinal == DECODE_ALLOC_SENTINEL {
            return;
        }
        // Inside a HIP graph capture, sync `hipFree` would abort the
        // capture (HIP forbids synchronous device calls during
        // capture) and pooling the buffer would let it be reused
        // mid-graph. Route through `hipFreeAsync` on the capture
        // stream so the free becomes a graph-managed node.
        unsafe {
            let captured_to = capture_active_stream();
            if !captured_to.is_null() {
                let _ = sys::hipFreeAsync(self.ptr, captured_to);
                return;
            }
        }
        // Outside capture: return the buffer to the per-device
        // workspace pool. The next op of the same byte size on this
        // device will pop it instead of calling hipMallocAsync. The
        // pool falls back to `hipFree` if it's disabled or the
        // bucket is at capacity.
        let size = self.len * std::mem::size_of::<T>();
        pool_give_back(self.device_ordinal, self.ptr, size);
    }
}

// ---------------------------------------------------------------------------
// Per-device workspace pool
// ---------------------------------------------------------------------------
//
// Candle's HIP backend allocates a fresh device buffer for every Tensor
// op output and frees it on drop. For an `n_layers × n_ops_per_layer`
// transformer forward this generates thousands of hipMallocAsync /
// hipFree pairs *per forward pass* — measured at 1964 alloc + 1964 free
// + 491 memset calls per forward of qwen35-9B.
//
// Each call is sub-millisecond on its own but they accumulate to
// ~70-100 ms of pure HIP API host overhead per forward, which is roughly
// equal to the GPU kernel work itself. llamacpp-turbo bypasses this
// entirely by allocating one large workspace at startup and slicing
// into it for every op — we measured ~0.4 alloc calls per forward in
// turbo vs candle's ~2000.
//
// The pool below closes most of this gap: alloc requests of the same
// byte size hit the pool with a HashMap lookup (~10s of ns) instead
// of going to the runtime allocator. Buffers returned to the pool on
// drop become available for the next op of the same size on the same
// device. Inference workloads reuse the exact same shapes on every
// forward, so once the pool is warm (~1 forward) the per-op allocator
// path costs essentially zero.
//
// Knobs:
// - `HIPDARC_DISABLE_POOL=1` skips the pool entirely (for A/B testing).
// - `HIPDARC_POOL_MAX_BYTES=N` caps total pool size per device. Default
//   is unbounded — for inference workloads the working set is bounded
//   by the model and tends to plateau after one forward.
//
// Capture-mode interaction: while a HIP graph capture is in progress,
// drops bypass the pool and go to `hipFreeAsync(capture_stream)` so the
// free becomes a graph-managed node (otherwise the buffer would be
// reused mid-graph and corrupt captured kernels). The pool is reused
// freely between captures.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Per-device pool of free buffers, keyed by byte size.
/// Stores raw device pointers; the size is implicit in the bucket key.
///
/// SAFETY: `sys::hipDeviceptr_t` is `*mut c_void` and not `Send` by
/// default, but device pointers are valid across host threads — the
/// HIP runtime treats them as opaque handles. The pool is guarded by
/// a `Mutex` so concurrent access is serialized.
unsafe impl Send for DevicePool {}
unsafe impl Sync for DevicePool {}

struct DevicePool {
    /// `(ordinal, hip raw stream)` — the pool needs to know which device
    /// owns these allocations so it can `hipSetDevice` before the
    /// fallback hipMalloc / hipFree paths.
    device_ordinal: i32,
    /// Buckets keyed by *byte* size. Each bucket is a stack of free
    /// pointers — LIFO so the most-recently-used buffer (likely still
    /// hot in caches) is reused first.
    buckets: HashMap<usize, Vec<sys::hipDeviceptr_t>>,
    /// Total bytes currently held by this pool (sum of bucket entries).
    /// Used for the optional cap. Pool *capacity* is unbounded by
    /// default; entries beyond the cap fall back to hipFree on drop.
    total_bytes: usize,
    max_bytes: usize,
}

impl DevicePool {
    fn new(device_ordinal: i32, max_bytes: usize) -> Self {
        Self {
            device_ordinal,
            buckets: HashMap::new(),
            total_bytes: 0,
            max_bytes,
        }
    }

    /// Take a buffer of exactly `size` bytes from the pool, or `None`
    /// if none available. Constant-time HashMap lookup + Vec::pop.
    fn try_take(&mut self, size: usize) -> Option<sys::hipDeviceptr_t> {
        let bucket = self.buckets.get_mut(&size)?;
        let ptr = bucket.pop()?;
        self.total_bytes -= size;
        Some(ptr)
    }

    /// Return a buffer of `size` bytes to the pool. If the pool is at
    /// capacity, frees the buffer instead.
    fn give_back(&mut self, ptr: sys::hipDeviceptr_t, size: usize) {
        // Cap check: if adding this buffer would exceed the cap, free
        // it via hipFree instead of pooling. This keeps the pool
        // bounded for workloads with unbounded shape variation (rare
        // in inference but possible).
        if self.max_bytes != 0 && self.total_bytes + size > self.max_bytes {
            unsafe {
                let _ = sys::hipSetDevice(self.device_ordinal);
                let _ = sys::hipFree(ptr);
            }
            return;
        }
        self.buckets.entry(size).or_default().push(ptr);
        self.total_bytes += size;
    }
}

impl Drop for DevicePool {
    fn drop(&mut self) {
        // Process exit: free everything we still hold. Best-effort —
        // the runtime is tearing down anyway.
        unsafe {
            let _ = sys::hipSetDevice(self.device_ordinal);
            for (_size, bucket) in self.buckets.drain() {
                for ptr in bucket {
                    let _ = sys::hipFree(ptr);
                }
            }
        }
    }
}

/// Global registry of per-device pools. Lazy-initialized on first
/// access. Each device has its own `Mutex<DevicePool>` so unrelated
/// devices don't contend.
static POOLS: OnceLock<Mutex<HashMap<i32, Arc<Mutex<DevicePool>>>>> = OnceLock::new();

/// Cached "is the pool enabled" flag — checked once per process. The
/// `HIPDARC_DISABLE_POOL` env var lets users opt out for benchmarking
/// without recompiling.
static POOL_ENABLED: OnceLock<bool> = OnceLock::new();
static POOL_MAX_BYTES: OnceLock<usize> = OnceLock::new();
static POOL_MAX_BUFFER_BYTES: OnceLock<usize> = OnceLock::new();

fn pool_enabled() -> bool {
    *POOL_ENABLED.get_or_init(|| std::env::var("HIPDARC_DISABLE_POOL").is_err())
}

fn pool_max_bytes() -> usize {
    *POOL_MAX_BYTES.get_or_init(|| {
        std::env::var("HIPDARC_POOL_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            // Default cap: 1 GiB per device. The per-buffer threshold
            // (see below) keeps the largest individual buffers out of
            // the pool entirely, so this cap is mostly belt-and-
            // braces for the case where many medium buffers pile up.
            // Override with `HIPDARC_POOL_MAX_BYTES` (0 = unbounded).
            .unwrap_or(1024 * 1024 * 1024)
    })
}

/// Maximum size of an individual buffer eligible for the pool.
/// Buffers larger than this skip the pool on both alloc and free.
///
/// Why this matters: candle's `ConcatKvCache` grows the KV cache by
/// one row per decode step via `Tensor::cat`, which produces a buffer
/// of a *new* size every step (1, 2, 3, ... rows). Each unique size
/// lives in its own bucket, hits *exactly once*, and never gets
/// reused — so pooling them just hoards VRAM. The per-token
/// activation buffers that *do* benefit from pooling (the wq/wk/wv
/// projections, ffn intermediate, layer norms) are all under ~256 KB
/// at typical hidden sizes. The default threshold filters the cache
/// growth out and keeps the activations.
///
/// Override with `HIPDARC_POOL_MAX_BUFFER_BYTES`.
fn pool_max_buffer_bytes() -> usize {
    *POOL_MAX_BUFFER_BYTES.get_or_init(|| {
        std::env::var("HIPDARC_POOL_MAX_BUFFER_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            // 1 MiB: covers all per-token activation buffers for
            // hidden_size up to ~32K with f32 dtype, while excluding
            // even modest KV-cache cat results.
            .unwrap_or(1024 * 1024)
    })
}

/// Get or create the per-device pool for `ordinal`.
fn pool_for(ordinal: i32) -> Arc<Mutex<DevicePool>> {
    let registry = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("pool registry poisoned");
    guard
        .entry(ordinal)
        .or_insert_with(|| Arc::new(Mutex::new(DevicePool::new(ordinal, pool_max_bytes()))))
        .clone()
}

/// Try to take a buffer of `size` bytes from the device pool. Returns
/// `None` if pooling is disabled, the size is above the per-buffer
/// threshold, or no buffer of that size is cached.
#[inline]
fn pool_try_take(ordinal: i32, size: usize) -> Option<sys::hipDeviceptr_t> {
    if !pool_enabled() || size > pool_max_buffer_bytes() {
        return None;
    }
    let pool = pool_for(ordinal);
    let mut guard = pool.lock().ok()?;
    guard.try_take(size)
}

/// Return a buffer to the device pool. If pooling is disabled or the
/// buffer is above the per-buffer threshold, falls back to `hipFree`.
#[inline]
fn pool_give_back(ordinal: i32, ptr: sys::hipDeviceptr_t, size: usize) {
    if !pool_enabled() || size > pool_max_buffer_bytes() {
        unsafe {
            let _ = sys::hipSetDevice(ordinal);
            let _ = sys::hipFree(ptr);
        }
        return;
    }
    let pool = pool_for(ordinal);
    let mut guard = match pool.lock() {
        Ok(g) => g,
        Err(_) => {
            // Poisoned — fall back to direct free.
            unsafe {
                let _ = sys::hipSetDevice(ordinal);
                let _ = sys::hipFree(ptr);
            }
            return;
        }
    };
    guard.give_back(ptr, size);
}

// ---------------------------------------------------------------------------
// Decode-mode allocator — stable buffer addresses for op replay (G2)
// ---------------------------------------------------------------------------
//
// During decode recording, every allocation is captured in an ordered
// table. On replay, alloc() serves from this table so all intermediate
// buffers get the exact same device address every token. This eliminates
// the LIFO-pool non-determinism that caused 44% dynamic args in the
// decode plan.
//
// Buffers owned by the decode allocator use `device_ordinal =
// DECODE_ALLOC_SENTINEL` so `HipSlice::Drop` is a no-op — the buffers
// stay alive across the entire decode session.

use std::cell::RefCell;

/// Pad alloc size during recording to accommodate kv_len growth.
/// Adds 25% headroom (minimum 4 KB).
#[inline]
fn pad_decode_size(size: usize) -> usize {
    size + std::cmp::max(4096, size / 4)
}

/// Decode allocator mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeAllocMode {
    /// Recording: new allocs are appended to entries, returned with sentinel.
    Recording,
    /// Replaying: allocs are served from entries (if size matches).
    Replaying,
    /// Paused: the table is kept alive but neither recording nor replaying.
    /// Allocations go through the normal pool.
    Paused,
}

/// Decode allocator state: records or replays a fixed sequence of
/// `(byte_size, device_ptr)` allocations.
struct DecodeAllocState {
    /// Ordered list of (padded_byte_size, device_ptr) captured during recording.
    /// Sizes are padded via `pad_decode_size` to absorb kv_len growth.
    entries: Vec<(usize, sys::hipDeviceptr_t)>,
    /// Read cursor — advances on each `try_take`, reset between forwards.
    cursor: usize,
    /// Current mode.
    mode: DecodeAllocMode,
}

thread_local! {
    static DECODE_ALLOC: RefCell<Option<DecodeAllocState>> = RefCell::new(None);
}

/// Start recording decode allocations. Every `alloc()`/`alloc_zeros()`
/// that happens while recording is active is appended to the table.
/// The returned `HipSlice` is sentinel-marked so its `Drop` is a no-op.
pub fn decode_alloc_start_record() {
    DECODE_ALLOC.with(|d| {
        *d.borrow_mut() = Some(DecodeAllocState {
            entries: Vec::with_capacity(512),
            cursor: 0,
            mode: DecodeAllocMode::Recording,
        });
    });
}

/// Switch to replay mode, keeping recorded entries. Resets cursor.
pub fn decode_alloc_start_replay() {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            state.mode = DecodeAllocMode::Replaying;
            state.cursor = 0;
        }
    });
}

/// Reset cursor for the next forward pass (call before each replay).
pub fn decode_alloc_reset() {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            state.cursor = 0;
        }
    });
}

/// Pause — subsequent allocs go through normal pool. Table preserved.
pub fn decode_alloc_pause() {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            state.mode = DecodeAllocMode::Paused;
        }
    });
}

/// Resume replay mode (re-enable serving from table).
pub fn decode_alloc_resume() {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            state.mode = DecodeAllocMode::Replaying;
        }
    });
}

/// Snapshot the current decode-alloc mode (or `None` if disabled).
/// Pair with [`decode_alloc_set_mode`] to save/restore around a section
/// that needs to temporarily change modes (e.g. pause inside a recording
/// without losing the Recording state).
pub fn decode_alloc_get_mode() -> Option<DecodeAllocMode> {
    DECODE_ALLOC.with(|d| d.borrow().as_ref().map(|s| s.mode))
}

/// Restore a mode captured via [`decode_alloc_get_mode`]. No-op if the
/// allocator has been stopped since the snapshot.
pub fn decode_alloc_set_mode(mode: DecodeAllocMode) {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            state.mode = mode;
        }
    });
}

/// Disable the decode allocator. Sentinel-marked buffers remain
/// allocated until process exit (GPU driver reclaims them).
pub fn decode_alloc_stop() {
    DECODE_ALLOC.with(|d| {
        *d.borrow_mut() = None;
    });
}

/// True if decode alloc is active in any mode.
#[inline]
fn decode_alloc_active() -> bool {
    DECODE_ALLOC.with(|d| d.borrow().is_some())
}

/// True if in recording mode.
#[inline]
fn decode_alloc_is_recording() -> bool {
    DECODE_ALLOC.with(|d| {
        d.borrow().as_ref().map_or(false, |s| s.mode == DecodeAllocMode::Recording)
    })
}

/// Replay mode: try to serve a buffer of `size` bytes from the table.
#[inline]
fn decode_alloc_try_take(size: usize) -> Option<sys::hipDeviceptr_t> {
    DECODE_ALLOC.with(|d| {
        let mut guard = d.borrow_mut();
        let state = guard.as_mut()?;
        if state.mode != DecodeAllocMode::Replaying {
            return None;
        }
        if state.cursor < state.entries.len() {
            let (entry_size, ptr) = state.entries[state.cursor];
            if entry_size >= size {
                state.cursor += 1;
                return Some(ptr);
            }
        }
        None
    })
}

/// Record mode: append `(size, ptr)` to the table.
#[inline]
fn decode_alloc_record(size: usize, ptr: sys::hipDeviceptr_t) {
    DECODE_ALLOC.with(|d| {
        if let Some(ref mut state) = *d.borrow_mut() {
            if state.mode == DecodeAllocMode::Recording {
                state.entries.push((size, ptr));
            }
        }
    });
}

/// Number of entries recorded.
pub fn decode_alloc_entry_count() -> usize {
    DECODE_ALLOC.with(|d| {
        d.borrow().as_ref().map_or(0, |s| s.entries.len())
    })
}

// ---------------------------------------------------------------------------
// Capture-aware allocator routing
// ---------------------------------------------------------------------------
//
// While a HIP graph capture is in progress on this thread, all `HipSlice`
// drops must be routed through `hipFreeAsync` on the capture stream so
// they're recorded as graph-managed free nodes — see `with_capture`
// below for the wrapper that sets this state up. Outside capture, drops
// take the pool fast path (or sync `hipFree` if the pool is disabled).

use std::cell::Cell;

thread_local! {
    /// Stream pointer of the capture currently active on this thread,
    /// or null if no capture is in progress.
    static CAPTURE_STREAM: Cell<sys::hipStream_t> = const { Cell::new(std::ptr::null_mut()) };
}

#[inline]
fn capture_active_stream() -> sys::hipStream_t {
    CAPTURE_STREAM.with(|c| c.get())
}

/// Run `f` while HIP graph capture is active on `stream`. While `f`
/// runs, every `HipSlice<T>` that goes out of scope on this thread is
/// freed via `hipFreeAsync(stream)` instead of synchronous `hipFree` —
/// the resulting free nodes get recorded into the graph alongside the
/// kernel launches and async allocs, so the temporary's full lifetime
/// is owned by the captured graph.
///
/// Use this to capture a model forward pass. Without it, temporaries
/// that get dropped mid-forward would trip the runtime's "synchronous
/// call inside capture" guard and abort the capture.
pub fn with_capture<F>(stream: &HipStream, f: F) -> Result<HipGraph, DriverError>
where
    F: FnOnce() -> Result<(), DriverError>,
{
    // Reentrant captures aren't supported — the thread-local can only
    // hold one stream at a time. Bail loudly so misuse is obvious.
    let prev = CAPTURE_STREAM.with(|c| c.replace(stream.raw));
    if !prev.is_null() {
        // Restore and error out.
        CAPTURE_STREAM.with(|c| c.set(prev));
        return Err(DriverError::Hip(sys::hipError_t::hipErrorInvalidValue));
    }
    let result = (|| {
        let capture = HipGraphCapture::begin(stream)?;
        f()?;
        capture.end()
    })();
    CAPTURE_STREAM.with(|c| c.set(std::ptr::null_mut()));
    result
}

impl<T> std::fmt::Debug for HipSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipSlice")
            .field("len", &self.len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// HipView — borrowed device memory
// ---------------------------------------------------------------------------

/// Borrowed view into device memory. Does NOT free on drop.
pub struct HipView<'a, T> {
    pub(crate) ptr: sys::hipDeviceptr_t,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

unsafe impl<T> Send for HipView<'_, T> {}
unsafe impl<T> Sync for HipView<'_, T> {}

impl<'a, T> HipView<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_ptr(&self) -> sys::hipDeviceptr_t {
        self.ptr
    }

    /// Create a sub-view into a sub-range of this view.
    pub fn slice(&self, range: std::ops::Range<usize>) -> HipView<'a, T> {
        assert!(range.end <= self.len, "slice out of bounds");
        let offset = range.start * std::mem::size_of::<T>();
        HipView {
            ptr: unsafe { (self.ptr as *const u8).add(offset) as sys::hipDeviceptr_t },
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for HipView<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipView")
            .field("len", &self.len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PushKernelArg — trait for things that can be passed as kernel arguments
// ---------------------------------------------------------------------------

/// Trait for types that can be pushed as kernel arguments.
///
/// # Safety
/// The implementation must return a valid pointer and correct size for the
/// kernel argument.
pub unsafe trait PushKernelArg {
    fn as_kernel_arg(&self) -> (*const c_void, usize);
}

unsafe impl<T: DeviceRepr> PushKernelArg for T {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (self as *const T as *const c_void, std::mem::size_of::<T>())
    }
}

unsafe impl<T> PushKernelArg for &HipSlice<T> {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (
            &self.ptr as *const sys::hipDeviceptr_t as *const c_void,
            std::mem::size_of::<sys::hipDeviceptr_t>(),
        )
    }
}

unsafe impl<T> PushKernelArg for HipSlice<T> {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (
            &self.ptr as *const sys::hipDeviceptr_t as *const c_void,
            std::mem::size_of::<sys::hipDeviceptr_t>(),
        )
    }
}

unsafe impl<T> PushKernelArg for &HipView<'_, T> {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (
            &self.ptr as *const sys::hipDeviceptr_t as *const c_void,
            std::mem::size_of::<sys::hipDeviceptr_t>(),
        )
    }
}

unsafe impl<T> PushKernelArg for HipView<'_, T> {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (
            &self.ptr as *const sys::hipDeviceptr_t as *const c_void,
            std::mem::size_of::<sys::hipDeviceptr_t>(),
        )
    }
}


/// A null device pointer, used for contiguous layout sentinel.
/// The null value is stored inline so the pointer to it remains valid.
pub struct NullDevicePtr {
    null: sys::hipDeviceptr_t,
}

impl NullDevicePtr {
    pub const fn new() -> Self {
        Self {
            null: std::ptr::null_mut(),
        }
    }
}

impl Default for NullDevicePtr {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl PushKernelArg for NullDevicePtr {
    fn as_kernel_arg(&self) -> (*const c_void, usize) {
        (
            &self.null as *const sys::hipDeviceptr_t as *const c_void,
            std::mem::size_of::<sys::hipDeviceptr_t>(),
        )
    }
}

// ---------------------------------------------------------------------------
// HipModule / HipFunction — compiled GPU code
// ---------------------------------------------------------------------------

/// A loaded GPU code module (from HSACO binary).
pub struct HipModule {
    raw: sys::hipModule_t,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    /// Load a module from an HSACO (or fat binary) byte slice.
    pub fn load_data(data: &[u8]) -> Result<Self, DriverError> {
        let mut raw = std::ptr::null_mut();
        unsafe { check_hip(sys::hipModuleLoadData(&mut raw, data.as_ptr() as *const c_void))? };
        Ok(Self { raw })
    }

    /// Look up a kernel function by name.
    pub fn load_function(&self, name: &str) -> Result<HipFunction, DriverError> {
        let c_name = std::ffi::CString::new(name).map_err(|_| {
            DriverError::Message(format!("invalid kernel name: {name}"))
        })?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            check_hip(sys::hipModuleGetFunction(&mut raw, self.raw, c_name.as_ptr()))?;
        }
        Ok(HipFunction { raw, name: name.to_string() })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::hipModuleUnload(self.raw);
            }
        }
    }
}

impl std::fmt::Debug for HipModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipModule").finish()
    }
}

/// A handle to a kernel function within a module.
pub struct HipFunction {
    raw: sys::hipFunction_t,
    /// Symbolic name (matches the `extern "C"` symbol the kernel was loaded
    /// from). Used by the launch recorder to attach human-readable labels
    /// to recorded ops so K10-style replay debugging doesn't have to
    /// match raw `hipFunction_t` opaque pointers against module dumps.
    name: String,
}

impl HipFunction {
    /// Symbolic name of this kernel.
    pub fn name(&self) -> &str {
        &self.name
    }
}

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

impl std::fmt::Debug for HipFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipFunction").field("name", &self.name).finish()
    }
}

// ---------------------------------------------------------------------------
// LaunchConfig
// ---------------------------------------------------------------------------

/// Kernel launch configuration (grid and block dimensions).
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Simple 1D config for `n` elements with a default block size of 256.
    pub fn for_num_elems(n: u32) -> Self {
        const BLOCK: u32 = 256;
        let grid_x = n.div_ceil(BLOCK);
        Self {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// HipFunc — function + stream pair (mirrors CudaFunc in candle)
// ---------------------------------------------------------------------------

/// A kernel function bound to a stream, ready for argument building and launch.
#[derive(Debug)]
pub struct HipFunc {
    func: HipFunction,
    stream: Arc<HipStream>,
}

impl HipFunc {
    pub fn new(func: HipFunction, stream: Arc<HipStream>) -> Self {
        Self { func, stream }
    }

    /// Create a launch argument builder.
    pub fn builder(&self) -> LaunchArgs<'_> {
        LaunchArgs {
            func: &self.func,
            stream: &self.stream,
            args: Vec::new(),
            arg_sizes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// LaunchArgs — kernel argument builder
// ---------------------------------------------------------------------------

/// Builder for kernel launch arguments. Collects argument pointers then
/// launches the kernel.
pub struct LaunchArgs<'a> {
    func: &'a HipFunction,
    stream: &'a HipStream,
    args: Vec<*mut c_void>,
    /// Byte size of each argument (from `PushKernelArg::as_kernel_arg`).
    /// Used by the launch recorder to capture the correct number of bytes
    /// per arg — over-reading into adjacent stack memory would produce
    /// meaningless composite values for packed int32 args.
    arg_sizes: Vec<usize>,
}

impl<'a> LaunchArgs<'a> {
    /// Push a kernel argument. The argument must outlive this builder (it is
    /// borrowed by pointer until `launch` is called).
    pub fn arg<A: PushKernelArg>(&mut self, val: &A) -> &mut Self {
        let (ptr, size) = val.as_kernel_arg();
        self.args.push(ptr as *mut c_void);
        self.arg_sizes.push(size);
        self
    }

    /// Launch the kernel with the given configuration.
    ///
    /// # Safety
    /// The caller must ensure that all arguments match the kernel's expected
    /// signature and that device pointers are valid.
    pub unsafe fn launch(self, cfg: LaunchConfig) -> Result<(), DriverError> {
        let mut args = self.args;
        let sizes = self.arg_sizes;

        // G2: invoke the recording callback (if registered) to capture
        // this kernel launch for decode op cache replay. We pass the
        // symbolic kernel name in addition to the raw handle so the
        // recorder can attach human-readable labels to each captured
        // op (essential for diagnosing replay-state bugs).
        LAUNCH_RECORDER.with(|rec| {
            if let Some(ref cb) = *rec.borrow() {
                cb(
                    self.func.raw,
                    self.func.name.as_str(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    self.stream.raw(),
                    &args,
                    &sizes,
                );
            }
        });

        check_hip(sys::hipModuleLaunchKernel(
            self.func.raw,
            cfg.grid_dim.0,
            cfg.grid_dim.1,
            cfg.grid_dim.2,
            cfg.block_dim.0,
            cfg.block_dim.1,
            cfg.block_dim.2,
            cfg.shared_mem_bytes,
            self.stream.raw(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

/// G2: Thread-local kernel launch recorder callback.
/// When set, every `LaunchArgs::launch()` call invokes this function
/// with the kernel function handle, kernel name, grid/block config, and
/// arg pointers.
pub type LaunchRecorderFn = Box<
    dyn Fn(
        sys::hipFunction_t,
        &str,
        (u32, u32, u32),
        (u32, u32, u32),
        u32,
        sys::hipStream_t,
        &[*mut std::ffi::c_void],
        &[usize],
    ),
>;

thread_local! {
    /// The active launch recorder, if any.
    pub static LAUNCH_RECORDER: std::cell::RefCell<Option<LaunchRecorderFn>> = std::cell::RefCell::new(None);
}

/// Set the launch recorder callback. Returns the previous one (if any).
pub fn set_launch_recorder(recorder: Option<LaunchRecorderFn>) -> Option<LaunchRecorderFn> {
    LAUNCH_RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let prev = r.take();
        *r = recorder;
        prev
    })
}

/// Convenience macro to push a scalar kernel argument.
/// Usage: `barg!(builder, my_u32_value);`
#[macro_export]
macro_rules! barg {
    ($builder:expr, $val:expr) => {
        $builder.arg(&$val)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: capture an `alloc_zeros` + an `htod` copy + a `dtod`
    /// clone into a HIP graph, instantiate it, and replay it twice. Reads
    /// the destination buffer back through `clone_dtoh` and verifies the
    /// content matches.
    ///
    /// This is the minimum viable proof that capture/replay works on
    /// gfx906 + ROCm 7.1.1 — if this test passes, the bindings are
    /// trustworthy enough to wire into the model forward.
    #[test]
    fn hip_graph_capture_replay() {
        let dev = match HipDevice::new(0) {
            Ok(d) => d,
            // No HIP device available — skip silently so unit tests still
            // pass on a CPU-only build host.
            Err(_) => return,
        };
        let stream = HipStream::new(&dev).expect("create stream");

        // Two device buffers — `src` filled with a known pattern outside
        // capture, `dst` allocated outside capture as well so the
        // captured graph can target a stable address.
        let pattern: Vec<u8> = (0..256u32).map(|x| x as u8).collect();
        let mut src = stream.alloc::<u8>(pattern.len()).expect("alloc src");
        stream.memcpy_htod(&mut src, &pattern).expect("htod src");
        let mut dst = stream.alloc::<u8>(pattern.len()).expect("alloc dst");
        stream.memcpy_dtod(&mut dst, &src).expect("warm dtod");
        stream.synchronize().expect("warmup sync");

        // Capture: a single dtod copy into `dst`. We use the existing
        // memcpy_dtod which goes through hipMemcpyAsync — captureable.
        let capture = HipGraphCapture::begin(&stream).expect("begin capture");
        stream.memcpy_dtod(&mut dst, &src).expect("dtod inside capture");
        let graph = capture.end().expect("end capture");
        let exec = graph.instantiate().expect("instantiate");

        // Zero `dst`, replay the graph, verify the bytes are restored.
        let zeros = vec![0u8; pattern.len()];
        stream.memcpy_htod(&mut dst, &zeros).expect("zero dst");
        stream.synchronize().expect("zero sync");
        let dst_zeroed = stream.clone_dtoh(&dst).expect("read zeros");
        assert!(dst_zeroed.iter().all(|&b| b == 0), "dst not zeroed");

        exec.launch(&stream).expect("graph launch 1");
        stream.synchronize().expect("launch 1 sync");
        let dst_after_launch1 = stream.clone_dtoh(&dst).expect("read after 1");
        assert_eq!(
            dst_after_launch1, pattern,
            "graph replay #1 did not restore the pattern"
        );

        // Replay a second time on a fresh-zeroed dst — proves the graph
        // is reusable, not a one-shot.
        stream.memcpy_htod(&mut dst, &zeros).expect("re-zero dst");
        stream.synchronize().expect("re-zero sync");
        exec.launch(&stream).expect("graph launch 2");
        stream.synchronize().expect("launch 2 sync");
        let dst_after_launch2 = stream.clone_dtoh(&dst).expect("read after 2");
        assert_eq!(
            dst_after_launch2, pattern,
            "graph replay #2 did not restore the pattern"
        );
    }

    /// Mostly a compile-test: prove that `HipGraph` and `HipGraphExec`
    /// implement `Send + Sync`. The model code holds `Arc<HipGraphExec>`
    /// across worker threads when graphs are wired into multi-GPU
    /// pipelines, and we want a static guarantee that's safe.
    #[test]
    fn hip_graph_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HipGraph>();
        assert_send_sync::<HipGraphExec>();
    }

    /// `with_capture` defers `HipSlice` drops while capture is active.
    /// This test exercises the path that the model forward will rely on:
    /// allocate temporary buffers inside `with_capture`, let them drop,
    /// and prove the capture survives + the buffers are freed afterwards.
    #[test]
    fn hip_graph_with_capture_defers_drops() {
        let dev = match HipDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let stream = HipStream::new(&dev).expect("create stream");

        // Pre-existing buffer that we'll write into during capture.
        let mut dst = stream.alloc::<u8>(64).expect("alloc dst");
        // Need a stable host source for the htod copy inside capture.
        let host = vec![0xABu8; 64];

        let graph = with_capture(&stream, || {
            // Allocate a tmp inside capture, copy data through it, drop
            // it. Without `with_capture`'s deferred-free, the drop would
            // call sync `hipFree` and abort the capture.
            let mut tmp = stream.alloc::<u8>(64)?;
            stream.memcpy_htod(&mut tmp, &host)?;
            stream.memcpy_dtod(&mut dst, &tmp)?;
            // tmp goes out of scope here — its drop should be deferred.
            Ok(())
        })
        .expect("with_capture");

        let exec = graph.instantiate().expect("instantiate");
        // Replay the graph and verify dst contains 0xAB.
        let zeros = vec![0u8; 64];
        stream.memcpy_htod(&mut dst, &zeros).expect("zero dst");
        stream.synchronize().expect("zero sync");
        exec.launch(&stream).expect("graph launch");
        stream.synchronize().expect("launch sync");
        let dst_after = stream.clone_dtoh(&dst).expect("read dst");
        assert!(
            dst_after.iter().all(|&b| b == 0xAB),
            "graph replay did not write the expected pattern"
        );
    }
}
