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
        if len == 0 {
            return Ok(HipSlice {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                _marker: PhantomData,
            });
        }
        let size = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        // Try the async pool first; fall back if it returns
        // hipErrorNotSupported (older ROCm or unsupported device).
        // The free path mirrors this: free_stream is set iff the alloc
        // came from the async pool — that way `Drop` knows whether to
        // call `hipFreeAsync(stream)` or sync `hipFree`.
        let mut free_stream: sys::hipStream_t = std::ptr::null_mut();
        unsafe {
            let rc = sys::hipMallocAsync(&mut ptr, size, self.raw);
            if rc != sys::hipError_t::hipSuccess {
                check_hip(sys::hipMalloc(&mut ptr, size))?;
            } else {
                free_stream = self.raw;
            }
        };
        Ok(HipSlice {
            ptr,
            len,
            free_stream,
            _marker: PhantomData,
        })
    }

    /// Allocate `len` elements of type `T` on the device, zeroed.
    pub fn alloc_zeros<T: ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<HipSlice<T>, DriverError> {
        self.device.set_current()?;
        if len == 0 {
            return Ok(HipSlice {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                _marker: PhantomData,
            });
        }
        let size = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        let mut free_stream: sys::hipStream_t = std::ptr::null_mut();
        unsafe {
            let rc = sys::hipMallocAsync(&mut ptr, size, self.raw);
            if rc != sys::hipError_t::hipSuccess {
                check_hip(sys::hipMalloc(&mut ptr, size))?;
            } else {
                free_stream = self.raw;
            }
            check_hip(sys::hipMemsetAsync(ptr, 0, size, self.raw))?;
        }
        Ok(HipSlice {
            ptr,
            len,
            free_stream,
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
    /// `hipMallocAsync`. Currently retained as scaffolding for a
    /// future async-free path — `Drop` falls back to sync `hipFree`
    /// because the back-to-back async-alloc + async-free pattern
    /// regressed decode by ~25% on ROCm 7.1.1. See `Drop` impl below.
    #[allow(dead_code)]
    pub(crate) free_stream: sys::hipStream_t,
    pub(crate) _marker: PhantomData<T>,
}

// SAFETY: Device pointers can be sent across threads; the HIP runtime is
// thread-safe for memory operations.
unsafe impl<T> Send for HipSlice<T> {}
unsafe impl<T> Sync for HipSlice<T> {}

impl<T> HipSlice<T> {
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

impl<T> Drop for HipSlice<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Inside a HIP graph capture, sync `hipFree` would abort
            // the capture (HIP forbids synchronous device calls during
            // capture). Inside capture we route through `hipFreeAsync`
            // on the capture stream — this records a graph-managed
            // free node, so the temporary's lifetime becomes part of
            // the graph itself.
            //
            // Outside capture, sync `hipFree` is the fast path. Tested
            // unconditional `hipFreeAsync` and it regressed decode by
            // ~25% on ROCm 7.1.1, so we keep the branch.
            unsafe {
                let captured_to = capture_active_stream();
                if !captured_to.is_null() {
                    let _ = sys::hipFreeAsync(self.ptr, captured_to);
                } else {
                    let _ = sys::hipFree(self.ptr);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Capture-aware allocator routing
// ---------------------------------------------------------------------------
//
// While a HIP graph capture is in progress on this thread, all `HipSlice`
// drops must be routed through `hipFreeAsync` on the capture stream so
// they're recorded as graph-managed free nodes — see `with_capture`
// below for the wrapper that sets this state up. Outside capture, drops
// take the synchronous `hipFree` fast path because `hipFreeAsync` is
// measurably slower in steady-state decode (see commit history).

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
        Ok(HipFunction { raw })
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
}

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

impl std::fmt::Debug for HipFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipFunction").finish()
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
}

impl<'a> LaunchArgs<'a> {
    /// Push a kernel argument. The argument must outlive this builder (it is
    /// borrowed by pointer until `launch` is called).
    pub fn arg<A: PushKernelArg>(&mut self, val: &A) -> &mut Self {
        let (ptr, _size) = val.as_kernel_arg();
        self.args.push(ptr as *mut c_void);
        self
    }

    /// Launch the kernel with the given configuration.
    ///
    /// # Safety
    /// The caller must ensure that all arguments match the kernel's expected
    /// signature and that device pointers are valid.
    pub unsafe fn launch(self, cfg: LaunchConfig) -> Result<(), DriverError> {
        let mut args = self.args;
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
